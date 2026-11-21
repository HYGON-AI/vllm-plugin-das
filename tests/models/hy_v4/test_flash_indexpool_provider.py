# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Cross-repository numerical contract: real Flash writer -> adapter reader."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU/CUDA and companion adapter-das")
@pytest.mark.parametrize("backend", ["emu", "triton"])
def test_flash_pool_writer_matches_adapter_fp32_reader(monkeypatch, backend):
    from hcu_adapter.bridges.vllm_paged_mqa import get_paged_mqa_provider
    from vllm_hcu.models.hy_v4_flash.nvidia.indexpool_ops import fwht_quantize, write_pool

    from hcu_adapter import hcu_config
    monkeypatch.setattr(hcu_config, "load", lambda: {"ops": {
        "vllm_paged_mqa_logits": backend}, "overrides": {}})
    if backend == "triton":
        from vllm.v1.worker import workspace
        monkeypatch.setattr(workspace, "_manager", workspace.WorkspaceManager(torch.device("cuda")))

    torch.manual_seed(33)
    device = "cuda"
    heads, width = 32, 128
    backing = torch.zeros(3, 64 * 132 + 256, dtype=torch.uint8, device=device)
    cache = backing[:, 128:128 + 64 * 132].view(3, 64, 132)
    table = torch.tensor([[1, 0], [2, 0]], dtype=torch.int32, device=device)
    owners = torch.tensor([0] * 65 + [1] * 3 + [0], dtype=torch.int64, device=device)
    groups = torch.cat((torch.arange(65), torch.arange(3), torch.tensor([0]))).to(device)
    active = torch.ones(69, dtype=torch.bool, device=device)
    active[-1] = False  # Must not overwrite the first group's bytes or scale.
    quant_keys, scales = fwht_quantize(torch.randn(69, 128, device=device).bfloat16())
    write_pool(cache, table, owners, groups, active, quant_keys, scales)
    quant_q, q_scales = fwht_quantize(torch.randn(2 * heads, 128, device=device).bfloat16())
    q = quant_q.reshape(2, 1, heads, 128)
    weights = torch.randn(2, heads, device=device)
    folded_weights = weights * q_scales.reshape(2, heads)
    lens = torch.tensor([[65], [3]], dtype=torch.int32, device=device)
    metadata, score = get_paged_mqa_provider()
    actual = score((q, None), cache.unsqueeze(2), folded_weights, lens, table,
                   metadata(lens, 64, 2), max_model_len=width)

    reference = torch.zeros_like(actual)
    dequant_keys = quant_keys.float() * scales[:, None]
    dequant_q = q[:, 0].float() * q_scales.reshape(2, heads, 1)
    for row, (start, end) in enumerate(((0, 65), (65, 68))):
        logits = dequant_q[row] @ dequant_keys[start:end].T
        reference[row, :end - start] = (logits.relu() * weights[row, :, None]).sum(0)
    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-4)
