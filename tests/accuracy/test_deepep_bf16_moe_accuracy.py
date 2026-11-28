# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Real BF16 DeepGEMM pipelines, including shared output/workspace storage."""

from math import prod
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.modular_kernel import ExpertTokensMetadata
from vllm_hcu.model_executor.layers.fused_moe.experts.deepep_bf16_moe import (
    DeepEPAutoBF16Experts,
)

pytestmark = pytest.mark.hcu


def _config():
    return SimpleNamespace(
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        max_num_tokens=256,
    )


def _buffers(experts, x, w1, w2, ids, meta):
    e, m, n, k, topk = experts.moe_problem_size(x, w1, w2, ids)
    s13, s2, so = experts.workspace_shapes(
        m, n, k, topk, e, e, meta, MoEActivation.SILU
    )
    common = torch.empty(max(prod(s13), prod(so)), device="cuda", dtype=x.dtype)
    return (
        common[: prod(so)].view(so),
        common[: prod(s13)].view(s13),
        torch.empty(s2, device="cuda", dtype=x.dtype),
    )


@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("hidden,intermediate", [(512, 256), (3072, 1024)])
def test_auto_ht_ll_ht_matches_reference(tokens, hidden, intermediate):
    torch.manual_seed(41 + tokens)
    e, k, h, topk = 4, hidden, intermediate, 2
    w1 = torch.randn(e, 2 * h, k, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(e, k, h, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(e, device="cuda")[:topk] for _ in range(tokens)]
    ).to(torch.int64)
    # Include an invalid/non-local route and leave expert 3 empty.
    ids[ids == 3] = -1
    weights = torch.softmax(torch.randn(tokens, topk, device="cuda"), -1)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for t in range(tokens):
        for slot in range(topk):
            expert = int(ids[t, slot])
            if expert < 0:
                continue
            gate, up = F.linear(x[t].float(), w1[expert].float()).chunk(2)
            activated = F.silu(gate) * up
            expected[t] += F.linear(activated, w2[expert].float()) * weights[t, slot]
    impl = DeepEPAutoBF16Experts(
        _config(),
        FUSED_MOE_UNQUANTIZED_CONFIG,
        max_num_tokens=256,
        num_dispatchers=1,
    )
    impl.process_weights_after_loading(SimpleNamespace(w13_weight=w1, w2_weight=w2))
    for low_latency in [False, True, False]:
        impl.set_deepep_auto_use_low_latency(low_latency)
        if low_latency:
            batched = torch.zeros(e, 256, k, device="cuda", dtype=x.dtype)
            counts = torch.zeros(e, device="cuda", dtype=torch.int32)
            slots = {}
            for expert in range(e):
                routes = (ids == expert).nonzero()
                counts[expert] = routes.size(0)
                for row, (t, slot) in enumerate(routes.tolist()):
                    batched[expert, row] = x[t]
                    slots[t, slot] = (expert, row)
            input_x = batched
            meta = ExpertTokensMetadata(counts, counts.cpu())
        else:
            input_x, meta = x, None
        out, ws13, ws2 = _buffers(impl, input_x, w1, w2, ids, meta)
        impl.apply(
            out,
            input_x,
            w1,
            w2,
            weights,
            ids,
            MoEActivation.SILU,
            e,
            None,
            None,
            None,
            ws13,
            ws2,
            meta,
            False,
        )
        if low_latency:
            combined = torch.zeros_like(expected)
            for (t, slot), (expert, row) in slots.items():
                combined[t] += out[expert, row].float() * weights[t, slot]
        else:
            combined = out.float()
        torch.cuda.synchronize()
        assert torch.isfinite(combined).all()
        error = combined - expected
        relative_l2 = error.norm() / expected.norm().clamp_min(1e-8)
        print(
            f"HT/LL={low_latency} shape={k}/{h} M={tokens} "
            f"max_abs={error.abs().max().item():.6f} "
            f"relative_l2={relative_l2.item():.6f}"
        )
        assert relative_l2 < 0.02
        torch.testing.assert_close(combined, expected, rtol=0.04, atol=0.02)
