# SPDX-License-Identifier: Apache-2.0
"""CPU sizing contracts for provenance-tagged DeepEP physical expert layouts."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.platforms import envs as henvs


def sizing():
    path = Path(__file__).parents[2] / "vllm_hcu/model_executor/layers/fused_moe/deep_gemm_utils.py"
    tree = ast.parse(path.read_text())
    names = {"compute_aligned_M_and_alignment", "expert_num_tokens_round_up_and_sum"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"torch": torch, "mk": SimpleNamespace(ExpertTokensMetadata=object),
          "henvs": henvs, "current_platform": SimpleNamespace(is_rocm=lambda: True),
          "_HCU_TOKEN_ALIGNMENT": 256,
          "round_up": lambda value, align: ((value + align - 1) // align) * align}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns["compute_aligned_M_and_alignment"]


@pytest.mark.parametrize("counts", [[0, 0, 0, 0], [1, 0, 2, 257], [511, 256, 0, 9]])
def test_exact_layout_matches_physical_count_padding(monkeypatch, counts):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_DEEPEP_HT_EXACT_LAYOUT", "1")
    rows, top_k = 1024, 6
    expected = sum(((c + 255) // 256) * 256 for c in counts)
    meta = SimpleNamespace(_hcu_deepep_ht_layout=(rows, top_k, 4, expected),
                           expert_num_tokens_cpu=torch.tensor(counts))
    assert sizing()(rows, top_k, 4, 256, meta) == (expected, 256)


@pytest.mark.parametrize("reason", ["off", "master", "rows", "topk", "experts", "align", "untagged"])
def test_conservative_fallback_for_untrusted_or_changed_layout(monkeypatch, reason):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0" if reason == "master" else "1")
    monkeypatch.setenv("VLLM_HCU_DEEPEP_HT_EXACT_LAYOUT", "0" if reason == "off" else "1")
    rows, top_k, experts, alignment = 1024, 6, 4, 256
    meta = SimpleNamespace(_hcu_deepep_ht_layout=(rows, top_k, experts, 768),
                           expert_num_tokens_cpu=torch.tensor([1, 0, 2, 257]))
    if reason == "rows": rows = 512
    if reason == "topk": top_k = 2
    if reason == "experts": experts = 8
    if reason == "align": alignment = 512
    if reason == "untagged": del meta._hcu_deepep_ht_layout
    expected = ((rows * top_k + experts * (alignment - 1) + alignment - 1) // alignment) * alignment
    assert sizing()(rows, top_k, experts, alignment, meta) == (expected, alignment)


@pytest.mark.parametrize("rows,top_k,counts,valid", [
    (0, 6, [0, 0], True),
    (300, 6, [257, 100, 0], True),
    (300, 6, [301, 0], False),
    (300, 1, [257, 100], False),
    (300, 6, [100, 100], False),
    (300, 6, [-1, 300], False),
    (300, 6, [], False),
])
def test_received_count_provenance_validation(rows, top_k, counts, valid):
    from vllm_hcu.model_executor.layers.fused_moe.deepep_runtime import _bind_ht_exact_layout
    meta = SimpleNamespace()
    assert _bind_ht_exact_layout(meta, rows, top_k, counts) is valid
    if valid:
        assert meta._hcu_deepep_ht_layout == (
            rows, top_k, len(counts), sum(((c + 255) // 256) * 256 for c in counts)
        )
    else:
        assert not hasattr(meta, "_hcu_deepep_ht_layout")


@pytest.mark.parametrize("enabled", [True, False])
def test_ht_receiver_preserves_physical_ids_and_binds_counts(monkeypatch, enabled):
    from vllm_hcu.model_executor.layers.fused_moe.deepep_runtime import ht_receiver
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_DEEPEP_HT_EXACT_LAYOUT", "1" if enabled else "0")
    counts = [2, 2]
    module = SimpleNamespace(
        torch=torch,
        dbo_current_ubatch_id=lambda: 0,
        mk=SimpleNamespace(ExpertTokensMetadata=SimpleNamespace(
            make_from_list=lambda counts, device: SimpleNamespace(
                expert_num_tokens_cpu=torch.tensor(counts),
                expert_num_tokens=torch.tensor(counts),
            )
        )),
        moe_kernel_quantize_input=lambda x, *args, **kwargs: (x, torch.ones((3, 1))),
    )
    quant = SimpleNamespace(_vllm_hcu_channel_fp8_deepgemm=True,
        use_fp8_w8a8=True, is_per_act_token=True, is_block_quantized=False,
        quant_dtype=torch.float8_e4m3fn, per_act_token_quant=True,
        block_shape=None, is_scale_swizzled=False)
    local_ids = torch.tensor([[0, 1], [0, -1], [1, -1]])
    result = ht_receiver(module, SimpleNamespace(rank_expert_offset=2),
        SimpleNamespace(event=None), False, torch.ones((3, 4)), local_ids, 4,
        counts, torch.ones((3, 2)), None, quant, False)
    ids, meta = result[3], result[2]
    canonical_map = torch.tensor([-1, -1, 0, 1])
    recovered = canonical_map[ids]
    assert torch.equal(recovered, local_ids)
    actual_counts = torch.bincount(recovered[recovered >= 0], minlength=2)
    assert torch.equal(actual_counts, meta.expert_num_tokens_cpu)
    if enabled:
        assert meta._hcu_deepep_ht_layout == (3, 2, 2, 512)
        assert sizing()(3, 2, 2, 256, meta) == (512, 256)
    else:
        assert not hasattr(meta, "_hcu_deepep_ht_layout")
        assert sizing()(3, 2, 2, 256, meta) == (768, 256)
