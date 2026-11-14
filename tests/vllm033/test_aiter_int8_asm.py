from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "adapter_extension" / "src")
)

from hcu_vllm_das.moe import fused_moe_aiter as provider, int8_asm


def arguments(device="cpu"):
    return dict(
        hidden_states=torch.empty((3, 8), dtype=torch.bfloat16, device=device),
        w1=torch.empty((4, 12, 8), dtype=torch.int8, device=device),
        w2=torch.empty((4, 8, 6), dtype=torch.int8, device=device),
        topk_ids=torch.zeros((3, 2), dtype=torch.int32, device=device),
        topk_weights=torch.ones((3, 2), dtype=torch.float32, device=device),
        w1_scale=torch.ones((4, 12, 1), dtype=torch.float32, device=device),
        w2_scale=torch.ones((4, 8, 1), dtype=torch.float32, device=device),
        expert_map=torch.tensor(
            [0, -1, 1, -1, 2, -1, 3, -1], dtype=torch.int32, device=device
        ),
        global_num_experts=8,
        activation="silu",
        output_dtype=torch.bfloat16,
        moe_config=SimpleNamespace(
            solution_type="asm",
            config=dict(SOL_ID1=10010, SOL_ID2=20000, BLOCK_SIZE_M=16),
        ),
        a1_scale=None,
        a2_scale=None,
        w1_zp=None,
        w2_zp=None,
    )


@pytest.mark.parametrize(
    "kind",
    (
        "missing-api",
        "padding",
        "static-activation",
        "activation",
        "scale-shape",
        "strided",
        "large-buffer",
    ),
)
def test_rejected_stage_contract_falls_back_before_native_and_preserves_map(
    monkeypatch, kind
):
    a = arguments("meta" if kind == "large-buffer" else "cpu")
    native = SimpleNamespace(
        moe_sorting_fwd=lambda *a: pytest.fail("no native sorting"),
        asm_fmoe_a8=lambda *a: pytest.fail("no native GEMM"),
    )
    if kind == "missing-api":
        del native.asm_fmoe_a8
    elif kind == "padding":
        a["moe_config"].config["PADDED_K"] = 16
    elif kind == "static-activation":
        a["a1_scale"] = torch.tensor(0.01)
    elif kind == "activation":
        a["activation"] = "gelu"
    elif kind == "scale-shape":
        a["w1_scale"] = a["w1_scale"].squeeze(-1)
    elif kind == "strided":
        a["w1"] = a["w1"].transpose(1, 2)
    elif kind == "large-buffer":
        a["hidden_states"] = torch.empty(
            (2**24, 8), dtype=torch.bfloat16, device="meta"
        )
        a["topk_ids"] = torch.empty((2**24, 8), dtype=torch.int32, device="meta")
        a["topk_weights"] = torch.empty((2**24, 8), dtype=torch.float32, device="meta")
    vendor = SimpleNamespace(
        MoeQuantType=SimpleNamespace(W8A8="int8_w8a8"),
        get_aiter_moe_config=lambda *a, **kw: None,
        aiter_moe=lambda **kw: pytest.fail("rejected stage must use official fallback"),
    )
    original_import = provider.import_module
    monkeypatch.setattr(
        provider,
        "import_module",
        lambda name: vendor
        if name == "aiter.moe"
        else native
        if name == "aiter"
        else original_import(name),
    )
    calls = []
    expected = object()
    monkeypatch.setattr(
        provider,
        "_load_triton_fallback",
        lambda: lambda **kw: calls.append(kw) or expected,
    )
    assert provider._load_aiter_api().aiter_moe(**a) is expected
    assert len(calls) == 1
    assert calls[0]["expert_map"] is a["expert_map"]
    assert calls[0]["w1"] is a["w1"]
    assert calls[0]["w1_scale"] is a["w1_scale"]


def test_stage_launch_errors_propagate_without_switching_provider(monkeypatch):
    a = arguments()
    native = SimpleNamespace(
        moe_sorting_fwd=lambda *a: None, asm_fmoe_a8=lambda *a: None
    )
    vendor = SimpleNamespace(
        MoeQuantType=SimpleNamespace(W8A8="int8_w8a8"),
        get_aiter_moe_config=lambda: None,
    )
    original_import = provider.import_module
    monkeypatch.setattr(
        provider,
        "import_module",
        lambda name: vendor
        if name == "aiter.moe"
        else native
        if name == "aiter"
        else original_import(name),
    )
    monkeypatch.setattr(
        provider,
        "_load_triton_fallback",
        lambda: pytest.fail("native launch errors must propagate"),
    )

    def fail(**kw):
        raise RuntimeError("native-stage-failed")

    monkeypatch.setattr(int8_asm, "execute", fail)
    with pytest.raises(RuntimeError, match="native-stage-failed"):
        provider._load_aiter_api().aiter_moe(**a)


@pytest.mark.parametrize(("mapping", "expected"), [
    ([0, 1, -1, -1], [[0, 1], [-1, -1]]),
    ([-1, -1, 0, 1], [[-1, -1], [0, 1]]),
    ([0, -1, 1, -1], [[0, -1], [1, -1]]),
    ([-1, 0, -1, 1], [[-1, 0], [-1, 1]]),
    ([1, 0, -1, -1], [[1, 0], [-1, -1]]),
    ([-1, 1, -1, 0], [[-1, 1], [-1, 0]]),
])
def test_sorting_routes_preserve_canonical_local_slots(mapping, expected):
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    canonical = torch.tensor(mapping, dtype=torch.int32)
    before = canonical.clone()
    local, domain = int8_asm._prepare_sorting_routes(ids, canonical, 4, 2)
    assert domain == 2
    assert local.dtype == torch.int32 and local.is_contiguous()
    assert torch.equal(local, torch.tensor(expected, dtype=torch.int32))
    assert torch.equal(canonical, before)
    assert torch.equal(ids, torch.tensor([[0, 1], [2, 3]], dtype=torch.int64))


def test_sorting_routes_do_not_alias_invalid_original_int64_ids():
    ids = torch.tensor([[-1, 4, 2**40, -(2**40), 0, 3]], dtype=torch.int64)
    # Entries after global_num_experts are not part of this routing domain.
    canonical = torch.tensor([1, -1, -1, 0, 1, 0], dtype=torch.int32)
    local, domain = int8_asm._prepare_sorting_routes(ids, canonical, 4, 2)
    assert domain == 2
    assert torch.equal(local, torch.tensor([[-1, -1, -1, -1, 1, 0]], dtype=torch.int32))


def test_sorting_routes_without_map_retain_global_domain():
    ids = torch.tensor([[0, 3], [1, 2]], dtype=torch.int32)
    local, domain = int8_asm._prepare_sorting_routes(ids, None, 4, 4)
    assert domain == 4 and local.dtype == torch.int32
    assert torch.equal(local, ids)


@pytest.mark.parametrize("local_experts", [0, 1])
def test_local_domain_smaller_than_topk_refuses_before_native(local_experts):
    a = arguments()
    a["w1"] = a["w1"][:local_experts]
    a["w2"] = a["w2"][:local_experts]
    a["w1_scale"] = a["w1_scale"][:local_experts]
    a["w2_scale"] = a["w2_scale"][:local_experts]
    native = SimpleNamespace(moe_sorting_fwd=lambda *a: None, asm_fmoe_a8=lambda *a: None)
    assert int8_asm.unsupported_reason(a, native) == "local-expert-domain"


def test_no_map_requires_weights_for_the_entire_global_domain():
    a = arguments()
    a['expert_map'] = None
    native = SimpleNamespace(moe_sorting_fwd=lambda *a: None, asm_fmoe_a8=lambda *a: None)
    assert int8_asm.unsupported_reason(a, native) == 'expert-domain'


def test_empty_canonical_map_refuses_before_native():
    a = arguments()
    a['expert_map'] = torch.empty(0, dtype=torch.int32)
    native = SimpleNamespace(moe_sorting_fwd=lambda *a: None, asm_fmoe_a8=lambda *a: None)
    assert int8_asm.unsupported_reason(a, native) is not None


def test_empty_topk_refuses_before_native():
    a = arguments()
    a['topk_ids'] = torch.empty((3, 0), dtype=torch.int32)
    a['topk_weights'] = torch.empty((3, 0), dtype=torch.float32)
    native = SimpleNamespace(moe_sorting_fwd=lambda *a: None, asm_fmoe_a8=lambda *a: None)
    assert int8_asm.unsupported_reason(a, native) == 'local-expert-domain'
