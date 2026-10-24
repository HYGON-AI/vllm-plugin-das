"""CPU/mock evidence for the blocked DSV4.1 MoE gate-fusion contract.

These tests deliberately verify the existing fail-safe boundary rather than
claiming a native fused gate implementation.  In particular, padding and
DeepEP workspace ownership must remain in their existing downstream consumers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def test_legacy_gate_abi_falls_back_for_dsv41_routing_options() -> None:
    from vllm_hcu.model_executor.layers.fused_moe.lightop_routing import (
        lightop_moe_gate_kwargs,
    )

    legacy_lightop = SimpleNamespace()
    assert lightop_moe_gate_kwargs(legacy_lightop, "sigmoid", True) == {}
    # DSV4.1's default sqrt-softplus route cannot be sent to the old
    # sigmoid+renormalize positional ABI.
    assert lightop_moe_gate_kwargs(legacy_lightop, "sqrtsoftplus", True) is None
    assert lightop_moe_gate_kwargs(legacy_lightop, "sigmoid", False) is None
    assert lightop_moe_gate_kwargs(legacy_lightop, "softmax", True) is None


def test_capability_hook_is_keyword_only_and_forwards_exact_options() -> None:
    from vllm_hcu.model_executor.layers.fused_moe.lightop_routing import (
        lightop_moe_gate_kwargs,
    )

    calls: list[tuple[str, bool]] = []

    def supports(*, scoring_func: str, renormalize: bool) -> bool:
        calls.append((scoring_func, renormalize))
        return scoring_func == "sqrtsoftplus" and not renormalize

    lightop = SimpleNamespace(supports_moe_fused_gate_routing=supports)
    assert lightop_moe_gate_kwargs(lightop, "sqrtsoftplus", False) == {
        "scoring_func": "sqrtsoftplus",
        "renormalize": False,
    }
    assert lightop_moe_gate_kwargs(lightop, "softmax", True) is None
    assert calls == [("sqrtsoftplus", False), ("softmax", True)]

    malformed = SimpleNamespace(
        supports_moe_fused_gate_routing=lambda _scoring_func, _renormalize: True
    )
    with pytest.raises(RuntimeError, match="keyword arguments"):
        lightop_moe_gate_kwargs(malformed, "sqrtsoftplus", True)


def test_sqrtsoftplus_mock_normalizes_int64_ids_without_gpu() -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing

    calls: list[tuple[object, ...]] = []

    def mock_gate(*args: object) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(args)
        return torch.ones((0, 2)), torch.zeros((0, 2), dtype=torch.int32)

    original_loader = sqrtsoftplus_routing._load_lightop_sqrtsoftplus
    sqrtsoftplus_routing._load_lightop_sqrtsoftplus = lambda: mock_gate
    try:
        weights, ids = sqrtsoftplus_routing.run_lightop_sqrtsoftplus(
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((4,), dtype=torch.float32),
            topk=2,
            renormalize=True,
            routed_scaling_factor=1.0,
            indices_dtype=torch.int64,
        )
    finally:
        sqrtsoftplus_routing._load_lightop_sqrtsoftplus = original_loader

    assert calls and calls[0][0].shape == (0, 4)
    assert weights.shape == (0, 2)
    assert ids.dtype == torch.int64
    assert ids.shape == (0, 2)


def test_v41_fused_contract_accepts_int32_or_int64_and_ignores_unhashed_ids() -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing

    class CudaMetadata:
        device = torch.device("cuda")
        dtype = torch.float32

        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def is_contiguous(self) -> bool:
            return True

    logits = CudaMetadata((1024, 256))
    bias = CudaMetadata((256,))
    assert sqrtsoftplus_routing.can_use_lightop_sqrtsoftplus_v41_fused(
        logits,
        bias,
        topk=6,
        indices_dtype=torch.int64,
        input_tokens=None,
        hash_indices_table=None,
    )
    assert sqrtsoftplus_routing.can_use_lightop_sqrtsoftplus_v41_fused(
        logits,
        bias,
        topk=6,
        indices_dtype=torch.int32,
        input_tokens=None,
        hash_indices_table=None,
    )
    # Without a hash table/VL bias, input_tokens are irrelevant to the
    # official sqrt-softplus route and must not make the safe route dormant.
    assert sqrtsoftplus_routing.can_use_lightop_sqrtsoftplus_v41_fused(
        logits,
        bias,
        topk=6,
        indices_dtype=torch.int64,
        input_tokens=torch.tensor([1]),
        hash_indices_table=None,
    )


def test_v41_fused_mock_omits_padding_count_and_preserves_int64_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing

    calls: list[tuple[object, ...]] = []

    def mock_gate(*args: object) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(args)
        return torch.ones((2, 2)), torch.zeros((2, 2), dtype=torch.int64)

    monkeypatch.setattr(
        sqrtsoftplus_routing,
        "_load_lightop_sqrtsoftplus_v41_fused",
        lambda: mock_gate,
    )
    weights, ids = sqrtsoftplus_routing.run_lightop_sqrtsoftplus_v41_fused(
        torch.ones((2, 4), dtype=torch.float32),
        torch.ones((4,), dtype=torch.float32),
        topk=2,
        renormalize=True,
        routed_scaling_factor=1.5,
        indices_dtype=torch.int64,
    )

    assert len(calls) == 1
    assert calls[0][2:7] == (2, 0, True, 1.5, True)
    assert len(calls[0]) == 7
    assert weights.shape == (2, 2)
    assert ids.dtype == torch.int64


def test_v41_fused_converts_native_int64_ids_to_int32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing

    monkeypatch.setattr(
        sqrtsoftplus_routing,
        "_load_lightop_sqrtsoftplus_v41_fused",
        lambda: lambda *_args: (
            torch.ones((1, 2)),
            torch.tensor([[3, 4]], dtype=torch.int64),
        ),
    )
    _, ids = sqrtsoftplus_routing.run_lightop_sqrtsoftplus_v41_fused(
        torch.ones((1, 4)),
        torch.ones((4,)),
        topk=2,
        renormalize=True,
        routed_scaling_factor=1.0,
        indices_dtype=torch.int32,
    )
    assert ids.dtype is torch.int32


def test_v41_2d_vl_bias_is_materialized_without_hash_semantics() -> None:
    from vllm_hcu.model_executor.layers.fused_moe.sqrtsoftplus_routing import (
        make_v41_bias,
    )

    text = torch.full((4,), 1.0)
    vl = torch.full((4,), 2.0)
    ids = torch.tensor([100, 200, 101, 300], dtype=torch.int64)
    actual = make_v41_bias(text, ids, vl, 100, 4)
    expected = torch.tensor(
        [[2.0] * 4, [1.0] * 4, [2.0] * 4, [1.0] * 4]
    )
    torch.testing.assert_close(actual, expected)


def test_v41_vl_sentinel_without_vl_bias_falls_back_contractually() -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing

    class CudaMetadata:
        device = torch.device("cuda")
        dtype = torch.float32
        shape = (1024, 256)

        def is_contiguous(self) -> bool:
            return True

    assert not sqrtsoftplus_routing.can_use_lightop_sqrtsoftplus_v41_fused(
        CudaMetadata(),
        CudaMetadata(),
        topk=6,
        indices_dtype=torch.int64,
        input_tokens=torch.tensor([100]),
        hash_indices_table=None,
        bias_vl=None,
        image_sentinel_lo=100,
    )


def test_v41_vl_bias_is_excluded_from_fused_contract() -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing

    class CudaMetadata:
        device = torch.device("cuda")
        dtype = torch.float32
        shape = (1024, 256)

        def is_contiguous(self) -> bool:
            return True

    assert not sqrtsoftplus_routing.can_use_lightop_sqrtsoftplus_v41_fused(
        CudaMetadata(),
        CudaMetadata(),
        topk=6,
        indices_dtype=torch.int64,
        input_tokens=torch.tensor([100]),
        hash_indices_table=None,
        bias_vl=torch.ones(256),
        image_sentinel_lo=100,
    )


def test_v41_fused_missing_export_falls_back_without_native_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing

    monkeypatch.setattr(
        sqrtsoftplus_routing,
        "_load_lightop_sqrtsoftplus_v41_fused",
        lambda: None,
    )
    with pytest.raises(RuntimeError, match="V41_FUSED_GATE"):
        sqrtsoftplus_routing.run_lightop_sqrtsoftplus_v41_fused(
            torch.ones((1, 4), dtype=torch.float32),
            torch.ones((4,), dtype=torch.float32),
            topk=2,
            renormalize=True,
            routed_scaling_factor=1.0,
            indices_dtype=torch.int64,
        )


def test_v41_fused_router_adapter_is_default_on_and_forwards_exact_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing
    from vllm_hcu.patch.worker.op_opt.moe import patch_fused_topk_bias_router
    from vllm_hcu.platforms import envs as henvs

    class CudaMetadata:
        device = torch.device("cuda")
        dtype = torch.float32
        shape = (1024, 256)

        def is_contiguous(self) -> bool:
            return True

    class BiasMetadata(CudaMetadata):
        shape = (256,)

    def original(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor=1.0,
        bias_vl=None,
        image_sentinel_lo=0,
    ) -> str:
        del (
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
            routed_scaling_factor,
            bias_vl,
            image_sentinel_lo,
        )
        return "official"

    module = ModuleType(patch_fused_topk_bias_router.TARGET_MODULE)
    module.vllm_topk_softplus_sqrt = original
    patch_fused_topk_bias_router.apply_to_module(module)
    logits = CudaMetadata()
    bias = BiasMetadata()
    indices = torch.empty((1024, 6), dtype=torch.int64)
    weights = torch.empty((1024, 6), dtype=torch.float32)
    token_expert = torch.empty((1024, 6), dtype=torch.int32)

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE", True, raising=False
    )
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_V41_FUSED_GATE",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        sqrtsoftplus_routing,
        "can_use_lightop_sqrtsoftplus",
        lambda *_args, **_kwargs: False,
    )
    assert module.vllm_topk_softplus_sqrt(
        weights, indices, token_expert, logits, True, bias
    ) == "official"

    forwarded: list[tuple[object, ...]] = []
    expected = (torch.ones((1024, 6)), torch.zeros((1024, 6), dtype=torch.int64))
    monkeypatch.setattr(
        henvs,
        "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_V41_FUSED_GATE",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        sqrtsoftplus_routing,
        "can_use_lightop_sqrtsoftplus_v41_fused",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        sqrtsoftplus_routing,
        "is_lightop_sqrtsoftplus_v41_fused_available",
        lambda: True,
    )
    monkeypatch.setattr(
        sqrtsoftplus_routing,
        "get_v41_padding_mask",
        lambda: None,
    )

    def run(*args: object, **kwargs: object):
        forwarded.append((args, kwargs))
        return expected

    monkeypatch.setattr(sqrtsoftplus_routing, "run_lightop_sqrtsoftplus_v41_fused", run)
    actual = module.vllm_topk_softplus_sqrt(
        weights, indices, token_expert, logits, True, bias, None, None, 1.25
    )
    assert actual[0] is expected[0]
    assert actual[1] is expected[1]
    assert len(forwarded) == 1
    assert forwarded[0][1]["topk"] == 6
    assert forwarded[0][1]["renormalize"] is True
    assert forwarded[0][1]["routed_scaling_factor"] == 1.25
    assert forwarded[0][1]["indices_dtype"] is torch.int64
    assert "num_token_non_padded" not in forwarded[0][1]


def test_v41_padding_contract_falls_back_when_context_publishes_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing

    context = SimpleNamespace(is_padding=torch.tensor([False, True]))
    forward_context = ModuleType("vllm.forward_context")
    forward_context.is_forward_context_available = lambda: True
    forward_context.get_forward_context = lambda: context
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_context)
    assert sqrtsoftplus_routing.is_v41_padding_free_forward() is False


def test_v41_padding_contract_restores_official_zero_and_minus_one() -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing

    weights = torch.ones((2, 3), dtype=torch.float32)
    ids = torch.tensor([[4, 5, 6], [7, 8, 9]], dtype=torch.int64)
    actual_weights, actual_ids = sqrtsoftplus_routing.apply_v41_padding_contract(
        weights,
        ids,
        torch.tensor([False, True]),
    )
    torch.testing.assert_close(
        actual_weights,
        torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
    )
    assert actual_ids.tolist() == [[4, 5, 6], [-1, -1, -1]]


def test_v41_incompatible_padding_mask_returns_fallback_signal() -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing

    result = sqrtsoftplus_routing.apply_v41_padding_contract(
        torch.ones((2, 2)),
        torch.ones((2, 2), dtype=torch.int64),
        torch.ones((3,), dtype=torch.bool),
    )
    assert result is None


def test_v41_real_entry_applies_padding_contract_after_fused_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing
    from vllm_hcu.patch.worker.op_opt.moe import patch_fused_topk_bias_router
    from vllm_hcu.platforms import envs as henvs

    class CudaMetadata:
        # This is a CPU/mock entrypoint test; the padding mask below is a CPU
        # tensor because the test hides all accelerator devices.
        device = torch.device("cpu")
        dtype = torch.float32
        shape = (2, 4)

        def is_contiguous(self) -> bool:
            return True

    class BiasMetadata(CudaMetadata):
        shape = (4,)

    def original(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor=1.0,
        bias_vl=None,
        image_sentinel_lo=0,
    ):
        del (
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
            routed_scaling_factor,
            bias_vl,
            image_sentinel_lo,
        )
        return "official"

    module = ModuleType(patch_fused_topk_bias_router.TARGET_MODULE)
    module.vllm_topk_softplus_sqrt = original
    patch_fused_topk_bias_router.apply_to_module(module)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        henvs, "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_V41_FUSED_GATE", True,
        raising=False,
    )
    monkeypatch.setattr(
        sqrtsoftplus_routing,
        "can_use_lightop_sqrtsoftplus_v41_fused",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        sqrtsoftplus_routing,
        "is_lightop_sqrtsoftplus_v41_fused_available",
        lambda: True,
    )
    monkeypatch.setattr(
        sqrtsoftplus_routing,
        "get_v41_padding_mask",
        lambda: torch.tensor([False, True]),
    )
    monkeypatch.setattr(
        sqrtsoftplus_routing,
        "run_lightop_sqrtsoftplus_v41_fused",
        lambda *_args, **_kwargs: (
            torch.ones((2, 2), dtype=torch.float32),
            torch.tensor([[2, 3], [4, 5]], dtype=torch.int64),
        ),
    )
    weights = torch.empty((2, 2), dtype=torch.float32)
    ids = torch.empty((2, 2), dtype=torch.int64)
    token_expert = torch.empty((2, 2), dtype=torch.int32)
    actual_weights, actual_ids = module.vllm_topk_softplus_sqrt(
        weights, ids, token_expert, CudaMetadata(), True, BiasMetadata()
    )
    torch.testing.assert_close(
        actual_weights,
        torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
    )
    assert actual_ids.tolist() == [[2, 3], [-1, -1]]


def test_v41_padding_contract_has_no_device_mask_scan_symbols() -> None:
    from vllm_hcu.model_executor.layers.fused_moe import sqrtsoftplus_routing

    source = Path(sqrtsoftplus_routing.__file__).read_text()
    function_source = source.split("def is_v41_padding_free_forward", 1)[1].split(
        "def run_lightop_sqrtsoftplus", 1
    )[0]
    assert "torch.nonzero(" not in function_source
    assert ".item(" not in function_source
    assert "torch.all(" not in function_source


def test_empty_shape_is_not_advertised_as_a_production_route() -> None:
    from vllm_hcu.model_executor.layers.fused_moe.sqrtsoftplus_routing import (
        is_lightop_sqrtsoftplus_shape_supported,
    )

    assert not is_lightop_sqrtsoftplus_shape_supported(0, 256, 8)
    assert is_lightop_sqrtsoftplus_shape_supported(1024, 256, 8)


def test_padding_and_deepep_workspace_are_not_gate_outputs() -> None:
    repo = Path(__file__).resolve().parents[2]
    gate_source = (repo / "vllm_hcu/ops/fuse_moe_gate.py").read_text()
    align_source = (
        repo / "vllm_hcu/patch/worker/op_opt/moe/patch_moe_align_block_size.py"
    ).read_text()
    deepep_source = (
        repo / "vllm_hcu/model_executor/layers/fused_moe/prepare_finalize/deepep_auto.py"
    ).read_text()

    lightop_call = gate_source.split(
        "from lightop.moe import moe_fused_gate as lightop_moe_fused_gate",
        1,
    )[1].split("except RuntimeError", 1)[0]
    assert "input_ids" not in lightop_call
    assert "workspace" not in lightop_call.lower()
    assert "topk_ids.numel()" in align_source
    assert "fill_value=-1" in align_source
    assert "prepare_async" in deepep_source
    assert "begin_moe_call" in deepep_source
