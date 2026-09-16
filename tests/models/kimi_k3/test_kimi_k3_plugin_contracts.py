"""CPU-level contracts for the HCU-owned Kimi-K3 plugin surface."""

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.platform.core_fix import (
    patch_kimi_k3_config,
    patch_kimi_k3_model_config,
)
from vllm_hcu.patch.worker.core_fix import patch_kimi_k3_model
from vllm_hcu.models.kimi_k3.amd.ops.situ import SituAndMul
from vllm_hcu.model_executor.layers.quantization.kimi_k3_w4a8 import (
    KIMI_K3_W4A8_PACKING,
    KimiK3W4A8MoEMethod,
    pack_int4_twos_complement,
    unpack_int4_twos_complement,
    validate_kimi_k3_w4a8_metadata,
)
from vllm_hcu.model_executor.layers.quantization.slimquant_facade import (
    KimiK3W4A8Facade,
)


def _ht_moe_config(ht=True, ll=False):
    return SimpleNamespace(
        activation=SimpleNamespace(value="situ"), swiglu_beta=None,
        moe_parallel_config=SimpleNamespace(
            use_deepep_ht_kernels=ht, use_deepep_ll_kernels=ll),
    )


def test_kimi_ht_ll_selection_rejects_ambiguous_backend():
    with pytest.raises(ValueError, match="simultaneously"):
        KimiK3W4A8MoEMethod(_ht_moe_config(ll=True))


@pytest.mark.parametrize("shared", [False, True])
def test_kimi_ll_graph_replay_refreshes_values_at_static_addresses(monkeypatch, shared):
    import sys
    from vllm_hcu.model_executor.layers.quantization import kimi_k3_graph_runtime as runtime

    method = SimpleNamespace(_hcu_kimi_ll_graph_boundary=True)
    layer = SimpleNamespace(_quant_method=SimpleNamespace(old_quant_method=method))
    monkeypatch.setitem(sys.modules,
        "vllm_hcu.model_executor.layers.fused_moe.moe_runner",
        SimpleNamespace(get_layer_from_name=lambda name: layer,
                        _resolve_layer_name=lambda name: name))
    monkeypatch.setattr(runtime, "is_breakable_cudagraph_enabled", lambda: True)
    monkeypatch.setattr(runtime, "weak_ref_tensor", lambda tensor: tensor)
    segments = []

    def add_eager(fn):
        segments.append(fn)
        return fn()

    capture = SimpleNamespace(_capturing=True, add_eager=add_eager)
    monkeypatch.setattr(runtime.BreakableCUDAGraphCapture, "current", lambda: capture)
    calls = []

    @runtime.kimi_ll_graph_boundary
    def forward(hidden, router, shared_input, ids, quanted, scale, weights, topk, name, width):
        calls.append(hidden.clone())
        return (shared_input + 3, hidden + 2) if shared else hidden + 2

    hidden = torch.ones(2, 4)
    shared_input = torch.ones(2, 6)
    result = forward(hidden, None, shared_input, None, None, None, None, None, "layer", 0)
    routed = result[1] if shared else result
    if shared:
        assert result[0].data_ptr() != shared_input.data_ptr()
        assert torch.equal(shared_input, torch.ones_like(shared_input))
        shared_address = result[0].data_ptr()
    assert routed.data_ptr() == hidden.data_ptr()
    assert torch.equal(hidden, torch.full_like(hidden, 3))
    hidden.fill_(10)
    shared_input.fill_(20)
    segments[0]()
    assert len(calls) == 2
    assert torch.equal(hidden, torch.full_like(hidden, 12))
    if shared:
        assert result[0].data_ptr() == shared_address
        assert torch.equal(result[0], torch.full_like(shared_input, 23))
        assert torch.equal(shared_input, torch.full_like(shared_input, 20))


def test_kimi_ht_quant_config_uses_int8_token_scales_and_high_nibble_compensation():
    method = KimiK3W4A8MoEMethod(_ht_moe_config())
    layer = SimpleNamespace(w13_weight_scale=torch.ones(2, 64, 1),
                            w2_weight_scale=torch.ones(2, 32, 1))
    quant = method.get_fused_moe_quant_config(layer)
    assert quant is not None
    assert quant.quant_dtype == torch.int8
    assert quant.per_act_token_quant
    assert torch.equal(quant.w1_scale, layer.w13_weight_scale * 16)
    assert torch.equal(quant.w2_scale, layer.w2_weight_scale * 16)
    assert torch.equal(layer.w13_weight_scale, torch.ones(2, 64, 1))


def test_kimi_ll_quant_config_keeps_scale_correction_at_masked_gemm():
    from vllm_hcu.patch.worker.op_opt.moe import patch_config

    patch_config.apply()
    method = KimiK3W4A8MoEMethod(_ht_moe_config(ht=False, ll=True))
    layer = SimpleNamespace(w13_weight_scale=torch.ones(2, 64, 1),
                            w2_weight_scale=torch.ones(2, 32, 1))
    quant = method.get_fused_moe_quant_config(layer)
    assert quant is not None
    assert quant.quant_dtype == torch.int8
    # The HCU config callback represents the per-token INT8 route through the
    # audited 256x256 descriptor because vLLM 0.25.1 cannot express both.
    assert not quant.per_act_token_quant
    assert quant.block_shape == [256, 256]
    assert quant.w1_scale is layer.w13_weight_scale
    assert quant.w2_scale is layer.w2_weight_scale


def test_kimi_ll_selects_masked_expert_and_binds_weights(monkeypatch):
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEActivationFormat,
    )
    from vllm_hcu.model_executor.layers.quantization import kimi_k3_ll_runtime

    calls = []

    class Expert:
        def __init__(self, *args):
            calls.append(args)

        def process_weights_after_loading(self, layer):
            calls.append(layer)

    monkeypatch.setattr(kimi_k3_ll_runtime, "KimiK3LLExperts", Expert)
    method = KimiK3W4A8MoEMethod(_ht_moe_config(ht=False, ll=True))
    method.moe_quant_config = object()
    layer = object()
    pf = SimpleNamespace(
        activation_format=FusedMoEActivationFormat.BatchedExperts,
        max_num_tokens_per_rank=lambda: 8,
        num_dispatchers=lambda: 2,
    )
    assert isinstance(method.select_gemm_impl(pf, layer), Expert)
    assert calls == [(method.moe, method.moe_quant_config, 8, 2), layer]
    assert pf._vllm_hcu_clean_low_latency_buffer is False
    assert pf._hcu_ll_cleaned_buffer_layout is None


def test_kimi_ht_cannot_fall_back_to_direct_aiter_apply():
    method = KimiK3W4A8MoEMethod(_ht_moe_config())
    with pytest.raises(RuntimeError, match="modular"):
        method.apply(None, None, None, None, None, None)


def test_kimi_ll_apply_uses_contiguous_workspace_and_native_situ(monkeypatch):
    import sys
    from vllm_hcu.model_executor.layers.quantization.kimi_k3_ll_runtime import KimiK3LLExperts

    expert = object.__new__(KimiK3LLExperts)
    expert._situ_beta, expert._situ_linear_beta = 4.0, 25.0
    expert._hcu_logical_n = 64
    expert._deepgemm_w13 = torch.zeros(2, 64, 64, dtype=torch.int8)
    expert._deepgemm_w2 = torch.zeros(2, 128, 16, dtype=torch.int8)
    expert.quant_config = SimpleNamespace(w1_scale=torch.ones(2, 64, 1),
                                        w2_scale=torch.ones(2, 128, 1))
    expert.estimate_expected_m = lambda *args: 4
    counts = torch.tensor([0, 3], dtype=torch.int32)
    calls = []

    def gemm(a, b, output, mask, expected):
        assert output.is_contiguous()
        assert mask is counts and expected == 4
        assert torch.all(b[1] == 16)
        output.zero_()
        calls.append('gemm')

    def situ(x, mask, **kwargs):
        assert x.shape == (2, 4, 64) and x.is_contiguous()
        assert mask is counts
        assert kwargs == dict(situ_beta=4.0, situ_linear_beta=25.0, expect_m=4)
        calls.append('situ')
        return torch.zeros(2, 4, 32, dtype=torch.int8), torch.ones(2, 4, 1)

    monkeypatch.setitem(sys.modules, 'deepgemm', SimpleNamespace(
        m_grouped_w4a8_gemm_nt_masked_hipc=gemm))
    monkeypatch.setitem(sys.modules, 'lightop.activation', SimpleNamespace(
        fuse_situ_mul_quant_ep=situ))
    expert.apply(output=torch.empty(2, 4, 128),
                 hidden_states=torch.zeros(2, 4, 128, dtype=torch.int8),
                 w1=expert._deepgemm_w13, w2=expert._deepgemm_w2,
                 topk_ids=torch.zeros(4, 2, dtype=torch.long), activation='situ',
                 global_num_experts=2, a1q_scale=torch.ones(2, 4, 1),
                 workspace13=torch.empty(2, 4, 128),
                 expert_tokens_meta=SimpleNamespace(expert_num_tokens=counts))
    assert calls == ['gemm', 'situ', 'gemm']


def test_kimi_ll_packing_uses_loaded_ep_dimensions_once(monkeypatch):
    from vllm_hcu.model_executor.layers.quantization import kimi_k3_ll_runtime as runtime

    expert = object.__new__(runtime.KimiK3LLExperts)
    expert._hcu_logical_n = 64  # Stale TP-sharded config dimension.
    expert._hcu_logical_k = 128
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.zeros(2, 256, 64, dtype=torch.int8), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.zeros(2, 128, 64, dtype=torch.int8), requires_grad=False)
    layer.w13_weight_scale = torch.ones(2, 256, 1)
    layer.w2_weight_scale = torch.ones(2, 128, 1)
    calls = []
    monkeypatch.setattr(runtime, 'pack_w4a8_moe_hipc_weight', lambda w: calls.append(w.shape) or w)
    monkeypatch.setattr(runtime, 'view_w4a8_moe_hipc_weight_n32_layout', lambda w: w)
    expert.process_weights_after_loading(layer)
    expected_calls = [(2, 256, 64), (2, 128, 64)]
    assert calls == expected_calls
    packed_w13 = layer.w13_weight
    packed_w2 = layer.w2_weight
    expert.process_weights_after_loading(layer)
    assert expert._hcu_logical_n == 256
    assert expert._hcu_logical_k == 128
    assert calls == expected_calls  # No repacking on the second invocation.
    assert layer.w13_weight is packed_w13
    assert layer.w2_weight is packed_w2
    assert expert._deepgemm_w13 is packed_w13
    assert expert._deepgemm_w2 is packed_w2


def test_kimi_feature_off_has_no_modular_quant_config():
    method = KimiK3W4A8MoEMethod(_ht_moe_config(ht=False))
    assert method.get_fused_moe_quant_config(None) is None


def test_kimi_ht_situ_quantization_preserves_source_parameters():
    from vllm_hcu.model_executor.layers.quantization.kimi_k3_ht_runtime import (
        KimiK3HTExperts,
    )
    expert = object.__new__(KimiK3HTExperts)
    expert._situ_beta, expert._situ_linear_beta = 1.25, 2.0
    calls = []
    result = (object(), object())
    expert._situ_quant = lambda x, **kw: (calls.append((x, kw)) or result)
    gateup = object()
    assert expert._quantize_activation(gateup, None, None) is result
    assert calls == [(gateup, {"beta": 1.25, "linear_beta": 2.0})]
    assert expert._permute_scale_kwargs(3584) == {"block_size": 3584}
    assert expert.adjust_N_for_activation(2048, "situ") == 1024
    with pytest.raises(ValueError, match="SiTU"):
        expert._validate_activation("silu")


def test_kimi_ht_rejects_batched_expert_format():
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEActivationFormat,
    )
    method = KimiK3W4A8MoEMethod(_ht_moe_config())
    pf = SimpleNamespace(activation_format=FusedMoEActivationFormat.BatchedExperts)
    with pytest.raises(ValueError, match="incompatible expert format"):
        method.select_gemm_impl(pf, None)


def test_kimi_ht_selects_contiguous_expert_and_binds_loaded_weights(monkeypatch):
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEActivationFormat,
    )
    from vllm_hcu.model_executor.layers.quantization import kimi_k3_ht_runtime

    calls = []

    class Expert:
        def __init__(self, moe, quant):
            calls.append((moe, quant))

        def process_weights_after_loading(self, layer):
            calls.append(layer)

    monkeypatch.setattr(kimi_k3_ht_runtime, "KimiK3HTExperts", Expert)
    method = KimiK3W4A8MoEMethod(_ht_moe_config())
    method.moe_quant_config = object()
    layer = object()
    pf = SimpleNamespace(activation_format=FusedMoEActivationFormat.Standard)
    assert isinstance(method.select_gemm_impl(pf, layer), Expert)
    assert calls == [(method.moe, method.moe_quant_config), layer]


def test_kimi_ht_missing_situ_api_fails_before_weight_packing(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_KIMI_HT_SITU_BACKEND", "lightop")
    import sys
    from vllm_hcu.model_executor.layers.quantization.kimi_k3_ht_runtime import (
        KimiK3HTExperts, DeepEPDeepGemmW4A8ContiguousExperts,
    )
    monkeypatch.setattr(DeepEPDeepGemmW4A8ContiguousExperts, "__init__",
                        lambda *args: None)
    monkeypatch.setitem(sys.modules, "lightop", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "lightop.activation", SimpleNamespace())
    config = SimpleNamespace(activation_situ_beta=1.0,
                             activation_situ_linear_beta=2.0)
    with pytest.raises(ImportError, match="fuse_situ_mul_quant"):
        KimiK3HTExperts(config, None)


def test_kimi_k3_w4a8_nibble_round_trip():
    values = torch.tensor([[-8, -1, 0, 7, 1, -7]], dtype=torch.int8)
    packed = pack_int4_twos_complement(values)
    assert packed.tolist() == [[0x8F, 0x07, 0x19]]
    assert torch.equal(unpack_int4_twos_complement(packed), values)


def test_kimi_k3_w4a8_metadata_rejects_wrong_packing():
    config = {
        "quant_method": "kimi_k3_w4a8",
        "format": "kimi-k3-int4-w4a8-v1",
        "model_version": "k3-test",
        "weight_bits": 4,
        "activation_bits": 8,
        "group_size": 32,
        "symmetric": True,
        "scale_dtype": "float32",
        "packing": "wrong",
        "num_experts": 2,
        "top_k": 1,
        "hidden_size": 32,
        "intermediate_size": 32,
    }
    try:
        validate_kimi_k3_w4a8_metadata(config)
    except ValueError as exc:
        assert "packing" in str(exc)
    else:
        raise AssertionError("invalid packing must fail closed")


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {
                "quant_method": "slimquant_w4a8",
                "num_experts": 2,
                "top_k": 3,
                "hidden_size": 32,
                "intermediate_size": 32,
            },
            "top_k <= num_experts",
        ),
        (
            {
                "quant_method": "slimquant_w4a8",
                "num_experts": 2,
                "top_k": 1,
                "hidden_size": 31,
                "intermediate_size": 32,
            },
            "divisible by 32",
        ),
    ],
)
def test_slimquant_w4a8_metadata_rejects_invalid_kernel_dimensions(config, message):
    with pytest.raises(ValueError, match=message):
        validate_kimi_k3_w4a8_metadata(config)


def test_situ_activation_shape_and_finiteness():
    activation = SituAndMul(beta=1.0, linear_beta=2.0)
    output = activation(torch.randn(4, 8))
    assert output.shape == (4, 4)
    assert torch.isfinite(output).all()


def test_packing_contract_constant_is_explicit():
    assert KIMI_K3_W4A8_PACKING == "twos-complement-high-even-low-odd"


def test_kimi_cli_quant_facade_materializes_without_hf_metadata():
    facade = KimiK3W4A8Facade()
    facade.maybe_update_config("Kimi-K3-INT4")
    assert facade.get_quant_method(torch.nn.Linear(4, 4), "model.layers.0") is None


def test_kimi_k3_patch_inventory_is_exact_and_version_neutral():
    assert patch_kimi_k3_config.TARGET_MODULE == "vllm.transformers_utils.config"
    assert patch_kimi_k3_model_config.TARGET_MODULE == "vllm.model_executor.models.config"
    assert patch_kimi_k3_model.TARGET_MODULE == "vllm.model_executor.models.registry"
    assert all(item.startswith("_vllm_hcu_") for item in (
        patch_kimi_k3_config._MARKER,
        patch_kimi_k3_model_config._MARKER,
        patch_kimi_k3_model._MARKER,
    ))
