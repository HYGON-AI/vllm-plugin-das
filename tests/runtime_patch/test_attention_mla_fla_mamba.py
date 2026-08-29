# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU runtime-contract tests for attention, MLA, FLA, and Mamba adapters."""

from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _adapter(name: str):
    return importlib.import_module(f"vllm_hcu.patch.worker.op_opt.{name}")


def _module(name: str, **values) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(values)
    return module


def _gdn_causal_conv1d_fn(
    x,
    weight,
    bias=None,
    conv_states=None,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
    pad_slot_id=-1,
    null_block_id=0,
    block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    num_computed_tokens=None,
    block_size_to_align=0,
    metadata=None,
    validate_data=False,
):
    return weight


def _gdn_causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    query_start_loc=None,
    max_query_len=-1,
    null_block_id=0,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    validate_data=False,
):
    return weight


def _install_fake_module(monkeypatch, name: str, **values) -> ModuleType:
    parts = name.split(".")
    for index in range(1, len(parts)):
        parent_name = ".".join(parts[:index])
        if parent_name not in sys.modules:
            monkeypatch.setitem(sys.modules, parent_name, ModuleType(parent_name))
    module = _module(name, **values)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_dense_attention_layer_installs_hcu_runtime_and_preserves_fallback(
    monkeypatch,
):
    adapter = _adapter("patch_attention_layer")
    calls: list[tuple[object, ...]] = []

    def original_init(layer, quant_config, prefix):
        calls.append(("official-init", layer, quant_config, prefix))
        return "official-init"

    class Attention:
        def forward(
            self,
            query,
            key,
            value,
            output_shape=None,
            output_dtype=None,
        ):
            calls.append(
                (
                    "official-forward",
                    query,
                    key,
                    value,
                    output_shape,
                    output_dtype,
                )
            )
            return "official-forward"

    class FusedQkvSplitRmsNormRopeAttention:
        pass

    runtime = _module(
        "vllm_hcu.model_executor.layers.attention_runtime",
        FusedQkvSplitRmsNormRopeAttention=FusedQkvSplitRmsNormRopeAttention,
        init_kv_cache_quant_e5m2=lambda *args: (
            calls.append(("hcu-init", *args)) or "hcu-init"
        ),
        attention_forward=lambda *args: (
            calls.append(("hcu-forward", *args)) or "hcu-forward"
        ),
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    import vllm_hcu.model_executor.layers as hcu_layers

    monkeypatch.setattr(hcu_layers, "attention_runtime", runtime, raising=False)
    monkeypatch.setattr(adapter, "_feature_flags", lambda: (False, False))
    module = _module(
        adapter.TARGET_MODULE,
        Attention=Attention,
        _init_kv_cache_quant=original_init,
    )

    assert adapter.apply_to_module(module) is True
    assert adapter.apply_to_module(module) is False
    assert (
        module.FusedQkvSplitRmsNormRopeAttention
        is FusedQkvSplitRmsNormRopeAttention
    )

    layer = SimpleNamespace(kv_cache_dtype="auto")
    assert module._init_kv_cache_quant(layer, "quant", "prefix") == "official-init"
    instance = Attention()
    assert instance.forward("q", "k", "v") == "official-forward"

    layer.kv_cache_dtype = "fp8_e5m2"
    assert module._init_kv_cache_quant(layer, "quant", "prefix") == "hcu-init"
    instance.kv_cache_dtype = "fp8_e5m2"
    assert instance.forward("q", "k", "v") == "hcu-forward"
    assert [call[0] for call in calls] == [
        "official-init",
        "official-forward",
        "hcu-init",
        "hcu-forward",
    ]


def test_dense_attention_public_export_is_idempotent_and_exact():
    adapter = _adapter("patch_attention_exports")

    class Attention:
        pass

    class FusedQkvSplitRmsNormRopeAttention:
        pass

    package = _module(
        adapter.TARGET_MODULE,
        Attention=Attention,
        attention=SimpleNamespace(
            FusedQkvSplitRmsNormRopeAttention=(
                FusedQkvSplitRmsNormRopeAttention
            )
        ),
        __all__=["Attention"],
    )

    assert adapter.apply_to_module(package) is True
    assert adapter.apply_to_module(package) is False
    assert (
        package.FusedQkvSplitRmsNormRopeAttention
        is FusedQkvSplitRmsNormRopeAttention
    )
    assert package.__all__ == [
        "Attention",
        "FusedQkvSplitRmsNormRopeAttention",
    ]


def test_flashmla_sparse_bf16_preserves_v0251_topk_length(monkeypatch):
    adapter = _adapter("patch_flashmla_sparse")
    calls: list[tuple[object, ...]] = []

    class FlashMLASparseMetadataBuilder:
        def build(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build=False,
        ):
            del self, common_prefix_len, common_attn_metadata, fast_build
            return SimpleNamespace(fp8_use_mixed_batch=False)

    class FlashMLASparseImpl:
        softmax_scale = 0.25
        num_heads = 2

        def _fp8_flash_mla_kernel(
            self,
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            kernel_metadata,
        ):
            del self, q, kv_c_and_k_pe_cache, topk_indices, kernel_metadata
            return "official-fp8"

        def _bf16_flash_mla_kernel(
            self,
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            topk_length=None,
        ):
            calls.append(
                ("official", q, kv_c_and_k_pe_cache, topk_indices, topk_length)
            )
            return "official-bf16"

    def sparse_fwd(q, cache, indices, softmax_scale, *, topk_length=None):
        calls.append(("hcu", q, cache, indices, softmax_scale, topk_length))
        return torch.ones(q.shape[0], 4, q.shape[-1]), None

    import vllm_hcu.v1.attention.ops.flashmla as hcu_flashmla

    monkeypatch.setattr(hcu_flashmla, "FlashMLASchedMeta", object)
    monkeypatch.setattr(hcu_flashmla, "flash_mla_sparse_fwd", sparse_fwd)
    monkeypatch.setattr(
        hcu_flashmla,
        "flash_mla_with_kvcache",
        lambda **kwargs: (kwargs, None),
    )
    monkeypatch.setattr(hcu_flashmla, "get_mla_metadata", lambda *a, **k: None)

    platform = SimpleNamespace(is_rocm=lambda: True)
    module = _module(
        adapter.TARGET_MODULE,
        FlashMLASparseMetadataBuilder=FlashMLASparseMetadataBuilder,
        FlashMLASparseImpl=FlashMLASparseImpl,
        current_platform=platform,
        torch=torch,
    )
    assert adapter.apply_to_module(module)
    impl = FlashMLASparseImpl()
    q = torch.ones(2, 2, 4)
    cache = torch.ones(2, 4)
    indices = torch.zeros(2, 3, dtype=torch.int64)
    topk_length = torch.tensor([1, 2])
    output = impl._bf16_flash_mla_kernel(
        q,
        cache,
        indices,
        topk_length,
    )
    assert output.shape == (2, 2, 4)
    assert calls[-1][-1] is topk_length

    platform.is_rocm = lambda: False
    assert (
        impl._bf16_flash_mla_kernel(q, cache, indices, topk_length)
        == "official-bf16"
    )
    assert calls[-1][-1] is topk_length


def test_attention_direct_forward_preserves_cpu_values_and_query_device():
    from vllm_hcu.model_executor.layers import attention_forward_runtime as runtime
    from vllm_hcu.model_executor.layers.mla_runtime import torch as runtime_torch

    assert runtime_torch is torch
    calls = {}

    class Impl:
        def do_kv_cache_update(self, layer, key, value, cache, slots):
            calls["dummy_device"] = key.device

    def attention(query, key, value, output, layer_name, **kwargs):
        output.copy_(query + key + value)

    upstream = SimpleNamespace(
        _encode_layer_name=lambda value: value,
        _resolve_layer_name=lambda value: value,
        get_attention_context=lambda value: (
            None,
            SimpleNamespace(impl=Impl()),
            (torch.empty(0), torch.empty(0)),
            torch.tensor([0]),
        ),
        unified_attention_with_output=attention,
    )
    self = SimpleNamespace(
        calculate_kv_scales=False,
        query_quant=None,
        num_heads=1,
        num_kv_heads=1,
        head_size=2,
        head_size_v=2,
        layer_name="layer",
        kv_cache_dtype="auto",
        attn_backend=SimpleNamespace(forward_includes_kv_cache_update=False),
        kv_sharing_target_layer_name=None,
    )
    query = torch.tensor([[1.0, 2.0]])
    key = torch.tensor([[3.0, 4.0]])
    value = torch.tensor([[5.0, 6.0]])
    output = runtime.attention_forward(
        upstream,
        self,
        query,
        key,
        value,
        output_dtype=torch.float64,
    )
    torch.testing.assert_close(
        output, torch.tensor([[9.0, 12.0]], dtype=torch.float64)
    )
    assert output.dtype is torch.float64
    assert calls["dummy_device"] == query.device


@pytest.mark.parametrize(
    ("kv_axis", "cache_shape"),
    [(0, (3, 2, 4)), (1, (2, 3, 4))],
)
def test_fused_attention_rejects_incompatible_stacked_kv_axis(
    kv_axis: int,
    cache_shape: tuple[int, ...],
):
    from vllm_hcu.model_executor.layers import kv_cache_utils as runtime

    with pytest.raises(ValueError, match="stacked KV cache dimension"):
        runtime.split_kv_cache(torch.empty(cache_shape), kv_axis=kv_axis)


def test_fused_kv_store_routes_block_first_cache_to_stride_aware_writer(
    monkeypatch: pytest.MonkeyPatch,
):
    if importlib.util.find_spec("vllm_hcu.hcu_ops") is None:
        pytest.skip("source-only test environment has no HCU native extension")
    try:
        runtime = importlib.import_module(
            "vllm_hcu.model_executor.layers.attention_runtime"
        )
        forward_context_module = importlib.import_module("vllm.forward_context")
        hcu_platform = importlib.import_module("vllm_hcu.platforms.hcu")
    except (ImportError, RuntimeError):
        pytest.skip("vLLM custom-op registry is unavailable in this environment")

    num_tokens, num_blocks, block_size = 2, 3, 4
    q_size, kv_size, head_size = 4, 2, 2
    kv_cache = torch.empty(num_blocks, 2, block_size, 1, head_size)
    slot_mapping = torch.tensor([0, 5])
    layer = SimpleNamespace(
        kv_cache=kv_cache,
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    context = SimpleNamespace(
        slot_mapping={"layer": slot_mapping},
        no_compile_layers={"layer": layer},
    )
    monkeypatch.setattr(forward_context_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(hcu_platform, "get_hcu_flash_attn_mode", lambda: "cutlass")

    lightop_calls: list[dict[str, object]] = []
    lightop_module = ModuleType("lightop")

    def fake_lightop(*args, **kwargs):
        del args
        lightop_calls.append(kwargs)
        assert kwargs["kv_cache_loc"] is None
        assert kwargs["k_buffer"].numel() == 0
        assert kwargs["v_buffer"].numel() == 0
        return (
            torch.arange(num_tokens * q_size, dtype=torch.float32).reshape(
                num_tokens, q_size
            ),
            torch.arange(num_tokens * kv_size, dtype=torch.float32).reshape(
                num_tokens, kv_size
            ),
            torch.arange(num_tokens * kv_size, dtype=torch.float32)
            .reshape(num_tokens, kv_size)
            .add(100),
        )

    lightop_module.split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant = (
        fake_lightop
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop_module)

    writer_calls: list[tuple[object, ...]] = []
    aiter_cache_module = ModuleType("aiter.ops.cache")
    aiter_cache_module.reshape_and_cache_flash = (
        lambda *args: writer_calls.append(args)
    )
    monkeypatch.setitem(sys.modules, "aiter.ops.cache", aiter_cache_module)

    q, key, value = runtime.fused_qkv_split_rmsnorm_rope_kv_store_impl(
        torch.zeros(num_tokens, q_size + 2 * kv_size),
        torch.arange(num_tokens),
        "layer",
        "auto",
        torch.empty(num_tokens, head_size),
        torch.ones(head_size),
        torch.ones(head_size),
        1e-5,
        head_size,
        head_size,
        q_size,
        kv_size,
        block_size,
    )

    assert len(lightop_calls) == 1
    assert len(writer_calls) == 1
    writer_key, writer_value, key_cache, value_cache, writer_slots, *_ = (
        writer_calls[0]
    )
    assert writer_key is key
    assert writer_value is value
    assert writer_slots is slot_mapping
    assert key_cache.stride(0) == 2 * block_size * head_size
    assert value_cache.stride(0) == 2 * block_size * head_size
    assert q.shape == (num_tokens, q_size // head_size, head_size)


def test_fla_chunk_o_feature_off_is_numerically_identical(monkeypatch):
    adapter = _adapter("patch_fla_chunk_o")

    def original(q, k, v, h, g=None, scale=None, cu_seqlens=None,
                 chunk_indices=None, chunk_size=64, core_attn_out=None):
        return q + k + v + h

    module = _module(
        adapter.TARGET_MODULE,
        FLA_CHUNK_SIZE=64,
        chunk_fwd_o=original,
        torch=torch,
    )
    assert adapter.apply_to_module(module)
    assert not adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_AITER_FLA", False)
    x = torch.arange(4, dtype=torch.float32)
    torch.testing.assert_close(module.chunk_fwd_o(x, x, x, x), original(x, x, x, x))


def test_fla_chunk_delta_h_enabled_missing_aiter_fails_clearly(monkeypatch):
    adapter = _adapter("patch_fla_chunk_delta_h")

    def original(k, w, u, g=None, gk=None, initial_state=None,
                 output_final_state=False, chunk_size=64, save_new_value=True,
                 cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
                 use_exp2=False):
        return k, u, None

    module = _module(
        adapter.TARGET_MODULE,
        FLA_CHUNK_SIZE=64,
        chunk_gated_delta_rule_fwd_h=original,
        torch=torch,
    )
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_AITER_FLA", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.triton.fla.vllm.chunk_delta_h",
        ModuleType("aiter.ops.triton.fla.vllm.chunk_delta_h"),
    )
    with pytest.raises(RuntimeError, match="enabled but unavailable"):
        module.chunk_gated_delta_rule_fwd_h(
            torch.empty(1, 1, 1, 1),
            torch.empty(1),
            torch.empty(1, 1, 1, 1),
        )


def test_mamba_nn_sharded_loader_cpu_numeric():
    from vllm_hcu.model_executor.layers.mamba_runtime import (
        mamba_v2_nn_sharded_weight_loader,
    )

    param = torch.zeros(2, 4)
    loaded = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    loader = mamba_v2_nn_sharded_weight_loader([(4, 0, False)], 1, 0)
    loader(param, loaded)
    torch.testing.assert_close(param, loaded)


def test_mamba1_conv_weight_keeps_causal_conv_layout(monkeypatch):
    adapter = _adapter("patch_mamba_mixer")

    class Weight:
        def __init__(self, data):
            self.data = data
            self.weight_loader = lambda param, loaded: None

    class MambaMixer:
        def __init__(self, hidden_size, ssm_state_size, conv_kernel_size,
                     intermediate_size, time_step_rank, use_conv_bias, use_bias,
                     use_rms_norm, rms_norm_has_weight=True, rms_norm_eps=1e-5,
                     activation="silu", is_lora_enabled=False, model_config=None,
                     cache_config=None, prefix=""):
            del hidden_size, ssm_state_size, time_step_rank, use_conv_bias, use_bias
            del use_rms_norm, rms_norm_has_weight, rms_norm_eps, activation
            del is_lora_enabled, model_config, cache_config, prefix
            self.conv1d = SimpleNamespace(
                weight=Weight(
                    torch.empty(conv_kernel_size, 1, intermediate_size)
                ),
                tp_rank=0,
            )

    module = _module(adapter.TARGET_MODULE, MambaMixer=MambaMixer)
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_USE_NN", True)
    instance = MambaMixer(4, 1, 4, 6, 1, True, False, True)
    assert instance.conv1d.weight.data.shape == (6, 1, 4)

    loaded = torch.arange(24, dtype=torch.float32).reshape(6, 1, 4)
    instance.conv1d.weight.weight_loader(instance.conv1d.weight, loaded)
    torch.testing.assert_close(instance.conv1d.weight.data, loaded)


def test_mamba_mixer_init_converts_conv_buffer_to_nn_layout(monkeypatch):
    adapter = _adapter("patch_mamba_mixer2")

    class MambaMixer2:
        def __init__(self, hidden_size, ssm_state_size, conv_kernel_size,
                     intermediate_size, use_conv_bias, use_bias, n_groups=1,
                     num_heads=128, head_dim=64, rms_norm_eps=1e-5,
                     activation="silu", use_rms_norm=True, model_config=None,
                     cache_config=None, quant_config=None, prefix=""):
            del hidden_size, ssm_state_size, intermediate_size, use_conv_bias, use_bias
            del n_groups, num_heads, head_dim, rms_norm_eps, activation, use_rms_norm
            del model_config, cache_config, quant_config, prefix
            self.conv1d = SimpleNamespace(
                weight=torch.arange(12, dtype=torch.float32).reshape(3, 1, conv_kernel_size)
            )
            self.conv_weights = self.conv1d.weight.view(3, conv_kernel_size)

    def loader(shard_spec, tp_size, tp_rank):
        return lambda param, weight: None

    module = _module(
        adapter.TARGET_MODULE,
        MambaMixer2=MambaMixer2,
        mamba_v2_sharded_weight_loader=loader,
    )
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_USE_NN", True)
    instance = MambaMixer2(4, 1, 4, 4, False, False)
    assert instance.conv_weights.shape == (4, 3)
    torch.testing.assert_close(
        instance.conv_weights, instance.conv1d.weight.squeeze(1).T.contiguous()
    )


def test_gdn_feature_off_delegates_official_state_dtype(monkeypatch):
    adapter = _adapter("patch_gdn_base")
    calls: list[str] = []

    def causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        pad_slot_id,
        null_block_id,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
        block_size_to_align,
        metadata,
        validate_data,
    ):
        return "official-causal"

    def recurrent(*args, **kwargs):
        return "official-recurrent"

    def sigmoid(*args, **kwargs):
        return "official-sigmoid"

    def causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        max_query_len,
        null_block_id,
        block_idx_last_scheduled_token,
        initial_state_idx,
        validate_data,
    ):
        return "official-update"

    class GatedDeltaNetAttention:
        def get_state_dtype(self):
            calls.append("official-state-dtype")
            return "official-dtype"

    class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
        pass

    class Calculator:
        @staticmethod
        def gated_delta_net_state_dtype(*args):
            raise AssertionError("feature-off path must not recompute upstream dtype")

    module = _module(
        adapter.TARGET_MODULE,
        causal_conv1d_fn=causal_conv1d_fn,
        causal_conv1d_update=causal_conv1d_update,
        fused_recurrent_gated_delta_rule_packed_decode=recurrent,
        fused_sigmoid_gating_delta_rule_update=sigmoid,
        QwenGatedDeltaNetAttention=QwenGatedDeltaNetAttention,
        MambaStateDtypeCalculator=Calculator,
    )
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_MAMBA_SSM_CACHE_DTYPE", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    instance = QwenGatedDeltaNetAttention()
    assert instance.get_state_dtype() == "official-dtype"
    assert calls == ["official-state-dtype"]


def test_gdn_runtime_adapter_has_stable_patch_id():
    assert _adapter("patch_gdn_causal_conv1d").PATCH_ID == (
        "worker.op_opt.mamba.gdn.causal_conv1d"
    )
    assert _adapter("patch_gdn_base").PATCH_ID == (
        "worker.op_opt.mamba.gdn.base_state_dtype"
    )
    assert _adapter("patch_gdn_linear_attention").PATCH_ID == (
        "worker.op_opt.mamba.gdn.qwen_kernel_bindings"
    )


def test_gdn_nn_layout_normalizes_all_conv_weight_consumers(monkeypatch):
    causal_adapter = _adapter("patch_gdn_causal_conv1d")
    qwen_adapter = _adapter("patch_gdn_linear_attention")
    captured = {}

    def causal_fn(*args, **kwargs):
        captured["causal_fn"] = args[1]
        return args[1]

    causal_fn.__signature__ = inspect.signature(_gdn_causal_conv1d_fn)  # type: ignore[attr-defined]

    def causal_update(*args, **kwargs):
        captured["causal_update"] = args[2]
        return args[2]

    causal_update.__signature__ = inspect.signature(_gdn_causal_conv1d_update)  # type: ignore[attr-defined]

    def aiter_update(
        x,
        num_actual_tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        ba,
        z_out,
        core_attn_out,
        conv_state,
        weight,
        bias=None,
        activation=None,
        conv_state_indices=None,
        num_accepted_tokens=None,
        query_start_loc=None,
        max_query_len=-1,
        pad_slot_id=-1,
        block_idx_last_scheduled_token=None,
        initial_state_idx=None,
        validate_data=False,
        qkvz_layout="interleaved",
    ):
        del (
            x,
            num_actual_tokens,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            ba,
            z_out,
            core_attn_out,
            conv_state,
            bias,
            activation,
            conv_state_indices,
            num_accepted_tokens,
            query_start_loc,
            max_query_len,
            pad_slot_id,
            block_idx_last_scheduled_token,
            initial_state_idx,
            validate_data,
            qkvz_layout,
        )
        captured["aiter_update"] = weight
        return weight

    class GatedDeltaNetAttention:
        def get_state_dtype(self):
            return "official-dtype"

    causal_module = _module(
        causal_adapter.TARGET_MODULE,
        causal_conv1d_fn=causal_fn,
        causal_conv1d_update=causal_update,
    )
    qwen_module = _module(
        qwen_adapter.TARGET_MODULE,
        GDN_AITER_TRITON_AVAILABLE=True,
        gdn_aiter_fused_reshape_causal_conv1d_update_single_token=aiter_update,
        fused_recurrent_gated_delta_rule_packed_decode=lambda *a, **k: "official-recurrent",
        fused_sigmoid_gating_delta_rule_update=lambda *a, **k: "official-sigmoid",
        GatedDeltaNetAttention=GatedDeltaNetAttention,
        MambaStateDtypeCalculator=SimpleNamespace(
            gated_delta_net_state_dtype=lambda *a: "calculator"
        ),
    )
    causal_adapter.apply_to_module(causal_module)
    qwen_adapter.apply_to_module(qwen_module)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_USE_NN", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D", False)

    physical_weight = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    expected = physical_weight.T.contiguous()
    conv_state = torch.empty(1, 8, 3)
    x_fn = torch.empty(8, 2)
    x_update = torch.empty(2, 8)

    causal_module.causal_conv1d_fn(
        x_fn,
        physical_weight,
        None,
        conv_states=conv_state,
        query_start_loc=torch.tensor([0, 2]),
    )
    causal_module.causal_conv1d_update(x_update, conv_state, physical_weight)
    qwen_module.gdn_aiter_fused_reshape_causal_conv1d_update_single_token(
        torch.empty(1),
        1,
        1,
        1,
        1,
        1,
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        conv_state,
        physical_weight,
        None,
        "silu",
    )

    for key in ("causal_fn", "causal_update", "aiter_update"):
        torch.testing.assert_close(captured[key], expected)

    monkeypatch.setattr(henvs, "VLLM_USE_NN", False)
    logical_weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    assert (
        causal_module.causal_conv1d_update(
            x_update, conv_state, logical_weight
        )
        is logical_weight
    )


def test_gdn_recurrent_and_sigmoid_remain_target_owned(monkeypatch):
    adapter = _adapter("patch_gdn_linear_attention")

    _install_fake_module(
        monkeypatch,
        "aiter.ops.triton.fla.fused_recurrent",
        fused_recurrent_gated_delta_rule_packed_decode=lambda *a, **k: pytest.fail(
            "retired HCU recurrent path must not be imported"
        ),
    )
    _install_fake_module(
        monkeypatch,
        "aiter.ops.triton.fla.fused_sigmoid_gating",
        fused_sigmoid_gating_delta_rule_update=lambda *a, **k: pytest.fail(
            "retired HCU sigmoid path must not be imported"
        ),
    )

    def recurrent(*args, **kwargs):
        del args, kwargs
        return "official-recurrent"

    def sigmoid(*args, **kwargs):
        del args, kwargs
        return "official-sigmoid"

    module = _module(
        adapter.TARGET_MODULE,
        GDN_AITER_TRITON_AVAILABLE=False,
        fused_recurrent_gated_delta_rule_packed_decode=recurrent,
        fused_sigmoid_gating_delta_rule_update=sigmoid,
    )
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    assert module.fused_recurrent_gated_delta_rule_packed_decode is recurrent
    assert module.fused_sigmoid_gating_delta_rule_update is sigmoid
    assert module.fused_recurrent_gated_delta_rule_packed_decode() == "official-recurrent"
    assert module.fused_sigmoid_gating_delta_rule_update() == "official-sigmoid"


def test_gdn_state_dtype_feature_on_uses_auto_ssm_dtype(monkeypatch):
    adapter = _adapter("patch_gdn_base")
    calls = []

    class GatedDeltaNetAttention:
        model_config = SimpleNamespace(dtype=torch.float16)
        cache_config = SimpleNamespace(
            mamba_cache_dtype="float32",
            mamba_ssm_cache_dtype="float32",
        )

        def get_state_dtype(self):
            return "official-dtype"

    class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
        pass

    class Calculator:
        @staticmethod
        def gated_delta_net_state_dtype(model_dtype, cache_dtype, ssm_dtype):
            calls.append((model_dtype, cache_dtype, ssm_dtype))
            return "hcu-dtype"

    module = _module(
        adapter.TARGET_MODULE,
        QwenGatedDeltaNetAttention=QwenGatedDeltaNetAttention,
    )
    _install_fake_module(
        monkeypatch,
        "vllm.model_executor.layers.mamba.mamba_utils",
        MambaStateDtypeCalculator=Calculator,
    )
    adapter.apply_to_module(module)
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_MAMBA_SSM_CACHE_DTYPE", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    assert QwenGatedDeltaNetAttention().get_state_dtype() == "hcu-dtype"
    assert GatedDeltaNetAttention().get_state_dtype() == "official-dtype"
    assert calls == [(torch.float16, "float32", "auto")]


def test_causal_conv_metadata_exact_callback():
    adapter = _adapter("patch_attention_backend_utils")

    def compute(query_start_loc_p_cpu, *, device):
        return {8: {"device": device}}, None, None

    def split_batch(
        common_attn_metadata,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        del (
            common_attn_metadata,
            decode_threshold,
            require_uniform,
            treat_short_extends_as_decodes,
        )
        return "official"

    module = _module(
        adapter.TARGET_MODULE,
        compute_causal_conv1d_metadata=compute,
        split_decodes_and_prefills=split_batch,
    )
    assert adapter.apply_to_module(module)
    result = module.compute_causal_conv1d_metadata(
        torch.tensor([0, 2, 5]), device=torch.device("cpu")
    )
    assert result[0]["seqlens"] == [2, 3]
    assert not adapter.apply_to_module(module)


def test_uniform_short_extends_use_prefill_classification():
    adapter = _adapter("patch_attention_backend_utils")

    def compute(query_start_loc_p_cpu, *, device):
        return {}, None, None

    def split_batch(
        common_attn_metadata,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        del (
            common_attn_metadata,
            decode_threshold,
            require_uniform,
            treat_short_extends_as_decodes,
        )
        return "official"

    module = _module(
        adapter.TARGET_MODULE,
        compute_causal_conv1d_metadata=compute,
        split_decodes_and_prefills=split_batch,
    )
    adapter.apply_to_module(module)
    common = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 2, 4]),
        is_prefilling=torch.tensor([False, True]),
        num_reqs=2,
        num_actual_tokens=4,
    )
    assert module.split_decodes_and_prefills(
        common,
        decode_threshold=2,
        require_uniform=True,
        treat_short_extends_as_decodes=False,
    ) == (1, 1, 2, 2)
    assert module.split_decodes_and_prefills(
        common,
        decode_threshold=2,
        require_uniform=True,
        treat_short_extends_as_decodes=True,
    ) == "official"


def test_common_attention_metadata_accepts_hcu_fields_and_unpads():
    adapter = _adapter("patch_attention_backend")

    class CommonAttentionMetadata:
        def __init__(self, query_start_loc, query_start_loc_cpu, seq_lens,
                     num_reqs, num_actual_tokens, max_query_len, max_seq_len,
                     block_table_tensor, slot_mapping):
            self.query_start_loc = query_start_loc
            self.query_start_loc_cpu = query_start_loc_cpu
            self.seq_lens = seq_lens
            self.num_reqs = num_reqs
            self.num_actual_tokens = num_actual_tokens
            self.max_query_len = max_query_len
            self.max_seq_len = max_seq_len
            self.block_table_tensor = block_table_tensor
            self.slot_mapping = slot_mapping

        def unpadded(self, num_actual_tokens, num_actual_reqs):
            return CommonAttentionMetadata(
                self.query_start_loc, self.query_start_loc_cpu, self.seq_lens,
                num_actual_reqs, num_actual_tokens, self.max_query_len,
                self.max_seq_len, self.block_table_tensor,
                self.slot_mapping[:num_actual_tokens],
            )

        def replace(self, **kwargs):
            values = dict(
                query_start_loc=self.query_start_loc,
                query_start_loc_cpu=self.query_start_loc_cpu,
                seq_lens=self.seq_lens,
                num_reqs=self.num_reqs,
                num_actual_tokens=self.num_actual_tokens,
                max_query_len=self.max_query_len,
                max_seq_len=self.max_seq_len,
                block_table_tensor=self.block_table_tensor,
                slot_mapping=self.slot_mapping,
            )
            values.update(kwargs)
            return CommonAttentionMetadata(**values)

    module = _module(
        adapter.TARGET_MODULE,
        CommonAttentionMetadata=CommonAttentionMetadata,
    )
    adapter.apply_to_module(module)
    tensor = torch.arange(4)
    metadata = CommonAttentionMetadata(
        tensor, tensor, tensor, 1, 4, 4, 4, tensor, tensor,
        num_kv_actual_tokens=7, gather_indexes_tensor=tensor,
    )
    assert metadata.num_kv_actual_tokens == 7
    assert metadata.gather_indexes_tensor is tensor
    assert metadata.replace(max_seq_len=9).num_kv_actual_tokens == 7
    assert metadata.unpadded(2, 1).num_kv_actual_tokens == 2
    assert module.CpCommonAttentionMetadata.__module__.startswith("vllm_hcu")


def test_indexer_wrappers_filter_zero_chunks_and_propagate_kv_count():
    adapter = _adapter("patch_mla_indexer")

    def split_chunks(seq_lens_cpu, query_lens_cpu, workspace_size,
                     max_logits_bytes, request_offset=0):
        return [(slice(0, 1), slice(0, 0)), (slice(1, 2), slice(0, 2))]

    def split_batch(common_attn_metadata, decode_threshold=1,
                    require_uniform=False, treat_short_extends_as_decodes=True):
        return treat_short_extends_as_decodes

    class Builder:
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return SimpleNamespace(decode=None)

    module = _module(
        adapter.TARGET_MODULE,
        split_indexer_prefill_chunks=split_chunks,
        split_decodes_and_prefills=split_batch,
        DeepseekV32IndexerMetadataBuilder=Builder,
        current_platform=SimpleNamespace(is_rocm=lambda: False),
    )
    adapter.apply_to_module(module)
    chunks = module.split_indexer_prefill_chunks(None, None, 1, 1)
    assert chunks == [(slice(1, 2), slice(0, 2))]
    common = SimpleNamespace(
        is_prefilling=torch.tensor([True]), num_actual_tokens=2,
        num_kv_actual_tokens=5,
    )
    assert module.split_decodes_and_prefills(common) is False
    assert Builder().build(0, common).num_kv_actual_tokens == 5


def test_mla_forward_slices_kv_with_independent_token_count():
    from vllm_hcu.model_executor.layers.mla_runtime import mla_forward_impl

    captured = {}

    class SparseBase:
        pass

    class Impl:
        dcp_world_size = 1

        def forward_mha(self, q, k, pe, kv_cache, metadata, scale, output):
            captured["q"] = q.shape[0]
            captured["k"] = k.shape[0]
            output.zero_()

    upstream = SimpleNamespace(
        _detect_output_quant_key=lambda *args: None,
        is_quantized_kv_cache=lambda value: False,
        SparseMLAAttentionImpl=SparseBase,
    )
    self = SimpleNamespace(
        impl=Impl(), kv_cache_dtype="auto", num_heads=1, v_head_dim=1,
        qk_nope_head_dim=1, qk_rope_head_dim=1,
        chunked_prefill_workspace_size=4, _k_scale=torch.tensor(1.0),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=2, num_kv_actual_tokens=4,
        num_decodes=0, num_prefills=1, num_decode_tokens=0,
    )
    q = torch.ones(2, 1, 2)
    k = torch.ones(4, 1)
    pe = torch.ones(4, 1, 1)
    output = torch.empty(2, 1)
    result = mla_forward_impl(
        upstream, self, q, k, pe, torch.empty(0), metadata, output
    )
    assert result is output
    assert captured == {"q": 2, "k": 4}


def test_mla_upstream_skip_topk_contract_and_feature_off_delegation(monkeypatch):
    adapter = _adapter("patch_mla_layer")

    class MultiHeadLatentAttentionWrapper:
        def __init__(
            self,
            hidden_size,
            num_heads,
            scale,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            mla_modules,
            cache_config=None,
            quant_config=None,
            prefix="",
            skip_topk=False,
            non_causal_multi_token_decode=False,
            allow_short_prefill_indexer_scoring_skip=False,
        ):
            self.skip_topk = skip_topk
            self.non_causal_multi_token_decode = non_causal_multi_token_decode
            self.allow_short_prefill_indexer_scoring_skip = (
                allow_short_prefill_indexer_scoring_skip
            )

        def forward(self, positions, hidden_states, llama_4_scaling=None):
            return (
                "official",
                self.skip_topk,
                positions,
                hidden_states,
                llama_4_scaling,
            )

    module = _module(
        adapter.TARGET_MODULE,
        MultiHeadLatentAttentionWrapper=MultiHeadLatentAttentionWrapper,
    )
    _install_fake_module(
        monkeypatch,
        "vllm.config",
        get_current_vllm_config_or_none=lambda: None,
    )
    assert adapter.apply_to_module(module) is True
    instance = MultiHeadLatentAttentionWrapper(
        16,
        2,
        0.5,
        4,
        4,
        8,
        4,
        4,
        object(),
        skip_topk=True,
        non_causal_multi_token_decode=True,
        allow_short_prefill_indexer_scoring_skip=True,
    )
    assert instance.skip_topk is True
    assert instance.non_causal_multi_token_decode is True
    assert instance.allow_short_prefill_indexer_scoring_skip is True
    assert instance.forward("positions", "hidden") == (
        "official",
        True,
        "positions",
        "hidden",
        None,
    )


def test_mla_attention_v028_dcp_contract_forwards_or_fails_closed(monkeypatch):
    adapter = _adapter("patch_mla_attention")
    calls = []

    class MLAAttention:
        def __init__(
            self,
            num_heads,
            scale,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            kv_b_proj,
            dcp_q_replicate=False,
            cache_config=None,
            quant_config=None,
            prefix="",
            attn_backend=None,
            use_sparse=False,
            indexer=None,
            topk_indices_buffer=None,
            non_causal_multi_token_decode=False,
            sliding_window=None,
            prefill_backend_cls=None,
            **extra_impl_args,
        ):
            del (
                num_heads,
                scale,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                q_lora_rank,
                kv_lora_rank,
                kv_b_proj,
                dcp_q_replicate,
                cache_config,
                quant_config,
                prefix,
                attn_backend,
                use_sparse,
                indexer,
                topk_indices_buffer,
                non_causal_multi_token_decode,
                sliding_window,
                prefill_backend_cls,
                extra_impl_args,
            )

        def forward_impl(
            self,
            q,
            k_c_normed,
            k_pe,
            kv_cache,
            attn_metadata,
            output,
            output_scale=None,
            output_block_scale=None,
            quant_group_size=None,
            quant_scale_ue8m0=None,
            quant_col_major=None,
            quant_tma_aligned=None,
            q_dcp_replicated=None,
        ):
            del self
            calls.append(
                (
                    q,
                    k_c_normed,
                    k_pe,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                    quant_group_size,
                    quant_scale_ue8m0,
                    quant_col_major,
                    quant_tma_aligned,
                    q_dcp_replicated,
                )
            )
            return "official"

        def process_weights_after_loading(self, act_dtype):
            return act_dtype

    class MLACommonMetadata:
        def __init__(self, num_actual_tokens):
            self.num_actual_tokens = num_actual_tokens

    class MLACommonMetadataBuilder:
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            del self, common_prefix_len, fast_build
            return MLACommonMetadata(common_attn_metadata.num_actual_tokens)

    def split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        return (
            common_attn_metadata,
            decode_threshold,
            require_uniform,
            treat_short_extends_as_decodes,
        )

    module = _module(
        adapter.TARGET_MODULE,
        MLAAttention=MLAAttention,
        MLACommonMetadata=MLACommonMetadata,
        MLACommonMetadataBuilder=MLACommonMetadataBuilder,
        split_decodes_and_prefills=split_decodes_and_prefills,
    )
    _install_fake_module(
        monkeypatch,
        "vllm.config",
        get_current_vllm_config_or_none=lambda: None,
    )
    feature_config = SimpleNamespace(enable_lightly_cp=False)
    monkeypatch.setattr(adapter, "get_hcu_config", lambda _: feature_config)

    assert adapter.apply_to_module(module) is True
    instance = MLAAttention(1, 1.0, 1, 1, 1, None, 1, object())
    q_dcp_replicated = object()
    assert instance.forward_impl(
        "q", "k", "pe", "cache", "metadata", "output",
        q_dcp_replicated=q_dcp_replicated,
    ) == "official"
    assert calls[-1][-1] is q_dcp_replicated

    feature_config.enable_lightly_cp = True
    with pytest.raises(NotImplementedError, match="DCP query replication"):
        instance.forward_impl(
            "q", "k", "pe", "cache", "metadata", "output",
            q_dcp_replicated=q_dcp_replicated,
        )


def test_wrong_exact_module_name_fails_before_mutation():
    adapter = _adapter("patch_fla_chunk_o")
    wrong = _module("vllm.wrong")
    with pytest.raises(RuntimeError, match="expected module"):
        adapter.apply_to_module(wrong)
