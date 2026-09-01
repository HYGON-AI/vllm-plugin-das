# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for target-owned MLA execution and HCU CAT routing."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers import mla_runtime


class _GenericStub:
    @classmethod
    def __class_getitem__(cls, item):
        del item
        return cls


class _MLACommonImplStub(_GenericStub):
    supports_pcp = False

    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        logits_soft_cap,
        attn_type,
        kv_sharing_target_layer_name,
        **mla_args,
    ):
        del (
            num_heads,
            head_size,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
        )
        self.scale = scale
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank = mla_args.get("kv_lora_rank", 128)
        self.supports_quant_query_input = True


def _install_stub(monkeypatch, name, **values):
    parts = name.split(".")
    for index in range(1, len(parts) + 1):
        module_name = ".".join(parts[:index])
        module = sys.modules.get(module_name)
        if module is None or index == len(parts):
            module = ModuleType(module_name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, module_name, module)
        if index > 1:
            parent_name = ".".join(parts[: index - 1])
            parent = sys.modules[parent_name]
            monkeypatch.setattr(
                parent,
                parts[index - 1],
                module,
                raising=False,
            )
    module.__dict__.update(values)
    return module


@pytest.fixture
def cpu_flashmla(monkeypatch):
    class AttentionCGSupport:
        UNIFORM_BATCH = "uniform-batch"

    class AttentionType:
        DECODER = "decoder"

    class QueryLenSupport:
        UNIFORM = "uniform"

    _install_stub(monkeypatch, "vllm.envs", VLLM_BATCH_INVARIANT=False)
    _install_stub(monkeypatch, "vllm.config", VllmConfig=type("VllmConfig", (), {}))
    _install_stub(
        monkeypatch,
        "vllm.config.cache",
        CacheDType=type("CacheDType", (), {}),
    )
    _install_stub(
        monkeypatch,
        "vllm.logger",
        init_logger=lambda name: SimpleNamespace(name=name),
    )
    _install_stub(
        monkeypatch,
        "vllm.model_executor.layers.attention.mla_attention",
        MLACommonBackend=_GenericStub,
        MLACommonDecodeMetadata=_GenericStub,
        MLACommonImpl=_MLACommonImplStub,
        MLACommonMetadata=_GenericStub,
        MLACommonMetadataBuilder=_GenericStub,
        QueryLenSupport=QueryLenSupport,
    )
    _install_stub(
        monkeypatch,
        "vllm.platforms.interface",
        DeviceCapability=type("DeviceCapability", (), {}),
    )
    _install_stub(
        monkeypatch,
        "vllm.utils.platform_utils",
        num_compute_units=lambda device_index: 1,
    )
    _install_stub(
        monkeypatch,
        "vllm.utils.torch_utils",
        is_quantized_kv_cache=lambda cache_dtype: str(cache_dtype).startswith(
            "fp8"
        ),
    )
    _install_stub(
        monkeypatch,
        "vllm.v1.attention.backend",
        AttentionCGSupport=AttentionCGSupport,
        AttentionLayer=type("AttentionLayer", (), {}),
        AttentionType=AttentionType,
        MultipleOf=type("MultipleOf", (), {}),
    )
    _install_stub(
        monkeypatch,
        "vllm.v1.attention.backends.utils",
        reshape_attn_output_for_spec_decode=lambda output: output,
        reshape_query_for_spec_decode=lambda query, num_decodes: query,
    )
    _install_stub(
        monkeypatch,
        "vllm.v1.kv_cache_interface",
        AttentionSpec=type("AttentionSpec", (), {}),
    )
    _install_stub(
        monkeypatch,
        "vllm_hcu.v1.attention.ops.flashmla",
        FlashMLASchedMeta=type("FlashMLASchedMeta", (), {}),
        flash_mla_with_kvcache=lambda **kwargs: (kwargs, None),
        flash_mla_with_kvcache_fp8=lambda **kwargs: (kwargs, None),
        flash_mla_with_kvcache_fp8_with_cat=lambda **kwargs: (kwargs, None),
        get_mla_metadata=lambda *args, **kwargs: (None, None),
        get_mla_metadata_dense_fp8=lambda *args, **kwargs: (None, None),
        is_flashmla_dense_supported=lambda: (True, None),
    )
    _install_stub(
        monkeypatch,
        "vllm_hcu.platforms.envs",
        VLLM_HCU_USE_CAT_MLA=False,
        VLLM_USE_OPT_CAT=False,
    )
    _install_stub(
        monkeypatch,
        "vllm_hcu.platforms.hcu",
        on_gfx938=lambda: False,
    )

    module_name = "_vllm_hcu_cpu_test_flashmla"
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/v1/attention/backends/mla/flashmla.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _adapter():
    return importlib.import_module(
        "vllm_hcu.patch.worker.op_opt.patch_mla_attention"
    )


def _fake_mla_module(adapter, target_calls, event_log=None):
    split_calls = []
    full_forward_calls = []

    def split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        split_calls.append(
            (
                common_attn_metadata,
                decode_threshold,
                require_uniform,
                treat_short_extends_as_decodes,
            )
        )
        return treat_short_extends_as_decodes

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
            cache_config=None,
            quant_config=None,
            prefix="",
            attn_backend=None,
            use_sparse=False,
            indexer=None,
            topk_indices_buffer=None,
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
                cache_config,
                quant_config,
                prefix,
                attn_backend,
                use_sparse,
                indexer,
                topk_indices_buffer,
                extra_impl_args,
            )
            self.use_direct_call = False

        def forward(
            self,
            q,
            kv_c_normed,
            k_pe,
            output_shape=None,
        ):
            args = (q, kv_c_normed, k_pe, output_shape)
            full_forward_calls.append((self, args))
            if event_log is not None:
                event_log.append("opaque_forward")
            return "target-opaque-v0.25.1"

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
        ):
            args = (
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
            )
            target_calls.append((self, args))
            if event_log is not None:
                event_log.append("forward_impl")
            return "target-v0.25.1"

        def process_weights_after_loading(self, act_dtype):
            return act_dtype

    class MLACommonMetadata:
        def __init__(self, num_actual_tokens):
            self.num_actual_tokens = num_actual_tokens

    class MLACommonMetadataBuilder:
        def build(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build=False,
        ):
            del self, common_prefix_len, fast_build
            return SimpleNamespace(
                num_actual_tokens=common_attn_metadata.num_actual_tokens
            )

    module = ModuleType(adapter.TARGET_MODULE)
    module.MLAAttention = MLAAttention
    module.MLACommonMetadata = MLACommonMetadata
    module.MLACommonMetadataBuilder = MLACommonMetadataBuilder
    module.split_decodes_and_prefills = split_decodes_and_prefills
    module.split_calls = split_calls
    module.full_forward_calls = full_forward_calls
    module.get_forward_context = lambda: pytest.fail(
        "test did not install a forward context"
    )
    module._encode_layer_name = lambda value: value
    module.torch = torch
    module.current_platform = SimpleNamespace(
        is_rocm=lambda: pytest.fail(
            "feature ownership must not depend on the target platform"
        )
    )
    return module


def _forward_args():
    return tuple(object() for _ in range(12))


@pytest.mark.parametrize("transposed_input", [False, True])
def test_mla_weight_processing_falls_back_to_bf16_bmm_for_both_layouts(
    transposed_input,
):
    torch.manual_seed(7)
    kv_lora_rank = 2
    num_heads = 3
    qk_nope_head_dim = 4
    v_head_dim = 5
    logical_weight = torch.randn(
        kv_lora_rank,
        num_heads * (qk_nope_head_dim + v_head_dim),
        dtype=torch.bfloat16,
    )
    physical_weight = (
        logical_weight.T.contiguous() if transposed_input else logical_weight
    )
    triton_calls: list[object] = []
    scale_calls: list[object] = []
    upstream = SimpleNamespace(
        get_and_maybe_dequant_weights=lambda layer, out_dtype: physical_weight,
        rocm_aiter_ops=SimpleNamespace(
            triton_fp8_bmm=lambda *args, **kwargs: triton_calls.append(
                (args, kwargs)
            )
        ),
        should_load_quant_weights=lambda quant_method: False,
        set_default_quant_scales=lambda layer, register_buffer: scale_calls.append(
            (layer, register_buffer)
        ),
    )
    mla = SimpleNamespace(
        kv_b_proj=object(),
        kv_lora_rank=kv_lora_rank,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        is_aiter_triton_fp4_bmm_enabled=False,
        is_aiter_triton_fp8_bmm_enabled=False,
        quant_config=None,
    )

    mla_runtime.mla_process_weights_nn(upstream, mla, torch.bfloat16)

    reshaped = logical_weight.view(
        kv_lora_rank,
        num_heads,
        qk_nope_head_dim + v_head_dim,
    )
    expected_w_uk, expected_w_uv = reshaped.split(
        [qk_nope_head_dim, v_head_dim], dim=-1
    )
    torch.testing.assert_close(
        mla.W_UK_T,
        expected_w_uk.permute(1, 2, 0),
    )
    torch.testing.assert_close(
        mla.W_UV,
        expected_w_uv.transpose(0, 1),
    )
    query = torch.randn(num_heads, 2, qk_nope_head_dim, dtype=torch.bfloat16)
    torch.testing.assert_close(
        torch.bmm(query, mla.W_UK_T),
        torch.bmm(query, expected_w_uk.permute(1, 2, 0)),
    )
    assert triton_calls == []
    assert scale_calls == [(mla, False)]


def test_mla_channel_fp8_weight_is_dequantized_without_linear_forward():
    weight = torch.tensor(
        [[1.0, 2.0, -3.0], [4.0, -5.0, 6.0]],
        dtype=torch.float8_e4m3fn,
    )
    weight.input_dim = 0
    weight.output_dim = 1
    scale = torch.tensor([[0.25], [0.5], [0.125]], dtype=torch.float32)
    layer = SimpleNamespace(weight=weight, weight_scale=scale)
    upstream = SimpleNamespace(
        get_and_maybe_dequant_weights=lambda *args, **kwargs: pytest.fail(
            "channel FP8 dequantization must not execute the quantized linear"
        )
    )

    actual = mla_runtime._get_mla_kv_b_proj_weight(
        upstream,
        layer,
        torch.bfloat16,
    )

    expected = (weight.float() * scale.T).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected)


def test_mla_feature_off_delegates_exact_v0251_forward_on_rocm():
    adapter = _adapter()
    target_calls = []
    module = _fake_mla_module(adapter, target_calls)
    assert adapter.apply_to_module(module) is True

    instance = object.__new__(module.MLAAttention)
    instance._hcu_feature_config = SimpleNamespace(enable_lightly_cp=False)
    args = _forward_args()

    assert instance.forward_impl(*args) == "target-v0.25.1"
    assert target_calls == [(instance, args)]
    common = SimpleNamespace(is_prefilling=torch.tensor([True]))
    assert module.split_decodes_and_prefills(common, 3, True, True) is False
    assert module.split_calls == [(common, 3, True, False)]
    warmup = SimpleNamespace(is_prefilling=None)
    assert module.split_decodes_and_prefills(warmup, 3, True, True) is True
    assert module.split_calls[-1] == (warmup, 3, True, True)


@pytest.mark.parametrize(
    ("kv_cache_memory_bytes", "has_metadata", "expected_delegate"),
    (
        pytest.param(3 * 1024**3, False, False, id="manual-kv-profile"),
        pytest.param(3 * 1024**3, True, True, id="manual-kv-real-request"),
        pytest.param(None, False, True, id="automatic-kv-profile"),
        pytest.param(0, False, True, id="zero-kv-profile"),
    ),
)
def test_mla_manual_kv_profile_guard_preserves_upstream_paths(
    kv_cache_memory_bytes,
    has_metadata,
    expected_delegate,
):
    adapter = _adapter()
    target_calls = []
    module = _fake_mla_module(adapter, target_calls)
    assert adapter.apply_to_module(module) is True

    instance = object.__new__(module.MLAAttention)
    instance._hcu_feature_config = SimpleNamespace(enable_lightly_cp=False)
    instance._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=kv_cache_memory_bytes
        )
    )
    output = torch.ones(2, 3)
    metadata = object() if has_metadata else None
    args = (
        object(),
        object(),
        object(),
        object(),
        metadata,
        output,
    )

    result = instance.forward_impl(*args)

    if expected_delegate:
        assert result == "target-v0.25.1"
        assert len(target_calls) == 1
        delegated_instance, delegated_args = target_calls[0]
        assert delegated_instance is instance
        for actual, expected in zip(delegated_args[:6], args, strict=True):
            assert actual is expected
        assert delegated_args[6:] == (None,) * 6
    else:
        assert result is output
        torch.testing.assert_close(result, torch.zeros_like(output))
        assert target_calls == []


def test_mla_feature_on_uses_hcu_lightly_cp_delta(monkeypatch):
    adapter = _adapter()
    target_calls = []
    hcu_calls = []
    module = _fake_mla_module(adapter, target_calls)

    runtime = ModuleType("vllm_hcu.model_executor.layers.mla_runtime")

    def mla_forward_impl(upstream, self, *args):
        hcu_calls.append((upstream, self, args))
        return "hcu-lightly-cp"

    runtime.mla_forward_impl = mla_forward_impl
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    assert adapter.apply_to_module(module) is True

    instance = object.__new__(module.MLAAttention)
    instance._hcu_feature_config = SimpleNamespace(enable_lightly_cp=True)
    args = _forward_args()

    assert instance.forward_impl(*args) == "hcu-lightly-cp"
    assert target_calls == []
    assert hcu_calls == [(module, instance, args)]


@pytest.mark.parametrize(("pcp_size", "expected_direct"), [(1, False), (2, True)])
def test_mla_init_enables_direct_calls_only_for_pcp(
    monkeypatch,
    pcp_size,
    expected_direct,
):
    adapter = _adapter()
    module = _fake_mla_module(adapter, [])
    monkeypatch.setattr(
        adapter,
        "get_hcu_config",
        lambda config: SimpleNamespace(enable_lightly_cp=False),
    )
    import vllm.config as vllm_config_module

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
        )
    )
    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config_or_none",
        lambda: config,
    )
    assert adapter.apply_to_module(module) is True
    assert adapter.apply_to_module(module) is False

    instance = module.MLAAttention(1, 1.0, 1, 1, 1, None, 1, object())

    assert instance.use_direct_call is expected_direct
    assert instance._hcu_pcp_world_size == pcp_size


def test_mla_init_rejects_combined_pcp_and_lightly_cp(monkeypatch):
    adapter = _adapter()
    module = _fake_mla_module(adapter, [])
    monkeypatch.setattr(
        adapter,
        "get_hcu_config",
        lambda config: SimpleNamespace(enable_lightly_cp=True),
    )
    import vllm.config as vllm_config_module

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2)
    )
    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config_or_none",
        lambda: config,
    )
    assert adapter.apply_to_module(module) is True

    with pytest.raises(RuntimeError, match="PCP.*lightly-CP"):
        module.MLAAttention(1, 1.0, 1, 1, 1, None, 1, object())


def test_mla_pcp_full_forward_gathers_cache_inputs_and_keeps_q_local(
    monkeypatch,
):
    adapter = _adapter()
    target_calls = []
    events = []
    module = _fake_mla_module(adapter, target_calls, events)
    context_holder = {}
    module.get_forward_context = lambda: context_holder["context"]
    assert adapter.apply_to_module(module) is True

    q = torch.tensor([[1.0], [2.0]])
    local_kv = torch.tensor([[10.0], [11.0]])
    local_rope = torch.tensor([[[20.0]], [[21.0]]])
    expanded_slots = torch.tensor([100, 101, 200], dtype=torch.int64)
    gathered_kv = torch.tensor([[10.0], [11.0], [12.0]])
    gathered_rope = torch.tensor([[[20.0]], [[21.0]], [[22.0]]])
    gathered_slots = torch.tensor([100, 101, 200], dtype=torch.int64)
    metadata = SimpleNamespace(
        pcp_world_size=2,
        num_decode_tokens=0,
        num_prefills=2,
    )
    context = SimpleNamespace(
        attn_metadata={"layer": metadata},
        slot_mapping={"layer": expanded_slots},
    )
    context_holder["context"] = context

    from vllm_hcu.model_executor.layers.attention import pcp

    def gather(kv, rope, slots, actual_metadata):
        events.append("gather")
        assert kv is local_kv
        assert rope is local_rope
        assert slots is expanded_slots
        assert actual_metadata is metadata
        return gathered_kv, gathered_rope, gathered_slots

    monkeypatch.setattr(pcp, "maybe_gather_mla_latent_cache_inputs", gather)

    class Impl:
        def do_kv_cache_update(
            self,
            kv,
            rope,
            cache,
            slots,
            cache_dtype,
            k_scale,
        ):
            events.append("cache")
            assert kv is gathered_kv
            assert rope is gathered_rope
            assert slots is gathered_slots

    instance = object.__new__(module.MLAAttention)
    instance._hcu_use_pcp = True
    instance._hcu_pcp_world_size = 2
    instance._hcu_feature_config = SimpleNamespace(enable_lightly_cp=False)
    instance.calculate_kv_scales = False
    instance.layer_name = "layer"
    instance.kv_cache = torch.empty(0)
    instance.kv_cache_dtype = "auto"
    instance._k_scale = torch.tensor(1.0)
    instance.impl = Impl()

    result = instance.forward(
        q,
        local_kv,
        local_rope,
        output_shape=torch.Size([2, 1]),
    )

    assert result.shape == (2, 1)
    assert events == ["gather", "cache", "forward_impl"]
    assert module.full_forward_calls == []
    assert len(target_calls) == 1
    forwarded_args = target_calls[0][1]
    assert forwarded_args[0] is q
    assert forwarded_args[1] is local_kv
    assert forwarded_args[2] is local_rope
    assert forwarded_args[4] is metadata


def test_mla_pcp_profile_forward_skips_cache_without_metadata(monkeypatch):
    adapter = _adapter()
    target_calls = []
    events = []
    module = _fake_mla_module(adapter, target_calls, events)
    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata=None,
        slot_mapping={},
    )
    assert adapter.apply_to_module(module) is True
    from vllm_hcu.model_executor.layers.attention import pcp

    monkeypatch.setattr(
        pcp,
        "maybe_gather_mla_latent_cache_inputs",
        lambda *args: pytest.fail("profile forward gathered cache inputs"),
    )

    class Impl:
        def do_kv_cache_update(self, *args):
            pytest.fail("profile forward inserted into the KV cache")

    instance = object.__new__(module.MLAAttention)
    instance._hcu_use_pcp = True
    instance._hcu_pcp_world_size = 2
    instance._hcu_feature_config = SimpleNamespace(enable_lightly_cp=False)
    instance.calculate_kv_scales = False
    instance.layer_name = "layer"
    instance.kv_cache = torch.empty(0)
    instance.kv_cache_dtype = "auto"
    instance._k_scale = torch.tensor(1.0)
    instance.impl = Impl()
    q = torch.tensor([[1.0], [2.0]])
    local_kv = torch.tensor([[10.0], [11.0]])
    local_rope = torch.tensor([[[20.0]], [[21.0]]])

    result = instance.forward(
        q,
        local_kv,
        local_rope,
        output_shape=torch.Size([2, 1]),
    )

    assert result.shape == (2, 1)
    assert events == ["forward_impl"]
    assert len(target_calls) == 1
    forwarded_args = target_calls[0][1]
    assert forwarded_args[0] is q
    assert forwarded_args[1] is local_kv
    assert forwarded_args[2] is local_rope
    assert forwarded_args[4] is None


def test_mla_pcp_real_forward_rejects_missing_layer_slots():
    adapter = _adapter()
    module = _fake_mla_module(adapter, [])
    metadata = SimpleNamespace(pcp_world_size=2)
    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={"layer": metadata},
        slot_mapping={},
    )
    assert adapter.apply_to_module(module) is True

    instance = object.__new__(module.MLAAttention)
    instance._hcu_use_pcp = True
    instance._hcu_pcp_world_size = 2
    instance.calculate_kv_scales = False
    instance.layer_name = "layer"

    with pytest.raises(RuntimeError, match="slot mapping is missing"):
        instance.forward(
            torch.ones(1, 1),
            torch.ones(1, 1),
            torch.ones(1, 1, 1),
            output_shape=torch.Size([1, 1]),
        )


def test_mla_pcp_one_keeps_target_opaque_full_forward(monkeypatch):
    adapter = _adapter()
    events = []
    module = _fake_mla_module(adapter, [], events)
    assert adapter.apply_to_module(module) is True
    from vllm_hcu.model_executor.layers.attention import pcp

    monkeypatch.setattr(
        pcp,
        "maybe_gather_mla_latent_cache_inputs",
        lambda *args: pytest.fail("PCP=1 gathered MLA cache inputs"),
    )
    instance = object.__new__(module.MLAAttention)
    instance._hcu_use_pcp = False
    q, kv, rope = object(), object(), object()

    assert instance.forward(q, kv, rope) == "target-opaque-v0.25.1"
    assert module.full_forward_calls == [(instance, (q, kv, rope, None))]
    assert events == ["opaque_forward"]


def test_replicated_mtp_scope_uses_target_mla_forward(monkeypatch):
    """A restored global MTP batch must not enter PCP cache gathers again."""

    adapter = _adapter()
    events = []
    module = _fake_mla_module(adapter, [], events)
    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={"layer": SimpleNamespace(pcp_world_size=2)},
        slot_mapping={"layer": torch.tensor([0], dtype=torch.int64)},
    )
    assert adapter.apply_to_module(module) is True
    from vllm_hcu.model_executor.layers.attention import pcp

    monkeypatch.setattr(
        pcp,
        "maybe_gather_mla_latent_cache_inputs",
        lambda *args: pytest.fail("global MTP input entered a PCP cache gather"),
    )
    instance = object.__new__(module.MLAAttention)
    instance._hcu_use_pcp = True
    instance._hcu_pcp_world_size = 2
    instance.calculate_kv_scales = False
    instance.layer_name = "layer"
    q, kv, rope = object(), object(), object()
    scope = getattr(pcp, "replicated_mtp_batch_scope", nullcontext)

    with scope():
        result = instance.forward(q, kv, rope)

    assert result == "target-opaque-v0.25.1"
    assert module.full_forward_calls == [(instance, (q, kv, rope, None))]
    assert events == ["opaque_forward"]


def test_dense_and_sparse_mla_metadata_carry_parallel_sizes(monkeypatch):
    adapter = _adapter()
    module = _fake_mla_module(adapter, [])
    assert adapter.apply_to_module(module) is True
    dense_builder = module.MLACommonMetadataBuilder()
    dense_builder.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2)
    )
    common = SimpleNamespace(num_actual_tokens=3, num_kv_actual_tokens=5)

    dense_metadata = dense_builder.build(0, common)

    assert dense_metadata.pcp_world_size == 2

    sparse_adapter = importlib.import_module(
        "vllm_hcu.patch.worker.op_opt.patch_flashmla_sparse"
    )

    class FlashMLASparseMetadataBuilder:
        def build(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build=False,
        ):
            del common_prefix_len, common_attn_metadata, fast_build
            return SimpleNamespace(fp8_use_mixed_batch=False)

    class FlashMLASparseImpl:
        def forward_mqa(
            self,
            q,
            kv_c_and_k_pe_cache,
            attn_metadata,
            layer,
        ):
            del kv_c_and_k_pe_cache, attn_metadata, layer
            return q

        def _fp8_flash_mla_kernel(
            self,
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            kernel_metadata,
        ):
            return q

        def _bf16_flash_mla_kernel(
            self,
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            topk_length=None,
        ):
            return q

    sparse_module = ModuleType(sparse_adapter.TARGET_MODULE)
    sparse_module.FlashMLASparseMetadataBuilder = FlashMLASparseMetadataBuilder
    sparse_module.FlashMLASparseImpl = FlashMLASparseImpl
    sparse_module.split_decodes_and_prefills = (
        lambda common_attn_metadata,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True: (0, 0, 0, 0)
    )
    sparse_module.current_platform = SimpleNamespace(is_rocm=lambda: False)
    sparse_module.torch = torch
    sparse_module.SimpleNamespace = SimpleNamespace
    exec(
        """
def _build_fp8_separate_prefill_decode(self, common_attn_metadata):
    counts = split_decodes_and_prefills(common_attn_metadata)
    return SimpleNamespace(num_prefills=counts[1])
""",
        sparse_module.__dict__,
    )
    FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode = (
        sparse_module._build_fp8_separate_prefill_decode
    )
    assert sparse_adapter.apply_to_module(sparse_module) is True
    sparse_builder = FlashMLASparseMetadataBuilder()
    sparse_builder.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            cp_kv_cache_interleave_size=4,
        )
    )

    sparse_metadata = sparse_builder.build(0, common)

    assert sparse_metadata.pcp_world_size == 2
    assert sparse_metadata.cp_kv_cache_interleave_size == 4

    from vllm_hcu.model_executor.layers.attention import pcp

    scope = getattr(pcp, "replicated_mtp_batch_scope", nullcontext)
    with scope():
        replicated_dense_metadata = dense_builder.build(0, common)
        replicated_sparse_metadata = sparse_builder.build(0, common)

    assert replicated_dense_metadata.pcp_world_size == 1
    assert replicated_sparse_metadata.pcp_world_size == 1


@pytest.mark.parametrize(
    ("is_gfx938", "use_cat_mla", "expected"),
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
def test_flashmla_impl_owns_quant_query_capability(
    monkeypatch,
    cpu_flashmla,
    is_gfx938,
    use_cat_mla,
    expected,
):
    flashmla = cpu_flashmla
    monkeypatch.setattr(flashmla, "on_gfx938", lambda: is_gfx938)
    monkeypatch.setattr(
        flashmla.henvs,
        "VLLM_HCU_USE_CAT_MLA",
        use_cat_mla,
        raising=False,
    )
    monkeypatch.setattr(
        flashmla,
        "is_flashmla_dense_supported",
        lambda: (True, None),
    )

    impl = flashmla.FlashMLAImpl(
        num_heads=1,
        head_size=192,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_e4m3",
        logits_soft_cap=None,
        attn_type=flashmla.AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
    )
    assert impl.supports_quant_query_input is expected


def test_only_hcu_dense_and_sparse_mla_impls_advertise_pcp(
    monkeypatch,
    cpu_flashmla,
):
    dense_impl = cpu_flashmla.HcuFlashMLABackend.get_impl_cls()
    assert dense_impl is cpu_flashmla.FlashMLAImpl
    assert dense_impl.supports_pcp is True
    assert _MLACommonImplStub.supports_pcp is False

    class UpstreamFlashMLASparseImpl:
        supports_pcp = False
        can_return_lse_for_decode = False

        def forward_mqa(self, q, kv_cache, attn_metadata, layer):
            del self, q, kv_cache, attn_metadata, layer
            return "upstream-output", None

    class UpstreamFlashMLASparseBackend:
        @staticmethod
        def get_impl_cls():
            return UpstreamFlashMLASparseImpl

    _install_stub(
        monkeypatch,
        "vllm.v1.attention.backends.mla.flashmla_sparse",
        FlashMLASparseBackend=UpstreamFlashMLASparseBackend,
        FlashMLASparseImpl=UpstreamFlashMLASparseImpl,
    )
    module_name = "_vllm_hcu_cpu_test_flashmla_sparse_backend"
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/v1/attention/backends/mla/flashmla_sparse.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    sparse = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, sparse)
    spec.loader.exec_module(sparse)

    sparse_impl = sparse.HcuFlashMLASparseBackend.get_impl_cls()
    assert sparse_impl is sparse.HcuFlashMLASparseImpl
    assert sparse_impl.supports_pcp is True
    assert sparse_impl.can_return_lse_for_decode is True
    assert UpstreamFlashMLASparseImpl.supports_pcp is False
    assert UpstreamFlashMLASparseImpl.can_return_lse_for_decode is False


def test_hcu_sparse_mla_dcp_localizes_owned_indices_and_masks_empty_rows(
    monkeypatch,
):
    calls: list[tuple[object, ...]] = []

    class UpstreamFlashMLASparseImpl:
        supports_pcp = False
        can_return_lse_for_decode = False

        def forward_mqa(self, q, kv_cache, attn_metadata, layer):
            calls.append(("upstream", q, kv_cache, attn_metadata, layer))
            return "upstream-output", None

    class UpstreamFlashMLASparseBackend:
        @staticmethod
        def get_impl_cls():
            return UpstreamFlashMLASparseImpl

    def concat_mla_q(q_nope, q_pe, output):
        output.copy_(torch.cat((q_nope, q_pe), dim=-1))

    def convert_indices(*args, **kwargs):
        del args, kwargs
        pytest.fail("DCP sparse MLA used the non-DCP index converter")

    def filter_dcp_indices(
        req_id_per_token,
        block_table,
        topk_indices,
        *,
        dcp_size,
        dcp_rank,
        cp_kv_cache_interleave_size,
        BLOCK_SIZE,
        NUM_TOPK_TOKENS,
        return_valid_counts,
    ):
        torch.testing.assert_close(
            req_id_per_token,
            torch.tensor([0, 1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            block_table,
            torch.tensor([[7], [11]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            topk_indices,
            torch.tensor(
                [[0, 1, 2, 3], [0, 2, 4, 6]],
                dtype=torch.int32,
            ),
        )
        calls.append(
            (
                "filter_dcp",
                req_id_per_token,
                block_table,
                topk_indices,
                dcp_size,
                dcp_rank,
                cp_kv_cache_interleave_size,
                BLOCK_SIZE,
                NUM_TOPK_TOKENS,
                return_valid_counts,
            )
        )
        # For DCP size=2/rank=1/interleave=1, global positions 1 and 3
        # become local positions 0 and 1. With physical block 7, those
        # are slots 112 and 113. The second row has no rank-1 ownership.
        return (
            torch.tensor(
                [[112, 113, -1, -1], [-1, -1, -1, -1]],
                dtype=torch.int32,
            ),
            torch.tensor([2, 0], dtype=torch.int32),
        )

    kernel_output = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    kernel_lse = torch.arange(8, dtype=torch.float32).view(2, 4)

    def sparse_fwd(q, cache, indices, scale, *, topk_length):
        calls.append(("kernel", q.clone(), cache, indices, scale, topk_length))
        return kernel_output.clone(), torch.empty(0), kernel_lse.clone()

    _install_stub(
        monkeypatch,
        "vllm.v1.attention.backends.mla.flashmla_sparse",
        FlashMLASparseBackend=UpstreamFlashMLASparseBackend,
        FlashMLASparseImpl=UpstreamFlashMLASparseImpl,
    )
    _install_stub(monkeypatch, "vllm._custom_ops", concat_mla_q=concat_mla_q)
    _install_stub(
        monkeypatch,
        "vllm.v1.attention.backends.mla.sparse_utils",
        triton_convert_req_index_to_global_index=convert_indices,
        triton_filter_and_convert_dcp_index=filter_dcp_indices,
    )
    _install_stub(
        monkeypatch,
        "vllm_hcu.v1.attention.ops.flashmla",
        flash_mla_sparse_fwd=sparse_fwd,
    )

    module_name = "_vllm_hcu_cpu_test_flashmla_sparse_dcp_backend"
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu/v1/attention/backends/mla/flashmla_sparse.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    sparse = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, sparse)
    spec.loader.exec_module(sparse)

    impl = sparse.HcuFlashMLASparseImpl()
    impl.dcp_world_size = 2
    impl.dcp_rank = 1
    impl.kv_cache_dtype = "auto"
    impl.softmax_scale = 0.25
    impl.q_concat_buffer = torch.empty(2, 4, 4)
    impl.topk_indices_buffer = torch.tensor(
        [[0, 1, 2, 3], [0, 2, 4, 6]], dtype=torch.int32
    )
    q_nope = torch.ones(2, 4, 3)
    q_pe = torch.full((2, 4, 1), 2.0)
    cache = torch.ones(6, 4)
    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        block_table=torch.tensor([[7], [11]], dtype=torch.int32),
        block_size=16,
        cp_kv_cache_interleave_size=1,
    )
    layer = object()

    output, lse = impl.forward_mqa((q_nope, q_pe), cache, metadata, layer)

    torch.testing.assert_close(output[0], kernel_output[0])
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
    torch.testing.assert_close(lse[0], kernel_lse[0])
    assert torch.isneginf(lse[1]).all()
    filter_call = calls[-2]
    assert filter_call[0] == "filter_dcp"
    assert filter_call[4:10] == (2, 1, 1, 16, 4, True)
    kernel_call = calls[-1]
    assert kernel_call[0] == "kernel"
    assert kernel_call[1].shape == (2, 4, 4)
    assert kernel_call[2].shape == (6, 1, 4)
    torch.testing.assert_close(
        kernel_call[3],
        torch.tensor(
            [[[112, 113, -1, -1]], [[-1, -1, -1, -1]]],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        kernel_call[-1], torch.tensor([2, 0], dtype=torch.int32)
    )

    impl.dcp_world_size = 1
    assert impl.forward_mqa(q_nope, cache, metadata, layer) == (
        "upstream-output",
        None,
    )
    assert calls[-1][0] == "upstream"


def test_flashmla_cat_route_consumes_split_query(
    monkeypatch,
    cpu_flashmla,
):
    flashmla = cpu_flashmla
    q_nope = torch.ones(1, 2, 128)
    q_pe = torch.ones(1, 2, 64)
    cache = torch.zeros(2, 1, 192, dtype=torch.uint8)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.ones(1, dtype=torch.int32)
    tile_metadata = torch.zeros(1, 8, dtype=torch.int32)
    num_splits = torch.zeros(2, dtype=torch.int32)
    calls = []

    def cat_kernel(**kwargs):
        calls.append(kwargs)
        return torch.ones(1, 2, 128), torch.ones(1, 2)

    monkeypatch.setattr(flashmla, "on_gfx938", lambda: True)
    monkeypatch.setattr(
        flashmla.henvs,
        "VLLM_HCU_USE_CAT_MLA",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        flashmla,
        "reshape_query_for_spec_decode",
        lambda query, num_decodes: query,
    )
    monkeypatch.setattr(
        flashmla,
        "reshape_attn_output_for_spec_decode",
        lambda output: output,
    )
    monkeypatch.setattr(
        flashmla,
        "flash_mla_with_kvcache_fp8_with_cat",
        cat_kernel,
    )
    monkeypatch.setattr(
        flashmla,
        "flash_mla_with_kvcache_fp8",
        lambda **kwargs: pytest.fail("CAT route used the fused-query kernel"),
    )

    impl = object.__new__(flashmla.FlashMLAImpl)
    impl.kv_cache_dtype = "fp8_e4m3"
    impl.kv_lora_rank = 128
    impl.scale = 0.5
    metadata = SimpleNamespace(
        num_decodes=1,
        decode=SimpleNamespace(
            block_table=block_table,
            seq_lens=seq_lens,
            scheduler_metadata=SimpleNamespace(
                tile_scheduler_metadata=tile_metadata,
                num_splits=num_splits,
            ),
        ),
    )
    layer = SimpleNamespace(
        _q_scale=torch.ones(1),
        _k_scale=torch.ones(1),
    )

    output, lse = impl.forward_mqa(
        (q_nope, q_pe), cache, metadata, layer
    )

    assert output.shape == (1, 2, 128)
    assert lse.shape == (1, 2)
    assert len(calls) == 1
    assert calls[0]["q_nope"] is q_nope
    assert calls[0]["q_pe"] is q_pe
    assert calls[0]["block_table"] is block_table
    assert calls[0]["cache_seqlens"] is seq_lens
