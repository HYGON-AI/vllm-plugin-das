# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for target-owned MLA execution and HCU CAT routing."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from contextlib import nullcontext
from dataclasses import make_dataclass, replace
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
        NEVER = "never"
        UNIFORM_BATCH = "uniform-batch"

    class AttentionType:
        DECODER = "decoder"

    class QueryLenSupport:
        UNIFORM = "uniform"

    class MLACommonMetadataBuilder(_GenericStub):
        @classmethod
        def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
            del vllm_config, kv_cache_spec
            return cls._cudagraph_support

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
        MLACommonMetadataBuilder=MLACommonMetadataBuilder,
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
        direct_register_custom_op=lambda **kwargs: None,
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
    sparse_mha_calls = []
    cache_gather_calls = []

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
            self.use_direct_call = False

        def forward(
            self,
            q,
            kv_c_normed,
            k_pe,
            output_shape=None,
            q_dcp_replicated=None,
        ):
            args = (q, kv_c_normed, k_pe, output_shape, q_dcp_replicated)
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
            q_dcp_replicated=None,
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
                q_dcp_replicated,
            )
            target_calls.append((self, args))
            if event_log is not None:
                event_log.append("forward_impl")
            return "target-v0.25.1"

        def process_weights_after_loading(self, act_dtype):
            return act_dtype

        def _use_sparse_mha(self, attn_metadata):
            sparse_mha_calls.append((self, attn_metadata))
            return True

        def bind_kv_cache(self, kv_cache):
            self.kv_cache = kv_cache

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

    def _get_kv_b_proj_input_dtype(kv_b_proj, use_fp8_prefill):
        del use_fp8_prefill
        weight = getattr(kv_b_proj, "weight", None)
        return weight.dtype if weight is not None else kv_b_proj.params_dtype

    module.MLAAttention = MLAAttention
    module.MLACommonMetadata = MLACommonMetadata
    module.MLACommonMetadataBuilder = MLACommonMetadataBuilder
    module.QueryLenSupport = SimpleNamespace(
        SINGLE_ONLY="single_only", UNIFORM="uniform", VARLEN="varlen"
    )
    module.MLACommonMetadataBuilder.query_len_support = (
        module.QueryLenSupport.SINGLE_ONLY
    )
    module.MLACommonMetadataBuilder.use_pcp = False
    module._get_kv_b_proj_input_dtype = _get_kv_b_proj_input_dtype
    module.split_decodes_and_prefills = split_decodes_and_prefills
    def maybe_gather_mla_latent_cache_inputs(
        kv_c_normed,
        k_pe,
        slot_mapping,
        num_decode_tokens,
        use_pcp,
    ):
        cache_gather_calls.append(
            (kv_c_normed, k_pe, slot_mapping, num_decode_tokens, use_pcp)
        )
        return (
            kv_c_normed,
            k_pe,
            slot_mapping,
        )

    module.maybe_gather_mla_latent_cache_inputs = (
        maybe_gather_mla_latent_cache_inputs
    )
    module.split_calls = split_calls
    module.full_forward_calls = full_forward_calls
    module.sparse_mha_calls = sparse_mha_calls
    module.cache_gather_calls = cache_gather_calls
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
    return tuple(object() for _ in range(13))


def test_mla_int8_kv_b_proj_keeps_floating_prefill_input_dtype():
    adapter = _adapter()
    module = _fake_mla_module(adapter, [])
    int8_projection = SimpleNamespace(
        weight=torch.empty(1, dtype=torch.int8),
        params_dtype=torch.bfloat16,
        scheme=SimpleNamespace(supports_quanted_inputs=lambda: False),
    )
    custom_projection = SimpleNamespace(
        weight=torch.empty(1, dtype=torch.int8),
        params_dtype=torch.bfloat16,
        scheme=SimpleNamespace(supports_quanted_inputs=lambda: True),
    )
    fp16_projection = SimpleNamespace(
        weight=torch.empty(1, dtype=torch.float16),
        params_dtype=torch.bfloat16,
    )
    assert module._get_kv_b_proj_input_dtype(int8_projection, False) is torch.int8

    assert adapter.apply_to_module(module) is True

    assert (
        module._get_kv_b_proj_input_dtype(int8_projection, False)
        is torch.bfloat16
    )
    assert (
        module._get_kv_b_proj_input_dtype(fp16_projection, False)
        is torch.float16
    )
    assert (
        module._get_kv_b_proj_input_dtype(custom_projection, False)
        is torch.bfloat16
    )


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
    impl_post_load_calls: list[torch.dtype] = []
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
        impl=SimpleNamespace(
            process_weights_after_loading=lambda dtype: (
                impl_post_load_calls.append(dtype)
            )
        ),
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
    assert impl_post_load_calls == [torch.bfloat16]


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


def test_mla_split_preserves_decode_fast_path_for_padded_full_graph():
    adapter = _adapter()
    module = _fake_mla_module(adapter, [])

    def split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        del require_uniform
        query_lens = (
            common_attn_metadata.query_start_loc_cpu[1:]
            - common_attn_metadata.query_start_loc_cpu[:-1]
        )
        if (
            common_attn_metadata.max_query_len <= decode_threshold
            and treat_short_extends_as_decodes
        ):
            return (
                common_attn_metadata.num_reqs,
                0,
                common_attn_metadata.num_actual_tokens,
                0,
            )
        is_prefill = query_lens > decode_threshold
        if not treat_short_extends_as_decodes:
            is_prefill |= common_attn_metadata.is_prefilling
        return (0, 0, 0, int(is_prefill.sum().item()))

    module.split_decodes_and_prefills = split_decodes_and_prefills
    assert adapter.apply_to_module(module) is True
    common = SimpleNamespace(
        query_start_loc_cpu=torch.arange(9, dtype=torch.int32),
        is_prefilling=torch.zeros(7, dtype=torch.bool),
        max_query_len=1,
        num_reqs=8,
        num_actual_tokens=8,
    )

    assert module.split_decodes_and_prefills(common) == (8, 0, 8, 0)


def test_mla_pcp_disables_rank_local_sparse_mha_prefill():
    adapter = _adapter()
    module = _fake_mla_module(adapter, [])
    assert adapter.apply_to_module(module) is True

    metadata = object()
    instance = object.__new__(module.MLAAttention)
    instance._hcu_use_pcp = True
    assert instance._use_sparse_mha(metadata) is False
    assert module.sparse_mha_calls == []

    instance._hcu_use_pcp = False
    assert instance._use_sparse_mha(metadata) is True
    assert module.sparse_mha_calls == [(instance, metadata)]


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


def test_mla_pcp_full_forward_delegates_current_upstream_ownership(
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
    # Current upstream no longer exposes the legacy calculate_kv_scales flag.
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

    assert result == "target-opaque-v0.25.1"
    assert events == ["opaque_forward"]
    assert module.full_forward_calls == [
        (
            instance,
            (q, local_kv, local_rope, torch.Size([2, 1]), None),
        )
    ]
    assert target_calls == []


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

    assert result == "target-opaque-v0.25.1"
    assert events == ["opaque_forward"]
    assert module.full_forward_calls == [
        (
            instance,
            (q, local_kv, local_rope, torch.Size([2, 1]), None),
        )
    ]
    assert target_calls == []


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
    assert module.full_forward_calls == [(instance, (q, kv, rope, None, None))]
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
        cache_inputs = module.maybe_gather_mla_latent_cache_inputs(
            kv,
            rope,
            "global-slots",
            0,
            True,
        )
        result = instance.forward(q, kv, rope)

    assert cache_inputs == (kv, rope, "global-slots")
    assert result == "target-opaque-v0.25.1"
    assert module.full_forward_calls == [(instance, (q, kv, rope, None, None))]
    assert events == ["opaque_forward"]


def test_mla_cache_gather_distinguishes_replicated_mtp_from_pcp_prefill():
    adapter = _adapter()
    module = _fake_mla_module(adapter, [])
    assert adapter.apply_to_module(module) is True

    kv = torch.ones(5, 1)
    rope = torch.ones(5, 1, 1)
    replicated_slots = torch.arange(5)
    assert module.maybe_gather_mla_latent_cache_inputs(
        kv, rope, replicated_slots, 4, True
    ) == (kv, rope, replicated_slots)
    assert module.cache_gather_calls == []

    partitioned_slots = torch.arange(10)
    assert module.maybe_gather_mla_latent_cache_inputs(
        kv, rope, partitioned_slots, 4, True
    ) == (kv, rope, partitioned_slots)
    assert module.cache_gather_calls == [
        (kv, rope, partitioned_slots, 4, True)
    ]


@pytest.mark.parametrize(
    (
        "query_lens", "prefilling", "device_lens", "upper_bound", "causal",
        "decode_threshold", "expected_context",
    ),
    [
        ([4], [False], [108], [110], True, 1, [104]),
        ([4, 12], [False, True], [108, 60], [110, 60], True, 1, [104, 48]),
        ([12], [True], [60], [60], True, 1, [48]),
        ([1], [False], [108], [110], True, 1, [109]),
        ([4], [False], [108], [110], False, 1, [106]),
        ([4], [False], [108], [110], True, 4, [106]),
        ([4, 4], [False, False], [108, 100], [110, 103], True, 1, [104, 96]),
    ],
)
def test_mla_multitoken_decode_context_excludes_rejected_drafts(
    query_lens, prefilling, device_lens, upper_bound, causal, decode_threshold,
    expected_context,
):
    adapter = _adapter()
    module = _fake_mla_module(adapter, [])

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return SimpleNamespace(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            context_lens=common_attn_metadata.seq_lens_cpu_upper_bound
            - common_attn_metadata.query_start_loc_cpu.diff(),
            common=common_attn_metadata,
        )

    module.MLACommonMetadataBuilder.build = build
    assert adapter.apply_to_module(module) is True
    metadata_builder = module.MLACommonMetadataBuilder()
    metadata_builder.reorder_batch_threshold = decode_threshold
    metadata_values = {
        "num_actual_tokens": sum(query_lens),
        "query_start_loc_cpu": torch.tensor([0] + query_lens).cumsum(0),
        "is_prefilling": torch.tensor(prefilling),
        "seq_lens": torch.tensor(device_lens),
        "seq_lens_cpu_upper_bound": torch.tensor(upper_bound),
        "causal": causal,
    }
    metadata_class = make_dataclass(
        "CommonAttentionMetadata", [(name, object) for name in metadata_values]
    )
    common = metadata_class(**metadata_values)
    common.replace = lambda **updates: replace(common, **updates)
    common.num_kv_actual_tokens = common.num_actual_tokens + 5

    result = metadata_builder.build(0, common)

    assert result.context_lens.tolist() == expected_context
    assert result.num_kv_actual_tokens == common.num_kv_actual_tokens
    assert common.seq_lens_cpu_upper_bound.tolist() == upper_bound
    if causal and any(
        length > decode_threshold and not phase
        for length, phase in zip(query_lens, prefilling)
    ):
        assert result.common is not common
    else:
        assert result.common is common


@pytest.mark.parametrize(
    ("query_len_support", "decode_threshold"),
    [("SINGLE_ONLY", 1), ("UNIFORM", 128), ("VARLEN", 128)],
)
@pytest.mark.parametrize(
    ("num_actual_reqs", "num_padded_reqs", "padding_query_len"),
    [
        pytest.param(3, 4, 0, id="full-3-to-4"),
        pytest.param(5, 8, 0, id="full-5-to-8"),
        pytest.param(6, 8, 0, id="full-6-to-8"),
        pytest.param(7, 8, 0, id="full-7-to-8"),
        pytest.param(8, 8, 0, id="full-unpadded"),
        pytest.param(3, 3, 0, id="piecewise-unpadded"),
        pytest.param(7, 8, 1, id="full-single-token-padding"),
        pytest.param(7, 8, 4, id="ignore-multitoken-padding"),
    ],
)
def test_mla_build_ignores_decode_graph_padding_without_device_sync(
    num_actual_reqs, num_padded_reqs, padding_query_len,
    query_len_support, decode_threshold,
):
    adapter = _adapter()
    module = _fake_mla_module(adapter, [])
    query_lens = [1] * num_actual_reqs + [padding_query_len] * (
        num_padded_reqs - num_actual_reqs
    )
    common = SimpleNamespace(
        num_reqs=num_padded_reqs,
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        query_start_loc_cpu=torch.tensor([0] + query_lens).cumsum(0),
        is_prefilling=torch.zeros(num_actual_reqs, dtype=torch.bool),
        seq_lens=SimpleNamespace(
            cpu=lambda: pytest.fail("ordinary decode synchronized device lengths")
        ),
        causal=True,
    )
    build_calls = []

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        build_calls.append((common_prefix_len, common_attn_metadata, fast_build))
        return SimpleNamespace(num_actual_tokens=common_attn_metadata.num_actual_tokens)

    module.MLACommonMetadataBuilder.build = build
    assert adapter.apply_to_module(module) is True
    metadata_builder = module.MLACommonMetadataBuilder()
    metadata_builder.query_len_support = getattr(
        module.QueryLenSupport, query_len_support
    )
    metadata_builder.reorder_batch_threshold = decode_threshold

    result = metadata_builder.build(0, common, fast_build=True)

    assert build_calls == [(0, common, True)]
    assert result.num_actual_tokens == common.num_actual_tokens
    assert result.num_kv_actual_tokens == common.num_actual_tokens


@pytest.mark.parametrize("num_padded_reqs", [2, 4, 8])
def test_mla_multitoken_decode_uses_actual_lengths_with_graph_padding(
    num_padded_reqs,
):
    adapter = _adapter()
    module = _fake_mla_module(adapter, [])
    num_padding_reqs = num_padded_reqs - 2
    query_lens = [4, 4] + [0] * num_padding_reqs
    upper_bound = [110, 103] + [0] * num_padding_reqs
    device_lens = torch.tensor([108, 100] + [0] * num_padding_reqs)
    sync_calls = []
    common = SimpleNamespace(
        num_reqs=num_padded_reqs,
        num_actual_tokens=8,
        query_start_loc_cpu=torch.tensor([0] + query_lens).cumsum(0),
        is_prefilling=torch.tensor([False, False]),
        seq_lens=SimpleNamespace(
            cpu=lambda: sync_calls.append("cpu") or device_lens,
        ),
        seq_lens_cpu_upper_bound=torch.tensor(upper_bound),
        causal=True,
        num_kv_actual_tokens=13,
    )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return SimpleNamespace(
            common=common_attn_metadata,
            context_lens=common_attn_metadata.seq_lens_cpu_upper_bound
            - common_attn_metadata.query_start_loc_cpu.diff(),
        )

    module.MLACommonMetadataBuilder.build = build
    assert adapter.apply_to_module(module) is True
    metadata_builder = module.MLACommonMetadataBuilder()
    metadata_builder.reorder_batch_threshold = 1

    result = metadata_builder.build(0, common)

    assert sync_calls == ["cpu"]
    assert result.context_lens.tolist() == [104, 96] + [0] * num_padding_reqs
    assert result.common is not common
    assert common.seq_lens_cpu_upper_bound.tolist() == upper_bound
    assert result.num_kv_actual_tokens == 13


@pytest.mark.parametrize(
    (
        "query_lens", "prefilling", "device_lens", "upper_bound",
        "query_len_support", "decode_threshold", "expected_num_decodes",
        "expected_context", "needs_length_refresh",
    ),
    [
        pytest.param(
            [4, 2], [False, False], [108, 106], [108, 108],
            "UNIFORM", 128, 1, [104], True, id="budget-truncated-4-2",
        ),
        pytest.param(
            [4, 1], [False, False], [108, 105], [108, 108],
            "UNIFORM", 128, 1, [104], True, id="budget-truncated-4-1",
        ),
        pytest.param(
            [4, 4, 2], [False] * 3, [108, 108, 106], [108] * 3,
            "UNIFORM", 128, 2, [104], True, id="budget-truncated-last-request",
        ),
        pytest.param(
            [4, 2, 2], [False] * 3, [108, 106, 106], [108] * 3,
            "UNIFORM", 128, 1, [104, 104], True, id="multiple-fallback-requests",
        ),
        pytest.param(
            [4, 2, 4], [False] * 3, [108, 106, 106], [108] * 3,
            "UNIFORM", 128, 1, [104, 102], True, id="whole-prefill-suffix",
        ),
        pytest.param(
            [2, 4], [False, False], [106, 108], [108, 110],
            "UNIFORM", 128, 1, [104], True, id="short-first-query",
        ),
        pytest.param(
            [1, 4], [False, False], [105, 108], [108, 110],
            "UNIFORM", 128, 1, [104], True, id="single-token-first-query",
        ),
        pytest.param(
            [4, 2], [False, False], [108, 106], [108, 108],
            "UNIFORM", 4, 1, [104], True, id="speculative-query-threshold",
        ),
        pytest.param(
            [4, 2], [False, False], [108, 106], [108, 108],
            "UNIFORM", 1, 0, [104, 104], True, id="dcp-single-query-threshold",
        ),
        pytest.param(
            [4, 2, 16], [False, False, True], [108, 106, 60], [108, 108, 60],
            "UNIFORM", 128, 1, [104, 44], True, id="fallback-and-real-prefill",
        ),
        pytest.param(
            [4, 16], [False, True], [106, 60], [108, 60],
            "UNIFORM", 128, 1, [44], False, id="real-prefill-only-no-sync",
        ),
        pytest.param(
            [4, 4, 16], [False, False, True], [106, 106, 60], [108, 108, 60],
            "UNIFORM", 128, 2, [44], False, id="uniform-decodes-and-prefill",
        ),
        pytest.param(
            [4, 4], [False, False], [106, 106], [108, 108],
            "UNIFORM", 128, 2, [], False, id="uniform-mtp-no-sync",
        ),
        pytest.param(
            [1, 1], [False, False], [106, 106], [108, 108],
            "UNIFORM", 128, 2, [], False, id="ordinary-decode-no-sync",
        ),
        pytest.param(
            [4, 2], [False, False], [108, 106], [108, 108],
            "VARLEN", 128, 2, [], False, id="varlen-mtp-no-sync",
        ),
        pytest.param(
            [4, 2], [False, False], [108, 106], [108, 108],
            "SINGLE_ONLY", 1, 0, [104, 104], True, id="single-only-fallback",
        ),
        pytest.param(
            [4, 2], [True, True], [108, 108], [108, 108],
            "UNIFORM", 128, 0, [104, 106], False, id="real-prefills-no-sync",
        ),
        pytest.param(
            [4, 4, 0, 0], [False, False], [106, 106, 0, 0], [108, 108, 0, 0],
            "UNIFORM", 128, 4, [], False, id="uniform-mtp-graph-padding",
        ),
        pytest.param(
            [4, 2, 0, 0], [False, False], [108, 106, 0, 0], [108, 108, 0, 0],
            "UNIFORM", 128, 1, [104, 0, 0], True, id="fallback-with-padding",
        ),
    ],
)
@pytest.mark.parametrize("has_cpu_mirror", [False, True], ids=["async", "sync"])
def test_mla_prefill_fallback_uses_actual_split_and_sequence_lengths(
    monkeypatch, query_lens, prefilling, device_lens, upper_bound,
    query_len_support, decode_threshold, expected_num_decodes,
    expected_context, needs_length_refresh, has_cpu_mirror,
):
    from vllm import envs
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", True)
    adapter = _adapter()
    module = _fake_mla_module(adapter, [])
    module.split_decodes_and_prefills = split_decodes_and_prefills
    sync_calls = []
    device_lengths = torch.tensor(device_lens)
    seq_lens_cpu = torch.tensor(device_lens) if has_cpu_mirror else None
    expected_upper_bound = device_lens if has_cpu_mirror else upper_bound
    original_upper_bound = (
        seq_lens_cpu if has_cpu_mirror else torch.tensor(upper_bound)
    )
    common = SimpleNamespace(
        num_reqs=len(query_lens),
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        query_start_loc_cpu=torch.tensor([0] + query_lens).cumsum(0),
        is_prefilling=torch.tensor(prefilling),
        seq_lens=SimpleNamespace(
            cpu=lambda: sync_calls.append("cpu") or device_lengths,
        ),
        _seq_lens_cpu=seq_lens_cpu,
        seq_lens_cpu_upper_bound=original_upper_bound,
        causal=True,
        num_kv_actual_tokens=sum(query_lens) + 5,
    )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        num_decodes = module.split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
            require_uniform=(
                self.query_len_support != module.QueryLenSupport.VARLEN
            ),
            treat_short_extends_as_decodes=not self.use_pcp,
        )[0]
        return SimpleNamespace(
            common=common_attn_metadata,
            num_decodes=num_decodes,
            context_lens=(
                common_attn_metadata.seq_lens_cpu_upper_bound[num_decodes:]
                - common_attn_metadata.query_start_loc_cpu.diff()[num_decodes:]
            ),
        )

    module.MLACommonMetadataBuilder.build = build
    assert adapter.apply_to_module(module) is True
    metadata_builder = module.MLACommonMetadataBuilder()
    metadata_builder.query_len_support = getattr(
        module.QueryLenSupport, query_len_support
    )
    metadata_builder.reorder_batch_threshold = decode_threshold

    result = metadata_builder.build(0, common, fast_build=True)

    assert result.num_decodes == expected_num_decodes
    assert result.context_lens.tolist() == expected_context
    assert sync_calls == (
        ["cpu"] if needs_length_refresh and not has_cpu_mirror else []
    )
    assert (result.common is not common) is needs_length_refresh
    assert result.common._seq_lens_cpu is seq_lens_cpu
    if has_cpu_mirror:
        assert result.common.seq_lens_cpu_upper_bound is seq_lens_cpu
    assert result.common.query_start_loc_cpu is common.query_start_loc_cpu
    assert result.common.num_reqs == common.num_reqs
    assert common.seq_lens_cpu_upper_bound is original_upper_bound
    assert common.seq_lens_cpu_upper_bound.tolist() == expected_upper_bound
    assert result.num_kv_actual_tokens == common.num_kv_actual_tokens

    for request_index in range(expected_num_decodes, len(prefilling)):
        query_len = query_lens[request_index]
        context_len = result.context_lens[
            request_index - expected_num_decodes
        ].item()
        expected_context_len = device_lens[request_index] - query_len
        query = torch.ones(1, query_len, 1)
        key = torch.zeros(1, upper_bound[request_index], 1)
        value = torch.arange(
            upper_bound[request_index], dtype=torch.float32
        ).view(1, -1, 1)
        query_positions = torch.arange(query_len).unsqueeze(1)
        key_positions = torch.arange(upper_bound[request_index]).unsqueeze(0)
        actual_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=key_positions <= context_len + query_positions,
        )
        expected_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=key_positions <= expected_context_len + query_positions,
        )
        torch.testing.assert_close(actual_output, expected_output)


@pytest.fixture
def mrv2_prepare_attn():
    import vllm
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.attention.backend import CommonAttentionMetadata

    assert CommonAttentionMetadata.__dataclass_fields__["_seq_lens_cpu"].default is None
    namespace = {
        "torch": torch,
        "CUDAGraphMode": CUDAGraphMode,
        "CommonAttentionMetadata": CommonAttentionMetadata,
    }
    for relative_path, owner, function_name in (
        ("v1/worker/gpu/attn_utils.py", None, "build_attn_metadata"),
        ("v1/worker/gpu/model_states/default.py", "DefaultModelState", "prepare_attn"),
    ):
        source_path = Path(vllm.__file__).parent / relative_path
        body = ast.parse(source_path.read_text()).body
        if owner is not None:
            body = next(
                node for node in body
                if isinstance(node, ast.ClassDef) and node.name == owner
            ).body
        function = next(
            node for node in body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        source = ast.Module(body=[function], type_ignores=[])
        exec(compile(source, str(source_path), "exec"), namespace)
    return namespace["prepare_attn"], CUDAGraphMode


@pytest.mark.parametrize(
    "query_lens,actual_lengths,upper_bound,query_len_support,decode_threshold",
    [
        pytest.param([4], [108], [110], "SINGLE_ONLY", 1, id="single-only"),
        pytest.param([4, 2], [108, 106], [108, 108], "UNIFORM", 128, id="uniform"),
    ],
)
@pytest.mark.parametrize("graph_mode", ["PIECEWISE", "FULL", "FULL_PADDED"])
@pytest.mark.parametrize(
    "async_scheduling,adaptive_verification,has_config,needs_sync",
    [
        pytest.param(False, False, True, False, id="sync-no-mirror"),
        pytest.param(True, False, True, True, id="async-rejection"),
        pytest.param(False, True, True, True, id="adaptive-verification"),
        pytest.param(None, False, True, True, id="unresolved-scheduling"),
        pytest.param(False, None, True, True, id="unknown-adaptive-state"),
        pytest.param(None, None, False, True, id="missing-config"),
    ],
)
def test_mrv2_prefill_fallback_reuses_exact_cpu_upper_bound(
    monkeypatch, mrv2_prepare_attn, query_lens, actual_lengths, upper_bound,
    query_len_support, decode_threshold, graph_mode, async_scheduling,
    adaptive_verification, has_config, needs_sync,
):
    from vllm import envs
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", True)
    prepare_attn, cudagraph_mode = mrv2_prepare_attn
    adapter = _adapter()
    module = _fake_mla_module(adapter, [])
    module.split_decodes_and_prefills = split_decodes_and_prefills
    produced_metadata = []

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        num_decodes = module.split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
            require_uniform=True,
            treat_short_extends_as_decodes=True,
        )[0]
        return SimpleNamespace(
            common=common_attn_metadata,
            context_lens=(
                common_attn_metadata.seq_lens_cpu_upper_bound[num_decodes:]
                - common_attn_metadata.query_start_loc_cpu.diff()[num_decodes:]
            ),
        )

    module.MLACommonMetadataBuilder.build = build
    assert adapter.apply_to_module(module) is True
    metadata_builder = module.MLACommonMetadataBuilder()
    metadata_builder.query_len_support = getattr(module.QueryLenSupport, query_len_support)
    metadata_builder.reorder_batch_threshold = decode_threshold
    if has_config:
        metadata_builder.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
            scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
            speculative_config=SimpleNamespace(
                enable_adaptive_verification=adaptive_verification,
            ),
        )

    def record_and_build(common_prefix_len, common_attn_metadata):
        produced_metadata.append(common_attn_metadata)
        return metadata_builder.build(common_prefix_len, common_attn_metadata)

    num_reqs = len(query_lens)
    num_padded_reqs = 4 if graph_mode == "FULL_PADDED" else num_reqs
    padding = [0] * (num_padded_reqs - num_reqs)
    device_lengths = torch.tensor(actual_lengths + padding, dtype=torch.int32)
    cpu_lengths = upper_bound if needs_sync else actual_lengths
    original_upper_bound = torch.tensor(cpu_lengths + padding, dtype=torch.int32)
    query_offsets = torch.tensor([0] + query_lens + padding, dtype=torch.int32).cumsum(0)
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_reqs_after_padding=num_padded_reqs,
        num_tokens=sum(query_lens),
        num_tokens_after_padding=sum(query_lens) + len(padding),
        query_start_loc_np=query_offsets.numpy(),
        query_start_loc=query_offsets,
        max_query_len=max(query_lens),
        seq_lens=device_lengths,
        seq_lens_cpu_upper_bound=original_upper_bound,
        dcp_local_seq_lens=None,
        positions=None,
        is_prefilling_np=torch.zeros(num_reqs, dtype=torch.bool).numpy(),
        prompt_lens=None,
    )
    state = SimpleNamespace(supports_mm_inputs=False)
    attn_group = SimpleNamespace(
        get_metadata_builder=lambda ubatch_idx: SimpleNamespace(build=record_and_build),
        layer_names=["mla.layer"],
    )
    cpu_calls = []
    original_cpu = torch.Tensor.cpu

    def record_cpu(tensor, *args, **kwargs):
        if tensor.untyped_storage().data_ptr() == device_lengths.untyped_storage().data_ptr():
            cpu_calls.append("cpu")
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)
    result = prepare_attn(
        state, input_batch,
        getattr(cudagraph_mode, "FULL" if graph_mode.startswith("FULL") else graph_mode),
        (torch.zeros(num_padded_reqs, 1, dtype=torch.int32),),
        torch.zeros(input_batch.num_tokens_after_padding, dtype=torch.int64),
        [[attn_group]], SimpleNamespace(kv_cache_groups=[object()]),
    )["mla.layer"]

    assert len(produced_metadata) == 1
    common = produced_metadata[0]
    assert common._seq_lens_cpu is None
    assert result.common._seq_lens_cpu is None
    assert cpu_calls == (["cpu"] if needs_sync else [])
    assert result.context_lens.tolist() == [104] + padding
    assert common.seq_lens_cpu_upper_bound.tolist() == cpu_lengths + padding
    assert original_upper_bound.tolist() == cpu_lengths + padding
    if not needs_sync:
        assert result.common.seq_lens_cpu_upper_bound is common.seq_lens_cpu_upper_bound


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
        def __init__(
            self,
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
        ):
            del kv_cache_spec, layer_names, device
            self.vllm_config = vllm_config

        def build(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build=False,
        ):
            del common_prefix_len, common_attn_metadata, fast_build
            return SimpleNamespace(fp8_use_mixed_batch=False)

    class FlashMLASparseImpl:
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
            actual_num_heads=None,
        ):
            return q

    sparse_module = ModuleType(sparse_adapter.TARGET_MODULE)
    sparse_module.FlashMLASparseMetadataBuilder = FlashMLASparseMetadataBuilder
    sparse_module.FlashMLASparseImpl = FlashMLASparseImpl
    sparse_module.current_platform = SimpleNamespace(is_rocm=lambda: False)
    sparse_module.torch = torch
    sparse_module.SimpleNamespace = SimpleNamespace
    exec(
        """
def _build_fp8_separate_prefill_decode(
    self, common_attn_metadata, metadata
):
    return SimpleNamespace(num_prefills=metadata.num_prefills)
""",
        sparse_module.__dict__,
    )
    FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode = (
        sparse_module._build_fp8_separate_prefill_decode
    )
    assert sparse_adapter.apply_to_module(sparse_module) is True
    sparse_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            cp_kv_cache_interleave_size=4,
        )
    )
    sparse_builder = FlashMLASparseMetadataBuilder(
        None,
        (),
        sparse_config,
        torch.device("cpu"),
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


@pytest.mark.parametrize(
    ("cache_dtype", "expected"),
    [
        ("auto", "uniform-batch"),
        ("float16", "uniform-batch"),
        ("bfloat16", "uniform-batch"),
        ("fp8", "uniform-batch"),
        ("fp8_e4m3", "uniform-batch"),
        ("fp8_e5m2", "uniform-batch"),
        ("fp8_ds_mla", "uniform-batch"),
    ],
)
def test_flashmla_supported_kv_dtypes_retain_full_graph_support(
    cpu_flashmla,
    cache_dtype,
    expected,
):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
    )

    support = cpu_flashmla.FlashMLAMetadataBuilder.get_cudagraph_support(
        config,
        SimpleNamespace(),
    )

    assert support == expected


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
        FlashMLASparseMetadataBuilder=type(
            "UpstreamSparseMetadataBuilder", (), {}
        ),
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

        def __init__(self):
            self.kv_cache_dtype = "auto"
            self.qk_rope_head_dim = 1
            self.q_concat_buffer = torch.empty(2, 4, 4)

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
        FlashMLASparseMetadataBuilder=type(
            "UpstreamSparseMetadataBuilder", (), {}
        ),
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

    impl.q_concat_buffer = torch.empty(2, 8, 3)
    q_pe_empty = torch.empty(2, 4, 0)
    assert impl.forward_mqa((q_nope, q_pe_empty), cache, metadata, layer) == (
        "upstream-output",
        None,
    )
    upstream_query = calls[-1][1]
    assert isinstance(upstream_query, torch.Tensor)
    assert upstream_query.shape == (2, 4, 3)
    torch.testing.assert_close(upstream_query, q_nope)


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


@pytest.mark.parametrize("initial_rows", [40, 160])
def test_flashmla_fp8_graph_metadata_grows_and_keeps_only_valid_rows(
    monkeypatch,
    cpu_flashmla,
    initial_rows,
):
    flashmla = cpu_flashmla
    valid_tile_metadata = torch.arange(80 * 8, dtype=torch.int32).reshape(80, 8)
    valid_num_splits = torch.tensor([0, 3], dtype=torch.int32)
    scheduler_metadata = SimpleNamespace()

    monkeypatch.setattr(
        flashmla,
        "get_mla_metadata",
        lambda *args, **kwargs: (scheduler_metadata, None),
    )
    monkeypatch.setattr(
        flashmla,
        "get_mla_metadata_dense_fp8",
        lambda *args, **kwargs: (valid_tile_metadata, valid_num_splits),
    )
    monkeypatch.setattr(
        flashmla,
        "FlashMLADecodeMetadata",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    builder = object.__new__(flashmla.FlashMLAMetadataBuilder)
    builder.num_q_heads = 10
    builder.dcp_world_size = 1
    builder.is_fp8_kvcache = True
    # A one-token decode needs 160 rows, while a later multi-token MTP
    # verification needs only 80. Reusing the larger persistent buffer must
    # not expose its zero-padded tail to the native kernel.
    builder.cg_buf_tile_scheduler_metadata = torch.full(
        (initial_rows, 8),
        -1,
        dtype=torch.int32,
    )
    builder.cg_buf_num_splits = torch.full((9,), -1, dtype=torch.int32)
    builder.compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: True)
    )

    result = builder._build_decode(
        block_table_tensor=torch.zeros(1, 3, dtype=torch.int32),
        seq_lens_device=torch.tensor([139], dtype=torch.int32),
        max_seq_len=139,
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_device=torch.tensor([0, 2], dtype=torch.int32),
        num_decode_tokens=2,
        dcp_tot_seq_lens_device=None,
    )

    assert result.scheduler_metadata.tile_scheduler_metadata.shape == (80, 8)
    torch.testing.assert_close(
        result.scheduler_metadata.tile_scheduler_metadata,
        valid_tile_metadata,
    )
    torch.testing.assert_close(
        result.scheduler_metadata.num_splits,
        valid_num_splits,
    )


@pytest.mark.parametrize("local_lse", [0.1, float("-inf")])
def test_dcp_single_sink_matches_global_softmax(local_lse):
    local_lse, other_lse, sink = torch.tensor(
        [local_lse, 0.7, 1.2], dtype=torch.float64
    )
    local_value, other_value = torch.tensor([2.0, 4.0], dtype=torch.float64)
    rank0_lse = torch.logaddexp(local_lse, sink)
    rank0_output = torch.where(
        torch.isneginf(local_lse),
        0.0,
        torch.exp(local_lse - rank0_lse) * local_value,
    )

    merged = (
        torch.exp(rank0_lse) * rank0_output
        + torch.exp(other_lse) * other_value
    ) / (torch.exp(rank0_lse) + torch.exp(other_lse))
    oracle = (
        torch.exp(local_lse) * local_value
        + torch.exp(other_lse) * other_value
    ) / (torch.exp(local_lse) + torch.exp(other_lse) + torch.exp(sink))
    double_sink = (
        torch.exp(local_lse) * local_value
        + torch.exp(other_lse) * other_value
    ) / (torch.exp(local_lse) + torch.exp(other_lse) + 2 * torch.exp(sink))

    torch.testing.assert_close(merged, oracle)
    assert not torch.isclose(double_sink, oracle)


def test_hyv4_fp8_dcp_keeps_scattered_local_slots_for_hcu_flashmla(monkeypatch):
    from vllm.v1.attention.backends.mla import flashmla_sparse as upstream
    from vllm_hcu.models.hy_v4 import hcu_sparse
    from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseImpl

    seen = {}

    def localize(req_ids, block_table, indices, **kwargs):
        seen.update(kwargs)
        torch.testing.assert_close(
            req_ids, torch.tensor([0, 1], dtype=torch.int32)
        )
        torch.testing.assert_close(
            block_table, torch.tensor([[7], [11]], dtype=torch.int32)
        )
        assert indices.shape == (2, 4)
        return torch.tensor(
            [[112, -1, 113, -1], [-1, -1, -1, -1]],
            dtype=torch.int32,
        )

    monkeypatch.setattr(
        hcu_sparse, "triton_filter_and_convert_dcp_index", localize
    )
    impl = object.__new__(HYV4FlashMLASparseImpl)
    impl.dcp_world_size = 2
    impl.dcp_rank = 1
    impl.need_to_return_lse_for_decode = True
    kernel_args = {}

    def fp8_kernel(**kwargs):
        kernel_args.update(kwargs)
        return (
            torch.ones(1, 2, 16, 512),
            torch.zeros(1, 16, 2),
        )

    monkeypatch.setattr(impl, "_fp8_flash_mla_kernel", fp8_kernel)
    fp8 = upstream.FlashMLASparseMetadata.FP8KernelMetadata(
        scheduler_metadata=None,
        cache_lens=torch.tensor([2], dtype=torch.int32),
        dummy_block_table=torch.zeros(1, 1, dtype=torch.int32),
    )
    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        block_table=torch.tensor([[7], [11]], dtype=torch.int32),
        cp_kv_cache_interleave_size=1,
        block_size=64,
        fp8_extra_metadata=fp8,
    )

    out, lse = impl._forward_fp8_kv_mixed_batch(
        torch.zeros(2, 16, 576),
        torch.zeros(4, 656, dtype=torch.uint8),
        torch.zeros(2, 4, dtype=torch.int32),
        metadata,
    )

    assert seen["dcp_size"] == 2 and seen["dcp_rank"] == 1
    assert seen["compact_valid_to_front"] is False
    assert seen["cp_kv_cache_interleave_size"] == 1
    torch.testing.assert_close(
        kernel_args["topk_indices"],
        torch.tensor(
            [[[112, -1, 113, -1], [-1, -1, -1, -1]]],
            dtype=torch.int32,
        ),
    )
    assert kernel_args["q"].shape == (1, 2, 16, 576)
    assert "topk_length" not in kernel_args
    assert torch.count_nonzero(out[1]) == 0
    assert torch.isneginf(lse[1]).all()
