# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for target-owned MLA execution and HCU CAT routing."""

from __future__ import annotations

import importlib
import importlib.util
import sys
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


def _fake_mla_module(adapter, target_calls):
    split_calls = []

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
    common = object()
    assert module.split_decodes_and_prefills(common, 3, True, True) is False
    assert module.split_calls == [(common, 3, True, False)]


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
