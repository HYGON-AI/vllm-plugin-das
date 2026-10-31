from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

PLUGIN_SOURCE_ROOT = (
    Path(__file__).resolve().parents[2] / "adapter_extension" / "src"
)
sys.path.insert(0, os.fspath(PLUGIN_SOURCE_ROOT))


def load_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_vllm_seams: bool = True,
    kv_cache_layout: str = "LBHNC",
):
    class FlashAttentionImpl:
        supports_dcp = True

        def __init__(self, *args, **kwargs):
            kv_cache_dtype = kwargs.get(
                "kv_cache_dtype",
                args[6] if len(args) > 6 else None,
            )
            self.upstream_kv_cache_dtype = kv_cache_dtype
            if kv_cache_dtype == "fp8_e5m2":
                raise NotImplementedError("upstream rejects fp8_e5m2")
            self.kv_cache_dtype = kv_cache_dtype
            self.supports_quant_query_input = True
            self.vllm_flash_attn_version = kwargs.pop("version", None)

    if with_vllm_seams:
        for name in (
            "_get_kv_cache_views",
            "_get_kv_cache_write_views",
            "_run_flash_attention",
        ):
            setattr(FlashAttentionImpl, name, lambda self, *args, **kwargs: None)

    class FlashAttentionBackend:
        @classmethod
        def get_preferred_block_size(cls, default_block_size):
            return default_block_size

        @classmethod
        def supports_kv_cache_dtype(cls, kv_cache_dtype):
            return kv_cache_dtype != "fp8_e5m2"

        @classmethod
        def supports_combination(
            cls,
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        ):
            if kv_cache_dtype == "fp8_e5m2":
                return "upstream rejects fp8_e5m2"
            return "delegated-combination-result"

    vllm = ModuleType("vllm")
    vllm.__path__ = []
    v1 = ModuleType("vllm.v1")
    v1.__path__ = []
    attention = ModuleType("vllm.v1.attention")
    attention.__path__ = []
    backends = ModuleType("vllm.v1.attention.backends")
    backends.__path__ = []
    official = ModuleType("vllm.v1.attention.backends.flash_attn")
    official.FlashAttentionImpl = FlashAttentionImpl
    official.FlashAttentionBackend = FlashAttentionBackend
    layouts = ModuleType("vllm.v1.kv_cache_layout")
    layouts.KVCacheLayout = SimpleNamespace(LBHNC="LBHNC", LBNHC="LBNHC")
    config = ModuleType("vllm.config")
    cache_config = SimpleNamespace(
        get_resolved_kv_cache_layout=lambda: kv_cache_layout
    )
    config.get_current_vllm_config_or_none = lambda: SimpleNamespace(
        cache_config=cache_config
    )
    for name, module in (
        ("vllm", vllm),
        ("vllm.config", config),
        ("vllm.v1", v1),
        ("vllm.v1.attention", attention),
        ("vllm.v1.attention.backends", backends),
        ("vllm.v1.attention.backends.flash_attn", official),
        ("vllm.v1.kv_cache_layout", layouts),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(
        sys.modules,
        "hcu_vllm_das.attention.flash_attn",
        raising=False,
    )
    return importlib.import_module("hcu_vllm_das.attention.flash_attn")


def test_import_rejects_vllm_without_required_flash_attention_seams(monkeypatch):
    with pytest.raises(RuntimeError, match="_get_kv_cache_write_views"):
        load_flash_attention(monkeypatch, with_vllm_seams=False)


@pytest.mark.parametrize(
    (
        "layout",
        "packed_inner_strides",
        "read_shape",
        "read_inner_strides",
    ),
    (
        (
            "LBHNC",
            lambda h, n, d: (n * 2 * d, 2 * d, 1),
            lambda b, h, n, d: (b, h, n, d),
            lambda h, n, d: (n * d, d, 1),
        ),
        (
            "LBNHC",
            lambda h, n, d: (2 * d, h * 2 * d, 1),
            lambda b, h, n, d: (b, n, h, d),
            lambda h, n, d: (h * d, d, 1),
        ),
    ),
)
def test_dense_views_separate_kv_and_preserve_block_stride(
    monkeypatch,
    layout,
    packed_inner_strides,
    read_shape,
    read_inner_strides,
):
    flash = load_flash_attention(monkeypatch, kv_cache_layout=layout)
    num_blocks, num_heads, block_size, head_size = 2, 3, 4, 5
    half_page = num_heads * block_size * head_size
    block_stride = 2 * half_page + 17
    storage_offset = 13
    raw = torch.zeros(storage_offset + block_stride * num_blocks)
    packed = torch.as_strided(
        raw,
        size=(num_blocks, num_heads, block_size, 2 * head_size),
        stride=(
            block_stride,
            *packed_inner_strides(num_heads, block_size, head_size),
        ),
        storage_offset=storage_offset,
    )

    impl = flash.HcuFlashAttentionImpl()
    impl.head_size = head_size
    key, value = impl._get_kv_cache_views(packed)
    write_key, write_value = impl._get_kv_cache_write_views(packed)
    key.fill_(11)
    value.fill_(22)

    assert key.shape == value.shape == read_shape(
        num_blocks, num_heads, block_size, head_size
    )
    assert key.stride() == value.stride() == (
        block_stride,
        *read_inner_strides(num_heads, block_size, head_size),
    )
    assert key.storage_offset() == storage_offset
    assert value.storage_offset() == storage_offset + half_page
    assert write_key.shape == write_value.shape == (
        num_blocks,
        block_size,
        num_heads,
        head_size,
    )
    expected_write_key = key.transpose(1, 2) if layout == "LBHNC" else key
    expected_write_value = value.transpose(1, 2) if layout == "LBHNC" else value
    assert write_key.stride() == expected_write_key.stride()
    assert write_value.stride() == expected_write_value.stride()
    torch.testing.assert_close(
        raw[storage_offset : storage_offset + half_page],
        torch.full((half_page,), 11.0),
    )
    torch.testing.assert_close(
        raw[storage_offset + half_page : storage_offset + 2 * half_page],
        torch.full((half_page,), 22.0),
    )
    torch.testing.assert_close(
        raw[
            storage_offset + block_stride : storage_offset
            + block_stride
            + half_page
        ],
        torch.full((half_page,), 11.0),
    )
    torch.testing.assert_close(
        raw[
            storage_offset + block_stride + half_page : storage_offset
            + block_stride
            + 2 * half_page
        ],
        torch.full((half_page,), 22.0),
    )
    torch.testing.assert_close(
        raw[storage_offset + 2 * half_page : storage_offset + block_stride],
        torch.zeros(17),
    )


def test_backend_selects_fa2_dual_layout_and_64_token_pages(monkeypatch):
    flash = load_flash_attention(monkeypatch)

    impl = flash.HcuFlashAttentionImpl()

    assert impl.vllm_flash_attn_version == 2
    assert flash.HcuFlashAttentionBackend.get_impl_cls() is type(impl)
    assert flash.HcuFlashAttentionBackend.supported_kv_cache_layouts() == (
        "LBHNC",
        "LBNHC",
    )
    assert flash.HcuFlashAttentionBackend.get_supported_kernel_block_sizes() == [64]
    assert flash.HcuFlashAttentionBackend.get_preferred_block_size(16) == 64
    assert flash.HcuFlashAttentionBackend.get_preferred_block_size(128) == 64
    assert flash.HcuFlashAttentionBackend.supports_block_size(64) is True
    assert flash.HcuFlashAttentionBackend.supports_block_size(128) is False
    assert flash.HcuFlashAttentionBackend.supported_kv_cache_dtypes == [
        "auto",
        "float16",
        "bfloat16",
        "fp8_e5m2",
    ]
    assert flash.HcuFlashAttentionBackend.supports_dcp() is False
    assert flash.HcuFlashAttentionBackend.supports_mm_prefix() is False
    assert flash.HcuFlashAttentionBackend.supports_sink() is False
    assert flash.HcuFlashAttentionBackend.supports_rswa() is False
    assert flash.HcuFlashAttentionBackend.supports_attn_type("decoder") is True
    assert flash.HcuFlashAttentionBackend.supports_attn_type("encoder") is False
    assert flash.HcuFlashAttentionBackend.supports_sliding_window() is False
    assert flash.HcuFlashAttentionBackend.supports_non_causal() is False
    assert flash.HcuFlashAttentionBackend.supports_batch_invariance() is False


def test_backend_accepts_only_hcu_e5m2_extension_and_delegates_other_dtypes(
    monkeypatch,
):
    flash = load_flash_attention(monkeypatch)
    backend = flash.HcuFlashAttentionBackend
    common = dict(
        head_size=128,
        dtype=torch.bfloat16,
        block_size=64,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=object(),
    )

    assert backend.supports_kv_cache_dtype("fp8_e5m2") is True
    assert (
        backend.supports_combination(
            kv_cache_dtype="fp8_e5m2",
            **common,
        )
        is None
    )
    assert backend.supports_kv_cache_dtype("bfloat16") is True
    assert (
        backend.supports_combination(
            kv_cache_dtype="bfloat16",
            **common,
        )
        == "delegated-combination-result"
    )


@pytest.mark.parametrize("constructor_style", ("keyword", "positional"))
def test_e5m2_constructor_bypasses_only_upstream_guard(
    monkeypatch,
    constructor_style,
):
    flash = load_flash_attention(monkeypatch)

    if constructor_style == "keyword":
        impl = flash.HcuFlashAttentionImpl(kv_cache_dtype="fp8_e5m2")
    else:
        impl = flash.HcuFlashAttentionImpl(
            4,
            64,
            0.125,
            2,
            None,
            None,
            "fp8_e5m2",
        )

    assert impl.upstream_kv_cache_dtype == "auto"
    assert impl.kv_cache_dtype == "fp8_e5m2"
    assert impl.supports_quant_query_input is False


def test_non_e5m2_constructor_preserves_upstream_dtype_state(monkeypatch):
    flash = load_flash_attention(monkeypatch)

    impl = flash.HcuFlashAttentionImpl(kv_cache_dtype="bfloat16")

    assert impl.upstream_kv_cache_dtype == "bfloat16"
    assert impl.kv_cache_dtype == "bfloat16"
    assert impl.supports_quant_query_input is True


@pytest.mark.parametrize(
    "incoming_dtype",
    (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2),
)
def test_e5m2_vendor_boundary_reinterprets_kv_bytes_only(
    monkeypatch,
    incoming_dtype,
):
    flash = load_flash_attention(monkeypatch)
    calls = []
    monkeypatch.setattr(
        flash,
        "_vendor_varlen_func",
        lambda: lambda **kwargs: calls.append(kwargs) or "output",
    )
    impl = flash.HcuFlashAttentionImpl()
    impl.kv_cache_dtype = "fp8_e5m2"
    query = torch.empty(1, 4, 64, dtype=torch.bfloat16)
    raw_key = torch.arange(64, dtype=torch.uint8)
    raw_value = torch.arange(64, dtype=torch.uint8).flip(0)
    key = raw_key if incoming_dtype == torch.uint8 else raw_key.view(incoming_dtype)
    value = (
        raw_value
        if incoming_dtype == torch.uint8
        else raw_value.view(incoming_dtype)
    )
    k_descale = torch.tensor(0.5)
    v_descale = torch.tensor(0.25)

    result = impl._run_flash_attention(
        q=query,
        k=key,
        v=value,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    assert result == "output"
    assert calls[0]["q"] is query
    assert calls[0]["k"].dtype == torch.float8_e5m2
    assert calls[0]["v"].dtype == torch.float8_e5m2
    assert calls[0]["k"].view(torch.uint8).data_ptr() == raw_key.data_ptr()
    assert calls[0]["v"].view(torch.uint8).data_ptr() == raw_value.data_ptr()
    assert calls[0]["k_descale"] is k_descale
    assert calls[0]["v_descale"] is v_descale


def test_e5m2_vendor_boundary_rejects_non_byte_kv_storage(monkeypatch):
    flash = load_flash_attention(monkeypatch)
    monkeypatch.setattr(
        flash,
        "_vendor_varlen_func",
        lambda: pytest.fail("vendor must not run"),
    )
    impl = flash.HcuFlashAttentionImpl()
    impl.kv_cache_dtype = "fp8_e5m2"

    with pytest.raises(ValueError, match="one-byte K/V storage"):
        impl._run_flash_attention(
            q=torch.empty(1, 4, 64, dtype=torch.bfloat16),
            k=torch.empty(64, dtype=torch.bfloat16),
            v=torch.empty(64, dtype=torch.bfloat16),
        )


def test_non_e5m2_vendor_boundary_preserves_kv_dtype(monkeypatch):
    flash = load_flash_attention(monkeypatch)
    calls = []
    monkeypatch.setattr(
        flash,
        "_vendor_varlen_func",
        lambda: lambda **kwargs: calls.append(kwargs),
    )
    impl = flash.HcuFlashAttentionImpl()
    impl.kv_cache_dtype = "auto"
    key = torch.empty(64, dtype=torch.float8_e4m3fn)
    value = torch.empty_like(key)

    impl._run_flash_attention(q="q", k=key, v=value)

    assert calls[0]["k"] is key
    assert calls[0]["v"] is value


@pytest.mark.parametrize("head_size", (64, 128, 192, 256))
def test_backend_accepts_vendor_supported_head_sizes(monkeypatch, head_size):
    flash = load_flash_attention(monkeypatch)

    assert flash.HcuFlashAttentionBackend.supports_head_size(head_size) is True


@pytest.mark.parametrize("head_size", (8, 32, 80, 96, 264))
def test_backend_rejects_vendor_unsupported_head_sizes(monkeypatch, head_size):
    flash = load_flash_attention(monkeypatch)

    assert flash.HcuFlashAttentionBackend.supports_head_size(head_size) is False


@pytest.mark.parametrize(
    ("kv_cache_layout", "kernel_layout"),
    (("LBHNC", "bhsd"), ("LBNHC", "bshd")),
)
def test_varlen_boundary_forwards_fa2_keywords_without_fa4_options(
    monkeypatch, kv_cache_layout, kernel_layout
):
    flash = load_flash_attention(
        monkeypatch, kv_cache_layout=kv_cache_layout
    )
    calls = []

    def vendor(**kwargs):
        calls.append(kwargs)
        return "vendor-output"

    monkeypatch.setattr(flash, "_vendor_varlen_func", lambda: vendor)
    impl = flash.HcuFlashAttentionImpl()

    result = impl._run_flash_attention(
        q="q",
        k="k",
        v="v",
        out="out",
        dynamic_causal=None,
        mask_mod=None,
        aux_tensors=None,
        num_splits=3,
    )

    assert result == "vendor-output"
    assert calls == [
        {
            "q": "q",
            "k": "k",
            "v": "v",
            "out": "out",
            "num_splits": 3,
            "layout": kernel_layout,
            "fa_version": 2,
        }
    ]


def test_layout_selection_uses_resolved_config_when_geometry_is_ambiguous(
    monkeypatch,
):
    flash = load_flash_attention(monkeypatch, kv_cache_layout="LBNHC")
    calls = []
    monkeypatch.setattr(
        flash,
        "_vendor_varlen_func",
        lambda: lambda **kwargs: calls.append(kwargs),
    )
    impl = flash.HcuFlashAttentionImpl()
    impl.head_size = 8
    packed = torch.empty_strided(
        (1, 4, 4, 16),
        (256, 16, 64, 1),
    )

    key, _ = impl._get_kv_cache_views(packed)
    impl._run_flash_attention(q="q", k=key, v=key)

    assert key.shape == (1, 4, 4, 8)
    assert key.stride() == (256, 32, 8, 1)
    assert calls[0]["layout"] == "bshd"


def test_resolved_layout_rejects_other_layout_strides(monkeypatch):
    flash = load_flash_attention(monkeypatch, kv_cache_layout="LBHNC")
    impl = flash.HcuFlashAttentionImpl()
    impl.head_size = 8
    nhd_packed = torch.empty_strided(
        (1, 4, 4, 16),
        (256, 16, 64, 1),
    )

    with pytest.raises(ValueError, match="strides do not match resolved layout"):
        impl._get_kv_cache_views(nhd_packed)


def test_unknown_resolved_layout_fails_closed(monkeypatch):
    flash = load_flash_attention(monkeypatch, kv_cache_layout="LHBNC")
    impl = flash.HcuFlashAttentionImpl()
    impl.head_size = 8

    with pytest.raises(ValueError, match="unsupported resolved KV cache layout"):
        impl._get_kv_cache_views(torch.empty(1, 1, 1, 16))


def test_varlen_boundary_avoids_vendor_short_prefill_mtp_misclassification(
    monkeypatch,
):
    flash = load_flash_attention(monkeypatch)
    calls = []

    def vendor(**kwargs):
        calls.append(kwargs)
        return "vendor-output"

    monkeypatch.setattr(flash, "_vendor_varlen_func", lambda: vendor)
    impl = flash.HcuFlashAttentionImpl()

    impl._run_flash_attention(
        q=torch.empty(5, 14, 64),
        k=torch.empty(1, 2, 64, 64),
        v=torch.empty(1, 2, 64, 64),
        cu_seqlens_q=torch.tensor([0, 5], dtype=torch.int32),
        max_seqlen_q=5,
        block_table=torch.zeros((1, 1), dtype=torch.int32),
    )

    assert calls[0]["max_seqlen_q"] == 6


@pytest.mark.parametrize(
    ("max_seqlen_q", "query_tokens", "batch_size", "has_block_table"),
    (
        (1, 1, 1, True),
        (16, 16, 1, True),
        (5, 9, 2, True),
        (5, 5, 1, False),
    ),
)
def test_varlen_boundary_preserves_non_rectangular_or_non_paged_shapes(
    monkeypatch,
    max_seqlen_q,
    query_tokens,
    batch_size,
    has_block_table,
):
    flash = load_flash_attention(monkeypatch)
    calls = []
    monkeypatch.setattr(
        flash,
        "_vendor_varlen_func",
        lambda: lambda **kwargs: calls.append(kwargs),
    )
    impl = flash.HcuFlashAttentionImpl()

    impl._run_flash_attention(
        q=torch.empty(query_tokens, 4, 64),
        k=torch.empty(1, 2, 64, 64),
        v=torch.empty(1, 2, 64, 64),
        cu_seqlens_q=torch.arange(
            batch_size + 1,
            dtype=torch.int32,
        ),
        max_seqlen_q=max_seqlen_q,
        block_table=(
            torch.zeros((batch_size, 1), dtype=torch.int32)
            if has_block_table
            else None
        ),
    )

    assert calls[0]["max_seqlen_q"] == max_seqlen_q


@pytest.mark.parametrize("name", ("dynamic_causal", "mask_mod", "aux_tensors"))
def test_varlen_boundary_rejects_active_fa4_options(monkeypatch, name):
    flash = load_flash_attention(monkeypatch)
    monkeypatch.setattr(
        flash,
        "_vendor_varlen_func",
        lambda: pytest.fail("vendor must not run"),
    )
    impl = flash.HcuFlashAttentionImpl()

    with pytest.raises(NotImplementedError, match=name):
        impl._run_flash_attention(q="q", k="k", v="v", **{name: object()})
