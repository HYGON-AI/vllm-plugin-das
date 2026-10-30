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
):
    class FlashAttentionImpl:
        supports_dcp = True

        def __init__(self, *args, **kwargs):
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
    layouts.KVCacheLayout = SimpleNamespace(LBHNC="LBHNC")
    for name, module in (
        ("vllm", vllm),
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


def test_dense_hnd_views_separate_kv_and_preserve_block_stride(monkeypatch):
    flash = load_flash_attention(monkeypatch)
    num_blocks, num_heads, block_size, head_size = 2, 3, 4, 5
    half_page = num_heads * block_size * head_size
    block_stride = 2 * half_page + 17
    storage_offset = 13
    raw = torch.zeros(storage_offset + block_stride * num_blocks)
    packed = torch.as_strided(
        raw,
        size=(num_blocks, num_heads, block_size, 2 * head_size),
        stride=(block_stride, block_size * 2 * head_size, 2 * head_size, 1),
        storage_offset=storage_offset,
    )

    key, value = flash._dense_hnd_kv_cache_views(packed, head_size)
    impl = flash.HcuFlashAttentionImpl()
    impl.head_size = head_size
    write_key, write_value = impl._get_kv_cache_write_views(packed)
    key.fill_(11)
    value.fill_(22)

    assert key.shape == value.shape == (
        num_blocks,
        num_heads,
        block_size,
        head_size,
    )
    assert key.stride() == value.stride() == (
        block_stride,
        block_size * head_size,
        head_size,
        1,
    )
    assert key.storage_offset() == storage_offset
    assert value.storage_offset() == storage_offset + half_page
    assert write_key.shape == write_value.shape == (
        num_blocks,
        block_size,
        num_heads,
        head_size,
    )
    assert write_key.stride() == key.transpose(1, 2).stride()
    assert write_value.stride() == value.transpose(1, 2).stride()
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


def test_backend_selects_fa2_hnd_layout_and_64_token_pages(monkeypatch):
    flash = load_flash_attention(monkeypatch)

    impl = flash.HcuFlashAttentionImpl()

    assert impl.vllm_flash_attn_version == 2
    assert flash.HcuFlashAttentionBackend.get_impl_cls() is type(impl)
    assert flash.HcuFlashAttentionBackend.supported_kv_cache_layouts() == (
        "LBHNC",
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
    ]
    assert flash.HcuFlashAttentionBackend.supports_dcp() is False
    assert flash.HcuFlashAttentionBackend.supports_mm_prefix() is False
    assert flash.HcuFlashAttentionBackend.supports_sink() is False
    assert flash.HcuFlashAttentionBackend.supports_rswa() is False


def test_varlen_boundary_forwards_fa2_keywords_without_fa4_options(monkeypatch):
    flash = load_flash_attention(monkeypatch)
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
            "layout": "bhsd",
            "fa_version": 2,
        }
    ]


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
