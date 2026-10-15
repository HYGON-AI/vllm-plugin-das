# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU layout preferences through the real vLLM negotiation API."""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.config import CacheConfig
from vllm.v1.attention.backends import utils
from vllm.v1.kv_cache_interface import (
    KVCacheTensor,
    MLAAttentionSpec,
    create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout as Layout
from vllm_hcu.platforms import hcu
from vllm_hcu.v1.attention.kv_cache_layout import prefer_hnd_layouts


BACKENDS = (
    ("vllm_hcu.v1.attention.backends.mla.flashmla", "HcuFlashMLABackend"),
    (
        "vllm_hcu.v1.attention.backends.mla.flashmla_sparse",
        "HcuFlashMLASparseBackend",
    ),
)


@pytest.fixture(autouse=True)
def layout_environment(monkeypatch):
    # vLLM may memoize env reads in a running process. Do not let another test's
    # explicit value affect negotiation here, or leak a cached value afterwards.
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    clear = getattr(envs.__getattr__, "cache_clear", lambda: None)
    clear()
    yield
    clear()


@pytest.fixture(params=BACKENDS, ids=[name for _, name in BACKENDS])
def backend(request):
    module, name = request.param
    return getattr(importlib.import_module(module), name)


def _set_architecture(monkeypatch, arch):
    monkeypatch.setattr(hcu, "on_gfx93x", lambda: arch in ("gfx936", "gfx938"))
    monkeypatch.setattr(hcu, "on_gfx938", lambda: arch == "gfx938")


@pytest.fixture
def target_arch(monkeypatch):
    _set_architecture(monkeypatch, "gfx938")
    return "gfx938"


def test_gfx938_preserves_unrestricted_support_without_export(target_arch):
    result = prefer_hnd_layouts(None)
    assert result[0] is Layout.LBHNC
    assert set(result) == set(Layout)
    assert "VLLM_KV_CACHE_LAYOUT" not in os.environ


@pytest.mark.parametrize(
    "supported",
    [None, (), (Layout.LBNHC,), (Layout.LBNHC, Layout.LBHNC)],
)
@pytest.mark.parametrize("arch", ("gfx936", "gfx90a"))
def test_other_architectures_keep_inherited_declaration(monkeypatch, supported, arch):
    _set_architecture(monkeypatch, arch)
    assert prefer_hnd_layouts(supported) is supported


@pytest.mark.parametrize(
    "supported",
    [(), (Layout.LBNHC,), (Layout.BLNHC, Layout.LBNHC)],
)
def test_hnd_is_never_added_to_a_restricted_supported_set(target_arch, supported):
    assert prefer_hnd_layouts(supported) is supported


def test_reordering_keeps_remaining_preferences(target_arch):
    supported = (Layout.BLNHC, Layout.LBNHC, Layout.LBHNC)
    assert prefer_hnd_layouts(supported) == (Layout.LBHNC, Layout.BLNHC, Layout.LBNHC)


def test_gfx938_backend_prefers_hnd_without_narrowing_support(backend, target_arch):
    inherited = super(backend, backend).supported_kv_cache_layouts()
    actual = backend.supported_kv_cache_layouts()
    expected_set = set(Layout) if inherited is None else set(inherited)
    assert set(actual) == expected_set
    assert actual[0] is Layout.LBHNC


def _resolve(backend, *, resolved=None):
    getattr(envs.__getattr__, "cache_clear", lambda: None)()
    cache_config = CacheConfig(block_size=64)
    cache_config.kv_cache_layout = resolved
    config = SimpleNamespace(
        cache_config=cache_config,
        kv_transfer_config=None,
    )
    supported = utils.get_supported_kv_cache_layouts([backend])
    result = utils.resolve_kv_cache_layout(config, [[v.name for v in supported]])
    assert config.cache_config.kv_cache_layout == result.name
    return result


@pytest.mark.parametrize("arch", ("gfx936", "gfx90a"))
def test_other_architectures_keep_backend_declaration_and_default(
    monkeypatch, backend, arch
):
    _set_architecture(monkeypatch, arch)
    inherited_backend = backend.__mro__[1]
    assert (
        backend.supported_kv_cache_layouts()
        == inherited_backend.supported_kv_cache_layouts()
    )
    assert _resolve(backend) is _resolve(inherited_backend)


@pytest.mark.parametrize("arch", ("gfx936", "gfx938", "gfx90a"))
def test_flash_attention_keeps_inherited_declaration_and_default(monkeypatch, arch):
    _set_architecture(monkeypatch, arch)
    from vllm_hcu.v1.attention.backends.flash_attn import HcuFlashAttentionBackend

    inherited_backend = HcuFlashAttentionBackend.__mro__[1]
    assert "supported_kv_cache_layouts" not in HcuFlashAttentionBackend.__dict__
    assert (
        HcuFlashAttentionBackend.supported_kv_cache_layouts()
        == inherited_backend.supported_kv_cache_layouts()
    )
    assert _resolve(HcuFlashAttentionBackend) is _resolve(inherited_backend)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, Layout.LBHNC), ("NHD", Layout.LBNHC), ("HND", Layout.LBHNC)],
)
def test_real_resolver_default_and_explicit_settings(
    monkeypatch, backend, target_arch, requested, expected
):
    if requested is not None:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", requested)
    assert _resolve(backend) is expected
    assert os.environ.get("VLLM_KV_CACHE_LAYOUT") == requested


def test_connector_preference_and_explicit_override(monkeypatch, backend, target_arch):
    monkeypatch.setattr(utils, "get_kv_connector_cache_layout", lambda _: "NHD")
    assert _resolve(backend) is Layout.LBNHC
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    assert _resolve(backend) is Layout.LBHNC


@pytest.mark.parametrize("layout", list(Layout), ids=lambda layout: layout.name)
def test_connector_layout_set_is_not_narrowed(
    monkeypatch, backend, target_arch, layout
):
    monkeypatch.setattr(utils, "get_kv_connector_cache_layout", lambda _: layout.name)
    assert _resolve(backend) is layout


def test_already_resolved_config_wins(monkeypatch, backend, target_arch):
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    assert _resolve(backend, resolved="LBNHC") is Layout.LBNHC


def test_hybrid_backend_restrictions_remain_authoritative(
    monkeypatch, backend, target_arch
):

    class NhdOnly:
        @classmethod
        def supported_kv_cache_layouts(cls):
            return (Layout.LBNHC,)

    supported = utils.get_supported_kv_cache_layouts([backend, NhdOnly])
    assert supported == [Layout.LBNHC]
    config = SimpleNamespace(cache_config=CacheConfig(), kv_transfer_config=None)
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    with pytest.raises(ValueError, match="does not satisfy"):
        utils.resolve_kv_cache_layout(config, [[v.name for v in supported]])


@pytest.mark.parametrize(
    ("dtype", "width"),
    [
        (torch.bfloat16, 576),
        (torch.bfloat16, 512),
        (torch.float8_e4m3fn, 576),
        (torch.float8_e5m2, 576),
        (torch.uint8, 656),
    ],
)
def test_single_head_mla_layout_views_have_identical_bytes(dtype, width):
    spec = MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=width, dtype=dtype)
    size = 3 * 2 * spec.page_size_bytes
    raw = (torch.arange(size, dtype=torch.int32) % 251).to(torch.int8)
    tensor = KVCacheTensor(
        size=size,
        layers=["layer0", "layer1"],
        layer_stride=3 * spec.page_size_bytes,
        block_stride=spec.page_size_bytes,
    )
    nhd = create_kv_cache_views(raw, spec, 3, Layout.LBNHC, tensor)
    hnd = create_kv_cache_views(raw, spec, 3, Layout.LBHNC, tensor)
    for first, second in zip(nhd, hnd):
        assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))
        assert [(n, s) for n, s in zip(first.shape, first.stride()) if n > 1] == [
            (n, s) for n, s in zip(second.shape, second.stride()) if n > 1
        ]
