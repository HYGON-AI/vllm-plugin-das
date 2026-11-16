# SPDX-License-Identifier: Apache-2.0
"""Manager blocks must preserve upstream page and SSM alignment."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from vllm.config import vllm as config_module
from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import MultipleOf
from vllm_hcu.platforms.hcu import HCUPlatform


def _backend(name, sizes, ssm=False):
    return type(name, (), {
        "is_ssm": staticmethod(lambda: ssm),
        "get_name": staticmethod(lambda: name),
        "get_supported_kernel_block_sizes": staticmethod(lambda: sizes),
    })


def _run(monkeypatch, upstream_size, rules, *, hybrid=False, heterogeneous=False,
         user_specified=False):
    cache = SimpleNamespace(
        block_size=16, user_specified_block_size=user_specified,
        kv_cache_dtype_skip_layers=[0] if heterogeneous else [],
        mamba_page_size_padded=8192 if hybrid else None,
        mamba_block_size=upstream_size if hybrid else None,
        skip_page_size_padded=8192 if heterogeneous else None,
    )
    config = SimpleNamespace(
        cache_config=cache, attention_config=SimpleNamespace(backend=None),
        model_config=SimpleNamespace(is_hybrid=hybrid),
    )
    backends = [_backend(f"Backend{i}", sizes) for i, sizes in enumerate(rules)]
    # The SSM's kernel block must never become an attention intersection rule.
    backends.append(_backend("Mamba", [999], ssm=True))
    layers = {
        str(i): SimpleNamespace(get_attn_backend=lambda b=b: b)
        for i, b in enumerate(backends)
    }
    monkeypatch.setattr(
        Platform, "update_block_size_for_backend",
        classmethod(lambda cls, cfg: setattr(cfg.cache_config, "block_size", upstream_size)),
    )
    monkeypatch.setattr(config_module, "get_layers_from_vllm_config", lambda *a: layers)
    monkeypatch.setattr(config_module, "set_current_vllm_config", lambda cfg: nullcontext())
    try:
        HCUPlatform.update_block_size_for_backend(config)
    except ValueError:
        assert cache.block_size == upstream_size
        assert cache.mamba_page_size_padded == (8192 if hybrid else None)
        assert cache.skip_page_size_padded == (8192 if heterogeneous else None)
        raise
    return cache


@pytest.mark.parametrize("hybrid,heterogeneous", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("manager_size", [256, 1152])
def test_preserves_large_manager_blocks_and_upstream_page_hints(
    monkeypatch, hybrid, heterogeneous, manager_size
):
    cache = _run(monkeypatch, manager_size, [[MultipleOf(16)], [64]],
                 hybrid=hybrid, heterogeneous=heterogeneous)
    assert cache.block_size == manager_size
    assert cache.mamba_block_size == (manager_size if hybrid else None)
    assert cache.mamba_page_size_padded == (8192 if hybrid else None)
    assert cache.skip_page_size_padded == (8192 if heterogeneous else None)


@pytest.mark.parametrize("upstream,rules,expected", [
    (16, [[MultipleOf(16)], [64]], 64),
    (16, [[MultipleOf(16)], [MultipleOf(24)]], 48),
    (16, [[MultipleOf(16)], [768]], 768),
    (96, [[MultipleOf(16)], [64]], 192),
    (128, [[64, 128], [MultipleOf(16)]], 128),
    (16, [[MultipleOf(24), MultipleOf(32)], [MultipleOf(16)]], 32),
])
def test_grows_ordinary_manager_using_actual_kernel_rules(monkeypatch, upstream, rules, expected):
    assert _run(monkeypatch, upstream, rules).block_size == expected


@pytest.mark.parametrize("hybrid,heterogeneous", [(True, False), (False, True)])
def test_incompatible_page_alignment_fails_without_changing_manager(
    monkeypatch, hybrid, heterogeneous
):
    with pytest.raises(ValueError, match="Upstream-aligned KV manager block size 96"):
        _run(monkeypatch, 96, [[MultipleOf(16)], [64]],
             hybrid=hybrid, heterogeneous=heterogeneous)


def test_disjoint_kernel_sizes_fail_instead_of_retaining_invalid_block(monkeypatch):
    with pytest.raises(ValueError, match="No common attention kernel block size"):
        _run(monkeypatch, 64, [[64], [128]])


@pytest.mark.parametrize("rules", [[[], [64]], [[0], [64]], [[MultipleOf(0)], [64]]])
def test_invalid_or_empty_kernel_rules_fail_closed(monkeypatch, rules):
    with pytest.raises(ValueError):
        _run(monkeypatch, 64, rules)


def test_user_specified_block_is_not_changed_by_mixed_backend_override(monkeypatch):
    assert _run(monkeypatch, 96, [[16], [64]], user_specified=True).block_size == 96
