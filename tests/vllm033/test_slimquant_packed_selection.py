# SPDX-License-Identifier: Apache-2.0
"""Actual format selector with CPU host/backend boundaries; no GPU claims."""

from types import SimpleNamespace

import pytest
from hcu_adapter.extensions import ExtensionCommitError
from tests.vllm033.test_slimquant import checkpoint_config
from tests.vllm033.test_slimquant_packed import api, moe
from tests.vllm033.test_slimquant_packed_activation import (
    vendor as vendor,
    registry as registry,
)

# Export imported pytest fixtures for collection in this module.
__all__ = ["registry", "vendor"]

PREFIX = "model.layers.0.mlp.experts"


def setup(monkeypatch, provider="boltops", ignores=None):
    import torch
    import hcu_vllm_das.slimquant as impl

    context = SimpleNamespace(
        additional_config={"hcu_slimquant": {"moe_provider": provider}}
    )
    monkeypatch.setattr(impl, "get_current_vllm_config_or_none", lambda: context)
    expanded = object()
    unquantized = object()
    # Backend constructors are metadata-only boundaries in this CPU selector
    # test. The selected native class and its preparation method remain real.
    monkeypatch.setattr(impl, "SlimQuantInt8Experts", lambda *args: expanded)
    monkeypatch.setattr(impl, "UnquantizedFusedMoEMethod", lambda *args: unquantized)
    layer = object.__new__(impl.RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = moe()
    raw = checkpoint_config()
    if ignores is not None:
        raw["ignore"] = ignores
    return impl.SlimQuantConfig.from_config(raw), layer, expanded, unquantized


def test_explicit_native_selection_returns_prepared_packed_method(vendor, monkeypatch):
    state, function, _ = vendor
    config, layer, _, _ = setup(monkeypatch)
    method = config.get_quant_method(layer, PREFIX)
    assert isinstance(method, api().SlimQuantPackedExperts)
    assert method._provider is function and state["events"].count("import") == 1


def test_native_requires_public_derived_buffer_api_before_vendor_prepare(
    vendor, monkeypatch
):
    from vllm.model_executor import utils

    state, _, _ = vendor
    config, layer, _, _ = setup(monkeypatch)
    monkeypatch.delattr(utils, "register_derived_buffer")
    with pytest.raises(ValueError, match="Core derived-buffer reload APIs"):
        config.get_quant_method(layer, PREFIX)
    assert not state["events"]


def test_default_expanded_selection_never_requires_optional_vendor(vendor, monkeypatch):
    state, _, _ = vendor
    state["missing"] = True
    config, layer, expanded, _ = setup(monkeypatch, "expanded-int8")
    assert config.get_quant_method(layer, PREFIX) is expanded
    assert not state["events"]


def test_native_option_keeps_fully_ignored_experts_unquantized(vendor, monkeypatch):
    state, _, _ = vendor
    state["missing"] = True
    config, layer, _, unquantized = setup(monkeypatch, ignores=[PREFIX])
    assert config.get_quant_method(layer, PREFIX) is unquantized
    assert not state["events"]


def test_partial_ignore_refuses_before_native_dependency_preparation(
    vendor, monkeypatch
):
    state, _, _ = vendor
    config, layer, _, _ = setup(monkeypatch, ignores=[PREFIX + ".gate_up_proj"])
    with pytest.raises(ValueError, match="partially ignored expert"):
        config.get_quant_method(layer, PREFIX)
    assert not state["events"]


def test_unknown_native_provider_is_not_silently_expanded(vendor, monkeypatch):
    config, layer, _, _ = setup(monkeypatch, "typo")
    with pytest.raises(ValueError, match="moe_provider"):
        config.get_quant_method(layer, PREFIX)


def test_poisoned_native_preparation_also_refuses_default_format_reuse(
    vendor, monkeypatch
):
    from hcu_vllm_das import slimquant_activation as activation

    config, layer, _, _ = setup(monkeypatch, "expanded-int8")
    activation._failure = True
    with pytest.raises(ExtensionCommitError):
        config.get_quant_method(layer, PREFIX)
    with pytest.raises(ExtensionCommitError):
        config.get_embedding_method("ignored.ple", object)
