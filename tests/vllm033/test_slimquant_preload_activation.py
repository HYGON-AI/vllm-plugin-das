# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import pytest
from hcu_adapter.extensions import ExtensionCommitError
from hcu_vllm_das.quantization import preload_activation as owner


def context(request=()):
    return SimpleNamespace(vllm_version="0.33.0rc0",requested_extensions=frozenset(request),requested_capabilities=frozenset())


def test_requires_explicit_request():
    with pytest.raises(RuntimeError,match="explicit"):
        owner.activate(context())


def test_public_registration_with_preload_bridge_double_and_latched_drift(monkeypatch):
    # The MiniMax adapter branch does not ship the optional Kimi preload bridge.
    # Model/quantization registration is real; only this deployment dependency
    # is supplied as a double, never claimed as actual preload validation.
    import sys
    from types import ModuleType
    bridge = ModuleType("hcu_adapter.bridges.vllm_kimi31_preload")
    bridge.selected_mode = lambda: "nvidia-load-only"
    bridge.require_healthy = lambda: None
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    import vllm.model_executor.layers.quantization as q
    from vllm.model_executor.models.registry import ModelRegistry as registry
    from vllm.platforms import current_platform
    monkeypatch.setattr(bridge,"selected_mode",lambda:"nvidia-load-only")
    monkeypatch.setattr(bridge,"require_healthy",lambda:None)
    previous_supported=list(current_platform.supported_quantization or [])
    assert "slimquant_w4a8" not in q.QUANTIZATION_METHODS
    try:
        owner.activate(context([owner.ENTRY]))
        owner.check_health()
        from hcu_vllm_das.quantization.kimi31_load_model import OUTER_ARCH, INNER_ARCH
        assert OUTER_ARCH in registry.models and INNER_ARCH in registry.models
        assert q.get_quantization_config("slimquant_w4a8").get_name() == "slimquant_w4a8"
        registry.models.pop(OUTER_ARCH)
        with pytest.raises(ExtensionCommitError,match="ownership"):
            owner.check_health()
        with pytest.raises(ExtensionCommitError): owner.activate(context([owner.ENTRY]))
    finally:
        q._CUSTOMIZED_METHOD_TO_QUANT_CONFIG.pop("slimquant_w4a8",None)
        if "slimquant_w4a8" in q.QUANTIZATION_METHODS: q.QUANTIZATION_METHODS.remove("slimquant_w4a8")
        if current_platform.supported_quantization is not None:
            current_platform.supported_quantization[:]=previous_supported
        registry.models.pop("HcuKimi31SlimQuantLoadOnly",None)
        registry.models.pop("HcuKimi31SlimQuantLoadOnlyInner",None)
        monkeypatch.setattr(owner,"_publication",None)
        monkeypatch.setattr(owner,"_failure",None)


def test_missing_optional_preload_bridge_refuses_before_publication(monkeypatch):
    import sys
    import vllm.model_executor.layers.quantization as q
    monkeypatch.setitem(sys.modules, "hcu_adapter.bridges.vllm_kimi31_preload", None)
    monkeypatch.setattr(owner, "_publication", None)
    monkeypatch.setattr(owner, "_failure", None)
    with pytest.raises(RuntimeError, match="optional Kimi preload bridge"):
        owner.activate(context([owner.ENTRY]))
    assert owner._publication is None
    assert owner._failure is None
    assert "slimquant_w4a8" not in q._CUSTOMIZED_METHOD_TO_QUANT_CONFIG
