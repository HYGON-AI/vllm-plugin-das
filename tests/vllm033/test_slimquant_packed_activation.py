# SPDX-License-Identifier: Apache-2.0
"""CPU native preparation contracts using public host/vendor boundary doubles."""

from concurrent.futures import ThreadPoolExecutor
import os
import signal
from types import SimpleNamespace

import pytest
from hcu_adapter.extensions import ExtensionCommitError
from hcu_vllm_das import slimquant_activation as activation
from tests.vllm033.test_slimquant_activation import registry as registry, context

# Export imported pytest fixtures for collection in this module.
__all__ = ['registry']


def prepare():
    function = getattr(activation, "prepare_packed_provider", None)
    assert callable(function), "Native provider preparation is missing"
    return function()


@pytest.fixture
def vendor(registry, monkeypatch):
    from hcu_adapter.bridges import vllm as host

    activation.activate(context())
    original = activation.import_module
    state = dict(events=[], missing=False, fail=False, hook=None)

    def fused_experts(*args, **kwargs):
        raise AssertionError("CPU contract must not execute native kernels")

    def normalize(name):
        assert name == "silu"
        state["events"].append("normalize")
        if state["hook"]:
            state["hook"]()
        if state["fail"]:

            class BadDescription(RuntimeError):
                def __str__(self):
                    raise AssertionError("original error formatting must not run")

            raise BadDescription()
        return ("silu", True)

    module = SimpleNamespace(
        fused_experts=fused_experts, normalize_moe_activation=normalize
    )

    def load(name):
        if name == "boltops.fused_moe":
            state["events"].append("import")
            return module
        return original(name)

    def healthy(capability):
        assert capability == activation.CAPABILITY
        assert not activation._lock._is_owned(), "host health queried under owner lock"
        state["events"].append("host")
        if activation._failure:
            raise ExtensionCommitError("host observed failed owner")
        activation.check_health()

    monkeypatch.setattr(activation, "_packed_provider", None, raising=False)
    monkeypatch.setattr(activation, "_packed_provider_pid", None, raising=False)
    monkeypatch.setattr(activation, "import_module", load)
    monkeypatch.setattr(
        activation,
        "find_spec",
        lambda name: None if state["missing"] else object(),
        raising=False,
    )
    monkeypatch.setattr(host, "require_extension_healthy", healthy)
    return state, fused_experts, host


def test_host_health_precedes_vendor_import_and_native_preparation_is_idempotent(
    vendor,
):
    state, function, _ = vendor
    assert prepare() is prepare() is function
    assert state["events"] == ["host", "import", "normalize", "host"]


def test_missing_optional_dependency_is_clean_and_expanded_owner_remains_healthy(
    vendor,
):
    state, function, _ = vendor
    state["missing"] = True
    with pytest.raises(RuntimeError, match="optional BoltOps"):
        prepare()
    assert not activation._failure and not activation._activating
    activation.check_health()
    state["missing"] = False
    assert prepare() is function


def test_missing_active_host_refuses_before_import(vendor, monkeypatch):
    state, _, host = vendor

    def refuse(_):
        raise ExtensionCommitError("no ACTIVE owner")

    monkeypatch.setattr(host, "require_extension_healthy", refuse)
    with pytest.raises(ExtensionCommitError, match="ACTIVE"):
        prepare()
    assert "import" not in state["events"]


def test_partial_vendor_import_failure_poison_is_sticky_before_diagnostics(vendor):
    state, _, _ = vendor
    state["fail"] = True
    with pytest.raises(ExtensionCommitError):
        prepare()
    assert activation._failure and not activation._activating
    state["fail"] = False
    with pytest.raises(ExtensionCommitError):
        prepare()
    with pytest.raises(ExtensionCommitError):
        activation.activate(context())
    assert state["events"].count("import") == 1


def test_concurrent_native_preparation_publishes_one_callable(vendor):
    state, function, _ = vendor
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: prepare(), range(8)))
    assert all(v is function for v in values)
    assert state["events"].count("import") == state["events"].count("normalize") == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires CPU fork")
@pytest.mark.parametrize("during_preparation", [False, True])
def test_fork_never_reuses_inherited_native_provider_or_inflight_import(
    vendor, during_preparation
):
    state, _, _ = vendor

    def fork_and_check():
        pid = os.fork()
        if pid == 0:
            signal.alarm(3)
            try:
                prepare()
            except ExtensionCommitError:
                os._exit(0)
            except BaseException:
                os._exit(2)
            os._exit(3)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0

    if during_preparation:
        state["hook"] = fork_and_check
        prepare()
    else:
        prepare()
        fork_and_check()
    activation.check_health()


def test_dependency_finder_failure_is_not_assumed_side_effect_free(vendor, monkeypatch):
    state, _, _ = vendor

    def find(_):
        state["finder_side_effect"] = True
        raise RuntimeError("finder failed after side effect")

    monkeypatch.setattr(activation, "find_spec", find)
    with pytest.raises(ExtensionCommitError):
        prepare()
    assert state["finder_side_effect"] and activation._failure
    with pytest.raises(ExtensionCommitError):
        activation.activate(context())


def test_async_callable_object_is_not_published_as_native_kernel(vendor, monkeypatch):
    _, _, _ = vendor
    original = activation.import_module

    class AsyncKernel:
        async def __call__(self, *args, **kwargs):
            return None

    def load(name):
        module = original(name)
        if name == "boltops.fused_moe":
            return SimpleNamespace(
                fused_experts=AsyncKernel(),
                normalize_moe_activation=module.normalize_moe_activation,
            )
        return module

    monkeypatch.setattr(activation, "import_module", load)
    with pytest.raises(ExtensionCommitError):
        prepare()
    assert activation._failure


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires CPU fork")
@pytest.mark.parametrize("operation", ["create", "load", "postload", "apply"])
def test_fork_cached_method_never_bypasses_owner_health(vendor, operation):
    import torch
    from tests.vllm033.test_slimquant_packed import storage, load_all

    method, layer, _ = storage()
    load_all(layer)
    method.process_weights_after_loading(layer)
    method.prepare()
    pid = os.fork()
    if pid == 0:
        signal.alarm(3)
        try:
            if operation == "create":
                method.create_weights(
                    torch.nn.Module(),
                    2,
                    8,
                    4,
                    torch.bfloat16,
                    weight_loader=lambda *a, **k: None,
                )
            elif operation == "load":
                p = layer.w13_weight
                p.weight_loader(
                    p, torch.ones((4, 4), dtype=torch.int8), "w13_weight", "w1", 0, True
                )
            elif operation == "postload":
                method.process_weights_after_loading(layer)
            else:
                method.apply(
                    layer,
                    torch.ones((1, 8), dtype=torch.bfloat16),
                    torch.ones((1, 1)),
                    torch.zeros((1, 1), dtype=torch.int32),
                )
        except ExtensionCommitError:
            os._exit(0)
        except BaseException:
            os._exit(2)
        os._exit(3)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    activation.check_health()
    method.process_weights_after_loading(layer)
