"""Admission is established before Core constructs any communication manager."""
from pathlib import Path
import importlib
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "adapter_extension/src"))
from hcu_vllm_das.parallel.runtime_sp_policy import custom_sp_requested


@pytest.mark.parametrize("extra,expected", [
    (None, False), ({}, False), ({"hcu": {}}, False),
    ({"hcu": {"enable_custom_sp": False}}, False),
    ({"hcu": {"enable_custom_sp": True}}, True),
])
def test_existing_sidecar_policy(extra, expected):
    assert custom_sp_requested(SimpleNamespace(additional_config=extra)) is expected
    assert custom_sp_requested(None) is False


@pytest.mark.parametrize("extra", [[], {"hcu": None}, {"hcu": []},
    {"hcu": {"enable_custom_sp": 1}}, {"hcu": {"enable_custom_sp": "false"}}])
def test_invalid_sidecar_cannot_silently_change_communication(extra):
    with pytest.raises(TypeError):
        custom_sp_requested(SimpleNamespace(additional_config=extra))


@pytest.fixture
def core(monkeypatch):
    calls = []
    config = SimpleNamespace(additional_config={"hcu": {"enable_custom_sp": True}})
    cfg = ModuleType("vllm.config")
    cfg.get_current_vllm_config_or_none = lambda: config

    class CudaCommunicator:
        def __init__(self, cpu_group, device=None, device_group=None, unique_name="",
                     global_ranks=None, global_world_size=None, tcp_store_group=None,
                     use_all2all=False):
            calls.append(locals().copy())
            self.world_size = 2
            self.device_group = device_group

    cuda = ModuleType("vllm.distributed.device_communicators.cuda_communicator")
    cuda.CudaCommunicator = CudaCommunicator
    monkeypatch.setitem(sys.modules, cfg.__name__, cfg)
    monkeypatch.setitem(sys.modules, cuda.__name__, cuda)
    name = "hcu_vllm_das.parallel.runtime_sp_communicator"
    old = sys.modules.pop(name, None)
    module = importlib.import_module(name)
    yield SimpleNamespace(module=module, calls=calls, config=config, cuda=cuda)
    sys.modules.pop(name, None)
    if old is not None:
        sys.modules[name] = old


@pytest.mark.parametrize("group", ["ep:0", "ep:stateless", "tp:0", "etp:0", "pp:0"])
@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("upstream_admission", [False, True])
def test_constructor_preserves_all_arguments_and_admits_only_requested_ep(
    core, group, requested, upstream_admission,
):
    core.config.additional_config["hcu"]["enable_custom_sp"] = requested
    cpu, device, device_group, ranks, store = (object() for _ in range(5))
    core.module.RuntimeSPCommunicator(cpu, device, device_group, group, ranks,
                                     19, store, upstream_admission)
    actual = core.calls[-1]
    assert actual["cpu_group"] is cpu
    assert actual["device"] is device
    assert actual["device_group"] is device_group
    assert actual["global_ranks"] is ranks
    assert actual["global_world_size"] == 19
    assert actual["tcp_store_group"] is store
    assert actual["unique_name"] == group
    assert actual["use_all2all"] is (
        upstream_admission or (requested and group.split(":")[0] == "ep")
    )


def test_exchange_uses_owned_group_and_returns_original_output(core, monkeypatch):
    group = object()
    comm = core.module.RuntimeSPCommunicator(object(), device_group=group)
    x, out = torch.arange(8), torch.empty(8, dtype=torch.int64)
    calls = []

    def exchange(output, input, *, group):
        calls.append((output, input, group))
        output.copy_(input.flip(0))

    monkeypatch.setattr(torch.distributed, "all_to_all_single", exchange)
    assert comm.all_to_all_single(out, x) is out
    assert torch.equal(out, x.flip(0))
    assert len(calls) == 1
    assert calls[0][0] is out and calls[0][1] is x and calls[0][2] is group
    comm.device_group = None
    with pytest.raises(RuntimeError, match="device process group"):
        comm.all_to_all_single(out, x)
    assert len(calls) == 1  # Never accidentally exchange on the world group.
    comm.world_size = 1
    assert comm.all_to_all_single(out, x) is out
    assert torch.equal(out, x)
    assert len(calls) == 1


def test_discovery_imports_no_framework():
    roots = [str(Path(__file__).resolve().parents[2] / "adapter_extension/src"),
             *sys.path]
    code = f'''
import sys, importlib.abc
sys.path[:0] = {roots!r}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'vllm', 'torch', 'aiter', 'lightop', 'vllm_hcu'}}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from hcu_vllm_das.parallel.runtime_sp_activation import extension_spec
assert extension_spec().capabilities == frozenset({{'runtime-sp-communication'}})
assert not set(sys.modules).intersection({{'vllm', 'torch', 'aiter', 'lightop', 'vllm_hcu'}})
'''
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def activation(core, monkeypatch):
    from hcu_adapter.bridges import vllm as bridge
    from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle
    from hcu_adapter.extensions import ExtensionCommitError
    from hcu_vllm_das.parallel import runtime_sp_activation as module

    class Platform:
        _device_communicator_selector = None

        @classmethod
        def register_device_communicator_selector(cls, selector):
            if cls._device_communicator_selector is not None:
                raise RuntimeError("competing owner")
            cls._device_communicator_selector = staticmethod(selector)

        @classmethod
        def owns_device_communicator_selector(cls, selector):
            return cls._device_communicator_selector is selector

    monkeypatch.setitem(bridge.__dict__, "HcuPlatform", Platform)
    monkeypatch.setattr(bridge, "require_extension_healthy", lambda capability: (
        module.check_health() if capability == module.CAPABILITY
        else pytest.fail("wrong required owner")
    ))
    monkeypatch.setattr(module, "_platform", None)
    monkeypatch.setattr(module, "_lifecycle", ProcessLocalLifecycle(
        runtime_action=module._ACTION, error_type=ExtensionCommitError))
    return SimpleNamespace(module=module, platform=Platform, core=core)


def _context(module, **overrides):
    from hcu_adapter.extensions import ExtensionContext
    args = dict(adapter_version="0.5.22", vllm_version="0.33.0",
                requested_capabilities=frozenset({module.CAPABILITY}))
    args.update(overrides)
    return ExtensionContext(**args)


def test_publication_is_explicit_idempotent_and_detects_owner_drift(activation):
    a = activation
    with pytest.raises(RuntimeError, match="explicit request"):
        a.module.activate(_context(a.module, requested_capabilities=frozenset()))
    assert a.platform._device_communicator_selector is None
    a.module.activate(_context(a.module))
    a.module.activate(_context(a.module))
    assert a.platform._device_communicator_selector is a.module.select_communicator
    assert a.module.select_communicator("official.Class") == a.module._PATH
    a.core.config.additional_config["hcu"]["enable_custom_sp"] = False
    assert a.module.select_communicator("official.Class") == "official.Class"
    a.platform._device_communicator_selector = lambda default: default
    with pytest.raises(RuntimeError, match="ownership changed"):
        a.module.check_health()


def test_constructor_drift_refuses_before_selector_publication(activation):
    a = activation
    a.core.cuda.CudaCommunicator.__init__ = lambda self, cpu_group: None
    with pytest.raises(RuntimeError, match="constructor ABI changed"):
        a.module.activate(_context(a.module))
    assert a.platform._device_communicator_selector is None


def test_failed_registration_is_sticky_and_preserves_competing_owner(activation):
    a = activation
    other = lambda default: default
    a.platform._device_communicator_selector = other
    for _ in range(2):
        with pytest.raises(RuntimeError, match="publication failed"):
            a.module.activate(_context(a.module))
    assert a.platform._device_communicator_selector is other


def test_selector_requires_exact_host_report_and_latches_report_replacement(activation, monkeypatch):
    from hcu_adapter.bridges import vllm as bridge
    from hcu_adapter.extensions import ExtensionReport, ExtensionStatus
    from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle

    a = activation
    a.module.activate(_context(a.module))
    host = ProcessLocalLifecycle(runtime_action="test-host")
    monkeypatch.setattr(bridge, "require_extension_healthy", host.require_extension_healthy)
    report = ExtensionReport(entry_point_name=a.module.ENTRY,
        extension_name=a.module.ENTRY, status=ExtensionStatus.ACTIVE,
        capabilities=frozenset({a.module.CAPABILITY}), reason="test activation")
    host.record_extension_reports((report,))
    host.record_extension_health(report, a.module.check_health)
    assert a.module.select_communicator("official.Class") == a.module._PATH
    replacement = ExtensionReport(entry_point_name=a.module.ENTRY,
        extension_name=a.module.ENTRY, status=ExtensionStatus.ACTIVE,
        capabilities=frozenset({a.module.CAPABILITY}), reason="test activation")
    host.record_extension_reports((replacement,))
    with pytest.raises(RuntimeError, match="exact current-process health binding"):
        a.module.select_communicator("official.Class")
    host.record_extension_reports((report,))
    with pytest.raises(RuntimeError, match="exact current-process health binding"):
        a.module.select_communicator("official.Class")
