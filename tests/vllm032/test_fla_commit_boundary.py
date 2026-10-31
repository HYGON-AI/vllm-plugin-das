"""CPU publication failure contracts; mounted markers are not native FLA proof."""

import os
import select
import signal
from types import SimpleNamespace as NS

import pytest

from hcu_adapter import extensions as host
from hcu_adapter.bridges import vllm as bridge
from hcu_vllm_das import extension
from test_extension import context, successful_importer


@pytest.fixture
def activation(monkeypatch):
    monkeypatch.setattr(extension, "_activated", False)
    monkeypatch.setattr(extension, "_commit_failure", None)
    monkeypatch.setattr(extension, "_committing_error_type", None)
    imports = []
    importer = successful_importer(imports)
    importer.fla.requested = True
    monkeypatch.setattr(extension, "_import_module", importer)
    return importer, imports


def point(name, priority, callback):
    return NS(name=name, load=lambda: lambda: host.ExtensionSpec(
        name=name, api_version=1, priority=priority, capabilities=(), activate=callback))


@pytest.mark.parametrize("stage", ["factor-conflict", "missing-oracle", "missing-fatal-abi", "missing-vendor"])
def test_prepare_failure_keeps_optional_baseline_and_no_publication(activation, stage, monkeypatch):
    importer, imports = activation
    envs = importer.modules["vllm.envs"]
    if stage == "factor-conflict":
        envs.environment_variables["VLLM_HCU_DAS_RMS_NORM_ENABLED"] = object()
    elif stage == "missing-oracle":
        importer.int8_oracle.install = None
    elif stage == "missing-fatal-abi":
        importer.modules["hcu_adapter.extensions"] = NS()
    else:
        importer.fla.install_result = False
    before = dict(envs.environment_variables)
    later = []
    # Missing requested FLA is clean at prepare; requiring it fails via the
    # host's existing required policy, without inventing a dirty publication.
    ctx = context(required_fla=stage == "missing-vendor")
    if stage == "missing-vendor":
        with pytest.raises(host.RequiredExtensionError):
            host.activate_extensions(ctx, [point("das-hcu", 10, extension.activate_extension)])
    else:
        reports = host.activate_extensions(ctx, [
            point("das-hcu", 10, extension.activate_extension),
            point("later", 0, lambda _: later.append(True))])
        assert [row.status.value for row in reports] == ["failed", "active"]
        assert later == [True]
    assert envs.environment_variables == before
    assert not importer.fla.mounted and not extension._activated
    assert extension._commit_failure is None
    assert "hcu_vllm_das.rms_norm" not in imports


@pytest.mark.parametrize("failure", [RuntimeError("late RMS failure"), SystemExit("late exit")])
def test_late_failure_stops_host_and_direct_retry_with_state_retained(activation, failure, monkeypatch):
    importer, imports = activation
    original = extension._import_module
    later = []

    def load(name):
        if name == "hcu_vllm_das.rms_norm":
            raise failure
        return original(name)

    monkeypatch.setattr(extension, "_import_module", load)
    with pytest.raises(host.ExtensionCommitError) as caught:
        host.activate_extensions(context(), [
            point("das-hcu", 10, extension.activate_extension),
            point("later", 0, lambda _: later.append(True))])
    assert not later
    assert [r.status.value for r in caught.value.reports] == ["failed"]
    assert importer.fla.mounted and not extension._activated
    assert importer.modules["vllm.envs"].environment_variables
    # Fixing the injected failure cannot heal the same process.
    monkeypatch.setattr(extension, "_import_module", original)
    before = list(imports)
    with pytest.raises(host.ExtensionCommitError, match="after publication"):
        extension.activate_extension(context())
    assert imports == before


def test_actual_runtime_latches_failed_optional_bundle_and_preserves_report(activation, monkeypatch):
    importer, _ = activation
    monkeypatch.setattr(importer.int8_oracle, "install", lambda: (_ for _ in ()).throw(RuntimeError("late oracle")))
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())
    monkeypatch.setattr(bridge, "detect_hcu", lambda: bridge.HcuDetection(True, "gfx936", "CPU double"))
    monkeypatch.setattr(bridge, "_ensure_vllm_compatible", lambda: "0.32.0")
    monkeypatch.setattr(bridge, "_ensure_hcu_platform_selected", lambda: None)
    monkeypatch.setattr(bridge, "configure_vllm_hip_runtime_path", lambda: None)
    monkeypatch.setattr(bridge, "_installed_adapter_version", lambda: "cpu")
    monkeypatch.setattr(host, "_entry_points", lambda: [point("das-hcu", 10, extension.activate_extension)])
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        bridge.hcu_runtime_plugin()
    assert [r.status.value for r in bridge.extension_reports()] == ["failed"]
    assert importer.fla.mounted
    with pytest.raises(bridge.LatchedVllmRuntimeError):
        bridge.require_runtime_healthy()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires actual fork")
def test_failed_owner_remains_failed_after_actual_fork(activation, monkeypatch):
    importer, _ = activation
    monkeypatch.setattr(importer.int8_oracle, "install", lambda: (_ for _ in ()).throw(RuntimeError("late oracle")))
    with pytest.raises(host.ExtensionCommitError):
        extension.activate_extension(context())
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        signal.alarm(4)
        try:
            extension.activate_extension(context())
        except host.ExtensionCommitError:
            os.write(write_fd, b"refused")
            os._exit(0)
        os._exit(1)
    os.close(write_fd)
    try:
        ready, _, _ = select.select([read_fd], [], [], 6)
        assert ready and os.read(read_fd, 32) == b"refused"
    finally:
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
