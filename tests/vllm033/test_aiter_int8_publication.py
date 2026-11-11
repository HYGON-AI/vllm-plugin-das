"""INT8 owner publication/fork contracts through real public host/Core APIs.

CPU fault injection replaces hardware discovery and native acquisition only.
The actual owner, Core registry and Adapter discovery/health sink run in fresh
processes. These tests do not execute vendor kernels or certify device support.
"""
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from threading import Barrier
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

TAG = "INT8_PUBLICATION_RESULT="


def _fork_check(action):
    """Bound the real owned child, including failed assertions and lock stalls."""
    pid = os.fork()
    if pid == 0:
        try:
            action()
        except BaseException:
            os._exit(2)
        os._exit(0)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        found, status = os.waitpid(pid, os.WNOHANG)
        if found:
            assert os.waitstatus_to_exitcode(status) == 0, "owned fork contract failed"
            return
        time.sleep(0.01)
    os.kill(pid, signal.SIGKILL)  # Our unreaped child cannot have its PID reused.
    os.waitpid(pid, 0)
    raise AssertionError("owned fork did not finish before the deadline")


def _scenario(case, error_kind="runtime"):
    import torch
    from hcu_adapter.bridges import vllm as host
    from hcu_adapter.extensions import (
        activate_extensions, ExtensionCommitError, ExtensionContext,
        ExtensionSpec, ExtensionStatus,
    )
    from hcu_vllm_das import int8_activation as owner
    from hcu_vllm_das.moe.int8_identity import FrozenFiles
    from vllm.model_executor.layers.fused_moe.oracle import int8_backend_registry as registry

    calls, rendered = [], []
    parent_pid = os.getpid()

    class Unprintable(BaseException):
        def __str__(self):
            rendered.append("unsafe rendering")
            raise KeyboardInterrupt("error rendering must not precede the latch")

    failure = {"runtime": RuntimeError, "interrupt": KeyboardInterrupt,
               "unprintable": Unprintable}[error_kind]("injected boundary failure")
    phase = "register" if case == "fork-after-failure" else case
    point = next(p for p in distribution("hcu-vllm-das-plugin").entry_points
                 if p.name == "das_hcu_aiter_int8")
    later = SimpleNamespace(name="later-owner", load=lambda: lambda: ExtensionSpec(
        "later-owner", 1, 1, {"test-later-owner"}, lambda _: calls.append("later")))
    context = ExtensionContext("0.5.22", "0.33.0",
                               requested_capabilities={owner.CAPABILITY})
    assert registry.get_int8_moe_backend("aiter") is None

    def discover():
        reports = activate_extensions(context, [later, point],
            health_check_sink=host._RUNTIME_ACTIONS.record_extension_health)
        host._RUNTIME_ACTIONS.record_extension_reports(reports)
        return reports

    def refused_in_child():
        assert registry.get_int8_moe_backend("aiter") is None
        for action in (lambda: owner.activate(context), owner.check_health,
                       owner.compile_factor):
            with pytest.raises(ExtensionCommitError):
                action()
        assert registry.get_int8_moe_backend("aiter") is None

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU publication test attempted GPU/native execution")

    with tempfile.TemporaryDirectory() as folder:
        native_file = Path(folder) / "controlled-native-identity"
        native_file.write_bytes(b"CPU fixture, never loaded as a native library")
        frozen = FrozenFiles.capture({"native": native_file})
        chip = ModuleType("aiter.jit.utils.chip_info")
        chip.get_gfx = lambda: "gfx938"
        asm_dir = folder + "/"
        original_register = registry.register_int8_moe_backend

        def prepare(*args):
            calls.append("prepare")
            if phase == "prepare":
                raise failure
            if case == "fork-during-prepare" and os.getpid() == parent_pid:
                _fork_check(refused_in_child)
                calls.append("inflight-child-refused")
            if case == "concurrent":
                time.sleep(0.05)  # Let the other caller reach the owner lock.
            return frozen, ((chip, "get_gfx", chip.get_gfx),), chip, os.fsencode(asm_dir)

        def register(*args, **kwargs):
            calls.append("register")
            result = original_register(*args, **kwargs)
            if phase == "register":
                raise failure  # The genuine public registry has already written.
            return result

        def validate_paths():
            if phase == "health":
                raise failure
            return original_validate_paths()

        original_validate_paths = frozen.validate_paths
        with patch.dict(os.environ, {"AITER_ASM_DIR": asm_dir}), \
             patch.dict(sys.modules, {"aiter": ModuleType("aiter"),
                 "aiter.moe": ModuleType("aiter.moe"), chip.__name__: chip}), \
             patch.object(host, "detect_hcu", return_value=host.HcuDetection(True, "gfx938", "")), \
             patch.object(owner, "_prepare_identity", side_effect=prepare), \
             patch.object(registry, "register_int8_moe_backend", side_effect=register), \
             patch.object(FrozenFiles, "validate_paths", side_effect=validate_paths), \
             patch.object(torch.cuda, "_lazy_init", forbidden), \
             patch.object(torch.cuda, "current_device", forbidden), \
             patch.object(torch.cuda, "get_device_properties", forbidden):
            if phase in {"prepare", "register", "health"}:
                with pytest.raises(ExtensionCommitError) as rejected:
                    discover()
                assert len(rejected.value.reports) == 1
                assert rejected.value.reports[0].status is ExtensionStatus.FAILED
                assert "later" not in calls and owner._failure and not owner._publishing
                count = calls.count("register")
                assert count == (phase != "prepare")
                assert (registry.get_int8_moe_backend("aiter") is not None) == (phase != "prepare")
                prepared = calls.count("prepare")
                phase = "cleared"  # Refusal must survive removal of its cause.
                # Direct retries, public discovery, read-only health and cache
                # factor must all refuse without registering again.
                for action in (discover, lambda: owner.activate(context),
                               owner.check_health, owner.compile_factor):
                    with pytest.raises(ExtensionCommitError):
                        action()
                assert calls.count("register") == count and "later" not in calls
                assert calls.count("prepare") == prepared
                assert not rendered
                if case == "fork-after-failure":
                    _fork_check(refused_in_child)
            else:
                if case == "concurrent":
                    # Discovery/report publication is a host responsibility;
                    # concurrently exercise only the owner's idempotent API.
                    start = Barrier(2)

                    def concurrent_activate():
                        start.wait(timeout=20)
                        owner.activate(context)

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        jobs = [executor.submit(concurrent_activate) for _ in range(2)]
                        for job in jobs:
                            job.result(timeout=20)
                reports = discover()
                assert all(r.status is ExtensionStatus.ACTIVE for r in reports)
                assert {r.entry_point_name for r in reports} == {point.name, later.name}
                assert calls.count("later") == 1
                assert calls.count("register") == 1 and calls.count("prepare") == 1
                descriptor = registry.get_int8_moe_backend("aiter")
                factor = owner.compile_factor()
                owner.activate(context)
                assert owner.compile_factor() == factor and calls.count("register") == 1
                if case == "fork-after-active":
                    def activate_child():
                        assert not host.extension_reports()
                        assert registry.get_int8_moe_backend("aiter") is None
                        with pytest.raises(RuntimeError, match="not active"):
                            owner.check_health()
                        with pytest.raises(ValueError, match="not active"):
                            registry.require_current_int8_moe_backend(descriptor)
                        child_reports = discover()
                        assert all(r.status is ExtensionStatus.ACTIVE for r in child_reports)
                        assert registry.get_int8_moe_backend("aiter") is not descriptor
                        assert owner.compile_factor() == factor
                    _fork_check(activate_child)
                    assert registry.get_int8_moe_backend("aiter") is descriptor
                    assert host.extension_reports() == reports
                    owner.check_health()
                if case == "fork-during-prepare":
                    assert "inflight-child-refused" in calls
            assert not torch.cuda.is_initialized()
    return dict(case=case, error=error_kind, register_calls=calls.count("register"),
                error_render_calls=len(rendered), gpu_initialized=False)


def _run(case, error_kind="runtime"):
    import hcu_vllm_das
    package_root = str(Path(hcu_vllm_das.__file__).resolve().parent.parent)
    env = dict(os.environ, HCU_ADAPTER_HOST="none", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", HIP_VISIBLE_DEVICES="",
        CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
        VLLM_ROCM_USE_AITER="0", VLLM_ROCM_USE_AITER_MOE="0",
        VLLM_ROCM_USE_AITER_CUSTOM_AR="0")
    env.pop("PYTHONPATH", None)
    code = ("import sys,runpy,json;sys.path.insert(0,sys.argv[1]);"
            "d=runpy.run_path(sys.argv[2]);print(" + repr(TAG)
            + "+json.dumps(d['_scenario'](*sys.argv[3:])))")
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code,
        package_root, __file__, case, error_kind], env=env,
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line[len(TAG):]) for line in result.stdout.splitlines()
            if line.startswith(TAG)]
    assert len(rows) == 1 and rows[0]["error_render_calls"] == 0
    assert rows[0]["gpu_initialized"] is False
    return rows[0]


@pytest.mark.parametrize("phase", ["prepare", "register", "health"])
@pytest.mark.parametrize("error_kind", ["runtime", "interrupt", "unprintable"])
def test_publication_failure_stops_later_owner_and_all_retries(phase, error_kind):
    assert _run(phase, error_kind)["register_calls"] == (phase != "prepare")


@pytest.mark.parametrize("case", ["normal", "concurrent", "fork-during-prepare",
                                  "fork-after-active", "fork-after-failure"])
def test_idempotence_and_actual_process_boundaries(case):
    if case.startswith("fork-") and not hasattr(os, "fork"):
        pytest.skip("requires real CPU fork")
    assert _run(case)["register_calls"] == 1
