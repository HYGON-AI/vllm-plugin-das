# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Clean-interpreter smoke tests for the complete runtime patch inventory."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from typing import Any

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
_RESULT_PREFIX = "VLLM_HCU_BOOTSTRAP_RESULT="


def _resolve_target_vllm_root() -> Path:
    configured_root = os.environ.get("VLLM_SOURCE_ROOT")
    vllm_spec = importlib.util.find_spec("vllm")
    discovered_root = (
        Path(vllm_spec.origin).resolve().parents[1]
        if vllm_spec is not None and vllm_spec.origin is not None
        else None
    )
    installed_roots = tuple(
        Path(path)
        for key in ("platlib", "purelib")
        if (path := sysconfig.get_path(key))
    )
    candidates = (
        (Path(configured_root),)
        if configured_root is not None
        else (discovered_root, *installed_roots)
    )
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if (resolved / "vllm" / "__init__.py").is_file():
            return resolved
    rendered = ", ".join(str(path) for path in candidates if path is not None)
    raise RuntimeError(
        "no current vLLM source/install was found; checked: " + rendered
    )


TARGET_VLLM_ROOT = _resolve_target_vllm_root()


_TARGET_SOURCE_ASSERTION = r"""
import os as _vllm_hcu_os
from pathlib import Path as _VllmHcuPath
from packaging.version import Version as _VllmHcuVersion
import vllm as _vllm_hcu_target
from vllm_hcu.version import __vllm_target_version__ as _vllm_hcu_target_version
_vllm_hcu_root = _VllmHcuPath(
    _vllm_hcu_os.environ["VLLM_SOURCE_ROOT"]
).resolve()
_vllm_hcu_file = _VllmHcuPath(_vllm_hcu_target.__file__).resolve()
assert _vllm_hcu_file.is_relative_to(_vllm_hcu_root), (
    f"vllm resolved outside target root: "
    f"{_vllm_hcu_file} not under {_vllm_hcu_root}"
)
assert _VllmHcuVersion(_vllm_hcu_target.__version__).release[:3] == (0, 28, 1)
assert _VllmHcuVersion(_vllm_hcu_target_version).release[:3] == (0, 28, 1)
assert "VLLM_V0251_SOURCE_ROOT" not in _vllm_hcu_os.environ
"""

_PORTABLE_BOOTSTRAP = rf"""
import builtins
import json

_original_import = builtins.__import__

import vllm_hcu
from vllm_hcu.patch import (
    IMPORT_COORDINATOR,
    apply_worker_patches,
    patch_report,
)

assert vllm_hcu.hcu_platform_plugin() == (
    "vllm_hcu.platforms.hcu.HCUPlatform"
)
vllm_hcu.hcu_platform_register_model()
apply_worker_patches(None)

def snapshot():
    return {{
        item.patch_id: (
            item.module_name,
            item.action.value,
            item.status,
            item.feature_enabled,
        )
        for item in IMPORT_COORDINATOR.registrations()
    }}

first = snapshot()
first_report = patch_report()
first_registrations = IMPORT_COORDINATOR.registrations()

assert first_registrations
assert len(first) == len(first_registrations)
assert len(first) == len({{item.patch_id for item in first_registrations}})
target_modules = {{item.module_name for item in first_registrations}}
assert target_modules
replacement_count = sum(
    item.action.value == "replacement" for item in first_registrations
)
callback_count = sum(
    item.action.value == "callback" for item in first_registrations
)
assert replacement_count > 0
assert callback_count > 0
assert replacement_count + callback_count == len(first_registrations)
assert set(first_report["patches"]) - set(first) == {{
    "plugin.general.model_registry"
}}
assert first_report["patches"]["plugin.general.model_registry"]["status"] == (
    "applied"
)
assert first_report["process_role"] == "Worker"

bad = {{
    patch_id: value
    for patch_id, value in first.items()
    if value[2] in {{"failed", "applying"}}
}}
assert bad == {{}}, bad

# Re-enter every public startup boundary.  No registration, wrapper, or model
# callback may be duplicated or change ownership.
assert vllm_hcu.hcu_platform_plugin() == (
    "vllm_hcu.platforms.hcu.HCUPlatform"
)
vllm_hcu.hcu_platform_register_model()
apply_worker_patches(None)

second = snapshot()
assert second == first
assert builtins.__import__ is _original_import

print(
    "{_RESULT_PREFIX}"
    + json.dumps(
        {{
            "callbacks": callback_count,
            "failed": [],
            "process_role": first_report["process_role"],
            "registrations": len(first),
            "replacements": replacement_count,
            "target_modules": len(target_modules),
        }},
        sort_keys=True,
    )
)
"""

_HCU_TARGET_IMPORT_BOOTSTRAP = rf"""
import builtins
import importlib
import json

_original_import = builtins.__import__

from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins

# Exercise the installed vLLM entry points before calling the public plugin
# functions again for the idempotency check below.
load_general_plugins()
assert type(current_platform).__module__ == "vllm_hcu.platforms.hcu"
assert type(current_platform).__name__ == "HCUPlatform"

import vllm_hcu
from vllm_hcu.patch import (
    IMPORT_COORDINATOR,
    apply_worker_patches,
    patch_report,
)
from vllm_hcu.patch.worker import validate_worker_patches

assert vllm_hcu.hcu_platform_plugin() == (
    "vllm_hcu.platforms.hcu.HCUPlatform"
)
vllm_hcu.hcu_platform_register_model()
vllm_hcu.hcu_platform_register_ops()
apply_worker_patches(None)

registrations = IMPORT_COORDINATOR.registrations()
assert registrations
assert len(registrations) == len({{item.patch_id for item in registrations}})

# The complete inventory is armed atomically before this point.  Import each
# target enabled by the default worker profile once.  Disabled optional
# profiles such as DeepEP and custom-SP must remain lazy: forcing their target
# modules into a normal startup would incorrectly turn optional dependencies
# into requirements of the base HCU environment.
all_target_modules = tuple(dict.fromkeys(
    item.module_name for item in registrations
))
enabled_target_modules = tuple(dict.fromkeys(
    item.module_name for item in registrations if item.feature_enabled
))
assert all_target_modules
assert enabled_target_modules
assert set(enabled_target_modules).issubset(all_target_modules)
for module_name in enabled_target_modules:
    importlib.import_module(module_name)

validate_worker_patches(require_applied=True)

def snapshot():
    return {{
        item.patch_id: (
            item.module_name,
            item.action.value,
            item.status,
            item.feature_enabled,
        )
        for item in IMPORT_COORDINATOR.registrations()
    }}

first = snapshot()
bad = {{
    patch_id: value
    for patch_id, value in first.items()
    if value[2] in {{"failed", "applying"}}
}}
assert bad == {{}}, bad
enabled_nonterminal = {{
    patch_id: value
    for patch_id, value in first.items()
    if value[3] and value[2] not in {{"applied", "skipped"}}
}}
assert enabled_nonterminal == {{}}, enabled_nonterminal

report = patch_report()
failed = {{
    patch_id: value["failure_reason"]
    for patch_id, value in report["patches"].items()
    if value["status"] == "failed"
}}
assert failed == {{}}, failed
assert report["patches"]["plugin.general.model_registry"]["status"] == (
    "applied"
)
assert report["patches"]["plugin.general.ops_registry"]["status"] == "applied"

# Re-enter all plugin/dispatcher boundaries and re-import every enabled
# target.  The resulting inventory must remain byte-for-byte equivalent.
assert vllm_hcu.hcu_platform_plugin() == (
    "vllm_hcu.platforms.hcu.HCUPlatform"
)
vllm_hcu.hcu_platform_register_model()
vllm_hcu.hcu_platform_register_ops()
apply_worker_patches(None)
for module_name in enabled_target_modules:
    importlib.import_module(module_name)

second = snapshot()
assert second == first
assert builtins.__import__ is _original_import

status_counts = {{
    status: sum(value[2] == status for value in first.values())
    for status in ("applied", "skipped", "armed")
}}
print(
    "{_RESULT_PREFIX}"
    + json.dumps(
        {{
            "failed": [],
            "enabled_target_modules": len(enabled_target_modules),
            "process_role": report["process_role"],
            "registrations": len(first),
            "status_counts": status_counts,
            "target_modules": len(all_target_modules),
        }},
        sort_keys=True,
    )
)
"""


def _clean_environment(*, plugins: str) -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("VLLM_HCU_") or name in {
            "VLLM_USE_NN",
            "VLLM_USE_OPT_CAT",
        }:
            environment.pop(name)
    environment["VLLM_PLUGINS"] = plugins
    environment.pop("VLLM_V0251_SOURCE_ROOT", None)
    environment["VLLM_SOURCE_ROOT"] = str(TARGET_VLLM_ROOT)
    python_path = (
        str(TARGET_VLLM_ROOT),
        str(REPOSITORY),
        environment.get("PYTHONPATH", ""),
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in python_path if part
    )
    return environment


def _run_clean_python(
    body: str,
    *,
    plugins: str = "__disabled__",
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    if not (TARGET_VLLM_ROOT / "vllm" / "__init__.py").is_file():
        raise RuntimeError(
            "VLLM_SOURCE_ROOT does not contain the target vllm package: "
            f"{TARGET_VLLM_ROOT}"
        )
    return subprocess.run(
        [sys.executable, "-c", _TARGET_SOURCE_ASSERTION + body],
        check=False,
        capture_output=True,
        text=True,
        env=_clean_environment(plugins=plugins),
        timeout=timeout,
    )


def _result_payload(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    lines = (
        line.removeprefix(_RESULT_PREFIX)
        for line in result.stdout.splitlines()
        if line.startswith(_RESULT_PREFIX)
    )
    try:
        return json.loads(next(lines))
    except StopIteration as exc:
        raise AssertionError(
            "clean bootstrap subprocess did not emit its result payload\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        ) from exc


def _require_live_hcu() -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch is required for the HCU bootstrap smoke test")
    if not torch.cuda.is_available():
        pytest.skip("a live HCU/ROCm device is required")
    properties = torch.cuda.get_device_properties(0)
    if not hasattr(properties, "gcnArchName"):
        pytest.skip("the active torch device is not an HCU/ROCm device")


def _require_local_hcu_extension() -> None:
    if importlib.util.find_spec("vllm_hcu.hcu_ops") is not None:
        return
    from vllm_hcu.version import __version__ as source_version

    try:
        installed_version = importlib.metadata.version("vllm_hcu")
    except importlib.metadata.PackageNotFoundError:
        installed_version = "not installed"
    pytest.fail(
        "the full HCU bootstrap smoke requires the local vllm_hcu.hcu_ops "
        "extension, but no hcu_ops*.so is importable from this checkout; "
        f"source version={source_version}, installed distribution="
        f"{installed_version}. The test intentionally imports the current "
        "checkout before site-packages. Install this checkout with "
        "`MAX_JOBS=8 python -m pip install -e . --no-build-isolation` "
        "and rerun this test"
    )


def test_clean_process_arms_complete_patch_inventory_and_is_idempotent() -> None:
    result = _run_clean_python(_PORTABLE_BOOTSTRAP, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = _result_payload(result)
    assert payload["failed"] == []
    assert payload["process_role"] == "Worker"
    assert payload["registrations"] > 0
    assert payload["callbacks"] > 0
    assert payload["replacements"] > 0
    assert (
        payload["callbacks"] + payload["replacements"]
        == payload["registrations"]
    )
    assert 0 < payload["target_modules"] <= payload["registrations"]


@pytest.mark.hcu
def test_clean_hcu_process_imports_every_enabled_patch_target() -> None:
    _require_live_hcu()
    _require_local_hcu_extension()
    result = _run_clean_python(
        _HCU_TARGET_IMPORT_BOOTSTRAP,
        plugins="hcu,hcu_model,hcu_ops",
        timeout=360,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = _result_payload(result)
    assert payload["failed"] == []
    assert payload["process_role"] == "Worker"
    assert payload["registrations"] > 0
    assert 0 < payload["enabled_target_modules"] <= payload["target_modules"]
    assert payload["target_modules"] <= payload["registrations"]
    assert sum(payload["status_counts"].values()) == payload["registrations"]
