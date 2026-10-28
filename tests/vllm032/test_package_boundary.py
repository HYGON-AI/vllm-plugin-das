from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tomllib
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "adapter_extension"
SOURCE_ROOT = PROJECT_ROOT / "src"


def test_project_metadata_has_only_the_adapter_extension_entry_point() -> None:
    metadata = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert metadata["name"] == "hcu-vllm-das-plugin"
    assert metadata["version"] == "0.2.0.dev0"
    assert metadata["requires-python"] == ">=3.11,<3.15"
    assert metadata["dependencies"] == [
        "sgl-kernel>=0.1.0",
        "packaging>=23",
    ]
    assert metadata["entry-points"] == {
        "hcu_adapter.extensions": {
            "das_hcu": "hcu_vllm_das.extension:extension_spec"
        }
    }
    assert "vllm.platform_plugins" not in metadata["entry-points"]
    assert "vllm.general_plugins" not in metadata["entry-points"]


def test_package_exports_the_exact_vllm_target_without_framework_imports() -> None:
    code = f"""
import sys
sys.path.insert(0, {os.fspath(SOURCE_ROOT)!r})
import hcu_vllm_das
assert hcu_vllm_das.__version__ == "0.2.0.dev0"
assert hcu_vllm_das.TARGET_VLLM_RELEASE == (0, 32, 0)
assert hcu_vllm_das.VLLM_UPSTREAM_COMMIT == "5df63a7203cdb9328d824a27b682e77c7b9928be"
blocked = {{"torch", "vllm", "lightop", "vllm_hcu"}}
assert blocked.isdisjoint(sys.modules), blocked.intersection(sys.modules)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_built_wheel_contains_only_the_new_plugin_namespace(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            os.fspath(wheel_dir),
            os.fspath(PROJECT_ROOT),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    wheels = tuple(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    with ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        entry_points_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        wheel_metadata = archive.read(metadata_name).decode("utf-8")
        entry_points = archive.read(entry_points_name).decode("utf-8")

    assert "Name: hcu-vllm-das-plugin\n" in wheel_metadata
    assert "Version: 0.2.0.dev0\n" in wheel_metadata
    assert "Requires-Dist: sgl-kernel>=0.1.0\n" in wheel_metadata
    assert "[hcu_adapter.extensions]\n" in entry_points
    assert "das_hcu = hcu_vllm_das.extension:extension_spec\n" in entry_points
    assert "[vllm.platform_plugins]\n" not in entry_points
    assert "[vllm.general_plugins]\n" not in entry_points
    assert any(name == "hcu_vllm_das/__init__.py" for name in names)
    license_names = {
        name.rsplit("/", 1)[-1]
        for name in names
        if ".dist-info/licenses/" in name
    }
    assert license_names == {"LICENSE", "NOTICE"}
    assert not any(name.startswith("vllm_hcu/") for name in names)
    assert not any(name.startswith("tests/") for name in names)
