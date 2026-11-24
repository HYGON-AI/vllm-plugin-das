from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "adapter_extension"
SOURCE_ROOT = PROJECT_ROOT / "src"

# The single expected extension table. Both the source metadata and the built
# wheel are checked against this, so the two cannot drift apart.
EXPECTED_ENTRY_POINTS = {
    "hcu_adapter.extensions": {
        "minimax31_slimquant": "hcu_vllm_das.quantization.minimax31_activation:extension_spec",
        "minimax31_pcp_ep": "hcu_vllm_das.parallel.minimax31_pcp_activation:extension_spec",
        "das_hcu": "hcu_vllm_das.extension:extension_spec",
        "das_hcu_flash": "hcu_vllm_das.flash_activation:extension_spec",
        "das_hcu_slimquant": "hcu_vllm_das.slimquant_activation:extension_spec",
        "das_hcu_aiter_int8": "hcu_vllm_das.int8_activation:extension_spec",
        "das_hcu_dsv41": "hcu_vllm_das.dsv41_activation:extension_spec",
        "das_hcu_ct_w4a8": "hcu_vllm_das.ct_w4a8_activation:extension_spec",
        "das_hcu_v41_indexer": "hcu_vllm_das.v41_indexer_activation:extension_spec",
        "das_hcu_v41_pcp": "hcu_vllm_das.v41_pcp_activation:extension_spec",
        "das_hcu_fp4_packer": "hcu_vllm_das.fp4_packer:extension_spec",
        "das_hcu_step5_config": "hcu_vllm_das.step5_activation:extension_spec",
        "das_hcu_hyv4_flash_config": "hcu_vllm_das.hyv4_flash_config_activation:extension_spec",
        "das_hcu_hy4_frontend": "hcu_vllm_das.hy4_frontend:extension_spec",
        "das_hcu_hy4_ct_w4a8": "hcu_vllm_das.hy4_quant_activation:extension_spec",
        "das_hcu_hy4_static": "hcu_vllm_das.hy4_static_activation:extension_spec",
        "das_hcu_wo_a_token_major": "hcu_vllm_das.wo_a_token_major:extension_spec",
    }
}


def _parse_entry_points(text: str) -> dict[str, dict[str, str]]:
    """Parse an entry_points.txt body into {group: {name: value}}."""
    parsed: dict[str, dict[str, str]] = {}
    group = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            group = line[1:-1]
            parsed.setdefault(group, {})
            continue
        if group is None or "=" not in line:
            continue
        name, value = line.split("=", 1)
        parsed[group][name.strip()] = value.strip()
    return parsed


def test_project_metadata_has_only_the_adapter_extension_entry_point() -> None:
    metadata = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert metadata["name"] == "hcu-vllm-das-plugin"
    assert metadata["version"] == "0.33.0"
    assert metadata["requires-python"] == ">=3.11,<3.15"
    assert metadata["dependencies"] == [
        "hcu-adapter>=0.5.22",
        "packaging>=23",
    ]
    assert metadata["entry-points"] == EXPECTED_ENTRY_POINTS
    assert "vllm.platform_plugins" not in metadata["entry-points"]
    assert "vllm.general_plugins" not in metadata["entry-points"]


def test_package_exports_the_exact_vllm_target_without_framework_imports() -> None:
    code = f"""
import sys
sys.path.insert(0, {os.fspath(SOURCE_ROOT)!r})
blocked = {{"torch", "vllm", "lightop", "vllm_hcu"}}
before = set(sys.modules)
import hcu_vllm_das
assert hcu_vllm_das.__version__ == "0.33.0"
assert hcu_vllm_das.TARGET_VLLM_RELEASE == (0, 33, 0)
assert hcu_vllm_das.VLLM_UPSTREAM_COMMIT == "4ba787ec9959e566d3b587a796dff07673324da5"
introduced = (set(sys.modules) - before).intersection(blocked)
assert not introduced, introduced
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
        prefill_kernel = "hcu_vllm_das/parallel/minimax31_prefill_triton.py"
        assert prefill_kernel in names
        installed = tmp_path / "installed"
        archive.extractall(installed)
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        entry_points_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        wheel_metadata = archive.read(metadata_name).decode("utf-8")
        entry_points = archive.read(entry_points_name).decode("utf-8")

    metadata_lines = wheel_metadata.splitlines()
    entry_point_lines = entry_points.splitlines()
    assert "Name: hcu-vllm-das-plugin" in metadata_lines
    assert "Version: 0.33.0" in metadata_lines
    assert "Requires-Dist: hcu-adapter>=0.5.22" in metadata_lines
    assert "[hcu_adapter.extensions]" in entry_point_lines
    # Parse the wheel's own entry points and require the full expected table,
    # not two spot checks: a build backend that drops or rewrites one extension
    # entry point would otherwise still pass while the installed plugin silently
    # loses an extension the adapter host looks for.
    wheel_entry_points = _parse_entry_points(entry_points)
    assert wheel_entry_points == {
        "hcu_adapter.extensions": {
            name: value
            for name, value in EXPECTED_ENTRY_POINTS["hcu_adapter.extensions"].items()
        }
    }
    assert "hcu_vllm_das/dsv41_activation.py" in names
    assert "hcu_vllm_das/hyv4_flash_config_activation.py" in names
    assert "hcu_vllm_das/models/hy_v4_flash/configs.py" in names
    assert "hcu_vllm_das/models/hy_v4_flash/vision_config.py" in names
    assert "hcu_vllm_das/quantization/dsv41_channel_fp8.py" in names
    assert "[vllm.platform_plugins]" not in entry_point_lines
    assert "[vllm.general_plugins]" not in entry_point_lines
    assert any(name == "hcu_vllm_das/__init__.py" for name in names)
    assert "hcu_vllm_das/attention/__init__.py" in names
    assert "hcu_vllm_das/models/dsv41_pcp.py" in names
    assert "hcu_vllm_das/dsv41/pcp_attention.py" in names
    assert "hcu_vllm_das/dsv41/worker.py" in names
    assert "hcu_vllm_das/quantization/dsv41_channel_fp8.py" in names
    assert "hcu_vllm_das/attention/flash_attn.py" in names
    assert "hcu_vllm_das/moe/__init__.py" in names
    assert "hcu_vllm_das/moe/fused_moe_aiter.py" in names
    assert "hcu_vllm_das/moe/int8_oracle.py" in names
    assert "hcu_vllm_das/silu_and_mul.py" not in names
    license_names = {
        name.rsplit("/", 1)[-1]
        for name in names
        if ".dist-info/licenses/" in name
    }
    assert license_names == {"LICENSE", "NOTICE"}
    assert not any(name.startswith("vllm_hcu/") for name in names)
    assert not any(name.startswith("tests/") for name in names)

    # The source CI lane has no Torch/Triton. Use import-only stand-ins here;
    # real Torch/Triton imports are checked separately in the HCU container.
    code = f"""
import importlib, importlib.abc, importlib.metadata, sys, types
from pathlib import Path
installed = Path({str(installed)!r})
sys.path.insert(0, str(installed))
class NoRepositoryModules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'triton_tmp_dir', 'vllm_hcu'}}:
            raise AssertionError('repository-root dependency: ' + fullname)
sys.meta_path.insert(0, NoRepositoryModules())
sys.modules['torch'] = types.ModuleType('torch')
triton = types.ModuleType('triton'); triton.__path__ = []
triton.jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
language = types.ModuleType('triton.language'); language.constexpr = object
triton.language = language
sys.modules['triton'] = triton; sys.modules['triton.language'] = language
module = importlib.import_module('hcu_vllm_das.parallel.minimax31_prefill_triton')
origin = Path(module.__file__).resolve()
assert origin == (installed / 'hcu_vllm_das/parallel/minimax31_prefill_triton.py').resolve()
record = importlib.metadata.distribution('hcu-vllm-das-plugin')
assert any(Path(record.locate_file(f)).resolve() == origin for f in record.files)
for name in ['force_local_pages','quant_q','store_pcp_cache','check_scales','select_physical_dynamic']:
    assert callable(getattr(module,name)), name
"""
    result = subprocess.run([sys.executable, '-I', '-S', '-c', code],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("frontend", ["setup", "pip"])
def test_root_build_refuses_legacy_package_before_loading_torch(
    frontend: str, tmp_path: Path
) -> None:
    if frontend == "setup":
        command = [sys.executable, "-I", "-S", "setup.py", "--version"]
    else:
        command = [
            sys.executable, "-m", "pip", "wheel", "--no-deps",
            "--no-build-isolation", "--wheel-dir", str(tmp_path), ".",
        ]
    result = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True,
        check=False, timeout=120,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "legacy vllm_hcu build is disabled" in output, output
    assert "./adapter_extension" in output, output
    assert "ModuleNotFoundError" not in output, output
    assert not list(tmp_path.glob("*.whl"))
