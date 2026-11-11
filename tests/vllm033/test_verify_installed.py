from __future__ import annotations

from enum import Enum
import json
import os
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace

import pytest


TOOLS_ROOT = (
    Path(__file__).resolve().parents[2] / "adapter_extension" / "tools"
)
sys.path.insert(0, os.fspath(TOOLS_ROOT))

import verify_installed


class FakeDistribution:
    def __init__(self, name: str, version: str, root: Path, files: list[str]):
        self.metadata = {"Name": name}
        self.version = version
        self.root = root
        self.files = [PurePosixPath(value) for value in files]
        self.record = "recorded-file,sha256=fixture,1\n"
        self.direct_url: str | None = None

    def locate_file(self, value: str | PurePosixPath) -> Path:
        return self.root / value

    def read_text(self, name: str) -> str | None:
        if name == "RECORD":
            return self.record
        if name == "direct_url.json":
            return self.direct_url
        return None


class FakeEntryPoint:
    def __init__(
        self,
        name: str,
        value: str,
        distribution: FakeDistribution,
    ) -> None:
        self.name = name
        self.value = value
        self.dist = distribution


class ActiveStatus:
    value = "active"


class FailedStatus:
    value = "failed"


class HcuRMSNorm:
    pass


HcuRMSNorm.__module__ = "hcu_vllm_das.rms_norm"


class TritonExperts:
    pass


TritonExperts.__module__ = "vllm.model_executor.layers.fused_moe.experts.triton_moe"


class Int8MoeBackend(Enum):
    TRITON = "TRITON"
    AITER = "AITER"


class HcuPlatform:
    pass


HcuPlatform.__module__ = "hcu_adapter.bridges.vllm"


def installed_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    site = tmp_path / "site-packages"
    distributions = {
        "vllm": FakeDistribution(
            "vllm",
            "0.33.0+rocm723",
            site,
            ["vllm/__init__.py", "vllm/model_executor/layers/fused_moe/oracle/int8.py", "vllm-0.33.0.dist-info/RECORD"],
        ),
        "sgl-kernel": FakeDistribution(
            "sgl-kernel",
            "0.1.0",
            site,
            [
                "hcu_adapter/extensions.py",
                "hcu_adapter/bridges/vllm.py",
                "sgl_kernel-0.1.0.dist-info/RECORD",
            ],
        ),
        "hcu-vllm-das-plugin": FakeDistribution(
            "hcu-vllm-das-plugin",
            "0.33.0",
            site,
            [
                "hcu_vllm_das/extension.py",
                "hcu_vllm_das/rms_norm.py",
                "hcu_vllm_das/moe/__init__.py",
                "hcu_vllm_das/moe/fused_moe_aiter.py",
                "hcu_vllm_das/moe/int8_oracle.py",
                "hcu_vllm_das_plugin-0.33.0.dist-info/RECORD",
            ],
        ),
    }
    entry_points = {
        "vllm.platform_plugins": [
            FakeEntryPoint(
                "hcu_adapter",
                "hcu_adapter.bridges.vllm:hcu_platform_plugin",
                distributions["sgl-kernel"],
            )
        ],
        "vllm.general_plugins": [
            FakeEntryPoint(
                "hcu_adapter",
                "hcu_adapter.bridges.vllm:hcu_runtime_plugin",
                distributions["sgl-kernel"],
            )
        ],
        "hcu_adapter.extensions": [
            FakeEntryPoint(
                "das_hcu",
                "hcu_vllm_das.extension:extension_spec",
                distributions["hcu-vllm-das-plugin"],
            )
        ],
    }
    reports = [
        SimpleNamespace(
            entry_point_name="das_hcu",
            extension_name="das-hcu",
            status=ActiveStatus(),
            capabilities=frozenset(
                {"lightop-rms-norm"}
            ),
            reason="",
        )
    ]

    class ExtensionContext:
        def __init__(self, *, adapter_version: str, vllm_version: str):
            self.adapter_version = adapter_version
            self.vllm_version = vllm_version

    def activate_extensions(context, points):
        assert context.adapter_version == (distributions.get("hcu-adapter") or distributions["sgl-kernel"]).version
        assert context.vllm_version == distributions["vllm"].version
        assert points == entry_points["hcu_adapter.extensions"]
        return tuple(reports)

    modules = {
        "vllm": SimpleNamespace(
            __file__=os.fspath(site / "vllm" / "__init__.py")
        ),
        "hcu_adapter.extensions": SimpleNamespace(
            __file__=os.fspath(site / "hcu_adapter" / "extensions.py"),
            ExtensionContext=ExtensionContext,
            activate_extensions=activate_extensions,
        ),
        "hcu_adapter.bridges.vllm": SimpleNamespace(
            __file__=os.fspath(site / "hcu_adapter" / "bridges" / "vllm.py")
        ),
        "hcu_vllm_das.extension": SimpleNamespace(
            __file__=os.fspath(site / "hcu_vllm_das" / "extension.py")
        ),
        "hcu_vllm_das.rms_norm": SimpleNamespace(
            __file__=os.fspath(site / "hcu_vllm_das" / "rms_norm.py")
        ),
        "vllm.model_executor.layers.fused_moe.oracle.int8": SimpleNamespace(
            __file__=os.fspath(site / "vllm/model_executor/layers/fused_moe/oracle/int8.py"),
            Int8MoeBackend=Int8MoeBackend,
            map_int8_backend=lambda value: Int8MoeBackend.TRITON,
            backend_to_kernel_cls=lambda backend: [TritonExperts],
            select_int8_moe_backend=lambda *args: None,
            convert_to_int8_moe_kernel_format=lambda *args: None,
            make_int8_moe_quant_config=lambda *args: None,
            make_int8_moe_kernel=lambda *args: None,
        ),
        "vllm.model_executor.custom_op": SimpleNamespace(
            op_registry_oot={"RMSNorm": HcuRMSNorm}
        ),
        "vllm.platforms": SimpleNamespace(current_platform=HcuPlatform()),
    }

    def find_distribution(name):
        try:
            return distributions[name]
        except KeyError:
            raise verify_installed.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(verify_installed.metadata,"distribution",find_distribution)
    monkeypatch.setattr(
        verify_installed.metadata,
        "entry_points",
        lambda *, group: tuple(entry_points[group]),
    )
    monkeypatch.setattr(
        verify_installed,
        "_import_module",
        lambda name: modules[name],
    )
    return distributions, entry_points, reports, modules


def test_attestation_accepts_owned_wheels_and_active_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installed_fixture(monkeypatch, tmp_path)

    report = verify_installed.collect_attestation()

    assert report["schema_version"] == 4
    assert report["vllm"]["version"] == "0.33.0+rocm723"
    assert report["adapter"]["version"] == "0.1.0"
    assert report["plugin"]["version"] == "0.33.0"
    assert report["extension"] == {
        "capabilities": ["lightop-rms-norm"],
        "entry_point": "das_hcu",
        "name": "das-hcu",
        "status": "active",
    }
    assert report["platform"] == "hcu_adapter.bridges.vllm.HcuPlatform"
    assert report["oot_classes"] == {
        "RMSNorm": "hcu_vllm_das.rms_norm.HcuRMSNorm",
    }
    assert report["int8_moe"] == {
        "backend": "TRITON",
        "official_symbols_unchanged": True,
        "aiter_integration": "unsupported: no public INT8 expert registry",
        "experts_class": "vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonExperts",
        "oracle_module_origin": os.fspath(
            tmp_path / "site-packages/vllm/model_executor/layers/fused_moe/oracle/int8.py"
        ),
    }
    json.dumps(report, sort_keys=True)


def test_attestation_rejects_a_non_033_vllm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    distributions, *_ = installed_fixture(monkeypatch, tmp_path)
    distributions["vllm"].version = "0.31.0"

    with pytest.raises(verify_installed.InstalledVerificationError, match="0.33.0"):
        verify_installed.collect_attestation()


def test_attestation_accepts_a_vllm_033_patch_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    distributions, *_ = installed_fixture(monkeypatch, tmp_path)
    distributions["vllm"].version = "0.33.1+rocm723"

    report = verify_installed.collect_attestation()

    assert report["vllm"]["version"] == "0.33.1+rocm723"


def test_attestation_rejects_editable_or_unrecorded_installations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    distributions, *_ = installed_fixture(monkeypatch, tmp_path)
    distributions["hcu-vllm-das-plugin"].direct_url = json.dumps(
        {"dir_info": {"editable": True}}
    )

    with pytest.raises(verify_installed.InstalledVerificationError, match="editable"):
        verify_installed.collect_attestation()

    distributions["hcu-vllm-das-plugin"].direct_url = None
    distributions["hcu-vllm-das-plugin"].record = ""
    with pytest.raises(verify_installed.InstalledVerificationError, match="RECORD"):
        verify_installed.collect_attestation()


def test_attestation_rejects_a_module_outside_its_distribution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _distributions, _points, _reports, modules = installed_fixture(
        monkeypatch, tmp_path
    )
    modules["hcu_vllm_das.extension"].__file__ = "/unowned/plugin.py"

    with pytest.raises(verify_installed.InstalledVerificationError, match="not owned"):
        verify_installed.collect_attestation()


def test_attestation_rejects_legacy_plugin_ownership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    distributions, points, *_ = installed_fixture(monkeypatch, tmp_path)
    points["vllm.general_plugins"].append(
        FakeEntryPoint(
            "hcu_ops",
            "hcu_vllm_das.legacy:register",
            distributions["hcu-vllm-das-plugin"],
        )
    )

    with pytest.raises(
        verify_installed.InstalledVerificationError,
        match="vllm.general_plugins",
    ):
        verify_installed.collect_attestation()


def test_attestation_rejects_a_missing_adapter_extension_entry_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _distributions, points, *_ = installed_fixture(monkeypatch, tmp_path)
    points["hcu_adapter.extensions"].clear()

    with pytest.raises(verify_installed.InstalledVerificationError, match="das_hcu"):
        verify_installed.collect_attestation()


def test_attestation_rejects_an_unexpected_selected_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _distributions, _points, _reports, modules = installed_fixture(
        monkeypatch, tmp_path
    )
    modules["vllm.platforms"].current_platform = SimpleNamespace()

    with pytest.raises(
        verify_installed.InstalledVerificationError,
        match="selected platform",
    ):
        verify_installed.collect_attestation()


def test_attestation_rejects_failed_activation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _distributions, _points, reports, _modules = installed_fixture(
        monkeypatch, tmp_path
    )
    reports[0].status = FailedStatus()
    reports[0].reason = "activation failed"

    with pytest.raises(
        verify_installed.InstalledVerificationError,
        match="activation failed",
    ):
        verify_installed.collect_attestation()


def test_attestation_rejects_a_missing_oot_registration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _distributions, _points, _reports, modules = installed_fixture(
        monkeypatch, tmp_path
    )
    del modules["vllm.model_executor.custom_op"].op_registry_oot["RMSNorm"]

    with pytest.raises(
        verify_installed.InstalledVerificationError,
        match="RMSNorm",
    ):
        verify_installed.collect_attestation()


def test_attestation_rejects_a_missing_required_capability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _distributions, _points, reports, _modules = installed_fixture(
        monkeypatch, tmp_path
    )
    reports[0].capabilities = frozenset()

    with pytest.raises(
        verify_installed.InstalledVerificationError,
        match="lightop-rms-norm",
    ):
        verify_installed.collect_attestation()


def test_attestation_rejects_an_unowned_int8_oracle_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _distributions, _points, _reports, modules = installed_fixture(
        monkeypatch, tmp_path
    )
    modules["vllm.model_executor.layers.fused_moe.oracle.int8"].__file__ = "/unowned/oracle.py"

    with pytest.raises(verify_installed.InstalledVerificationError, match="not owned"):
        verify_installed.collect_attestation()


def test_attestation_rejects_a_non_official_int8_selector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _distributions, _points, _reports, modules = installed_fixture(
        monkeypatch, tmp_path
    )
    official_oracle = modules[
        "vllm.model_executor.layers.fused_moe.oracle.int8"
    ]
    official_oracle.map_int8_backend = lambda value: Int8MoeBackend.AITER

    with pytest.raises(
        verify_installed.InstalledVerificationError,
        match="official Triton",
    ):
        verify_installed.collect_attestation()


def test_attestation_detects_int8_rebinding_during_activation(monkeypatch, tmp_path):
    _, _, _, modules = installed_fixture(monkeypatch, tmp_path)
    host = modules["hcu_adapter.extensions"]
    original = host.activate_extensions
    def mutate(context, points):
        reports = original(context, points)
        modules["vllm.model_executor.layers.fused_moe.oracle.int8"].map_int8_backend = lambda _: None
        return reports
    host.activate_extensions = mutate
    with pytest.raises(verify_installed.InstalledVerificationError, match="modified official INT8"):
        verify_installed.collect_attestation()


def test_attestation_accepts_split_host_distribution(monkeypatch,tmp_path):
    distributions,points,_,_=installed_fixture(monkeypatch,tmp_path)
    kernel=distributions['sgl-kernel']
    host=FakeDistribution('hcu-adapter','0.5.22+hcu.1',kernel.root,
        ['hcu_adapter/extensions.py','hcu_adapter/bridges/vllm.py','hcu_adapter-0.5.22.dist-info/RECORD'])
    distributions['hcu-adapter']=host
    kernel.files=[PurePosixPath('sgl_kernel/__init__.py'),PurePosixPath('sgl_kernel-0.0.dist-info/RECORD')]
    kernel.version='0.0.0+hcu.dev0'
    for group in ['vllm.platform_plugins','vllm.general_plugins']:
        for point in points[group]:point.dist=host
    report=verify_installed.collect_attestation()
    assert report['adapter']['distribution']=='hcu-adapter'


def test_attestation_refuses_duplicate_host_record_owners(monkeypatch,tmp_path):
    distributions,_,_,_=installed_fixture(monkeypatch,tmp_path)
    kernel=distributions['sgl-kernel']
    distributions['hcu-adapter']=FakeDistribution('hcu-adapter','0.5.22+hcu.1',kernel.root,
        ['hcu_adapter/extensions.py','hcu_adapter/bridges/vllm.py'])
    with pytest.raises(verify_installed.InstalledVerificationError,match='unique|multiple|ambig'):
        verify_installed.collect_attestation()
