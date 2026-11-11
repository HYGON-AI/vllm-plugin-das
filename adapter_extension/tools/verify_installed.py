#!/usr/bin/env python3
"""Verify an installed vLLM 0.33 adapter-das plus DAS extension stack."""

from __future__ import annotations

import hashlib
from importlib import import_module as _import_module
from importlib import metadata
import json
from pathlib import Path
import sys

from packaging.version import InvalidVersion, Version


_REQUIRED_CAPABILITIES = frozenset(
    {"lightop-rms-norm"}
)
_REQUIRED_OOT_CLASSES = {
    "RMSNorm": "hcu_vllm_das.rms_norm.HcuRMSNorm",
}
_REQUIRED_PLATFORM = "hcu_adapter.bridges.vllm.HcuPlatform"
_INT8_SYMBOLS = (
    "Int8MoeBackend", "map_int8_backend", "backend_to_kernel_cls",
    "select_int8_moe_backend", "convert_to_int8_moe_kernel_format",
    "make_int8_moe_quant_config", "make_int8_moe_kernel",
)


class InstalledVerificationError(RuntimeError):
    """Raised when installed files or runtime ownership are not proven."""


def _canonical_name(value: str) -> str:
    return value.casefold().replace("_", "-")


def _distribution(name: str):
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError as error:
        raise InstalledVerificationError(
            f"required distribution is not installed: {name}"
        ) from error
    files = distribution.files
    if not files:
        raise InstalledVerificationError(f"{name} has no installed file manifest")
    record = distribution.read_text("RECORD")
    if not record:
        raise InstalledVerificationError(f"{name} has no non-empty RECORD")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            editable = bool(
                json.loads(direct_url).get("dir_info", {}).get("editable", False)
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise InstalledVerificationError(
                f"{name} has malformed direct_url.json"
            ) from error
        if editable:
            raise InstalledVerificationError(
                f"{name} is an editable installation"
            )
    owned = frozenset(
        Path(distribution.locate_file(item)).resolve() for item in files
    )
    return distribution, owned, hashlib.sha256(record.encode()).hexdigest()


def _module_origin(module: object, owned: frozenset[Path], owner: str) -> str:
    raw_origin = getattr(module, "__file__", None)
    if not raw_origin:
        raise InstalledVerificationError(f"{owner} module has no file origin")
    origin = Path(raw_origin).resolve()
    if origin not in owned:
        raise InstalledVerificationError(
            f"module {origin} is not owned by {owner}"
        )
    return str(origin)


def _entry_points(group: str) -> tuple[object, ...]:
    return tuple(metadata.entry_points(group=group))


def _entry_point_owner(point: object) -> str:
    distribution = getattr(point, "dist", None)
    name = getattr(distribution, "metadata", {}).get("Name", "")
    return _canonical_name(name) if isinstance(name, str) else ""


def _require_entry_point(
    group: str,
    *,
    name: str,
    value: str,
    distribution: str,
) -> object:
    points = _entry_points(group)
    plugin_owned = [
        point
        for point in points
        if _entry_point_owner(point) == "hcu-vllm-das-plugin"
    ]
    if group in {"vllm.platform_plugins", "vllm.general_plugins"} and plugin_owned:
        raise InstalledVerificationError(
            f"DAS plugin must not own {group}: "
            + ", ".join(point.name for point in plugin_owned)
        )
    matches = [
        point
        for point in points
        if point.name == name
        and point.value == value
        and _entry_point_owner(point) == _canonical_name(distribution)
    ]
    if len(matches) != 1:
        raise InstalledVerificationError(
            f"expected exactly one {group} entry point {name!r} = {value!r} "
            f"owned by {distribution}"
        )
    return matches[0]


def _distribution_report(distribution: object, record_sha256: str) -> dict[str, str]:
    return {
        "record_sha256": record_sha256,
        "version": distribution.version,
    }


def _adapter_owner_name():
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('_hcu_plugin_installed_adapter_owner',
        Path(__file__).with_name('installed_adapter_owner.py'))
    helper=module_from_spec(spec)
    spec.loader.exec_module(helper)
    try:
        return helper.adapter_distribution_name()
    except RuntimeError as error:
        raise InstalledVerificationError(str(error)) from error


def collect_attestation() -> dict[str, object]:
    """Return a fail-closed installed-state attestation."""

    vllm_dist, vllm_owned, vllm_record = _distribution("vllm")
    adapter_name = _adapter_owner_name()
    adapter_dist, adapter_owned, adapter_record = _distribution(adapter_name)
    plugin_dist, plugin_owned, plugin_record = _distribution(
        "hcu-vllm-das-plugin"
    )
    try:
        parsed_vllm = Version(vllm_dist.version)
    except (InvalidVersion, TypeError) as error:
        raise InstalledVerificationError(
            f"installed vLLM version is invalid: {vllm_dist.version!r}"
        ) from error
    if parsed_vllm.epoch != 0 or parsed_vllm.release[:2] != (0, 33):
        raise InstalledVerificationError(
            "installed vLLM must satisfy >=0.33.0,<0.34.0, "
            f"got {vllm_dist.version!r}"
        )

    _require_entry_point(
        "vllm.platform_plugins",
        name="hcu_adapter",
        value="hcu_adapter.bridges.vllm:hcu_platform_plugin",
        distribution=adapter_name,
    )
    _require_entry_point(
        "vllm.general_plugins",
        name="hcu_adapter",
        value="hcu_adapter.bridges.vllm:hcu_runtime_plugin",
        distribution=adapter_name,
    )
    extension_point = _require_entry_point(
        "hcu_adapter.extensions",
        name="das_hcu",
        value="hcu_vllm_das.extension:extension_spec",
        distribution="hcu-vllm-das-plugin",
    )

    vllm_module = _import_module("vllm")
    adapter_extensions = _import_module("hcu_adapter.extensions")
    adapter_bridge = _import_module("hcu_adapter.bridges.vllm")
    plugin_extension = _import_module("hcu_vllm_das.extension")
    vllm_origin = _module_origin(vllm_module, vllm_owned, "vllm")
    adapter_origin = _module_origin(
        adapter_extensions, adapter_owned, adapter_name
    )
    adapter_bridge_origin = _module_origin(
        adapter_bridge, adapter_owned, adapter_name
    )
    plugin_origin = _module_origin(
        plugin_extension, plugin_owned, "hcu-vllm-das-plugin"
    )

    context = adapter_extensions.ExtensionContext(
        adapter_version=adapter_dist.version,
        vllm_version=vllm_dist.version,
    )
    official_int8_oracle = _import_module(
        "vllm.model_executor.layers.fused_moe.oracle.int8"
    )
    oracle_origin = _module_origin(official_int8_oracle, vllm_owned, "vllm")
    before = {name: getattr(official_int8_oracle, name, None) for name in _INT8_SYMBOLS}
    if any(value is None for value in before.values()):
        raise InstalledVerificationError("official INT8 oracle contract unavailable")
    if getattr(official_int8_oracle, "_hcu_vllm_das_aiter_int8_installed", False):
        raise InstalledVerificationError("retired private INT8 oracle integration is active")
    reports = adapter_extensions.activate_extensions(context, [extension_point])
    matching = [
        report for report in reports if report.entry_point_name == "das_hcu"
    ]
    if len(matching) != 1:
        raise InstalledVerificationError(
            "DAS activation did not produce exactly one das_hcu report"
        )
    extension_report = matching[0]
    status = getattr(extension_report.status, "value", extension_report.status)
    if status != "active":
        raise InstalledVerificationError(
            "DAS extension is not active: "
            + (extension_report.reason or str(status))
        )
    capabilities = frozenset(extension_report.capabilities)
    missing_capabilities = _REQUIRED_CAPABILITIES - capabilities
    if missing_capabilities:
        raise InstalledVerificationError(
            "DAS extension is missing required capabilities: "
            + ", ".join(sorted(missing_capabilities))
        )

    platforms = _import_module("vllm.platforms")
    selected_platform = getattr(platforms, "current_platform", None)
    platform_class = selected_platform.__class__
    platform_name = f"{platform_class.__module__}.{platform_class.__qualname__}"
    if platform_name != _REQUIRED_PLATFORM:
        raise InstalledVerificationError(
            f"unexpected selected platform: {platform_name}; "
            f"expected {_REQUIRED_PLATFORM}"
        )

    custom_op = _import_module("vllm.model_executor.custom_op")
    oot_classes: dict[str, str] = {}
    for class_name, expected_name in _REQUIRED_OOT_CLASSES.items():
        oot_class = custom_op.op_registry_oot.get(class_name)
        if oot_class is None:
            raise InstalledVerificationError(
                f"vLLM OOT registry has no DAS {class_name} implementation"
            )
        oot_name = f"{oot_class.__module__}.{oot_class.__qualname__}"
        if oot_name != expected_name:
            raise InstalledVerificationError(
                f"vLLM OOT {class_name} has unexpected owner: {oot_name}"
            )
        oot_module = _import_module(oot_class.__module__)
        _module_origin(oot_module, plugin_owned, "hcu-vllm-das-plugin")
        oot_classes[class_name] = oot_name

    if any(getattr(official_int8_oracle, name) is not value for name, value in before.items()):
        raise InstalledVerificationError("DAS activation modified official INT8 oracle")
    if "aiter-int8-moe" in capabilities:
        raise InstalledVerificationError("unregistered AITER INT8 capability advertised")
    selected_backend = official_int8_oracle.map_int8_backend("triton")
    selected_classes = official_int8_oracle.backend_to_kernel_cls(selected_backend)
    backend_name = getattr(selected_backend, "value", None)
    if backend_name != "TRITON" or not selected_classes:
        raise InstalledVerificationError("official Triton INT8 selector unavailable")
    experts_name = f"{selected_classes[0].__module__}.{selected_classes[0].__qualname__}"
    if not experts_name.startswith("vllm."):
        raise InstalledVerificationError("INT8 expert is not owned by official vLLM")

    return {
        "schema_version": 4,
        "vllm": {
            **_distribution_report(vllm_dist, vllm_record),
            "module_origin": vllm_origin,
        },
        "adapter": {
            "distribution": adapter_name,
            **_distribution_report(adapter_dist, adapter_record),
            "module_origin": adapter_origin,
            "bridge_origin": adapter_bridge_origin,
        },
        "plugin": {
            **_distribution_report(plugin_dist, plugin_record),
            "module_origin": plugin_origin,
        },
        "extension": {
            "entry_point": extension_report.entry_point_name,
            "name": extension_report.extension_name,
            "status": status,
            "capabilities": sorted(capabilities),
        },
        "platform": platform_name,
        "oot_classes": oot_classes,
        "int8_moe": {
            "backend": backend_name,
            "official_symbols_unchanged": True,
            "aiter_integration": "unsupported: no public INT8 expert registry",
            "experts_class": experts_name,
            "oracle_module_origin": oracle_origin,
        },
    }


def main() -> int:
    try:
        report = collect_attestation()
    except (InstalledVerificationError, OSError) as error:
        print(f"installed verification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
