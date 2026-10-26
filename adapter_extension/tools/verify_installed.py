#!/usr/bin/env python3
"""Verify an installed vLLM 0.32 adapter plus DAS extension stack."""

from __future__ import annotations

import hashlib
from importlib import import_module as _import_module
from importlib import metadata
import json
from pathlib import Path
import sys

from packaging.version import InvalidVersion, Version


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


def collect_attestation() -> dict[str, object]:
    """Return a fail-closed installed-state attestation."""

    vllm_dist, vllm_owned, vllm_record = _distribution("vllm")
    adapter_dist, adapter_owned, adapter_record = _distribution(
        "hcu-vllm-adapter"
    )
    plugin_dist, plugin_owned, plugin_record = _distribution(
        "hcu-vllm-das-plugin"
    )
    try:
        parsed_vllm = Version(vllm_dist.version)
    except (InvalidVersion, TypeError) as error:
        raise InstalledVerificationError(
            f"installed vLLM version is invalid: {vllm_dist.version!r}"
        ) from error
    if parsed_vllm.epoch != 0 or parsed_vllm.release != (0, 32, 0):
        raise InstalledVerificationError(
            f"installed vLLM must be release 0.32.0, got {vllm_dist.version!r}"
        )

    _require_entry_point(
        "vllm.platform_plugins",
        name="hcu_adapter",
        value="hcu_vllm_adapter.plugin:hcu_platform_plugin",
        distribution="hcu-vllm-adapter",
    )
    _require_entry_point(
        "vllm.general_plugins",
        name="hcu_adapter",
        value="hcu_vllm_adapter.plugin:hcu_runtime_plugin",
        distribution="hcu-vllm-adapter",
    )
    extension_point = _require_entry_point(
        "hcu_vllm_adapter.extensions",
        name="das_hcu",
        value="hcu_vllm_das.extension:extension_spec",
        distribution="hcu-vllm-das-plugin",
    )

    vllm_module = _import_module("vllm")
    adapter_extensions = _import_module("hcu_vllm_adapter.extensions")
    plugin_extension = _import_module("hcu_vllm_das.extension")
    vllm_origin = _module_origin(vllm_module, vllm_owned, "vllm")
    adapter_origin = _module_origin(
        adapter_extensions, adapter_owned, "hcu-vllm-adapter"
    )
    plugin_origin = _module_origin(
        plugin_extension, plugin_owned, "hcu-vllm-das-plugin"
    )

    context = adapter_extensions.ExtensionContext(
        adapter_version=adapter_dist.version,
        vllm_version=vllm_dist.version,
    )
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

    custom_op = _import_module("vllm.model_executor.custom_op")
    oot_class = custom_op.op_registry_oot.get("SiluAndMul")
    if oot_class is None:
        raise InstalledVerificationError(
            "vLLM OOT registry has no DAS SiluAndMul implementation"
        )
    oot_name = f"{oot_class.__module__}.{oot_class.__qualname__}"
    if not oot_name.startswith("hcu_vllm_das.silu_and_mul."):
        raise InstalledVerificationError(
            f"vLLM OOT SiluAndMul has unexpected owner: {oot_name}"
        )
    oot_module = _import_module(oot_class.__module__)
    _module_origin(oot_module, plugin_owned, "hcu-vllm-das-plugin")

    return {
        "schema_version": 1,
        "vllm": {
            **_distribution_report(vllm_dist, vllm_record),
            "module_origin": vllm_origin,
        },
        "adapter": {
            **_distribution_report(adapter_dist, adapter_record),
            "module_origin": adapter_origin,
        },
        "plugin": {
            **_distribution_report(plugin_dist, plugin_record),
            "module_origin": plugin_origin,
        },
        "extension": {
            "entry_point": extension_report.entry_point_name,
            "name": extension_report.extension_name,
            "status": status,
            "capabilities": sorted(extension_report.capabilities),
        },
        "oot_class": oot_name,
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
