# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU-owned feature configuration stored in ``additional_config['hcu']``.

The sidecar keeps HCU-only fields out of vLLM's Pydantic/dataclass schemas.
It accepts both an object-style ``VllmConfig`` and its dictionary form so the
same normalization is used before pickling and in spawned processes.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, MutableMapping


_FEATURE_FIELDS = (
    "enable_lightly_cp",
    "enable_lightly_cplb",
    "enable_custom_sp",
    "enable_multi_layers_mtp",
    "deepep_auto",
    "moe_backend",
    "hcu_flash_attn_mode",
)
_BOOLEAN_FIELDS = _FEATURE_FIELDS[:5]
_SUPPORTED_MOE_BACKENDS = frozenset({"auto", "deep_gemm"})
_LEGACY_DEEP_GEMM_BACKEND = "dpsk_deep_gemm"
_DEEP_GEMM_BACKEND = "deep_gemm"
_legacy_backend_warning_emitted = False
_legacy_backend_warning_lock = threading.Lock()
_SUPPORTED_FLASH_ATTN_MODES = frozenset(
    {"classic", "cutlass", "custom", "varlen"}
)


def normalize_hcu_moe_backend(value: Any) -> Any:
    """Map the public legacy alias to the canonical vLLM backend name."""

    if value != _LEGACY_DEEP_GEMM_BACKEND:
        return value
    global _legacy_backend_warning_emitted
    with _legacy_backend_warning_lock:
        if not _legacy_backend_warning_emitted:
            warnings.warn(
                "moe_backend='dpsk_deep_gemm' is deprecated; use "
                "moe_backend='deep_gemm' instead",
                FutureWarning,
                stacklevel=3,
            )
            _legacy_backend_warning_emitted = True
    return _DEEP_GEMM_BACKEND


@dataclass(frozen=True, slots=True)
class HcuFeatureConfig:
    """Normalized HCU-specific switches.

    ``moe_backend='auto'`` delegates non-DeepEP selection to vLLM's normal
    oracle. The explicit ``deep_gemm`` value mirrors vLLM's official backend
    request; ``deepep_auto`` owns the HCU Channel-FP8/INT8 specialization.
    """

    enable_lightly_cp: bool = False
    enable_lightly_cplb: bool = False
    enable_custom_sp: bool = False
    enable_multi_layers_mtp: bool = False
    deepep_auto: bool = False
    moe_backend: str = "auto"
    hcu_flash_attn_mode: str | None = None

    def __post_init__(self) -> None:
        for name in _BOOLEAN_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(
                    f"HCU config field {name!r} must be bool, "
                    f"got {type(value).__name__}"
                )
        if not isinstance(self.moe_backend, str):
            raise TypeError(
                "HCU config field 'moe_backend' must be str, "
                f"got {type(self.moe_backend).__name__}"
            )
        normalized_backend = normalize_hcu_moe_backend(self.moe_backend)
        if normalized_backend != self.moe_backend:
            object.__setattr__(self, "moe_backend", normalized_backend)
        if self.moe_backend not in _SUPPORTED_MOE_BACKENDS:
            supported = ", ".join(sorted(_SUPPORTED_MOE_BACKENDS))
            raise ValueError(
                f"unsupported HCU moe_backend {self.moe_backend!r}; expected one of {supported}"
            )
        if self.hcu_flash_attn_mode is not None:
            if not isinstance(self.hcu_flash_attn_mode, str):
                raise TypeError(
                    "HCU config field 'hcu_flash_attn_mode' must be str or None, "
                    f"got {type(self.hcu_flash_attn_mode).__name__}"
                )
            if self.hcu_flash_attn_mode not in _SUPPORTED_FLASH_ATTN_MODES:
                supported = ", ".join(sorted(_SUPPORTED_FLASH_ATTN_MODES))
                raise ValueError(
                    "unsupported HCU hcu_flash_attn_mode "
                    f"{self.hcu_flash_attn_mode!r}; expected one of {supported}"
                )
        if self.enable_lightly_cplb and not self.enable_lightly_cp:
            raise ValueError("enable_lightly_cplb requires enable_lightly_cp")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "HcuFeatureConfig":
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise TypeError(
                "additional_config['hcu'] must be a mapping or HcuFeatureConfig, "
                f"got {type(values).__name__}"
            )
        unknown = set(values).difference(_FEATURE_FIELDS)
        if unknown:
            names = ", ".join(sorted(str(item) for item in unknown))
            raise ValueError(f"unknown HCU config field(s): {names}")
        return cls(**dict(values))

    def with_updates(self, **updates: Any) -> "HcuFeatureConfig":
        unknown = set(updates).difference(_FEATURE_FIELDS)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown HCU config field(s): {names}")
        return replace(self, **updates)

    def to_dict(self) -> dict[str, bool | str | None]:
        return asdict(self)


def _config_from_payload(payload: object) -> HcuFeatureConfig:
    if payload is None:
        return HcuFeatureConfig()
    if isinstance(payload, HcuFeatureConfig):
        return payload
    if isinstance(payload, Mapping):
        return HcuFeatureConfig.from_mapping(payload)

    # Compatibility with a pickled/simple object representation.  Missing
    # fields use sidecar defaults; an object with none of the fields is almost
    # certainly the wrong level and is rejected.
    present = {name: getattr(payload, name) for name in _FEATURE_FIELDS if hasattr(payload, name)}
    if not present:
        raise TypeError(
            "HCU config payload must be a mapping, HcuFeatureConfig, or object "
            "with HCU feature attributes"
        )
    return HcuFeatureConfig.from_mapping(present)


def _read_additional_config(container: object) -> object:
    if container is None or isinstance(container, HcuFeatureConfig):
        return container

    if isinstance(container, Mapping):
        # Direct HCU payload is useful in tests and internal adapters.
        if set(container).issubset(_FEATURE_FIELDS) and "hcu" not in container:
            return container
        # A mapping with ``hcu`` is an additional_config mapping.
        if "hcu" in container and "additional_config" not in container:
            return container.get("hcu")
        # Otherwise treat it as a dictionary representation of VllmConfig.
        additional = container.get("additional_config")
    elif hasattr(container, "additional_config"):
        additional = getattr(container, "additional_config")
    else:
        # Accept a direct object-form HCU payload.
        return container

    if additional is None:
        return None
    if not isinstance(additional, Mapping):
        raise TypeError(
            "vllm_config.additional_config must be a mapping or None, "
            f"got {type(additional).__name__}"
        )
    return additional.get("hcu")


def get_hcu_config(vllm_config: object | None) -> HcuFeatureConfig:
    """Read and validate ``vllm_config.additional_config['hcu']``.

    A direct HCU mapping/config object is also accepted for lightweight
    adapters.  The returned value is always immutable and fully populated.
    """

    return _config_from_payload(_read_additional_config(vllm_config))


def write_hcu_config(
    additional_config: MutableMapping[str, Any],
    config: HcuFeatureConfig | Mapping[str, Any],
) -> HcuFeatureConfig:
    """Write a normalized sidecar to an existing additional-config mapping."""

    if not isinstance(additional_config, MutableMapping):
        raise TypeError("additional_config must be a mutable mapping")
    normalized = (
        config
        if isinstance(config, HcuFeatureConfig)
        else HcuFeatureConfig.from_mapping(config)
    )
    additional_config["hcu"] = normalized.to_dict()
    return normalized


def set_hcu_config(
    vllm_config: object,
    config: HcuFeatureConfig | Mapping[str, Any] | None = None,
    **updates: Any,
) -> HcuFeatureConfig:
    """Normalize and store the HCU sidecar on a dict/object VllmConfig.

    Existing unrelated ``additional_config`` keys are preserved.  Passing no
    explicit ``config`` starts from the currently stored sidecar, which makes
    CLI/Python compatibility wrappers able to merge one switch at a time.
    """

    if config is None:
        normalized = get_hcu_config(vllm_config)
    elif isinstance(config, HcuFeatureConfig):
        normalized = config
    else:
        normalized = HcuFeatureConfig.from_mapping(config)
    if updates:
        normalized = normalized.with_updates(**updates)

    if isinstance(vllm_config, MutableMapping):
        if "hcu" in vllm_config and "additional_config" not in vllm_config:
            # The caller explicitly supplied an additional_config mapping.
            write_hcu_config(vllm_config, normalized)
            return normalized
        additional = vllm_config.get("additional_config")
        if additional is None:
            additional = {}
            vllm_config["additional_config"] = additional
        elif not isinstance(additional, MutableMapping):
            if not isinstance(additional, Mapping):
                raise TypeError("vllm_config['additional_config'] must be a mapping or None")
            additional = dict(additional)
            vllm_config["additional_config"] = additional
        write_hcu_config(additional, normalized)
        return normalized

    if not hasattr(vllm_config, "additional_config"):
        raise TypeError("vllm_config must be a mutable mapping or expose additional_config")
    additional = getattr(vllm_config, "additional_config")
    if additional is None:
        additional = {}
        setattr(vllm_config, "additional_config", additional)
    elif not isinstance(additional, MutableMapping):
        if not isinstance(additional, Mapping):
            raise TypeError("vllm_config.additional_config must be a mapping or None")
        additional = dict(additional)
        setattr(vllm_config, "additional_config", additional)
    write_hcu_config(additional, normalized)
    return normalized


def pop_hcu_feature_kwargs(
    kwargs: MutableMapping[str, Any],
    *,
    base: HcuFeatureConfig | Mapping[str, Any] | None = None,
) -> HcuFeatureConfig:
    """Extract legacy HCU keyword arguments before constructing VllmConfig.

    Only the five owned fields are removed.  This helper is intentionally
    independent of vLLM's constructor and can be used by CLI and Python-entry
    wrappers without adding fields to upstream classes.
    """

    if not isinstance(kwargs, MutableMapping):
        raise TypeError("kwargs must be a mutable mapping")
    if base is None:
        normalized = HcuFeatureConfig()
    elif isinstance(base, HcuFeatureConfig):
        normalized = base
    else:
        normalized = HcuFeatureConfig.from_mapping(base)
    updates = {name: kwargs.pop(name) for name in _FEATURE_FIELDS if name in kwargs}
    return normalized.with_updates(**updates) if updates else normalized


__all__ = [
    "HcuFeatureConfig",
    "get_hcu_config",
    "normalize_hcu_moe_backend",
    "pop_hcu_feature_kwargs",
    "set_hcu_config",
    "write_hcu_config",
]
