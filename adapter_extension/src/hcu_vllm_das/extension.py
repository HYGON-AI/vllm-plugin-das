"""Adapter extension lifecycle for migrated DAS optimizations."""

from __future__ import annotations

from importlib import import_module as _import_module
import threading

from packaging.version import InvalidVersion, Version

from hcu_vllm_adapter.extensions import (
    ExtensionContext,
    ExtensionSpec,
)

from .version import TARGET_VLLM_RELEASE


_CAPABILITIES = frozenset({"lightop-rms-norm", "lightop-silu-and-mul"})
_IMPLEMENTED_EXTENSION_API_VERSION = 1
_activation_lock = threading.Lock()
_activated = False


class DasExtensionError(RuntimeError):
    """Raised before an incompatible optimization can be activated."""


def _require_vllm_032(raw_version: str | None) -> None:
    try:
        parsed = Version(raw_version) if raw_version is not None else None
    except (InvalidVersion, TypeError):
        parsed = None
    if (
        parsed is None
        or parsed.epoch != 0
        or parsed.release != TARGET_VLLM_RELEASE
    ):
        raise DasExtensionError(
            "DAS adapter extension requires vLLM 0.32.0; "
            f"received {raw_version!r}"
        )


def _require_callable(owner: object, name: str, qualified_name: str) -> None:
    if not callable(getattr(owner, name, None)):
        raise DasExtensionError(f"required callable is unavailable: {qualified_name}")


def activate_extension(context: ExtensionContext) -> None:
    """Validate and register the migrated optimization capability once."""

    global _activated
    _require_vllm_032(context.vllm_version)
    with _activation_lock:
        if _activated:
            return
        try:
            lightop_activation = _import_module("lightop.activation")
        except Exception as error:
            raise DasExtensionError(
                f"cannot import required LightOp activation module: {error}"
            ) from error
        _require_callable(
            lightop_activation,
            "silu_and_mul_opt",
            "lightop.activation.silu_and_mul_opt",
        )
        try:
            lightop_norm = _import_module("lightop.norm")
        except Exception as error:
            raise DasExtensionError(
                f"cannot import required LightOp norm module: {error}"
            ) from error
        _require_callable(
            lightop_norm,
            "rmsnorm_forward_autograd",
            "lightop.norm.rmsnorm_forward_autograd",
        )
        _require_callable(
            lightop_norm,
            "fused_add_rms_norm",
            "lightop.norm.fused_add_rms_norm",
        )

        try:
            vllm_activation = _import_module(
                "vllm.model_executor.layers.activation"
            )
            vllm_layernorm = _import_module(
                "vllm.model_executor.layers.layernorm"
            )
            torch_utils = _import_module("vllm.utils.torch_utils")
        except Exception as error:
            raise DasExtensionError(
                f"cannot import required vLLM 0.32 operator API: {error}"
            ) from error
        silu_and_mul = getattr(vllm_activation, "SiluAndMul", None)
        _require_callable(
            silu_and_mul,
            "register_oot",
            "vllm.model_executor.layers.activation.SiluAndMul.register_oot",
        )
        rms_norm = getattr(vllm_layernorm, "RMSNorm", None)
        _require_callable(
            rms_norm,
            "register_oot",
            "vllm.model_executor.layers.layernorm.RMSNorm.register_oot",
        )
        _require_callable(
            torch_utils,
            "direct_register_custom_op",
            "vllm.utils.torch_utils.direct_register_custom_op",
        )

        try:
            _import_module("hcu_vllm_das.silu_and_mul")
            _import_module("hcu_vllm_das.rms_norm")
        except Exception as error:
            raise DasExtensionError(
                f"cannot register DAS operator capabilities: {error}"
            ) from error
        _activated = True


def extension_spec() -> ExtensionSpec:
    """Return the adapter-owned discovery contract without runtime imports."""

    return ExtensionSpec(
        name="das-hcu",
        api_version=_IMPLEMENTED_EXTENSION_API_VERSION,
        priority=100,
        capabilities=_CAPABILITIES,
        activate=activate_extension,
    )


__all__ = [
    "DasExtensionError",
    "activate_extension",
    "extension_spec",
]
