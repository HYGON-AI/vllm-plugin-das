"""Adapter extension lifecycle for migrated DAS optimizations."""

from __future__ import annotations

from importlib import import_module as _import_module
import inspect
import threading

from packaging.version import InvalidVersion, Version

from hcu_adapter.extensions import (
    ExtensionContext,
    ExtensionSpec,
)

from .environment import register_compile_factors
from .version import TARGET_VLLM_RELEASE


_CAPABILITIES = frozenset({"lightop-rms-norm", "aiter-int8-moe"})
_IMPLEMENTED_EXTENSION_API_VERSION = 1
_AITER_SIGNATURES = {
    "get_aiter_moe_config": (
        "M",
        "E",
        "N1",
        "N2",
        "K",
        "top_k",
        "block_size",
        "dtype",
        "quant_type",
        "activation",
        "gated",
        "spec_sol_type",
        "use_shuffle",
    ),
    "aiter_moe": (
        "hidden_states",
        "w1",
        "w2",
        "topk_weights",
        "topk_ids",
        "moe_config",
        "inplace",
        "activation",
        "w1_scale",
        "w2_scale",
        "w1_zp",
        "w2_zp",
        "a1_scale",
        "a2_scale",
        "block_shape",
        "global_num_experts",
        "expert_map",
        "routed_scaling_factor",
        "use_weight_shuffle",
        "output_dtype",
        "gemm1_alpha",
        "gemm1_limit",
    ),
}
_activation_lock = threading.Lock()
_activated = False


class DasExtensionError(RuntimeError):
    """Raised before an incompatible optimization can be activated."""


def _require_vllm_032_release_line(raw_version: str | None) -> None:
    try:
        parsed = Version(raw_version) if raw_version is not None else None
    except (InvalidVersion, TypeError):
        parsed = None
    if (
        parsed is None
        or parsed.epoch != 0
        or parsed.release[:2] != TARGET_VLLM_RELEASE[:2]
    ):
        raise DasExtensionError(
            "DAS adapter extension requires vLLM >=0.32.0,<0.33.0; "
            f"received {raw_version!r}"
        )


def _require_callable(owner: object, name: str, qualified_name: str) -> None:
    if not callable(getattr(owner, name, None)):
        raise DasExtensionError(f"required callable is unavailable: {qualified_name}")


def _require_signature(owner: object, name: str, qualified_name: str) -> None:
    _require_callable(owner, name, qualified_name)
    value = getattr(owner, name)
    try:
        actual = tuple(inspect.signature(value).parameters)
    except (TypeError, ValueError) as error:
        raise DasExtensionError(
            f"cannot inspect required callable: {qualified_name}"
        ) from error
    expected = _AITER_SIGNATURES[name]
    if actual != expected:
        raise DasExtensionError(
            f"required callable signature changed for {qualified_name}: "
            f"expected {expected}, got {actual}"
        )


def activate_extension(context: ExtensionContext) -> None:
    """Validate and register the migrated optimization capability once."""

    global _activated
    _require_vllm_032_release_line(context.vllm_version)
    with _activation_lock:
        if _activated:
            return
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
            aiter_moe = _import_module("aiter.moe")
        except Exception as error:
            raise DasExtensionError(
                f"cannot import required AITER MoE module: {error}"
            ) from error
        moe_quant_type = getattr(aiter_moe, "MoeQuantType", None)
        if getattr(moe_quant_type, "W8A8", None) != "int8_w8a8":
            raise DasExtensionError(
                "required AITER quant type is unavailable or changed: "
                "aiter.moe.MoeQuantType.W8A8"
            )
        for name in _AITER_SIGNATURES:
            _require_signature(aiter_moe, name, f"aiter.moe.{name}")

        try:
            vllm_layernorm = _import_module(
                "vllm.model_executor.layers.layernorm"
            )
            torch_utils = _import_module("vllm.utils.torch_utils")
        except Exception as error:
            raise DasExtensionError(
                f"cannot import required vLLM 0.32 operator API: {error}"
            ) from error
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
            register_compile_factors(_import_module("vllm.envs"))
            _import_module("hcu_vllm_das.rms_norm")
            int8_oracle = _import_module("hcu_vllm_das.moe.int8_oracle")
            install_oracle = getattr(int8_oracle, "install", None)
            if not callable(install_oracle):
                raise DasExtensionError(
                    "required callable is unavailable: "
                    "hcu_vllm_das.moe.int8_oracle.install"
                )
            install_oracle()
        except Exception as error:
            if isinstance(error, DasExtensionError):
                raise
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
