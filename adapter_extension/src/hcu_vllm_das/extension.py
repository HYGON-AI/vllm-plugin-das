"""Adapter extension lifecycle for migrated DAS optimizations."""

from __future__ import annotations

from importlib import import_module as _import_module
import ast
import inspect
import threading

from packaging.version import InvalidVersion, Version

from hcu_adapter.extensions import (
    ExtensionContext,
    ExtensionSpec,
)

from .environment import register_compile_factors
from .version import TARGET_VLLM_RELEASE


_CAPABILITIES = frozenset({"lightop-rms-norm"})
_IMPLEMENTED_EXTENSION_API_VERSION = 1
_activation_lock = threading.Lock()
_activated = False


class DasExtensionError(RuntimeError):
    """Raised before an incompatible optimization can be activated."""


def _require_vllm_033_release_line(raw_version: str | None) -> None:
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
            "DAS adapter extension requires vLLM >=0.33.0,<0.34.0; "
            f"received {raw_version!r}"
        )


def _require_callable(owner: object, name: str, qualified_name: str) -> None:
    if not callable(getattr(owner, name, None)):
        raise DasExtensionError(f"required callable is unavailable: {qualified_name}")


def _require_rms_call_arity(function, name, arguments):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as error:
        if not inspect.isbuiltin(function):
            raise
        # Pybind exports can have generated overload documentation without
        # __text_signature__. Parse declarations without evaluating any code.
        # This checks the same positional call shape as Signature.bind; it is
        # not evidence that a native kernel or a numerical contract passed.
        for line in (inspect.getdoc(function) or "").splitlines():
            declaration = line.strip()
            prefix, separator, rest = declaration.partition(". ")
            if separator and prefix.isdecimal():
                declaration = rest
            if not declaration.startswith(name + "("):
                continue
            try:
                definition = ast.parse("def " + declaration + ":\n    pass\n").body[0]
            except SyntaxError:
                continue
            parameters = definition.args
            if parameters.vararg or parameters.kwarg:
                continue
            positional = parameters.posonlyargs + parameters.args
            required = len(positional) - len(parameters.defaults)
            if (required <= len(arguments) <= len(positional)
                    and all(default is not None for default in parameters.kw_defaults)):
                return
        raise DasExtensionError(
            f"lightop.norm.{name} builtin lacks a compatible documented "
            f"{len(arguments)}-argument call signature"
        ) from error
    else:
        signature.bind(*arguments)


def activate_extension(context: ExtensionContext) -> None:
    """Validate and register the migrated optimization capability once."""

    global _activated
    _require_vllm_033_release_line(context.vllm_version)
    with _activation_lock:
        if _activated:
            return
        try:
            lightop_norm = _import_module("lightop.norm")
        except Exception as error:
            raise DasExtensionError(
                f"cannot import required LightOp norm module: {error}"
            ) from error
        for name, arguments in (
            ("rmsnorm_forward_autograd", (object(), object(), 1e-6, False)),
            ("fused_add_rms_norm", (object(), object(), object(), 1e-6)),
        ):
            _require_callable(lightop_norm, name, f"lightop.norm.{name}")
            function = getattr(lightop_norm, name)
            implementations = (function,) if inspect.isroutine(function) else (function, function.__call__)
            if any(check(implementation) for implementation in implementations for check in (
                inspect.iscoroutinefunction, inspect.isasyncgenfunction, inspect.isgeneratorfunction,
            )):
                raise DasExtensionError(f"lightop.norm.{name} must be synchronous")
            try:
                _require_rms_call_arity(function, name, arguments)
            except (TypeError, ValueError) as error:
                raise DasExtensionError(
                    f"required four-positional-argument RMS ABI unavailable: lightop.norm.{name}"
                ) from error

        try:
            vllm_layernorm = _import_module(
                "vllm.model_executor.layers.layernorm"
            )
            torch_utils = _import_module("vllm.utils.torch_utils")
        except Exception as error:
            raise DasExtensionError(
                f"cannot import required vLLM 0.33 operator API: {error}"
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
