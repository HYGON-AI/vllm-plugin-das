# SPDX-License-Identifier: Apache-2.0
"""Independent DSV4.1 configuration registration through public vLLM APIs.

Discovery imports no framework or vendor. This capability describes routing,
not checkpoint loading or kernel/model acceptance. It does not install the
legacy HCU package or replace an official quantization method.
"""
from __future__ import annotations

from importlib import import_module as _import_module
import inspect
import os
from threading import RLock

from hcu_adapter.extensions import (
    ExtensionCommitError,
    ExtensionContext,
    ExtensionSpec,
)
from packaging.version import InvalidVersion, Version

QUANTIZATION_METHOD = "hcu_dsv41_channel_fp8"
CAPABILITY = "dsv41-channel-fp8-routing"
_lock = RLock()
_publication = None
_failure: str | None = None
_publishing = False


class Dsv41ActivationError(RuntimeError):
    """The routing capability cannot be registered under the current contract."""


def _after_fork_child() -> None:
    global _lock, _failure
    _lock = RLock()
    if _publishing and _failure is None:
        _failure = "fork inherited incomplete DSV4.1 registration; use a fresh process"


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def _fatal(reason: str) -> None:
    global _failure
    if _failure is None:
        _failure = reason
    raise ExtensionCommitError(_failure)


def _require_release(raw_version: str | None) -> None:
    try:
        version = Version(raw_version) if raw_version is not None else None
    except (InvalidVersion, TypeError):
        version = None
    if version is None or version.epoch != 0 or version.release[:2] != (0, 33):
        raise Dsv41ActivationError(
            f"DSV4.1 routing requires vLLM >=0.33.0,<0.34.0; got {raw_version!r}"
        )


def check_health() -> None:
    """Read the public selector and refuse a lost owner without repairing it."""
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _publication is None:
            raise Dsv41ActivationError("DSV4.1 routing registration is not active")
        quant, config_class, register, resolve = _publication
        try:
            owned = (
                quant.register_quantization_config is register
                and quant.get_quantization_config is resolve
                and QUANTIZATION_METHOD in quant.QUANTIZATION_METHODS
                and resolve(QUANTIZATION_METHOD) is config_class
            )
        except BaseException as error:
            try:
                _fatal("DSV4.1 public quantization selector is unavailable")
            except ExtensionCommitError as refusal:
                raise refusal from error
        if not owned:
            _fatal("DSV4.1 public quantization registration changed owner")


def activate_extension(context: ExtensionContext) -> None:
    """Publish one distinct config; never overwrite FP8, CT or oracle owners."""
    global _publication, _publishing
    with _lock:
        if _failure is not None:
            raise ExtensionCommitError(_failure)
        if _publishing:
            _fatal("recursive DSV4.1 quantization registration")
        if _publication is not None:
            _publishing = True
            try:
                _require_release(context.vllm_version)
                check_health()
            except BaseException as error:
                try:
                    _fatal("DSV4.1 contract changed after public registration")
                except ExtensionCommitError as refusal:
                    raise refusal from error
            finally:
                _publishing = False
            return
        _require_release(context.vllm_version)
        _publishing = True
        try:
            try:
                quant = _import_module("vllm.model_executor.layers.quantization")
                candidate = _import_module(
                    "hcu_vllm_das.quantization.dsv41_channel_fp8"
                )
            except BaseException as error:
                # Native imports may register operators before failing. Do not
                # label such a partially prepared process as a clean fallback.
                try:
                    _fatal("DSV4.1 native quantization preparation failed")
                except ExtensionCommitError as refusal:
                    raise refusal from error
            config_class = candidate.HcuDSV41ChannelFP8Config
            for name in ("register_quantization_config", "get_quantization_config"):
                function = getattr(quant, name, None)
                if (
                    not callable(function)
                    or tuple(inspect.signature(function).parameters)
                    != ("quantization",)
                ):
                    raise Dsv41ActivationError(f"public quantization ABI changed: {name}")
            if not isinstance(quant.QUANTIZATION_METHODS, list):
                raise Dsv41ActivationError("public quantization registry contract changed")
            register = quant.register_quantization_config
            resolve = quant.get_quantization_config
            if QUANTIZATION_METHOD in quant.QUANTIZATION_METHODS:
                if resolve(QUANTIZATION_METHOD) is not config_class:
                    raise Dsv41ActivationError(
                        "DSV4.1 quantization name is already registered by another owner"
                    )
            else:
                try:
                    register(QUANTIZATION_METHOD)(config_class)
                    if resolve(QUANTIZATION_METHOD) is not config_class:
                        _fatal("DSV4.1 public registration did not retain its owner")
                except BaseException as error:
                    try:
                        _fatal("DSV4.1 public registration failed; use a fresh process")
                    except ExtensionCommitError as refusal:
                        raise refusal from error
            _publication = (quant, config_class, register, resolve)
        finally:
            _publishing = False


def extension_spec() -> ExtensionSpec:
    return ExtensionSpec(
        name="das-hcu-dsv41", api_version=1, priority=100,
        capabilities=frozenset({CAPABILITY}),
        activate=activate_extension, check_health=check_health,
    )


__all__ = [
    "QUANTIZATION_METHOD", "CAPABILITY", "Dsv41ActivationError",
    "activate_extension", "check_health", "extension_spec",
]
