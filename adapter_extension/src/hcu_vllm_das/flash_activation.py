# SPDX-License-Identifier: Apache-2.0
"""Independent declaration and preparation of the explicit HCU Flash backend.

This top-level module avoids attention.__init__'s backend import. Discovery
imports no framework or vendor. Activation checks public contracts;
ACTIVE does not establish a model's configuration, numerical or GPU support.
This owner does not install RMS, AITER, quantization aliases or lifecycle hooks.
"""

from __future__ import annotations

from importlib import import_module

from hcu_adapter.extensions import ExtensionContext, ExtensionSpec
from packaging.version import InvalidVersion, Version


CAPABILITY = "hcu-flash-attention"


class FlashActivationError(RuntimeError):
    """The explicit Flash capability cannot be prepared on this host."""


def activate(context: ExtensionContext) -> None:
    try:
        version = Version(context.vllm_version or "")
    except InvalidVersion as error:
        raise FlashActivationError("HCU Flash requires the vLLM 0.32 release line") from error
    if version.epoch != 0 or version.release[:2] != (0, 32):
        raise FlashActivationError("HCU Flash requires the vLLM 0.32 release line")

    bridge = import_module("hcu_adapter.bridges.vllm")
    detection = bridge.detect_hcu()
    if not detection.detected:
        raise FlashActivationError(f"HCU Flash requires a detected HCU: {detection.reason}")

    # The backend import checks the three official attention seams. Resolving
    # the public callable does not launch a kernel or install another owner.
    try:
        backend = import_module("hcu_vllm_das.attention.flash_attn")
        backend._vendor_varlen_func()
    except Exception as error:
        raise FlashActivationError(
            f"cannot prepare HCU Flash backend: {type(error).__name__}: {error}"
        ) from error


def extension_spec() -> ExtensionSpec:
    return ExtensionSpec(
        name="das-hcu-flash",
        api_version=1,
        priority=80,
        capabilities=frozenset({CAPABILITY}),
        activate=activate,
    )
