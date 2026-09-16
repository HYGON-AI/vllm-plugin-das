# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 model — hardware-isolated entry point.

The implementation lives under ``nvidia/`` and ``amd/``; this module picks the
right one for the current platform and re-exports the public classes used by
the model registry. (Mirrors ``vllm.models.minimax_m3``.)
"""

from typing import TYPE_CHECKING

# The plugin owns the HCU implementation.  Keep imports lazy so plugin
# discovery on CPU and non-ROCm hosts can still inspect the registry without
# importing accelerator-only kernels.
if TYPE_CHECKING:
    from .amd.linear import KimiLinearForCausalLM
    from .amd.model import KimiK3ForConditionalGeneration
    from .amd.mtp import KimiK3MTP


def __getattr__(name: str):
    if name == "KimiK3ForConditionalGeneration":
        from .amd.model import KimiK3ForConditionalGeneration

        return KimiK3ForConditionalGeneration
    if name == "KimiK3MTP":
        from .amd.mtp import KimiK3MTP

        return KimiK3MTP
    if name == "KimiLinearForCausalLM":
        from .amd.linear import KimiLinearForCausalLM

        return KimiLinearForCausalLM
    raise AttributeError(name)

__all__ = [
    "KimiK3ForConditionalGeneration",
    "KimiK3MTP",
    "KimiLinearForCausalLM",
]
