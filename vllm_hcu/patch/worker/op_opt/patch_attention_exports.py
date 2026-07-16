# SPDX-License-Identifier: Apache-2.0
"""Export the HCU fused attention class from vLLM's public layer package."""

from __future__ import annotations

from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module, require_class

TARGET_MODULE = "vllm.model_executor.layers.attention"
PATCH_ID = "worker.op_opt.attention.fused_qkv_public_export"
TARGETS = (f"{TARGET_MODULE}.FusedQkvSplitRmsNormRopeAttention",)
_MARKER = "_vllm_hcu_fused_attention_export_applied"


def apply_to_module(module: ModuleType) -> bool:
    package = load_exact_module(TARGET_MODULE, module)
    if getattr(package, _MARKER, False):
        exported = require_class(package, "FusedQkvSplitRmsNormRopeAttention", TARGETS[0])
        if "FusedQkvSplitRmsNormRopeAttention" not in package.__all__:
            raise PatchCompatibilityError(f"required HCU export marker for {TARGETS[0]} is stale")
        return False
    if hasattr(package, "FusedQkvSplitRmsNormRopeAttention"):
        raise PatchCompatibilityError(f"required HCU export {TARGETS[0]} already exists")
    source = require_class(
        package.attention,
        "FusedQkvSplitRmsNormRopeAttention",
        f"{TARGET_MODULE}.attention.FusedQkvSplitRmsNormRopeAttention",
    )
    exports = getattr(package, "__all__", None)
    if not isinstance(exports, list) or "Attention" not in exports:
        raise PatchCompatibilityError(f"required HCU patch target {TARGET_MODULE}.__all__ is incompatible")
    setattr(package, "FusedQkvSplitRmsNormRopeAttention", source)
    exports.append("FusedQkvSplitRmsNormRopeAttention")
    setattr(package, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
