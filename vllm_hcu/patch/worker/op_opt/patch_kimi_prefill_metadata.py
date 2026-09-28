# SPDX-License-Identifier: Apache-2.0
"""Preserve CPU lengths at the GDN builder's captured convolution helper."""
from functools import wraps
from types import ModuleType

from ._common import (
    PatchCompatibilityError, load_exact_module, require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.attention.backends.gdn_attn"
PATCH_ID = "worker.op_opt.kimi.prefill_cpu_lengths"
TARGETS = (f"{TARGET_MODULE}.compute_causal_conv1d_metadata",)
_MARKER = "_vllm_hcu_kimi_prefill_cpu_lengths"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    original = require_callable(target, "compute_causal_conv1d_metadata", TARGETS[0])
    if getattr(target, _MARKER, False):
        if getattr(original, _MARKER, False):
            return False
        raise PatchCompatibilityError(f"stale marker for {TARGETS[0]}; restart worker")
    if getattr(original, _MARKER, False):
        raise PatchCompatibilityError(f"partial marker for {TARGETS[0]}; restart worker")
    require_exact_signature(original, TARGETS[0],
                            positional=("query_start_loc_p_cpu",),
                            keyword_only=("device",))
    logged = False

    @wraps(original)
    def metadata(query_start_loc_p_cpu, *, device):
        nonlocal logged
        from vllm_hcu.models.kimi_k3.amd.ops.prefill_metadata import LENGTHS_KEY, enabled
        result = original(query_start_loc_p_cpu, device=device)
        if enabled():
            if query_start_loc_p_cpu.device.type != "cpu":
                raise ValueError("Kimi prefill metadata requires CPU query boundaries")
            # A fresh Python list survives reuse of the scheduler's pinned buffer.
            result[0][LENGTHS_KEY] = query_start_loc_p_cpu.diff().tolist()
            if not logged:
                from vllm.logger import init_logger
                # Match existing HCU route markers: INFO may be filtered in workers.
                init_logger(__name__).warning(
                    "HCU Kimi prefill CPU sequence lengths enabled "
                    "(VLLM_HCU_USE_KIMI_PREFILL_CPU_LENGTHS=1)."
                )
                logged = True
        return result

    setattr(metadata, _MARKER, True)
    target.compute_causal_conv1d_metadata = metadata
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
