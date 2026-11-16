# SPDX-License-Identifier: Apache-2.0
# Ported from vllm-hcu-main d095594: HCU micro-optimizations for the Qwen GDN
# layer.
#
# 1. rearrange_mixed_qkv: replace the three-way torch.split + rearrange +
#    three .contiguous() copies (3 aten::copy_ launches per decode step,
#    invoked twice per step via spec + non_spec paths) with the single fused
#    hcu_rearrange_mixed_qkv triton op. Falls back to the reference path when
#    the op is unavailable or the input layout is unexpected.
# 2. strided-z part-3 output projection: v0.25's _output_projection reshapes
#    z to 2D before the gated RMSNorm, materializing a contiguous copy
#    (~4.96us on qwen3.5 int8wo decode). The replacement feeds the original
#    [L, num_v, head_v] strided z view straight into the registered custom op
#    hcu_rms_norm_gated_strided_z (opaque to dynamo — compile-safe), then
#    performs the same flatten + out_proj. Ineligible layouts fall back to
#    the original method verbatim.

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_exact_signature,
    require_unpatched,
)

TARGET_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
PATCH_ID = "worker.op_opt.mamba.gdn.hcu_fused_ops"
TARGETS = (
    f"{TARGET_MODULE}.rearrange_mixed_qkv",
    f"{TARGET_MODULE}._output_projection",
)
_MARKER = "_vllm_hcu_qwen_gdn_fused_ops_applied"
_REARRANGE_WRAPPER = "_vllm_hcu_gdn_rearrange_wrapper"
_PROJ_WRAPPER = "_vllm_hcu_gdn_output_projection_wrapper"

_REARRANGE_PARAMETERS = ("self", "mixed_qkv")


def _find_rearrange_owner(module: ModuleType):
    for value in vars(module).values():
        if isinstance(value, type) and "rearrange_mixed_qkv" in vars(value):
            return value
    return None


def apply_to_module(module: ModuleType) -> bool:
    # Import for op registration side effects (getattr guards below
    # still handle absence gracefully).
    import vllm_hcu.ops.hcu_rearrange_mixed_qkv  # noqa: F401
    import vllm_hcu.ops.hcu_rms_norm_gated_strided_z_op  # noqa: F401
    qwen = load_exact_module(TARGET_MODULE, module)
    if getattr(qwen, _MARKER, False):
        return False

    # --- 1. fused rearrange_mixed_qkv -------------------------------------
    owner = _find_rearrange_owner(qwen)
    if owner is None:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} is missing"
        )
    original_rearrange = require_unpatched(
        owner, "rearrange_mixed_qkv", TARGETS[0], _REARRANGE_WRAPPER
    )
    require_exact_signature(
        original_rearrange,
        TARGETS[0],
        positional=_REARRANGE_PARAMETERS,
    )

    @functools.wraps(original_rearrange)
    def hcu_rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is not None:
            import vllm_hcu.platforms.envs as henvs

            _op = getattr(torch.ops.vllm, "hcu_rearrange_mixed_qkv", None)
            if (
                henvs.VLLM_HCU_USE_CUSTOM_OPS
                and henvs.VLLM_HCU_USE_CUSTOM_REARRANGE_MIXED_QKV
                and _op is not None
                and mixed_qkv.dim() == 2
                and mixed_qkv.stride(-1) == 1
            ):
                try:
                    return _op(
                        mixed_qkv,
                        self.key_dim // self.tp_size // self.head_k_dim,
                        self.head_k_dim,
                        self.value_dim // self.tp_size // self.head_v_dim,
                        self.head_v_dim,
                    )
                except (AttributeError, RuntimeError):
                    pass
        return original_rearrange(self, mixed_qkv)

    setattr(hcu_rearrange_mixed_qkv, _REARRANGE_WRAPPER, True)
    owner._vllm_hcu_original_rearrange_mixed_qkv = original_rearrange
    owner.rearrange_mixed_qkv = hcu_rearrange_mixed_qkv

    # --- 2. strided-z gated RMSNorm via _output_projection ----------------
    # In v0.25 the part-3 norm receives z already reshaped to 2D:
    #   ``z = z.reshape(-1, z.shape[-1])`` in _output_projection — and that
    #   reshape is exactly the contiguous-copy (~4.96us on qwen3.5 int8wo
    #   decode) the 0.18 patch avoided. Replace _output_projection and feed
    #   the ORIGINAL [L, num_v, head_v] strided z view straight into
    #   torch.ops.vllm.hcu_rms_norm_gated_strided_z. The op is a registered
    #   custom op, so dynamo treats it as opaque under full-graph capture
    #   (never trace into rmsnorm_fn / input_guard). Ineligible layouts fall
    #   back to the original method verbatim.
    proj_original = require_unpatched(
        owner, "_output_projection", TARGETS[1], _PROJ_WRAPPER
    )
    # 0.28 signature: returns the projection result (no output/num_tokens
    # out-params like 0.25).
    require_exact_signature(
        proj_original,
        TARGETS[1],
        positional=("self", "core_attn_out", "z"),
    )

    @functools.wraps(proj_original)
    def hcu_output_projection(self, core_attn_out, z):
        import vllm_hcu.platforms.envs as henvs

        if (
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_USE_CUSTOM_STRIDED_Z
            and z.dim() in (2, 3)
            and z.stride(-1) == 1
        ):
            norm = self.norm
            if getattr(norm, "bias", None) is None and (
                norm.group_size is None or norm.group_size == z.shape[-1]
            ):
                _op = getattr(
                    torch.ops.vllm, "hcu_rms_norm_gated_strided_z", None
                )
                core_2d = core_attn_out.reshape(
                    -1, core_attn_out.shape[-1]
                )
                rows = core_2d.shape[0]
                if _op is not None and (
                    (z.dim() == 3 and rows == z.shape[0] * z.shape[1])
                    or (z.dim() == 2 and rows == z.shape[0])
                ):
                    # [M, N] z rides the same kernel as [L, num_v, N] via a
                    # trivial unsqueeze (rows_per_outer = 1).
                    z_view = z.unsqueeze(1) if z.dim() == 2 else z
                    out_norm = _op(
                        core_2d,
                        norm.weight,
                        z_view,
                        norm.eps,
                        bool(norm.norm_before_gate),
                        getattr(norm, "activation", "swish"),
                    )
                    out_norm = out_norm.reshape(z.shape)
                    core_attn_out = out_norm.flatten(-2)  # ... h d -> ... (h d)
                    out, _ = self.out_proj(core_attn_out)
                    return out
        return proj_original(self, core_attn_out, z)

    setattr(hcu_output_projection, _PROJ_WRAPPER, True)
    owner._vllm_hcu_original_output_projection = proj_original
    owner._output_projection = hcu_output_projection

    setattr(qwen, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
