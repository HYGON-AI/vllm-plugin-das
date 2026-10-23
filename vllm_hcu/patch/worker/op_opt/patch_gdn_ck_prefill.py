# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route eligible complete Qwen GDN prefills through AITER CK."""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._aiter_ck_gdn import (
    aiter_ck_cu_seqlens,
    aiter_ck_runtime_safety_reason,
    aiter_ck_unsupported,
    aiter_ck_workspace,
    resolve_aiter_ck_gdn,
)
from ._common import (
    already_applied,
    load_exact_module,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
PATCH_ID = "worker.op_opt.mamba.gdn.aiter_ck_prefill"
TARGETS = (f"{TARGET_MODULE}.ChunkGatedDeltaRule.forward_native",)
_MARKER = "_vllm_hcu_aiter_ck_prefill_applied"
_WRAPPER = "_vllm_hcu_aiter_ck_prefill_wrapper"


def _store_core_output(output: torch.Tensor, core_attn_out: torch.Tensor) -> None:
    if output.dtype != core_attn_out.dtype or output.device != core_attn_out.device:
        raise ValueError("core_attn_out must share output dtype and device")
    source = output.reshape(-1)
    target = core_attn_out.reshape(-1)
    if target.numel() < source.numel():
        raise ValueError("core_attn_out is too small for CK GDN output")
    target[: source.numel()].copy_(source)


def apply_to_module(module: ModuleType) -> bool:
    qwen = load_exact_module(TARGET_MODULE, module)
    cls = require_class(qwen, "ChunkGatedDeltaRule", TARGETS[0])
    wrapped = ((cls, "forward_native", TARGETS[0], _WRAPPER),)
    if already_applied(qwen, _MARKER, wrapped):
        return False
    original = cls.forward_native
    require_exact_signature(
        original,
        TARGETS[0],
        positional=(
            "self", "q", "k", "v", "g", "beta", "initial_state",
            "output_final_state", "cu_seqlens", "chunk_indices",
            "chunk_offsets", "use_qk_l2norm_in_kernel", "core_attn_out",
        ),
        defaults={
            "cu_seqlens": None, "chunk_indices": None, "chunk_offsets": None,
            "use_qk_l2norm_in_kernel": True, "core_attn_out": None,
        },
    )
    safety_reason = aiter_ck_runtime_safety_reason()
    ops = None if safety_reason else resolve_aiter_ck_gdn()

    @functools.wraps(original)
    def hcu_forward_native(
        self, q, k, v, g, beta, initial_state, output_final_state,
        cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
        use_qk_l2norm_in_kernel=True, core_attn_out=None,
    ):
        def fallback():
            return original(
                self, q, k, v, g, beta, initial_state, output_final_state,
                cu_seqlens, chunk_indices, chunk_offsets,
                use_qk_l2norm_in_kernel, core_attn_out,
            )

        if safety_reason:
            qwen.logger.warning_once(
                "GDN fused CK prefill is disabled because %s; using the "
                "existing prefill path.", safety_reason,
            )
            return fallback()
        if ops is None or use_qk_l2norm_in_kernel:
            return fallback()
        if any(x.ndim < 1 or x.shape[0] != 1 for x in (q, k, v, g, beta)):
            return fallback()
        packed = (q[0], k[0], v[0], g[0], beta[0])
        capturing = torch.cuda.is_current_stream_capturing()
        reason = aiter_ck_unsupported(
            *packed, initial_state, cu_seqlens, capturing=capturing
        )
        if reason is not None:
            qwen.logger.warning_once(
                "GDN fused CK prefill cannot be used because %s; using the "
                "existing prefill path.", reason,
            )
            return fallback()
        ck_cu = aiter_ck_cu_seqlens(packed[0], cu_seqlens)
        flags = {
            "use_g": True, "use_gk": False, "use_exp2": False,
            "transpose_state": True, "qk_l2norm": False,
            "gate_in_kernel": False, "has_dt_bias": False,
            "beta_sigmoid_in_kernel": False,
            # AITER 0.1.7's Python/C++ binding cannot marshal the empty
            # final-state result produced when this is false.  Compute it and
            # discard it below to preserve the vLLM contract.
            "output_final_state": True,
            "cp_context": False, "cp_world_size": 2,
        }
        workspace = aiter_ck_workspace(ops, packed[0], packed[2], ck_cu, flags)
        output, final_state = ops.prefill(
            *packed, workspace, initial_state=initial_state,
            cu_seqlens=ck_cu, scale=packed[1].shape[-1] ** -0.5,
            **flags, cp_rank=1,
        )
        if not capturing:
            qwen.logger.info_once(
                "GDN fused CK prefill is active for supported inputs."
            )
        output = output.unsqueeze(0)
        if core_attn_out is not None:
            _store_core_output(output, core_attn_out)
        if not output_final_state:
            final_state = None
        return output, final_state

    setattr(hcu_forward_native, _WRAPPER, True)
    setattr(cls, "_vllm_hcu_original_forward_native", original)
    setattr(cls, "forward_native", hcu_forward_native)
    setattr(qwen, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
