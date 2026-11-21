# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Run Qwen GDN prefill as one native aiter CK call (``hcu-aiter-ck``).

Selected with ``--gdn-prefill-backend hcu-aiter-ck``.  Only the module-local
``fla_chunk_gated_delta_rule`` symbol of the Qwen GDN module is wrapped; any
other prefill backend goes to the original function.  An explicit
``hcu-aiter-ck`` request never falls back: a missing CK build or inputs CK
cannot take raise instead of silently running another pipeline.
"""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

import torch
from vllm.logger import init_logger

from ._aiter_ck_gdn import resolve_ck_gdn_prefill
from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)
from ._gdn_backend import AITER_CK, requested_backend

logger = init_logger(f"vllm.{__name__}")

TARGET_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
PATCH_ID = "worker.op_opt.mamba.gdn.qwen_ck_prefill"
TARGETS = (f"{TARGET_MODULE}.fla_chunk_gated_delta_rule",)
_MARKER = "_vllm_hcu_qwen_gdn_ck_prefill_applied"
_WRAPPER = "_vllm_hcu_qwen_gdn_ck_prefill_wrapper"

_HEAD_DIM = 128
_MAX_HEAD_RATIO = 4
_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16)


def _ck_unsupported(arguments) -> str | None:
    """Return why CK cannot take this call, or None."""
    q, k, v = arguments["q"], arguments["k"], arguments["v"]
    if arguments["cu_seqlens"] is None or q.shape[0] != 1:
        return "inputs without a flattened batch and cu_seqlens"
    if arguments["g"] is None:
        return "a missing gate"
    if not (q.dtype == k.dtype == v.dtype) or q.dtype not in _ACTIVATION_DTYPES:
        return f"{q.dtype} activations"
    if q.shape[-1] != _HEAD_DIM or v.shape[-1] != _HEAD_DIM:
        return "head dims other than K=V=128"
    h_qk, h_v = q.shape[-2], v.shape[-2]
    if h_v % h_qk != 0 or h_v // h_qk > _MAX_HEAD_RATIO:
        return f"h_v/h_qk = {h_v}/{h_qk}"
    return None


def apply_to_module(module: ModuleType) -> bool:
    qwen = load_exact_module(TARGET_MODULE, module)
    wrapped = ((qwen, "fla_chunk_gated_delta_rule", TARGETS[0], _WRAPPER),)
    if already_applied(qwen, _MARKER, wrapped):
        return False

    official = require_callable(qwen, "fla_chunk_gated_delta_rule", TARGETS[0])
    require_exact_signature(
        official,
        TARGETS[0],
        positional=(
            "q", "k", "v", "g", "beta", "scale", "initial_state",
            "output_final_state", "cu_seqlens", "chunk_indices",
            "chunk_offsets", "use_qk_l2norm_in_kernel", "core_attn_out",
        ),
        defaults={
            "scale": None,
            "initial_state": None,
            "output_final_state": False,
            "cu_seqlens": None,
            "chunk_indices": None,
            "chunk_offsets": None,
            "use_qk_l2norm_in_kernel": False,
            "core_attn_out": None,
        },
    )
    signature = inspect.signature(official)

    @torch.compiler.disable
    @functools.wraps(official)
    def hcu_chunk_gated_delta_rule(*args, **kwargs):
        if requested_backend("prefill") != AITER_CK:
            return official(*args, **kwargs)
        kernel = resolve_ck_gdn_prefill()
        if kernel is None:
            raise RuntimeError(
                "GDN prefill hcu-aiter-ck was requested but aiter CK GDN "
                "(aiter.ops.ck_gdn.ck_gdn_prefill) is not installed"
            )
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        reason = _ck_unsupported(arguments)
        if reason is not None:
            raise RuntimeError(
                f"GDN prefill hcu-aiter-ck does not support {reason}; "
                "select another --gdn-prefill-backend"
            )
        logger.info_once("GDN prefill chunk_gated_delta_rule uses hcu-aiter-ck.")

        initial_state = arguments["initial_state"]
        if initial_state is not None:
            initial_state = initial_state.to(torch.float32).contiguous()
        scale = arguments["scale"]
        # vLLM hands q/k/v/g/beta as [1, T, H, D]; CK takes packed [T, H, D].
        # g is the final log-decay and beta is already sigmoid-applied, so the
        # in-kernel gate/sigmoid paths stay off.  use_exp2=False keeps the
        # natural-log convention; transpose_state=True matches the [V, K]
        # recurrent-state layout used by the vLLM pool.
        output, final_state = kernel(
            arguments["q"].squeeze(0).contiguous(),
            arguments["k"].squeeze(0).contiguous(),
            arguments["v"].squeeze(0).contiguous(),
            arguments["g"].squeeze(0).to(torch.float32).contiguous(),
            arguments["beta"].squeeze(0).to(torch.float32).contiguous(),
            initial_state=initial_state,
            cu_seqlens=arguments["cu_seqlens"].to(torch.int64),
            scale=None if scale is None else float(scale),
            use_g=True,
            use_exp2=False,
            transpose_state=True,
            qk_l2norm=arguments["use_qk_l2norm_in_kernel"],
            output_final_state=arguments["output_final_state"],
        )
        output = output.unsqueeze(0)
        core_attn_out = arguments["core_attn_out"]
        if core_attn_out is not None:
            flat = output.reshape(-1)
            core_attn_out.reshape(-1)[: flat.numel()].copy_(flat)
        return output, final_state

    setattr(hcu_chunk_gated_delta_rule, _WRAPPER, True)
    setattr(qwen, "_vllm_hcu_original_fla_chunk_gated_delta_rule", official)
    setattr(qwen, "fla_chunk_gated_delta_rule", hcu_chunk_gated_delta_rule)
    setattr(qwen, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
