# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Optional per-channel int8 (W8A16) lm_head for decode-sized batches.

Checkpoints commonly leave lm_head in 16 bit.  For a 248k-vocab head the
fp16 GEMV reads ~1 GB per decode step (~750 us on BW150); a per-output-row
int8 copy read through DeepGEMM's W8A16 GEMV halves that (~400 us).  This is
a precision trade-off and therefore opt-in: VLLM_HCU_INT8_LM_HEAD=1.
Batches above ``_MAX_M`` rows keep the 16-bit weight.
"""

from __future__ import annotations

from types import ModuleType

import torch

from ._common import load_exact_module, require_class, require_unpatched

TARGET_MODULE = "vllm.model_executor.layers.logits_processor"
PATCH_ID = "worker.op_opt.logits_processor.int8_lm_head"
TARGETS = (f"{TARGET_MODULE}.LogitsProcessor._apply_head",)
_MARKER = "_vllm_hcu_int8_lm_head_applied"
_WRAPPER = "_vllm_hcu_int8_lm_head_wrapper"
_STATE = "_vllm_hcu_int8_lm_head"
_UNSET = object()
_MAX_M = 32

# HCU-measured DeepGEMM configs for (N=248320, K=2048), marlin layout.
_CFG_SMALL = dict(NUM_K_WARPS=4, N_PER_WARP=32, K_PER_WARP=256, NUM_N_WARPS=2,
                  SPLITK=1, BLOCK_M=16, KERNEL_TYPE=0, WEIGHT_LAYOUT=1)
_CFG_LARGE = dict(NUM_K_WARPS=1, N_PER_WARP=32, K_PER_WARP=128, NUM_N_WARPS=1,
                  SPLITK=1, BLOCK_M=16, KERNEL_TYPE=0, WEIGHT_LAYOUT=1)


# 512-wide K tile for K not divisible by 1024 (Qwen3.5-4B, K=2560): measured
# 520-556 us at M=1..16 vs ~1050 us fp16 and 580-1210 us DeepGEMM default.
_CFG_SMALL_K512 = dict(NUM_K_WARPS=2, N_PER_WARP=32, K_PER_WARP=256,
                       NUM_N_WARPS=2, SPLITK=1, BLOCK_M=16, KERNEL_TYPE=0,
                       WEIGHT_LAYOUT=1)


def _k_tile(cfg: dict) -> int:
    return cfg["NUM_K_WARPS"] * cfg["K_PER_WARP"] * cfg["SPLITK"]


def _config(m: int, n: int, k: int) -> dict:
    """First tuned config whose K tile divides K, else DeepGEMM's default.

    A partial K tile makes the kernel return NaN (_CFG_SMALL's 1024-wide
    tile with Qwen3.5-4B's K=2560 produced garbage output), so a tuned
    config is only used when K is a multiple of its K tile.
    """
    import deepgemm

    candidates = (_CFG_SMALL, _CFG_SMALL_K512) if m <= 16 else (_CFG_LARGE,)
    for cfg in candidates:
        if k % _k_tile(cfg) == 0:
            return cfg
    return {**deepgemm.get_w8a16_linear_config(m, n, k), "WEIGHT_LAYOUT": 1}


def _build_state(lm_head):
    weight = getattr(lm_head, "weight", None)
    if (
        weight is None
        or weight.dim() != 2
        or weight.dtype not in (torch.float16, torch.bfloat16)
        or type(lm_head.quant_method).__name__
        not in ("UnquantizedEmbeddingMethod", "UnquantizedLinearMethod")
        or weight.shape[0] % 16
        or weight.shape[1] % 128
    ):
        return None
    try:
        import deepgemm
    except ImportError:
        return None
    w = weight.data
    scale = (w.float().abs().amax(dim=1) / 127.0).clamp_(min=1e-8)
    q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    rows = 8192  # bound the fp32 temporaries
    for i in range(0, w.shape[0], rows):
        q[i:i + rows] = torch.round(
            w[i:i + rows].float() / scale[i:i + rows, None]).clamp_(-127, 127)
    return deepgemm.w8a16_marlin_weight(q), scale.to(w.dtype)


def apply_to_module(module: ModuleType) -> bool:
    lp = load_exact_module(TARGET_MODULE, module)
    cls = require_class(lp, "LogitsProcessor", TARGETS[0])
    if getattr(lp, _MARKER, False):
        return False
    original = require_unpatched(cls, "_apply_head", TARGETS[0], _WRAPPER)

    def hcu_apply_head(self, lm_head, hidden_states, embedding_bias):
        import vllm_hcu.platforms.envs as henvs

        if (
            not henvs.optional_custom_op_enabled(henvs.VLLM_HCU_INT8_LM_HEAD)
            or embedding_bias is not None
            or (self.head_dtype is not None
                and self.head_dtype != hidden_states.dtype)
            or hidden_states.dim() != 2
            or hidden_states.shape[0] > _MAX_M
            or hidden_states.shape[0] == 0
        ):
            return original(self, lm_head, hidden_states, embedding_bias)
        state = getattr(lm_head, _STATE, _UNSET)
        if state is _UNSET:
            if torch.cuda.is_current_stream_capturing():
                return original(self, lm_head, hidden_states, embedding_bias)
            state = _build_state(lm_head)
            setattr(lm_head, _STATE, state)
        if state is None or hidden_states.dtype != state[1].dtype:
            return original(self, lm_head, hidden_states, embedding_bias)
        import deepgemm

        cfg = _config(hidden_states.shape[0], *state[0].shape)
        return deepgemm.w8a16_linear(
            hidden_states.contiguous(), state[0], state[1], config=cfg)

    hcu_apply_head.__wrapped__ = original
    setattr(hcu_apply_head, _WRAPPER, True)
    cls._vllm_hcu_original_apply_head = original
    cls._apply_head = hcu_apply_head
    setattr(lp, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
