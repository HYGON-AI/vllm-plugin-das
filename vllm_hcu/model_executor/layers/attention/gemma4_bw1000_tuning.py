# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Gemma4 BW1000 launch-parameter tuning for the stock Triton unified attention.

Ported from the validated stage1 bundle. No new kernel is introduced: the
existing ``kernel_unified_attention`` is launched with different parameters.
Persistent KV stays FP8; shapes outside the validated set use stock behaviour.

  * global head512 (Q16/KV2), q_len 2..1100, 9..MAX_SEQS sequences: BLOCK_M=64,
    BLOCK_Q=8, TILE_SIZE=32, 8 warps (the grid is recomputed for BLOCK_Q);
  * SWA head256 (Q16/KV8, window 1024) with K/V scale==1: read FP8 as BF16
    directly (``KV_QUANT_MODE=0``), skipping the FP32 scaling, at any sequence
    count; the warp/stage choice applies from 32 sequences up, with no ceiling.
"""

from __future__ import annotations

import functools
import threading

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_state = threading.local()
_installed = False
def _max_tuned_seqs() -> int:
    """Upper bound on running sequences for the *launch-parameter* tuning.

    Applies to the global head512 tiling only. That path's tiling, warp and
    stage choices have been measured at 9..64 sequences and nowhere else, so the
    default stays there; raising it applies them to shapes nobody has timed,
    which can be slower rather than faster.

    The SWA path is no longer governed by this: its warp/stage choice was swept
    at the operator level and holds from 32 sequences upward with no ceiling,
    while its unit-scale KV cast is worth about 1.5x independently of the
    sequence count and is not bounded at all.
    """
    import os

    raw = os.environ.get("VLLM_HCU_GEMMA4_BW1000_MAX_SEQS", "64").strip()
    value = int(raw)
    if value < 9:
        raise ValueError("VLLM_HCU_GEMMA4_BW1000_MAX_SEQS must be >= 9")
    return value


_unit_pairs: set[tuple[int, int]] = set()
_unit_checked: set[int] = set()
_unit_logged = False


def _global_tiling_eligible(kw: dict) -> bool:
    from vllm import envs
    from vllm.v1.kv_cache_interface import KVQuantMode

    q, k = kw["q"], kw["k"]
    return (
        q.shape[2] == 512
        and q.shape[1] == 16
        and k.shape[2] == 2
        and 2 <= kw["max_seqlen_q"] <= 1100
        and 9 <= len(kw["seqused_k"]) <= _max_tuned_seqs()
        and q.dtype == torch.bfloat16
        and kw.get("kv_quant_mode") == KVQuantMode.FP8_PER_TENSOR
        and kw.get("output_scale") is None
        and kw.get("window_size") == (-1, -1)
        and not kw.get("use_td", False)
        and not envs.VLLM_BATCH_INVARIANT
    )


def install() -> None:
    """Install the tuning hooks. Call once per worker process."""
    global _installed
    if _installed:
        return
    import vllm.v1.attention.backends.triton_attn as backend
    import vllm.v1.attention.ops.triton_unified_attention as ops

    # Host side: remember whether this call is global-tiling eligible; the
    # kernel launch below cannot see ``max_seqlen_q``.
    stock = backend.unified_attention

    @functools.wraps(stock)
    def unified(*a, **kw):
        if a:
            return stock(*a, **kw)
        _state.tile = _global_tiling_eligible(kw)
        try:
            return stock(**kw)
        finally:
            _state.tile = False

    backend.unified_attention = unified

    # SWA unit-scale audit, done before CUDA graph capture.
    original_forward = backend.TritonAttentionImpl.forward

    @functools.wraps(original_forward)
    def forward(self, layer, *args, **kwargs):
        if (
            self.head_size == 256
            and self.num_kv_heads == 8
            and self.sliding_window == (1023, 0)
            and id(layer) not in _unit_checked
            and not torch.cuda.is_current_stream_capturing()
        ):
            _unit_checked.add(id(layer))
            valid = (
                getattr(layer, "quant_config", "missing") is None
                and getattr(layer, "_k_scale_float", None) == 1.0
                and getattr(layer, "_v_scale_float", None) == 1.0
            )
            if valid:
                valid = (
                    float(layer._k_scale.item()) == 1.0
                    and float(layer._v_scale.item()) == 1.0
                )
            if valid:
                _unit_pairs.add((layer._k_scale.data_ptr(), layer._v_scale.data_ptr()))
            logger.info(
                "Gemma4 BW1000 SWA unit-KV audit: layer=%s enabled=%s",
                getattr(layer, "layer_name", "unknown"), valid,
            )
        return original_forward(self, layer, *args, **kwargs)

    backend.TritonAttentionImpl.forward = forward

    producer = ops.kernel_unified_attention

    class Launch:
        def __getattr__(self, name):
            return getattr(producer, name)

        def __getitem__(self, grid):
            def invoke(*args, **kw):
                global _unit_logged
                call = producer[grid]
                if (
                    not args
                    and getattr(_state, "tile", False)
                    and kw.get("KV_QUANT_MODE") == 1
                    and kw.get("HEAD_SIZE") == 512
                    and not kw.get("IS_3D", True)
                ):
                    kw.update(BLOCK_M=64, BLOCK_Q=8, TILE_SIZE=32)
                    kw["num_warps"] = 8
                    kw["num_stages"] = 1
                    call = producer[
                        (kw["query_ptr"].shape[0] // 8 + kw["num_seqs"], *grid[1:])
                    ]
                elif (
                    not args
                    and kw.get("KV_QUANT_MODE") == 1
                    and not kw.get("Q_IS_FP8", True)
                    and kw.get("HEAD_SIZE") == 256
                    and kw.get("num_query_heads") == 16
                    and kw.get("num_queries_per_kv") == 2
                    and kw.get("SLIDING_WINDOW") == 1024
                    and not kw.get("IS_3D", True)
                    and kw["query_ptr"].dtype == torch.bfloat16
                    and kw["key_cache_ptr"].dtype == torch.float8_e4m3fn
                    and kw.get("k_scale") is not None
                    and kw.get("v_scale") is not None
                    and (kw["k_scale"].data_ptr(), kw["v_scale"].data_ptr())
                    in _unit_pairs
                ):
                    # Reading FP8 as BF16 is exact: every e4m3 value is
                    # representable in bfloat16, and with both scales verified
                    # to be 1 the FP32 dequantization step is the identity.
                    # That argument does not involve the sequence count, so the
                    # cast is not bounded by it.
                    kw["KV_QUANT_MODE"] = 0
                    # The warp/stage choice, unlike the cast, was measured at
                    # 32..64 sequences only; leave other shapes on the stock
                    # launch parameters.
                    # Measured on this path (head 256, 16/8 heads, window
                    # 1024, q_len 5, unit scales): the warp/stage choice is
                    # 1.058x at 32 sequences, rising to 1.144x at 64, 1.194x at
                    # 128 and 1.139x at 256, so there is no upper bound to hold.
                    # Below 32 it is a loss, not a smaller win -- 0.892x at 8,
                    # 0.893x at 16, 0.924x at 24 -- so that floor stays.
                    if kw.get("num_seqs", 0) >= 32:
                        kw["num_stages"] = 1
                        kw.setdefault("num_warps", 4)
                    if not _unit_logged:
                        _unit_logged = True
                        logger.info(
                            "Gemma4 BW1000 SWA unit-KV cast active "
                            "(head256 Q16/KV8 window1024, any N); "
                            "launch tuning applies at 32..%d",
                            _max_tuned_seqs(),
                        )
                return call(*args, **kw)

            return invoke

    ops.kernel_unified_attention = Launch()
    _installed = True
    logger.info("Gemma4 BW1000 attention launch-parameter tuning installed")
