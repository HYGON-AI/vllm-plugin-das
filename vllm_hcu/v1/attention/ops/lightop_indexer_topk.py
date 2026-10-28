# SPDX-License-Identifier: Apache-2.0
"""Small, lazy LightOp adapter for the ordinary sparse-MLA decode selector."""

from __future__ import annotations

from collections.abc import Callable

import torch


def decode_topk(
    top_k_per_row_decode: Callable[..., None],
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> bool:
    """Call LightOp decode TopK with an ABI-safe output view.

    LightOp writes ``indices[row * top_k]`` directly.  The vLLM/HCU caller owns
    the destination and may provide a strided view when decode and prefill share
    a buffer, so use a contiguous temporary for the native call and copy back.
    ``row_ends`` has already expanded speculative metadata into one effective
    end per logits row; therefore this adapter deliberately passes ``next_n=1``.

    Any ABI precondition miss returns ``False`` so the caller keeps its exact
    Torch fallback; runtime errors from a native call that already started
    intentionally propagate.
    """
    if not callable(top_k_per_row_decode):
        return False
    if topk_tokens <= 0:
        return False
    if logits.dim() != 2 or logits.shape[1] == 0:
        return False
    if (
        topk_indices.dim() != 2
        or topk_indices.shape != (logits.shape[0], topk_tokens)
    ):
        return False
    if row_ends.dim() != 1 or row_ends.numel() != logits.shape[0]:
        return False
    if topk_indices.dtype != torch.int32:
        return False
    if logits.dtype != torch.float32:
        return False
    if logits.device != topk_indices.device:
        return False

    row_ends_i32 = row_ends.to(
        device=logits.device, dtype=torch.int32
    ).contiguous()
    native_output = (
        topk_indices
        if topk_indices.is_contiguous()
        else torch.empty_like(topk_indices, memory_format=torch.contiguous_format)
    )
    top_k_per_row_decode(
        logits,
        1,
        row_ends_i32,
        native_output,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )
    if native_output is not topk_indices:
        topk_indices.copy_(native_output)
    return True


def decode_topk_v2(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> bool:
    """Try LightOp adaptive ragged-v2 for the protected decode K=512 case.

    Import and ABI precondition failures return ``False`` so the caller can
    retain its exact Torch fallback. Once the tensors satisfy the v2 contract,
    runtime errors from the native call intentionally propagate.
    """
    if topk_tokens != 512:
        return False
    if (
        logits.dim() != 2
        or logits.shape[1] == 0
        or logits.dtype != torch.float32
        or logits.device.type != "cuda"
        or logits.stride(1) != 1
        or logits.stride(0) < logits.shape[1]
        or row_ends.dim() != 1
        or row_ends.numel() != logits.shape[0]
        or topk_indices.dim() != 2
        or topk_indices.shape != (logits.shape[0], topk_tokens)
        or topk_indices.dtype != torch.int32
        or logits.device != row_ends.device
        or logits.device != topk_indices.device
    ):
        return False
    topk_transform_ragged_v2 = _load_topk_transform_ragged_v2()
    if topk_transform_ragged_v2 is None:
        return False

    lengths = row_ends.to(dtype=torch.int32, device=logits.device).contiguous()
    row_starts = torch.zeros_like(lengths)
    offsets = torch.zeros_like(lengths)
    native_output = torch.empty(
        topk_indices.shape, dtype=torch.int32, device=topk_indices.device
    )
    topk_transform_ragged_v2(
        logits,
        lengths,
        out_offsets=offsets,
        out_indices=native_output,
        row_starts=row_starts,
    )
    _mask_inf_selections(logits, row_ends, native_output, topk_tokens)
    topk_indices.copy_(native_output)
    return True


def _mask_inf_selections(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    selected: torch.Tensor,
    topk_tokens: int,
) -> None:
    """Publish -1 for -inf selections on long rows, in place.

    The ragged-v2 kernel validates only ``0 <= id < length`` and can publish a
    masked-out (``-inf``) candidate as a valid index when a long row holds
    fewer than ``topk_tokens`` finite scores.  The Torch reference converts a
    selected ``-inf`` to ``-1`` on such rows, so mirror it here.  Short rows
    (``row_ends <= topk_tokens``) keep sequence order in both paths, including
    -inf positions, and must stay untouched.  Cost is O(rows * topk_tokens).
    """
    if logits.shape[1] == 0:
        selected.fill_(-1)
        return
    scores = logits.gather(1, selected.clamp(min=0))
    invalid = (
        (selected >= 0)
        & (scores == float("-inf"))
        & (row_ends > topk_tokens).unsqueeze(1)
    )
    selected.masked_fill_(invalid, -1)


def _load_topk_transform_ragged_v2() -> Callable[..., None] | None:
    """Resolve the top-level LightOp v2 export without touching device state.

    The Python wrapper defers to ``lightop.op.topk_transform_ragged_v2_interface``
    at call time, so a stale or partial native extension would otherwise raise
    after the adapter has claimed success. Probe the native symbol here so any
    ABI resolution miss degrades to the Torch fallback.
    """
    try:
        from lightop import topk_transform_ragged_v2
        from lightop import op as lightop_op

        if not callable(
            getattr(lightop_op, "topk_transform_ragged_v2_interface", None)
        ):
            return None
    except (ImportError, AttributeError, OSError):
        return None
    return topk_transform_ragged_v2 if callable(topk_transform_ragged_v2) else None


__all__ = ["decode_topk", "decode_topk_v2"]
