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
) -> None:
    """Call LightOp decode TopK with an ABI-safe output view.

    LightOp writes ``indices[row * top_k]`` directly.  The vLLM/HCU caller owns
    the destination and may provide a strided view when decode and prefill share
    a buffer, so use a contiguous temporary for the native call and copy back.
    ``row_ends`` has already expanded speculative metadata into one effective
    end per logits row; therefore this adapter deliberately passes ``next_n=1``.
    """
    if not callable(top_k_per_row_decode):
        raise RuntimeError("LightOp decode TopK symbol is unavailable")
    if logits.dim() != 2:
        raise RuntimeError(f"decode topk expects 2D logits, got {logits.shape}")
    if topk_indices.dim() != 2 or topk_indices.shape[0] != logits.shape[0]:
        raise RuntimeError(
            "decode topk output/logits row mismatch: "
            f"logits={tuple(logits.shape)}, output={tuple(topk_indices.shape)}"
        )
    if topk_indices.dtype != torch.int32:
        raise RuntimeError(
            f"decode topk output must be int32, got {topk_indices.dtype}"
        )
    if logits.dtype != torch.float32:
        raise RuntimeError(f"decode topk logits must be float32, got {logits.dtype}")

    native_output = (
        topk_indices
        if topk_indices.is_contiguous()
        else torch.empty_like(topk_indices, memory_format=torch.contiguous_format)
    )
    top_k_per_row_decode(
        logits,
        1,
        row_ends.to(device=logits.device, dtype=torch.int32).reshape(-1),
        native_output,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )
    if native_output is not topk_indices:
        topk_indices.copy_(native_output)


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
    topk_indices.copy_(native_output)
    return True


def _load_topk_transform_ragged_v2() -> Callable[..., None] | None:
    """Resolve the top-level LightOp v2 export without touching device state."""
    try:
        from lightop import topk_transform_ragged_v2
    except (ImportError, AttributeError):
        return None
    return topk_transform_ragged_v2 if callable(topk_transform_ragged_v2) else None


__all__ = ["decode_topk"]
