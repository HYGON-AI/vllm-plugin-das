"""Apply KV descales around the vendor's mixed-dtype paged attention.

Q*(K*ks) equals (Q*ks)*K; attention(V*vs) equals attention(V)*vs.
Use two small GPU row operations, preserving native gather/paged attention and
the caller's output buffer. Q and output products round to the query dtype;
non-unit-scale FP32-reference and changed-input Graph tests are required.
Remove when the vendor mixed-dtype API applies descales directly and passes
the same tests. Clearing the native descales avoids future double application.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _scale_rows(
    source, destination, scale, cu_lens,
    source_token_stride: tl.constexpr, source_head_stride: tl.constexpr,
    dest_token_stride: tl.constexpr, dest_head_stride: tl.constexpr,
    scale_batch_stride: tl.constexpr, scale_head_stride: tl.constexpr,
    BATCH: tl.constexpr, HEADS: tl.constexpr, GROUPS: tl.constexpr,
    DIM: tl.constexpr, TILE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * TILE + tl.arange(0, TILE)
    head, component = offsets // DIM, offsets % DIM
    # Find the rightmost sequence start, including empty ragged sequences.
    low, high = 0, BATCH
    while low + 1 < high:
        middle = (low + high) // 2
        start = tl.load(cu_lens + middle)
        at_or_after = token >= start
        low = tl.where(at_or_after, middle, low)
        high = tl.where(at_or_after, high, middle)
    mask = head < HEADS
    value = tl.load(
        source + token * source_token_stride + head * source_head_stride + component,
        mask, other=0,
    ).to(tl.float32)
    factor = tl.load(
        scale + low * scale_batch_stride + (head // GROUPS) * scale_head_stride,
        mask, other=1,
    )
    tl.store(
        destination + token * dest_token_stride + head * dest_head_stride + component,
        value * factor, mask,
    )


def _scale_strides(scale: torch.Tensor, batch: int, heads: int) -> tuple[int, int]:
    if scale.numel() == 1:
        return 0, 0
    if scale.ndim == 2 and scale.shape == (batch, heads):
        return scale.stride()
    raise ValueError("HCU E5M2 descales must be scalar or [batch, KV heads]")


def run_e5m2_attention(vendor: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Handle layer scales and broadcast strides without CPU tensor reads."""
    query, key = kwargs["q"], kwargs["k"]
    tokens, heads, dimension = query.shape
    batch = kwargs["cu_seqlens_q"].shape[0] - 1
    kv_heads = key.shape[1] if kwargs["layout"] == "bhsd" else key.shape[2]
    if (
        query.dtype not in (torch.bfloat16, torch.float16)
        or kwargs.get("q_descale") is not None
        or batch < 1 or heads % kv_heads or query.stride(-1) != 1
    ):
        raise ValueError("HCU E5M2 attention requires BF16/FP16 grouped queries")
    scales = []
    for name in ("k_descale", "v_descale"):
        scale = kwargs.get(name)
        if scale is not None:
            if scale.device != query.device or scale.dtype != torch.float32:
                raise ValueError("HCU E5M2 descales must be FP32 on the query device")
            _scale_strides(scale, batch, kv_heads)
        scales.append(scale)

    def multiply(source: torch.Tensor, destination: torch.Tensor, scale: torch.Tensor):
        _scale_rows[(tokens, triton.cdiv(heads * dimension, 1024))](
            source, destination, scale, kwargs["cu_seqlens_q"],
            source.stride(0), source.stride(1),
            destination.stride(0), destination.stride(1),
            *_scale_strides(scale, batch, kv_heads),
            BATCH=batch, HEADS=heads, GROUPS=heads // kv_heads,
            DIM=dimension, TILE=1024,
        )

    arguments = dict(kwargs)
    if scales[0] is not None:
        scaled = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        multiply(query, scaled, scales[0])
        arguments["q"] = scaled
    arguments.update(k_descale=None, v_descale=None)
    result = vendor(**arguments)
    output = result[0] if isinstance(result, tuple) else result
    if scales[1] is not None:
        if output.shape != query.shape or output.stride(-1) != 1:
            raise ValueError("HCU E5M2 attention received an incompatible output")
        multiply(output, output, scales[1])
    return result
