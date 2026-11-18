"""E4M3 paged decode and native byte gather for BF16/FP16 prefill.

The vendor paged-prefill wrapper casts Q to the cache dtype. Its E4M3
kernel only implements head size 128; its mixed-dtype gather does not
preserve E4M3 values. Use Core's byte-preserving gather and Torch's explicit
E4M3 conversion before the vendor's ordinary unquantized varlen attention.
Allocation uses static metadata; cumulative lengths stay on the device.
"""

from collections.abc import Callable
from typing import Any

import torch

# Target the combined FP8 gathers and BF16/FP16 conversions. A single
# context can exceed this target, but batch size must never multiply it.
_PREFILL_SCRATCH_TARGET_BYTES = 64 * 1024**2
# The vendor's varlen tiles can load masked V lanes past the final real
# token, including during changed-length Graph replay. Give its largest
# 256-token tile initialized backing storage; masked NaNs still taint PV.
_VARLEN_KV_TILE = 256


def _padded_max_k(max_k: int) -> int:
    return ((max_k + _VARLEN_KV_TILE - 1) // _VARLEN_KV_TILE) * _VARLEN_KV_TILE


def _prefill_chunk(vendor: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    from vllm._custom_ops import cp_gather_cache

    query, key, value = (kwargs[name] for name in ("q", "k", "v"))
    if kwargs["layout"] == "bhsd":
        key, value = key.transpose(1, 2), value.transpose(1, 2)
    batch = kwargs["cu_seqlens_q"].shape[0] - 1
    max_k = kwargs["max_seqlen_k"]
    capacity = (batch - 1) * max_k + _padded_max_k(max_k)
    shape = (capacity, key.shape[2], key.shape[3])
    # Initialize unused capacity so conversion never consumes indeterminate
    # FP8 encodings. Attention still consumes only the device-side lengths.
    gathered_k = torch.zeros(shape, dtype=key.dtype, device=key.device)
    gathered_v = torch.zeros_like(gathered_k)
    cumulative = torch.zeros_like(kwargs["cu_seqlens_q"])
    torch.cumsum(kwargs["seqused_k"], dim=0, out=cumulative[1:])
    for head in range(key.shape[2]):
        cp_gather_cache(key[:, :, head, :], gathered_k[:, head, :],
                        kwargs["block_table"], cumulative, batch)
        cp_gather_cache(value[:, :, head, :], gathered_v[:, head, :],
                        kwargs["block_table"], cumulative, batch)
    arguments = dict(kwargs, k=gathered_k.to(query.dtype),
                     v=gathered_v.to(query.dtype), cu_seqlens_k=cumulative)
    for name in ("block_table", "seqused_k", "layout"):
        arguments.pop(name, None)
    wants_lse = arguments.pop("return_softmax_lse", False)
    if wants_lse:
        arguments["return_attn_probs"] = True
    result = vendor(**arguments)
    produced = result[0] if isinstance(result, tuple) else result
    output = kwargs["out"]
    if produced.data_ptr() != output.data_ptr():
        output.copy_(produced)
    if isinstance(result, tuple):
        return (output, *result[1:])
    return output


def _prefill(vendor: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Microbatch sequences when their static gather bound is too large.

    KV scratch is bounded by max(64 MiB, one maximum-length context), plus
    O(query tokens) compaction/scatter buffers. All query offsets remain on
    device, including under changed-length Graph replay. No CPU length read
    or special empty-KV vendor contract is needed.
    """
    query, key = kwargs["q"], kwargs["k"]
    cuq = kwargs["cu_seqlens_q"]
    batch = cuq.shape[0] - 1
    kv_heads = key.shape[1] if kwargs["layout"] == "bhsd" else key.shape[2]
    per_sequence = (_padded_max_k(kwargs["max_seqlen_k"]) * kv_heads * key.shape[-1]
                    * (2 * key.element_size() + 2 * query.element_size()))
    chunk_size = max(1, _PREFILL_SCRATCH_TARGET_BYTES // max(1, per_sequence))
    if batch <= chunk_size:
        return _prefill_chunk(vendor, kwargs)

    output = kwargs["out"]
    positions = torch.arange(query.shape[0], device=query.device)
    lse, auxiliary = None, ()
    for begin in range(0, batch, chunk_size):
        end = min(begin + chunk_size, batch)
        # Static capacity may include padding; the rebased cumulative lengths
        # identify only actual queries. Clamping protects the ignored tail.
        capacity = min(query.shape[0], (end - begin) * kwargs["max_seqlen_q"])
        indices = (torch.arange(capacity, device=query.device) + cuq[begin]).clamp(
            max=query.shape[0] - 1)
        compact_q = query.index_select(0, indices)
        arguments = dict(kwargs, q=compact_q, out=torch.empty_like(compact_q),
            cu_seqlens_q=cuq[begin:end + 1] - cuq[begin],
            seqused_k=kwargs["seqused_k"][begin:end],
            block_table=kwargs["block_table"][begin:end])
        slopes = kwargs.get("alibi_slopes")
        if slopes is not None and slopes.ndim == 2:
            arguments["alibi_slopes"] = slopes[begin:end]
        result = _prefill_chunk(vendor, arguments)
        produced = result[0] if isinstance(result, tuple) else result
        belongs = (positions >= cuq[begin]) & (positions < cuq[end])
        relative = (positions - cuq[begin]).clamp(0, capacity - 1)
        torch.where(belongs[:, None, None], produced.index_select(0, relative),
                    output, out=output)
        if isinstance(result, tuple):
            if lse is None:
                lse = torch.empty((result[1].shape[0], query.shape[0]),
                                  device=result[1].device, dtype=result[1].dtype)
            torch.where(belongs[None, :], result[1].index_select(1, relative),
                        lse, out=lse)
            # Inference requests LSE, not dropout probabilities. The vendor
            # returns an empty probability tensor at dropout=0; never silently
            # return one chunk's nonempty auxiliary as a full-batch result.
            auxiliary = result[2:]
            if any(isinstance(x, torch.Tensor) and x.numel() for x in auxiliary):
                raise NotImplementedError("chunked E4M3 prefill requires empty attention-probability output")
    return (output, lse, *auxiliary) if lse is not None else output


def run_e4m3_attention(vendor: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Retain quantized storage and native decode; share the descale adapter."""
    from hcu_vllm_das.attention.e5m2_scales import run_e5m2_attention

    def execute(**arguments: Any) -> Any:
        # Paged single-token decode's fast path drops requested LSE. The
        # byte-gather/ordinary-varlen path retains BF16/FP16 Q and exposes LSE
        # via return_attn_probs, including for one query per sequence.
        if arguments.get("block_table") is not None and (
            arguments["max_seqlen_q"] > 1
            or arguments.get("return_softmax_lse", False)
        ):
            return _prefill(vendor, arguments)
        return vendor(**arguments)

    return run_e5m2_attention(execute, kwargs)
