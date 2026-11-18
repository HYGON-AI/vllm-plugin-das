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


def _prefill(vendor: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    from vllm._custom_ops import cp_gather_cache

    query, key, value = (kwargs[name] for name in ("q", "k", "v"))
    if kwargs["layout"] == "bhsd":
        key, value = key.transpose(1, 2), value.transpose(1, 2)
    batch = kwargs["cu_seqlens_q"].shape[0] - 1
    capacity = batch * kwargs["max_seqlen_k"]
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


def run_e4m3_attention(vendor: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Retain quantized storage and native decode; share the descale adapter."""
    from hcu_vllm_das.attention.e5m2_scales import run_e5m2_attention

    def execute(**arguments: Any) -> Any:
        if arguments.get("block_table") is not None and arguments["max_seqlen_q"] > 1:
            return _prefill(vendor, arguments)
        return vendor(**arguments)

    return run_e5m2_attention(execute, kwargs)
