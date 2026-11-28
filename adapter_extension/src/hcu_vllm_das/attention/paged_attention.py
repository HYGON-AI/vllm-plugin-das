"""Adapt dense-within-page KV views to the vendor's contiguous PA ABI.

Core owns physical pages and their writer views. For packed [K0,V0,K1,V1]
storage, expose dense read-only K=[K0,V0,K1] and V=[V0,K1,V1] views and
double the block indices. Each real K/V page keeps its original address;
neither the allocator nor the cache contents change. Only the small block
table and noncontiguous query/metadata need materialization during decode.
"""

from collections.abc import Callable
from math import prod
from typing import Any

import torch


def _page_view(cache: torch.Tensor) -> tuple[torch.Tensor, int | None]:
    if cache.is_contiguous():
        return cache, 1
    if cache.ndim != 4 or cache.shape[0] == 0 or not cache[0].is_contiguous():
        raise ValueError("HCU PA requires dense contents within each KV page")
    page_elements = prod(cache.shape[1:])
    if cache.stride(0) < page_elements:
        raise ValueError("HCU PA KV pages overlap")
    if cache.stride(0) % page_elements:
        return cache, None
    ratio = cache.stride(0) // page_elements
    # Stop at the end of the final real page, including for an offset V view.
    # Expanding to num_blocks * ratio would overrun the final allocation.
    shape = ((cache.shape[0] - 1) * ratio + 1, *cache.shape[1:])
    view = torch.as_strided(cache, shape, (page_elements, *cache.stride()[1:]))
    return view, ratio


def run_paged_attention(vendor: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    # The Plugin bumps rectangular short-prefill's upper bound before this
    # seam. Restrict normalization to ordinary single-token paged decode.
    if kwargs.get("block_table") is None or kwargs.get("max_seqlen_q") != 1:
        return vendor(**kwargs)
    arguments = dict(kwargs)
    key, kratio = _page_view(arguments["k"])
    value, vratio = _page_view(arguments["v"])
    if kratio != vratio:
        raise ValueError("HCU PA requires K and V to have the same page stride")
    arguments.update(k=key, v=value)
    table = arguments["block_table"]
    if kratio is not None and kratio != 1:
        table = table * kratio
    # Core's gather also assumes dense table columns on the padding fallback.
    arguments["block_table"] = table.contiguous()
    for name in ("q", "seqused_k", "cu_seqlens_q", "alibi_slopes",
                 "q_descale", "k_descale", "v_descale"):
        tensor = arguments.get(name)
        if tensor is not None:
            arguments[name] = tensor.contiguous()
    output = arguments.get("out")
    if output is not None and not output.is_contiguous():
        arguments["out"] = torch.empty_like(output, memory_format=torch.contiguous_format)
    if kratio is None:
        # A subpage padding gap cannot be represented by integer PA page IDs.
        # Gather only the active contexts with the existing bounded native
        # composition; never materialize the complete allocated KV cache.
        from hcu_vllm_das.attention.e4m3_attention import _prefill

        if output is None:
            arguments["out"] = torch.empty_like(
                arguments["q"], memory_format=torch.contiguous_format)
        result = _prefill(vendor, arguments)
    else:
        result = vendor(**arguments)
    if output is not None and arguments["out"] is not output:
        output.copy_(arguments["out"])
        return output
    return result
