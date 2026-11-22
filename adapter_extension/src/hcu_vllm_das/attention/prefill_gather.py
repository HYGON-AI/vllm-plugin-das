"""Bounded BF16/FP16 prefill composition using existing official C++ gather.

No custom gather kernel. Single-token paged decode and FP8 stay on their
existing paths. Device lengths determine cumulative offsets inside Graph;
allocation capacity uses the official static maximum and batch shape only.
"""
from typing import Any, Callable

import torch

_MAX_CAPACITY = 262144


def prepare() -> None:
    from vllm._custom_ops import cp_gather_cache

    if not callable(cp_gather_cache) or not callable(
        torch.ops._C_cache_ops.cp_gather_cache
    ):
        raise RuntimeError("official cp_gather_cache ABI is unavailable")


def can_run(kwargs: dict[str, Any], layout: str) -> bool:
    q, k, v = (kwargs.get(name) for name in ("q", "k", "v"))
    cuq = kwargs.get("cu_seqlens_q")
    max_q, max_k = kwargs.get("max_seqlen_q"), kwargs.get("max_seqlen_k")
    return (
        layout in ("LBHNC", "LBNHC")
        and type(max_q) is int and max_q > 1
        and type(max_k) is int and max_k > 0
        and isinstance(cuq, torch.Tensor) and cuq.ndim == 1
        and 0 < (cuq.shape[0] - 1) * max_k <= _MAX_CAPACITY
        and isinstance(q, torch.Tensor) and q.ndim == 3
        and isinstance(kwargs.get("out"), torch.Tensor)
        and q.dtype in (torch.float16, torch.bfloat16)
        and isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor)
        and k.ndim == v.ndim == 4 and k.dtype == v.dtype == q.dtype
        and k.shape == v.shape and k.shape[-1] == q.shape[-1]
        and isinstance(kwargs.get("block_table"), torch.Tensor)
        and isinstance(kwargs.get("seqused_k"), torch.Tensor)
        and all(kwargs.get(name) is None for name in ("q_descale", "k_descale", "v_descale"))
    )


def run(vendor: Callable[..., Any], kwargs: dict[str, Any], layout: str) -> Any:
    from vllm._custom_ops import cp_gather_cache

    cuq = kwargs["cu_seqlens_q"]
    batch = cuq.shape[0] - 1
    k, v = kwargs["k"], kwargs["v"]
    if layout == "LBHNC":
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    capacity = batch * kwargs["max_seqlen_k"]
    gathered_k = torch.empty((capacity, k.shape[2], k.shape[3]), device=k.device, dtype=k.dtype)
    gathered_v = torch.empty_like(gathered_k)
    cuk = torch.zeros_like(cuq)
    torch.cumsum(kwargs["seqused_k"], dim=0, out=cuk[1:])
    # Each head slice has a contiguous page/content plane. The official C++
    # helper retains actual page and destination token strides.
    for head in range(k.shape[2]):
        cp_gather_cache(k[:, :, head, :], gathered_k[:, head, :], kwargs["block_table"], cuk, batch)
        cp_gather_cache(v[:, :, head, :], gathered_v[:, head, :], kwargs["block_table"], cuk, batch)
    arguments = dict(kwargs, k=gathered_k, v=gathered_v, cu_seqlens_k=cuk, fa_version=2)
    for name in ("block_table", "seqused_k", "layout"):
        arguments.pop(name, None)
    result = vendor(**arguments)
    # The installed public non-paged wrapper may allocate its own output.
    out = kwargs["out"]
    if result.data_ptr() != out.data_ptr():
        out.copy_(result)
    return out
