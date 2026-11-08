# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Correct sparse-attention flat views over non-affine paged KV storage.

The sparse kernels consume affine [physical-token, head, channel] arrays.
Head-major pages cannot flatten block and token axes without copying. Keep
the real paged cache for transport and materialize only referenced pages in
a shared FP8 read workspace. HCU eager DBO restores each microbatch's metadata
and orders attention on its compute stream before the next MoE handoff.
"""
from __future__ import annotations

import os
from typing import Any

import torch

_FLAT_BUFFERS: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _read_blocks(metadata: Any, blocks: int, page: int, device: torch.device):
    if metadata is None:
        return torch.arange(blocks, dtype=torch.long, device=device)
    key = (device, blocks, page)
    cached = getattr(metadata, "_step5_flat_read_blocks", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    host_lengths = getattr(metadata, "seq_lens_cpu", None)
    lengths = (host_lengths if host_lengths is not None else metadata.seq_lens.detach().cpu()).tolist()
    table = getattr(metadata, "block_table_np", None)
    if table is None:
        table = metadata.block_table.detach().cpu().tolist()
    rows = int(getattr(metadata, "num_decodes", 0) + getattr(metadata, "num_prefills", 0))
    if not rows:
        rows = len(lengths)
    used = set()
    for row, length in enumerate(lengths[:rows]):
        count = (int(length) + page - 1) // page
        used.update(int(x) for x in table[row][:count])
    if used and (min(used) < 0 or max(used) >= blocks):
        raise ValueError("Sparse KV read references a block outside its cache")
    result = torch.tensor(sorted(used), dtype=torch.long, device=device)
    metadata._step5_flat_read_blocks = (key, result)
    return result


def write_paged_kv_and_get_flat(
    cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    metadata: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache.ndim != 4 or cache.shape[-1] % 2 or cache.stride(-1) != 1:
        raise ValueError("Paged sparse KV requires logical [B,H,N,2D] storage")
    if cache.is_cuda and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("Non-affine sparse KV read bridge requires eager execution")
    if k_scale <= 0 or v_scale <= 0:
        raise ValueError("KV quantization scales must be positive")
    base = cache.view(torch.float8_e4m3fn) if cache.dtype == torch.uint8 else cache
    blocks, heads, page, content = base.shape
    dim = content // 2
    if key.shape[1:] != (heads, dim) or value.shape != key.shape:
        raise ValueError("Paged KV input dimensions differ from the cache")
    slots = slot_mapping[:key.shape[0]].long()
    valid = slots >= 0
    positions = slots[valid]
    block_ids, token_ids = positions // page, positions % page
    if base.dtype == torch.float8_e4m3fn:
        # Match the existing sparse writer: divide in FP32, saturate, then cast.
        keys = (key[valid].float() / k_scale).clamp(-448, 448).to(base.dtype)
        values = (value[valid].float() / v_scale).clamp(-448, 448).to(base.dtype)
        raw = base.view(torch.uint8)
        raw[block_ids, :, token_ids, :dim] = keys.contiguous().view(torch.uint8)
        raw[block_ids, :, token_ids, dim:] = values.contiguous().view(torch.uint8)
    else:
        base[block_ids, :, token_ids, :dim] = key[valid].to(base.dtype)
        base[block_ids, :, token_ids, dim:] = value[valid].to(base.dtype)

    # HCU DBO consumes these views within the current attention call, before
    # yielding at MoE. Reuse on that ordered stream is safe; distinct streams
    # need independent storage. The connector validates producer DBO ownership.
    stream = torch.cuda.current_stream(base.device).cuda_stream if base.is_cuda else 0
    workspace_key = (base.device, base.dtype, blocks, heads, page, dim, stream)
    if workspace_key not in _FLAT_BUFFERS:
        shape = (blocks, page, heads, dim)
        _FLAT_BUFFERS[workspace_key] = (
            torch.empty(shape, dtype=base.dtype, device=base.device),
            torch.empty(shape, dtype=base.dtype, device=base.device),
        )
    flat_k, flat_v = _FLAT_BUFFERS[workspace_key]
    used = _read_blocks(metadata, blocks, page, base.device)
    if (
        os.environ.get("HCU_STEP5_EP8_SPARSE_GROUP_MAJOR") == "1"
        and base.is_cuda and base.dtype == torch.float8_e4m3fn
        and heads == 4 and dim == 192
        and getattr(metadata, "num_prefills", 0) > 0
        and getattr(metadata, "num_decodes", 0) == 0
    ):
        from vllm.logger import init_logger
        from .ep8_sparse_group_major import copy

        init_logger("vllm.hcu.step5_paged_kv").info_once(
            "Step5 EP8 group-major sparse KV workspace active", scope="process"
        )
        return copy(base, used, (flat_k.view(torch.uint8), flat_v.view(torch.uint8)))
    k_pages = base[..., :dim].transpose(1, 2)
    v_pages = base[..., dim:].transpose(1, 2)
    # Copy bytes so FP8 representations are preserved exactly and the path
    # does not depend on index_copy support for a float8 arithmetic dtype.
    flat_k.view(torch.uint8).index_copy_(0, used, k_pages.view(torch.uint8).index_select(0, used))
    flat_v.view(torch.uint8).index_copy_(0, used, v_pages.view(torch.uint8).index_select(0, used))
    return flat_k.view(blocks * page, heads, dim), flat_v.view(blocks * page, heads, dim)
