# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-owned Engram lookup and storage checks.

The byte-copy kernel follows Core033 qwen4_exp/common/ngram_embedding.py's
pinned-host row lookup. Keeping this small closure local avoids importing an
unrelated model implementation. CPU lookup is an explicit eager reference
path, not evidence for UVA, GPU hashing, or graph capture correctness.
"""

import torch

from vllm.triton_utils import tl, triton


def get_engram_cpu_offload(storage, *, additional_config=None) -> bool:
    option = "hyv4_flash_engram_cpu_offload"
    selected = None
    if additional_config is not None and option in additional_config:
        selected = additional_config[option]
        if type(selected) is not bool:
            raise ValueError(f"{option} must be a boolean")
    if storage is None:
        return selected is True
    if any(getattr(storage, name, False) for name in (
        "disk_offload", "embedding_across_dp", "dp_shared_memory",
    )):
        raise ValueError("Flash Engram supports TP-local tables, not DP/disk offload")
    configured = bool(getattr(storage, "cpu_offload", False))
    if selected is not None and selected != configured:
        raise ValueError(f"{option} conflicts with engram_config.cpu_offload")
    return configured


@triton.jit
def _lookup_rows_kernel(
    weight_ptr, ids_ptr, output_ptr, row_bytes, vocab_start, vocab_end,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    index = tl.load(ids_ptr + row)
    valid = (index >= vocab_start) & (index < vocab_end)
    local = tl.where(valid, index - vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    values = tl.load(
        weight_ptr + local * row_bytes + offsets,
        mask=(offsets < row_bytes) & valid, other=0,
    )
    tl.store(output_ptr + row * row_bytes + offsets, values, mask=offsets < row_bytes)


def lookup_rows(weight, indices, output, vocab_start, vocab_end):
    """Gather whole local rows; invalid/unowned ids produce all-zero bytes."""
    if not (weight.is_contiguous() and indices.is_contiguous()
            and output.is_contiguous()):
        raise ValueError("Flash Engram lookup requires contiguous storage")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Flash Engram lookup requires integer indices")
    if weight.ndim not in (1, 2) or weight.shape[0] != vocab_end - vocab_start:
        raise ValueError("Flash Engram lookup table does not match TP row bounds")
    if output.shape != (*indices.shape, *weight.shape[1:]):
        raise ValueError("Flash Engram lookup output shape mismatch")
    if output.dtype != weight.dtype or output.device != indices.device:
        raise ValueError("Flash Engram lookup output dtype/device mismatch")
    if indices.numel() == 0:
        return
    row_bytes = weight.element_size() * (weight.shape[1] if weight.ndim == 2 else 1)
    if indices.device.type == "cpu":
        if weight.device.type != "cpu":
            raise ValueError("CPU Engram reference lookup requires CPU table storage")
        # Byte gather works even when CPU lacks FP8 indexing arithmetic.
        local = (indices - vocab_start).flatten()
        valid = (indices.flatten() >= vocab_start) & (indices.flatten() < vocab_end)
        dst = output.view(torch.uint8).reshape(-1, row_bytes)
        dst.zero_()
        dst[valid] = weight.view(torch.uint8).reshape(-1, row_bytes)[local[valid]]
        return
    if weight.device != indices.device:
        raise ValueError("Engram GPU lookup requires a device table or an UVA view")
    _lookup_rows_kernel[(indices.numel(),)](
        weight.view(torch.uint8), indices, output.view(torch.uint8),
        row_bytes, vocab_start, vocab_end,
        BLOCK_D=triton.next_power_of_2(row_bytes),
    )
