"""Prepare native FlashMLA SWA and compressed indices in one launch."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=[
    "swa_stride", "lens_stride", "topk_stride", "table_stride", "table_rows", "table_width",
])
def _prepare_decode_indices_kernel(
    swa_indices, swa_lens, main_output,
    topk_indices, token_to_req, block_table, valid_tokens,
    extra_output, extra_lens,
    swa_stride, lens_stride,
    topk_stride, table_stride,
    swa_width: tl.constexpr, topk_width: tl.constexpr,
    table_rows, table_width,
    page_size: tl.constexpr, has_extra: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    main_length = tl.load(swa_lens + row * lens_stride)
    main_slots = tl.load(
        swa_indices + row * swa_stride + columns,
        mask=(columns < swa_width) & (columns < main_length), other=-1,
    )
    tl.store(main_output + row * swa_width + columns, main_slots, columns < swa_width)
    if has_extra:
        row_valid = tl.load(valid_tokens + row)
        request = tl.load(token_to_req + row)
        local = tl.load(
            topk_indices + row * topk_stride + columns,
            mask=(columns < topk_width) & row_valid, other=-1,
        )
        logical_block = local // page_size
        mapped_block = tl.load(
            block_table + request * table_stride + logical_block,
            mask=(columns < topk_width) & row_valid & (local >= 0)
            & (logical_block < table_width) & (request >= 0) & (request < table_rows),
            other=-1,
        )
        physical = mapped_block * page_size + local % page_size
        keep = (columns < topk_width) & (local >= 0) & (mapped_block >= 0) & row_valid
        offsets = tl.cumsum(keep.to(tl.int32)) - 1
        count = tl.sum(keep.to(tl.int32), 0)
        tl.store(extra_output + row * topk_width + offsets, physical, keep)
        tl.store(
            extra_output + row * topk_width + columns, -1,
            (columns < topk_width) & (columns >= count),
        )
        tl.store(extra_lens + row, count)


def prepare_decode_indices(
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    topk_indices: torch.Tensor | None = None,
    token_to_req_indices: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    page_size: int = 1,
    is_valid_token: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    rows, swa_width = swa_indices.shape[0], swa_indices.shape[-1]
    has_extra = topk_indices is not None
    topk_width = topk_indices.shape[-1] if has_extra else 0
    main_output = torch.empty((rows, 1, swa_width), device=swa_indices.device, dtype=torch.int32)
    extra_output = torch.empty(
        (rows, 1, topk_width), device=swa_indices.device, dtype=torch.int32,
    ) if has_extra else None
    extra_lens = torch.empty((rows,), device=swa_indices.device, dtype=torch.int32) if has_extra else None
    _prepare_decode_indices_kernel[(rows,)](
        swa_indices, swa_lens, main_output,
        topk_indices, token_to_req_indices, block_table, is_valid_token,
        extra_output, extra_lens,
        swa_indices.stride(0), swa_lens.stride(0),
        topk_indices.stride(0) if has_extra else 0,
        block_table.stride(0) if has_extra else 0,
        swa_width, topk_width,
        block_table.shape[0] if has_extra else 0,
        block_table.shape[1] if has_extra else 0,
        page_size, has_extra, triton.next_power_of_2(max(swa_width, topk_width)),
        num_warps=4,
    )
    return main_output, extra_output, extra_lens
