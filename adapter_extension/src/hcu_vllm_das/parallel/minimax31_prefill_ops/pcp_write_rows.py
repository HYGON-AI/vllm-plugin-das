"""Grouped main and index cache writes for PCP packed FP8 rows."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _write_fp8_rows(
    source, main_slots, index_slots, main, index, rows,
    row_stride: tl.constexpr, index_stride: tl.constexpr,
    tile_rows: tl.constexpr, kv_groups: tl.constexpr,
):
    row = tl.program_id(0) * tile_rows + tl.arange(0, tile_rows)
    tile = tl.program_id(1)
    column = tile * (128 * kv_groups) + tl.arange(0, 128 * kv_groups)
    main_slot = tl.load(main_slots + row, row < rows, -1)
    main_valid = (row < rows) & (main_slot >= 0)
    values = tl.load(source + row[:, None] * row_stride + column[None, :],
                     main_valid[:, None], 0)
    main_col = (column // 128 % 4) * 256 + (column // 512) * 128 + column % 128
    main_addr = main_slot[:, None] * 1024 + main_col[None, :]
    tl.store(main + main_addr, values, main_valid[:, None])
    if tile == 0:
        index_slot = tl.load(index_slots + row, row < rows, -1)
        valid = (row < rows) & (index_slot >= 0)
        d = tl.arange(0, 128)
        index_data = tl.load(source + row[:, None] * row_stride + 1024 + d[None, :],
                             valid[:, None], 0)
        base = index_slot // 128 * index_stride + index_slot % 128 * 128
        tl.store(index + base[:, None] + d[None, :], index_data, valid[:, None])


def supports_layout(rows, count, slots, index_slots, cache, index):
    """Validate before the collective so unsupported shapes keep the BF16 path."""
    if index is None or index_slots is None:
        return False
    return (
        rows.is_cuda and cache.ndim == 4 and cache.shape[1:] == (128, 4, 128)
        and cache.dtype in (torch.uint8, torch.float8_e4m3fn)
        and cache.stride() == (131072, 1024, 256, 1)
        and index.ndim == 3 and index.shape[1:] == (128, 128)
        and index.stride()[1:] == (128, 1) and index.dtype == torch.float8_e4m3fn
        and slots.is_contiguous() and index_slots.is_contiguous()
        and slots.dtype == index_slots.dtype == torch.int64
        and slots.shape == index_slots.shape == (count,)
        and rows.device == slots.device == index_slots.device == cache.device == index.device
    )


def supports(rows, slots, index_slots, cache, index):
    return (
        rows.ndim == 2 and rows.shape[1] == 1152 and rows.stride(1) == 1
        and rows.dtype == torch.float8_e4m3fn
        and supports_layout(rows, rows.shape[0], slots, index_slots, cache, index)
    )


def write(rows, slots, index_slots, cache, index, tile_rows, kv_groups, warps):
    if not supports(rows, slots, index_slots, cache, index):
        raise ValueError("The grouped writer requires the packed FP8 PCP cache format")
    _write_fp8_rows[(triton.cdiv(rows.shape[0], tile_rows), 8 // kv_groups)](
        rows.view(torch.uint8), slots, index_slots, cache.view(torch.uint8), index.view(torch.uint8),
        rows.shape[0], rows.stride(0), index.stride(0), tile_rows, kv_groups, num_warps=warps,
    )
