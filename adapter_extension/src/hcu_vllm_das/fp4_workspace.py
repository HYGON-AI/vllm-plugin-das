# SPDX-License-Identifier: Apache-2.0
"""Tensor-lifetime budget for the LightOp reader (not an allocator limit).

The vendor's private scratch is not measurable here. Device peak validation
is required before treating this estimate as a complete memory guarantee.
"""


# Bound per-row selection scratch even when the gathered K window is short.
# This is the row capacity of the widest dense LightOp Top512 input.
_PREFILL_CHUNK_WIDTH_FLOOR = 32768


def query_chunk_rows(max_model_len: int, max_logits_bytes: int) -> int:
    if max_model_len <= 0 or max_logits_bytes <= 0:
        raise ValueError('FP4 score width and logits budget must be positive')
    width = (max_model_len + 3) // 4 * 4
    return max(1, max_logits_bytes // (4 * width))


def prefill_chunk_rows(max_model_len: int, live_width: int, max_logits_bytes: int) -> int:
    width_floor = min(max_model_len, _PREFILL_CHUNK_WIDTH_FLOOR)
    return query_chunk_rows(max(live_width, width_floor), max_logits_bytes)


def workspace_bytes(*, query_rows: int, decode_rows: int, max_model_len: int,
                    total_seq_lens: int, topk_tokens: int, num_query_heads: int,
                    max_logits_bytes: int, physical_page_size: int,
                    persistent_staging: bool = False) -> int:
    """Bound the explicit gather, selection and optional decode-staging peaks.

    Prefill and decode execute sequentially and release temporaries between
    phases. The caller freezes decode capacity from the scheduler at model
    construction, independently of the dummy prefill token count.
    """
    rows = min(query_rows, prefill_chunk_rows(max_model_len, 0, max_logits_bytes))
    if min(query_rows, decode_rows, total_seq_lens, topk_tokens) < 0:
        raise ValueError('FP4 workspace extents cannot be negative')
    if physical_page_size not in (32, 64, 128):
        raise ValueError('FP4 workspace requires allocator page geometry')
    if query_rows == 0:
        return 0
    decode_rows = min(query_rows, decode_rows)
    width = (max_model_len + 3) // 4 * 4
    decode_width = (max_model_len + 255) // 256 * 256
    staging_width = (max_model_len + 63) // 64 * 64

    # Slot building is released before gather; slots are released before
    # scoring. 76 B/record covers gathered payload/scales plus slot indices.
    # Runtime can trade a narrower matrix for more query rows. Bound the
    # matrix by the logits budget, allowing one row even if it exceeds it.
    prefill_elements = min(rows * width, max(width, max_logits_bytes // 4))
    prefill = total_seq_lens * 76 + prefill_elements * 10 + width * 8
    prefill += rows * (topk_tokens * 96 + num_query_heads * 128 * 2)
    staging = pool = 0
    if physical_page_size != 64 and decode_rows:
        staging = decode_rows * staging_width * (2 * 68 + 3 * 8 + 1)
        staging += staging_width * 40 + decode_rows * (staging_width // 64) * 4
        pool = decode_rows * staging_width * 68
    if persistent_staging:
        staging -= pool
        pool = 0
    decode = pool + decode_rows * (decode_width * 10 + topk_tokens * 96)
    decode += decode_width * 8
    decode += decode_rows * num_query_heads * 68
    return max(prefill, staging, decode)
