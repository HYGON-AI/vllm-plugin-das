# SPDX-License-Identifier: Apache-2.0
"""Reuse official uniform request rows without inspecting device values."""
from __future__ import annotations


def uniform_decode_rows(meta, *, topk_tokens):
    """Return a proven request width, otherwise preserve per-token scoring.

    Core records uniformity before flattening query rows. Divisibility alone
    cannot prove ownership for ragged, padded or PCP-remapped metadata.
    Only the native bounded TopK consumer is admitted; its other K values
    use the unchanged Torch selection path, whose tie membership can vary.
    """
    decode = meta.decode
    width = getattr(decode, 'write_max_decode_len', 0)
    requests = meta.num_decodes
    rows = meta.num_decode_tokens
    if (topk_tokens not in (512, 1024, 2048)
            or type(width) is not int or width <= 1
            or type(requests) is not int or requests <= 0
            or type(rows) is not int or rows != requests * width
            or getattr(decode, 'decode_is_uniform', False) is not True
            or decode.requires_padding
            or getattr(decode, 'indices', None) is not None
            or decode.seq_lens.shape not in ((rows,), (rows, 1))
            or decode.block_table.ndim != 2 or decode.block_table.shape[0] != rows):
        return 1
    return width


def stage_request_rows(cache, table, lengths, width, gather, *,
                       rows_per_request, page_records, stage):
    """Stage each proven request once, preserving per-row causal lengths.

    Uniform rows are consecutive and their compressed lengths are monotone.
    The final row's table/length therefore contains the shared visible prefix.
    Scoring still uses each original row's length. Canonical pages already
    share physical addresses and need no additional materialization.
    """
    if rows_per_request == 1 or page_records == 64:
        return stage(cache, table, lengths, width, gather, live_only=True)
    last = rows_per_request - 1
    pool, per_request = stage(
        cache, table[last::rows_per_request],
        lengths[last::rows_per_request].contiguous(),
        width, gather, live_only=True)
    return pool, per_request.repeat_interleave(rows_per_request, dim=0)
