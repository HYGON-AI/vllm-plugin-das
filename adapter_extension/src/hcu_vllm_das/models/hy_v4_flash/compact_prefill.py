# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Capacity bounds for the gfx936 sparse prefill workspace."""

from itertools import accumulate


def compact_prefill_offsets(
    seq_lens: list[int], num_prefill_tokens: int
) -> list[int] | None:
    if num_prefill_tokens < 1024 or not seq_lens or min(seq_lens) <= 0:
        return None
    offsets = list(accumulate(seq_lens, initial=0))
    # Full-context dequantization is bounded even when chunked prefill admits
    # long sequences. Short extends over a long prefix use mixed FP8 instead.
    if offsets[-1] > min(16384, 2 * num_prefill_tokens):
        return None
    return offsets
