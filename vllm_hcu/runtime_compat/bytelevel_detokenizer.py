# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Conservative prompt context for the native ByteLevel DecodeStream."""
from tokenizers.decoders import ByteLevel


def choose_prompt_tail(
    tokenizer,
    prompt_token_ids: list[int] | None,
    skip_special_tokens: bool,
) -> list[int] | None:
    """Keep a clean nonempty suffix, or preserve the complete prompt.

    ByteLevel decoding is byte concatenation followed by UTF-8 decoding. An
    incomplete character or skipped-special-only tail must not lose context.
    Other decoder types can need unbounded context and are left unchanged.
    """
    if (
        not prompt_token_ids
        or len(prompt_token_ids) <= 32
        or type(tokenizer.decoder) is not ByteLevel
    ):
        return prompt_token_ids

    size = 32
    while size < len(prompt_token_ids):
        tail = prompt_token_ids[-size:]
        try:
            decoded = tokenizer.decode(tail, skip_special_tokens=skip_special_tokens)
        except (OverflowError, TypeError, ValueError):
            # Let the unchanged upstream path handle invalid input IDs.
            return prompt_token_ids
        if decoded and "\ufffd" not in decoded:
            return tail
        size *= 2
    return prompt_token_ids
