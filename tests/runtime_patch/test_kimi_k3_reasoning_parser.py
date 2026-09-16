# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from vllm_hcu.runtime_compat.kimi_k3_reasoning_parser import KimiK3ReasoningParser


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return {
            "<|open|>think<|sep|>": [1, 2, 3],
            "<|close|>think<|sep|>": [4, 2, 3],
        }[text]


class _Request:
    tools = None
    tool_choice = "none"
    skip_special_tokens = True
    spaces_between_special_tokens = True


def test_k3_instruct_parser_discards_leaked_think_channel_and_xtml_wrappers():
    parser = KimiK3ReasoningParser(
        _Tokenizer(), chat_template_kwargs={"thinking": False}
    )
    generated = (
        "analysis that must not reach the caller"
        "<|close|>think<|sep|><|open|>response<|sep|>"
        "def answer():\n    return 42\n"
        "<|close|>response<|sep|><|close|>message<|sep|>"
    )

    reasoning, content = parser.extract_reasoning(generated, _Request())

    assert reasoning is None
    assert content == "def answer():\n    return 42\n"
    assert "<|" not in content


def test_k3_thinking_parser_splits_reasoning_and_response():
    parser = KimiK3ReasoningParser(_Tokenizer(), chat_template_kwargs={"thinking": True})
    generated = (
        "<|open|>think<|sep|>reasoning"
        "<|close|>think<|sep|><|open|>response<|sep|>answer"
        "<|close|>response<|sep|><|close|>message<|sep|>"
    )

    reasoning, content = parser.extract_reasoning(generated, _Request())

    assert reasoning == "reasoning"
    assert content == "answer"
