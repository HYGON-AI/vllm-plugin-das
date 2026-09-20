# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Kimi K3 XTML reasoning parser owned by the HCU plugin."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def _last_subsequence_index(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """Find the final occurrence of a token-id marker sequence."""
    length = len(needle)
    if not length:
        return -1
    for index in range(len(haystack) - length, -1, -1):
        if list(haystack[index : index + length]) == list(needle):
            return index
    return -1


class KimiK3ReasoningParser(ReasoningParser):
    """Separate K3's XTML think and response channels for OpenAI serving."""

    _THINK_OPEN = "<|open|>think<|sep|>"
    _THINK_CLOSE = "<|close|>think<|sep|>"
    _RESPONSE_OPEN = "<|open|>response<|sep|>"
    _RESPONSE_CLOSE = "<|close|>response<|sep|>"
    _MESSAGE_CLOSE = "<|close|>message<|sep|>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs) -> None:
        super().__init__(tokenizer)
        if not self.model_tokenizer:
            raise ValueError("Kimi K3 reasoning parsing requires the model tokenizer")

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking")
        if thinking is None:
            thinking = chat_kwargs.get("enable_thinking", True)
        self._thinking_enabled = bool(thinking)

        open_marker = r"<\|open\|>"
        close_marker = r"<\|close\|>"
        sep_marker = r"<\|sep\|>"
        self._think_open_re = re.compile(open_marker + r"\s*think\s*" + sep_marker)
        self._think_close_re = re.compile(close_marker + r"\s*think\s*" + sep_marker)
        self._response_open_re = re.compile(
            open_marker + r"\s*response\s*" + sep_marker
        )
        self._response_close_re = re.compile(
            close_marker + r"\s*response\s*" + sep_marker
        )
        self._message_close_re = re.compile(
            close_marker + r"\s*message\s*" + sep_marker
        )
        self._think_open_ids = tokenizer.encode(self._THINK_OPEN, add_special_tokens=False)
        self._think_close_ids = tokenizer.encode(
            self._THINK_CLOSE, add_special_tokens=False
        )
        self._last_streaming_delta_ids: tuple[int, ...] | None = None
        self._last_streaming_content_ids: list[int] | None = None
        self._content_pending = ""
        self._content_started = False
        self._content_prompt_checked = False
        self._content_closed = False
        self._content_wrapper_re = re.compile(
            "(?:" + "|".join((
                self._response_open_re.pattern,
                self._response_close_re.pattern,
                self._message_close_re.pattern,
            )) + ")"
        )

    @property
    def reasoning_start_str(self) -> str:
        return self._THINK_OPEN

    @property
    def reasoning_end_str(self) -> str:
        return self._THINK_CLOSE

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        request.skip_special_tokens = False
        if hasattr(request, "spaces_between_special_tokens"):
            request.spaces_between_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if not self._thinking_enabled:
            return True
        last_close = _last_subsequence_index(input_ids, self._think_close_ids)
        last_open = _last_subsequence_index(input_ids, self._think_open_ids)
        return last_close != -1 if last_open == -1 else last_close > last_open

    def _extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self._thinking_enabled:
            return input_ids
        index = _last_subsequence_index(input_ids, self._think_close_ids)
        if index == -1:
            return []
        return input_ids[index + len(self._think_close_ids) :]

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        cached_ids = self._last_streaming_delta_ids
        cached_content = self._last_streaming_content_ids
        self._last_streaming_delta_ids = None
        self._last_streaming_content_ids = None
        if cached_ids == tuple(input_ids) and cached_content is not None:
            return cached_content
        return self._extract_content_ids(input_ids)

    @staticmethod
    def _preserve_tool_channels(request: "ChatCompletionRequest | ResponsesRequest") -> bool:
        return bool(getattr(request, "tools", None)) and getattr(
            request, "tool_choice", None
        ) != "none"

    def _strip_content_wrapper(self, text: str) -> str:
        response_open = self._response_open_re.search(text)
        response_close = self._response_close_re.search(
            text, response_open.end() if response_open else 0
        )
        if response_open is not None and response_close is not None:
            text = text[response_open.end() : response_close.start()]
        elif response_open is not None:
            text = text[response_open.end() :]
        else:
            text = self._response_open_re.sub("", text)
            text = self._response_close_re.sub("", text)
        return self._message_close_re.sub("", text)

    def _content_after_reasoning(
        self, text: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> str | None:
        if self._preserve_tool_channels(request):
            return text or None
        return self._strip_content_wrapper(text) or None

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        if not self._thinking_enabled:
            return None, self._content_after_reasoning(model_output, request)

        open_match = self._think_open_re.search(model_output)
        content_start = open_match.end() if open_match is not None else 0
        if open_match is None and self._think_close_re.search(model_output) is None:
            return None, self._content_after_reasoning(model_output, request)

        close_match = self._think_close_re.search(model_output, content_start)
        if close_match is None:
            return model_output[content_start:] or None, None
        reasoning = model_output[content_start : close_match.start()]
        content = self._content_after_reasoning(model_output[close_match.end() :], request)
        return reasoning or None, content

    @staticmethod
    def _without_partial_marker(text: str, markers: Sequence[str]) -> str:
        overlap = 0
        for marker in markers:
            for length in range(min(len(marker) - 1, len(text)), 0, -1):
                if text.endswith(marker[:length]):
                    overlap = max(overlap, length)
                    break
        return text[:-overlap] if overlap else text

    def initialize_content_from_prompt(
        self, prompt_token_ids: Sequence[int] | None,
    ) -> None:
        """Recognize the response opener consumed by K3's native prompt."""
        if self._content_prompt_checked or prompt_token_ids is None:
            return
        self._content_prompt_checked = True
        if not prompt_token_ids or self._content_started or self._content_closed:
            return
        response_ids = self.model_tokenizer.encode(
            self._RESPONSE_OPEN, add_special_tokens=False
        )
        # Only the generation prefix matters, not response tags in earlier
        # turns. Compare token ids to preserve structural-token identity.
        if response_ids and list(prompt_token_ids[-len(response_ids):]) == list(
            response_ids
        ):
            self._content_started = True

    def filter_content_delta(self, text: str, *, finished: bool) -> str:
        """Filter actual delegating-parser output, buffering split XTML tags."""
        if self._content_closed:
            return ""
        pending = self._content_pending + text
        self._content_pending = ""
        if not self._content_started:
            # Until response opens, text may still be residual analysis even
            # when thinking is disabled. Already emitted deltas cannot be
            # retracted when the channel marker arrives in a later chunk.
            response_open = self._response_open_re.search(pending)
            if response_open is None:
                if finished:
                    self._content_closed = True
                    # Unwrapped output remains supported, matching the
                    # non-streaming channel selection at end of generation.
                    return self._strip_content_wrapper(pending)
                self._content_pending = pending
                return ""
            self._content_started = True
            pending = pending[response_open.end():]
        output = []
        while pending:
            match = self._content_wrapper_re.search(pending, partial=True)
            if match is None:
                output.append(pending)
                break
            output.append(pending[:match.start()])
            if match.partial:
                tail = pending[match.start():]
                if finished:
                    output.append(tail)
                else:
                    self._content_pending = tail
                break
            if self._response_close_re.fullmatch(match.group()):
                self._content_closed = True
                break
            pending = pending[match.end():]
        return "".join(output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        self._last_streaming_delta_ids = None
        self._last_streaming_content_ids = None
        if not self._thinking_enabled:
            return DeltaMessage(content=delta_text)
        if self._think_close_re.search(previous_text):
            return DeltaMessage(content=delta_text)

        close_match = self._think_close_re.search(current_text)
        if close_match is not None:
            self._last_streaming_delta_ids = tuple(delta_token_ids)
            self._last_streaming_content_ids = self._extract_content_ids(
                list(current_token_ids)
            )
            open_match = self._think_open_re.search(current_text)
            start = open_match.end() if open_match is not None else 0
            reasoning = current_text[start : close_match.start()]
            previous_reasoning = self._reasoning_ready_to_emit(previous_text)
            reasoning_delta = (
                reasoning[len(previous_reasoning) :]
                if reasoning.startswith(previous_reasoning)
                else reasoning
            )
            return DeltaMessage(
                reasoning=reasoning_delta or None,
                content=current_text[close_match.end() :] or None,
            )

        current_reasoning = self._reasoning_ready_to_emit(current_text)
        previous_reasoning = self._reasoning_ready_to_emit(previous_text)
        delta = (
            current_reasoning[len(previous_reasoning) :]
            if current_reasoning.startswith(previous_reasoning)
            else current_reasoning
        )
        return DeltaMessage(reasoning=delta) if delta else None

    def _reasoning_ready_to_emit(self, text: str) -> str:
        open_match = self._think_open_re.search(text)
        if open_match is not None:
            text = text[open_match.end() :]
        return self._without_partial_marker(text, (self._THINK_OPEN, self._THINK_CLOSE))

    extract_reasoning_content = extract_reasoning
    extract_reasoning_content_streaming = extract_reasoning_streaming


__all__ = ["KimiK3ReasoningParser"]
