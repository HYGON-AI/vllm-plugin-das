# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Step5 tool parser registered by the HCU runtime patch coordinator.

Step5 answers with the same XML tool-call format as Step3.5, so extraction and
streaming are inherited from ``Step3p5ToolParser`` unchanged. The two inherited
behaviours that must differ are the required/named tool-choice dispatch flag and
the generation-side request adjustment; see the class docstring.
"""

from __future__ import annotations

from vllm.entrypoints.generate.base.protocol import (
    ExtractedToolCallInformation,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.logger import init_logger
from vllm.tool_parsers.step3p5_tool_parser import Step3p5ToolParser

logger = init_logger(__name__)

# These inherited behaviours are only safe while Step3p5 keeps the standard
# flag-based dispatch and XML extraction. Fail when the parser is resolved
# instead of silently serving a different protocol after an upstream change.
if getattr(Step3p5ToolParser, "supports_required_and_named", None) is not True:
    raise RuntimeError(
        "Step3p5ToolParser.supports_required_and_named is no longer True; "
        "re-review the Step5 required/named tool-choice contract"
    )
for _method_name in ("extract_tool_calls", "extract_tool_calls_streaming"):
    if not callable(getattr(Step3p5ToolParser, _method_name, None)):
        raise RuntimeError(
            f"Step3p5ToolParser.{_method_name} is missing; "
            "re-review the Step5 parser contract"
        )


def _named_tool_choice_function(request: ChatCompletionRequest) -> str | None:
    """Function name pinned by a named tool choice, or None for the other modes."""
    tool_choice = request.tool_choice
    if tool_choice is None or isinstance(tool_choice, str):
        return None
    return tool_choice.function.name


class Step5ToolParser(Step3p5ToolParser):
    """Tool parser for Step5.

    Step5 answers with the same XML tool-call format as Step3.5, so extraction is
    inherited unchanged. Two inherited behaviours do have to change.

    ``supports_required_and_named`` is False: the generic serving path reads a
    True value as "this model satisfies required/named tool choice through the
    JSON grammar the server installs". Step5 replies in XML, so those two modes
    must be parsed by this parser instead -- which is exactly what the flag
    selects.

    ``adjust_request`` is overridden for the matching reason on the generation
    side; see the method.
    """

    supports_required_and_named = False

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        # ``ToolParser.adjust_request`` derives a JSON Schema from the tool
        # definitions for required and named tool choice and installs it as
        # ``structured_outputs``, i.e. guided decoding. That grammar holds the
        # model to JSON and therefore to something other than the ``<tool_call>``
        # block it was trained to emit. Leave the request untouched for those
        # modes and let the constraint be applied to the parsed calls instead.
        # auto/none derive no schema, so they keep the inherited behaviour.
        if request.tool_choice in (None, "auto"):
            return super().adjust_request(request)
        return request

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        info = super().extract_tool_calls(model_output, request)

        requested = _named_tool_choice_function(request)
        if requested is not None and requested not in [
            call.function.name for call in info.tool_calls or []
        ]:
            # The model picked a different function. The calls it really made are
            # returned as they are -- relabelling them with the requested name
            # would report a call that never happened.
            logger.warning(
                "Model was asked for function %r but called %s; returning the "
                "calls as generated.",
                requested,
                [call.function.name for call in info.tool_calls or []],
            )
        elif request.tool_choice == "required" and not info.tools_called:
            # Answering in plain text under "required" is a model-side choice;
            # the response still reports finish_reason="tool_calls", so log it
            # rather than let the empty tool_calls list look like a parser fault.
            logger.warning(
                "tool_choice='required' but the model produced no tool call."
            )
        return info


__all__ = ["Step5ToolParser"]
