# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Step5 tool parser registered by the HCU runtime patch coordinator.

Step5 answers with the same XML tool-call format as Step3.5, so extraction and
streaming are inherited from ``Step3p5ToolParser`` unchanged. The inherited
behaviours that must differ are the required/named tool-choice dispatch flag,
the generation-side request adjustment and the Responses API schemas; see the
class docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai.types.responses import FunctionTool
from openai.types.responses.response import ToolChoiceFunction
from vllm.entrypoints.generate.base.protocol import (
    ExtractedToolCallInformation,
    FunctionDefinition,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.step3p5_tool_parser import Step3p5ToolParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

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


def _named_tool_choice_function(
    request: ChatCompletionRequest | ResponsesRequest,
) -> str | None:
    """Function name pinned by a named tool choice, or None for the other modes.

    Chat Completions nests the name under ``function``; the Responses API puts
    it on the choice itself. Mirrors ``DelegatingParser._get_function_name``.
    """
    tool_choice = request.tool_choice
    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        return tool_choice.function.name
    if isinstance(tool_choice, ToolChoiceFunction):
        return tool_choice.name
    return None


def _chat_completion_tools(
    tools: list[ChatCompletionToolsParam | FunctionTool] | None,
) -> list[ChatCompletionToolsParam | FunctionTool]:
    """Express Responses ``FunctionTool`` entries in the nested Chat shape.

    ``ToolParser`` keeps both ``ChatCompletionToolsParam`` and Responses
    ``FunctionTool`` objects, but the inherited XML parser locates parameter
    types through ``tool.function.parameters``. A flat Responses tool carries
    ``name``/``parameters`` on the tool itself, so without this mapping every
    argument would be parsed as a string.
    """
    if not tools:
        return []
    normalized: list[ChatCompletionToolsParam | FunctionTool] = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            normalized.append(
                ChatCompletionToolsParam(
                    type="function",
                    function=FunctionDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                        strict=tool.strict,
                        defer_loading=getattr(tool, "defer_loading", None),
                    ),
                )
            )
        else:
            normalized.append(tool)
    return normalized


class Step5ToolParser(Step3p5ToolParser):
    """Tool parser for Step5.

    Step5 answers with the same XML tool-call format as Step3.5, so extraction is
    inherited unchanged. Three inherited behaviours do have to change.

    ``supports_required_and_named`` is False: the generic serving path reads a
    True value as "this model satisfies required/named tool choice through the
    JSON grammar the server installs". Step5 replies in XML, so those two modes
    must be parsed by this parser instead -- which is exactly what the flag
    selects.

    ``adjust_request`` is overridden for the matching reason on the generation
    side; see the method.

    ``__init__`` maps Responses ``FunctionTool`` entries onto the Chat
    Completions shape the inherited XML parser reads; ``extract_tool_calls``
    reads the named tool choice from either request schema.
    """

    supports_required_and_named = False

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[ChatCompletionToolsParam | FunctionTool] | None = None,
    ) -> None:
        super().__init__(tokenizer, _chat_completion_tools(tools))

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
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
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
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
