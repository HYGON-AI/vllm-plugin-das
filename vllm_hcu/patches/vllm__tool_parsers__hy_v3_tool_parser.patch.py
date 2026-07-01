"""
Patch for vllm/tool_parsers/hy_v3_tool_parser.py
"""

PATCHES = [
(
'''
        if "type" in arg_schema:
            return [arg_schema]
''',
'''
        if "type" in arg_schema:
            type_val = arg_schema["type"]
            # JSON Schema allows "type" to be an array to represent union types,
            # e.g. "type": ["string", "object"].
            # Expand it into an anyOf-equivalent format:
            #   [{"type": "string"}, {"type": "object"}]
            # so that _get_types / _parse_value can handle it uniformly later.
            if isinstance(type_val, list):
                return [{"type": t} for t in type_val]
            return [arg_schema]
''',
),
(
'''
        self.tool_calls_start_token: str = "<tool_calls>"
        self.tool_calls_end_token: str = "</tool_calls>"

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        self.tool_sep_token: str = "<tool_sep>"

        self.arg_key_start_token: str = "<arg_key>"
        self.arg_key_end_token: str = "</arg_key>"

        self.arg_value_start_token: str = "<arg_value>"
        self.arg_value_end_token: str = "</arg_value>"
''',
'''
        init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
        self.suffix: str = init_kwargs.get("token_suffix") or ""

        self.tool_calls_start_token: str = f"<tool_calls{self.suffix}>"
        self.tool_calls_end_token: str = f"</tool_calls{self.suffix}>"

        self.tool_call_start_token: str = f"<tool_call{self.suffix}>"
        self.tool_call_end_token: str = f"</tool_call{self.suffix}>"

        self.tool_sep_token: str = f"<tool_sep{self.suffix}>"

        self.arg_key_start_token: str = f"<arg_key{self.suffix}>"
        self.arg_key_end_token: str = f"</arg_key{self.suffix}>"

        self.arg_value_start_token: str = f"<arg_value{self.suffix}>"
        self.arg_value_end_token: str = f"</arg_value{self.suffix}>"
''',
),
]