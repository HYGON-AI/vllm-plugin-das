"""
Patch for vllm/reasoning/hy_v3_reasoning_parser.py
"""

PATCHES = [
(
'''
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
''',
'''
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
        self.suffix: str = init_kwargs.get("token_suffix") or ""
        super().__init__(tokenizer, *args, **kwargs)
''',
),
(
'''
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"
''',
'''
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return f"<think{self.suffix}>"
''',
),
(
'''
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"
''',
'''
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return f"</think{self.suffix}>"
''',
),
]