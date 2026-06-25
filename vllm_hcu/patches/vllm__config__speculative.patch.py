PATCHES = [
(
'''    use_local_argmax_reduction: bool = False
    """Use vocab-parallel local argmax instead of all-gathering full logits
    for draft token generation. Reduces communication from O(vocab_size) to
    O(2 * tp_size) per token. Only applies to greedy draft selection in
    non-tree speculation."""
''',
'''    use_local_argmax_reduction: bool = False
    """Use vocab-parallel local argmax instead of all-gathering full logits
    for draft token generation. Reduces communication from O(vocab_size) to
    O(2 * tp_size) per token. Only applies to greedy draft selection in
    non-tree speculation."""
    enable_multi_layers_mtp: bool = False
    """Enable using all configured MTP layers for models that provide multiple
    next-token-prediction layers. Currently this is used by Step3.5 MTP."""
''',
),

(
'''    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
''',
'''    @staticmethod
    def hf_config_override(
        hf_config: PretrainedConfig,
        *,
        enable_multi_layers_mtp: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
''',
),
]