from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)

from vllm_hcu.patch.worker.core_fix.patch_glm5next_channel_fp8 import (
    _expand_glm5next_multimodal_ignore_aliases,
)


def test_glm53_multimodal_regex_ignore_matches_runtime_prefix():
    original = [
        "re:^model\\.layers\\.[012]\\..*",
        "re:^model\\.layers\\.\\d+\\.self_attn\\.o_proj$",
        "visual",
    ]

    expanded = _expand_glm5next_multimodal_ignore_aliases(original)

    assert expanded[: len(original)] == original
    assert "re:^language_model\\.model\\.layers\\.[012]\\..*" in expanded
    assert should_ignore_layer(
        "language_model.model.layers.0.mlp.down_proj",
        ignore=expanded,
    )
    assert should_ignore_layer(
        "language_model.model.layers.4.self_attn.o_proj",
        ignore=expanded,
    )


def test_glm53_multimodal_regex_ignore_expansion_is_idempotent():
    original = ["re:^model\\.layers\\.[012]\\..*"]
    once = _expand_glm5next_multimodal_ignore_aliases(original)

    assert _expand_glm5next_multimodal_ignore_aliases(once) == once
