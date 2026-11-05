# SPDX-License-Identifier: Apache-2.0
"""CPU configuration regressions for eager Transformers RoPE normalization."""

from copy import deepcopy
import json

import pytest

from vllm_hcu.models.step5_config import Step5Config, Step5VLConfig


def llama3_scaling():
    return {
        "rope_type": "llama3",
        "factor": 2.0,
        "original_max_position_embeddings": 4096,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
    }


def test_default_context_length_is_preserved():
    config = Step5Config()
    assert config.max_seq_len == config.max_position_embeddings == 4096


@pytest.mark.parametrize("explicit_context", [None, 16384])
def test_scaled_rope_initialization_resolves_context_first(explicit_context):
    # The original context key is deliberately present: dict.setdefault still
    # evaluates its default expression, so it used to read the missing member.
    scaling = llama3_scaling()
    config = Step5Config(
        max_seq_len=8192,
        max_position_embeddings=explicit_context,
        rope_theta=500000,
        rope_scaling=deepcopy(scaling),
    )
    assert config.max_position_embeddings == (explicit_context or 8192)
    assert config.max_seq_len == 8192
    assert config.rope_theta == 500000
    for key, value in scaling.items():
        assert config.rope_scaling[key] == value


def test_composite_config_roundtrip_preserves_rope_and_mtp_layer_geometry():
    layer_types = ["sliding_attention", "full_attention", "sliding_attention"]
    theta = [10000, 500000, 10000]
    scaling = llama3_scaling()
    config = Step5VLConfig(
        image_token_id=42,
        text_config={
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_attention_groups": 2,
            "head_dim": 32,
            "vocab_size": 128,
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 1,
            "max_seq_len": 8192,
            "max_position_embeddings": 16384,
            "layer_types": layer_types,
            "rope_theta": theta,
            "rope_scaling": deepcopy(scaling),
        },
    )
    # Exercise the JSON-compatible wire representation, not just object reuse.
    restored = Step5VLConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert isinstance(restored.text_config, Step5Config)
    for text in [config.text_config, restored.text_config]:
        assert text.max_seq_len == 8192
        assert text.max_position_embeddings == 16384
        assert text.rope_theta == theta
        assert text.layer_types == layer_types[:2]
        assert text.layer_types_with_mtp == layer_types
        assert text.num_nextn_predict_layers == 1
        for key, value in scaling.items():
            assert text.rope_scaling[key] == value
    assert restored.vision_config.width == config.vision_config.width
    assert restored.image_token_id == config.image_token_id == 42


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_position_embeddings": 0}, "max_position_embeddings must be positive"),
        ({"max_position_embeddings": -1}, "max_position_embeddings must be positive"),
        ({"max_seq_len": 0}, "max_seq_len must be positive"),
        ({"max_position_embedding": 100}, "max_position_embedding"),
    ],
)
def test_context_validation_is_not_bypassed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Step5Config(**kwargs)
