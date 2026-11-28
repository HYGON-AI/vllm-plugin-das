# SPDX-License-Identifier: Apache-2.0
"""Real stock rotary math for HY4 vision grids larger than the initial table."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("height,width", [(256, 64), (1638, 10)])
def test_vision_2d_rotary_covers_admitted_tall_grid(height, width):
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLVisionConfig
    from hcu_vllm_das.models.hy4_standard.vision import HYV4VisionTransformer
    from vllm.config import (
        CompilationConfig,
        DeviceConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.layers.rotary_embedding import get_rope

    config = HYV4VLVisionConfig(hidden_size=32, num_attention_heads=4)
    runtime = VllmConfig(
        device_config=DeviceConfig("cpu"),
        compilation_config=CompilationConfig(custom_ops=["none"]),
    )
    kwargs = {
        "head_size": 8,
        "is_neox_style": True,
        "rope_parameters": {
            "rope_theta": config.rope_2d_theta,
            "partial_rotary_factor": 0.5,
        },
        "dtype": torch.float32,
    }
    with set_current_vllm_config(runtime):
        initial = get_rope(
            max_position=config.max_image_size // config.patch_size + 1, **kwargs
        )
        old_cos, old_sin = initial.get_cos_sin(129)
        # This isolates the real rotary method from unrelated distributed layers.
        tower = SimpleNamespace(
            config=config, rotary_pos_emb=initial, device=torch.device("cpu")
        )
        cos, sin = HYV4VisionTransformer._rope_cos_sin(tower, [[1, height, width]])
        reference = get_rope(max_position=max(height, width), **kwargs)
        ref_cos, ref_sin = reference.get_cos_sin(max(height, width))
        indices = torch.stack(
            (
                torch.arange(height).repeat_interleave(width),
                torch.arange(width).repeat(height),
            ),
            dim=-1,
        )
        torch.testing.assert_close(cos, ref_cos[indices].flatten(1), rtol=0, atol=0)
        torch.testing.assert_close(sin, ref_sin[indices].flatten(1), rtol=0, atol=0)
        expanded = tower.rotary_pos_emb
        new_cos, new_sin = expanded.get_cos_sin(129)
        torch.testing.assert_close(new_cos, old_cos, rtol=0, atol=0)
        torch.testing.assert_close(new_sin, old_sin, rtol=0, atol=0)
        HYV4VisionTransformer._rope_cos_sin(tower, [[1, 4, 4]])
        assert tower.rotary_pos_emb is expanded
        assert not torch.cuda.is_initialized()
