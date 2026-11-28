# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standard HY4 VL configuration, preprocessing and position contracts."""

from copy import deepcopy

import pytest
import torch
from PIL import Image


def test_vl_config_roundtrip_preserves_checkpoint_metadata(tmp_path):
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLConfig

    quant = {"quant_method": "compressed-tensors", "format": "pack-quantized"}
    config = HYV4VLConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        quantization_config=deepcopy(quant),
        dfly_num_hidden_layers=5,
        dfly_block_size=7,
        num_nextn_predict_layers=0,
    )
    config.save_pretrained(tmp_path)
    restored = HYV4VLConfig.from_pretrained(tmp_path)
    assert restored.model_type == "hy_v4_vl"
    assert restored.architectures == ["HYV4VLForConditionalGeneration"]
    assert restored.get_text_config().model_type == "hy_v4"
    assert restored.text_config.quantization_config == quant
    assert restored.dfly_num_hidden_layers == 5
    assert restored.dfly_block_size == 7
    assert restored.text_config.num_nextn_predict_layers == 0
    assert restored.vision_config.out_hidden_size == 256
    assert restored.text_config.rope_parameters["mrope_section"] == [12, 10, 10]


def test_nested_text_config_overrides_flat_values():
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLConfig

    config = HYV4VLConfig(hidden_size=256, text_config={"hidden_size": 512})
    assert config.text_config.hidden_size == 512
    assert config.vision_config.out_hidden_size == 512
    with pytest.raises(ValueError, match="projector output"):
        HYV4VLConfig(hidden_size=256, vision_config={"out_hidden_size": 128})


def test_owned_vl_text_marker_survives_roundtrip_without_claiming_plain_text(tmp_path):
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLConfig
    from vllm.transformers_utils.configs.hy_v4 import HYV4Config

    original = HYV4Config(
        rope_parameters={"rope_type": "default", "mrope_section": [12, 10, 10]}
    )
    config = HYV4VLConfig(text_config=original)
    assert getattr(original, "hy4_standard_vl", False) is False
    assert config.text_config is not original
    assert config.text_config.hy4_standard_vl is True
    assert isinstance(config.text_config, HYV4Config)
    config.save_pretrained(tmp_path)
    restored = HYV4VLConfig.from_pretrained(tmp_path)
    assert restored.text_config.hy4_standard_vl is True
    assert restored.text_config.model_type == "hy_v4"


def test_real_image_processor_matches_patch_grid():
    from hcu_vllm_das.models.hy4_standard.processor import HYV4VLImageProcessor

    processor = HYV4VLImageProcessor(min_pixels=4096, max_pixels=4096)
    output = processor(Image.new("RGB", (64, 64)), return_tensors="pt")
    assert output["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert output["pixel_values"].shape == (16, 768)
    assert torch.isfinite(output["pixel_values"]).all()
    with pytest.raises(ValueError, match="align"):
        processor(Image.new("RGB", (63, 64)), do_resize=False)


@pytest.mark.parametrize("grid", [(1, 0, 4), (1, 3, 4), (0, 4, 4)])
def test_invalid_visual_grids_are_rejected(grid):
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLVisionConfig
    from hcu_vllm_das.models.hy4_standard.processor import vision_token_grid

    with pytest.raises(ValueError):
        vision_token_grid(grid, HYV4VLVisionConfig(), "image")


def test_image_mrope_compresses_media_without_losing_text_positions():
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLConfig
    from hcu_vllm_das.models.hy4_standard.mrope import build_mrope_positions

    config = HYV4VLConfig()
    tokens = [17, config.image_start_token_id]
    tokens += [config.image_token_id] * 6
    tokens += [config.image_end_token_id, 18]
    positions, delta = build_mrope_positions(config, tokens, [[1, 4, 4]], [])
    assert positions.tolist() == [
        [0, 1, 2, 2, 2, 2, 2, 2, 5, 6],
        [0, 1, 2, 2, 2, 3, 3, 3, 5, 6],
        [0, 1, 2, 3, 4, 2, 3, 4, 5, 6],
    ]
    assert delta == -3
    with pytest.raises(ValueError, match="frame boundaries and grid"):
        build_mrope_positions(config, tokens[:-3] + tokens[-2:], [[1, 4, 4]], [])
    with pytest.raises(ValueError, match="paired"):
        build_mrope_positions(config, tokens[:-2], [[1, 4, 4]], [])


def test_text_only_mrope_keeps_all_three_axes_equal():
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLConfig
    from hcu_vllm_das.models.hy4_standard.mrope import build_mrope_positions

    positions, delta = build_mrope_positions(HYV4VLConfig(), [3, 7, 9], [], [])
    torch.testing.assert_close(positions, torch.arange(3).repeat(3, 1))
    assert delta == 0


def test_video_extra_tokens_advance_compressed_positions():
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLConfig
    from hcu_vllm_das.models.hy4_standard.mrope import build_mrope_positions

    config = HYV4VLConfig(vision_config={"cat_extra_token": 1})
    frame = [config.image_start_token_id] + [config.video_token_id] * 8
    frame += [config.image_end_token_id]
    tokens = [101, config.video_start_token_id, *frame, 99, *frame]
    tokens += [config.video_end_token_id, 102]
    positions, delta = build_mrope_positions(config, tokens, [], [[2, 4, 4]])
    assert positions[:, [3, 10, 14, 21]].tolist() == [[3, 7, 11, 15]] * 3
    assert positions[:, -1].tolist() == [18, 18, 18]
    assert delta == -6


@pytest.mark.parametrize("extra", [0, 1])
@pytest.mark.parametrize("mixed_video", [False, True])
def test_multiple_media_keep_text_delta_consistent(extra, mixed_video):
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLConfig
    from hcu_vllm_das.models.hy4_standard.mrope import build_mrope_positions

    config = HYV4VLConfig(vision_config={"cat_extra_token": extra})

    def frame(pad):
        return (
            [config.image_start_token_id]
            + [pad] * (6 + 2 * extra)
            + [config.image_end_token_id]
        )

    tokens = [101, *frame(config.image_token_id), 103]
    images = [[1, 4, 4]]
    if mixed_video:
        tokens += [
            config.video_start_token_id,
            *frame(config.video_token_id),
            99,
            *frame(config.video_token_id),
            config.video_end_token_id,
        ]
        videos = [[2, 4, 4]]
        blocks = 3
    else:
        tokens += frame(config.image_token_id)
        images += [[1, 4, 4]]
        videos = []
        blocks = 2
    tokens += [102]
    positions, delta = build_mrope_positions(config, tokens, images, videos)
    assert delta == -3 * blocks
    assert positions[:, -1].tolist() == [len(tokens) - 1 - 3 * blocks] * 3


def test_mrope_cache_is_rebuilt_after_text_only_forward():
    from hcu_vllm_das.models.hy4_standard.mrope import set_token_rope, token_rope
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    with set_current_vllm_config(VllmConfig()):
        rotary = MRotaryEmbedding(
            64,
            64,
            128,
            10000,
            False,
            torch.float32,
            mrope_section=[12, 10, 10],
            mrope_interleaved=True,
        )
    hidden = torch.zeros(4, 64)
    image_positions = torch.tensor([[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 1, 2]])
    set_token_rope(rotary, image_positions, hidden)
    _, image_rows = token_rope(rotary, image_positions, rotary.cos_sin_cache)
    text_positions = torch.arange(4)
    set_token_rope(rotary, text_positions, hidden)
    text_ids, text_cache = token_rope(rotary, text_positions, rotary.cos_sin_cache)
    assert text_ids is text_positions
    assert text_cache is rotary.cos_sin_cache
    changed_positions = image_positions + 7
    set_token_rope(rotary, changed_positions, hidden)
    changed_ids, changed_rows = token_rope(
        rotary, changed_positions, rotary.cos_sin_cache
    )
    torch.testing.assert_close(changed_ids, torch.arange(4))
    assert not torch.equal(image_rows, changed_rows)


@pytest.mark.parametrize("video_limit,has_axis", [(0, False), (1, True)])
def test_encoder_item_selection_matches_capture_axes(video_limit, has_axis):
    from hcu_vllm_das.models.hy4_standard.vl_cudagraph import (
        HYV4VLEncoderCudaGraphMixin,
    )
    from vllm.config.multimodal import MultiModalConfig
    from vllm.v1.worker.encoder_cudagraph_defs import (
        ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG,
    )

    encoder = HYV4VLEncoderCudaGraphMixin()
    encoder.multimodal_config = MultiModalConfig(
        limit_per_prompt={"image": 1, "video": video_limit}
    )
    kwargs = {
        "pixel_values": torch.zeros(16, 768),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    for indices in ([], [0]):
        selected = encoder.select_encoder_cudagraph_items(kwargs, indices)
        assert (ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG in selected) is has_axis


def test_metadata_only_images_require_consumer_mode():
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLVisionConfig
    from hcu_vllm_das.models.hy4_standard.mm_preprocess import HYV4VLDataParser

    data = {"image_grid_thw": torch.tensor([[1, 4, 4]])}
    producer = HYV4VLDataParser(HYV4VLVisionConfig(), max_model_len=128)
    with pytest.raises(ValueError, match="consumer"):
        producer.parse_mm_data({"image": data})
    consumer = HYV4VLDataParser(
        HYV4VLVisionConfig(), max_model_len=128, allow_missing_mm_embeddings=True
    )
    parsed = consumer.parse_mm_data({"image": data})
    assert len(parsed["image"]) == 1
    with pytest.raises(ValueError, match="context"):
        HYV4VLDataParser(
            HYV4VLVisionConfig(), max_model_len=2, allow_missing_mm_embeddings=True
        ).parse_mm_data({"image": data})


def test_registry_resolves_standard_vl_to_owned_wrapper_on_stock_core(tmp_path):
    from hcu_adapter.extensions import ExtensionContext
    from hcu_vllm_das.hy4_vl_activation import activate
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLConfig
    from vllm.config import ModelConfig
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.renderers.registry import RENDERER_REGISTRY

    activate(ExtensionContext(adapter_version="test", vllm_version="0.33.0"))
    HYV4VLConfig(hidden_size=256, num_hidden_layers=2).save_pretrained(tmp_path)
    config = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )
    cls, arch = ModelRegistry.resolve_model_cls(
        ["HYV4VLForConditionalGeneration"], model_config=config
    )
    assert arch == "HYV4VLForConditionalGeneration"
    assert cls.language_model_architecture == "HYV4ForCausalLM"
    assert cls.__module__ == "hcu_vllm_das.models.hy4_standard.vl_model"
    assert config.tokenizer_mode in ("auto", "hf")
    renderer = RENDERER_REGISTRY.load_renderer_cls("hf")
    assert renderer.__name__ == "HY4VLHfRenderer"
    assert not config.multimodal_config.mm_device_do_normalize


def test_target_loader_skips_only_checkpoint_draft_prefixes():
    from hcu_vllm_das.models.hy4_standard.weight_utils import filter_target_weights
    from vllm.model_executor.models.utils import AutoWeightsLoader

    target = torch.nn.Module()
    target.model = torch.nn.Module()
    target.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
    target.lm_head = torch.nn.Linear(2, 2, bias=False)
    weights = [
        ("model.layers.0.weight", torch.ones(2, 2)),
        ("lm_head.weight", torch.full((2, 2), 2.0)),
        ("model.dfly_layers.0.weight", torch.zeros(2, 2)),
    ]
    loaded = AutoWeightsLoader(target).load_weights(filter_target_weights(weights))
    assert loaded == {"model.layers.0.weight", "lm_head.weight"}
    torch.testing.assert_close(target.model.layers[0].weight, torch.ones(2, 2))
    torch.testing.assert_close(target.lm_head.weight, torch.full((2, 2), 2.0))
    with pytest.raises(ValueError, match="no module or parameter"):
        AutoWeightsLoader(target).load_weights(
            filter_target_weights([("model.unexpected.weight", torch.zeros(2, 2))])
        )


def test_target_weight_filter_preserves_unknown_and_nearby_prefixes():
    from hcu_vllm_das.models.hy4_standard.weight_utils import filter_target_weights

    tensor = torch.ones(1)
    names = [
        "model.layers.0.weight",
        "model.dfly_layers.0.weight",
        "model.dfly_layers_extra.0.weight",
        "model.unexpected.weight",
        "other.model.dfly_layers.0.weight",
    ]
    filtered = list(filter_target_weights((name, tensor) for name in names))
    assert [name for name, _ in filtered] == [names[0], *names[2:]]
    assert all(value is tensor for _, value in filtered)
