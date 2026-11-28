# SPDX-License-Identifier: Apache-2.0
"""Regression contracts for independent video config and resize limits."""

import json

import pytest


@pytest.fixture
def visual_checkpoint(tmp_path):
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    special = [
        "<unk>",
        "<image_start>",
        "<image_end>",
        "<image>",
        "<image_newline>",
        "<video_start>",
        "<video_end>",
        "<video>",
    ]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(dict(zip(special, range(len(special)))), unk_token="<unk>")
        ),
        unk_token="<unk>",
        additional_special_tokens=special[1:],
    )
    tokenizer.save_pretrained(tmp_path)
    HYV4VLConfig(
        image_start_token_id=1,
        image_end_token_id=2,
        image_token_id=3,
        image_newline_token_id=4,
        video_start_token_id=5,
        video_end_token_id=6,
        video_token_id=7,
    ).save_pretrained(tmp_path)
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "size": {"shortest_edge": 4096, "longest_edge": 65536},
                "image_mean": [0.1, 0.2, 0.3],
                "image_std": [0.4, 0.5, 0.6],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _write_video_config(checkpoint):
    (checkpoint / "video_preprocessor_config.json").write_text(
        json.dumps(
            {
                "size": {"shortest_edge": 4096, "longest_edge": 1048576},
                "image_mean": [0.7, 0.8, 0.9],
                "image_std": [0.2, 0.3, 0.4],
                "fps": 6,
                "max_frames": 12,
                "video_max_image_size": 320,
                "timestamp_token_reserve": 16,
            }
        ),
        encoding="utf-8",
    )


def test_from_pretrained_uses_separate_video_config(visual_checkpoint):
    from hcu_vllm_das.models.hy4_standard.processor import HYV4VLProcessor

    _write_video_config(visual_checkpoint)
    processor = HYV4VLProcessor.from_pretrained(
        visual_checkpoint, local_files_only=True
    )
    assert processor.image_processor.size["longest_edge"] == 65536
    assert list(processor.image_processor.image_mean) == [0.1, 0.2, 0.3]
    assert processor.video_processor.size["longest_edge"] == 1048576
    assert list(processor.video_processor.image_mean) == [0.7, 0.8, 0.9]
    assert list(processor.video_processor.image_std) == [0.2, 0.3, 0.4]
    assert processor.video_processor.fps == 6
    assert processor.video_processor.max_frames == 12
    assert processor.video_processor.video_max_image_size == 320
    assert processor.video_processor.timestamp_token_reserve == 16


def test_from_pretrained_video_config_falls_back_to_image(visual_checkpoint):
    from hcu_vllm_das.models.hy4_standard.processor import HYV4VLProcessor

    processor = HYV4VLProcessor.from_pretrained(
        visual_checkpoint, local_files_only=True
    )
    assert processor.video_processor.size["longest_edge"] == 65536
    assert list(processor.video_processor.image_mean) == [0.1, 0.2, 0.3]


def test_from_pretrained_explicit_video_overrides_preserve_other_config(
    visual_checkpoint,
):
    from hcu_vllm_das.models.hy4_standard.processor import HYV4VLProcessor

    _write_video_config(visual_checkpoint)
    processor = HYV4VLProcessor.from_pretrained(
        visual_checkpoint,
        local_files_only=True,
        images_kwargs={"image_mean": [0.0, 0.0, 0.0]},
        videos_kwargs={"fps": 3, "video_max_image_size": 256},
        max_frames=8,
    )
    assert processor.video_processor.fps == 3
    assert processor.video_processor.video_max_image_size == 256
    assert processor.video_processor.max_frames == 8
    assert processor.video_processor.size["longest_edge"] == 1048576
    assert list(processor.video_processor.image_mean) == [0.7, 0.8, 0.9]
    assert list(processor.image_processor.image_mean) == [0.0, 0.0, 0.0]
    assert processor.image_processor.size["longest_edge"] == 65536


def test_video_loader_receives_checkpoint_resolution_options(
    visual_checkpoint, monkeypatch, tmp_path
):
    from hcu_vllm_das.models.hy4_standard.processor import (
        HYV4VLProcessor,
        HYV4VLVideoProcessor,
    )

    _write_video_config(visual_checkpoint)
    original_loader = HYV4VLVideoProcessor.get_video_processor_dict
    received = []

    def record_loader(cls, checkpoint, **kwargs):
        received.append((checkpoint, dict(kwargs)))
        return original_loader(checkpoint, **kwargs)

    monkeypatch.setattr(
        HYV4VLVideoProcessor,
        "get_video_processor_dict",
        classmethod(record_loader),
    )
    load_options = {
        "revision": "test-revision",
        "cache_dir": str(tmp_path / "cache"),
        "local_files_only": True,
        "token": "local-fixture-token",
    }
    processor = HYV4VLProcessor.from_pretrained(visual_checkpoint, **load_options)
    assert received == [(visual_checkpoint, load_options)]
    assert processor.video_processor.fps == 6


@pytest.mark.parametrize(
    "height,width,num_frames,max_pixels,side,expected",
    [
        (2048, 2048, 2, 4194304, 256, (256, 256)),
        (1024, 2048, 2, 4194304, 256, (160, 352)),
        (2048, 2048, 2, 4194304, 2048, (1024, 1024)),
        (64, 64, 2, 33554432, 1024, (64, 64)),
        (64, 64, 2, 1024, 256, (32, 32)),
        (4096, 32, 2, 33554432, 256, (2048, 32)),
        (32, 4096, 2, 33554432, 256, (32, 2048)),
    ],
)
def test_adaptive_resize_applies_budget_and_frame_area_limits(
    height, width, num_frames, max_pixels, side, expected
):
    from hcu_vllm_das.models.hy4_standard.processor import adaptive_resize

    resized = adaptive_resize(
        height,
        width,
        num_frames,
        max_pixels=max_pixels,
        video_max_image_size=side,
    )
    assert resized == expected
    assert resized[0] % 32 == resized[1] % 32 == 0
    assert resized[0] * resized[1] <= side**2


def test_adaptive_resize_rejects_frame_area_smaller_than_one_patch():
    from hcu_vllm_das.models.hy4_standard.processor import adaptive_resize

    with pytest.raises(ValueError, match="video_max_image_size"):
        adaptive_resize(64, 64, 2, video_max_image_size=16)


@pytest.mark.parametrize(
    "height,width,num_frames,max_pixels",
    [
        (0, 64, 2, 4194304),
        (64, 0, 2, 4194304),
        (64, 64, 0, 4194304),
        (64, 64, 2, 0),
    ],
)
def test_adaptive_resize_keeps_invalid_budget_errors(
    height, width, num_frames, max_pixels
):
    from hcu_vllm_das.models.hy4_standard.processor import adaptive_resize

    with pytest.raises(ValueError, match="must be positive"):
        adaptive_resize(height, width, num_frames, max_pixels=max_pixels)
