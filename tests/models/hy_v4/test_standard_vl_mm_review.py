# SPDX-License-Identifier: Apache-2.0
"""MR550 regressions using installed Core caching and real CPU HF processing.

The in-memory HF processor avoids checkpoint downloads. Core context, parser,
cache, prompt replacement, field splitting, and HF preprocessing are real.
"""

from copy import deepcopy

import numpy as np
import pytest
import torch


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    from hcu_vllm_das.models.hy4_standard.mm_preprocess import (
        HYV4VLDummyInputsBuilder,
        HYV4VLMultiModalProcessor,
        HYV4VLProcessingInfo,
    )
    from hcu_vllm_das.models.hy4_standard.processor import (
        HYV4VLImageProcessor,
        HYV4VLProcessor,
        HYV4VLVideoProcessor,
    )
    from test_standard_vl_runtime import _cpu_config
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache
    from vllm.multimodal.processing.context import InputProcessingContext

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config = _cpu_config(tmp_path)
    special = [
        "<image_start>",
        "<image_end>",
        "<image>",
        "<image_newline>",
        "<video_start>",
        "<video_end>",
        "<video>",
    ]
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {word: i for i, word in enumerate(alphabet + special)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, additional_special_tokens=special
    )
    hf_config = config.model_config.hf_config
    for name in (
        "image_start",
        "image_end",
        "image",
        "image_newline",
        "video_start",
        "video_end",
        "video",
    ):
        setattr(hf_config, f"{name}_token_id", vocab[f"<{name}>"])
    hf = HYV4VLProcessor(
        HYV4VLImageProcessor(
            patch_size=16,
            merge_size=2,
            size={"shortest_edge": 1024, "longest_edge": 4096},
        ),
        tokenizer,
        HYV4VLVideoProcessor(patch_size=16, merge_size=2),
        config=hf_config,
    )

    class LocalProcessingInfo(HYV4VLProcessingInfo):
        def get_hf_processor(self, **kwargs):
            return hf

    info = LocalProcessingInfo(InputProcessingContext(config.model_config, tokenizer))
    processor = HYV4VLMultiModalProcessor(info, HYV4VLDummyInputsBuilder(info))
    return processor, hf, MultiModalProcessorOnlyCache(config.model_config)


def _video():
    return np.zeros((4, 32, 32, 3), dtype=np.uint8), {
        "fps": 2.0,
        "total_num_frames": 4,
        "duration": 2.0,
        "frames_indices": [0, 1, 2, 3],
        "do_sample_frames": False,
    }


def _process(pipeline, intervals, uuids, *, global_options=False):
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    processor, hf, cache = pipeline
    kwargs = {"videos_kwargs": {"do_resize": False}}
    if global_options:
        processor.info.ctx.get_mm_config().mm_processor_kwargs = {
            "second_per_grid_ts": intervals
        }
    else:
        kwargs["second_per_grid_ts"] = intervals
    original = deepcopy(kwargs)
    inputs = ProcessorInputs(
        prompt=hf.tokenizer.encode("<video>" * len(uuids), add_special_tokens=False),
        mm_data_items=processor.info.parse_mm_data(
            {"video": [_video() for _ in uuids]}, validate=False
        ),
        mm_uuid_items={"video": uuids},
        hf_processor_mm_kwargs=kwargs,
        cache=cache,
    )
    original_uuids = deepcopy(inputs.mm_uuid_items)
    result = processor.apply(inputs, TimingContext(enabled=False))
    assert kwargs == original
    assert inputs.mm_uuid_items == original_uuids
    return result


def _timestamps(result):
    return [item["video_timestamps"].data for item in result["mm_kwargs"]["video"]]


@pytest.mark.parametrize("global_options", [False, True])
def test_same_uuid_changed_interval_reprocesses_timestamps_and_prompt(
    pipeline, global_options
):
    first = _process(pipeline, [0.5], ["same"], global_options=global_options)
    repeat = _process(pipeline, [0.5], ["same"], global_options=global_options)
    changed = _process(pipeline, [1.0], ["same"], global_options=global_options)
    assert first["mm_hashes"] == repeat["mm_hashes"]
    assert first["prompt_token_ids"] == repeat["prompt_token_ids"]
    torch.testing.assert_close(
        _timestamps(first)[0], torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float32)
    )
    torch.testing.assert_close(
        _timestamps(changed)[0], torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    )
    assert changed["mm_hashes"] != first["mm_hashes"]
    assert changed["prompt_token_ids"] != first["prompt_token_ids"]


@pytest.mark.parametrize("global_options", [False, True])
def test_multi_video_partial_cache_hit_keeps_per_item_interval(
    pipeline, global_options
):
    warm = _process(pipeline, [0.5, 1.0], ["a", "b"], global_options=global_options)
    partial = _process(pipeline, [0.5, 2.0], ["a", "b"], global_options=global_options)
    assert partial["mm_hashes"]["video"][0] == warm["mm_hashes"]["video"][0]
    assert partial["mm_hashes"]["video"][1] != warm["mm_hashes"]["video"][1]
    torch.testing.assert_close(
        _timestamps(partial)[0], torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float32)
    )
    torch.testing.assert_close(
        _timestamps(partial)[1], torch.tensor([0.0, 2.0, 4.0, 6.0], dtype=torch.float32)
    )


def test_global_interval_vector_is_not_reintroduced_for_missing_subset(
    pipeline, monkeypatch
):
    _, hf, _ = pipeline
    calls = []
    original = type(hf).__call__

    def observe(self, *args, **kwargs):
        calls.append((len(kwargs["videos"]), kwargs.get("second_per_grid_ts")))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(hf), "__call__", observe)
    _process(pipeline, [0.5, 1.0], ["a", "b"], global_options=True)
    _process(pipeline, [0.5, 1.0], ["a", "c"], global_options=True)
    assert calls == [(2, None), (1, None)]


@pytest.mark.parametrize("metadata_interval", [None, 0.5])
def test_uuid_only_cache_retrieval_without_interval_override(
    pipeline, monkeypatch, metadata_interval
):
    from vllm.multimodal.parse import MultiModalDataItems, VideoProcessorItems
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    processor, hf, cache = pipeline
    video = _video()
    if metadata_interval is not None:
        video[1]["second_per_grid_t"] = metadata_interval
    common = {
        "prompt": hf.tokenizer.encode("<video>", add_special_tokens=False),
        "mm_uuid_items": {"video": ["cached-no-override"]},
        "hf_processor_mm_kwargs": {"videos_kwargs": {"do_resize": False}},
        "cache": cache,
    }
    warm = processor.apply(
        ProcessorInputs(
            mm_data_items=processor.info.parse_mm_data(
                {"video": [video]}, validate=False
            ),
            **common,
        ),
        TimingContext(enabled=False),
    )

    def unexpected_hf(*args, **kwargs):
        pytest.fail("UUID-only cache hit must not call the HF processor")

    monkeypatch.setattr(type(hf), "__call__", unexpected_hf)
    cached = processor.apply(
        ProcessorInputs(
            mm_data_items=MultiModalDataItems({"video": VideoProcessorItems([None])}),
            **common,
        ),
        TimingContext(enabled=False),
    )
    assert cached["mm_hashes"] == warm["mm_hashes"]
    assert cached["prompt_token_ids"] == warm["prompt_token_ids"]
    torch.testing.assert_close(_timestamps(cached)[0], _timestamps(warm)[0])


@pytest.mark.parametrize("counts", [{}, {"image": 1, "video": 0}])
def test_disabled_video_does_not_search_video_shape(pipeline, monkeypatch, counts):
    processor, _, _ = pipeline

    def disabled(*args):
        pytest.fail("disabled video modality must not search/allocate video data")

    monkeypatch.setattr(processor.info, "get_dummy_video_shape", disabled)
    data = processor.dummy_inputs.get_dummy_mm_data(128, counts, {})
    assert not data.get("video")
    assert len(data.get("image", [])) == counts.get("image", 0)


@pytest.mark.parametrize("seq_len", [1, 128])
def test_spatial_compression_dummy_shape_is_supported(pipeline, seq_len):
    from hcu_vllm_das.models.hy4_standard.processor import vision_token_count

    processor, hf, _ = pipeline
    config = hf.config.vision_config
    config.spatial_patch_size = 2
    # Keep the configured profiling ceiling even if no candidate fits seq_len.
    processor.info.ctx.get_mm_config().mm_processor_kwargs = {
        "videos_kwargs": {
            "max_frames": 4,
            "max_pixels": 16384,
            "video_max_image_size": 64,
        }
    }
    frames, height, width = processor.info.get_dummy_video_shape(seq_len, {"video": 1})
    assert 2 <= frames <= 4
    assert height % (config.patch_size * config.spatial_merge_size * 2) == 0
    assert width % (config.patch_size * config.spatial_merge_size * 2) == 0
    assert vision_token_count((frames, height // 16, width // 16), config, "video") > 0
    assert processor.info.get_mm_max_tokens_per_item(128, {"video": 1})["video"] >= 8


def test_compressed_dummy_shape_survives_public_profiling_processing(pipeline):
    from vllm.multimodal.processing.context import TimingContext

    processor, hf, _ = pipeline
    hf.config.vision_config.spatial_patch_size = 2
    hf.video_processor.spatial_patch_size = 2
    processor.info.ctx.get_mm_config().mm_processor_kwargs = {
        "videos_kwargs": {
            "max_frames": 4,
            "max_pixels": 16384,
            "video_max_image_size": 64,
        }
    }
    inputs = processor.get_dummy_inputs(128, {"video": 1}, {})
    result = processor.apply(inputs, TimingContext(enabled=False))
    grid = result["mm_kwargs"]["video"][0]["video_grid_thw"].data
    assert 2 <= int(grid[0]) <= 4
    assert int(grid[1]) % 4 == 0
    assert int(grid[2]) % 4 == 0
    assert result["prompt_token_ids"]
    assert result["prompt_token_ids"].count(hf.video_token_id) > 0
    assert len(_timestamps(result)[0]) == int(grid[0])
