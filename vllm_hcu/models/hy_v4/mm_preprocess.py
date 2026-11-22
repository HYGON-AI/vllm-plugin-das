# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The HunYuan team.

import math
from collections.abc import Mapping
from dataclasses import replace
from functools import partial

import torch

from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdateDetails,
    cached_encode,
)
from vllm.multimodal.processing.processor import replace_token_matches
from vllm_hcu.models.hy_v4.config import HYV4VLConfig
from vllm_hcu.models.hy_v4.processor import (
    HYV4VLProcessor,
    smart_resize,
    vision_token_count,
)


class HYV4VLDataParser(MultiModalDataParser):
    def _parse_image_data(self, data):
        if isinstance(data, dict) or self.is_embeddings(data):
            raise ValueError("HYV4 VL does not support precomputed image embeddings")
        return super()._parse_image_data(data)

    def _parse_video_data(self, data):
        if isinstance(data, dict) or self.is_embeddings(data):
            raise ValueError("HYV4 VL does not support precomputed video embeddings")
        return super()._parse_video_data(data)


class HYV4VLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> HYV4VLConfig:
        return self.ctx.get_hf_config(HYV4VLConfig)

    def get_hf_processor(self, **kwargs) -> HYV4VLProcessor:
        return self.ctx.get_hf_processor(HYV4VLProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_max_frames_per_video(self) -> int:
        processor = self.get_hf_processor().video_processor
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "videos_kwargs", {}
        )
        return int(options.get("max_frames", processor.max_frames))

    def get_data_parser(self):
        return HYV4VLDataParser(video_needs_metadata=True)

    def get_image_size_with_most_features(self):
        processor = self.get_hf_processor().image_processor
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "images_kwargs", {}
        )
        factor = processor.patch_size * processor.merge_size
        maximum = options.get("max_pixels", processor.size["longest_edge"])
        minimum = options.get("min_pixels", processor.size["shortest_edge"])
        cells = max(1, maximum // factor**2)
        candidates = []
        for rows in range(1, math.isqrt(cells * 200) + 1):
            cols = min(cells // rows, rows * 200)
            if cols and rows <= cols * 200:
                candidates.append((rows * (cols + 1), rows, cols))
        _, rows, cols = max(candidates)
        height, width = smart_resize(
            rows * factor, cols * factor, factor, minimum, maximum
        )
        return ImageSize(width=width, height=height)

    def get_dummy_video_shape(self, seq_len, mm_counts):
        hf_processor = self.get_hf_processor()
        processor = hf_processor.video_processor
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "videos_kwargs", {}
        )
        config = self.get_hf_config().vision_config
        maximum = options.get("max_pixels", processor.size["longest_edge"])
        max_frames = options.get("max_frames", processor.max_frames)
        side = options.get("video_max_image_size", processor.video_max_image_size)
        resize = options.get("do_resize", processor.do_resize)
        timestamp_reserve = options.get(
            "timestamp_token_reserve", processor.timestamp_token_reserve
        )
        timestamp_per_frame = options.get(
            "timestamp_token_per_frame", processor.timestamp_token_per_frame
        )
        available = max(1, seq_len // max(1, sum(mm_counts.values())))
        min_frames = 2 if config.spatial_patch_size != 1 else 1
        if max_frames < min_frames:
            raise ValueError(
                "Video max_frames cannot represent the vision configuration"
            )
        factor = (
            config.patch_size * config.spatial_merge_size * config.spatial_patch_size
        )
        max_cells = maximum // factor**2
        newline = int(config.use_image_newline_token)
        extra = 2 * int(bool(config.cat_extra_token))
        # The optional vision 2D RoPE uses a fixed position-edge cache.
        max_edge = None
        if config.use_2d_rope:
            max_edge = (config.max_image_size // config.patch_size + 1) // (
                config.spatial_merge_size * config.spatial_patch_size
            )
        best = None
        timestamp_lengths = {}
        # Jointly search temporal length and aligned spatial grids. Maximizing
        # ViT input patches (not only compressed LLM tokens) avoids severe
        # underprofiling when no full-resolution video fits a short context.
        for frames in range(min_frames, min(max_frames, max_cells) + 1):
            tau = config.temporal_patch_size
            indices = list(range(frames))
            indices += [indices[-1]] * (-frames % tau)
            timestamps = [
                sum(indices[i : i + tau]) / tau / 2 for i in range(0, len(indices), tau)
            ]
            # Match the dummy builder's fps=2 metadata and the processor's
            # formatted, temporally merged timestamps. Include frame delimiters
            # and the two outer video tokens, not a guessed tokens/frame count.
            overhead = 2 + 2 * len(timestamps)
            for timestamp in timestamps:
                text = f"<{timestamp:.1f} seconds>"
                if text not in timestamp_lengths:
                    timestamp_lengths[text] = len(
                        hf_processor.tokenizer.encode(text, add_special_tokens=False)
                    )
                overhead += timestamp_lengths[text]
            visual_cells = (available - overhead) // len(timestamps) - extra
            frame_cells = max_cells // frames
            if resize:
                frame_cells = min(frame_cells, side**2 // factor**2)
                resize_budget = (
                    max(1, max_cells - timestamp_reserve - timestamp_per_frame * frames)
                    // frames
                )
            else:
                resize_budget = None
            # Swapping portrait to landscape preserves patch count and resize
            # limits while never increasing newline tokens, so rows <= cols
            # suffices. Enumerating rows and solving cols visits the maximum
            # area for every possible row count, without a pixel-size scan.
            row_limit = math.isqrt(max(0, min(frame_cells, visual_cells)))
            if max_edge is not None:
                row_limit = min(row_limit, max_edge)
            for rows in range(1, row_limit + 1):
                cols = min(frame_cells // rows, visual_cells // rows - newline)
                if max_edge is not None:
                    cols = min(cols, max_edge)
                if resize_budget is not None:
                    cols = min(cols, resize_budget // (rows + 1))
                if cols < rows:
                    continue
                height, width = rows * factor, cols * factor
                patches = frames * rows * cols
                candidate = (patches, rows * cols, frames, height, width)
                if best is None or candidate > best:
                    best = candidate
            # adaptive_resize clamps both dimensions to at least one cell even
            # if its reserved token budget is exhausted. Keep that legal edge
            # candidate too, but never let it replace a larger fitting grid.
            minimum_tokens = len(timestamps) * (1 + newline + extra) + overhead
            if minimum_tokens <= available:
                candidate = (frames, 1, frames, factor, factor)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            # No complete video placeholder fits at all; retain the smallest
            # legal encoder shape for the existing tiny-context profile ABI.
            return min_frames, factor, factor
        return best[2:]

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        config = self.get_hf_config().vision_config
        size = self.get_image_size_with_most_features()
        image_tokens = vision_token_count(
            (1, size.height // config.patch_size, size.width // config.patch_size),
            config,
            "image",
        )
        video = self.get_hf_processor().video_processor
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "videos_kwargs", {}
        )
        factor = config.patch_size * config.spatial_merge_size
        max_frames = options.get("max_frames", video.max_frames)
        max_pixels = options.get("max_pixels", video.size["longest_edge"])
        video_tokens = 2 * (max_pixels // factor**2 + max_frames)
        video_tokens += 2 * max_frames * int(bool(config.cat_extra_token))
        return {"image": image_tokens, "video": video_tokens}


class HYV4VLDummyInputsBuilder(BaseDummyInputsBuilder[HYV4VLProcessingInfo]):
    def get_dummy_text(self, mm_counts):
        processor = self.info.get_hf_processor()
        return "".join(
            processor.get_placeholder(modality) * mm_counts.get(modality, 0)
            for modality in ("image", "video")
        )

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        size = self.info.get_image_size_with_most_features()
        videos = []
        if mm_counts.get("video", 0):
            frames, height, width = self.info.get_dummy_video_shape(seq_len, mm_counts)
            videos = self._get_dummy_videos(
                width=width,
                height=height,
                num_frames=frames,
                num_videos=mm_counts["video"],
                overrides=mm_options.get("video"),
            )
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": [
                (
                    video,
                    {
                        "fps": 2.0,
                        "duration": len(video) / 2,
                        "total_num_frames": len(video),
                        "frames_indices": list(range(len(video))),
                        "do_sample_frames": False,
                        "video_backend": "opencv",
                    },
                )
                for video in videos
            ],
        }


class HYV4VLMultiModalProcessor(BaseMultiModalProcessor[HYV4VLProcessingInfo]):
    def apply(self, inputs, timing_ctx):
        processor = self.info.get_hf_processor(**inputs.hf_processor_mm_kwargs)
        prompt = inputs.prompt
        for modality in ("image", "video"):
            if inputs.mm_data_items.get_count(modality, strict=False):
                target = [
                    getattr(processor, f"{modality}_start_token_id"),
                    getattr(processor, f"{modality}_token_id"),
                    getattr(processor, f"{modality}_end_token_id"),
                ]
                prompt = replace_token_matches(
                    prompt,
                    target,
                    cached_encode(
                        processor.tokenizer, f"<{modality}>", add_special_tokens=False
                    ),
                )
        inputs = replace(inputs, prompt=prompt)
        options = self.info.ctx.get_merged_mm_kwargs(inputs.hf_processor_mm_kwargs)
        intervals = options.get("second_per_grid_ts")
        if intervals is not None:
            videos = inputs.mm_data_items.get("video", ())
            if len(intervals) != len(videos):
                raise ValueError("second_per_grid_ts must match video count")
            normalized = []
            for item, interval in zip(videos, intervals):
                if item is None:
                    raise ValueError("Video timing overrides require media metadata")
                frames, metadata = item
                normalized.append(
                    (frames, {**metadata, "second_per_grid_t": float(interval)})
                )
            mm_items = MultiModalDataItems(
                {**inputs.mm_data_items, "video": VideoProcessorItems(normalized)}
            )
            kwargs = dict(inputs.hf_processor_mm_kwargs)
            kwargs.pop("second_per_grid_ts", None)
            inputs = replace(
                inputs, mm_data_items=mm_items, hf_processor_mm_kwargs=kwargs
            )
        return super().apply(inputs, timing_ctx)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        result = {}
        for modality, pixel_key in (
            ("image", "pixel_values"),
            ("video", "pixel_values_videos"),
        ):
            grid_key = f"{modality}_grid_thw"
            grid = hf_inputs.get(grid_key, torch.empty((0, 3), dtype=torch.long))
            result[pixel_key] = MultiModalFieldConfig.flat_from_sizes(
                modality, grid.prod(-1)
            )
            result[grid_key] = MultiModalFieldConfig.batched(modality, keep_on_cpu=True)
            if modality == "video":
                tau = self.info.get_hf_config().vision_config.temporal_patch_size
                result["video_timestamps"] = MultiModalFieldConfig.flat_from_sizes(
                    "video", (grid[:, 0] + tau - 1) // tau, keep_on_cpu=True
                )
        return result

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        def replacement(index: int, modality: str) -> PromptUpdateDetails:
            item = out_mm_kwargs[modality][index]
            grid = item[f"{modality}_grid_thw"].data
            timestamps = item["video_timestamps"].data if modality == "video" else None
            tokens = [
                getattr(processor, f"{modality}_start_token_id"),
                *processor.get_replacement(modality, grid, timestamps),
                getattr(processor, f"{modality}_end_token_id"),
            ]
            token_id = getattr(processor, f"{modality}_token_id")
            return PromptUpdateDetails.select_token_id(tokens, token_id)

        return [
            PromptReplacement(
                modality=modality,
                target=cached_encode(
                    processor.tokenizer, f"<{modality}>", add_special_tokens=False
                ),
                replacement=partial(replacement, modality=modality),
            )
            for modality in ("image", "video")
        ]
