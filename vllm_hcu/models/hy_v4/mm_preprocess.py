# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The HunYuan team.

import math
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
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
    adaptive_resize,
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
        newline = int(self.get_hf_config().vision_config.use_image_newline_token)
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "images_kwargs", {}
        )
        factor = processor.patch_size * processor.merge_size
        maximum = options.get("max_pixels", processor.size["longest_edge"])
        minimum = options.get("min_pixels", processor.size["shortest_edge"])
        cells = maximum // factor**2
        if cells < 1 or minimum <= 0 or maximum < minimum:
            raise ValueError("No image shape satisfies pixel bounds and patch alignment")
        if not options.get("do_resize", getattr(processor, "do_resize", True)):
            # The no-resize processor checks alignment/max_pixels, not aspect.
            return ImageSize(width=factor, height=cells * factor)

        best = None

        def consider(height, width):
            nonlocal best
            h, w = smart_resize(height, width, factor, minimum, maximum)
            score = (h // factor) * (w // factor + newline)
            candidate = (score, h, w, height, width)
            if best is None or candidate[:3] > best[:3]:
                best = candidate

        def rounding_interval(count):
            low = max(1, math.ceil((count - 0.5) * factor))
            high = math.floor((count + 0.5) * factor)
            # Python round uses ties-to-even, including the one-cell edge.
            if round(low / factor) != count:
                low += 1
            if round(high / factor) != count:
                high -= 1
            return low, high

        def consider_downscale(ratio):
            height, width = ratio.numerator, ratio.denominator
            scale = max(1, math.isqrt(maximum // (height * width)))
            while (round(height * scale / factor) * round(width * scale / factor)
                   * factor**2 <= maximum):
                scale += 1
            try:
                consider(height * scale, width * scale)
            except ValueError:
                # Clamping a short axis to one cell can exceed max_pixels
                # at an extreme ratio. The real processor rejects that input.
                pass

        # Nearest rounding checks aspect on the INPUT, not the aligned output.
        # For each output row count only its largest reachable column count
        # can maximize rows * (cols + newline). The slack bounds the search;
        # it does not impose a new limit on requests.
        extent = min(cells, math.isqrt(cells * 200) + 201)
        for rows in range(1, extent + 1):
            low_h, high_h = rounding_interval(rows)
            cols = min(cells // rows, round(200 * high_h / factor))
            if not cols or rows * cols * factor**2 < minimum:
                continue
            if max(rows, cols) <= 200 * min(rows, cols):
                consider(rows * factor, cols * factor)
                continue
            low_w, high_w = rounding_interval(cols)
            height = max(low_h, math.ceil(low_w / 200))
            width = max(low_w, math.ceil(height / 200))
            if height <= high_h and width <= high_w:
                consider(height, width)

        # Downscaling floors both axes and may return fewer than min_pixels.
        # Its normalized axes are sqrt(area * ratio), sqrt(area / ratio).
        # For each row count, intersect their floor intervals with [1/200,200]
        # and choose an integer input ratio strictly inside. Upscaling needs no
        # extra search: ceil preserves this integer aspect limit, so every
        # accepted upscaled grid was already covered by the aligned candidates.
        area = Fraction(maximum, factor**2)
        edge = math.isqrt(maximum * 200) // factor
        # The original aspect endpoints are closed. At a nonintegral cell
        # budget an endpoint can supply the unique maximal floor grid.
        for ratio in (Fraction(1, 200), Fraction(200)):
            consider_downscale(ratio)
        for rows in range(1, min(cells, edge) + 1):
            cols = min(cells // rows, edge)
            lower = max(Fraction(1, 200),
                        Fraction(rows**2, 1) / area if rows > 1 else 0,
                        area / (cols + 1)**2)
            upper = min(Fraction(200), Fraction((rows + 1)**2, 1) / area,
                        area / cols**2 if cols > 1 else 200)
            if lower >= upper:
                # Single-point intervals were covered by aligned inputs or
                # the closed aspect endpoints above.
                continue
            middle = (lower + upper) / 2
            denominator = 1
            ratio = middle.limit_denominator(denominator)
            while not lower < ratio < upper:
                denominator *= 2
                ratio = middle.limit_denominator(denominator)
            consider_downscale(ratio)

        if best is None:
            raise ValueError("No image shape satisfies pixel bounds and patch alignment")
        _, _, _, height, width = best
        return ImageSize(width=width, height=height)

    def get_dummy_video_shape(self, seq_len, mm_counts):
        processor = self.get_hf_processor().video_processor
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "videos_kwargs", {}
        )
        config = self.get_hf_config().vision_config
        maximum = options.get("max_pixels", processor.size["longest_edge"])
        max_frames = options.get("max_frames", processor.max_frames)
        side = options.get("video_max_image_size", processor.video_max_image_size)
        available = max(1, seq_len // max(1, sum(mm_counts.values())))
        # Spatially compressed video has no single-frame encoding. Search
        # only legal shapes, including when the available token budget is tiny.
        min_frames = 2 if config.spatial_patch_size != 1 else 1
        if max_frames < min_frames:
            raise ValueError("Video max_frames cannot represent the vision configuration")
        factor = config.patch_size * config.spatial_merge_size * config.spatial_patch_size
        best = (0, min_frames, factor, factor)
        for frames in range(min_frames, min(max_frames, available) + 1):
            height, width = adaptive_resize(
                side,
                side,
                frames,
                config.patch_size,
                config.spatial_merge_size * config.spatial_patch_size,
                maximum,
                side,
                options.get(
                    "timestamp_token_per_frame", processor.timestamp_token_per_frame
                ),
                options.get(
                    "timestamp_token_reserve", processor.timestamp_token_reserve
                ),
            )
            grid = (frames, height // config.patch_size, width // config.patch_size)
            count = vision_token_count(grid, config, "video")
            overhead = frames * (processor.timestamp_token_per_frame + 2) + 2
            if count + overhead <= available and count > best[0]:
                best = (count, frames, height, width)
        return best[1:]

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        config = self.get_hf_config().vision_config
        size = self.get_image_size_with_most_features()
        processor = self.get_hf_processor()
        image = processor.image_processor
        image_options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "images_kwargs", {}
        )
        height, width = size.height, size.width
        if image_options.get("do_resize", getattr(image, "do_resize", True)):
            height, width = smart_resize(
                height, width, image.patch_size * image.merge_size,
                image_options.get("min_pixels", image.size["shortest_edge"]),
                image_options.get("max_pixels", image.size["longest_edge"]),
            )
        image_tokens = vision_token_count(
            (1, height // config.patch_size, width // config.patch_size),
            config,
            "image",
        )
        video = processor.video_processor
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
