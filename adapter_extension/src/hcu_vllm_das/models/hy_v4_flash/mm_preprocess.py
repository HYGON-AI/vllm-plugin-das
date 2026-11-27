# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The HunYuan team.

import math
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from functools import cached_property, partial

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
from vllm.transformers_utils.processor import get_processor_kwargs_keys

from .configs import HYV4FlashVLConfig
from .processor import (
    HYV4VLProcessor,
    HYV4VLProcessorKwargs,
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


class HYV4FlashVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> HYV4FlashVLConfig:
        return self.ctx.get_hf_config(HYV4FlashVLConfig)

    def _load_processor(self, kwargs) -> HYV4VLProcessor:
        # This processor belongs to the installed adapter, not Transformers'
        # AutoProcessor registry. Separate call-time options from init options
        # using the actual Core helper and reuse the renderer's tokenizer.
        dynamic_keys = get_processor_kwargs_keys(HYV4VLProcessorKwargs)
        options = {
            key: value
            for key, value in self.ctx.get_merged_mm_kwargs(kwargs).items()
            if key not in dynamic_keys
        }
        options["config"] = self.get_hf_config()
        options["tokenizer"] = self.ctx.tokenizer
        return HYV4VLProcessor.from_pretrained(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            **options,
        )

    @cached_property
    def _default_processor(self) -> HYV4VLProcessor:
        return self._load_processor({})

    def get_hf_processor(self, **kwargs) -> HYV4VLProcessor:
        return self._load_processor(kwargs) if kwargs else self._default_processor

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
        config = self.get_hf_config().vision_config
        newline = int(config.use_image_newline_token)
        options = self._merge_and_resolve_mm_processor_kwargs({}).get(
            "images_kwargs", {}
        )
        factor = processor.patch_size * processor.merge_size
        maximum = options.get("max_pixels", processor.size["longest_edge"])
        minimum = options.get("min_pixels", processor.size["shortest_edge"])
        cells = maximum // factor**2
        if cells < 1 or minimum <= 0 or maximum < minimum:
            raise ValueError("No image shape satisfies pixel bounds and patch alignment")
        max_edge = None
        if getattr(config, "use_2d_rope", False):
            # Cache positions index raw patches, before the spatial merger.
            max_edge = (config.max_image_size // config.patch_size + 1) // (
                processor.merge_size
            )
            if max_edge < 1:
                raise ValueError("No image shape fits the vision 2D RoPE cache")
        if not options.get("do_resize", getattr(processor, "do_resize", True)):
            # No resize checks alignment/max_pixels, not aspect/min_pixels.
            if max_edge is None:
                return ImageSize(width=factor, height=cells * factor)
            _, rows, cols = max(
                (rows * (min(cells // rows, max_edge) + newline),
                 rows, min(cells // rows, max_edge))
                for rows in range(1, min(cells, max_edge) + 1)
            )
            return ImageSize(width=cols * factor, height=rows * factor)

        best = None

        def consider(height, width):
            nonlocal best
            h, w = smart_resize(height, width, factor, minimum, maximum)
            # Raw witnesses may be unaligned or larger than the cache.
            # Only the actual public resize output indexes its positions.
            if max_edge is not None and (
                h // factor > max_edge or w // factor > max_edge
            ):
                return
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
        if max_edge is not None:
            extent = min(extent, max_edge)
        for rows in range(1, extent + 1):
            low_h, high_h = rounding_interval(rows)
            cols = min(cells // rows, round(200 * high_h / factor))
            if max_edge is not None:
                cols = min(cols, max_edge)
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
        if max_edge is not None:
            edge = min(edge, max_edge)
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
            if max_edge is not None:
                raise ValueError("No image shape fits the vision 2D RoPE cache")
            raise ValueError("No image shape satisfies pixel bounds and patch alignment")
        _, _, _, height, width = best
        return ImageSize(width=width, height=height)

    def get_dummy_video_shape(
        self, seq_len, mm_counts, *, return_raw=False, video_options=None
    ):
        """Maximal processed grid, or a mathematical raw resize witness.

        Profiling allocates the bounded processed grid and disables video resize
        only on its dummy ProcessorInputs. A raw witness can be arbitrarily large:
        return_raw is for arithmetic checks, never for allocating dummy media.
        """
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
        # Core options describe raw profile dimensions. Consume them BEFORE the
        # processed-grid search: Core's later min() crop is not patch aligned.
        # The minimum-edge clamp can legitimately enlarge a sub-factor input.
        height_cells = width_cells = None
        if video_options is not None:
            if video_options.height is not None:
                height_cells = max(1, video_options.height // factor)
            if video_options.width is not None:
                width_cells = max(1, video_options.width // factor)
            if video_options.num_frames is not None:
                # A spatially compressed video requires at least two frames;
                # conservatively profile that minimum even for a one-frame hint.
                max_frames = max(min_frames, min(max_frames, video_options.num_frames))
        max_cells = maximum // factor**2
        newline = int(config.use_image_newline_token)
        extra = 2 * int(bool(config.cat_extra_token))
        # The optional vision 2D RoPE uses a fixed position-edge cache.
        max_edge = None
        if config.use_2d_rope:
            max_edge = (config.max_image_size // config.patch_size + 1) // (
                config.spatial_merge_size * config.spatial_patch_size
            )
        if max_edge is not None and max_edge < 1:
            raise ValueError("Minimum video grid exceeds the vision 2D RoPE cache")
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
            resize_budget = max(
                1, max_cells - timestamp_reserve - timestamp_per_frame * frames
            )
            # Without axis-specific options, swapping portrait to landscape
            # preserves patches and limits while not increasing newline tokens.
            # Otherwise retain both orientations. Solve cols per row rather
            # than scanning every raw pixel size.
            if height_cells is None and width_cells is None:
                row_limit = math.isqrt(max(0, min(frame_cells, visual_cells)))
            else:
                # Width/height overrides break portrait/landscape symmetry.
                row_limit = min(frame_cells, max(0, visual_cells // (1 + newline)))
            if height_cells is not None:
                row_limit = min(row_limit, height_cells)
            if max_edge is not None:
                row_limit = min(row_limit, max_edge)
            for rows in range(1, row_limit + 1):
                cols = min(frame_cells // rows, visual_cells // rows - newline)
                if max_edge is not None:
                    cols = min(cols, max_edge)
                if width_cells is not None:
                    cols = min(cols, width_cells)
                if resize and rows > 1 and cols > 1:
                    # For non-clamped outputs, area is bounded by BOTH resize
                    # scales. floor(raw_area) + floor(raw_long_edge) is strictly
                    # greater than raw_area when the long edge >= one cell,
                    # hence the strict total-budget bound. This is NOT the
                    # fixed-point condition rows*cols + cols <= budget/frames.
                    cols = max(
                        1,  # A one-cell width has the same minimum-edge clamp.
                        min(
                            cols,
                            ((resize_budget - 1) // frames) // rows,
                            (side**2 // factor**2) // rows,
                        ),
                    )
                # With either edge==1 the minimum-edge clamp defeats both
                # resize area bounds: thin raw inputs can yield any long edge.
                # Only the post-resize max_pixels, context and RoPE caps apply.
                if cols < 1 or (
                    height_cells is None and width_cells is None and cols < rows
                ):
                    continue
                height, width = rows * factor, cols * factor
                patches = frames * rows * cols
                candidate = (patches, rows * cols, frames, height, width)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            # No complete video placeholder fits at all; retain the smallest
            # legal encoder shape for the existing tiny-context profile ABI.
            return min_frames, factor, factor
        _, _, frames, height, width = best
        if not resize or not return_raw:
            return frames, height, width

        def resized(raw_height, raw_width):
            return adaptive_resize(
                raw_height,
                raw_width,
                frames,
                config.patch_size,
                config.spatial_merge_size * config.spatial_patch_size,
                maximum,
                side,
                timestamp_per_frame,
                timestamp_reserve,
            )

        # Return RAW input dimensions, not the desired output grid. Refeeding
        # an adaptive_resize output is not idempotent. The builder and public
        # HF preprocessor must perform exactly one resize on this witness.
        if resized(height, width) == (height, width):
            return frames, height, width
        if height == factor:
            # A sub-factor short edge clamps to one output cell; W*scale grows without
            # bound. Bracket and bisect a crossing of the desired output width.
            # scale is non-increasing in integer W, so an upward step is <=1
            # raw pixel and cannot skip an aligned output width (even though
            # floor terms can cause downward steps; monotonicity is NOT assumed).
            # Core builds channels-last NumPy videos. H=1 or H=3 would make
            # HF channel inference ambiguous, so use H=2 (patch factor >= 2).
            raw_height = 2
            low, high = 0, width
            while resized(raw_height, high)[1] < width:
                low, high = high, high * 2
            while high - low > 1:
                mid = (low + high) // 2
                if resized(raw_height, mid)[1] >= width:
                    high = mid
                else:
                    low = mid
            if resized(raw_height, high) != (height, width):
                raise RuntimeError("Cannot construct the video profile clamp witness")
            return frames, raw_height, high

        # Without a clamped edge, the limiting scaled area is Q. If the aligned
        # grid is not already a fixed point, Q lies inside its rounding box.
        # Choose an interior aspect ratio and increase raw resolution until the
        # diminishing raw-long-edge overhead leaves exactly this output grid.
        budget = max(1, max_cells - timestamp_reserve - timestamp_per_frame * frames)
        rows, cols = height // factor, width // factor
        area = min(budget / frames, side**2 / factor**2)
        lower, upper = max(rows, area / (cols + 1)), min(rows + 1, area / cols)
        target_h = (lower + upper) / 2
        target_w = area / target_h
        for power in range(64):
            scale = 2**power
            raw_h = math.ceil(target_h * factor * scale)
            raw_w = math.ceil(target_w * factor * scale)
            if resized(raw_h, raw_w) == (height, width):
                return frames, raw_h, raw_w
        # Do not silently substitute a smaller, underprofiling fixed point if
        # numeric construction fails for an unsupported extreme configuration.
        raise RuntimeError("Cannot construct the video profile resize witness")

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


class HYV4VLDummyInputsBuilder(BaseDummyInputsBuilder[HYV4FlashVLProcessingInfo]):
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
            video_options = mm_options.get("video")
            shape_options = (
                {"video_options": video_options} if video_options is not None else {}
            )
            frames, height, width = self.info.get_dummy_video_shape(
                seq_len, mm_counts, **shape_options
            )
            videos = self._get_dummy_videos(
                width=width,
                height=height,
                num_frames=frames,
                num_videos=mm_counts["video"],
                # Options were resolved into an aligned processed grid above.
                # Do not let Core crop it again to unaligned raw dimensions.
                overrides=None,
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


class HYV4VLMultiModalProcessor(BaseMultiModalProcessor[HYV4FlashVLProcessingInfo]):
    def get_dummy_inputs(self, seq_len, mm_counts, mm_options):
        inputs = super().get_dummy_inputs(seq_len, mm_counts, mm_options)
        if mm_counts.get("video", 0):
            # The builder already allocates the conservative processed grid.
            # adaptive_resize is not idempotent: resizing it again underprofiles,
            # while materializing an inverse raw witness can exhaust host RAM.
            # Scope this override to dummy videos; real requests and dummy images
            # retain their configured preprocessing and no shared state is changed.
            kwargs = dict(inputs.hf_processor_mm_kwargs)
            video_kwargs = dict(kwargs.get("videos_kwargs", {}))
            video_kwargs["do_resize"] = False
            kwargs["videos_kwargs"] = video_kwargs
            inputs = replace(inputs, hf_processor_mm_kwargs=kwargs)
        return inputs

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
