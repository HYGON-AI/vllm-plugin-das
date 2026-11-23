# SPDX-License-Identifier: Apache-2.0
"""Bounded video profiling allocation and profile-only HF resize overrides.

NativeProfileAllocationTests uses the real Core builder, parser, ProcessorInputs
and public HF wrapper on CPU; no resize/tensor mocks, weights or downloads. A
builder subclass rejects oversized requests BEFORE delegating to Core's actual
NumPy allocation, so reintroducing the 1.2 TB raw witness cannot OOM the runner.
The processing-info/context is an explicit shell, not full Core apply/encoder,
GPU, distributed, or model acceptance. Allow >1 GiB host headroom for HF float
patches/intermediates even though the uint8 dummy video itself is only ~75 MB.

CoreDummyKwargsContractTests patches ONLY the parent get_dummy_inputs result to
exercise preservation of existing ProcessorInputs fields/kwargs; it is a merge
contract test, not public HF integration. StdlibAllocationSpyTests AST-extracts
ONLY the builder method and uses fake shape witnesses/allocations; it is NOT
Torch/HF/Core runtime validation. Missing native dependencies skip the native
lanes, never count as a successful native run.
"""

import ast
from copy import deepcopy
from dataclasses import fields
import importlib.util
import math
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


MODEL = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4"
EXPECTED_SHAPE = (26, 832, 1152)
EXPECTED_BYTES = 74_760_192  # 26 * 832 * 1152 * 3 uint8 channels (71.3 MiB).
ALLOCATION_LIMIT = 128 * 1024**2
NATIVE_MODULES = (
    "torch",
    "torchvision",
    "transformers",
    "tokenizers",
    "numpy",
    "PIL",
    "vllm",
)
HAS_NATIVE = all(importlib.util.find_spec(name) is not None for name in NATIVE_MODULES)
NATIVE_REASON = "requires real Torch/torchvision/HF/tokenizers/NumPy/Pillow/Core"


class StdlibAllocationSpyTests(unittest.TestCase):
    """Source-only safety lane: fake witnesses, no actual arrays or Core imports."""

    def run_builder(self, counts):
        tree = ast.parse((MODEL / "mm_preprocess.py").read_text(encoding="utf-8"))
        builder = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "HYV4VLDummyInputsBuilder"
        )
        method = next(
            n
            for n in builder.body
            if isinstance(n, ast.FunctionDef) and n.name == "get_dummy_mm_data"
        )
        namespace = {}
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]), "<allocation-spy>", "exec"
            ),
            namespace,
        )
        shape_calls, allocations = [], []

        def shape(seq_len, mm_counts, **kwargs):
            self.assertEqual(seq_len, 8192)
            shape_calls.append(dict(kwargs))
            if kwargs.get("return_raw", False):
                return 26, 106532, 147505  # Reported inverse raw witness: >1 TB.
            return EXPECTED_SHAPE

        def allocate(**kwargs):
            nbytes = math.prod(kwargs[k] for k in ("num_frames", "height", "width")) * 3
            allocations.append((kwargs, nbytes))
            self.assertLessEqual(
                nbytes,
                ALLOCATION_LIMIT,
                "builder requested an unbounded raw-video allocation",
            )
            # Only __len__ is used when generating metadata; allocate no pixels.
            return [range(kwargs["num_frames"])] * kwargs["num_videos"]

        info = NS(
            get_image_size_with_most_features=lambda: NS(height=32, width=32),
            get_dummy_video_shape=shape,
        )
        shell = NS(
            info=info, _get_dummy_videos=allocate, _get_dummy_images=lambda **kwargs: []
        )
        data = namespace["get_dummy_mm_data"](shell, 8192, counts, {})
        return data, shape_calls, allocations

    def test_builder_uses_processed_shape_without_requesting_raw_witness(self):
        data, calls, allocations = self.run_builder({"video": 1})
        self.assertEqual(calls, [{}])
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0][1], EXPECTED_BYTES)
        self.assertEqual(len(data["video"][0][0]), 26)
        self.assertEqual(data["video"][0][1]["total_num_frames"], 26)

    def test_no_video_does_not_search_or_allocate_a_video(self):
        for counts in ({}, {"image": 1}, {"video": 0}):
            with self.subTest(counts=counts):
                data, calls, allocations = self.run_builder(counts)
                self.assertEqual(data["video"], [])
                self.assertEqual(calls, [])
                self.assertEqual(allocations, [])


@unittest.skipUnless(HAS_NATIVE, NATIVE_REASON)
class CoreDummyKwargsContractTests(unittest.TestCase):
    """Real Core dataclasses, patched parent result; not native preprocessing."""

    @classmethod
    def setUpClass(cls):
        from vllm.multimodal.parse import MultiModalDataItems
        from vllm.multimodal.processing import BaseMultiModalProcessor
        from vllm.multimodal.processing.inputs import ProcessorInputs
        from vllm_hcu.models.hy_v4.mm_preprocess import HYV4VLMultiModalProcessor

        cls.items_class = MultiModalDataItems
        cls.inputs_class = ProcessorInputs
        cls.base_class = BaseMultiModalProcessor
        cls.processor_class = HYV4VLMultiModalProcessor

    def invoke(self, original, counts):
        # Constructor/context and parent builder are deliberately outside this
        # kwargs-only unit test. The separate native class exercises both.
        shell = object.__new__(self.processor_class)
        with patch.object(
            self.base_class, "get_dummy_inputs", return_value=original
        ) as parent:
            result = shell.get_dummy_inputs(8192, counts, {})
            parent.assert_called_once_with(8192, counts, {})
        return result

    def test_only_video_resize_is_overridden_and_existing_fields_are_preserved(self):
        kwargs = {
            "videos_kwargs": {
                "do_resize": True,
                "max_pixels": 81920,
                "do_normalize": False,
                "fps": 3.0,
            },
            "images_kwargs": {
                "do_resize": True,
                "min_pixels": 1024,
                "max_pixels": 8192,
            },
            "text_kwargs": {"padding": False},
            "common_kwargs": {"return_tensors": "pt"},
            "second_per_grid_ts": [0.5],
        }
        before = deepcopy(kwargs)
        original = self.inputs_class(
            prompt=[7, 8],
            mm_data_items=self.items_class({}),
            mm_uuid_items={"video": ["keep-video-uuid"]},
            hf_processor_mm_kwargs=kwargs,
            media_io_kwargs={"video": {"video_backend": "opencv"}},
        )
        original.cache = (
            object()
        )  # Opaque cache identity must survive dataclass replace.
        result = self.invoke(original, {"video": 1, "image": 1})
        self.assertIsInstance(result, self.inputs_class)
        expected = deepcopy(before)
        expected["videos_kwargs"]["do_resize"] = False
        self.assertEqual(result.hf_processor_mm_kwargs, expected)
        self.assertEqual(
            original.hf_processor_mm_kwargs,
            before,
            "profiling must not mutate the parent's shared nested kwargs",
        )
        for field in fields(original):
            if field.name != "hf_processor_mm_kwargs":
                self.assertIs(
                    getattr(result, field.name), getattr(original, field.name)
                )

    def test_no_video_adds_no_override_even_when_image_or_video_options_exist(self):
        for counts in ({}, {"image": 1}, {"video": 0}, {"image": 1, "video": 0}):
            for kwargs in (
                {},
                {"images_kwargs": {"do_resize": True}},
                {"videos_kwargs": {"do_resize": True, "max_pixels": 81920}},
            ):
                with self.subTest(counts=counts, kwargs=kwargs):
                    original = self.inputs_class(
                        prompt=[],
                        mm_data_items=self.items_class({}),
                        hf_processor_mm_kwargs=deepcopy(kwargs),
                    )
                    result = self.invoke(original, counts)
                    self.assertEqual(result.hf_processor_mm_kwargs, kwargs)
                    self.assertEqual(original.hf_processor_mm_kwargs, kwargs)


@unittest.skipUnless(HAS_NATIVE, NATIVE_REASON)
class NativeProfileAllocationTests(unittest.TestCase):
    """Actual bounded Core allocation -> actual parser -> public HF CPU preprocess."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from vllm_hcu.models.hy_v4 import processor
        from vllm_hcu.models.hy_v4.mm_preprocess import (
            HYV4VLDataParser,
            HYV4VLDummyInputsBuilder,
            HYV4VLMultiModalProcessor,
            HYV4VLProcessingInfo,
        )

        cls.np, cls.torch, cls.processor = np, torch, processor
        cls.parser_class, cls.info_class = HYV4VLDataParser, HYV4VLProcessingInfo
        cls.builder_class, cls.mm_processor_class = (
            HYV4VLDummyInputsBuilder,
            HYV4VLMultiModalProcessor,
        )

    def setUp(self):
        initialized = self.torch.cuda.is_initialized()
        self.addCleanup(
            lambda: self.assertEqual(self.torch.cuda.is_initialized(), initialized)
        )

    def make_wrapper(self, *, small=False):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        names = (
            "image_start_token",
            "image_end_token",
            "image_token",
            "image_newline_token",
            "video_start_token",
            "video_end_token",
            "video_token",
        )
        specials = [f"<{name}>" for name in names]
        vocab = {
            token: i
            for i, token in enumerate(
                sorted(pre_tokenizers.ByteLevel.alphabet()) + specials
            )
        }
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, additional_special_tokens=specials
        )
        self.assertEqual(
            len(tokenizer.encode("<0.0 seconds>", add_special_tokens=False)), 13
        )
        config = self.processor.HYV4VLConfig(
            vision_config={
                "spatial_patch_size": 2,
                "temporal_patch_size": 1,
                "use_image_newline_token": True,
                "cat_extra_token": False,
                "use_2d_rope": True,
                "max_image_size": 2048,
            },
            **{f"{name}_id": vocab[token] for name, token in zip(names, specials)},
        )
        common = {"spatial_patch_size": 2, "temporal_patch_size": 1}
        image_options, video_options = dict(common), dict(common)
        if small:
            image_options["size"] = {"shortest_edge": 1024, "longest_edge": 8192}
            video_options.update(
                size={"shortest_edge": 4096, "longest_edge": 81920},
                max_frames=6,
                video_max_image_size=128,
                timestamp_token_reserve=0,
                timestamp_token_per_frame=1,
            )
        return self.processor.HYV4VLProcessor(
            self.processor.HYV4VLImageProcessor(**image_options),
            tokenizer,
            self.processor.HYV4VLVideoProcessor(**video_options),
            config=config,
        )

    def make_core_processor(self, wrapper):
        parser = self.parser_class(video_needs_metadata=True)
        info = NS(
            ctx=NS(tokenizer=wrapper.tokenizer),
            get_hf_processor=lambda **kwargs: wrapper,
            get_hf_config=lambda: wrapper.config,
            _merge_and_resolve_mm_processor_kwargs=lambda kwargs: deepcopy(kwargs),
            get_data_parser=lambda: parser,
            parse_mm_data=lambda data, validate=True: parser.parse_mm_data(data),
        )
        for name in (
            "get_dummy_video_shape",
            "get_image_size_with_most_features",
            "get_max_frames_per_video",
        ):
            setattr(info, name, MethodType(getattr(self.info_class, name), info))
        base = self.builder_class

        class GuardedActualBuilder(base):
            """Fence oversized requests, then perform the actual Core allocation."""

            def __init__(self, info):
                super().__init__(info)
                self.allocations = []

            def _get_dummy_videos(self, **kwargs):
                request = (kwargs["num_frames"], kwargs["height"], kwargs["width"])
                nbytes = math.prod(request) * 3
                if nbytes > ALLOCATION_LIMIT:
                    raise AssertionError(
                        f"Unsafe Core dummy allocation {request}: {nbytes} bytes; "
                        "builder must use the processed profile, not return_raw=True"
                    )
                videos = super()._get_dummy_videos(**kwargs)
                self.allocations.extend((tuple(v.shape), v.nbytes) for v in videos)
                return videos

        builder = GuardedActualBuilder(info)
        return self.mm_processor_class(info, builder), info, builder

    def public_profile(self, wrapper, inputs, *, images=None):
        from vllm.multimodal.parse import VideoProcessorItems

        kwargs = deepcopy(inputs.hf_processor_mm_kwargs)
        video_kwargs = kwargs.setdefault("videos_kwargs", {})
        self.assertIs(video_kwargs.get("do_resize"), False)
        # Add only output/layout controls, NEVER silently fix the resize flag.
        video_kwargs.update(
            return_tensors="pt",
            input_data_format="channels_last",
            do_rescale=False,
            do_normalize=False,
        )
        videos = inputs.mm_data_items.get_items("video", VideoProcessorItems)
        self.assertEqual(videos.get_count(), 1)
        frames, metadata = videos.get(0)
        text = wrapper.get_placeholder("video")
        if images is not None:
            text = wrapper.get_placeholder("image") + text
            kwargs.setdefault("images_kwargs", {}).update(
                return_tensors="pt", do_rescale=False, do_normalize=False
            )
        output = wrapper(
            text=text, videos=[(frames, metadata)], images=images, **kwargs
        )
        return output, frames

    def test_default_8192_rope_spatial2_allocates_75mb_and_keeps_public_grid(self):
        from vllm.multimodal.processing.inputs import ProcessorInputs

        wrapper = self.make_wrapper()
        core, info, builder = self.make_core_processor(wrapper)
        counts = {"video": 1}
        self.assertEqual(info.get_dummy_video_shape(8192, counts), EXPECTED_SHAPE)
        inputs = core.get_dummy_inputs(8192, counts, {})
        self.assertIsInstance(inputs, ProcessorInputs)
        self.assertEqual(builder.allocations, [((*EXPECTED_SHAPE, 3), EXPECTED_BYTES)])
        output, frames = self.public_profile(wrapper, inputs)
        self.assertEqual(tuple(frames.shape), (*EXPECTED_SHAPE, 3))
        self.assertEqual(frames.dtype, self.np.dtype("uint8"))
        self.assertEqual(frames.nbytes, EXPECTED_BYTES)
        self.assertTrue(frames.flags.c_contiguous)
        grid = output["video_grid_thw"][0].tolist()
        self.assertEqual(grid, [26, 52, 72])
        self.assertEqual(output["pixel_values_videos"].shape[0], 97_344)
        self.assertEqual(output["pixel_values_videos"].numel(), 97_344 * 3 * 16**2)
        self.assertEqual(len(output["video_timestamps"]), 26)
        self.assertEqual(
            len(output["input_ids"][0]),
            2 + len(wrapper.get_replacement("video", grid, output["video_timestamps"])),
        )
        self.assertLessEqual(len(output["input_ids"][0]), 8192)
        # This is a non-idempotent resize fixture: accidentally running the
        # public default resize again MUST differ from the planned grid.
        video = wrapper.video_processor
        resized = self.processor.adaptive_resize(
            832,
            1152,
            26,
            16,
            4,
            video.size["longest_edge"],
            video.video_max_image_size,
            video.timestamp_token_per_frame,
            video.timestamp_token_reserve,
        )
        self.assertNotEqual(resized, (832, 1152))
        self.assertTrue(wrapper.video_processor.do_resize)
        self.assertTrue(wrapper.image_processor.do_resize)

    def test_real_video_still_resizes_after_dummy_inputs_on_the_same_processor(self):
        wrapper = self.make_wrapper(small=True)
        core, _, _ = self.make_core_processor(wrapper)
        core.get_dummy_inputs(128, {"video": 1}, {})
        frames = self.np.zeros((3, 128, 256, 3), dtype=self.np.uint8)
        metadata = dict(
            fps=2.0,
            total_num_frames=3,
            frames_indices=[0, 1, 2],
            do_sample_frames=False,
        )
        # Real request: no dummy kwargs and no explicit do_resize=True escape hatch.
        output = wrapper(
            text=wrapper.get_placeholder("video"),
            videos=[(frames, metadata)],
            videos_kwargs=dict(
                return_tensors="pt",
                input_data_format="channels_last",
                do_rescale=False,
                do_normalize=False,
            ),
        )
        video = wrapper.video_processor
        expected = self.processor.adaptive_resize(
            128,
            256,
            3,
            16,
            4,
            video.size["longest_edge"],
            video.video_max_image_size,
            video.timestamp_token_per_frame,
            video.timestamp_token_reserve,
        )
        self.assertNotEqual(expected, (128, 256))
        self.assertEqual(
            output["video_grid_thw"][0].tolist(),
            [3, expected[0] // 16, expected[1] // 16],
        )
        self.assertTrue(video.do_resize)

    def test_mixed_profile_disables_only_video_and_image_public_resize_still_runs(self):
        from PIL import Image

        wrapper = self.make_wrapper(small=True)
        core, _, _ = self.make_core_processor(wrapper)
        inputs = core.get_dummy_inputs(128, {"video": 1, "image": 1}, {})
        self.assertIsNot(
            inputs.hf_processor_mm_kwargs.get("images_kwargs", {}).get("do_resize"),
            False,
        )
        # Unaligned input would fail public image patch checks without resize.
        output, frames = self.public_profile(
            wrapper, inputs, images=[Image.new("RGB", (130, 70))]
        )
        self.assertEqual(output["image_grid_thw"][0].tolist(), [1, 4, 8])
        self.assertEqual(
            output["video_grid_thw"][0].tolist(),
            [frames.shape[0], frames.shape[1] // 16, frames.shape[2] // 16],
        )
        self.assertTrue(wrapper.image_processor.do_resize)

    def test_no_video_actual_core_inputs_do_not_acquire_resize_overrides(self):
        wrapper = self.make_wrapper(small=True)
        core, _, builder = self.make_core_processor(wrapper)
        for counts in ({}, {"image": 1}, {"video": 0}):
            with self.subTest(counts=counts):
                inputs = core.get_dummy_inputs(128, counts, {})
                self.assertNotIn("videos_kwargs", inputs.hf_processor_mm_kwargs)
                self.assertNotIn("do_resize", inputs.hf_processor_mm_kwargs)
                self.assertEqual(builder.allocations, [])


if __name__ == "__main__":
    unittest.main()
