# SPDX-License-Identifier: Apache-2.0
"""Video total-token and per-frame resize budgets must apply together.

Run with ``python -B tests/models/hy_v4/test_flash_video_resize_budget.py -v``
(or pytest). Arithmetic tests isolate the production function via AST. Public
HF dispatch / HYV4 wrapper tests use real CPU Torch, torchvision, transformers
and the local HYV4 config; they skip only if Torch/transformers are unavailable.
No model weights, downloads or processor mocks are used; preprocessing is CPU-only.
"""

import ast
import importlib.util
import math
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4/processor.py"
HAS_PROCESSING_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers")
)


class AdaptiveResizeBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "adaptive_resize"
        )
        namespace = {"math": math}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"),
            namespace,
        )
        cls.resize = staticmethod(namespace["adaptive_resize"])

    def test_uhd_dual_budget_witness_and_total_budget_still_wins(self):
        for spatial, frames, expected in (
            (1, 3, (768, 1344)),
            (1, 4, (768, 1344)),
            (2, 3, (768, 1344)),
            (2, 4, (768, 1344)),
            (1, 100, (384, 704)),
            (2, 100, (320, 576)),
        ):
            for transpose in (False, True):
                with self.subTest(spatial=spatial, frames=frames, transpose=transpose):
                    height, width = (3840, 2160) if transpose else (2160, 3840)
                    h, w = self.resize(height, width, frames, merge_size=2 * spatial)
                    self.assertEqual((h, w), expected[::-1] if transpose else expected)
                    factor = 32 * spatial
                    self.assertEqual(h % factor, 0)
                    self.assertEqual(w % factor, 0)
                    self.assertLessEqual(h * w, 1024**2)
                    self.assertLessEqual(frames * h * w, 33554432)
                    # This is an independent count on the aligned output grid,
                    # including newline cells in either orientation.
                    budget = 33554432 // factor**2 - 2000 - 6 * frames
                    tokens = frames * (h // factor) * (w // factor + 1)
                    self.assertLessEqual(tokens, budget)

    def test_independent_constraints_and_unscaled_shapes_are_unchanged(self):
        for spatial, kwargs, expected in (
            (1, {"max_pixels": 1024**4}, (768, 1344)),
            (2, {"max_pixels": 1024**4}, (768, 1344)),
            (1, {"video_max_image_size": 8192}, (2080, 3712)),
            (2, {"video_max_image_size": 8192}, (1856, 3264)),
            (1, {"video_max_image_size": 512}, (384, 672)),
            (2, {"video_max_image_size": 512}, (384, 640)),
        ):
            with self.subTest(spatial=spatial, kwargs=kwargs):
                self.assertEqual(
                    self.resize(2160, 3840, 4, merge_size=2 * spatial, **kwargs),
                    expected,
                )
        for spatial, side, expected in (
            (1, 96, 96),
            (2, 96, 64),
            (2, 128, 128),
            (2, 32, 64),
        ):
            with self.subTest(spatial=spatial, side=side):
                self.assertEqual(
                    self.resize(side, side, 2, merge_size=2 * spatial),
                    (expected, expected),
                )

    def test_timestamp_options_still_reduce_total_budget(self):
        self.assertEqual(
            self.resize(
                2160, 3840, 100, timestamp_token_per_frame=0, timestamp_token_reserve=0
            ),
            (416, 736),
        )
        self.assertEqual(
            self.resize(
                2160,
                3840,
                100,
                timestamp_token_per_frame=100,
                timestamp_token_reserve=10000,
            ),
            (256, 448),
        )

    def test_nonpositive_dimensions_and_budget_still_reject(self):
        for kwargs in (
            {"height": 0},
            {"width": 0},
            {"num_frames": 0},
            {"max_pixels": 0},
            {"patch_size": 0},
            {"merge_size": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.resize(**{"height": 96, "width": 96, "num_frames": 2, **kwargs})


@unittest.skipUnless(
    HAS_PROCESSING_DEPS,
    "public HF dispatch requires Torch, torchvision and transformers",
)
class PublicVideoResizeBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast

        from vllm_hcu.models.hy_v4 import processor

        cls.torch = torch
        cls.processor = processor
        cls.tokenizer_class = PreTrainedTokenizerFast
        cls.tokenizer_backend = Tokenizer
        cls.tokenizer_model = WordLevel

    def make_processors(self, spatial=1):
        tokens = {
            "image_start_token": "<image_start>",
            "image_end_token": "<image_end>",
            "image_token": "<image>",
            "image_newline_token": "<image_newline>",
            "video_start_token": "<video_start>",
            "video_end_token": "<video_end>",
            "video_token": "<video>",
        }
        vocab = {"[UNK]": 0, **{token: i for i, token in enumerate(tokens.values(), 1)}}
        tokenizer = self.tokenizer_class(
            tokenizer_object=self.tokenizer_backend(
                self.tokenizer_model(vocab=vocab, unk_token="[UNK]")
            ),
            unk_token="[UNK]",
            additional_special_tokens=list(tokens.values()),
        )
        config = self.processor.HYV4VLConfig(
            vision_config={
                "spatial_patch_size": spatial,
                "temporal_patch_size": 1,
                "use_image_newline_token": True,
                "cat_extra_token": 0,
            },
            **{f"{name}_id": vocab[token] for name, token in tokens.items()},
        )
        image = self.processor.HYV4VLImageProcessor(spatial_patch_size=spatial)
        video = self.processor.HYV4VLVideoProcessor(spatial_patch_size=spatial)
        wrapper = self.processor.HYV4VLProcessor(image, tokenizer, video, config=config)
        return video, wrapper

    def frames(self, count, height, width):
        # A real CPU tensor with no large input allocation. The real HF resize
        # and patchification still run, without substituting dispatcher methods.
        return self.torch.zeros(1, 3, 1, 1, dtype=self.torch.uint8).expand(
            count, 3, height, width
        )

    @staticmethod
    def options(**kwargs):
        return {
            "do_convert_rgb": False,
            "do_sample_frames": False,
            "do_rescale": False,
            "do_normalize": False,
            **kwargs,
        }

    def check_output(self, output, count, height, width, spatial):
        self.assertEqual(
            output["video_grid_thw"].tolist(), [[count, height // 16, width // 16]]
        )
        pixels = output["pixel_values_videos"]
        self.assertEqual(
            tuple(pixels.shape), (count * (height // 16) * (width // 16), 3 * 16**2)
        )
        self.assertEqual(pixels.device.type, "cpu")
        self.assertEqual(height % (32 * spatial), 0)
        self.assertEqual(width % (32 * spatial), 0)
        self.assertLessEqual(height * width, 1024**2)
        self.assertLessEqual(count * height * width, 33554432)

    def test_public_hf_dispatch_applies_both_budgets(self):
        for spatial in (1, 2):
            video, _ = self.make_processors(spatial)
            for count in (3, 4):
                with self.subTest(spatial=spatial, frames=count):
                    initialized = self.torch.cuda.is_initialized()
                    output = video(
                        [self.frames(count, 2160, 3840)],
                        **self.options(return_tensors="pt"),
                    )
                    self.check_output(output, count, 768, 1344, spatial)
                    self.assertEqual(self.torch.cuda.is_initialized(), initialized)

    def test_public_hyv4_wrapper_preserves_grid_tokens_and_timestamps(self):
        for spatial in (1, 2):
            _, wrapper = self.make_processors(spatial)
            for count in (3, 4):
                with self.subTest(spatial=spatial, frames=count):
                    output = wrapper(
                        text=wrapper.video_token,
                        videos=[self.frames(count, 2160, 3840)],
                        videos_kwargs=self.options(return_tensors="pt"),
                    )
                    self.check_output(output, count, 768, 1344, spatial)
                    expected_tokens = (
                        count * (768 // (32 * spatial)) * (1344 // (32 * spatial) + 1)
                    )
                    ids = output["input_ids"][0]
                    if hasattr(ids, "tolist"):
                        ids = ids.tolist()
                    self.assertEqual(ids.count(wrapper.video_token_id), expected_tokens)
                    self.assertEqual(
                        expected_tokens,
                        self.processor.vision_token_count(
                            output["video_grid_thw"][0],
                            wrapper.config.vision_config,
                            "video",
                        ),
                    )
                    self.torch.testing.assert_close(
                        output["video_timestamps"],
                        self.torch.arange(count, dtype=self.torch.float64) / 24,
                    )

    def test_public_dispatch_and_wrapper_keep_spatial_alignment_boundaries(self):
        for spatial, side, resized, aligned in (
            (1, 96, 96, True),
            (2, 96, 64, False),
            (2, 128, 128, True),
        ):
            video, wrapper = self.make_processors(spatial)
            for use_wrapper in (False, True):

                def process(do_resize):
                    frames = [self.frames(2, side, side)]
                    options = self.options(do_resize=do_resize, return_tensors="pt")
                    if use_wrapper:
                        return wrapper(videos=frames, videos_kwargs=options)
                    return video(frames, **options)

                with self.subTest(spatial=spatial, side=side, wrapper=use_wrapper):
                    self.check_output(process(True), 2, resized, resized, spatial)
                    if aligned:
                        self.check_output(process(False), 2, side, side, spatial)
                    else:
                        with self.assertRaisesRegex(ValueError, "align"):
                            process(False)


if __name__ == "__main__":
    unittest.main()
