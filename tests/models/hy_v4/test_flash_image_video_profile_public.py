# SPDX-License-Identifier: Apache-2.0
"""Native CPU public HF preprocessing and real Core 2D-RoPE regressions.

No model weights/downloads, tensor/resize mocks or GPU jobs. Processing-info
context is an explicit shell; this is NOT full Core encoder/GPU acceptance.
Missing Torch/HF skips this lane (never reported as a successful native run).
"""

import importlib.util
import math
from types import MethodType, SimpleNamespace as NS
import unittest
from itertools import product

HAS_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers")
)


@unittest.skipUnless(HAS_DEPS, "native public preprocess requires Torch/HF/Core")
class PublicImageVideoProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from vllm_hcu.models.hy_v4 import processor
        from vllm_hcu.models.hy_v4.mm_preprocess import (
            HYV4VLDummyInputsBuilder,
            HYV4VLProcessingInfo,
        )

        cls.torch, cls.processor = torch, processor
        cls.info_class, cls.builder_class = (
            HYV4VLProcessingInfo,
            HYV4VLDummyInputsBuilder,
        )

    def make_wrapper(
        self, spatial=2, tau=1, newline=True, extra=False, rope=False, cache_side=2048
    ):
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
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        vocab = {token: i for i, token in enumerate(alphabet + specials)}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, additional_special_tokens=specials
        )
        # No BPE merges: timestamp text has exactly its ASCII character length.
        self.assertEqual(
            len(tokenizer.encode("<0.0 seconds>", add_special_tokens=False)), 13
        )
        config = self.processor.HYV4VLConfig(
            vision_config={
                "spatial_patch_size": spatial,
                "temporal_patch_size": tau,
                "use_image_newline_token": newline,
                "cat_extra_token": extra,
                "use_2d_rope": rope,
                "max_image_size": cache_side,
            },
            **{f"{name}_id": vocab[token] for name, token in zip(names, specials)},
        )
        options = dict(spatial_patch_size=spatial, temporal_patch_size=tau)
        image_options = dict(options)
        if cache_side < 512:
            image_options["size"] = {
                "shortest_edge": 1024,
                "longest_edge": cache_side**2,
            }
        image = self.processor.HYV4VLImageProcessor(**image_options)
        video = self.processor.HYV4VLVideoProcessor(**options)
        return self.processor.HYV4VLProcessor(image, tokenizer, video, config=config)

    def info(self, wrapper, options):
        info = NS(
            get_hf_processor=lambda: wrapper,
            get_hf_config=lambda: wrapper.config,
            _merge_and_resolve_mm_processor_kwargs=lambda _: {"videos_kwargs": options},
        )
        for name in ("get_dummy_video_shape", "get_image_size_with_most_features"):
            setattr(info, name, MethodType(getattr(self.info_class, name), info))
        return info

    def raw_frames(self, shape):
        t, h, w = shape
        return self.torch.zeros((1, 3, 1, 1), dtype=self.torch.uint8).expand(t, 3, h, w)

    def public_video(self, wrapper, frames, options):
        t = len(frames)
        return wrapper(
            text=wrapper.get_placeholder("video"),
            videos=[
                (
                    frames,
                    dict(
                        fps=2.0,
                        total_num_frames=t,
                        duration=t / 2,
                        frames_indices=list(range(t)),
                        do_sample_frames=False,
                    ),
                )
            ],
            videos_kwargs={
                "input_data_format": "channels_first",
                "return_tensors": "pt",
                "do_rescale": False,
                "do_normalize": False,
                **options,
            },
        )

    def assert_core_rope(self, grid, config):
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
        from vllm_hcu.models.hy_v4.vision import HYV4VisionTransformer

        # Use the actual Core fixed cache and production 2D indexing on CPU.
        with set_current_vllm_config(VllmConfig()), self.torch.device("cpu"):
            rope = RotaryEmbedding(
                head_size=16,
                rotary_dim=8,
                max_position_embeddings=config.max_image_size // 16 + 1,
                base=config.rope_2d_theta,
                is_neox_style=True,
                dtype=self.torch.float32,
            )
        # Check the fixed Core cache directly: the child vision implementation
        # can grow a local cache, so its fallback must not mask a profile overflow.
        t, h, w = grid
        indices = self.torch.stack(
            (self.torch.arange(h).repeat_interleave(w), self.torch.arange(w).repeat(h)),
            dim=-1,
        ).repeat(t, 1)
        cos_table, sin_table = rope.get_cos_sin(max(h, w))
        expected = cos_table[indices].flatten(1), sin_table[indices].flatten(1)
        shell = self.torch.nn.Module()
        shell.rotary_pos_emb = rope
        shell.register_buffer("_vision_cos_sin_cache", None, persistent=False)
        cos, sin = HYV4VisionTransformer._rope_cos_sin(shell, [grid])
        self.assertEqual(cos.shape[0], math.prod(grid))
        self.assertEqual(cos.shape, sin.shape)
        for actual, reference in zip((cos, sin), expected):
            self.torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def check_video_bound(
        self, wrapper, raw, options, budget, expected_grid=None, expected_tokens=None
    ):
        initialized = self.torch.cuda.is_initialized()
        real = self.public_video(wrapper, self.raw_frames(raw), options)
        grid = real["video_grid_thw"][0].tolist()
        real_patches = real["pixel_values_videos"].shape[0]
        real_tokens = 2 + len(
            wrapper.get_replacement("video", grid, real["video_timestamps"])
        )
        self.assertEqual(real_patches, math.prod(grid))
        self.assertEqual(len(real["input_ids"][0]), real_tokens)
        self.assertLessEqual(real_tokens, budget)
        if expected_grid is not None:
            self.assertEqual(grid, expected_grid)
        if expected_tokens is not None:
            self.assertEqual(real_tokens, expected_tokens)
        if wrapper.config.vision_config.use_2d_rope:
            self.assert_core_rope(grid, wrapper.config.vision_config)
        del real
        info = self.info(wrapper, options)
        # Real Core builder -> raw NumPy video -> public HF preprocess, ONCE.
        data = self.builder_class(info).get_dummy_mm_data(budget, {"video": 1}, {})
        expected_output = info.get_dummy_video_shape(budget, {"video": 1})
        frames, metadata = data["video"][0]
        dummy = wrapper(
            text=wrapper.get_placeholder("video"),
            videos=[(frames, metadata)],
            videos_kwargs={
                "return_tensors": "pt",
                "do_rescale": False,
                "do_normalize": False,
                **options,
            },
        )
        dummy_grid = dummy["video_grid_thw"][0].tolist()
        self.assertEqual(
            dummy_grid,
            [expected_output[0], expected_output[1] // 16, expected_output[2] // 16],
        )
        self.assertGreaterEqual(dummy["pixel_values_videos"].shape[0], real_patches)
        self.assertLessEqual(len(dummy["input_ids"][0]), budget)
        self.assertLessEqual(
            2
            + len(
                wrapper.get_replacement("video", dummy_grid, dummy["video_timestamps"])
            ),
            budget,
        )
        if wrapper.config.vision_config.use_2d_rope:
            self.assert_core_rope(dummy_grid, wrapper.config.vision_config)
        self.assertEqual(self.torch.cuda.is_initialized(), initialized)

    def test_reported_default_raw_video_exceeds_old_90720_patch_profile(self):
        self.check_video_bound(
            self.make_wrapper(), (25, 2688, 3264), {}, 8192, [25, 56, 68], 6682
        )

    def test_reported_skinny_raw_video_exceeds_old_four_patch_profile(self):
        self.check_video_bound(
            self.make_wrapper(spatial=1),
            (1, 32, 4096),
            {"max_pixels": 81920},
            32,
            [1, 2, 16],
            26,
        )

    def test_public_once_resize_matrix_covers_flags_compression_and_rope(self):
        for spatial, tau, newline, extra, rope in product(
            (1, 2), (1, 3), (False, True), (False, True), (False, True)
        ):
            with self.subTest(
                spatial=spatial, tau=tau, newline=newline, extra=extra, rope=rope
            ):
                wrapper = self.make_wrapper(
                    spatial, tau, newline, extra, rope, cache_side=128
                )
                self.check_video_bound(
                    wrapper,
                    (3, 128, 128),
                    {
                        "max_pixels": 81920,
                        "max_frames": 6,
                        "video_max_image_size": 128,
                        "timestamp_token_reserve": 0,
                        "timestamp_token_per_frame": 1,
                    },
                    128,
                )

    def test_public_image_only_default_profile_fits_actual_rope_cache(self):
        for newline in (False, True):
            wrapper = self.make_wrapper(rope=True, newline=newline)
            # Negative control: Core really rejects the old cache-overflow grid.
            with self.assertRaises(IndexError):
                self.assert_core_rope([1, 1638, 10], wrapper.config.vision_config)
            info = self.info(wrapper, {})
            data = self.builder_class(info).get_dummy_mm_data(8192, {"image": 1}, {})
            # Explicit public preprocess, with the actual image processor defaults.
            output = wrapper.image_processor.preprocess(
                data["image"], return_tensors="pt"
            )
            grid = output["image_grid_thw"][0].tolist()
            self.assertEqual(grid, [1, 128, 128])
            self.assertEqual(
                output["pixel_values"].numel() // (3 * 16**2), math.prod(grid)
            )
            self.assertLessEqual(2 + len(wrapper.get_replacement("image", grid)), 8192)
            self.assert_core_rope(grid, wrapper.config.vision_config)


if __name__ == "__main__":
    unittest.main()
