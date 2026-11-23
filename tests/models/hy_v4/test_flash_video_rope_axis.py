# SPDX-License-Identifier: Apache-2.0
"""Fixed video RoPE axes: production AST preprocessing/binding + real public HF calls.

The stdlib lane executes production methods with shape/container shells only;
it does not emulate HF dispatch and is not Torch/Core/encoder evidence.
The native lane needs real Torch, torchvision, transformers and vLLM, but no
GPU, weights or downloads. It is intentionally skipped without processing deps.
"""

import ast
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

SOURCE = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4/processor.py"
HAS_PROCESSING_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "torchvision", "transformers", "tokenizers", "vllm")
)
TOKENS = (
    "image_start_token",
    "image_end_token",
    "image_token",
    "image_newline_token",
    "video_start_token",
    "video_end_token",
    "video_token",
)


def isolated_classes():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    class Mixin:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    body = [next(n for n in tree.body if getattr(n, "name", "") == "adaptive_resize")]
    for name, method, base in (
        ("HYV4VLVideoProcessor", "_preprocess", "object"),
        ("HYV4VLProcessor", "__init__", "Mixin"),
    ):
        original = next(n for n in tree.body if getattr(n, "name", "") == name)
        body.append(
            ast.ClassDef(
                name=name,
                bases=[ast.Name(id=base, ctx=ast.Load())],
                keywords=[],
                body=[
                    next(n for n in original.body if getattr(n, "name", "") == method)
                ],
                decorator_list=[],
            )
        )
    namespace = {
        "math": math,
        "Mixin": Mixin,
        "SizeDict": NS,
        "PILImageResampling": NS(BICUBIC=3),
        "BatchFeature": lambda data, tensor_type: data,
        "torch": NS(cat=lambda pixels: pixels, tensor=lambda grids: grids),
        "patchify": lambda video, patch: None,
    }
    exec(  # noqa: S102 - execute only selected local production methods
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace


class VideoRopeAxisAstTests(unittest.TestCase):
    def setUp(self):
        self.ns = isolated_classes()

    def video(self, rope=True, cap=2048, patch=16, merge=2, spatial=2):
        video = self.ns["HYV4VLVideoProcessor"]()
        video.patch_size, video.merge_size = patch, merge
        video.spatial_patch_size, video.temporal_patch_size = spatial, 1
        video.resize = lambda value, size, resample: NS(
            shape=(value.shape[0], 3, size.height, size.width)
        )
        video.rescale_and_normalize = lambda value, *args: value
        video.max_frames = 100
        video.video_max_image_size = 1024
        video.timestamp_token_per_frame, video.timestamp_token_reserve = 4, 2000
        video._rope_axis_patch_limit = 999999  # stale/factory value must be replaced
        self.bind(video, rope=rope, cap=cap)
        return video

    def bind(self, video, rope, cap=2048):
        vision = NS(
            use_2d_rope=rope,
            max_image_size=cap,
            patch_size=video.patch_size,
            spatial_merge_size=video.merge_size,
            spatial_patch_size=video.spatial_patch_size,
            temporal_patch_size=video.temporal_patch_size,
        )
        config = NS(
            vision_config=vision, **{f"{name}_id": i for i, name in enumerate(TOKENS)}
        )
        tokenizer = NS(
            convert_ids_to_tokens=lambda i: TOKENS[i],
            convert_tokens_to_ids=TOKENS.index,
        )
        self.ns["HYV4VLProcessor"](video, tokenizer, video, config=config)

    @staticmethod
    def size(video, height=128, width=4096, frames=4, **options):
        patch = options.get("patch_size", video.patch_size)
        output = video._preprocess(
            [NS(shape=(frames, 3, height, width))],
            do_resize=options.get("do_resize", True),
            size=NS(longest_edge=options.get("max_pixels", 33554432)),
            do_convert_rgb=False,
            do_rescale=False,
            do_normalize=False,
            patch_size=patch,
            merge_size=options.get("merge_size", video.merge_size),
            spatial_patch_size=options.get(
                "spatial_patch_size", video.spatial_patch_size
            ),
            video_max_image_size=options.get("video_max_image_size"),
            timestamp_token_per_frame=options.get("timestamp_token_per_frame"),
            timestamp_token_reserve=options.get("timestamp_token_reserve"),
            # A per-call private override must not replace the bound model cap.
            _rope_axis_patch_limit=999999,
        )
        t, h, w = output["video_grid_thw"][0]
        if t != frames:
            raise AssertionError("Preprocessing changed the frame count")
        return h * patch, w * patch

    def test_real_witness_and_transpose_use_raw_not_compressed_axes(self):
        video = self.video()
        self.assertEqual(
            self.ns["adaptive_resize"](128, 4096, 4, merge_size=4), (128, 4096)
        )
        self.assertEqual(video._rope_axis_patch_limit, 129)
        self.assertEqual(self.size(video), (64, 2048))
        self.assertEqual(self.size(video, 4096, 128), (2048, 64))

    def test_no_resize_rejects_witness_in_both_orientations(self):
        for height, width in ((128, 4096), (4096, 128)):
            with self.assertRaisesRegex(ValueError, "2D RoPE cache.*do_resize=False"):
                self.size(self.video(), height, width, do_resize=False)

    def test_exact_cache_boundary_and_spatial_merge_alignment(self):
        video = self.video(cap=128, merge=1, spatial=1)
        for height, width in ((144, 32), (32, 144)):
            self.assertEqual(
                self.size(video, height, width, do_resize=False), (height, width)
            )
        self.assertEqual(self.size(video, 160, 32), (144, 16))
        with self.assertRaisesRegex(ValueError, "2D RoPE cache.*do_resize=False"):
            self.size(video, 160, 32, do_resize=False)
        self.assertEqual(self.size(self.video(cap=128), 256, 64), (128, 64))
        with self.assertRaisesRegex(ValueError, "cannot fit one merged video patch"):
            self.size(self.video(cap=32), 64, 64)

    def test_pixel_patch_spatial_and_timestamp_overrides_keep_fixed_raw_cap(self):
        video = self.video()
        for patch, merge, spatial in ((8, 2, 2), (16, 1, 1), (16, 2, 3), (32, 4, 2)):
            options = {
                "patch_size": patch,
                "merge_size": merge,
                "spatial_patch_size": spatial,
                "max_pixels": 10**9,
                "video_max_image_size": 8192,
                "timestamp_token_per_frame": 0,
                "timestamp_token_reserve": 0,
            }
            h, w = self.size(video, **options)
            self.assertLessEqual(max(h, w) // patch, 129)
            self.assertEqual(h % (patch * merge * spatial), 0)
            self.assertEqual(w % (patch * merge * spatial), 0)
        self.assertEqual(video._rope_axis_patch_limit, 129)
        self.assertEqual(self.video(patch=8)._rope_axis_patch_limit, 257)

    def test_valid_inputs_and_existing_dual_budgets_are_unchanged(self):
        for spatial in (1, 2):
            video = self.video(spatial=spatial)
            for t, h, w in ((4, 128, 128), (4, 2160, 3840), (100, 2160, 3840)):
                expected = self.ns["adaptive_resize"](
                    h, w, t, merge_size=2 * spatial, timestamp_token_per_frame=4
                )
                self.assertEqual(self.size(video, h, w, frames=t), expected)
            self.assertEqual(self.size(video, 2048, 64, do_resize=False), (2048, 64))
        with self.assertRaisesRegex(ValueError, "align with patch, merge and spatial"):
            self.size(self.video(), 65, 64, do_resize=False)
        with self.assertRaisesRegex(ValueError, "total max_pixels budget"):
            self.size(self.video(), 128, 128, do_resize=False, max_pixels=4096)

    def test_disabled_unbound_and_rebound_components_keep_old_behavior(self):
        video = self.video(rope=False)
        self.assertIsNone(video._rope_axis_patch_limit)
        for resizing in (True, False):
            self.assertEqual(self.size(video, do_resize=resizing), (128, 4096))
        self.bind(video, rope=True)
        self.assertEqual(self.size(video), (64, 2048))
        self.bind(video, rope=False)
        self.assertIsNone(video._rope_axis_patch_limit)
        self.assertEqual(self.size(video), (128, 4096))
        # Match an unbound standalone processor's class default.
        del video._rope_axis_patch_limit
        type(video)._rope_axis_patch_limit = None
        self.assertEqual(self.size(video), (128, 4096))

    def test_private_axis_limit_is_not_a_public_request_option(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        kwargs = next(
            n for n in tree.body if getattr(n, "name", "") == "HYV4VideoProcessorKwargs"
        )
        self.assertNotIn(
            "_rope_axis_patch_limit",
            [n.target.id for n in kwargs.body if isinstance(n, ast.AnnAssign)],
        )


@unittest.skipUnless(
    HAS_PROCESSING_DEPS,
    "real public HF video dispatch requires Torch, torchvision and transformers",
)
class PublicVideoRopeAxisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast

        from vllm_hcu.models.hy_v4 import processor

        cls.torch, cls.processor = torch, processor
        vocab = {"[UNK]": 0, **{f"<{name}>": i for i, name in enumerate(TOKENS, 1)}}
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]")),
            unk_token="[UNK]",
            additional_special_tokens=[f"<{name}>" for name in TOKENS],
        )
        cls.token_ids = {f"{name}_id": vocab[f"<{name}>"] for name in TOKENS}

    def wrapper(self, rope=True, cap=2048, merge=2, spatial=2, tau=1):
        p = self.processor
        config = p.HYV4VLConfig(
            vision_config={
                "use_2d_rope": rope,
                "max_image_size": cap,
                "spatial_merge_size": merge,
                "spatial_patch_size": spatial,
                "temporal_patch_size": tau,
                "use_image_newline_token": True,
                "cat_extra_token": 0,
            },
            **self.token_ids,
        )
        options = {
            "merge_size": merge,
            "spatial_patch_size": spatial,
            "temporal_patch_size": tau,
        }
        return p.HYV4VLProcessor(
            p.HYV4VLImageProcessor(**options),
            self.tokenizer,
            p.HYV4VLVideoProcessor(**options),
            config=config,
        )

    def frames(self, height=128, width=4096):
        return self.torch.zeros(1, 3, 1, 1, dtype=self.torch.uint8).expand(
            4, 3, height, width
        )

    @staticmethod
    def options(**kwargs):
        return dict(
            do_convert_rgb=False,
            do_sample_frames=False,
            do_rescale=False,
            do_normalize=False,
            return_tensors="pt",
            **kwargs,
        )

    def assert_output(self, wrapper, output, grids, patch=16):
        self.assertEqual(output["video_grid_thw"].tolist(), grids)
        self.assertEqual(
            tuple(output["pixel_values_videos"].shape),
            (sum(t * h * w for t, h, w in grids), 3 * patch**2),
        )
        if wrapper.config.vision_config.use_2d_rope:
            vision = wrapper.config.vision_config
            from vllm.config import VllmConfig, set_current_vllm_config
            from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

            from vllm_hcu.models.hy_v4.vision import HYV4VisionTransformer

            # Real pinned Core fixed cache and actual encoder indexing, no
            # substitute position table or GPU/full ViT/model acceptance claim.
            with set_current_vllm_config(VllmConfig()), self.torch.device("cpu"):
                rope = RotaryEmbedding(
                    head_size=16,
                    rotary_dim=8,
                    max_position_embeddings=vision.max_image_size // vision.patch_size
                    + 1,
                    base=vision.rope_2d_theta,
                    is_neox_style=True,
                    dtype=self.torch.float32,
                )
            cos, sin = HYV4VisionTransformer._rope_cos_sin(
                NS(rotary_pos_emb=rope), grids
            )
            self.assertEqual(cos.shape[0], sum(math.prod(grid) for grid in grids))
            self.assertEqual(cos.shape, sin.shape)

    def test_public_hf_witness_transpose_and_batch_patch_counts(self):
        wrapper = self.wrapper()
        output = wrapper.video_processor(
            [self.frames(), self.frames(4096, 128), self.frames()], **self.options()
        )
        self.assert_output(wrapper, output, [[4, 4, 128], [4, 128, 4], [4, 4, 128]])

    def test_wrapper_public_tokens_and_timestamps_follow_clamped_grid(self):
        for tau in (1, 2):
            wrapper = self.wrapper(tau=tau)
            output = wrapper(
                text=wrapper.video_token,
                videos=[self.frames()],
                videos_kwargs=self.options(),
            )
            self.assert_output(wrapper, output, [[4, 4, 128]])
            ids = output["input_ids"][0]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            self.assertEqual(
                ids.count(wrapper.video_token_id), (4 // tau) * 1 * (32 + 1)
            )
            times = (
                self.torch.arange(4, dtype=self.torch.float64).reshape(-1, tau).mean(1)
                / 24
            )
            self.torch.testing.assert_close(output["video_timestamps"], times)

    def test_no_resize_rejects_witness_and_private_request_key_is_rejected(self):
        wrapper = self.wrapper()
        for height, width in ((128, 4096), (4096, 128)):
            with self.assertRaisesRegex(ValueError, "2D RoPE cache.*do_resize=False"):
                wrapper.video_processor(
                    [self.frames(height, width)],
                    **self.options(do_resize=False, max_pixels=10**9),
                )
        with self.assertRaisesRegex(ValueError, "2D RoPE cache.*do_resize=False"):
            wrapper(videos=[self.frames()], videos_kwargs=self.options(do_resize=False))
        # HF's typed video kwargs reject this private key before _preprocess;
        # rejection is a valid fence, not a successful resize/axis invocation.
        with self.assertRaisesRegex(TypeError, "_rope_axis_patch_limit"):
            wrapper.video_processor(
                [self.frames()],
                **self.options(do_resize=False, _rope_axis_patch_limit=999999),
            )
        self.assertEqual(wrapper.video_processor._rope_axis_patch_limit, 129)

    def test_exact_boundary_and_too_small_cache(self):
        wrapper = self.wrapper(cap=128, merge=1, spatial=1)
        output = wrapper.video_processor(
            [self.frames(144, 32)], **self.options(do_resize=False)
        )
        self.assert_output(wrapper, output, [[4, 9, 2]])
        with self.assertRaisesRegex(ValueError, "2D RoPE cache.*do_resize=False"):
            wrapper.video_processor(
                [self.frames(160, 32)], **self.options(do_resize=False)
            )
        with self.assertRaisesRegex(ValueError, "cannot fit one merged video patch"):
            self.wrapper(cap=32).video_processor(
                [self.frames(64, 64)], **self.options()
            )

    def test_request_overrides_cannot_expand_model_cache(self):
        wrapper = self.wrapper()
        for overrides in (
            {
                "max_pixels": 10**9,
                "video_max_image_size": 8192,
            },
            {"size": {"shortest_edge": 4096, "longest_edge": 10**9}},
            {"patch_size": 8, "merge_size": 2, "spatial_patch_size": 2},
            {"merge_size": 1, "spatial_patch_size": 1},
        ):
            with self.subTest(overrides=overrides):
                output = wrapper.video_processor(
                    [self.frames()], **self.options(**overrides)
                )
                grids = output["video_grid_thw"].tolist()
                self.assert_output(
                    wrapper, output, grids, overrides.get("patch_size", 16)
                )
                factor = overrides.get("merge_size", 2) * overrides.get(
                    "spatial_patch_size", 2
                )
                self.assertTrue(all(axis % factor == 0 for axis in grids[0][1:]))
                self.assertEqual(wrapper.video_processor._rope_axis_patch_limit, 129)
        output = wrapper(
            videos=[self.frames()],
            videos_kwargs=self.options(max_pixels=10**9, video_max_image_size=8192),
        )
        self.assert_output(wrapper, output, [[4, 4, 128]])

    def test_valid_disabled_standalone_and_rebinding(self):
        wrapper = self.wrapper()
        output = wrapper.video_processor(
            [self.frames(128, 128)], **self.options(do_resize=False)
        )
        self.assert_output(wrapper, output, [[4, 8, 8]])
        disabled = self.wrapper(rope=False)
        for resize in (True, False):
            output = disabled.video_processor(
                [self.frames()], **self.options(do_resize=resize)
            )
            self.assert_output(disabled, output, [[4, 8, 256]])
        self.processor.HYV4VLProcessor(
            wrapper.image_processor,
            self.tokenizer,
            wrapper.video_processor,
            config=disabled.config,
        )
        self.assertIsNone(wrapper.video_processor._rope_axis_patch_limit)
        output = wrapper.video_processor([self.frames()], **self.options())
        self.assert_output(disabled, output, [[4, 8, 256]])
        standalone = self.processor.HYV4VLVideoProcessor(spatial_patch_size=2)
        self.assertIsNone(standalone._rope_axis_patch_limit)
        output = standalone([self.frames()], **self.options())
        self.assert_output(disabled, output, [[4, 8, 256]])

    def test_local_factory_overrides_are_rebound_to_actual_vision_config(self):
        wrapper = self.wrapper()
        with tempfile.TemporaryDirectory() as directory:
            wrapper.image_processor.save_pretrained(directory)
            wrapper.video_processor.save_pretrained(directory)
            wrapper.config.save_pretrained(directory)
            loaded = self.processor.HYV4VLProcessor.from_pretrained(
                directory,
                config=wrapper.config,
                tokenizer=self.tokenizer,
                local_files_only=True,
                videos_kwargs={
                    "max_pixels": 10**9,
                    "video_max_image_size": 8192,
                    "_rope_axis_patch_limit": 999999,
                },
            )
            self.assertEqual(loaded.video_processor._rope_axis_patch_limit, 129)
            output = loaded.video_processor([self.frames()], **self.options())
            self.assert_output(loaded, output, [[4, 4, 128]])


if __name__ == "__main__":
    unittest.main()
