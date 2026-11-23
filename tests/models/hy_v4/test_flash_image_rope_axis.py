# SPDX-License-Identifier: Apache-2.0
"""Fixed image RoPE axes: dependency-free arithmetic + real public HF dispatch.

The AST lane executes production sizing/binding methods, not a HF emulator.
The native lane uses real PIL images, CPU Torch, HF loaders and a local tokenizer;
it requires the normal vLLM environment, but no GPU, weights or downloads.
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
    for name in ("torch", "torchvision", "transformers")
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

    class Backend:
        def _standardize_kwargs(self, size, **kwargs):
            return {"size": NS(**size), **kwargs}

    class Mixin:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    body = [next(n for n in tree.body if getattr(n, "name", "") == "smart_resize")]
    for name, methods, base in (
        (
            "HYV4VLImageProcessor",
            {
                "_get_image_size",
                "get_number_of_image_patches",
                "_standardize_kwargs",
            },
            "Backend",
        ),
        ("HYV4VLProcessor", {"__init__"}, "Mixin"),
    ):
        original = next(n for n in tree.body if getattr(n, "name", "") == name)
        selected = [
            n
            for n in original.body
            if isinstance(n, ast.FunctionDef) and n.name in methods
        ]
        body.append(
            ast.ClassDef(
                name=name,
                bases=[ast.Name(id=base, ctx=ast.Load())],
                keywords=[],
                body=selected,
                decorator_list=[],
            )
        )
    ns = {
        "math": math,
        "Backend": Backend,
        "Mixin": Mixin,
        "PILImageResampling": NS(LANCZOS=1, BICUBIC=3),
    }
    exec(  # noqa: S102 - execute only local production AST selected above
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        ns,
    )
    return ns["HYV4VLImageProcessor"], ns["HYV4VLProcessor"], ns["smart_resize"]


class ImageRopeAxisAstTests(unittest.TestCase):
    def setUp(self):
        self.image_class, self.wrapper_class, self.smart_resize = isolated_classes()

    def image(self, rope=True, cap=2048, patch=16, merge=2, minimum=262144):
        image = self.image_class()
        image.patch_size, image.merge_size = patch, merge
        image.spatial_patch_size = image.temporal_patch_size = 1
        image.min_pixels = image.max_pixels = None
        image.size = {"shortest_edge": minimum, "longest_edge": 4194304}
        image.do_resize = True
        vision = NS(
            use_2d_rope=rope,
            max_image_size=cap,
            patch_size=patch,
            spatial_merge_size=merge,
            spatial_patch_size=1,
            temporal_patch_size=1,
        )
        config = NS(
            vision_config=vision, **{f"{name}_id": i for i, name in enumerate(TOKENS)}
        )
        tokenizer = NS(
            convert_ids_to_tokens=lambda i: TOKENS[i],
            convert_tokens_to_ids=TOKENS.index,
        )
        self.wrapper_class(image, tokenizer, image, config=config)
        return image

    def test_landscape_and_portrait_real_witness(self):
        image = self.image()
        self.assertEqual(self.smart_resize(1024, 3072), (1024, 3072))
        self.assertEqual(image.get_number_of_image_patches(1024, 3072), (42, 128))
        self.assertEqual(image.get_number_of_image_patches(3072, 1024), (128, 42))

    def test_cache_boundary_counts_raw_patches_including_last_entry(self):
        image = self.image(cap=128, merge=1, minimum=256)
        for height, width in ((144, 32), (32, 144)):
            self.assertEqual(
                image.get_number_of_image_patches(height, width, {"do_resize": False}),
                (height // 16, width // 16),
            )
        self.assertEqual(image.get_number_of_image_patches(160, 32), (9, 1))
        with self.assertRaisesRegex(ValueError, "2D RoPE cache.*do_resize=False"):
            image.get_number_of_image_patches(160, 32, {"do_resize": False})
        merged = self.image(cap=128, minimum=256)
        self.assertEqual(merged.get_number_of_image_patches(160, 32), (8, 2))

    def test_no_resize_witness_fails_before_patchification(self):
        for shape in ((1024, 3072), (3072, 1024)):
            with (
                self.subTest(shape=shape),
                self.assertRaisesRegex(ValueError, "2D RoPE cache.*do_resize=False"),
            ):
                self.image().get_number_of_image_patches(*shape, {"do_resize": False})

    def test_pixel_and_grid_overrides_cannot_expand_actual_cache(self):
        image = self.image()
        options = {
            "min_pixels": 1024,
            "max_pixels": 10**9,
            "_rope_axis_patch_limit": 999999,
        }
        for patch, merge in ((8, 2), (16, 1), (16, 2), (32, 4)):
            grid = image.get_number_of_image_patches(
                1024, 3072, {**options, "patch_size": patch, "merge_size": merge}
            )
            self.assertLessEqual(max(grid), 129)
            self.assertTrue(all(axis % merge == 0 for axis in grid))
        self.assertEqual(image._rope_axis_patch_limit, 129)

    def test_size_and_instance_pixel_options_use_shared_standardization(self):
        image = self.image()
        image.min_pixels = 1024
        image.max_pixels = 65536
        self.assertEqual(image.get_number_of_image_patches(512, 512), (16, 16))
        image.min_pixels = image.max_pixels = None
        options = {"size": {"shortest_edge": 1024, "longest_edge": 65536}}
        self.assertEqual(image.get_number_of_image_patches(512, 512, options), (16, 16))
        self.assertEqual(options["size"]["longest_edge"], 65536)

    def test_minimum_conflicts_with_axis_or_aspect_are_explicit(self):
        for image, shape, options in (
            (self.image(cap=128), (128, 128), {}),
            (self.image(), (64, 8192), {}),
            (self.image(), (1024, 3072), {"min_pixels": 2048**2, "max_pixels": 10**9}),
        ):
            with (
                self.subTest(shape=shape),
                self.assertRaisesRegex(ValueError, "min_pixels.*2D RoPE cache"),
            ):
                image.get_number_of_image_patches(*shape, options)
        with self.assertRaisesRegex(ValueError, "cannot fit one merged"):
            self.image(cap=8).get_number_of_image_patches(32, 32)

    def test_rope_disabled_and_unbound_images_preserve_original_resize(self):
        image = self.image(rope=False)
        for shape in ((1024, 3072), (3072, 1024), (96, 160), (4096, 4096)):
            expected = self.smart_resize(*shape, 32, 262144, 4194304)
            self.assertEqual(
                image.get_number_of_image_patches(*shape),
                tuple(axis // 16 for axis in expected),
            )
        self.assertIsNone(image._rope_axis_patch_limit)
        self.assertEqual(
            image.get_number_of_image_patches(1024, 3072, {"do_resize": False}),
            (64, 192),
        )

    def test_sizing_and_cache_validation_are_on_actual_preprocess_path(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        image = next(
            n for n in tree.body if getattr(n, "name", "") == "HYV4VLImageProcessor"
        )
        preprocess = next(
            n for n in image.body if getattr(n, "name", "") == "_preprocess"
        )
        calls = [
            n
            for n in ast.walk(preprocess)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        self.assertTrue(any(n.func.attr == "_get_image_size" for n in calls))
        self.assertFalse(any(n.func.attr == "smart_resize" for n in calls))


@unittest.skipUnless(
    HAS_PROCESSING_DEPS,
    "real public HF image dispatch requires Torch, torchvision and transformers",
)
class PublicImageRopeAxisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from PIL import Image
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast

        from vllm_hcu.models.hy_v4 import processor

        cls.torch, cls.Image, cls.processor = torch, Image, processor
        vocab = {"[UNK]": 0, **{f"<{name}>": i for i, name in enumerate(TOKENS, 1)}}
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]")),
            unk_token="[UNK]",
            additional_special_tokens=[f"<{name}>" for name in TOKENS],
        )
        cls.token_ids = {f"{name}_id": vocab[f"<{name}>"] for name in TOKENS}

    def wrapper(self, rope=True, cap=2048, merge=2, **image_options):
        p = self.processor
        config = p.HYV4VLConfig(
            vision_config={
                "use_2d_rope": rope,
                "max_image_size": cap,
                "spatial_merge_size": merge,
            },
            **self.token_ids,
        )
        image = p.HYV4VLImageProcessor(merge_size=merge, **image_options)
        video = p.HYV4VLVideoProcessor(merge_size=merge)
        return p.HYV4VLProcessor(image, self.tokenizer, video, config=config)

    def picture(self, height=1024, width=3072):
        return self.Image.new("RGB", (width, height), (17, 61, 103))

    def assert_grid(self, wrapper, height, width, options=None, expected=None):
        options = options or {}
        result = wrapper.image_processor(
            self.picture(height, width), return_tensors="pt", **options
        )
        grid = result["image_grid_thw"][0].tolist()
        patches = wrapper.image_processor.get_number_of_image_patches(
            height, width, options
        )
        self.assertEqual(grid, [1, *patches])
        patch = options.get("patch_size", wrapper.image_processor.patch_size)
        self.assertEqual(
            tuple(result["pixel_values"].shape), (grid[1] * grid[2], 3 * patch**2)
        )
        if expected is not None:
            self.assertEqual(grid, expected)
        if wrapper.config.vision_config.use_2d_rope:
            cap = wrapper.config.vision_config.max_image_size // 16 + 1
            # Independently execute the fixed table index boundary on CPU.
            table = self.torch.zeros(cap, 1)
            table[self.torch.arange(grid[1])]
            table[self.torch.arange(grid[2])]
        return result

    def test_direct_public_hf_dispatch_for_witness_and_transpose(self):
        wrapper = self.wrapper()
        self.assert_grid(wrapper, 1024, 3072, expected=[1, 42, 128])
        self.assert_grid(wrapper, 3072, 1024, expected=[1, 128, 42])

    def test_wrapper_public_text_expansion_matches_real_image_tokens(self):
        wrapper = self.wrapper()
        for height, width in ((1024, 3072), (3072, 1024)):
            result = wrapper(
                images=[self.picture(height, width)],
                text=wrapper.get_placeholder("image"),
                return_tensors="pt",
            )
            grid = result["image_grid_thw"][0].tolist()
            self.assertLessEqual(max(grid[1:]), 129)
            count = len(wrapper.get_replacement("image", grid))
            self.assertEqual(
                int((result["input_ids"] == wrapper.image_token_id).sum()), count
            )

    def test_batch_grouping_keeps_grid_and_patch_counts(self):
        wrapper = self.wrapper()
        result = wrapper.image_processor(
            [self.picture(), self.picture(3072, 1024), self.picture()],
            return_tensors="pt",
        )
        grids = [[1, 42, 128], [1, 128, 42], [1, 42, 128]]
        self.assertEqual(result["image_grid_thw"].tolist(), grids)
        self.assertEqual(
            result["pixel_values"].shape[0], sum(h * w for _, h, w in grids)
        )

    def test_fixed_cache_boundary_and_resize_disabled_errors(self):
        wrapper = self.wrapper(cap=128, merge=1, min_pixels=256)
        self.assert_grid(wrapper, 144, 32, {"do_resize": False}, [1, 9, 2])
        self.assert_grid(wrapper, 160, 32, expected=[1, 9, 1])
        for height, width in ((160, 32), (32, 160)):
            with self.assertRaisesRegex(ValueError, "2D RoPE cache.*do_resize=False"):
                wrapper.image_processor(self.picture(height, width), do_resize=False)
        wrapper = self.wrapper()
        self.assert_grid(wrapper, 2048, 32, {"do_resize": False}, [1, 128, 2])
        with self.assertRaisesRegex(ValueError, "2D RoPE cache.*do_resize=False"):
            wrapper(images=self.picture(), images_kwargs={"do_resize": False})

    def test_runtime_overrides_are_safe_and_helpers_match(self):
        wrapper = self.wrapper()
        for options in (
            {"min_pixels": 1024, "max_pixels": 10**9},
            {"size": {"shortest_edge": 1024, "longest_edge": 10**9}},
            {"min_pixels": 1024, "max_pixels": 65536},
            {"min_pixels": 1024, "patch_size": 8, "merge_size": 2},
            {"min_pixels": 1024, "merge_size": 1},
        ):
            with self.subTest(options=options):
                self.assert_grid(wrapper, 1024, 3072, options)
        result = wrapper(
            images=self.picture(),
            images_kwargs={"min_pixels": 1024, "max_pixels": 10**9},
            return_tensors="pt",
        )
        self.assertEqual(result["image_grid_thw"].tolist(), [[1, 42, 128]])

    def test_instance_options_match_real_dispatch(self):
        self.assert_grid(
            self.wrapper(min_pixels=1024, max_pixels=65536),
            512,
            512,
            expected=[1, 16, 16],
        )
        self.assert_grid(self.wrapper(do_resize=False), 512, 512, expected=[1, 32, 32])

    def test_conflicting_minimum_has_same_clear_failure_in_both_paths(self):
        for wrapper, shape, options in (
            (self.wrapper(cap=128), (128, 128), {}),
            (self.wrapper(), (64, 8192), {}),
            (
                self.wrapper(),
                (1024, 3072),
                {"min_pixels": 2048**2, "max_pixels": 10**9},
            ),
        ):
            for helper in (False, True):
                with (
                    self.subTest(helper=helper, shape=shape),
                    self.assertRaisesRegex(ValueError, "min_pixels.*2D RoPE cache"),
                ):
                    if helper:
                        wrapper.image_processor.get_number_of_image_patches(
                            *shape, options
                        )
                    else:
                        wrapper.image_processor(self.picture(*shape), **options)

    def test_rope_disabled_and_rebinding_preserve_unrestricted_witness(self):
        wrapper = self.wrapper(rope=False)
        self.assert_grid(wrapper, 1024, 3072, expected=[1, 64, 192])
        self.assert_grid(wrapper, 3072, 1024, {"do_resize": False}, [1, 192, 64])
        # A component previously bound to a RoPE model must also be cleared.
        enabled = self.wrapper()
        self.processor.HYV4VLProcessor(
            enabled.image_processor,
            self.tokenizer,
            enabled.video_processor,
            config=wrapper.config,
        )
        self.assertIsNone(enabled.image_processor._rope_axis_patch_limit)
        self.assert_grid(wrapper, 1024, 3072, {"do_resize": False}, [1, 64, 192])

    def test_local_from_pretrained_overrides_cannot_replace_model_axis_limit(self):
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
                images_kwargs={
                    "min_pixels": 1024,
                    "max_pixels": 10**9,
                    "_rope_axis_patch_limit": 999999,
                },
            )
            self.assertEqual(loaded.image_processor._rope_axis_patch_limit, 129)
            self.assert_grid(loaded, 1024, 3072, expected=[1, 42, 128])
            loaded = self.processor.HYV4VLProcessor.from_pretrained(
                directory,
                config=wrapper.config,
                tokenizer=self.tokenizer,
                local_files_only=True,
                images_kwargs={"do_resize": False},
            )
            with self.assertRaisesRegex(ValueError, "2D RoPE cache.*do_resize=False"):
                loaded.image_processor(self.picture())


if __name__ == "__main__":
    unittest.main()
