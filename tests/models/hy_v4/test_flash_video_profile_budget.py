# SPDX-License-Identifier: Apache-2.0
"""Joint video profile regression: production AST and independent grid oracle.

These stdlib-only tests use an explicit character tokenizer and processing-info
shell. They are source/shape tests, not Torch, Core profile or model evidence.
The real HF dispatch regression is in test_flash_video_resize_budget.py.
"""

import ast
from itertools import product
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[3]
PROCESSOR = ROOT / "vllm_hcu/models/hy_v4/processor.py"
PROFILE = ROOT / "vllm_hcu/models/hy_v4/mm_preprocess.py"


def functions(profile_text=None):
    namespace = {"math": math}
    tree = ast.parse(PROCESSOR.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("adaptive_resize", "vision_token_grid", "vision_token_count")
    ]
    tree = ast.parse(profile_text or PROFILE.read_text(encoding="utf-8"))
    nodes.append(
        next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_dummy_video_shape"
        )
    )
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(PROFILE), "exec"),
        namespace,
    )
    return namespace


def settings(spatial=2, tau=1, newline=True, extra=False, options=None, **defaults):
    config = NS(
        patch_size=16,
        spatial_merge_size=2,
        spatial_patch_size=spatial,
        temporal_patch_size=tau,
        use_image_newline_token=newline,
        cat_extra_token=extra,
        use_2d_rope=False,
        max_image_size=2048,
    )
    processor = NS(
        size={"longest_edge": defaults.get("maximum", 33554432)},
        max_frames=defaults.get("max_frames", 100),
        video_max_image_size=defaults.get("side", 1024),
        do_resize=True,
        timestamp_token_reserve=defaults.get("reserve", 2000),
        timestamp_token_per_frame=defaults.get("per_frame", 4),
    )
    tokenizer = NS(encode=lambda text, **kwargs: list(text))
    hf = NS(video_processor=processor, tokenizer=tokenizer)
    info = NS(
        get_hf_processor=lambda: hf,
        get_hf_config=lambda: NS(vision_config=config),
        _merge_and_resolve_mm_processor_kwargs=lambda _: {
            "videos_kwargs": options or {}
        },
    )
    return info, config, processor, tokenizer


def placeholder_tokens(frames, rows, cols, config, tokenizer):
    # Independent formula for the public wrapper + outer video token pair.
    indices = list(range(frames))
    indices.extend([indices[-1]] * (-len(indices) % config.temporal_patch_size))
    times = [
        sum(indices[i : i + config.temporal_patch_size])
        / config.temporal_patch_size
        / 2
        for i in range(0, len(indices), config.temporal_patch_size)
    ]
    per_frame = rows * (cols + int(config.use_image_newline_token))
    per_frame += 2 * int(bool(config.cat_extra_token))
    return 2 + sum(
        per_frame + 2 + len(tokenizer.encode(f"<{t:.1f} seconds>")) for t in times
    )


class JointProfileBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = functions()

    def run_shape(self, info, length=512, counts=None):
        return self.ns["get_dummy_video_shape"](info, length, counts or {"video": 1})

    def assert_accepted(self, shape, config, processor, options=None):
        options = options or {}
        frames, height, width = shape
        factor = (
            config.patch_size * config.spatial_merge_size * config.spatial_patch_size
        )
        self.assertEqual(height % factor, 0)
        self.assertEqual(width % factor, 0)
        maximum = options.get("max_pixels", processor.size["longest_edge"])
        self.assertLessEqual(frames * height * width, maximum)
        self.assertLessEqual(frames, options.get("max_frames", processor.max_frames))
        if options.get("do_resize", processor.do_resize):
            self.assertEqual(
                self.ns["adaptive_resize"](
                    height,
                    width,
                    frames,
                    config.patch_size,
                    config.spatial_merge_size * config.spatial_patch_size,
                    maximum,
                    options.get("video_max_image_size", processor.video_max_image_size),
                    options.get(
                        "timestamp_token_per_frame", processor.timestamp_token_per_frame
                    ),
                    options.get(
                        "timestamp_token_reserve", processor.timestamp_token_reserve
                    ),
                ),
                (height, width),
            )

    def test_short_context_witness_profiles_at_least_the_accepted_encoder_work(self):
        for spatial, witness_patches in ((2, 2304), (1, 1024)):
            with self.subTest(spatial=spatial):
                info, config, processor, tokenizer = settings(spatial=spatial)
                shape = self.run_shape(info)
                self.assert_accepted(shape, config, processor)
                t, h, w = shape
                patches = t * (h // 16) * (w // 16)
                self.assertGreaterEqual(patches, witness_patches)
                f = 32 * spatial
                self.assertLessEqual(
                    placeholder_tokens(t, h // f, w // f, config, tokenizer), 512
                )
                self.assertNotEqual(shape, (2 if spatial == 2 else 1, f, f))

    def test_small_grid_search_matches_independent_exhaustive_oracle(self):
        for spatial, tau, newline, extra, budget, resize in product(
            (1, 2), (1, 2), (False, True), (False, True), (48, 128, 512), (False, True)
        ):
            options = {"do_resize": resize}
            info, config, processor, tokenizer = settings(
                spatial,
                tau,
                newline,
                extra,
                options=options,
                maximum=81920,
                max_frames=6,
                side=128,
                reserve=0,
                per_frame=1,
            )
            factor = 32 * spatial
            max_cells = processor.size["longest_edge"] // factor**2
            expected = 0
            for frames in range(2 if spatial == 2 else 1, processor.max_frames + 1):
                for rows in range(1, max_cells // frames + 1):
                    for cols in range(1, max_cells // frames // rows + 1):
                        h, w = rows * factor, cols * factor
                        if (
                            placeholder_tokens(frames, rows, cols, config, tokenizer)
                            > budget
                        ):
                            continue
                        if resize and self.ns["adaptive_resize"](
                            h, w, frames, 16, 2 * spatial, 81920, 128, 1, 0
                        ) != (h, w):
                            continue
                        expected = max(expected, frames * rows * cols)
            with self.subTest(
                spatial=spatial,
                tau=tau,
                newline=newline,
                extra=extra,
                budget=budget,
                resize=resize,
            ):
                shape = self.run_shape(info, budget)
                frames, h, w = shape
                self.assertGreater(expected, 0, "oracle fixture must admit some video")
                self.assertEqual(frames * (h // factor) * (w // factor), expected)
                self.assert_accepted(shape, config, processor, options)
                self.assertLessEqual(
                    placeholder_tokens(
                        frames, h // factor, w // factor, config, tokenizer
                    ),
                    budget,
                )

    def test_optional_2d_rope_stays_inside_actual_encoder_position_cache(self):
        info, config, processor, tokenizer = settings()
        config.use_2d_rope = True
        shape = self.run_shape(info)
        frames, height, width = shape
        edge = config.max_image_size // config.patch_size + 1
        self.assertLessEqual(max(height, width) // config.patch_size, edge)
        self.assertGreaterEqual(frames * (height // 16) * (width // 16), 2304)
        self.assert_accepted(shape, config, processor)
        self.assertLessEqual(
            placeholder_tokens(frames, height // 64, width // 64, config, tokenizer),
            512,
        )

    def test_temporally_compressed_frames_may_exceed_token_budget(self):
        info, config, processor, tokenizer = settings(
            tau=100, side=64, reserve=0, per_frame=0
        )
        frames, h, w = self.run_shape(info, 32)
        self.assertEqual(frames, 100)
        self.assertLessEqual(
            placeholder_tokens(frames, h // 64, w // 64, config, tokenizer), 32
        )
        self.assert_accepted((frames, h, w), config, processor)

    def test_explicit_limits_and_shared_media_budget_are_used(self):
        options = {
            "max_pixels": 65536,
            "max_frames": 3,
            "video_max_image_size": 128,
            "timestamp_token_reserve": 0,
            "timestamp_token_per_frame": 0,
        }
        info, config, processor, tokenizer = settings(options=options)
        shape = self.run_shape(info, 512, {"image": 1, "video": 2})
        self.assert_accepted(shape, config, processor, options)
        t, h, w = shape
        self.assertLessEqual(
            placeholder_tokens(t, h // 64, w // 64, config, tokenizer), 512 // 3
        )

    def test_minimum_shape_survives_exhausted_resize_reserve(self):
        info, config, processor, tokenizer = settings(
            maximum=8192, side=64, max_frames=2
        )
        self.assertEqual(self.run_shape(info, 128), (2, 64, 64))

    def test_truly_impossible_context_keeps_existing_minimum_profile_abi(self):
        info, _, _, _ = settings()
        self.assertEqual(self.run_shape(info, 1), (2, 64, 64))

    def test_illegal_single_frame_only_spatial_configuration_rejected(self):
        info, _, _, _ = settings(max_frames=1)
        with self.assertRaisesRegex(ValueError, "max_frames"):
            self.run_shape(info)


if __name__ == "__main__":
    unittest.main()
