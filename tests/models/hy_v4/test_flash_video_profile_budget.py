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

    def run_shape(self, info, length=512, counts=None, *, return_raw=False):
        return self.ns["get_dummy_video_shape"](
            info, length, counts or {"video": 1}, return_raw=return_raw
        )

    def test_dummy_dimension_options_search_aligned_grids_in_both_orientations(self):
        for height, width in (
            (224, 224),
            (224, None),
            (None, 224),
            (65, 127),
            (127, 65),
            (1, 1),
            (10**9, 10**9),
        ):
            with self.subTest(height=height, width=width):
                info, config, processor, tokenizer = settings()
                config.use_2d_rope = True
                option = NS(height=height, width=width, num_frames=None)
                shape = self.ns["get_dummy_video_shape"](
                    info, 8192, {"video": 1}, video_options=option
                )
                frames, out_height, out_width = shape
                for cap, edge in ((height, out_height), (width, out_width)):
                    self.assertEqual(edge % 64, 0)
                    self.assertLessEqual(edge, 2048)
                    if cap is not None:
                        self.assertLessEqual(edge, max(64, cap // 64 * 64))
                self.assertLessEqual(
                    placeholder_tokens(
                        frames, out_height // 64, out_width // 64, config, tokenizer
                    ),
                    8192,
                )
                if height == width == 224:
                    self.assertEqual(shape, (processor.max_frames, 192, 192))
                if height is None:
                    self.assertGreater(out_height, out_width)
                if width is None:
                    self.assertGreater(out_width, out_height)

    def resize_shape(self, shape, config, processor, options=None):
        options = options or {}
        frames, height, width = shape
        if not options.get("do_resize", processor.do_resize):
            return shape
        height, width = self.ns["adaptive_resize"](
            height,
            width,
            frames,
            config.patch_size,
            config.spatial_merge_size * config.spatial_patch_size,
            options.get("max_pixels", processor.size["longest_edge"]),
            options.get("video_max_image_size", processor.video_max_image_size),
            options.get(
                "timestamp_token_per_frame", processor.timestamp_token_per_frame
            ),
            options.get("timestamp_token_reserve", processor.timestamp_token_reserve),
        )
        return frames, height, width

    def assert_accepted(
        self, shape, info, config, processor, options=None, length=512, counts=None
    ):
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
        # The processed optimum need not be a resize fixed point. Validate a
        # raw witness through exactly ONE production resize; never resize the
        # processed grid and pretend that it was a supported raw input.
        raw = self.run_shape(info, length, counts, return_raw=True)
        self.assertEqual(raw[0], frames)
        self.assertTrue(all(isinstance(value, int) and value > 0 for value in raw))
        self.assertEqual(self.resize_shape(raw, config, processor, options), shape)

    def test_short_context_witness_profiles_at_least_the_accepted_encoder_work(self):
        for spatial, witness_patches in ((2, 2304), (1, 1024)):
            with self.subTest(spatial=spatial):
                info, config, processor, tokenizer = settings(spatial=spatial)
                shape = self.run_shape(info)
                self.assert_accepted(shape, info, config, processor)
                t, h, w = shape
                patches = t * (h // 16) * (w // 16)
                self.assertGreaterEqual(patches, witness_patches)
                f = 32 * spatial
                self.assertLessEqual(
                    placeholder_tokens(t, h // f, w // f, config, tokenizer), 512
                )
                self.assertNotEqual(shape, (2 if spatial == 2 else 1, f, f))

    def test_output_grid_search_matches_exhaustive_upper_bound(self):
        for spatial, tau, newline, extra, budget, resize in product(
            (1, 2),
            (1, 2, 3),
            (False, True),
            (False, True),
            (48, 128, 512),
            (False, True),
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
            expected = None
            for frames in range(2 if spatial == 2 else 1, processor.max_frames + 1):
                for rows in range(1, max_cells // frames + 1):
                    for cols in range(1, max_cells // frames // rows + 1):
                        h, w = rows * factor, cols * factor
                        if (
                            placeholder_tokens(frames, rows, cols, config, tokenizer)
                            > budget
                        ):
                            continue
                        # Exhaust every output grid, rather than the optimized
                        # row/column solver. Non-clamped grids have strict total
                        # resize area and inclusive side-area bounds. A one-cell
                        # short edge can instead be reached by a thin raw input.
                        if resize and min(rows, cols) > 1:
                            resize_cells = max(1, max_cells - frames)
                            if frames * rows * cols >= resize_cells:
                                continue
                            if h * w > processor.video_max_image_size**2:
                                continue
                        candidate = (
                            frames * rows * cols,
                            rows * cols,
                            frames,
                            min(h, w),
                            max(h, w),
                        )
                        if expected is None or candidate > expected:
                            expected = candidate
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
                self.assertIsNotNone(expected, "oracle fixture must admit some video")
                self.assertEqual(shape, expected[2:])
                self.assert_accepted(shape, info, config, processor, options, budget)
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
        self.assert_accepted(shape, info, config, processor)
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
        self.assert_accepted((frames, h, w), info, config, processor, length=32)

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
        self.assert_accepted(
            shape, info, config, processor, options, 512, {"image": 1, "video": 2}
        )
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

    def test_reported_non_idempotent_raw_inputs_are_covered(self):
        info, config, processor, tokenizer = settings()
        real = self.resize_shape((25, 2688, 3264), config, processor)
        self.assertEqual(real, (25, 896, 1088))
        real_patches = real[0] * (real[1] // 16) * (real[2] // 16)
        self.assertEqual(real_patches, 95200)
        self.assertGreater(real_patches, 90720)  # Old fixed-point profile.
        self.assertEqual(placeholder_tokens(25, 14, 17, config, tokenizer), 6682)
        shape = self.run_shape(info, 8192)
        self.assertEqual(shape, (2, 64, 261056))
        self.assertGreaterEqual(
            shape[0] * (shape[1] // 16) * (shape[2] // 16), real_patches
        )
        raw = self.run_shape(info, 8192, return_raw=True)
        self.assertEqual(raw[1], 2)  # Neither H=1 nor H=3: HF channel ambiguity.
        self.assertNotEqual(raw, shape)
        self.assertNotEqual(self.resize_shape(shape, config, processor), shape)
        self.assert_accepted(shape, info, config, processor, length=8192)
        self.assertEqual(placeholder_tokens(2, 1, 4079, config, tokenizer), 8192)
        # The short-edge clamp can exceed the nominal per-frame side-area cap.
        self.assertGreater(shape[1] * shape[2], processor.video_max_image_size**2)

    def test_small_pixel_budget_raw_counterexample_uses_thin_input_clamp(self):
        options = {"max_pixels": 81920}
        info, config, processor, tokenizer = settings(spatial=1, options=options)
        real = self.resize_shape((1, 32, 4096), config, processor, options)
        self.assertEqual(real, (1, 32, 256))
        self.assertEqual(placeholder_tokens(1, 1, 8, config, tokenizer), 26)
        shape = self.run_shape(info, 32)
        self.assertEqual(shape, (1, 32, 448))
        self.assertGreater(shape[0] * (shape[1] // 16) * (shape[2] // 16), 32)
        self.assertEqual(self.run_shape(info, 32, return_raw=True)[1], 2)
        self.assert_accepted(shape, info, config, processor, options, 32)
        self.assertEqual(placeholder_tokens(1, 1, 14, config, tokenizer), 32)

    def test_nonclamped_raw_witness_resizes_once_to_cache_bounded_optimum(self):
        info, config, processor, tokenizer = settings()
        config.use_2d_rope = True
        shape = self.run_shape(info, 8192)
        self.assertEqual(shape, (26, 832, 1152))
        self.assertGreater(min(shape[1:]), 64)  # Exercise non-clamp construction.
        self.assertNotEqual(self.resize_shape(shape, config, processor), shape)
        raw = self.run_shape(info, 8192, return_raw=True)
        self.assertNotEqual(raw, shape)
        self.assertGreater(raw[0] * raw[1] * raw[2], processor.size["longest_edge"])
        self.assert_accepted(shape, info, config, processor, length=8192)
        self.assertGreaterEqual(shape[0] * (shape[1] // 16) * (shape[2] // 16), 95200)
        edge = config.max_image_size // config.patch_size + 1
        self.assertLessEqual(max(shape[1:]) // config.patch_size, edge)
        self.assertLessEqual(placeholder_tokens(26, 13, 18, config, tokenizer), 8192)

    def test_already_fixed_point_raw_witness_is_returned_unchanged(self):
        info, config, processor, _ = settings(spatial=1)
        shape = self.run_shape(info, 32)
        self.assertEqual(shape, (1, 32, 448))
        self.assertEqual(self.run_shape(info, 32, return_raw=True), shape)
        self.assert_accepted(shape, info, config, processor, length=32)

    def test_resize_disabled_does_not_apply_reserves_or_construct_raw_witness(self):
        options = {
            "do_resize": False,
            "timestamp_token_reserve": 1000000,
            "timestamp_token_per_frame": 1000000,
            "video_max_image_size": 1,
        }
        info, config, processor, _ = settings(spatial=1, options=options)
        shape = self.run_shape(info, 128)
        self.assertEqual(self.run_shape(info, 128, return_raw=True), shape)
        self.assert_accepted(shape, info, config, processor, options, 128)
        default_info, _, _, _ = settings(spatial=1, options={"do_resize": False})
        self.assertEqual(shape, self.run_shape(default_info, 128))

    def test_resolved_resize_overrides_match_equivalent_processor_defaults(self):
        for reserve, per_frame in ((0, 0), (4, 1), (1000, 6)):
            options = {
                "max_pixels": 81920,
                "max_frames": 6,
                "video_max_image_size": 128,
                "timestamp_token_reserve": reserve,
                "timestamp_token_per_frame": per_frame,
            }
            override, config, processor, _ = settings(spatial=1, options=options)
            defaults, _, _, _ = settings(
                spatial=1,
                maximum=81920,
                max_frames=6,
                side=128,
                reserve=reserve,
                per_frame=per_frame,
            )
            with self.subTest(reserve=reserve, per_frame=per_frame):
                for raw in (False, True):
                    self.assertEqual(
                        self.run_shape(override, 128, return_raw=raw),
                        self.run_shape(defaults, 128, return_raw=raw),
                    )
                self.assert_accepted(
                    self.run_shape(override, 128),
                    override,
                    config,
                    processor,
                    options,
                    128,
                )

    def test_one_pass_raw_space_not_only_aligned_fixed_points(self):
        # Independent RAW inputs, not the output-grid feasibility inequalities
        # used by the exhaustive oracle. Include transposes, sub-factor edges,
        # floor boundaries, and widths much larger than the per-frame side cap.
        edges = (1, 2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 1024)
        accepted = 0
        for spatial, tau, newline, extra in product(
            (1, 2), (1, 3), (False, True), (False, True)
        ):
            info, config, processor, tokenizer = settings(
                spatial,
                tau,
                newline,
                extra,
                maximum=81920,
                max_frames=6,
                side=128,
                reserve=0,
                per_frame=1,
            )
            factor = 32 * spatial
            profile = self.run_shape(info, 128)
            work = profile[0] * (profile[1] // factor) * (profile[2] // factor)
            for frames in range(2 if spatial == 2 else 1, 7):
                for height, width in product(edges, repeat=2):
                    _, h, w = self.resize_shape(
                        (frames, height, width), config, processor
                    )
                    if frames * h * w > processor.size["longest_edge"]:
                        continue
                    if (
                        placeholder_tokens(
                            frames, h // factor, w // factor, config, tokenizer
                        )
                        > 128
                    ):
                        continue
                    accepted += 1
                    self.assertLessEqual(
                        frames * (h // factor) * (w // factor),
                        work,
                        (spatial, tau, newline, extra, frames, height, width, h, w),
                    )
            self.assert_accepted(profile, info, config, processor, length=128)
        self.assertGreater(accepted, 1000)  # Ensure the scan is not vacuous.

    def test_tiny_context_raw_query_preserves_minimum_abi_not_a_fit_claim(self):
        for resize in (False, True):
            info, config, _, tokenizer = settings(options={"do_resize": resize})
            shape = self.run_shape(info, 1)
            self.assertEqual(shape, (2, 64, 64))
            self.assertEqual(self.run_shape(info, 1, return_raw=True), shape)
            self.assertGreater(placeholder_tokens(2, 1, 1, config, tokenizer), 1)

    def test_unconstructible_nonclamped_raw_witness_fails_without_underprofiling(self):
        info, config, _, _ = settings()
        config.use_2d_rope = True
        namespace = functions()
        # Inject failure only in the bounded inverse-construction path; the
        # output-grid optimum must not silently degrade to a smaller fixed point.
        namespace["adaptive_resize"] = lambda *args: (64, 64)
        with self.assertRaisesRegex(RuntimeError, "resize witness"):
            namespace["get_dummy_video_shape"](
                info, 8192, {"video": 1}, return_raw=True
            )

    def test_video_grid_rejects_too_small_2d_rope_cache(self):
        info, config, _, _ = settings()
        config.use_2d_rope = True
        config.max_image_size = 16
        for raw in (False, True):
            with self.subTest(return_raw=raw):
                with self.assertRaisesRegex(ValueError, "2D RoPE cache"):
                    self.run_shape(info, 128, return_raw=raw)


if __name__ == "__main__":
    unittest.main()
