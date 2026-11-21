# SPDX-License-Identifier: Apache-2.0
"""Profile the image grids accepted before aspect-ratio rounding."""
import ast
from fractions import Fraction
import math
from pathlib import Path
import random
from types import SimpleNamespace as NS

import pytest

MODEL = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4"


def _functions():
    nodes = []
    for filename, names in (
        ("processor.py", {"smart_resize", "vision_token_grid", "vision_token_count"}),
        ("mm_preprocess.py", {"get_image_size_with_most_features", "get_mm_max_tokens_per_item"}),
    ):
        nodes.extend(n for n in ast.walk(ast.parse((MODEL / filename).read_text()))
                     if isinstance(n, ast.FunctionDef) and n.name in names)
    namespace = {"math": math, "Fraction": Fraction, "ImageSize": NS}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<production-rounding-profile>", "exec"), namespace)
    return namespace


def _profile(minimum, maximum, newline=True, *, do_resize=True):
    functions = _functions()
    config = NS(patch_size=16, spatial_merge_size=2, spatial_patch_size=1,
                temporal_patch_size=1, use_image_newline_token=newline, cat_extra_token=False)
    image = NS(patch_size=16, merge_size=2, do_resize=do_resize,
               size={"shortest_edge": minimum, "longest_edge": maximum})
    video = NS(max_frames=2, size={"longest_edge": 4096})
    info = NS(get_hf_config=lambda: NS(vision_config=config),
              get_hf_processor=lambda: NS(image_processor=image, video_processor=video),
              _merge_and_resolve_mm_processor_kwargs=lambda _: {
                  "images_kwargs": {"do_resize": do_resize}})
    info.get_image_size_with_most_features = lambda: functions["get_image_size_with_most_features"](info)
    size = info.get_image_size_with_most_features()
    output = functions["smart_resize"](size.height, size.width, 32, minimum, maximum) if do_resize else (size.height, size.width)
    budget = functions["get_mm_max_tokens_per_item"](info, 4096, {"image": 1})["image"]
    actual = functions["vision_token_count"]((1, output[0] // 16, output[1] // 16), config, "image")
    return functions, size, output, budget, actual


@pytest.mark.parametrize("height,width", [(8192, 41), (41, 8192)])
def test_profile_covers_input_ratio_checked_before_rounding(height, width):
    functions, _, _, budget, _ = _profile(512**2, 512**2)
    h, w = functions["smart_resize"](height, width, 32, 512**2, 512**2)
    assert max(h, w) > 200 * min(h, w)
    assert budget >= (h // 32) * (w // 32 + 1)


@pytest.mark.parametrize("pixels,expected", [(512**2, 512), (1024**2, 1364), (2048**2, 4914)])
@pytest.mark.parametrize("newline", [False, True])
def test_profile_counts_actual_resized_grid_for_tight_limits(pixels, expected, newline):
    _, size, output, budget, actual = _profile(pixels, pixels, newline)
    assert max(size.height, size.width) <= 200 * min(size.height, size.width)
    assert output[0] % 32 == output[1] % 32 == 0
    assert output[0] * output[1] <= pixels
    assert budget == actual == (expected if newline else pixels // 32**2)


def test_downscale_floor_output_can_be_below_minimum():
    functions, _, output, budget, actual = _profile(1025, 1025)
    assert functions["smart_resize"](64, 64, 32, 1025, 1025) == (32, 32)
    assert output == (32, 32) and budget == actual == 2


def test_downscale_floor_witness_for_tight_1024_budget():
    functions, _, _, budget, _ = _profile(1024**2, 1024**2)
    # Twice the 341x3 output grid takes the downscale branch. Flooring leaves
    # 1023 cells below min_pixels, yet it is accepted and has 1364 tokens.
    assert functions["smart_resize"](21824, 192, 32, 1024**2, 1024**2) == (10912, 96)
    assert budget == 341 * (3 + 1)


def test_closed_input_aspect_endpoint_survives_fractional_cell_budget():
    functions, _, _, budget, _ = _profile(860672, 860672)
    # 840.5 cells: the maximal row count occurs exactly at input ratio 200.
    assert functions["smart_resize"](13600, 68, 32, 860672, 860672) == (13120, 64)
    assert budget == 410 * (2 + 1)


@pytest.mark.parametrize("newline", [False, True])
def test_no_resize_uses_actual_aligned_pixel_budget_without_ratio_limit(newline):
    _, size, output, budget, actual = _profile(4096, 512**2, newline, do_resize=False)
    assert (size.height, size.width) == output
    assert output[0] * output[1] == 512**2
    assert budget == actual == 256 * (1 + int(newline))


@pytest.mark.parametrize("minimum,maximum", [(4096, 512**2), (512**2, 512**2),
                                           (512**2, 2048**2), (1024**2, 1024**2),
                                           (512**2-1, 512**2-1), (512**2+1, 512**2+1),
                                           (860672, 860672), (1024**2+511, 1024**2+511)])
@pytest.mark.parametrize("newline", [False, True])
def test_budget_covers_real_resize_on_independent_input_samples(minimum, maximum, newline):
    functions, _, _, budget, actual = _profile(minimum, maximum, newline)
    assert budget == actual
    random_source = random.Random(493)
    # Raw images, not grids constrained by the resized aspect ratio. Include
    # both extreme orientations and large inputs which take the floor branch.
    shapes = [(8192, 41), (41, 8192), (32000, 160), (160, 32000)]
    shapes += [(short * ratio + jitter, short)
               for short in (1, 17, 31, 32, 41, 47, 48, 64, 80, 160, 1024)
               for ratio in (1, 2, 16, 199, 200) for jitter in (-1, 0, 1)]
    shapes += [(random_source.randint(1, 100000), random_source.randint(1, 100000))
               for _ in range(500)]
    accepted = 0
    for height, width in shapes:
        if min(height, width) <= 0:
            continue
        try:
            h, w = functions["smart_resize"](height, width, 32, minimum, maximum)
        except ValueError:
            continue
        accepted += 1
        assert (h // 32) * (w // 32 + int(newline)) <= budget, (height, width, h, w)
    assert accepted > 400
