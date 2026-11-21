# SPDX-License-Identifier: Apache-2.0
"""Exercise real profiling/resize functions without framework or model imports."""
import ast
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

MODEL = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4"


def _profile_shape(minimum, maximum, overrides=None, *, newline=True):
    processor = NS(patch_size=16, merge_size=2,
                   size={"shortest_edge": minimum, "longest_edge": maximum})
    info = NS(get_hf_processor=lambda: NS(image_processor=processor),
              get_hf_config=lambda: NS(vision_config=NS(use_image_newline_token=newline)),
              _merge_and_resolve_mm_processor_kwargs=lambda _: {
                  "images_kwargs": overrides or {}})
    nodes = []
    for filename, name in (("processor.py", "smart_resize"),
                           ("mm_preprocess.py", "get_image_size_with_most_features")):
        tree = ast.parse((MODEL / filename).read_text())
        nodes.append(deepcopy(next(n for n in ast.walk(tree)
                                   if isinstance(n, ast.FunctionDef) and n.name == name)))
    namespace = {"math": math, "ImageSize": NS}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<production-image-profile>", "exec"), namespace)
    return namespace["get_image_size_with_most_features"](info), namespace["smart_resize"]


@pytest.mark.parametrize("minimum,maximum,overrides", [
    (512**2, 512**2, None),
    (512**2, 1024**2, {"max_pixels": 512**2}),
    (4096, 512**2, None),
    (128**2, 128**2 + 1023, None),
    (512**2, 1024**2, {"min_pixels": 128**2, "max_pixels": 128**2}),
])
def test_profile_shape_obeys_pixel_bounds_and_real_resize(minimum, maximum, overrides):
    size, resize = _profile_shape(minimum, maximum, overrides)
    limits = overrides or {}
    minimum, maximum = limits.get("min_pixels", minimum), limits.get("max_pixels", maximum)
    assert size.height % 32 == size.width % 32 == 0
    assert minimum <= size.height * size.width <= maximum
    assert max(size.height, size.width) / min(size.height, size.width) <= 200
    assert resize(size.height, size.width, 32, minimum, maximum) == (size.height, size.width)


def test_tight_budget_still_maximizes_image_and_newline_tokens():
    size, _ = _profile_shape(512**2, 512**2)
    # 128 rows of two merge cells use the exact budget and yield 384 tokens
    # including row newlines, versus 272 tokens for the feasible square.
    assert (size.height, size.width) == (4096, 64)


@pytest.mark.parametrize("newline", [False, True])
@pytest.mark.parametrize("maximum", [512**2, 2048**2])
def test_maximum_matches_all_feasible_grids_with_configured_newlines(newline, maximum):
    minimum = 512**2
    size, _ = _profile_shape(minimum, maximum, newline=newline)
    actual = (size.height // 32) * (size.width // 32 + int(newline))
    cells = maximum // 32**2
    # Independent exhaustive oracle, including every smaller column count.
    expected = max(rows * (cols + int(newline))
                   for rows in range(1, cells + 1)
                   for cols in range(1, cells // rows + 1)
                   if rows * cols * 32**2 >= minimum
                   and max(rows, cols) <= 200 * min(rows, cols))
    assert actual == expected


@pytest.mark.parametrize("minimum,maximum", [(1000, 1000), (1025, 1025)])
def test_impossible_aligned_pixel_budget_reports_no_shape(minimum, maximum):
    with pytest.raises(ValueError):
        _profile_shape(minimum, maximum)
