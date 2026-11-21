# SPDX-License-Identifier: Apache-2.0
"""Exercise real profiling/resize functions without framework or model imports."""
import ast
from copy import deepcopy
from fractions import Fraction
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
    namespace = {"math": math, "Fraction": Fraction, "ImageSize": NS}
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
    assert max(size.height, size.width) / min(size.height, size.width) <= 200
    height, width = resize(size.height, size.width, 32, minimum, maximum)
    assert height % 32 == width % 32 == 0
    assert height * width <= maximum


def test_tight_budget_still_maximizes_image_and_newline_tokens():
    size, resize = _profile_shape(512**2, 512**2)
    # The raw aspect is legal; rounding reaches 256 one-cell rows and adds
    # a newline to each. A post-round 200:1 constraint misses this shape.
    assert resize(size.height, size.width, 32, 512**2, 512**2) == (8192, 32)


@pytest.mark.parametrize("newline", [False, True])
@pytest.mark.parametrize("maximum", [512**2, 2048**2])
def test_maximum_matches_independent_raw_input_family(newline, maximum):
    minimum = 512**2
    size, resize = _profile_shape(minimum, maximum, newline=newline)
    height, width = resize(size.height, size.width, 32, minimum, maximum)
    actual = (height // 32) * (width // 32 + int(newline))
    cells = maximum // 32**2
    # Exhaust the aligned RAW inputs up to four times the pixel budget,
    # including downscales, and add independently known rounded panoramas.
    # Acceptance and output sizes come from the real preprocessing function;
    # this is not a claim that aligned output grids preserve the input ratio.
    shapes = ((rows * 32, cols * 32)
              for rows in range(1, 4 * cells + 1)
              for cols in range(1, 4 * cells // rows + 1)
              if max(rows, cols) <= 200 * min(rows, cols))
    expected = 0
    for h, w in (*shapes, (8192, 41), (41, 8192), (21824, 192)):
        try:
            h, w = resize(h, w, 32, minimum, maximum)
        except ValueError:
            continue
        expected = max(expected, (h // 32) * (w // 32 + int(newline)))
    assert actual == expected


@pytest.mark.parametrize("minimum,maximum", [(1000, 1000), (1, 1023)])
def test_impossible_aligned_pixel_budget_reports_no_shape(minimum, maximum):
    with pytest.raises(ValueError):
        _profile_shape(minimum, maximum)


def _vision_rope():
    """Real Core RoPE with only the vision layer/constructor shell isolated."""
    import torch
    from vllm.model_executor.layers.rotary_embedding import get_rope

    source = MODEL / "vision.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_rope_cos_sin")
    namespace = {"torch": torch, "get_rope": get_rope}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    shell = torch.nn.Module()
    shell.rotary_pos_emb = get_rope(head_size=64, max_position=129,
        is_neox_style=True, dtype=torch.float32,
        rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.5})
    shell.register_buffer("_vision_cos_sin_cache", None, persistent=False)
    return shell, namespace["_rope_cos_sin"]


@pytest.fixture
def vision_rope_context():
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config

    initialized = torch.cuda.is_initialized()
    with set_current_vllm_config(VllmConfig()):
        yield
    assert torch.cuda.is_initialized() == initialized


def _check_rope_values(actual, grid):
    import torch

    # Independent unbounded 2D definition, not another cached Core lookup.
    indices = torch.cat([torch.stack((torch.arange(h).repeat_interleave(w),
                                     torch.arange(w).repeat(h)), -1).repeat(t, 1)
                         for t, h, w in grid])
    angles = indices.float()[..., None] / (10000.0 ** (torch.arange(0, 32, 2).float() / 32))
    expected = angles.cos().flatten(1), angles.sin().flatten(1)
    for value, reference in zip(actual, expected):
        assert value.dtype == torch.float32 and value.device.type == "cpu"
        torch.testing.assert_close(value, reference, rtol=0, atol=1e-4)


@pytest.mark.parametrize("newline", [False, True])
@pytest.mark.parametrize("maximum", [512**2, 2048**2])
def test_profile_image_grid_has_complete_vision_rope(newline, maximum, vision_rope_context):
    size, resize = _profile_shape(512**2, maximum, newline=newline)
    height, width = resize(size.height, size.width, 32, 512**2, maximum)
    grid = [[1, height // 16, width // 16]]
    shell, rope = _vision_rope()
    _check_rope_values(rope(shell, grid), grid)


@pytest.mark.parametrize("grid", [[[1, 1638, 10]], [[1, 10, 1638]], [[2, 132, 130]]])
def test_vision_rope_grows_for_rectangular_and_multiframe_grids(grid, vision_rope_context):
    import torch
    from vllm.config import set_current_vllm_config

    shell, rope = _vision_rope()
    before = tuple(value.clone() for value in rope(shell, [[1, 4, 6]]))
    original_cache = shell.rotary_pos_emb.cos_sin_cache.clone()
    # Core CustomOps require a construction context. Runtime growth must
    # work after that context is gone and must not mutate a shared object.
    with set_current_vllm_config(None):
        _check_rope_values(rope(shell, grid), grid)
        pointer = shell._vision_cos_sin_cache.data_ptr()
        _check_rope_values(rope(shell, grid), grid)
    assert shell._vision_cos_sin_cache.data_ptr() == pointer
    torch.testing.assert_close(shell.rotary_pos_emb.cos_sin_cache, original_cache, rtol=0, atol=0)
    assert not any("cos_sin_cache" in name for name in shell.state_dict())
    for value, original in zip(rope(shell, [[1, 4, 6]]), before):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_vision_rope_extends_again_for_larger_mixed_grid_without_config(vision_rope_context):
    import torch
    from vllm.config import set_current_vllm_config

    shell, rope = _vision_rope()
    with set_current_vllm_config(None):
        first = [[1, 132, 4]]
        old = tuple(value.clone() for value in rope(shell, first))
        old_capacity = shell._vision_cos_sin_cache.shape[0]
        grid = [[1, 8, 12], [2, old_capacity + 2, 4]]
        _check_rope_values(rope(shell, grid), grid)
        assert shell._vision_cos_sin_cache.shape[0] >= old_capacity + 2
        for value, original in zip(rope(shell, first), old):
            torch.testing.assert_close(value, original, rtol=0, atol=0)
