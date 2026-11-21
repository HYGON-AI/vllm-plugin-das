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
    assert resize(size.height, size.width, 32, 512**2, maximum) == (size.height, size.width)
    grid = [[1, size.height // 16, size.width // 16]]
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
