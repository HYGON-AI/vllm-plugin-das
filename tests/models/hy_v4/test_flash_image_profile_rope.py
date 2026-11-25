# SPDX-License-Identifier: Apache-2.0
"""Raw image witnesses: public resize outputs must fit both RoPE cache axes.

The AST lane has no Torch/Core dependency and is not encoder/GPU validation.
"""

import ast
from fractions import Fraction
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4"


def image_functions():
    namespace = {"math": math, "Fraction": Fraction, "ImageSize": NS}
    nodes = []
    for path, names in (
        (
            ROOT / "processor.py",
            {"smart_resize", "vision_token_grid", "vision_token_count"},
        ),
        (
            ROOT / "mm_preprocess.py",
            {"get_image_size_with_most_features", "get_mm_max_tokens_per_item"},
        ),
    ):
        nodes.extend(
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name in names
        )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT), "exec"), namespace)
    return namespace


def profile(
    *,
    newline=True,
    rope=True,
    cache_side=2048,
    minimum=262144,
    maximum=4194304,
    options=None,
    do_resize=True,
):
    ns = image_functions()
    config = NS(
        use_2d_rope=rope,
        max_image_size=cache_side,
        patch_size=16,
        spatial_merge_size=2,
        spatial_patch_size=1,
        temporal_patch_size=1,
        use_image_newline_token=newline,
        cat_extra_token=False,
    )
    image = NS(
        patch_size=16,
        merge_size=2,
        do_resize=do_resize,
        size={"longest_edge": maximum, "shortest_edge": minimum},
    )
    video = NS(max_frames=2, size={"longest_edge": 4096})
    options = {} if options is None else options
    info = NS(
        get_hf_config=lambda: NS(vision_config=config),
        get_hf_processor=lambda: NS(image_processor=image, video_processor=video),
        _merge_and_resolve_mm_processor_kwargs=lambda _: {"images_kwargs": options},
    )
    info.get_image_size_with_most_features = lambda: ns[
        "get_image_size_with_most_features"
    ](info)
    size = info.get_image_size_with_most_features()
    if options.get("do_resize", do_resize):
        output = ns["smart_resize"](
            size.height,
            size.width,
            32,
            options.get("min_pixels", minimum),
            options.get("max_pixels", maximum),
        )
    else:
        output = size.height, size.width
    if rope:
        edge = cache_side // 16 + 1
        assert output[0] // 16 <= edge
        assert output[1] // 16 <= edge
    # get_mm_max_tokens_per_item must resize the RAW witness, not count it as a grid.
    budget = ns["get_mm_max_tokens_per_item"](info, 8192, {"image": 1})["image"]
    actual = ns["vision_token_count"](
        (1, output[0] // 16, output[1] // 16), config, "image"
    )
    assert budget == actual
    return ns, size, output, budget


@pytest.mark.parametrize("newline", [False, True])
def test_default_image_only_profile_respects_real_raw_patch_cache(newline):
    _, _, output, _ = profile(newline=newline)
    assert output == (2048, 2048)
    # The old unconstrained output really is unsafe for the fixed cache.
    assert 26208 // 16 > 2048 // 16 + 1


@pytest.mark.parametrize("newline", [False, True])
def test_disabling_rope_preserves_unrestricted_image_profile(newline):
    _, _, output, budget = profile(newline=newline, rope=False)
    assert budget == (4914 if newline else 4096)
    if newline:
        assert output == (26208, 160)


def test_overrides_and_minimum_upscale_cannot_escape_cache():
    _, _, output, _ = profile(
        cache_side=128, options={"min_pixels": 1024, "max_pixels": 8192}
    )
    assert output[0] * output[1] <= 8192
    with pytest.raises(ValueError, match="2D RoPE cache"):
        profile(cache_side=128, options={"min_pixels": 65536, "max_pixels": 65536})


@pytest.mark.parametrize("newline", [False, True])
@pytest.mark.parametrize(
    "minimum,maximum,expected",
    [
        (262144, 262144, 512),
        (1048576, 1048576, 1364),
        (860672, 860672, 1230),
        (1025, 1025, 2),
    ],
)
def test_rope_keeps_fractional_floor_rounding_and_closed_aspect_witnesses(
    newline, minimum, maximum, expected
):
    _, _, _, budget = profile(
        newline=newline, cache_side=32768, minimum=minimum, maximum=maximum
    )
    assert budget == (expected if newline else maximum // 1024)


@pytest.mark.parametrize("newline", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_resize_disabled_still_maximizes_with_both_cache_axes(newline, override):
    options = {"do_resize": False} if override else {}
    _, size, output, budget = profile(
        newline=newline,
        cache_side=128,
        minimum=4096,
        maximum=32768,
        do_resize=override,
        options=options,
    )
    assert (size.height, size.width) == output == (128, 128)
    assert budget == 4 * (4 + int(newline))


@pytest.mark.parametrize("newline", [False, True])
def test_resize_disabled_keeps_its_no_minimum_pixel_semantics(newline):
    # The public no-resize path does not upscale to min_pixels.
    _, _, output, budget = profile(
        newline=newline, cache_side=128, minimum=65536, maximum=65536, do_resize=False
    )
    assert output == (128, 128)
    assert budget == 4 * (4 + int(newline))


@pytest.mark.parametrize("do_resize", [False, True])
def test_cache_too_small_for_one_merged_cell_fails_explicitly(do_resize):
    with pytest.raises(ValueError, match="2D RoPE cache"):
        profile(cache_side=0, minimum=1024, maximum=8192, do_resize=do_resize)


@pytest.mark.parametrize("do_resize", [False, True])
def test_cache_position_count_includes_last_raw_patch(do_resize):
    _, _, output, _ = profile(
        cache_side=16, minimum=1024, maximum=1024, do_resize=do_resize
    )
    assert output == (32, 32)  # Two raw positions: max_image_size // patch_size + 1.


@pytest.mark.parametrize("newline", [False, True])
@pytest.mark.parametrize(
    "minimum,maximum,cache_side",
    [
        (1025, 1025, 16),  # floor output below min_pixels is still feasible
        (1024, 5120, 64),  # a cropped search must try a smaller column count
        (4096, 9216, 80),  # cache-side division has an odd raw-patch remainder
        (8192, 8192, 96),  # tight pixel target with cache-constrained aspect
        (12289, 12289, 128),  # fractional cell budget and downscale witnesses
        (1024, 32768, 128),
    ],
)
def test_bounded_profile_covers_independent_raw_resize_candidates(
    newline, minimum, maximum, cache_side
):
    ns, _, _, budget = profile(
        newline=newline, cache_side=cache_side, minimum=minimum, maximum=maximum
    )
    edge = cache_side // 16 + 1
    expected = 0
    # Independent raw inputs include unaligned rounding, floor/downscale,
    # upscaling, both orientations and exact aspect endpoints. Reject based
    # on ACTUAL output axes, not the raw dimensions or an aspect proxy.
    shapes = [(h, w) for h in range(1, 193) for w in range(1, 193)]
    shapes += [
        (short * ratio + jitter, short)
        for short in (1, 17, 31, 32, 41, 48, 64, 80, 160)
        for ratio in (1, 2, 16, 199, 200)
        for jitter in (-1, 0, 1)
    ]
    shapes += [(w, h) for h, w in shapes[-135:]]
    for height, width in shapes:
        if min(height, width) < 1:
            continue
        try:
            h, w = ns["smart_resize"](height, width, 32, minimum, maximum)
        except ValueError:
            continue
        if h // 16 > edge or w // 16 > edge:
            continue
        expected = max(expected, (h // 32) * (w // 32 + int(newline)))
    assert budget == expected


@pytest.mark.parametrize(
    "minimum,maximum,cache_side,expected",
    [
        (1025, 1025, 16, (32, 32)),
        (16385, 16385, 128, (128, 128)),
    ],
)
def test_raw_witness_may_exceed_cache_when_real_floor_output_fits(
    minimum, maximum, cache_side, expected
):
    _, size, output, _ = profile(
        cache_side=cache_side, minimum=minimum, maximum=maximum
    )
    assert output == expected
    assert (size.height, size.width) != output
    # In the first case the raw 48x24 witness is intentionally not patch-aligned.
    # Testing the raw size (or forcing it to the output) rejects this valid floor.
    assert max(size.height, size.width) > (cache_side // 16 + 1) * 16


@pytest.mark.parametrize("newline", [False, True])
@pytest.mark.parametrize("maximum", [5120, 6144, 8192, 12288, 32768])
def test_no_resize_searches_cache_bounded_rectangles_not_only_a_skinny_witness(
    newline, maximum
):
    _, _, _, budget = profile(
        newline=newline, minimum=4096, maximum=maximum, cache_side=128, do_resize=False
    )
    expected = max(
        h * (w + int(newline))
        for h in range(1, 5)
        for w in range(1, 5)
        if h * w * 1024 <= maximum
    )
    assert budget == expected
