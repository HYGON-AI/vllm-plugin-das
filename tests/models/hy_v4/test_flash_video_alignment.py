# SPDX-License-Identifier: Apache-2.0
"""Production video preprocessing AST with real Torch frames.

Only framework constructors, image resize and the result container are shells;
this does not qualify the installed multimodal dispatcher or model inference.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4/processor.py"


def preprocess():
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("adaptive_resize", "patchify", "vision_token_grid")]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "HYV4VLVideoProcessor")
    nodes += [n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "_preprocess"]
    namespace = dict(torch=torch, math=math, SizeDict=lambda **kw: kw,
                     PILImageResampling=SimpleNamespace(BICUBIC=3),
                     BatchFeature=lambda data, tensor_type: data)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("spatial", [1, 2])
@pytest.mark.parametrize("shape", [(2, 224, 224), (4, 96, 96), (3, 160, 224)])
def test_resized_real_video_grid_includes_spatial_compression(spatial, shape):
    ns = preprocess()
    frames, height, width = shape
    video = torch.ones(frames, 3, height, width)
    sizes = []

    def resize(value, size, resample):
        sizes.append(size)
        return torch.nn.functional.interpolate(value, size=(size["height"], size["width"]), mode="nearest")

    processor = SimpleNamespace(max_frames=100, video_max_image_size=1024,
        timestamp_token_per_frame=0, timestamp_token_reserve=0, resize=resize,
        rescale_and_normalize=lambda value, *args: value)
    output = ns["_preprocess"](processor, [video], do_convert_rgb=False,
        size=SimpleNamespace(longest_edge=33554432), spatial_patch_size=spatial)
    factor = 16 * 2 * spatial
    assert sizes == [{"height": height // factor * factor, "width": width // factor * factor}]
    grid = output["video_grid_thw"][0].tolist()
    config = SimpleNamespace(spatial_merge_size=2, spatial_patch_size=spatial,
                             temporal_patch_size=2, use_image_newline_token=False)
    encoded = ns["vision_token_grid"](grid, config, "video")
    assert encoded == (math.ceil(frames / 2), sizes[0]["height"] // factor, sizes[0]["width"] // factor)
    assert output["pixel_values_videos"].shape[0] == math.prod(grid)


@pytest.mark.parametrize("spatial,side,valid", [(1, 96, True), (2, 96, False), (2, 128, True)])
def test_resize_disabled_checks_effective_video_alignment(spatial, side, valid):
    ns = preprocess()
    processor = SimpleNamespace(max_frames=100,
        rescale_and_normalize=lambda value, *args: value)
    kwargs = dict(do_resize=False, do_convert_rgb=False,
        size=SimpleNamespace(longest_edge=33554432), spatial_patch_size=spatial)
    video = torch.ones(2, 3, side, side)
    if valid:
        output = ns["_preprocess"](processor, [video], **kwargs)
        assert output["video_grid_thw"].tolist() == [[2, side // 16, side // 16]]
    else:
        with pytest.raises(ValueError, match="align"):
            ns["_preprocess"](processor, [video], **kwargs)
