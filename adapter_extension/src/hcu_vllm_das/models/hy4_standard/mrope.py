# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared per-forward M-RoPE rows for HY4 fused rotary kernels."""

import torch
from torch import nn


def set_token_rope(
    rotary_emb: nn.Module, positions: torch.Tensor, x: torch.Tensor
) -> None:
    """Build this forward's per-token M-RoPE cos/sin rows once, before the
    layers run; `token_rope` reads them. `x` sets the cache dtype."""
    if positions.ndim == 1:
        rotary_emb._hy4_token_rope = None
        return
    sel = getattr(rotary_emb, "_hy4_mrope_sel", None)
    if sel is None:
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            apply_interleaved_rope,
        )

        # T/H/W position row of each cos/sin channel, via vLLM's M-RoPE layout.
        sec = rotary_emb.mrope_section
        ids = torch.arange(3, device=positions.device)[:, None].expand(3, sum(sec))
        if rotary_emb.mrope_interleaved:
            sel = apply_interleaved_rope(ids, sec)
        else:
            sel = torch.cat([m[i] for i, m in enumerate(ids.split(sec, dim=-1))])
        sel = rotary_emb._hy4_mrope_sel = sel.repeat(2)
    num_tokens = positions.shape[1]
    cache = rotary_emb._match_cos_sin_cache_dtype(x)
    rows = cache[positions].gather(0, sel.expand(1, num_tokens, -1))[0]
    arange = torch.arange(num_tokens, device=positions.device)
    rotary_emb._hy4_token_rope = (arange, rows)


def token_rope(
    rotary_emb: nn.Module, positions: torch.Tensor, cos_sin_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(positions, cos_sin_cache)` for the 1-D fused RoPE kernels.

    [3, T] M-RoPE positions become `arange(T)` into the per-token cos/sin
    rows built by `set_token_rope` for this forward.
    """
    if positions.ndim == 1:
        return positions, cos_sin_cache
    rows = getattr(rotary_emb, "_hy4_token_rope", None)
    assert rows is not None and rows[0].shape[0] == positions.shape[1], (
        "set_token_rope must run before the layers"
    )
    return rows


def apply_mrope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate strided Q/K in place using this forward's shared token rows."""
    rope_pos, cos_sin = token_rope(
        rotary_emb, positions, rotary_emb._match_cos_sin_cache_dtype(query)
    )
    torch.ops._C.rotary_embedding(
        rope_pos,
        query,
        key,
        rotary_emb.head_size,
        cos_sin,
        rotary_emb.is_neox_style,
    )
    return query, key


def build_mrope_positions(config, input_tokens, image_grids, video_grids):
    """Build checkpoint T/H/W positions from validated media grids."""
    from .processor import vision_token_grid

    tokens = torch.tensor(input_tokens, dtype=torch.long)
    positions = torch.arange(len(input_tokens)).repeat(3, 1)
    if not input_tokens or not (image_grids or video_grids):
        return positions, 0
    vision = config.vision_config
    extra = int(bool(vision.cat_extra_token))
    starts = torch.where(tokens == config.image_start_token_id)[0].tolist()
    ends = torch.where(tokens == config.image_end_token_id)[0].tolist()
    if len(starts) != len(ends):
        raise ValueError("HYV4 image/frame start and end tokens must be paired")
    start_pos = start_index = mm_index = scan = 0
    image_index = video_index = -1
    inside_video = False
    while mm_index < len(starts):
        marker = starts[mm_index]
        while scan < marker:
            if input_tokens[scan] == config.video_start_token_id:
                inside_video = True
            elif input_tokens[scan] == config.video_end_token_id:
                inside_video = False
            scan += 1
        is_video = inside_video or (
            image_index + 1 >= len(image_grids) and video_index + 1 < len(video_grids)
        )
        if is_video:
            video_index += 1
            if video_index >= len(video_grids):
                raise ValueError("Video placeholders do not match video_grid_thw")
            frames, rows, cols = vision_token_grid(
                video_grids[video_index], vision, "video"
            )
        else:
            image_index += 1
            if image_index >= len(image_grids):
                raise ValueError("Image placeholders do not match image_grid_thw")
            frames, rows, cols = vision_token_grid(
                image_grids[image_index], vision, "image"
            )
        if mm_index + frames > len(starts) or mm_index + frames > len(ends):
            raise ValueError("Missing HYV4 image/frame boundary tokens")
        count = rows * cols
        for frame in range(frames):
            begin = starts[mm_index + frame] + 1 + extra
            end = begin + count
            pad_id = config.video_token_id if is_video else config.image_token_id
            if end + extra != ends[mm_index + frame] or not torch.all(
                tokens[begin - extra : end + extra] == pad_id
            ):
                raise ValueError(
                    "HYV4 visual placeholders do not match frame boundaries and grid"
                )
            if start_pos < begin:
                positions[:, start_pos:begin] = (
                    torch.arange(begin - start_pos) + start_index
                )
                start_index += begin - start_pos
            positions[0, begin:end] = start_index
            positions[1, begin:end] = (
                torch.arange(rows).repeat_interleave(cols) + start_index
            )
            positions[2, begin:end] = torch.arange(cols).repeat(rows) + start_index
            start_index += max(rows, cols)
            # The next prefix includes this frame's trailing extra token,
            # boundary/timestamp tokens, and the next frame's leading extra.
            start_pos = end
        mm_index += frames
    if start_pos < len(input_tokens):
        positions[:, start_pos:] = (
            torch.arange(len(input_tokens) - start_pos) + start_index
        )
    if image_index + 1 != len(image_grids) or video_index + 1 != len(video_grids):
        raise ValueError("Unused HYV4 media grids after position construction")
    return positions, int(positions.max()) + 1 - len(input_tokens)
