# SPDX-License-Identifier: Apache-2.0
"""Grid sizing for the fused MoE shared-expert SiLU-mul.

The kernel writes exactly ``BLOCK`` columns per program, so the launch grid
must tile ``inter`` without truncation. A plain ``inter // blk`` silently
dropped the tail when ``inter`` was not a multiple of 1024 (inter=1536 left
columns 1024..1535 holding ``torch.empty`` memory that then reached
down_proj). These tests pin the tiling rule for the shapes in use and for
the ones that used to truncate.
"""

from __future__ import annotations

import pytest

from vllm_hcu.ops import qwen35_moe_block as fused


def _tiling(inter: int) -> tuple[int, int]:
    """Mirror of the launch site: (BLOCK, grid columns).

    BLOCK stays a power of two because ``tl.arange`` requires one; the kernel
    masks the tail, so any ``cdiv`` grid is exact.
    """
    blk = 1024 if inter % 1024 == 0 else 512
    return blk, -(-inter // blk)


@pytest.mark.parametrize(
    ("inter", "block", "grid"),
    [
        (512, 512, 1),     # Qwen3.5-35B shared_expert_intermediate_size
        (1024, 1024, 1),
        (1536, 512, 3),    # used to truncate to 1024 columns
        (2048, 1024, 2),
        (2560, 512, 5),    # Qwen3.5-4B
        (3072, 1024, 3),
    ],
)
def test_silu_mul_grid_covers_every_column(inter, block, grid):
    blk, g = _tiling(inter)
    assert (blk, g) == (block, grid)
    assert blk & (blk - 1) == 0, "BLOCK must be a power of two for tl.arange"
    assert blk * g >= inter, "grid must not leave a tail unwritten"


def test_launch_site_uses_the_tiling_rule():
    """Guard against the call site drifting from `_tiling`."""
    import inspect

    src = inspect.getsource(fused.fused_moe_block_pending)
    assert "blk = 1024 if inter % 1024 == 0 else 512" in src
    assert "triton.cdiv(inter, blk)" in src


def test_silu_mul_matches_reference_on_tail_columns():
    """End-to-end shape check on the columns that used to be skipped.

    Runs the Triton kernel over a full row and compares against the same
    arithmetic in torch, for both the shipped 512 and the 1536 shape.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("kernel needs a GPU")

    for inter in (512, 1536):
        m, ldy = 3, 2 * inter
        y = torch.randn(m, ldy, dtype=torch.float16, device="cuda")
        act = torch.empty(m, inter, dtype=torch.float16, device="cuda")
        block, grid = _tiling(inter)
        fused._strided_silu_mul_kernel[(m, grid)](
            y, ldy, act, D=inter, BLOCK=block, num_warps=4)
        g = y[:, :inter].float()
        ref = (g * torch.sigmoid(g) * y[:, inter:].float()).half()
        torch.testing.assert_close(act, ref, rtol=2e-3, atol=2e-3)
