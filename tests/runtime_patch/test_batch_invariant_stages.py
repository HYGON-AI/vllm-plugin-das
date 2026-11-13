# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import ModuleType

import pytest
from triton.runtime.errors import OutOfResources

from vllm_hcu.patch.worker.core_fix import patch_batch_invariant_stages
from vllm_hcu.patch.worker.core_fix.patch_batch_invariant_stages import (
    fallback_configs,
)


class _Tensor:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype


class _Kernel:
    """Fails with OutOfResources unless ``fits(BLOCK_SIZE_N, num_stages)``."""

    def __init__(self, fits=lambda block_n, stages: True) -> None:
        self.fits = fits
        self.calls: list[tuple[object, tuple, dict]] = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, dict(kwargs)))
            if not self.fits(kwargs["BLOCK_SIZE_N"], kwargs["num_stages"]):
                raise OutOfResources(98304, 65536, "shared memory")
            return "launched"

        return launch


def _module(fits=lambda block_n, stages: True):
    module = ModuleType(patch_batch_invariant_stages.TARGET_MODULE)
    module.matmul_kernel_persistent = _Kernel(fits)
    module.bmm_kernel = _Kernel(fits)
    return module


def _launch(module, *, m=128, n=4, k=128, dtype="float32", block_n=128, stages=3):
    return module.matmul_kernel_persistent[(4,)](
        _Tensor(dtype),
        None,
        None,
        None,
        m,
        n,
        k,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=32,
        num_stages=stages,
        num_warps=8,
    )


def _launched(kernel):
    return [(kw["BLOCK_SIZE_N"], kw["num_stages"]) for _grid, _args, kw in kernel.calls]


def test_fallback_order_keeps_two_stages_before_dropping_to_one():
    assert fallback_configs(128, 3) == [(128, 2), (64, 2), (32, 2), (128, 1)]
    assert fallback_configs(256, 3) == [(256, 2), (128, 2), (64, 2), (32, 2), (256, 1)]
    assert fallback_configs(32, 2) == [(32, 1)]
    assert fallback_configs(32, 1) == []


def test_config_that_fits_is_launched_unchanged():
    module = _module()
    patch_batch_invariant_stages.apply_to_module(module)
    assert _launch(module, dtype="bfloat16") == "launched"
    assert _launched(module.matmul_kernel_persistent._kernel) == [(128, 3)]


def test_unaligned_fp32_falls_back_to_narrow_two_stage_tile():
    # HCU fp32 with N % 16 != 0: 128/2 still needs 98304 bytes, 64/2 fits.
    module = _module(fits=lambda block_n, stages: block_n <= 64 or stages == 1)
    patch_batch_invariant_stages.apply_to_module(module)
    assert _launch(module) == "launched"
    assert _launched(module.matmul_kernel_persistent._kernel) == [
        (128, 3),
        (128, 2),
        (64, 2),
    ]


def test_fallback_is_cached_and_shared_across_m():
    module = _module(fits=lambda block_n, stages: block_n <= 64)
    patch_batch_invariant_stages.apply_to_module(module)
    _launch(module, m=128)
    kernel = module.matmul_kernel_persistent._kernel
    kernel.calls.clear()
    _launch(module, m=1)
    _launch(module, m=4096)
    assert _launched(kernel) == [(64, 2), (64, 2)]


def test_cache_is_per_shape():
    module = _module(fits=lambda block_n, stages: block_n <= 64)
    patch_batch_invariant_stages.apply_to_module(module)
    _launch(module, n=4)
    kernel = module.matmul_kernel_persistent._kernel
    kernel.fits = lambda block_n, stages: True
    kernel.calls.clear()
    _launch(module, n=128)
    assert _launched(kernel) == [(128, 3)]


def test_both_kernels_are_wrapped_and_patch_is_idempotent():
    module = _module(fits=lambda block_n, stages: stages <= 2)
    assert patch_batch_invariant_stages.apply_to_module(module) is True
    assert patch_batch_invariant_stages.apply_to_module(module) is False
    assert module.bmm_kernel[(2, 8)](
        _Tensor("bfloat16"),
        None,
        None,
        2,
        128,
        128,
        128,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        num_stages=3,
    ) == "launched"
    assert _launched(module.bmm_kernel._kernel) == [(128, 3), (128, 2)]


def test_error_is_raised_when_nothing_fits():
    module = _module(fits=lambda block_n, stages: False)
    patch_batch_invariant_stages.apply_to_module(module)
    with pytest.raises(OutOfResources):
        _launch(module)


def test_other_errors_are_not_retried():
    module = _module()
    patch_batch_invariant_stages.apply_to_module(module)
    kernel = module.matmul_kernel_persistent._kernel

    def broken(block_n, stages):
        raise ValueError("bad launch")

    kernel.fits = broken
    with pytest.raises(ValueError):
        _launch(module)
    assert len(kernel.calls) == 1


def _cdiv(value: int, block: int) -> int:
    return (value + block - 1) // block


def _blank(batch: int, rows: int, cols: int, fill: object = None) -> list:
    return [[[fill for _ in range(cols)] for _ in range(rows)] for _ in range(batch)]


class _TileKernel:
    """One program writes one tile, matching upstream ``bmm_kernel``."""

    def __init__(self, fits=lambda block_n, stages: True) -> None:
        self.fits = fits
        self.grids: list[object] = []

    def __getitem__(self, grid):
        def launch(a, b, out, batch, rows, cols, k, **kwargs):
            del a, b, k
            self.grids.append(grid)
            block_m = kwargs["BLOCK_SIZE_M"]
            block_n = kwargs["BLOCK_SIZE_N"]
            if not self.fits(block_n, kwargs["num_stages"]):
                raise OutOfResources(98304, 65536, "shared memory")
            num_pid_n = _cdiv(cols, block_n)
            num_pid_m = _cdiv(rows, block_m)
            for pid_b in range(grid[0]):
                for pid in range(grid[1]):
                    pid_m = pid // num_pid_n
                    pid_n = pid % num_pid_n
                    if pid_m >= num_pid_m or pid_n >= num_pid_n:
                        continue
                    for row in range(pid_m * block_m, min((pid_m + 1) * block_m, rows)):
                        for col in range(pid_n * block_n, min((pid_n + 1) * block_n, cols)):
                            out[pid_b][row][col] = 1
            return out

        return launch


def _launch_bmm(module, out, *, batch, rows, cols, block_n=128, stages=3):
    # Upstream freezes this from the requested tile, before any fallback.
    grid = (batch, _cdiv(rows, 128) * _cdiv(cols, block_n))
    return module.bmm_kernel[grid](
        _Tensor("float32"),
        None,
        out,
        batch,
        rows,
        cols,
        128,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=32,
        num_stages=stages,
        num_warps=8,
    )


def _written(out) -> bool:
    return all(value == 1 for batch in out for row in batch for value in row)


def test_bmm_narrow_tile_covers_output_on_retry_and_cache_hit():
    module = _module(fits=lambda block_n, stages: block_n <= 64)
    module.bmm_kernel = _TileKernel(fits=lambda block_n, stages: block_n <= 64)
    patch_batch_invariant_stages.apply_to_module(module)

    out = _blank(1, 128, 65, fill="old")
    _launch_bmm(module, out, batch=1, rows=128, cols=65)
    assert _written(out)
    assert module.bmm_kernel._kernel.grids[-1] == (1, 2)

    cached = _blank(1, 256, 65, fill="old")
    _launch_bmm(module, cached, batch=1, rows=256, cols=65)
    assert _written(cached)
    assert module.bmm_kernel._kernel.grids[-1] == (1, 4)

    wider = _blank(2, 128, 129, fill="old")
    _launch_bmm(module, wider, batch=2, rows=128, cols=129)
    assert _written(wider)
    assert module.bmm_kernel._kernel.grids[-1] == (2, 3)


def test_bmm_stage_drop_keeps_the_original_grid():
    module = _module(fits=lambda block_n, stages: stages <= 2)
    module.bmm_kernel = _TileKernel(fits=lambda block_n, stages: stages <= 2)
    patch_batch_invariant_stages.apply_to_module(module)

    out = _blank(1, 128, 128, fill="old")
    _launch_bmm(module, out, batch=1, rows=128, cols=128)
    assert _written(out)
    assert module.bmm_kernel._kernel.grids[-1] == (1, 1)

    narrow_n = _blank(1, 128, 4, fill="old")
    module.bmm_kernel._kernel.fits = lambda block_n, stages: block_n <= 64
    _launch_bmm(module, narrow_n, batch=1, rows=128, cols=4)
    assert _written(narrow_n)
    assert module.bmm_kernel._kernel.grids[-1] == (1, 1)


def test_persistent_matmul_keeps_callable_grid():
    module = _module(fits=lambda block_n, stages: stages <= 2)

    def grid(meta):
        return (meta["BLOCK_SIZE_N"],)

    patch_batch_invariant_stages.apply_to_module(module)
    assert (
        module.matmul_kernel_persistent[grid](
            _Tensor("float32"),
            None,
            None,
            None,
            128,
            65,
            128,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=32,
            num_stages=3,
            num_warps=8,
        )
        == "launched"
    )
    assert [item[0] for item in module.matmul_kernel_persistent._kernel.calls] == [
        grid,
        grid,
    ]


def test_plain_functions_are_skipped_when_triton_is_disabled():
    module = ModuleType(patch_batch_invariant_stages.TARGET_MODULE)

    def matmul(*args, **kwargs):
        del args, kwargs

    def bmm(*args, **kwargs):
        del args, kwargs

    module.matmul_kernel_persistent = matmul
    module.bmm_kernel = bmm
    assert patch_batch_invariant_stages.apply_to_module(module) is False
    assert patch_batch_invariant_stages.apply_to_module(module) is False
    assert module.matmul_kernel_persistent is matmul
    assert module.bmm_kernel is bmm


def test_missing_kernel_is_rejected():
    module = ModuleType(patch_batch_invariant_stages.TARGET_MODULE)
    module.matmul_kernel_persistent = _Kernel()
    with pytest.raises(patch_batch_invariant_stages.PatchCompatibilityError):
        patch_batch_invariant_stages.apply_to_module(module)


def test_stale_marker_is_rejected():
    module = _module()
    setattr(module, "_vllm_hcu_batch_invariant_lds_stages_applied", True)
    with pytest.raises(patch_batch_invariant_stages.PatchCompatibilityError):
        patch_batch_invariant_stages.apply_to_module(module)
