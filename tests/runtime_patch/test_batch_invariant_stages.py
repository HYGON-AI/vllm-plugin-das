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


def test_stale_marker_is_rejected():
    module = _module()
    setattr(module, "_vllm_hcu_batch_invariant_lds_stages_applied", True)
    with pytest.raises(patch_batch_invariant_stages.PatchCompatibilityError):
        patch_batch_invariant_stages.apply_to_module(module)
