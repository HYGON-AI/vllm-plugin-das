# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fall back to smaller batch-invariant GEMM tiles when HCU LDS overflows.

``matmul_persistent`` and ``bmm_batch_invariant`` hardcode their tile and
``num_stages``.  Upstream only shrinks them on CUDA, so on HCU (64KiB LDS)
some launches fail with ``OutOfResources``.  The A/B tile size does not
predict the real LDS use on ROCm: bf16/fp16 128x128 with 3 stages fits,
while fp32 with N or K not a multiple of 16 needs ``(stages + 1) * tile``.

So launch the upstream config first and only on ``OutOfResources`` walk a
ladder: fewer stages (not below 2), then half ``BLOCK_SIZE_N``, then a single
stage.  One stage is about 4x slower, so it is the last resort.

``BLOCK_SIZE_N`` and ``num_stages`` do not change the K reduction order, so
the fallback keeps results batch invariant.  The chosen config is cached by
dtype, N, K and the requested config, never by M.

``bmm_batch_invariant`` freezes ``grid`` from the original ``BLOCK_SIZE_N``
before this wrapper runs.  Narrowing that tile adds output tiles, and each
program writes only one tile, so every BMM launch rebuilds ``grid`` from the
tile actually used.  ``matmul_kernel_persistent`` keeps the callable grid,
which Triton evaluates from the launch meta.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from vllm.logger import init_logger

from ._common import PatchCompatibilityError, load_exact_module

logger = init_logger(__name__)

TARGET_MODULE = "vllm.model_executor.determinism.batch_invariant"
PATCH_ID = "worker.core_fix.batch_invariant.lds_stages"
TARGETS = (
    f"{TARGET_MODULE}.matmul_kernel_persistent",
    f"{TARGET_MODULE}.bmm_kernel",
)
_MARKER = "_vllm_hcu_batch_invariant_lds_stages_applied"
_PROXY_MARKER = "_vllm_hcu_batch_invariant_lds_stage_proxy"
_KERNEL_NAMES = ("matmul_kernel_persistent", "bmm_kernel")
# Both kernels take (a, b, c, bias_or_B, M, N, K, ...) positionally.
# For bmm_kernel the fourth value is the batch count, not a bias.
_B_ARG = 3
_M_ARG = 4
_N_ARG = 5
_K_ARG = 6
_MIN_BLOCK_N = 32


def _cdiv(value: int, block: int) -> int:
    return (value + block - 1) // block


def fallback_configs(block_n: int, num_stages: int) -> list[tuple[int, int]]:
    """Smaller ``(BLOCK_SIZE_N, num_stages)`` configs to try, fastest first."""

    configs = [(block_n, stages) for stages in range(num_stages - 1, 1, -1)]
    pipelined = min(num_stages, 2)
    narrow = block_n // 2
    while narrow >= _MIN_BLOCK_N:
        configs.append((narrow, pipelined))
        narrow //= 2
    if num_stages > 1:
        configs.append((block_n, 1))
    return configs


def _out_of_resources() -> type[Exception]:
    from triton.runtime.errors import OutOfResources

    return OutOfResources


class _LdsFallbackKernel:
    """Preserve ``kernel[grid](*args, **kwargs)`` and retry on LDS overflow."""

    def __init__(self, kernel: Any, name: str) -> None:
        self._kernel = kernel
        self._name = name
        self._chosen: dict[tuple[Any, ...], tuple[int, int]] = {}
        setattr(self, _PROXY_MARKER, True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)

    def __getitem__(self, grid: Any) -> Any:
        def launch_within_lds(*args: Any, **kwargs: Any) -> Any:
            return self._launch(grid, args, kwargs)

        return launch_within_lds

    def _launch(
        self, grid: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        def invoke() -> Any:
            return self._kernel[self._grid(grid, args, kwargs)](*args, **kwargs)

        block_n = kwargs.get("BLOCK_SIZE_N")
        num_stages = kwargs.get("num_stages", 3)
        if not isinstance(block_n, int) or not isinstance(num_stages, int):
            return invoke()

        key = self._cache_key(args, kwargs, block_n, num_stages)
        chosen = self._chosen.get(key)
        if chosen is not None:
            kwargs["BLOCK_SIZE_N"], kwargs["num_stages"] = chosen
            return invoke()

        out_of_resources = _out_of_resources()
        try:
            return invoke()
        except out_of_resources as exc:
            error = exc
        for candidate in fallback_configs(block_n, num_stages):
            kwargs["BLOCK_SIZE_N"], kwargs["num_stages"] = candidate
            try:
                result = invoke()
            except out_of_resources as exc:
                error = exc
                continue
            self._chosen[key] = candidate
            logger.info_once(
                "HCU batch-invariant %s: BLOCK_SIZE_N=%d num_stages=%d exceeds "
                "LDS, using BLOCK_SIZE_N=%d num_stages=%d",
                self._name,
                block_n,
                num_stages,
                *candidate,
            )
            return result
        raise error

    def _grid(
        self, grid: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Rebuild a frozen BMM tile grid from the tile about to be launched.

        Persistent matmul passes a callable of the launch meta, so Triton
        already grows that grid when ``BLOCK_SIZE_N`` shrinks.
        """

        if self._name != "bmm_kernel" or not isinstance(grid, tuple):
            return grid
        block_m = kwargs.get("BLOCK_SIZE_M")
        block_n = kwargs.get("BLOCK_SIZE_N")
        if (
            not isinstance(block_m, int)
            or not isinstance(block_n, int)
            or block_m <= 0
            or block_n <= 0
            or len(args) <= _N_ARG
        ):
            return grid
        batch, rows, cols = args[_B_ARG], args[_M_ARG], args[_N_ARG]
        if not all(isinstance(value, int) for value in (batch, rows, cols)):
            return grid
        tiles = _cdiv(rows, block_m) * _cdiv(cols, block_n)
        return (batch, tiles)

    @staticmethod
    def _cache_key(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        block_n: int,
        num_stages: int,
    ) -> tuple[Any, ...]:
        dtype = getattr(args[0], "dtype", None) if args else None
        n = args[_N_ARG] if len(args) > _N_ARG else None
        k = args[_K_ARG] if len(args) > _K_ARG else None
        return (
            dtype,
            n,
            k,
            kwargs.get("BLOCK_SIZE_M"),
            block_n,
            kwargs.get("BLOCK_SIZE_K"),
            num_stages,
            kwargs.get("num_warps"),
        )


def apply_to_module(module: ModuleType) -> bool:
    batch_invariant = load_exact_module(TARGET_MODULE, module)
    kernels = [getattr(batch_invariant, name, None) for name in _KERNEL_NAMES]
    if getattr(batch_invariant, _MARKER, False):
        if any(not getattr(kernel, _PROXY_MARKER, False) for kernel in kernels):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE} is stale; "
                "restart the process"
            )
        return False

    for name, kernel in zip(_KERNEL_NAMES, kernels):
        if kernel is None:
            raise PatchCompatibilityError(
                f"required HCU patch target {TARGET_MODULE}.{name} "
                "is missing Triton launch semantics"
            )
    launchable = [
        callable(getattr(kernel, "__getitem__", None)) for kernel in kernels
    ]
    if not any(launchable):
        # No GPU driver: vLLM replaces @triton.jit with the plain function.
        # CPU contract tests import this module, and there is no kernel to wrap.
        return False
    if not all(launchable):
        missing = _KERNEL_NAMES[launchable.index(False)]
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.{missing} "
            "is missing Triton launch semantics"
        )

    for name, kernel in zip(_KERNEL_NAMES, kernels):
        setattr(batch_invariant, f"_vllm_hcu_original_{name}", kernel)
        setattr(batch_invariant, name, _LdsFallbackKernel(kernel, name))
    setattr(batch_invariant, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "fallback_configs",
]
