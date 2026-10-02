# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Use registered anonymous host memory for vLLM UVA buffers on HCU.

ROCm/HCU ``pin_memory=True`` uses a large host-allocation path which is
unreliable for the request-state buffers created during worker startup.  Keep
the vLLM ``UvaBuffer`` API unchanged, but allocate anonymous mmap storage and
register it with ``cudaHostRegister`` before creating the UVA view.
"""

from __future__ import annotations

import functools
import mmap
import weakref
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.buffer_utils"
PATCH_ID = "worker.core_fix.hcu_uva_buffer"
TARGETS = (f"{TARGET_MODULE}.UvaBuffer.__init__",)
_MARKER = "_vllm_hcu_uva_buffer_applied"
_INIT_MARKER = "_vllm_hcu_uva_buffer_init"


def _cuda_code(result: object) -> int:
    return int(getattr(result, "value", result))


def _unregister(pointer: int) -> None:
    result = torch.cuda.cudart().cudaHostUnregister(pointer)
    if _cuda_code(result) != 0:
        raise RuntimeError(f"cudaHostUnregister failed: {result}")


def _allocate_registered_buffer(
    size: int | tuple[int, ...], dtype: torch.dtype, owner: object
) -> tuple[torch.Tensor, mmap.mmap]:
    shape = (size,) if isinstance(size, int) else tuple(size)
    numel = 1
    for dim in shape:
        numel *= dim
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    mapping = mmap.mmap(
        -1,
        nbytes,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    raw = torch.frombuffer(mapping, dtype=dtype, count=numel).view(shape)
    pointer = raw.data_ptr()
    try:
        result = torch.cuda.cudart().cudaHostRegister(pointer, nbytes, 0)
        if _cuda_code(result) != 0:
            raise RuntimeError(f"cudaHostRegister failed: {result}")
        if not raw.is_pinned():
            raise RuntimeError("HCU did not recognize registered UVA buffer as pinned")
        raw.zero_()
        # The owner retains ``mapping`` and the tensor view. Unregister first;
        # mmap closes naturally after those exported views are destroyed.
        finalizer = weakref.finalize(owner, _unregister, pointer)
        finalizer.atexit = False  # type: ignore[misc]
        return raw, mapping
    except BaseException:
        try:
            _unregister(pointer)
        except Exception:
            pass
        mapping.close()
        raise


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    uva_buffer = require_class(target, "UvaBuffer", f"{TARGET_MODULE}.UvaBuffer")
    original = require_callable(uva_buffer, "__init__", TARGETS[0])
    require_exact_signature(original, TARGETS[0], positional=("self", "size", "dtype"))
    if getattr(original, _INIT_MARKER, False):
        return False

    @functools.wraps(original)
    def hcu_init(self, size, dtype):
        try:
            from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

            self.cpu, self._hcu_uva_mapping = _allocate_registered_buffer(
                size, dtype, self
            )
            self.np = self.cpu.numpy()
            self._uva = get_accelerator_view_from_cpu_tensor(self.cpu)
        except Exception as exc:
            raise RuntimeError(
                "HCU UvaBuffer initialization failed while creating registered "
                "anonymous host memory"
            ) from exc

    setattr(hcu_init, _INIT_MARKER, True)
    setattr(uva_buffer, "_vllm_hcu_original_init", original)
    setattr(uva_buffer, "__init__", hcu_init)
    setattr(uva_buffer, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
