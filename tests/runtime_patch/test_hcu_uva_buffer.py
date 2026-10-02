# SPDX-License-Identifier: Apache-2.0
"""Tests for the HCU registered-host-memory UvaBuffer patch."""

from __future__ import annotations

from types import ModuleType

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_hcu_uva_buffer as patch
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


def _target_module() -> ModuleType:
    module = ModuleType(patch.TARGET_MODULE)

    class UvaBuffer:
        def __init__(self, size, dtype):
            self.cpu = torch.zeros(size, dtype=dtype)
            self.np = self.cpu.numpy()
            self._uva = self.cpu

    module.UvaBuffer = UvaBuffer
    return module


def test_patch_is_idempotent():
    module = _target_module()
    assert patch.apply_to_module(module) is True
    assert patch.apply_to_module(module) is False


def test_patch_rejects_incompatible_signature():
    module = _target_module()

    def incompatible(self, size):
        del self, size

    module.UvaBuffer.__init__ = incompatible
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch.apply_to_module(module)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
def test_registered_buffer_has_cpu_numpy_and_uva_view():
    import vllm._C_stable_libtorch  # noqa: F401

    import vllm.v1.worker.gpu.buffer_utils as target

    patch.apply_to_module(target)
    buf = target.UvaBuffer((8, 16), torch.int32)
    assert buf.cpu.device.type == "cpu"
    assert buf.cpu.is_pinned()
    assert buf.np.shape == (8, 16)
    assert buf._uva.device.type == "cuda"
    assert buf._uva.shape == (8, 16)
    buf.cpu.fill_(7)
    assert torch.equal(buf._uva.cpu(), torch.full((8, 16), 7, dtype=torch.int32))
