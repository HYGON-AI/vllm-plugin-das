# SPDX-License-Identifier: Apache-2.0
"""Tests for the HCU registered-host-memory UvaBuffer patch."""

from __future__ import annotations

from types import ModuleType

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_hcu_uva_buffer as patch
from vllm_hcu.patch.worker.core_fix import patch_hcu_uva_offloader as offload_patch
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


def _offloader_target():
    module = ModuleType(offload_patch.TARGET_MODULE)

    class UVAOffloader:
        uva_offloading = True
        pin_memory = True

        def _maybe_offload_to_cpu(self, module, prefix=""):
            self.original_call = (module, prefix)
            return module

    module.UVAOffloader = UVAOffloader
    return module


def test_registered_offloader_preserves_original_path_when_disabled(monkeypatch):
    target = _offloader_target()
    monkeypatch.delenv("VLLM_HCU_REGISTERED_CPU_OFFLOAD", raising=False)
    assert offload_patch.apply_to_module(target)
    offloader = target.UVAOffloader()
    layer = object()
    assert offloader._maybe_offload_to_cpu(layer, "layers.0.") is layer
    assert offloader.original_call == (layer, "layers.0.")
    assert not offload_patch.apply_to_module(target)


def test_registered_offloader_preserves_non_uva_fallback(monkeypatch):
    target = _offloader_target()
    monkeypatch.setenv("VLLM_HCU_REGISTERED_CPU_OFFLOAD", "1")
    offload_patch.apply_to_module(target)
    offloader = target.UVAOffloader()
    offloader.uva_offloading = False
    layer = object()
    assert offloader._maybe_offload_to_cpu(layer) is layer
    assert offloader.original_call == (layer, "")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
def test_registered_offloader_retains_exact_storage_and_preserves_int8_values(
    monkeypatch,
):
    import vllm.model_executor.offloader.uva as target
    from vllm.platforms import current_platform

    current_platform.import_kernels()
    monkeypatch.setenv("VLLM_HCU_REGISTERED_CPU_OFFLOAD", "1")
    offload_patch.apply_to_module(target)
    layer = torch.nn.Module()
    original = torch.arange(4608 * 128, device="cuda").to(torch.int8)
    layer.w13_weight = torch.nn.Parameter(
        original.view(1, 4608, 128).clone(), requires_grad=False
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones((1, 4608, 1), device="cuda"), requires_grad=False
    )
    scales_pointer = layer.w13_weight_scale.data_ptr()
    offloader = target.UVAOffloader(original.numel(), {"w13_weight"})
    assert offloader.uva_offloading and offloader.pin_memory
    assert offloader._maybe_offload_to_cpu(layer) is layer
    cpu, mapping = layer._hcu_registered_offload_storage[0]
    assert len(mapping) == original.numel()
    assert cpu.is_pinned()
    assert layer.w13_weight.device.type == "cuda"
    assert layer.w13_weight_scale.data_ptr() == scales_pointer
    assert torch.equal(layer.w13_weight.flatten().cpu(), original.cpu())
    assert offloader.cpu_offload_bytes == original.numel()
    offloader.cpu_offload_max_bytes *= 2
    offloader._maybe_offload_to_cpu(layer)
    assert len(layer._hcu_registered_offload_storage) == 1
    assert offloader.cpu_offload_bytes == original.numel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
def test_registered_offloader_survives_inplace_deepgemm_packing(monkeypatch):
    import deepgemm
    import vllm.model_executor.offloader.uva as target
    from vllm.platforms import current_platform

    current_platform.import_kernels()
    monkeypatch.setenv("VLLM_HCU_REGISTERED_CPU_OFFLOAD", "1")
    offload_patch.apply_to_module(target)
    hidden, out, tokens = 5120, 4608, 256
    torch.manual_seed(42)
    weight = torch.randint(
        -32, 32, (1, out, hidden), device="cuda", dtype=torch.int8
    )
    reference_weight = weight.clone()
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(weight, requires_grad=False)
    offloader = target.UVAOffloader(weight.numel(), {"w13_weight"})
    offloader._maybe_offload_to_cpu(layer)
    packed = deepgemm.marlin_i8_contiguous_weight(layer.w13_weight)
    layer.w13_weight = torch.nn.Parameter(packed, requires_grad=False)
    activation = torch.randint(
        -32, 32, (tokens, hidden), device="cuda", dtype=torch.int8
    )
    a_scale = torch.full((tokens, 1), 0.01, device="cuda")
    w_scale = torch.full((1, out, 1), 0.01, device="cuda")
    output = torch.empty((tokens, out), device="cuda", dtype=torch.bfloat16)
    deepgemm.m_grouped_i8_gemm_nt_contiguous(
        (activation, a_scale), (layer.w13_weight, w_scale), output,
        torch.zeros(tokens, device="cuda", dtype=torch.int32),
    )
    expected = (activation.float() * a_scale) @ (
        reference_weight[0].float() * w_scale[0]
    ).T
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=1e-2)
    assert len(layer._hcu_registered_offload_storage[0][1]) == weight.numel()
