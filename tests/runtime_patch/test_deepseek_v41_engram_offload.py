# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Tests for DeepSeek V4.1 Engram CPU offload on HCU."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v41_engram_offload as patch
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


def _layout():
    return SimpleNamespace(
        num_embeddings=(16,),
        head_dim=32,
        primes=(((16,),),),
    )


def _target_module():
    module = ModuleType(patch.TARGET_MODULE)

    class ParallelEngramEmbedding(torch.nn.Module):
        def __init__(self, num_embeddings, dim, head_sizes, block_size=32):
            super().__init__()
            self.part_num_embeddings = num_embeddings
            self.dim = dim
            self.block_size = block_size
            weight, scales = self._allocate_weights()
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)

        def _allocate_weights(self):
            return (
                torch.empty(self.part_num_embeddings, self.dim),
                torch.empty(
                    self.part_num_embeddings,
                    self.dim // self.block_size,
                    dtype=torch.uint8,
                ),
            )

    class Engram:
        def _create_embedding(self, layout, layer_hash_index):
            return ("resident", layout, layer_hash_index)

    module.ParallelEngramEmbedding = ParallelEngramEmbedding
    module.Engram = Engram
    return module


def test_patch_is_idempotent_and_disabled_config_keeps_resident(monkeypatch):
    module = _target_module()
    monkeypatch.setattr(
        patch,
        "_engram_config",
        lambda: SimpleNamespace(
            cpu_offload=False,
            dp_shared_memory=False,
            embedding_across_dp=False,
        ),
    )

    assert patch.apply_to_module(module) is True
    assert patch.apply_to_module(module) is False
    result = module.Engram()._create_embedding(_layout(), 0)
    assert result[0] == "resident"


@pytest.mark.parametrize("field", ("dp_shared_memory", "embedding_across_dp"))
def test_offload_rejects_unimplemented_parallel_modes(monkeypatch, field):
    module = _target_module()
    config = dict(
        cpu_offload=True,
        dp_shared_memory=False,
        embedding_across_dp=False,
    )
    config[field] = True
    monkeypatch.setattr(patch, "_engram_config", lambda: SimpleNamespace(**config))
    monkeypatch.setattr(patch, "_is_pin_memory_available", lambda: True)
    monkeypatch.setattr(patch, "_is_uva_available", lambda: True)
    patch.apply_to_module(module)

    with pytest.raises(ValueError, match=field):
        module.Engram()._create_embedding(_layout(), 0)


def test_offload_requires_uva(monkeypatch):
    module = _target_module()
    monkeypatch.setattr(
        patch,
        "_engram_config",
        lambda: SimpleNamespace(
            cpu_offload=True,
            dp_shared_memory=False,
            embedding_across_dp=False,
        ),
    )
    monkeypatch.setattr(patch, "_is_pin_memory_available", lambda: True)
    monkeypatch.setattr(patch, "_is_uva_available", lambda: False)
    patch.apply_to_module(module)

    with pytest.raises(RuntimeError, match="requires the HCU UVA operator"):
        module.Engram()._create_embedding(_layout(), 0)


def test_patch_rejects_incompatible_create_signature():
    module = _target_module()

    def incompatible(self, layout):
        del self, layout

    module.Engram._create_embedding = incompatible
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch.apply_to_module(module)


_needs_accelerator = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an HCU accelerator"
)


@_needs_accelerator
def test_offload_allocates_pinned_cpu_storage_and_caches_views(monkeypatch):
    import vllm._C_stable_libtorch  # noqa: F401

    module = _target_module()
    monkeypatch.setattr(
        patch,
        "_engram_config",
        lambda: SimpleNamespace(
            cpu_offload=True,
            dp_shared_memory=False,
            embedding_across_dp=False,
        ),
    )
    patch.apply_to_module(module)
    embedding = module.Engram()._create_embedding(_layout(), 0)

    assert embedding.weight.device.type == "cpu"
    assert embedding.weight_scale_inv.device.type == "cpu"
    assert embedding.weight.is_pinned()
    assert embedding.weight_scale_inv.is_pinned()

    first = embedding._storage()
    second = embedding._storage()
    assert first is second
    assert first[0].device.type == "cuda"
    assert first[1].device.type == "cuda"


@_needs_accelerator
def test_storage_rebuilds_views_when_parameter_storage_changes(monkeypatch):
    import vllm._C_stable_libtorch  # noqa: F401

    module = _target_module()
    monkeypatch.setattr(
        patch,
        "_engram_config",
        lambda: SimpleNamespace(
            cpu_offload=True,
            dp_shared_memory=False,
            embedding_across_dp=False,
        ),
    )
    patch.apply_to_module(module)
    embedding = module.Engram()._create_embedding(_layout(), 0)
    first = embedding._storage()

    replacement = torch.empty(
        embedding.weight.shape,
        dtype=embedding.weight.dtype,
        device="cpu",
    )
    patch._register_host_tensor(replacement, embedding)
    embedding.weight = torch.nn.Parameter(replacement, requires_grad=False)
    second = embedding._storage()
    assert first is not second
    assert first[0].data_ptr() != second[0].data_ptr()


@_needs_accelerator
def test_register_host_tensor_pins_anonymous_allocation(monkeypatch):
    import vllm._C_stable_libtorch  # noqa: F401

    class Owner:
        pass

    # Keep the finalizer anchor alive for the duration of the check.
    owner = Owner()
    tensor = torch.empty(4096, dtype=torch.uint8, device="cpu")
    assert not tensor.is_pinned()
    patch._register_host_tensor(tensor, owner)
    assert tensor.is_pinned()
