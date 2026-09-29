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

        def _get_shard_info(self):
            return 1, 0

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
        def _init_staging(self, max_tokens, head_dim):
            del max_tokens, head_dim

        def _create_embedding(self, layout, layer_hash_index):
            return ("resident", layout, layer_hash_index)

        def prepare_embeddings(self, hash_ids):
            del hash_ids

        def embed(self, hash_ids):
            del hash_ids

        def forward(self, hidden_states, hash_ids, token_mask=None):
            del hash_ids, token_mask
            return hidden_states.clone()

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


@pytest.mark.parametrize("field", ("dp_shared_memory",))
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


def test_dp_offload_allows_embedding_across_dp(monkeypatch):
    module = _target_module()
    monkeypatch.setattr(patch, "_require_supported_config", lambda: None)
    monkeypatch.setattr(patch, "_is_pin_memory_available", lambda: True)
    monkeypatch.setattr(patch, "_is_uva_available", lambda: True)
    monkeypatch.setattr(
        patch,
        "_engram_dp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0),
    )
    monkeypatch.setattr(
        patch,
        "_pcp_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        patch,
        "_engram_config",
        lambda: SimpleNamespace(
            cpu_offload=True,
            dp_shared_memory=False,
            embedding_across_dp=True,
        ),
    )
    monkeypatch.setattr(
        patch,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                data_parallel_size=2,
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        patch,
        "_allocate_registered_host_tensor",
        lambda shape, dtype, owner: torch.empty(shape, dtype=dtype),
    )
    patch.apply_to_module(module)

    result = module.Engram()._create_embedding(_layout(), 0)

    assert result is not None


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


def test_pcp_hash_gather_pads_to_common_token_slot(monkeypatch):
    class Group:
        world_size = 2

        def all_gather(self, tensor, dim=0):
            if tensor.numel() == 1:
                return torch.tensor([2, 3], dtype=tensor.dtype)
            remote = tensor.new_full((1, tensor.shape[1]), 7)
            return torch.cat((tensor, remote), dim=dim)

    monkeypatch.setattr(patch, "_pcp_group", lambda: Group())
    hashes = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)

    gathered, slot = patch._gather_padded_hash_ids(hashes)

    assert slot == 3
    assert gathered.shape == (4, 2)
    assert torch.equal(gathered[:3], torch.tensor([[1, 2], [3, 4], [-1, -1]]))
    assert torch.equal(gathered[3], torch.tensor([7, 7]))


def test_pcp_shard_info_is_tp_major():
    shards = [
        patch._pcp_shard_info(4, tp_rank, 2, pcp_rank)[1]
        for tp_rank in range(4)
        for pcp_rank in range(2)
    ]

    assert shards == list(range(8))


def test_dp_shard_info_is_tp_major():
    shards = [
        patch._engram_dp_shard_info(1, 0, 8, dp_rank)[1]
        for dp_rank in range(8)
    ]

    assert shards == list(range(8))


def test_dp_hash_gather_pads_to_common_token_slot(monkeypatch):
    class Group:
        world_size = 2

        rank_in_group = 0

        def all_gather(self, tensor, dim=0):
            if tensor.numel() == 1:
                return torch.tensor([2, 3], dtype=tensor.dtype)
            remote = tensor.new_full((3, tensor.shape[1]), 7)
            return torch.cat((tensor, remote), dim=dim)

    class Metadata:
        num_tokens_across_dp_cpu = torch.tensor([2, 3], dtype=torch.int64)

    class ForwardContext:
        dp_metadata = Metadata()

    monkeypatch.setattr(patch, "_engram_dp_group", lambda: Group())
    monkeypatch.setattr(
        patch, "_engram_config", lambda: SimpleNamespace(embedding_across_dp=True)
    )
    monkeypatch.setattr(
        "vllm.forward_context.get_forward_context", lambda: ForwardContext()
    )
    hashes = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)

    gathered, slot = patch._gather_padded_dp_hash_ids(hashes)

    assert slot == 3
    assert gathered.shape == (6, 2)
    assert torch.equal(gathered[:3], torch.tensor([[1, 2], [3, 4], [-1, -1]]))
    assert torch.equal(gathered[3:], torch.tensor([[7, 7], [7, 7], [7, 7]]))


def test_zero_token_engram_forward_skips_hcu_wkv(monkeypatch):
    module = _target_module()
    monkeypatch.setattr(
        patch,
        "_engram_config",
        lambda: SimpleNamespace(
            cpu_offload=True,
            dp_shared_memory=False,
            embedding_across_dp=True,
        ),
    )
    monkeypatch.setattr(patch, "_engram_dp_size", lambda: 2)
    patch.apply_to_module(module)
    original = module.Engram.forward
    hidden = torch.empty(0, 4, 32)
    result = original(module.Engram(), hidden, torch.empty(0, 2, dtype=torch.int32))

    assert result.shape == hidden.shape


def test_dp_forward_uses_shard_size_captured_on_embedding(monkeypatch):
    module = _target_module()
    monkeypatch.setattr(patch, "_engram_dp_size", lambda: 2)
    patch.apply_to_module(module)
    original = module.Engram.forward
    instance = module.Engram()
    instance.embed_tokens = SimpleNamespace(**{patch._DP_SIZE_MARKER: 2})
    instance.embed = lambda hash_ids: torch.empty(
        0, 2, 32, dtype=torch.bfloat16
    )
    monkeypatch.setattr(patch, "_engram_dp_size", lambda: 1)

    hidden = torch.ones(2, 4, 32)
    result = original(instance, hidden, torch.ones(2, 2, dtype=torch.int32))

    assert torch.equal(result, torch.zeros_like(hidden))


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


def test_allocate_registered_host_tensor_keeps_mmap_alive(monkeypatch):
    owner = SimpleNamespace()
    registered: list[torch.Tensor] = []

    def fake_register(tensor, _owner):
        registered.append(tensor)
        return tensor

    monkeypatch.setattr(patch, "_register_host_tensor", fake_register)
    tensor = patch._allocate_registered_host_tensor((8, 16), torch.uint8, owner)

    assert tensor.shape == (8, 16)
    assert tensor.device.type == "cpu"
    assert len(owner._hcu_engram_mappings) == 1
    assert registered == [tensor]
    tensor.zero_()
    assert int(tensor.sum()) == 0
