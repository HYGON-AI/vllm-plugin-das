# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""GLM-5.2 PCP keeps complete per-rank KV cache ownership."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch


os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_VLLM_ROOT = Path(
    os.environ.get("VLLM_V0251_SOURCE_ROOT", REPO_ROOT.parent / "vllm_0251")
).resolve()
if not (TARGET_VLLM_ROOT / "vllm/__init__.py").is_file():
    raise RuntimeError(
        f"VLLM_V0251_SOURCE_ROOT does not contain vllm: {TARGET_VLLM_ROOT}"
    )
if str(TARGET_VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(TARGET_VLLM_ROOT))


def _full_attention_spec(block_size: int):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )


def _kv_cache_config(*block_sizes: int, num_blocks: int = 32):
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[f"model.layers.{index}.self_attn"],
                kv_cache_spec=_full_attention_spec(block_size),
            )
            for index, block_size in enumerate(block_sizes)
        ],
    )


def _vllm_config(
    *,
    block_size: int,
    dcp: int,
    pcp: int,
    max_model_len: int = 4096,
    enable_prefix_caching: bool = True,
    hash_block_size: int | None = None,
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
            hash_block_size=hash_block_size,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        kv_transfer_config=None,
    )


@pytest.fixture(scope="module")
def patched_cache_modules():
    from vllm.v1 import kv_cache_interface
    from vllm.v1.core import (
        kv_cache_coordinator,
        kv_cache_utils,
        single_type_kv_cache_manager,
    )
    from vllm_hcu.patch.platform.framework_opt import (
        patch_pcp_kv_cache_coordinator,
        patch_pcp_kv_cache_interface,
        patch_pcp_kv_cache_utils,
        patch_pcp_single_type_kv_cache_manager,
    )

    # Exercise only the adapters owned by this test. Calling the complete
    # platform dispatcher here is order-dependent on v0.25.1: an earlier test
    # may legitimately import vllm._aiter_ops before the fail-closed AITER
    # replacement is registered.
    # Registration is idempotent: configuration tests or plugin discovery may
    # already have installed these adapters in the shared interpreter.  A False
    # return therefore means "already active", not a setup failure; incompatible
    # or partially applied targets still fail closed inside apply_to_module().
    patch_pcp_kv_cache_utils.apply_to_module(kv_cache_utils)
    patch_pcp_kv_cache_interface.apply_to_module(kv_cache_interface)
    patch_pcp_single_type_kv_cache_manager.apply_to_module(
        single_type_kv_cache_manager
    )
    patch_pcp_kv_cache_coordinator.apply_to_module(kv_cache_coordinator)

    return SimpleNamespace(
        interface=kv_cache_interface,
        coordinator=kv_cache_coordinator,
        resolve=kv_cache_utils.resolve_kv_cache_block_sizes,
        manager=single_type_kv_cache_manager,
    )


@pytest.mark.parametrize(
    ("dcp", "pcp", "expected_effective_block_size"),
    [(1, 1, 64), (1, 2, 64), (2, 1, 128)],
)
def test_attention_kv_ownership_is_dcp_only(
    patched_cache_modules, dcp, pcp, expected_effective_block_size
):
    scheduler_size, hash_size = patched_cache_modules.resolve(
        _kv_cache_config(64),
        _vllm_config(block_size=64, dcp=dcp, pcp=pcp),
    )

    assert scheduler_size == expected_effective_block_size
    assert hash_size == expected_effective_block_size


@pytest.mark.parametrize(
    ("dcp", "pcp", "expected_scheduler", "expected_hash"),
    [(1, 1, 48, 8), (1, 2, 48, 8), (2, 2, 96, 16)],
)
def test_multi_group_resolution_scales_attention_groups_by_dcp_only(
    patched_cache_modules, dcp, pcp, expected_scheduler, expected_hash
):
    scheduler_size, hash_size = patched_cache_modules.resolve(
        _kv_cache_config(16, 24),
        _vllm_config(block_size=16, dcp=dcp, pcp=pcp),
    )

    assert scheduler_size == expected_scheduler
    assert hash_size == expected_hash


@pytest.mark.parametrize(
    ("dcp", "pcp", "expected_blocks"),
    [(1, 1, 5), (1, 2, 5), (2, 1, 3)],
)
def test_full_attention_memory_capacity_is_not_divided_by_pcp(
    patched_cache_modules, dcp, pcp, expected_blocks
):
    spec = _full_attention_spec(16)
    config = _vllm_config(
        block_size=16,
        dcp=dcp,
        pcp=pcp,
        max_model_len=65,
    )

    assert spec.max_memory_usage_bytes(config) == expected_blocks * 512


@pytest.mark.parametrize(
    ("dcp", "pcp", "expected_block_size"),
    [(1, 1, 16), (1, 2, 16), (2, 1, 32)],
)
def test_single_type_manager_allocation_granularity_is_dcp_only(
    patched_cache_modules, dcp, pcp, expected_block_size
):
    pool = SimpleNamespace(null_block=object())
    manager = patched_cache_modules.manager.FullAttentionManager(
        kv_cache_spec=_full_attention_spec(16),
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=expected_block_size,
        dcp_world_size=dcp,
        pcp_world_size=pcp,
    )

    assert manager.block_size == expected_block_size
    assert manager.dcp_world_size == dcp
    assert manager.pcp_world_size == pcp


@pytest.mark.parametrize(
    ("dcp", "pcp", "expected_hits"),
    [(1, 1, 2), (1, 2, 2), (2, 1, 1)],
)
def test_full_attention_hash_lookup_uses_dcp_only(
    patched_cache_modules, dcp, pcp, expected_hits
):
    cached = [object(), object()]

    class BlockPool:
        def get_cached_block(self, block_hash, kv_cache_group_ids):
            del kv_cache_group_ids
            return [cached[int(block_hash)]]

    hits = patched_cache_modules.manager.FullAttentionManager.find_longest_cache_hit(
        block_hashes=[0, 1],
        max_length=32,
        kv_cache_group_ids=[0],
        block_pool=BlockPool(),
        kv_cache_spec=_full_attention_spec(16),
        drop_eagle_block=False,
        alignment_tokens=16 * dcp,
        dcp_world_size=dcp,
        pcp_world_size=pcp,
    )

    assert len(hits[0]) == expected_hits


@pytest.mark.parametrize(
    ("dcp", "pcp", "expected_block_size"),
    [(1, 1, 16), (1, 2, 16), (2, 1, 32)],
)
def test_unitary_coordinator_block_size_is_dcp_only(
    monkeypatch, patched_cache_modules, dcp, pcp, expected_block_size
):
    coordinator_module = patched_cache_modules.coordinator
    base_calls: list[tuple[int, int]] = []

    def base_init(
        self,
        kv_cache_config,
        max_model_len,
        max_num_batched_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size,
        pcp_world_size,
        scheduler_block_size,
        hash_block_size,
        metrics_collector=None,
    ):
        del (
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            scheduler_block_size,
            hash_block_size,
            metrics_collector,
        )
        base_calls.append((dcp_world_size, pcp_world_size))
        self.kv_cache_config = kv_cache_config
        self.eagle_group_ids = set()
        self.single_type_managers = [SimpleNamespace(use_eagle=False)]

    monkeypatch.setattr(coordinator_module.KVCacheCoordinator, "__init__", base_init)
    coordinator = coordinator_module.UnitaryKVCacheCoordinator(
        _kv_cache_config(16),
        4096,
        256,
        False,
        False,
        False,
        dcp,
        pcp,
        expected_block_size,
        expected_block_size,
    )

    assert coordinator.block_size == expected_block_size
    assert coordinator.dcp_world_size == dcp
    assert coordinator.pcp_world_size == pcp
    assert base_calls == [(dcp, pcp)]


@pytest.mark.parametrize(
    ("dcp", "pcp", "expected_max_blocks"),
    [(1, 1, 16), (1, 4, 16), (2, 4, 8)],
)
def test_mrv2_block_table_capacity_already_depends_on_dcp_only(
    monkeypatch, dcp, pcp, expected_max_blocks
):
    from vllm.v1.worker.gpu import model_runner

    captured: dict[str, object] = {}

    class StopAfterBlockTables(RuntimeError):
        pass

    class BlockTables:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise StopAfterBlockTables

    monkeypatch.setattr(
        model_runner,
        "init_attn_backend",
        lambda *args, **kwargs: (
            [],
            SimpleNamespace(min_cg_support=None, min_cg_attn_backend=None),
            [64],
        ),
    )
    monkeypatch.setattr(model_runner, "BlockTables", BlockTables)

    runner = object.__new__(model_runner.GPUModelRunner)
    runner.max_model_len = 1024
    runner.is_encoder_decoder = False
    runner.max_num_reqs = 4
    runner.max_num_tokens = 256
    runner.device = torch.device("cpu")
    runner.dcp_size = dcp
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp,
        prefill_context_parallel_size=pcp,
    )
    runner.vllm_config = SimpleNamespace()

    with pytest.raises(StopAfterBlockTables):
        model_runner.GPUModelRunner.initialize_kv_cache(
            runner, _kv_cache_config(64)
        )

    assert captured["max_num_blocks_per_group"] == [expected_max_blocks]
    assert captured["cp_size"] == dcp


def test_cache_adapters_are_idempotent_after_registration(
    patched_cache_modules,
):
    del patched_cache_modules
    from vllm_hcu.patch.platform.framework_opt import (
        patch_pcp_kv_cache_coordinator,
        patch_pcp_kv_cache_interface,
        patch_pcp_kv_cache_utils,
        patch_pcp_single_type_kv_cache_manager,
    )

    assert patch_pcp_kv_cache_utils.apply() is False
    assert patch_pcp_kv_cache_interface.apply() is False
    assert patch_pcp_single_type_kv_cache_manager.apply() is False
    assert patch_pcp_kv_cache_coordinator.apply() is False


def test_cache_adapters_fail_closed_on_signature_drift():
    from vllm_hcu.patch.platform.framework_opt import (
        patch_pcp_kv_cache_coordinator,
        patch_pcp_kv_cache_interface,
        patch_pcp_kv_cache_utils,
        patch_pcp_single_type_kv_cache_manager,
    )
    from vllm_hcu.patch.platform.framework_opt._common import (
        PatchCompatibilityError,
    )

    utils = ModuleType(patch_pcp_kv_cache_utils.TARGET_MODULE)
    utils.resolve_kv_cache_block_sizes = lambda config: config
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_pcp_kv_cache_utils.apply_to_module(utils)

    interface = ModuleType(patch_pcp_kv_cache_interface.TARGET_MODULE)

    class FullAttentionSpec:
        def max_memory_usage_bytes(self, config, extra):
            del self, config, extra

    interface.FullAttentionSpec = FullAttentionSpec
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_pcp_kv_cache_interface.apply_to_module(interface)

    managers = ModuleType(patch_pcp_single_type_kv_cache_manager.TARGET_MODULE)

    class SingleTypeKVCacheManager:
        def __init__(self, kv_cache_spec):
            del self, kv_cache_spec

    class FullAttentionManager(SingleTypeKVCacheManager):
        @classmethod
        def find_longest_cache_hit(cls):
            del cls

    managers.SingleTypeKVCacheManager = SingleTypeKVCacheManager
    managers.FullAttentionManager = FullAttentionManager
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_pcp_single_type_kv_cache_manager.apply_to_module(managers)

    coordinator = ModuleType(patch_pcp_kv_cache_coordinator.TARGET_MODULE)

    class UnitaryKVCacheCoordinator:
        def __init__(self, kv_cache_config):
            del self, kv_cache_config

    coordinator.UnitaryKVCacheCoordinator = UnitaryKVCacheCoordinator
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch_pcp_kv_cache_coordinator.apply_to_module(coordinator)


def test_pcp1_cache_adapters_delegate_to_original_implementations():
    from vllm_hcu.patch.platform.framework_opt import (
        patch_pcp_kv_cache_coordinator,
        patch_pcp_kv_cache_interface,
        patch_pcp_kv_cache_utils,
        patch_pcp_single_type_kv_cache_manager,
    )

    calls: list[tuple[str, int]] = []
    config = _vllm_config(block_size=16, dcp=1, pcp=1)

    utils = ModuleType(patch_pcp_kv_cache_utils.TARGET_MODULE)

    class AttentionSpec:
        pass

    class MambaSpec:
        pass

    def resolve(kv_cache_config, vllm_config):
        del kv_cache_config
        calls.append(
            (
                "resolve",
                vllm_config.parallel_config.prefill_context_parallel_size,
            )
        )
        return 7, 11

    utils.AttentionSpec = AttentionSpec
    utils.MambaSpec = MambaSpec
    utils.resolve_kv_cache_block_sizes = resolve
    assert patch_pcp_kv_cache_utils.apply_to_module(utils) is True
    assert utils.resolve_kv_cache_block_sizes(SimpleNamespace(), config) == (7, 11)

    interface = ModuleType(patch_pcp_kv_cache_interface.TARGET_MODULE)

    class FullAttentionSpec:
        def max_memory_usage_bytes(self, vllm_config):
            calls.append(
                (
                    "memory",
                    vllm_config.parallel_config.prefill_context_parallel_size,
                )
            )
            return 13

    interface.FullAttentionSpec = FullAttentionSpec
    interface.cdiv = lambda numerator, denominator: (
        numerator + denominator - 1
    ) // denominator
    assert patch_pcp_kv_cache_interface.apply_to_module(interface) is True
    assert FullAttentionSpec().max_memory_usage_bytes(config) == 13

    managers = ModuleType(patch_pcp_single_type_kv_cache_manager.TARGET_MODULE)

    class SingleTypeKVCacheManager:
        def __init__(
            self,
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            scheduler_block_size,
            dcp_world_size=1,
            pcp_world_size=1,
            max_admission_blocks_per_request=None,
        ):
            del (
                kv_cache_spec,
                block_pool,
                enable_caching,
                kv_cache_group_id,
                scheduler_block_size,
                dcp_world_size,
                max_admission_blocks_per_request,
            )
            calls.append(("manager", pcp_world_size))

    class FullAttentionManager(SingleTypeKVCacheManager):
        @classmethod
        def find_longest_cache_hit(
            cls,
            block_hashes,
            max_length,
            kv_cache_group_ids,
            block_pool,
            kv_cache_spec,
            drop_eagle_block,
            alignment_tokens,
            dcp_world_size=1,
            pcp_world_size=1,
        ):
            del (
                cls,
                block_hashes,
                max_length,
                kv_cache_group_ids,
                block_pool,
                kv_cache_spec,
                drop_eagle_block,
                alignment_tokens,
                dcp_world_size,
            )
            calls.append(("hash", pcp_world_size))
            return (["original"],)

    managers.SingleTypeKVCacheManager = SingleTypeKVCacheManager
    managers.FullAttentionManager = FullAttentionManager
    assert patch_pcp_single_type_kv_cache_manager.apply_to_module(managers) is True
    FullAttentionManager(None, None, False, 0, 16, pcp_world_size=1)
    assert FullAttentionManager.find_longest_cache_hit(
        [], 0, [0], None, None, False, 16, pcp_world_size=1
    ) == (["original"],)

    coordinator = ModuleType(patch_pcp_kv_cache_coordinator.TARGET_MODULE)

    class KVCacheCoordinator:
        pass

    class UnitaryKVCacheCoordinator(KVCacheCoordinator):
        def __init__(
            self,
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
            pcp_world_size,
            scheduler_block_size,
            hash_block_size,
            metrics_collector=None,
        ):
            del (
                self,
                kv_cache_config,
                max_model_len,
                max_num_batched_tokens,
                use_eagle,
                enable_caching,
                enable_kv_cache_events,
                dcp_world_size,
                scheduler_block_size,
                hash_block_size,
                metrics_collector,
            )
            calls.append(("coordinator", pcp_world_size))

    coordinator.KVCacheCoordinator = KVCacheCoordinator
    coordinator.UnitaryKVCacheCoordinator = UnitaryKVCacheCoordinator
    assert patch_pcp_kv_cache_coordinator.apply_to_module(coordinator) is True
    UnitaryKVCacheCoordinator(None, 1, 1, False, False, False, 1, 1, 16, 16)

    assert calls == [
        ("resolve", 1),
        ("memory", 1),
        ("manager", 1),
        ("hash", 1),
        ("coordinator", 1),
    ]
