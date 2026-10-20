# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

import asyncio
import importlib
import os
from types import SimpleNamespace

import msgspec
import pytest
import torch

os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

from vllm_hcu.patch.platform.framework_opt import patch_mooncake_connector


@pytest.fixture(scope="module")
def mooncake():
    return importlib.import_module(patch_mooncake_connector.TARGET_MODULE)


class _RegistrationEngine:
    def __init__(self):
        self.calls: list[tuple[list[int], list[int]]] = []

    def batch_register_memory(self, ptrs, lengths):
        self.calls.append((list(ptrs), list(lengths)))
        return 0


def _worker(
    mooncake,
    *,
    blocks_first: bool,
    split_k_and_v: bool = False,
    use_mla: bool = False,
    tp_size: int = 2,
    tp_rank: int = 0,
):
    worker = object.__new__(mooncake.MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker.use_mla = use_mla
    # Use the real v0.28 topology: it has no v0.25 cache-layout helpers.
    worker.transfer_topo = mooncake.TransferTopology(
        tp_rank=tp_rank, tp_size=tp_size, block_size=16, engine_id="test",
        is_mla=use_mla, is_mamba=False, total_num_kv_heads=16,
        attn_backends=[],
    )
    worker._pp_layer_range = None
    worker.pp_rank = 0
    worker.pp_size = 2
    worker.vllm_config = SimpleNamespace(speculative_config=None)
    worker.engine = _RegistrationEngine()
    worker.is_kv_consumer = True
    worker._layer_specs = {}
    worker._layer_group_indices = {}
    worker._physical_blocks_per_logical_kv_block = 1
    worker.tp_rank = tp_rank
    worker.tp_size = tp_size
    worker.finished_recving_reqs = set()
    worker._kv_load_error = None
    # object.__new__ skips MooncakeConnectorWorker.__init__.
    worker.dsa_layer_split_enabled = False
    worker._remote_pcp_info = {}
    return worker


def _metadata(mooncake, **overrides):
    fields = {
        "remote_hostname": "127.0.0.1",
        "remote_port": 1234,
        "remote_tp_size": 2,
        "remote_tp_rank": 0,
        "req_blocks": {},
        "kv_caches_base_addr": [],
        "block_lens": [],
        "kv_block_lens": [],
        "registered_layer_names": [],
        "registered_layer_indices": [],
        "registered_group_indices": [],
    }
    fields.update(overrides)
    return mooncake.MooncakeXferMetadata(**fields)


def _single_group_regions(mooncake, *, local_payload=8, remote_payload=8):
    local = [
        mooncake.TransferRegion(
            "model.layers.0.self_attn", 0, 1000, 32, local_payload, 0
        )
    ]
    remote = [
        mooncake.TransferRegion(
            "model.layers.0.self_attn", 0, 2000, 32, remote_payload, 0
        )
    ]
    return local, remote


def test_target_metadata_schema_and_round_trip(mooncake):
    assert mooncake.MooncakeXferMetadata.__struct_fields__ == (
        "remote_hostname",
        "remote_port",
        "remote_tp_size",
        "remote_tp_rank",
        "req_blocks",
        "kv_caches_base_addr",
        "block_lens",
        "kv_block_lens",
        "registered_layer_names",
        "registered_layer_indices",
        "registered_group_indices",
        "num_blocks_per_layer",
    )
    metadata = _metadata(
        mooncake,
        req_blocks={"r": ("x", [[3, 1]])},
        kv_caches_base_addr=[100],
        block_lens=[64],
        kv_block_lens=[24],
        num_blocks_per_layer=[8],
        registered_layer_names=["model.layers.0.self_attn"],
        registered_layer_indices=[0],
        registered_group_indices=[0],
    )
    encoded = msgspec.msgpack.encode(metadata)
    decoded = msgspec.msgpack.decode(encoded, type=mooncake.MooncakeXferMetadata)
    assert decoded == metadata


def test_mooncake_uses_v028_page_layout(mooncake):
    """Whole head slices must be contiguous within v0.28 pages."""
    config = SimpleNamespace(model_config=SimpleNamespace(use_mla=False))
    assert mooncake.MooncakeConnector.get_required_kvcache_layout(config) == "LBHNC"

    mla_config = SimpleNamespace(model_config=SimpleNamespace(use_mla=True))
    assert mooncake.MooncakeConnector.get_required_kvcache_layout(mla_config) is None

    incomplete_config = SimpleNamespace(model_config=None)
    assert (
        mooncake.MooncakeConnector.get_required_kvcache_layout(incomplete_config)
        is None
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"remote_block_lens": [64, 64]}, "inconsistent cache-entry counts"),
        ({"local_layer_indices": []}, "local identity fields"),
        ({"remote_base_addrs": [0]}, "positive base addresses"),
        ({"remote_block_lens": [0]}, "positive physical block strides"),
        ({"remote_kv_block_lens": [0]}, "positive KV payload sizes"),
        ({"remote_kv_block_lens": [65]}, "exceeds its physical block stride"),
        ({"remote_layer_names": [""]}, "non-empty layer name"),
        ({"remote_layer_indices": [-1]}, "non-negative layer indices"),
        ({"remote_group_indices": [-1]}, "non-negative group indices"),
        ({"remote_num_blocks_per_layer": []}, "positive block capacity"),
        ({"remote_num_blocks_per_layer": [0]}, "positive block capacity"),
        ({"remote_num_blocks_per_layer": [-1]}, "positive block capacity"),
        ({"remote_num_blocks_per_layer": [True]}, "positive block capacity"),
        ({"remote_num_blocks_per_layer": [2, 3]}, "positive block capacity"),
    ],
)
def test_phase1_metadata_is_structural_and_fail_closed(
    mooncake, overrides, message
):
    kwargs = {
        "local_layer_names": ["model.layers.0.self_attn"],
        "local_layer_indices": [0],
        "local_group_indices": [0],
        "remote_base_addrs": [100],
        "remote_block_lens": [64],
        "remote_kv_block_lens": [24],
        "remote_layer_names": ["model.layers.0.self_attn"],
        "remote_layer_indices": [0],
        "remote_group_indices": [0],
    }
    kwargs.update(overrides)
    assert message in mooncake._validate_phase1_metadata(**kwargs)


def test_phase1_metadata_defers_identity_to_region_alignment(mooncake):
    assert (
        mooncake._validate_phase1_metadata(
            local_layer_names=[
                "model.layers.1.self_attn",
                "model.layers.0.self_attn",
            ],
            local_layer_indices=[1, 0],
            local_group_indices=[0, 0],
            remote_base_addrs=[100, 200],
            remote_block_lens=[64, 64],
            remote_kv_block_lens=[24, 24],
            remote_layer_names=[
                "model.layers.0.self_attn",
                "model.layers.1.self_attn",
            ],
            remote_layer_indices=[0, 1],
            remote_group_indices=[0, 0],
        )
        is None
    )


def test_region_alignment_matches_occurrences_independent_of_order(mooncake):
    local = [
        mooncake.TransferRegion("layer.1", 1, 101, 32, 16, 0),
        mooncake.TransferRegion("layer.0", 0, 102, 32, 16, 0),
        mooncake.TransferRegion("layer.0", 0, 103, 32, 16, 0),
    ]
    remote = [
        mooncake.TransferRegion("layer.0", 0, 201, 32, 16, 0),
        mooncake.TransferRegion("layer.1", 1, 202, 32, 16, 0),
        mooncake.TransferRegion("layer.0", 0, 203, 32, 16, 0),
    ]

    aligned_local, aligned_remote, error = mooncake._align_transfer_regions(
        local, remote
    )

    assert error is None
    assert aligned_local == local
    assert [region.base_addr for region in aligned_remote] == [202, 201, 203]


def test_region_alignment_rejects_missing_repeated_occurrence(mooncake):
    local = [
        mooncake.TransferRegion("layer.0", 0, 101, 32, 16, 0),
        mooncake.TransferRegion("layer.0", 0, 102, 32, 16, 0),
    ]
    remote = [mooncake.TransferRegion("layer.0", 0, 201, 32, 16, 0)]

    aligned_local, aligned_remote, error = mooncake._align_transfer_regions(
        local, remote
    )

    assert aligned_local == []
    assert aligned_remote == []
    assert error is not None and "occurrence 1" in error


@pytest.mark.parametrize(
    ("remote", "message"),
    [
        (["layer.0", 1, 201, 32, 16, 0], "layer index mismatch"),
        (["layer.0", 0, 201, 32, 16, 1], "group index mismatch"),
    ],
)
def test_region_alignment_rejects_identity_mismatch(mooncake, remote, message):
    local = [mooncake.TransferRegion("layer.0", 0, 101, 32, 16, 0)]

    aligned_local, aligned_remote, error = mooncake._align_transfer_regions(
        local, [mooncake.TransferRegion(*remote)]
    )

    assert aligned_local == []
    assert aligned_remote == []
    assert error is not None and message in error


def test_phase1_feature_gates(mooncake):
    valid = SimpleNamespace(has_mamba_layers=False, kv_cache_groups=[object()])
    mooncake._validate_phase1_kv_cache_config(valid, 1)
    mooncake._validate_phase1_kv_cache_config(valid, 2)
    mooncake._validate_phase1_kv_cache_config(
        SimpleNamespace(
            has_mamba_layers=False,
            kv_cache_groups=[object(), object()],
        ),
        1,
    )
    with pytest.raises(ValueError, match="positive number"):
        mooncake._validate_phase1_kv_cache_config(valid, 0)
    with pytest.raises(NotImplementedError, match="Mamba/GDN"):
        mooncake._validate_phase1_kv_cache_config(
            SimpleNamespace(has_mamba_layers=True, kv_cache_groups=[object()]), 1
        )
    with pytest.raises(ValueError, match="at least one"):
        mooncake._validate_phase1_kv_cache_config(
            SimpleNamespace(has_mamba_layers=False, kv_cache_groups=[]), 1
        )


def test_logical_attention_blocks_expand_to_physical_kernel_blocks(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    worker._physical_blocks_per_logical_kv_block = 2
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
    )

    assert worker._logical_to_kernel_block_ids([[2, 0, 1]]) == [
        [4, 5, 0, 1, 2, 3]
    ]


def test_sync_block_size_enables_exact_logical_to_physical_ratio(
    mooncake, monkeypatch
):
    worker = object.__new__(mooncake.MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker.block_size = 64
    worker.vllm_config = object()
    worker._physical_blocks_per_logical_kv_block = 1
    monkeypatch.setattr(mooncake, "get_current_attn_backends", lambda _: [])
    monkeypatch.setattr(
        mooncake,
        "select_common_block_size",
        lambda logical, backends: 32,
    )

    worker._sync_block_size_with_kernel()

    assert worker.block_size == 32
    assert worker._physical_blocks_per_logical_kv_block == 2


@pytest.mark.parametrize("kernel_block_size", [0, 48])
def test_sync_block_size_rejects_invalid_physical_ratio(
    mooncake, monkeypatch, kernel_block_size
):
    worker = object.__new__(mooncake.MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker.block_size = 64
    worker.vllm_config = object()
    worker._physical_blocks_per_logical_kv_block = 1
    monkeypatch.setattr(mooncake, "get_current_attn_backends", lambda _: [])
    monkeypatch.setattr(
        mooncake,
        "select_common_block_size",
        lambda logical, backends: kernel_block_size,
    )

    with pytest.raises(ValueError, match="positive|integer multiple"):
        worker._sync_block_size_with_kernel()


def test_phase1_gate_runs_before_transfer_engine_creation(mooncake, monkeypatch):
    engine_creations = []

    class _ForbiddenEngine:
        def __init__(self):
            engine_creations.append(True)

    monkeypatch.setattr(mooncake, "TransferEngine", _ForbiddenEngine)
    monkeypatch.setattr(
        mooncake.torch.accelerator,
        "current_device_index",
        lambda: 0,
    )
    monkeypatch.setattr(mooncake.current_platform, "set_device", lambda _: None)
    monkeypatch.setattr(
        mooncake.MooncakeConnectorWorker,
        "_sync_block_size_with_kernel",
        lambda self: None,
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(block_size=64),
        model_config=SimpleNamespace(use_mla=False),
    )
    unsupported = SimpleNamespace(
        has_mamba_layers=False,
        kv_cache_groups=[],
    )

    with pytest.raises(ValueError, match="at least one"):
        mooncake.MooncakeConnectorWorker(config, "engine", unsupported)

    assert engine_creations == []


def test_region_expansion_splits_combined_kv_only_when_requested(mooncake):
    common = {
        "base_addrs": [100],
        "block_lens": [40],
        "kv_block_lens": [16],
        "layer_names": ["model.layers.0.self_attn"],
        "layer_indices": [0],
        "group_indices": [0],
    }
    split = mooncake._expand_transfer_regions(
        **common, is_kv_layout_blocks_first=True
    )
    assert [(r.base_addr, r.block_len, r.kv_block_len) for r in split] == [
        (100, 40, 16),
        (116, 40, 16),
    ]
    unsplit = mooncake._expand_transfer_regions(
        **common, is_kv_layout_blocks_first=False
    )
    assert len(unsplit) == 1
    assert unsplit[0].layer_name == "model.layers.0.self_attn"


def test_region_expansion_rejects_payload_larger_than_stride(mooncake):
    with pytest.raises(ValueError, match="exceeds its physical block stride"):
        mooncake._expand_transfer_regions(
            base_addrs=[100],
            block_lens=[24],
            kv_block_lens=[16],
            layer_names=["model.layers.0.self_attn"],
            layer_indices=[0],
            group_indices=[0],
            is_kv_layout_blocks_first=True,
        )


def _attention_spec(mooncake, *, heads=1, block_size=2, head_size=2, **kwargs):
    return mooncake.FullAttentionSpec(
        block_size=block_size, num_kv_heads=heads, head_size=head_size,
        dtype=torch.float16, **kwargs,
    )


def test_registration_preserves_padded_stride_and_deduplicates_storage(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    raw = torch.zeros(128, dtype=torch.float16)
    caches = {
        f"model.layers.{i}.self_attn": torch.as_strided(
            raw, size=(2, 1, 2, 4), stride=(16, 8, 4, 1), storage_offset=offset,
        )
        for i, offset in enumerate((0, 40))
    }
    for i, name in enumerate(caches):
        worker._layer_specs[name] = _attention_spec(mooncake)
        worker._layer_group_indices[name] = i
    worker.register_kv_caches(caches)
    assert worker.block_len_per_layer == [32, 32]
    assert worker.kv_block_len_per_layer == [16, 16]
    assert worker.registered_layer_indices == [0, 1]
    assert worker.registered_group_indices == [0, 1]
    assert worker.kv_caches_base_addr == [cache.data_ptr() for cache in caches.values()]
    assert worker.engine.calls == [
        ([raw.untyped_storage().data_ptr()], [raw.untyped_storage().nbytes()])
    ]
    regions = worker._get_transfer_regions(
        worker.kv_caches_base_addr, worker.block_len_per_layer,
        worker.kv_block_len_per_layer, worker.registered_layer_names,
        worker.registered_layer_indices, worker.registered_group_indices,
    )
    # K and V are packed per head/state in C: do not split whole pages in half.
    assert len(regions) == 2
    assert [r.kv_block_len for r in regions] == [16, 16]
    assert [r.group_index for r in regions] == [0, 1]


@pytest.mark.parametrize("spec_type", ["MLAAttentionSpec", "SlidingWindowMLASpec"])
def test_registration_supports_aligned_mla_pages(mooncake, spec_type):
    worker = _worker(mooncake, blocks_first=False, use_mla=True)
    kwargs = {"sliding_window": 8} if spec_type == "SlidingWindowMLASpec" else {}
    spec = getattr(mooncake, spec_type)(
        block_size=2, num_kv_heads=1, head_size=3, dtype=torch.float16,
        alignment=32, **kwargs,
    )
    name = "model.layers.0.self_attn"
    worker._layer_specs[name] = spec
    worker._layer_group_indices[name] = 0
    raw = torch.zeros(32, dtype=torch.float16)
    cache = torch.as_strided(raw, size=(2, 1, 2, 3), stride=(16, 6, 3, 1))
    worker.register_kv_caches({name: cache})
    assert worker.kv_block_len_per_layer == [32]
    assert worker.block_len_per_layer == [32]


def test_registration_scales_payload_for_kernel_block_splitting(mooncake):
    worker = _worker(mooncake, blocks_first=False)
    worker._physical_blocks_per_logical_kv_block = 2
    name = "model.layers.0.self_attn"
    worker._layer_specs[name] = _attention_spec(mooncake, block_size=4)
    worker._layer_group_indices[name] = 0
    worker.register_kv_caches({name: torch.zeros((4, 1, 2, 4), dtype=torch.float16)})
    assert worker.kv_block_len_per_layer == [16]
    assert worker.block_len_per_layer == [16]


@pytest.mark.parametrize("cache", [
    torch.zeros((2, 2, 2, 4)).transpose(0, 1),
    (torch.zeros((2, 1, 2, 2)), torch.zeros((2, 1, 2, 2))),
])
def test_registration_rejects_non_page_or_scattered_layouts(mooncake, cache):
    worker = _worker(mooncake, blocks_first=False)
    name = "model.layers.0.self_attn"
    worker._layer_specs[name] = _attention_spec(mooncake)
    worker._layer_group_indices[name] = 0
    with pytest.raises(ValueError, match="canonical contiguous|page tensor"):
        worker.register_kv_caches({name: cache})
    assert worker.engine.calls == []


def test_registration_rejects_payload_larger_than_stride(mooncake):
    worker = _worker(mooncake, blocks_first=False)
    name = "model.layers.0.self_attn"
    worker._layer_specs[name] = _attention_spec(mooncake, page_size_padded=64)
    worker._layer_group_indices[name] = 0
    with pytest.raises(ValueError, match="exceeds its physical block stride"):
        worker.register_kv_caches({name: torch.zeros((2, 1, 2, 4), dtype=torch.float16)})
    assert worker.engine.calls == []


def test_registration_rejects_view_outside_storage(mooncake, monkeypatch):
    worker = _worker(mooncake, blocks_first=True)
    name = "model.layers.0.self_attn"
    worker._layer_specs[name] = _attention_spec(mooncake)
    worker._layer_group_indices[name] = 0
    cache = torch.zeros((2, 1, 2, 4), dtype=torch.float16)
    real_storage = cache.untyped_storage()
    monkeypatch.setattr(
        torch.Tensor, "untyped_storage",
        lambda self: SimpleNamespace(data_ptr=real_storage.data_ptr, nbytes=lambda: 1),
    )
    with pytest.raises(ValueError, match="exceeds its registered storage"):
        worker.register_kv_caches({name: cache})


def test_random_block_transfer_uses_stride_and_payload(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
    )
    local_regions = mooncake._expand_transfer_regions(
        base_addrs=[1000],
        block_lens=[32],
        kv_block_lens=[8],
        layer_names=["model.layers.0.self_attn"],
        layer_indices=[0],
        group_indices=[0],
        is_kv_layout_blocks_first=True,
    )
    remote_regions = mooncake._expand_transfer_regions(
        base_addrs=[2000],
        block_lens=[40],
        kv_block_lens=[8],
        layer_names=["model.layers.0.self_attn"],
        layer_indices=[0],
        group_indices=[0],
        is_kv_layout_blocks_first=True,
    )
    send_meta = mooncake.SendBlockMeta(
        p_req_id="p",
        transfer_id="x",
        local_block_ids=[[2, 0, 1]],
        ready=asyncio.Event(),
    )
    metadata = _metadata(
        mooncake,
        req_blocks={"d": ("x", [[1, 2, 0]])},
    )
    src, dst, lengths, errors, message = asyncio.run(
        worker._build_transfer_params(
            [("d", send_meta)], metadata, local_regions, remote_regions
        )
    )
    assert errors == [] and message is None
    assert lengths == [8] * 6
    assert src == [1064, 1000, 1032, 1072, 1008, 1040]
    assert dst == [2040, 2080, 2000, 2048, 2088, 2008]


@pytest.mark.parametrize(
    ("local_blocks", "remote_blocks", "message"),
    [
        ([[0]], [[0], [1]], "KV group count mismatch"),
        ([[0]], [[0, 1]], "P num blocks less than D"),
        ([[-1]], [[0]], "Mooncake producer block ID -1 is negative"),
        ([[0]], [[-1]], "Mooncake consumer block ID -1 is negative"),
    ],
)
def test_transfer_planning_rejects_incompatible_block_tables(
    mooncake, local_blocks, remote_blocks, message
):
    worker = _worker(mooncake, blocks_first=True)
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
    )
    local, remote = _single_group_regions(mooncake)
    send_meta = mooncake.SendBlockMeta(
        p_req_id="p",
        transfer_id="x",
        local_block_ids=local_blocks,
        ready=asyncio.Event(),
    )
    metadata = _metadata(mooncake, req_blocks={"d": ("x", remote_blocks)})

    src, dst, lengths, errors, error = asyncio.run(
        worker._build_transfer_params([("d", send_meta)], metadata, local, remote)
    )

    assert (src, dst, lengths) == ([], [], [])
    assert errors == ["d"]
    assert error is not None and message in error


def test_failed_pull_result_is_fatal_without_marking_kv_ready(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    pull_meta = mooncake.PullReqMeta(
        d_req_id="d",
        transfer_id="x",
        local_block_ids=[[0]],
        remote_engine_id="engine",
        remote_bootstrap_addr="bootstrap",
        pull_tasks_count=1,
    )

    worker.process_pulling_result(
        mooncake.MooncakeXferResponse(
            status=mooncake.MooncakeXferResponseStatus.ERROR,
            err_reqs=["d"],
            err_msg="transfer failed",
        ),
        {"d": pull_meta},
    )

    assert worker.finished_recving_reqs == set()
    with pytest.raises(RuntimeError, match="Mooncake KV receive failed"):
        asyncio.run(worker.fetch_finished_recving_reqs())


def test_bootstrap_failure_is_fatal_without_started_tasks(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    pull_meta = mooncake.PullReqMeta(
        d_req_id="d",
        transfer_id="x",
        local_block_ids=[[0]],
        remote_engine_id="missing",
        remote_bootstrap_addr="bootstrap",
    )

    worker._fail_pull_metas({"d": pull_meta}, "engine not found")

    assert worker.finished_recving_reqs == set()
    with pytest.raises(RuntimeError, match="Mooncake KV receive failed"):
        asyncio.run(worker.fetch_finished_recving_reqs())


def test_transfer_planning_uses_only_uncached_suffix(mooncake):
    worker = _worker(mooncake, blocks_first=False)
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
    )
    local, remote = _single_group_regions(mooncake)
    send_meta = mooncake.SendBlockMeta(
        p_req_id="p",
        transfer_id="x",
        local_block_ids=[[7, 8, 9]],
        ready=asyncio.Event(),
    )
    metadata = _metadata(mooncake, req_blocks={"d": ("x", [[3, 4]])})

    src, dst, lengths, errors, error = asyncio.run(
        worker._build_transfer_params([("d", send_meta)], metadata, local, remote)
    )

    assert (src, dst, lengths) == ([1256, 1288], [2096, 2128], [8, 8])
    assert errors == [] and error is None


@pytest.mark.parametrize(
    (
        "local_tp_rank",
        "local_tp_size",
        "remote_tp_rank",
        "remote_tp_size",
        "local_payload",
        "remote_payload",
        "expected_src",
        "expected_dst",
    ),
    [
        (0, 2, 0, 4, 8, 4, [1000, 1008], [2000, 2004]),
        (0, 2, 1, 4, 8, 4, [1004, 1012], [2000, 2004]),
        (0, 4, 0, 2, 4, 8, [1000, 1004], [2000, 2008]),
        (1, 4, 0, 2, 4, 8, [1000, 1004], [2004, 2012]),
    ],
)
def test_cutlass_asymmetric_tp_uses_hnd_head_slice_transfer_plan(
    mooncake,
    local_tp_rank,
    local_tp_size,
    remote_tp_rank,
    remote_tp_size,
    local_payload,
    remote_payload,
    expected_src,
    expected_dst,
):
    """Cover asymmetric TP planning only for the CUTLASS/HND contract."""
    worker = _worker(
        mooncake, blocks_first=True, tp_rank=local_tp_rank, tp_size=local_tp_size,
    )
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
    )
    local = mooncake._expand_transfer_regions(
        base_addrs=[1000],
        block_lens=[32],
        kv_block_lens=[local_payload],
        layer_names=["model.layers.0.self_attn"],
        layer_indices=[0],
        group_indices=[0],
        is_kv_layout_blocks_first=True,
    )
    remote = mooncake._expand_transfer_regions(
        base_addrs=[2000],
        block_lens=[32],
        kv_block_lens=[remote_payload],
        layer_names=["model.layers.0.self_attn"],
        layer_indices=[0],
        group_indices=[0],
        is_kv_layout_blocks_first=True,
    )
    send_meta = mooncake.SendBlockMeta(
        p_req_id="p", transfer_id="x", local_block_ids=[[0]], ready=asyncio.Event()
    )
    metadata = _metadata(
        mooncake,
        remote_tp_rank=remote_tp_rank,
        remote_tp_size=remote_tp_size,
        req_blocks={"d": ("x", [[0]])},
    )
    src, dst, lengths, errors, message = asyncio.run(
        worker._build_transfer_params([("d", send_meta)], metadata, local, remote)
    )
    assert errors == [] and message is None
    assert src == expected_src
    assert dst == expected_dst
    assert lengths == [min(local_payload, remote_payload)] * 2


def test_contiguous_dense_blocks_can_coalesce(mooncake):
    worker = _worker(mooncake, blocks_first=False)
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
    )
    local = [
        mooncake.TransferRegion("model.layers.0.self_attn", 0, 100, 8, 8, 0)
    ]
    remote = [
        mooncake.TransferRegion("model.layers.0.self_attn", 0, 200, 8, 8, 0)
    ]
    send_meta = mooncake.SendBlockMeta(
        p_req_id="p",
        transfer_id="x",
        local_block_ids=[[1, 2, 3]],
        ready=asyncio.Event(),
    )
    metadata = _metadata(
        mooncake,
        req_blocks={"d": ("x", [[5, 6, 7]])},
    )
    src, dst, lengths, errors, message = asyncio.run(
        worker._build_transfer_params([("d", send_meta)], metadata, local, remote)
    )
    assert (src, dst, lengths) == ([108], [240], [24])
    assert errors == [] and message is None


@pytest.mark.parametrize(
    ("tp_rank", "pp_rank", "local_only", "dp_local", "dp_index", "expected"),
    [
        (1, 0, True, 0, 0, False),
        (0, 1, True, 0, 0, False),
        (0, 0, True, 0, 3, True),
        (0, 0, True, 1, 0, False),
        (0, 0, False, 0, 0, True),
        (0, 0, False, 0, 1, False),
    ],
)
def test_bootstrap_launch_is_owned_by_designated_tp_pp_dp_rank(
    mooncake,
    monkeypatch,
    tp_rank,
    pp_rank,
    local_only,
    dp_local,
    dp_index,
    expected,
):
    monkeypatch.setattr(mooncake, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(
        mooncake, "get_pp_group", lambda: SimpleNamespace(rank_in_group=pp_rank)
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            local_engines_only=local_only,
            data_parallel_rank_local=dp_local,
            data_parallel_index=dp_index,
        )
    )

    assert mooncake.should_launch_bootstrap_server(config) is expected


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", "kv_both", None])
def test_hyv4_producer_only_pcp_launches_one_bootstrap_per_global_first_rank(
    mooncake, monkeypatch, role,
):
    from vllm.distributed import parallel_state

    # GroupCoordinator.rank is global; each PCP group is strided across TP.
    group = object.__new__(parallel_state.GroupCoordinator)
    monkeypatch.setattr(parallel_state, "_PCP", group)
    monkeypatch.setattr(mooncake, "get_tensor_model_parallel_rank", lambda: group.rank % 2)
    monkeypatch.setattr(mooncake, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role=role),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=4, local_engines_only=False,
            data_parallel_index=0,
        ),
    )
    owners = []
    for global_rank in range(8):
        group.rank = global_rank
        group.ranks = [0, 2, 4, 6] if global_rank % 2 == 0 else [1, 3, 5, 7]
        group.rank_in_group = global_rank // 2
        if mooncake.should_launch_bootstrap_server(config):
            owners.append(global_rank)
    assert owners == ([0] if role == "kv_producer" else [])


@pytest.mark.parametrize("role,expected", [
    ("kv_producer", True), ("kv_both", True), ("kv_consumer", False), (None, False),
])
def test_mooncake_pp1_bootstrap_preserves_sending_roles(mooncake, monkeypatch, role, expected):
    monkeypatch.setattr(mooncake, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mooncake, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role=role),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, local_engines_only=False,
            data_parallel_index=0,
        ),
    )
    assert mooncake.should_launch_bootstrap_server(config) is expected


@pytest.mark.parametrize(
    ("local_only", "nodes", "expected_host"),
    [
        (True, 1, "127.0.0.1"),
        (False, 2, "model-master"),
        (False, 1, "dp-master"),
    ],
)
def test_bootstrap_address_follows_engine_topology(
    mooncake, monkeypatch, local_only, nodes, expected_host
):
    monkeypatch.setattr(mooncake.envs, "VLLM_MOONCAKE_BOOTSTRAP_PORT", 9876)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            local_engines_only=local_only,
            nnodes_within_dp=nodes,
            master_addr="model-master",
            data_parallel_master_ip="dp-master",
        )
    )

    assert mooncake.get_mooncake_bootstrap_addr(config) == (expected_host, 9876)


def test_receive_kv_selects_matching_remote_pp_workers(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    worker.pp_size = 2
    worker.pp_rank = 1
    worker._tp_size = {"engine": 2}
    worker._remote_agents = {
        "engine": {
            0: {0: "tp0-pp0", 1: "tp0-pp1"},
            1: {0: "tp1-pp0", 1: "tp1-pp1"},
        }
    }
    worker.transfer_topo.handshake_target_ranks = lambda remote_tp_size: [0, 1]
    calls = []

    async def receive(worker_addr, pull_metas):
        calls.append((worker_addr, pull_metas))

    worker.receive_kv_from_single_worker = receive
    pull_meta = SimpleNamespace(pull_tasks_count=0)
    pull_metas = {"request": pull_meta}

    async def run():
        worker.receive_kv("engine", pull_metas)
        await asyncio.sleep(0)

    asyncio.run(run())

    assert [address for address, _ in calls] == ["tp0-pp1", "tp1-pp1"]
    assert all(metadata is pull_metas for _, metadata in calls)
    assert pull_meta.pull_tasks_count == 2


def test_target_first_patch_contract_is_idempotent(
    mooncake,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delattr(
        mooncake,
        patch_mooncake_connector._MARKER,
        raising=False,
    )
    first = patch_mooncake_connector.apply_to_module(mooncake)
    second = patch_mooncake_connector.apply_to_module(mooncake)
    assert first is True
    assert second is False


def test_ttft_transfer_identity(mooncake):
    req_id = "cmpl-12345678-1234-1234-1234-123456789abc-extra"
    assert (
        mooncake.transfer_id_from_req(req_id)
        == "xfer-12345678-1234-1234-1234-123456789abc"
    )
    assert mooncake.transfer_id_from_req(req_id, {"transfer_id": "explicit"}) == (
        "explicit"
    )


@pytest.mark.parametrize("partition", ["41,37", " 41, 37 "])
def test_mooncake_custom_pp_partition_and_stage_ranges(mooncake, monkeypatch, partition):
    monkeypatch.setattr(mooncake.envs, "VLLM_PP_LAYER_PARTITION", partition)
    assert mooncake._normalize_pp_layer_partition(partition, 78, 2) == (41, 37)
    assert mooncake._get_pp_stage_layer_range(78, 0, 2) == (0, 41)
    assert mooncake._get_pp_stage_layer_range(78, 1, 2) == (41, 78)


@pytest.mark.parametrize("partition", [
    "", "41", "41,37,0", "41,,37", "41,36", "-1,79", "0,78",
    "41.0,37", "+41,37", "True,37",
])
def test_mooncake_invalid_pp_partition_is_rejected(mooncake, monkeypatch, partition):
    monkeypatch.setattr(mooncake.envs, "VLLM_PP_LAYER_PARTITION", partition)
    with pytest.raises(ValueError, match="partition"):
        mooncake._get_pp_stage_layer_range(78, 0, 2)


@pytest.mark.parametrize("layers,rank,size", [
    (0, 0, 1), (78, -1, 2), (78, 2, 2), (78, 0, 0),
    (1, 0, 2), (78, True, 2), (78.0, 0, 2),
])
def test_mooncake_invalid_pp_range_is_rejected(mooncake, monkeypatch, layers, rank, size):
    monkeypatch.setattr(mooncake.envs, "VLLM_PP_LAYER_PARTITION", None)
    with pytest.raises(ValueError, match="PP"):
        mooncake._get_pp_stage_layer_range(layers, rank, size)


def test_mooncake_default_pp_ranges_keep_upstream_distribution(mooncake, monkeypatch):
    monkeypatch.setattr(mooncake.envs, "VLLM_PP_LAYER_PARTITION", None)
    assert mooncake._get_pp_stage_layer_range(78, 0, 1) == (0, 78)
    assert [mooncake._get_pp_stage_layer_range(10, rank, 4) for rank in range(4)] == [
        (0, 2), (2, 5), (5, 8), (8, 10),
    ]


def test_mooncake_pp_registration_rejects_foreign_stage_layer(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    worker._pp_layer_range = (41, 78)
    name = "model.layers.40.self_attn"
    worker._layer_specs[name] = _attention_spec(mooncake)
    worker._layer_group_indices[name] = 0
    with pytest.raises(ValueError, match="stage.*41.*78"):
        worker.register_kv_caches({name: torch.zeros((2, 1, 2, 4))})
    assert worker.engine.calls == []


@pytest.mark.parametrize("local_tp,remote_tp", [(1, 1), (2, 4), (4, 2)])
@pytest.mark.parametrize("local_payload,remote_payload", [(8, 4), (4, 8)])
def test_mooncake_replicated_regions_reject_mismatched_lengths(
    mooncake, local_tp, remote_tp, local_payload, remote_payload,
):
    local, remote = _single_group_regions(
        mooncake, local_payload=local_payload, remote_payload=remote_payload,
    )
    error = mooncake._validate_asymmetric_region_lengths(
        local, remote,
        _worker(mooncake, blocks_first=False, tp_size=local_tp).transfer_topo,
        remote_tp, True,
    )
    assert error is not None and "replicated" in error


@pytest.mark.parametrize("local_tp,remote_tp,local_payload,remote_payload,replicated", [
    (1, 1, 8, 8, False), (4, 2, 4, 8, False), (2, 4, 8, 4, False),
    (1, 1, 8, 8, True), (4, 2, 8, 8, True), (2, 4, 8, 8, True),
])
def test_mooncake_valid_asymmetric_region_lengths(
    mooncake, local_tp, remote_tp, local_payload, remote_payload, replicated,
):
    local, remote = _single_group_regions(
        mooncake, local_payload=local_payload, remote_payload=remote_payload,
    )
    assert mooncake._validate_asymmetric_region_lengths(
        local, remote,
        _worker(mooncake, blocks_first=False, tp_size=local_tp).transfer_topo,
        remote_tp, replicated,
    ) is None


@pytest.mark.parametrize("role,expected", [
    ("kv_producer", True), ("kv_both", True), ("kv_consumer", False), (None, False),
])
def test_mooncake_pp1_bootstrap_preserves_sending_roles(mooncake, monkeypatch, role, expected):
    monkeypatch.setattr(mooncake, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mooncake, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role=role),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, local_engines_only=False,
            data_parallel_index=0,
        ),
    )
    assert mooncake.should_launch_bootstrap_server(config) is expected


@pytest.mark.parametrize("producer_tp,consumer_tp,use_mla", [
    (1, 2, False), (2, 1, False), (8, 1, True), (1, 8, True),
])
@pytest.mark.parametrize("padded", [False, True])
def test_v028_allocated_pages_transfer_exact_bytes(
    mooncake, producer_tp, consumer_tp, use_mla, padded,
):
    """Exercise allocator views, real topology, and transfer address planning.

    Copy planned CPU byte spans to check both K/V content, MLA replication,
    nonconsecutive block IDs, and page padding without using real RDMA.
    """
    import ctypes
    from vllm.v1.kv_cache_interface import KVCacheTensor, create_kv_cache_views
    from vllm.v1.kv_cache_layout import KVCacheLayout

    name = "model.layers.0.self_attn"
    global_heads = 1 if use_mla else 4
    content = 3 if use_mla else 5  # Unequal K/V widths catch accidental half-splits.
    expected = torch.arange(3 * global_heads * 2 * content, dtype=torch.int16).reshape(
        3, global_heads, 2, content,
    ).to(torch.float16)

    def make(rank, tp_size, producer):
        worker = _worker(mooncake, blocks_first=False, use_mla=use_mla)
        worker.tp_rank, worker.tp_size = rank, tp_size
        worker.transfer_topo = mooncake.TransferTopology(
            tp_rank=rank, tp_size=tp_size, block_size=2,
            engine_id=f"{'p' if producer else 'd'}-{rank}",
            is_mla=use_mla, is_mamba=False, total_num_kv_heads=global_heads,
            attn_backends=[],
        )
        heads = 1 if use_mla else global_heads // tp_size
        if use_mla:
            spec = mooncake.MLAAttentionSpec(
                block_size=2, num_kv_heads=heads, head_size=content,
                dtype=torch.float16, alignment=32 if padded else None,
            )
        else:
            spec = _attention_spec(
                mooncake, heads=heads, head_size=2, head_size_v=3,
                page_size_padded=(24 * heads if padded else None),
            )
        # Use v0.28's allocator view builder, including padded MLA pages.
        raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
        placement = KVCacheTensor(
            size=raw.numel(), layers=[name],
            layer_stride=raw.numel(), block_stride=spec.page_size_bytes,
        )
        cache = create_kv_cache_views(
            raw, spec, 3, KVCacheLayout.LBHNC, placement,
        )[0]
        if producer:
            cache.copy_(expected if use_mla else expected[:, rank * heads:(rank + 1) * heads])
        worker._layer_specs[name] = spec
        worker._layer_group_indices[name] = 0
        worker.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])
        worker.register_kv_caches({name: cache})
        regions = worker._get_transfer_regions(
            worker.kv_caches_base_addr, worker.block_len_per_layer,
            worker.kv_block_len_per_layer, worker.registered_layer_names,
            worker.registered_layer_indices, worker.registered_group_indices,
            num_blocks_per_layer=worker.num_blocks_per_layer,
        )
        assert regions[0].num_blocks == 3
        assert len(regions) == 1
        return worker, cache, regions

    producers = [make(rank, producer_tp, True) for rank in range(producer_tp)]
    consumers = [make(rank, consumer_tp, False) for rank in range(consumer_tp)]
    for producer, _, local_regions in producers:
        for remote_rank in producer.transfer_topo.handshake_target_ranks(consumer_tp):
            consumer, _, remote_regions = consumers[remote_rank]
            assert mooncake._validate_asymmetric_region_lengths(
                local_regions, remote_regions, producer.transfer_topo, consumer_tp, use_mla,
            ) is None
            metadata = _metadata(
                mooncake, remote_tp_rank=remote_rank, remote_tp_size=consumer_tp,
                req_blocks={"d": ("x", [[1, 2]])},
                num_blocks_per_layer=consumer.num_blocks_per_layer,
            )
            send = mooncake.SendBlockMeta(
                p_req_id="p", transfer_id="x", local_block_ids=[[2, 0]],
                ready=asyncio.Event(),
            )
            src, dst, lengths, errors, message = asyncio.run(
                producer._build_transfer_params(
                    [("d", send)], metadata, local_regions, remote_regions,
                )
            )
            assert errors == [] and message is None
            for src_ptr, dst_ptr, length in zip(src, dst, lengths):
                ctypes.memmove(dst_ptr, src_ptr, length)
    for rank, (_, cache, _) in enumerate(consumers):
        heads = 1 if use_mla else global_heads // consumer_tp
        wanted = expected[[2, 0]] if use_mla else expected[[2, 0], rank * heads:(rank + 1) * heads]
        torch.testing.assert_close(cache[[1, 2]], wanted)
        assert torch.count_nonzero(cache[0]) == 0


def test_registration_rejects_token_major_multi_head_pages(mooncake):
    worker = _worker(mooncake, blocks_first=False)
    name = "model.layers.0.self_attn"
    worker._layer_specs[name] = _attention_spec(mooncake, heads=2)
    worker._layer_group_indices[name] = 0
    # Logical [B,H,N,C], physical [B,N,H,C]. Page is dense, heads aren't.
    cache = torch.zeros((2, 2, 2, 4), dtype=torch.float16).transpose(1, 2)
    with pytest.raises(ValueError, match="canonical contiguous"):
        worker.register_kv_caches({name: cache})
    assert worker.engine.calls == []


def test_worker_rejects_dcp_before_creating_transfer_engine(mooncake, monkeypatch):
    monkeypatch.setattr(mooncake, "TransferEngine", lambda: pytest.fail("engine was created"))
    config = SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=2))
    with pytest.raises(NotImplementedError, match="decode_context_parallel_size=1"):
        mooncake.MooncakeConnectorWorker(config, "engine", None)


def test_registration_rejects_dense_transposed_page_contents(mooncake):
    worker = _worker(mooncake, blocks_first=False)
    name = "model.layers.0.self_attn"
    worker._layer_specs[name] = _attention_spec(mooncake)
    worker._layer_group_indices[name] = 0
    cache = torch.zeros((2, 1, 4, 2), dtype=torch.float16).transpose(2, 3)
    with pytest.raises(ValueError, match="canonical contiguous"):
        worker.register_kv_caches({name: cache})
    assert worker.engine.calls == []


def test_successful_pull_still_completes_and_error_remains_fatal(mooncake):
    worker = _worker(mooncake, blocks_first=False)
    meta = mooncake.PullReqMeta(
        d_req_id="d", transfer_id="x", local_block_ids=[[0]],
        remote_engine_id="engine", remote_bootstrap_addr="bootstrap",
        pull_tasks_count=1,
    )
    worker.process_pulling_result(
        mooncake.MooncakeXferResponse(
            status=mooncake.MooncakeXferResponseStatus.FINISH, ok_reqs=["d"],
        ), {"d": meta},
    )
    assert asyncio.run(worker.fetch_finished_recving_reqs()) == {"d"}
    assert asyncio.run(worker.fetch_finished_recving_reqs()) == set()
    worker._fail_pull_metas({"d": meta}, "connection lost")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="connection lost"):
            asyncio.run(worker.fetch_finished_recving_reqs())


@pytest.mark.parametrize("side", ["producer", "consumer"])
@pytest.mark.parametrize("kernel_ratio,block_id", [(1, 2), (1, 99), (2, 1)])
def test_transfer_rejects_block_ids_at_or_above_capacity(
    mooncake, side, kernel_ratio, block_id,
):
    from dataclasses import replace
    worker = _worker(mooncake, blocks_first=False)
    worker._physical_blocks_per_logical_kv_block = kernel_ratio
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
    )
    local, remote = _single_group_regions(mooncake)
    local = [replace(local[0], num_blocks=2)]
    remote = [replace(remote[0], num_blocks=2)]
    send = mooncake.SendBlockMeta(
        p_req_id="p", transfer_id="x",
        local_block_ids=[[block_id if side == "producer" else 0]],
        ready=asyncio.Event(),
    )
    metadata = _metadata(mooncake, req_blocks={
        "d": ("x", [[block_id if side == "consumer" else 0]]),
    })
    src, dst, lengths, errors, message = asyncio.run(worker._build_transfer_params(
        [("d", send)], metadata, local, remote,
    ))
    assert (src, dst, lengths) == ([], [], [])
    assert errors == ["d"] and side in message and "capacity" in message


@pytest.mark.parametrize("counts", [[], [0], [-1], [True], [2, 3]])
def test_region_capacities_reject_missing_or_invalid_counts(mooncake, counts):
    with pytest.raises(ValueError, match="positive block capacity"):
        mooncake._expand_transfer_regions(
            base_addrs=[100], block_lens=[16], kv_block_lens=[8],
            layer_names=["model.layers.0.self_attn"], layer_indices=[0],
            is_kv_layout_blocks_first=False, num_blocks_per_layer=counts,
        )


def test_capacity_error_does_not_emit_partial_request_descriptors(mooncake):
    from dataclasses import replace
    worker = _worker(mooncake, blocks_first=False)
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
    )
    local, remote = _single_group_regions(mooncake)
    local = [replace(local[0], num_blocks=2), replace(local[0], num_blocks=1)]
    remote = [replace(remote[0], num_blocks=2)] * 2
    send = mooncake.SendBlockMeta(
        p_req_id="p", transfer_id="x", local_block_ids=[[1]],
        ready=asyncio.Event(),
    )
    metadata = _metadata(mooncake, req_blocks={"d": ("x", [[1]])})
    src, dst, lengths, errors, message = asyncio.run(worker._build_transfer_params(
        [("d", send)], metadata, local, remote,
    ))
    assert (src, dst, lengths) == ([], [], [])
    assert errors == ["d"] and "capacity" in message


@pytest.mark.parametrize("method", ["mtp", "dspark", None])
@pytest.mark.parametrize("pp_rank,layer_index,with_spec,without_spec", [
    (0, 40, True, True), (0, 41, False, False),
    (0, 78, False, False), (0, 79, False, False),
    (1, 40, False, False), (1, 41, True, True), (1, 77, True, True),
    (1, 78, True, False), (1, 79, True, False),
])
def test_pp_registration_checks_backbone_and_local_draft_ownership(
    mooncake, method, pp_rank, layer_index, with_spec, without_spec,
):
    worker = _worker(mooncake, blocks_first=False)
    worker.pp_rank = pp_rank
    worker._pp_layer_range = [(0, 41), (41, 78)][pp_rank]
    worker.vllm_config.speculative_config = (
        SimpleNamespace(method=method) if method is not None else None
    )
    name = f"model.layers.{layer_index}.self_attn"
    # Only caches declared in this worker's KVCacheConfig can be registered.
    worker._layer_specs[name] = _attention_spec(mooncake)
    worker._layer_group_indices[name] = 0
    cache = torch.zeros((2, 1, 2, 4), dtype=torch.float16)
    owned = with_spec if method is not None else without_spec
    if not owned:
        with pytest.raises(ValueError, match="outside PP stage"):
            worker.register_kv_caches({name: cache})
        assert worker.engine.calls == []
    else:
        worker.register_kv_caches({name: cache})
        assert worker.registered_layer_names == [name]
        assert worker.registered_layer_indices == [layer_index]
        assert len(worker.engine.calls) == 1


def test_final_pp_stage_registers_backbone_and_only_declared_draft_cache(mooncake):
    worker = _worker(mooncake, blocks_first=False)
    worker.pp_rank = 1
    worker._pp_layer_range = (41, 78)
    worker.vllm_config.speculative_config = SimpleNamespace(method="mtp")
    names = [f"model.layers.{i}.self_attn" for i in (77, 78)]
    for name in names:
        worker._layer_specs[name] = _attention_spec(mooncake)
        worker._layer_group_indices[name] = 0
    cache = torch.zeros((2, 1, 2, 4), dtype=torch.float16)
    caches = dict.fromkeys(names, cache)
    caches["model.layers.79.self_attn"] = cache  # No local spec: skip this cache.
    worker.register_kv_caches(caches)
    assert worker.registered_layer_names == names
    assert worker.registered_layer_indices == [77, 78]
    assert len(worker.engine.calls) == 1
