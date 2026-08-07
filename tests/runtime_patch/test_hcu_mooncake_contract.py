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
):
    worker = object.__new__(mooncake.MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker.use_mla = use_mla
    worker.transfer_topo = SimpleNamespace(
        virtually_split_kv_in_blocks=blocks_first,
        is_kv_layout_blocks_first=blocks_first,
        split_k_and_v=split_k_and_v,
        local_replicates_kv_cache=False,
        get_transfer_cache_regions=lambda cache, spec: (
            cache if split_k_and_v else [cache]
        ),
    )
    worker.engine = _RegistrationEngine()
    worker.is_kv_consumer = True
    worker._layer_specs = {}
    worker._layer_group_indices = {}
    worker._physical_blocks_per_logical_kv_block = 1
    worker.tp_rank = 0
    worker.tp_size = 2
    worker.finished_recving_reqs = set()
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
    )
    metadata = _metadata(
        mooncake,
        req_blocks={"r": ("x", [[3, 1]])},
        kv_caches_base_addr=[100],
        block_lens=[64],
        kv_block_lens=[24],
        registered_layer_names=["model.layers.0.self_attn"],
        registered_layer_indices=[0],
        registered_group_indices=[0],
    )
    encoded = msgspec.msgpack.encode(metadata)
    decoded = msgspec.msgpack.decode(encoded, type=mooncake.MooncakeXferMetadata)
    assert decoded == metadata


def test_hcu_mooncake_uses_nhd_until_hnd_backend_support(mooncake):
    config = SimpleNamespace(model_config=SimpleNamespace(use_mla=False))
    assert mooncake.MooncakeConnector.get_required_kvcache_layout(config) == "NHD"

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
    with pytest.raises(ValueError, match="positive number"):
        mooncake._validate_phase1_kv_cache_config(valid, 0)
    with pytest.raises(NotImplementedError, match="Mamba/GDN"):
        mooncake._validate_phase1_kv_cache_config(
            SimpleNamespace(has_mamba_layers=True, kv_cache_groups=[object()]), 1
        )
    with pytest.raises(NotImplementedError, match="exactly one"):
        mooncake._validate_phase1_kv_cache_config(
            SimpleNamespace(has_mamba_layers=False, kv_cache_groups=[1, 2]), 1
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
        cache_config=SimpleNamespace(block_size=64),
        model_config=SimpleNamespace(use_mla=False),
    )
    unsupported = SimpleNamespace(
        has_mamba_layers=False,
        kv_cache_groups=[object(), object()],
    )

    with pytest.raises(NotImplementedError, match="exactly one"):
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


def test_registration_preserves_padded_stride_and_deduplicates_storage(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    raw = torch.zeros(128, dtype=torch.float16)
    first = torch.as_strided(
        raw,
        size=(2, 2, 1, 2, 2),
        stride=(16, 4, 4, 2, 1),
        storage_offset=0,
    )
    second = torch.as_strided(
        raw,
        size=(2, 2, 1, 2, 2),
        stride=(16, 4, 4, 2, 1),
        storage_offset=40,
    )
    caches = {
        "model.layers.0.self_attn": first,
        "model.layers.1.self_attn": second,
    }
    for name in caches:
        worker._layer_specs[name] = object()
        worker._layer_group_indices[name] = 0

    worker.register_kv_caches(caches)

    assert worker.block_len_per_layer == [32, 32]
    assert worker.kv_block_len_per_layer == [8, 8]
    assert worker.registered_layer_indices == [0, 1]
    assert worker.kv_caches_base_addr == [first.data_ptr(), second.data_ptr()]
    assert worker.engine.calls == [
        ([raw.untyped_storage().data_ptr()], [raw.untyped_storage().nbytes()])
    ]


def test_registration_supports_split_tuple_and_mla(mooncake, monkeypatch):
    split = _worker(mooncake, blocks_first=False, split_k_and_v=True)
    split_name = "model.layers.0.self_attn"
    split._layer_specs[split_name] = object()
    split._layer_group_indices[split_name] = 0
    key = torch.zeros((2, 1, 2, 2), dtype=torch.float16)
    value = torch.zeros_like(key)
    split.register_kv_caches({split_name: (key, value)})
    assert split.kv_caches_base_addr == [key.data_ptr(), value.data_ptr()]
    assert split.kv_block_len_per_layer == [8, 8]

    class _FakeMLASpec:
        page_size_bytes = 12

    monkeypatch.setattr(mooncake, "MLAAttentionSpec", _FakeMLASpec)
    monkeypatch.setattr(mooncake, "SlidingWindowMLASpec", _FakeMLASpec)
    mla = _worker(mooncake, blocks_first=False, use_mla=True)
    mla_name = "model.layers.0.self_attn"
    mla._layer_specs[mla_name] = _FakeMLASpec()
    mla._layer_group_indices[mla_name] = 0
    cache = torch.zeros((2, 2, 3), dtype=torch.float16)
    mla.register_kv_caches({mla_name: cache})
    assert mla.kv_block_len_per_layer == [12]


def test_registration_rejects_view_outside_storage(mooncake, monkeypatch):
    worker = _worker(mooncake, blocks_first=True)
    name = "model.layers.0.self_attn"
    worker._layer_specs[name] = object()
    worker._layer_group_indices[name] = 0
    cache = torch.zeros((2, 2, 1, 2, 2), dtype=torch.float16)
    real_storage = cache.untyped_storage()
    monkeypatch.setattr(
        torch.Tensor,
        "untyped_storage",
        lambda self: SimpleNamespace(
            data_ptr=real_storage.data_ptr,
            nbytes=lambda: 1,
        ),
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


def test_failed_pull_result_completes_the_receive_task(mooncake):
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

    assert pull_meta.pull_tasks_count == 0
    assert worker.finished_recving_reqs == {"d"}


def test_bootstrap_failure_completes_pull_without_started_tasks(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    pull_meta = mooncake.PullReqMeta(
        d_req_id="d",
        transfer_id="x",
        local_block_ids=[[0]],
        remote_engine_id="missing",
        remote_bootstrap_addr="bootstrap",
    )

    worker._fail_pull_metas({"d": pull_meta}, "engine not found")

    assert pull_meta.pull_tasks_count == 0
    assert worker.finished_recving_reqs == {"d"}


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


def test_asymmetric_tp_fails_closed_without_hnd_backend_support(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
    )
    local = mooncake._expand_transfer_regions(
        base_addrs=[1000],
        block_lens=[32],
        kv_block_lens=[8],
        layer_names=["model.layers.0.self_attn"],
        layer_indices=[0],
        group_indices=[0],
        is_kv_layout_blocks_first=True,
    )
    remote = mooncake._expand_transfer_regions(
        base_addrs=[2000],
        block_lens=[32],
        kv_block_lens=[4],
        layer_names=["model.layers.0.self_attn"],
        layer_indices=[0],
        group_indices=[0],
        is_kv_layout_blocks_first=True,
    )
    send_meta = mooncake.SendBlockMeta(
        p_req_id="p", transfer_id="x", local_block_ids=[[0]], ready=asyncio.Event()
    )
    metadata = _metadata(
        mooncake, remote_tp_size=4, req_blocks={"d": ("x", [[0]])}
    )
    src, dst, lengths, errors, message = asyncio.run(
        worker._build_transfer_params([("d", send_meta)], metadata, local, remote)
    )
    assert (src, dst, lengths) == ([], [], [])
    assert errors == ["d"]
    assert message is not None and "asymmetric TP transfer is disabled" in message


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
        parallel_config=SimpleNamespace(
            local_engines_only=local_only,
            data_parallel_rank_local=dp_local,
            data_parallel_index=dp_index,
        )
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
