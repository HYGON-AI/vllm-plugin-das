# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

import asyncio
import importlib
import os
import threading
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
    worker._pp_layer_range = None
    worker.pcp_size = 1
    worker.pcp_rank = 0
    worker.dsa_layer_split_enabled = False
    worker._pcp_dummy_sender = False
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


def test_cutlass_mooncake_uses_target_hnd_layout(mooncake):
    """Cover HND only for the supported CUTLASS deployment contract."""
    config = SimpleNamespace(model_config=SimpleNamespace(use_mla=False))
    assert mooncake.MooncakeConnector.get_required_kvcache_layout(config) == "HND"

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
    worker = _worker(mooncake, blocks_first=True)
    worker.tp_rank = local_tp_rank
    worker.tp_size = local_tp_size
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
    worker._layer_specs[name] = object()
    worker._layer_group_indices[name] = 0
    with pytest.raises(ValueError, match="stage.*41.*78"):
        worker.register_kv_caches({name: torch.zeros((2, 2, 1, 2, 2))})
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
        local, remote, local_tp, remote_tp, True,
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
        local, remote, local_tp, remote_tp, replicated,
    ) is None


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("callable_flag", [True, False])
def test_mooncake_dense_flag_accepts_boolean_properties_and_methods(
    mooncake, flag, callable_flag,
):
    tensor = SimpleNamespace(
        is_non_overlapping_and_dense=(lambda: flag) if callable_flag else flag,
    )
    assert mooncake._get_tensor_dense_flag(tensor) is flag


@pytest.mark.parametrize("flag", [1, "false", lambda: "true"])
def test_mooncake_dense_flag_rejects_truthy_non_booleans(mooncake, flag):
    with pytest.raises(ValueError, match="dense.*bool"):
        mooncake._get_tensor_dense_flag(SimpleNamespace(is_non_overlapping_and_dense=flag))


def test_mooncake_dense_flag_missing_is_unknown(mooncake):
    assert mooncake._get_tensor_dense_flag(SimpleNamespace()) is None


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


def test_pcp_disabled_does_not_query_pcp_group(mooncake, monkeypatch):
    def _unused_group():
        raise AssertionError("PCP group is not consulted when pcp_size <= 1")

    monkeypatch.setattr(mooncake, "get_pcp_group", _unused_group)
    size, rank = mooncake._pcp_kv_replica_coord(
        SimpleNamespace(prefill_context_parallel_size=1)
    )
    worker = object.__new__(mooncake.MooncakeConnectorWorker)
    worker.pcp_size = size
    worker.pcp_rank = rank

    assert (size, rank) == (1, 0)
    assert worker._is_canonical_pcp_kv_replica()


def test_pcp_tp_registers_one_sender_per_tp_group_and_pairs_decode(
    mooncake, monkeypatch
):
    """HYV4 producer PCP=2 TP=4 keeps four senders and one bootstrap server.

    PCP groups are [0, 4], [1, 5], [2, 6], [3, 7]. Global rank 0 is the only
    bootstrap owner. Each group's rank_in_group 0 (global ranks 0-3) must
    still register, or Decode infers producer TP size 1 and rejects D ranks
    1-3.
    """

    from vllm.distributed import parallel_state

    tp_size = 4
    pcp_size = 2
    world = tp_size * pcp_size
    group = object.__new__(parallel_state.GroupCoordinator)
    monkeypatch.setattr(mooncake, "get_pcp_group", lambda: group)
    monkeypatch.setattr(
        mooncake, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            local_engines_only=False,
            data_parallel_index=0,
        ),
    )
    senders: dict[int, int] = {}
    bootstrap_owners: list[int] = []
    for global_rank in range(world):
        tp_rank = global_rank % tp_size
        members = list(range(tp_rank, world, tp_size))
        group.rank = global_rank
        group.ranks = members
        group.rank_in_group = members.index(global_rank)
        monkeypatch.setattr(
            mooncake,
            "get_tensor_model_parallel_rank",
            lambda tp_rank=tp_rank: tp_rank,
        )
        size, pcp_rank = mooncake._pcp_kv_replica_coord(config.parallel_config)
        worker = object.__new__(mooncake.MooncakeConnectorWorker)
        worker.pcp_size = size
        worker.pcp_rank = pcp_rank
        if worker._is_canonical_pcp_kv_replica():
            senders[tp_rank] = global_rank
        if mooncake.should_launch_bootstrap_server(config):
            bootstrap_owners.append(global_rank)

    assert senders == {0: 0, 1: 1, 2: 2, 3: 3}
    assert bootstrap_owners == [0]

    registry = {
        "0": {
            "engine_id": "prefill",
            "worker_addr": {
                str(tp_rank): {"0": f"tcp://prefill:{9000 + tp_rank}"}
                for tp_rank in senders
            },
        }
    }

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return registry

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def get(self, url):
            assert url.endswith("/query")
            return _Response()

    monkeypatch.setattr(mooncake.httpx, "AsyncClient", lambda *_a, **_k: _Client())
    decoder = object.__new__(mooncake.MooncakeConnectorWorker)
    decoder._remote_agents = {}
    decoder._tp_size = {}
    decoder._pending_bootstrap_queries = {"http://prefill:1": asyncio.Event()}
    asyncio.run(decoder._connect_to_prefiller_bootstrap("http://prefill:1"))

    assert decoder._tp_size["prefill"] == tp_size
    assert sorted(decoder._remote_agents["prefill"]) == [0, 1, 2, 3]

    class _KvShapeBackend:
        """Satisfy vLLM 0.25.1 TransferTopology.__post_init__ attn_backends[0]."""

        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
            return (2, num_blocks, num_kv_heads, block_size, head_size)

    def _topology(tp_rank: int):
        return mooncake.TransferTopology(
            tp_rank=tp_rank,
            tp_size=tp_size,
            block_size=16,
            engine_id="engine",
            is_mla=True,
            is_mamba=False,
            total_num_kv_heads=tp_size,
            attn_backends=[_KvShapeBackend],
        )

    for decode_rank in range(tp_size):
        targets = _topology(decode_rank).handshake_target_ranks(
            decoder._tp_size["prefill"]
        )
        assert targets == [decode_rank]
        accepted = _topology(targets[0]).handshake_target_ranks(tp_size)
        assert decode_rank in accepted


def test_pcp_tp_multi_sender_holds_blocks_until_every_sender_finishes(
    mooncake, monkeypatch
):
    """PCP=2 TP=4 keeps four live senders and does not free blocks early.

    Replica ranks report completion without RDMA. The scheduler aggregator
    still waits until each TP group's PCP-local rank 0 finishes its send.
    """

    from vllm.distributed import parallel_state
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

    tp_size = 4
    pcp_size = 2
    world = tp_size * pcp_size
    req_id = "prefill-1"
    transfer_id = "xfer-pcp"
    group = object.__new__(parallel_state.GroupCoordinator)
    monkeypatch.setattr(mooncake, "get_pcp_group", lambda: group)

    scheduler = object.__new__(mooncake.MooncakeConnectorScheduler)
    scheduler.is_kv_producer = True
    scheduler.is_kv_consumer = False
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_send = {}
    scheduler._reqs_not_processed = set()
    scheduler._is_hma_required = False
    scheduler._has_mamba = False
    scheduler.blocks_per_sw = []
    request = SimpleNamespace(
        request_id=req_id,
        kv_transfer_params={"do_remote_decode": True, "transfer_id": transfer_id},
        status=mooncake.RequestStatus.FINISHED_LENGTH_CAPPED,
    )
    delay_free, _params = scheduler.request_finished(request, ([11, 12],))
    metadata = scheduler.build_connector_meta(SimpleNamespace())
    assert delay_free is True
    assert metadata.reqs_to_send[req_id] == (transfer_id, [[11, 12]])

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    scheduled: list[asyncio.Future] = []
    real_submit = asyncio.run_coroutine_threadsafe

    def _submit(coro, target_loop):
        future = real_submit(coro, target_loop)
        scheduled.append(future)
        return future

    monkeypatch.setattr(mooncake.asyncio, "run_coroutine_threadsafe", _submit)
    workers = []
    try:
        for global_rank in range(world):
            tp_rank = global_rank % tp_size
            members = list(range(tp_rank, world, tp_size))
            group.rank = global_rank
            group.ranks = members
            group.rank_in_group = members.index(global_rank)
            size, pcp_rank = mooncake._pcp_kv_replica_coord(
                SimpleNamespace(prefill_context_parallel_size=pcp_size)
            )
            worker = object.__new__(mooncake.MooncakeConnectorWorker)
            worker.sender_loop = loop
            worker.is_kv_producer = True
            worker.is_kv_consumer = False
            worker.tp_rank = tp_rank
            worker.tp_size = tp_size
            worker.pcp_size = size
            worker.pcp_rank = pcp_rank
            worker.finished_sending_reqs = set()
            worker.finished_recving_reqs = set()
            worker.reqs_need_send = {}
            worker.registered_layer_names = []
            worker.registered_layer_indices = []
            worker.registered_group_indices = []
            worker.kv_caches_base_addr = []
            worker.block_len_per_layer = []
            worker.kv_block_len_per_layer = []
            worker._encoder = msgspec.msgpack.Encoder()
            worker._get_transfer_regions = lambda *_args, **_kwargs: []
            worker.transfer_topo = SimpleNamespace(
                local_replicates_kv_cache=False,
                handshake_target_ranks=lambda remote_tp_size, rank=tp_rank: [rank]
                if remote_tp_size == tp_size
                else [],
            )
            if worker._is_canonical_pcp_kv_replica():
                worker.reqs_need_send[transfer_id] = mooncake.SendBlockMeta(
                    p_req_id="",
                    transfer_id=transfer_id,
                    local_block_ids=[],
                    ready=asyncio.Event(),
                )
            before = len(scheduled)
            worker.start_load_kv(metadata)
            scheduled[before].result(timeout=5)
            workers.append(worker)

        canonical = [worker for worker in workers if worker._is_canonical_pcp_kv_replica()]
        replicas = [worker for worker in workers if worker not in canonical]
        assert [worker.tp_rank for worker in canonical] == [0, 1, 2, 3]
        assert all(worker.pcp_rank == 0 for worker in canonical)
        assert all(worker.finished_sending_reqs == set() for worker in canonical)
        assert all(
            worker.reqs_need_send[transfer_id].ready.is_set()
            and worker.reqs_need_send[transfer_id].local_block_ids == [[11, 12]]
            for worker in canonical
        )
        assert all(worker.finished_sending_reqs == {req_id} for worker in replicas)
        assert all(transfer_id not in worker.reqs_need_send for worker in replicas)

        def _runner(finished: set[str] | None) -> ModelRunnerOutput:
            return ModelRunnerOutput(
                req_ids=[req_id],
                req_id_to_index={req_id: 0},
                kv_connector_output=KVConnectorOutput(finished_sending=finished),
            )

        aggregator = KVOutputAggregator(world)
        held = aggregator.aggregate(
            [
                _runner(set(worker.finished_sending_reqs) or None)
                for worker in workers
            ]
        )
        assert held is not None
        assert not held.kv_connector_output.finished_sending

        transfer_calls: list[int] = []

        async def _stand_in_transfer(
            ready_reqs,
            _agent_meta,
            _local_regions,
            _remote_regions,
            tp_rank,
        ):
            assert ready_reqs[0][1].local_block_ids == [[11, 12]]
            transfer_calls.append(tp_rank)
            return [], [], [], [], None

        class _Sock:
            def __init__(self):
                self.sent: list[tuple] = []

            async def send_multipart(self, parts):
                self.sent.append(parts)

        def _transfer_for(tp_rank: int):
            async def _run(ready_reqs, agent_meta, local_regions, remote_regions):
                return await _stand_in_transfer(
                    ready_reqs,
                    agent_meta,
                    local_regions,
                    remote_regions,
                    tp_rank,
                )

            return _run

        for worker in canonical:
            worker._build_transfer_params = _transfer_for(worker.tp_rank)
            sock = _Sock()
            xfer = _metadata(
                mooncake,
                remote_tp_size=tp_size,
                remote_tp_rank=worker.tp_rank,
                req_blocks={"decode-1": (transfer_id, [[21, 22]])},
            )
            real_submit(
                worker.send_kv_to_decode(b"decode", sock, xfer), loop
            ).result(timeout=5)
            response = msgspec.msgpack.decode(
                sock.sent[-1][1], type=mooncake.MooncakeXferResponse
            )
            assert response.status == mooncake.MooncakeXferResponseStatus.FINISH
            assert response.err_reqs is None
            finished = real_submit(
                worker.fetch_finished_sending_reqs(), loop
            ).result(timeout=5)
            assert finished == {req_id}

        assert transfer_calls == [0, 1, 2, 3]
        released = aggregator.aggregate(
            [_runner({req_id}) for _worker in canonical]
            + [_runner(None) for _worker in replicas]
        )
        assert released.kv_connector_output.finished_sending == {req_id}
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


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


@pytest.fixture
def constructed_mooncake_workers(mooncake, monkeypatch):
    """Run the real constructor; replace device/network/thread startup only."""
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed import parallel_state

    group = object.__new__(parallel_state.GroupCoordinator)
    group.rank = 0
    group.ranks = [0, 1, 2, 3]
    group.rank_in_group = 0
    monkeypatch.setattr(parallel_state, "_PCP", group)
    pp_group = SimpleNamespace(rank_in_group=0)
    monkeypatch.setattr(mooncake, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(mooncake, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mooncake, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(mooncake, "current_platform", SimpleNamespace(set_device=lambda _: None))
    monkeypatch.setattr(mooncake, "get_ip", lambda: "127.0.0.1")
    backend = SimpleNamespace(
        get_name=lambda: "TEST_MLA", get_kv_cache_shape=lambda **_: (1, 16, 1),
    )
    monkeypatch.setattr(mooncake, "get_current_attn_backends", lambda _: [backend])
    monkeypatch.setattr(mooncake, "select_common_block_size", lambda size, _: size)
    monkeypatch.setattr(mooncake, "get_kv_cache_layout", lambda: "HND")
    monkeypatch.setattr(mooncake.threading.Thread, "start", lambda _: None)

    class Engine:
        def initialize(self, *args):
            return 0

        def get_rpc_port(self):
            return 1234

    launches = []

    class Bootstrap:
        def __init__(self, host, port):
            self.rank = group.rank

        def start(self):
            launches.append(self.rank)

        def shutdown(self):
            pass

    monkeypatch.setattr(mooncake, "TransferEngine", Engine)
    monkeypatch.setattr(mooncake, "MooncakeBootstrapServer", Bootstrap)
    monkeypatch.setattr(mooncake, "HCUMooncakeBootstrapServer", Bootstrap)
    workers = []

    def make(*, role="kv_producer", pp_size=1, pp_rank=0, pcp_size=4):
        pp_group.rank_in_group = pp_rank
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                use_mla=True, get_total_num_kv_heads=lambda: 1,
                get_total_num_hidden_layers=lambda: 78,
            ),
            cache_config=SimpleNamespace(block_size=16),
            kv_transfer_config=KVTransferConfig(
                kv_connector="MooncakeConnector", kv_role=role, kv_buffer_device="cpu",
            ),
            parallel_config=SimpleNamespace(
                rank=0, tensor_parallel_size=1, pipeline_parallel_size=pp_size,
                prefill_context_parallel_size=pcp_size, local_engines_only=False,
                data_parallel_index=0, data_parallel_rank_local=0,
                nnodes_within_dp=1, data_parallel_master_ip="127.0.0.1",
            ),
        )
        cache_config = SimpleNamespace(
            has_mamba_layers=False,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=object(), layer_names=[])],
        )
        worker = mooncake.MooncakeConnectorWorker(config, "test-engine", cache_config)
        workers.append(worker)
        return worker

    yield make, group, launches
    for worker in workers:
        worker.shutdown()
        worker.shutdown = lambda: None
        for loop_name in ("sender_loop", "receiver_loop"):
            if hasattr(worker, loop_name):
                getattr(worker, loop_name).close()


def test_hyv4_worker_construction_starts_bootstrap_once(constructed_mooncake_workers):
    make, group, launches = constructed_mooncake_workers
    for role in ("kv_producer", "kv_consumer"):
        for rank in range(4):
            group.rank = rank
            group.rank_in_group = rank
            make(role=role)
    assert launches == [0]


def test_mooncake_worker_constructs_stage_range_from_pp_group(
    mooncake, monkeypatch, constructed_mooncake_workers,
):
    monkeypatch.setattr(mooncake.envs, "VLLM_PP_LAYER_PARTITION", "41,37")
    make, _, _ = constructed_mooncake_workers
    worker = make(role="kv_consumer", pp_size=2, pp_rank=1, pcp_size=1)
    assert worker._pp_layer_range == (41, 78)


# ---- DSA cache layer split over PD transfer (SGLang #29421) ----------------


def test_layer_split_registration_skips_non_owned_layers(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    worker.pcp_size = 4
    worker.pcp_rank = 1
    worker.dsa_layer_split_enabled = True
    worker._dsa_total_layers = 8
    worker._dsa_owned_layer_range = (2, 4)
    caches = {}
    for idx in range(8):
        name = f"model.layers.{idx}.self_attn"
        caches[name] = torch.zeros((2, 2, 1, 2, 2), dtype=torch.float16)
        worker._layer_specs[name] = object()
        worker._layer_group_indices[name] = 0

    worker.register_kv_caches(caches)

    assert worker.registered_layer_indices == [2, 3]


def test_layer_split_last_rank_carries_mtp_layers(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    worker.pcp_size = 4
    worker.pcp_rank = 3
    worker.dsa_layer_split_enabled = True
    worker._dsa_total_layers = 8
    worker._dsa_owned_layer_range = (6, 8)
    caches = {}
    for idx in (5, 6, 7, 8):  # layer 8 is the MTP layer
        name = f"model.layers.{idx}.self_attn"
        caches[name] = torch.zeros((2, 2, 1, 2, 2), dtype=torch.float16)
        worker._layer_specs[name] = object()
        worker._layer_group_indices[name] = 0

    worker.register_kv_caches(caches)

    assert worker.registered_layer_indices == [6, 7, 8]
    # A non-last rank must not transfer the MTP layer.
    assert worker._dsa_owns_layer(8)
    worker.pcp_rank = 0
    assert not worker._dsa_owns_layer(8)


def _receive_kv_calls(mooncake, worker, engine_id):
    calls = []

    async def receive(worker_addr, pull_metas):
        calls.append(worker_addr)

    worker.receive_kv_from_single_worker = receive
    pull_meta = SimpleNamespace(pull_tasks_count=0)

    async def run():
        worker.receive_kv(engine_id, {"request": pull_meta})
        await asyncio.sleep(0)

    asyncio.run(run())
    return calls, pull_meta


def test_receive_kv_pulls_every_pcp_rank_under_layer_split(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    worker.pp_size = 1
    worker.pp_rank = 0
    worker._tp_size = {"engine": 1}
    worker._remote_pcp_info = {"engine": (4, True)}
    worker._remote_agents = {
        "engine": {rank: {0: f"vtp{rank}-pp0"} for rank in range(4)}
    }
    worker.transfer_topo.handshake_target_ranks = lambda remote_tp_size: [0]

    calls, pull_meta = _receive_kv_calls(mooncake, worker, "engine")

    assert calls == ["vtp0-pp0", "vtp1-pp0", "vtp2-pp0", "vtp3-pp0"]
    assert pull_meta.pull_tasks_count == 4


def test_receive_kv_pulls_rank0_only_without_layer_split(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    worker.pp_size = 1
    worker.pp_rank = 0
    worker._tp_size = {"engine": 1}
    worker._remote_pcp_info = {"engine": (4, False)}
    worker._remote_agents = {
        "engine": {rank: {0: f"vtp{rank}-pp0"} for rank in range(4)}
    }
    worker.transfer_topo.handshake_target_ranks = lambda remote_tp_size: [0]

    calls, pull_meta = _receive_kv_calls(mooncake, worker, "engine")

    assert calls == ["vtp0-pp0"]
    assert pull_meta.pull_tasks_count == 1


def test_bootstrap_payload_carries_pcp_topology(mooncake):
    payload = mooncake.HCURegisterWorkerPayload(
        engine_id="e",
        dp_rank=0,
        tp_rank=5,
        pp_rank=0,
        addr="tcp://127.0.0.1:1",
        pcp_size=8,
        enable_dsa_cache_layer_split=True,
    )
    dumped = payload.model_dump()
    assert dumped["pcp_size"] == 8
    assert dumped["enable_dsa_cache_layer_split"] is True
    # Defaults keep non-PCP producers wire-compatible.
    bare = mooncake.HCURegisterWorkerPayload(
        engine_id="e", dp_rank=0, tp_rank=0, pp_rank=0, addr="tcp://127.0.0.1:1",
    )
    assert bare.pcp_size == 1
    assert bare.enable_dsa_cache_layer_split is False


def test_bootstrap_query_recovers_real_tp_size(mooncake):
    worker = _worker(mooncake, blocks_first=True)
    worker._remote_agents = {}
    worker._tp_size = {}
    worker._remote_pcp_info = {}
    worker._pending_bootstrap_queries = {
        "http://p:8998": asyncio.Event(),
    }

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "0": {
                    "engine_id": "engine",
                    "pcp_size": 4,
                    "enable_dsa_cache_layer_split": True,
                    "worker_addr": {
                        str(rank): {"0": f"vtp{rank}"} for rank in range(8)
                    },
                }
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            return _Resp()

    orig_client = mooncake.httpx.AsyncClient
    mooncake.httpx.AsyncClient = _Client
    try:
        asyncio.run(worker._connect_to_prefiller_bootstrap("http://p:8998"))
    finally:
        mooncake.httpx.AsyncClient = orig_client

    assert worker._tp_size["engine"] == 2  # 8 virtual ranks / pcp_size 4
    assert worker._remote_pcp_info["engine"] == (4, True)
    assert set(worker._remote_agents["engine"]) == set(range(8))
