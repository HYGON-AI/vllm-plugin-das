# SPDX-License-Identifier: Apache-2.0
"""Stock Core function bodies on CPU; no RDMA/device/engine initialization."""

from types import SimpleNamespace as NS

import pytest
import torch
from hcu_vllm_das import hy4_pcp_mooncake as owned
from hcu_vllm_das.models.hy4_standard import pd_policy
from vllm.distributed.kv_transfer.kv_connector.v1 import base
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_connector as native,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.request import Request, RequestStatus


def config(*, standard=True, method="dfly", producer=False):
    hf = NS(model_type="hy_v4")
    if standard:
        hf = NS(
            model_type="hy_v4_vl",
            architectures=["HYV4VLForConditionalGeneration"],
            text_config=NS(model_type="hy_v4"),
        )
    return NS(
        model_config=NS(hf_config=hf, enforce_eager=True),
        parallel_config=NS(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=16 if producer else 1,
            data_parallel_size=1 if producer else 16,
            data_parallel_rank=0,
            decode_context_parallel_size=1,
            enable_expert_parallel=True,
            all2all_backend="deepep_high_throughput",
        ),
        speculative_config=NS(method=method),
        cache_config=NS(block_size=64),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=True),
        kv_transfer_config=NS(
            engine_id="cpu",
            kv_role="kv_producer" if producer else "kv_consumer",
            kv_connector="MooncakeConnector",
            kv_connector_module_path=pd_policy.DIRECT_MOONCAKE_MODULE,
        ),
    )


def cache_config():
    return NS(
        kv_cache_groups=[],
        transfer_groups=[],
        has_mamba_layers=False,
        transfer_group_ids=[0],
        select_transfer_block_ids=lambda ids: ids,
    )


def scheduler(cfg):
    return owned.HY4PCPMooncakeScheduler(cfg, "cpu", cache_config())


def request(n, *, producer=False):
    params = {
        "do_remote_prefill": not producer,
        "do_remote_decode": producer,
        "remote_engine_id": "p-engine",
        "remote_bootstrap_addr": "http://p:19033",
        "transfer_id": "xfer-same",
    }
    return Request(
        "request-same",
        list(range(n)),
        SamplingParams(
            max_tokens=1 if producer else 12, extra_args={"kv_transfer_params": params}
        ),
        None,
        cache_salt="same-salt",
    )


@pytest.mark.parametrize("n", [1, 2, 64, 65, 85000])
def test_consumer_replays_exact_last_original_token_with_partial_prefix(n):
    obj = scheduler(config())
    req = request(n)
    for local in (0, max(n - 2, 0), n - 1):
        matched, asynchronous = obj.get_num_new_matched_tokens(req, local)
        assert matched == max(n - 1 - local, 0)
        assert asynchronous == (matched > 0)
    assert req.prompt_token_ids == list(range(n)) and req.num_prompt_tokens == n
    assert req.kv_transfer_params["transfer_id"] == "xfer-same"


def test_native_allocation_full_cache_ack_and_producer_lease_unchanged():
    consumer = scheduler(config())
    req = request(65)

    class Blocks:
        def get_unhashed_block_ids_all_groups(self):
            return ([41],)

    consumer.update_state_after_alloc(req, Blocks(), 1)
    assert consumer._reqs_need_recv[req.request_id][1] == [[41]]
    assert req.kv_transfer_params["do_remote_prefill"] is False
    full = request(65)
    consumer.update_state_after_alloc(full, Blocks(), 0)
    assert consumer._reqs_need_recv[full.request_id][1] == []
    producer = scheduler(config(producer=True))
    req = request(65, producer=True)
    producer.on_new_request(req)
    assert req.num_prompt_tokens == 65 and req.prompt_token_ids[-1] == 64
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    assert producer.request_finished(req, ([7, 8],)) == (True, None)
    assert producer._reqs_need_send[req.request_id][1] == [[7, 8]]


@pytest.mark.parametrize("method", [None, "mtp", "dfly"])
def test_plain_hy4_and_mtp_keep_stock_remote_count(method):
    obj = scheduler(config(standard=False, method=method))
    assert obj._get_remote_prefill_token_count(65) == 65
    obj._has_mamba = True
    assert obj._get_remote_prefill_token_count(65) == 64


def test_external_connector_binds_owned_scheduler_without_global_patch(monkeypatch):
    original = native.MooncakeConnector.__init__
    monkeypatch.setattr(base.KVConnectorBase_V1, "__init__", lambda *args: None)
    obj = owned.MooncakeConnector(
        config(), base.KVConnectorRole.SCHEDULER, cache_config()
    )
    assert type(obj.connector_scheduler) is owned.HY4PCPMooncakeScheduler
    assert (
        original.__globals__["MooncakeConnectorScheduler"]
        is native.MooncakeConnectorScheduler
    )


@pytest.mark.parametrize("alias_has_spec", [False, True])
def test_native_mla_alias_storage_and_transfer_region_are_deduplicated(alias_has_spec):
    # Run the actual stock registration body with real shared CPU storage.
    # Only the external transfer-engine registration leaf is replaced.
    # The owned destructor tolerates pre-constructor failures; no transport
    # resources exist in this registration-only probe.
    obj = object.__new__(owned.HY4PCPMooncakeWorker)
    obj.use_mla = True
    obj.is_kv_consumer = True
    obj._physical_blocks_per_logical_kv_block = 1
    obj._prepared_transfer_regions = {}
    obj._layout_by_peer = {}
    obj._acked_layout_peers = set()
    names = ["model.layers.0.self_attn.attn", "model.layers.78.self_attn.attn"]
    spec = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=16, dtype=torch.float32
    )
    obj._layer_specs = dict.fromkeys(names if alias_has_spec else names[:1], spec)
    obj.kv_cache_config = NS(
        num_blocks=4, transfer_group_index_by_layer=dict.fromkeys(names, 0)
    )
    registrations = []
    obj.engine = NS(
        batch_register_memory=lambda ptrs, sizes: (
            registrations.append((ptrs, sizes)) or 0
        )
    )
    cache = torch.zeros(4, 64, 16)
    native.MooncakeConnectorWorker.register_kv_caches(
        obj, {names[0]: cache, names[1]: cache.view_as(cache)}
    )
    assert len(registrations) == 1 and len(registrations[0][0]) == 1
    assert obj.kv_caches_base_addr == [cache.data_ptr()]
    assert obj.registered_layer_names == [names[0]]
    assert obj.region_shared_groups == [(0,)]
