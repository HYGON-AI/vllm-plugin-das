# SPDX-License-Identifier: Apache-2.0
"""Actual stock CPU requests, PCP partitions and dense draft configuration."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from hcu_vllm_das.models.hy4_standard.pcp_draft import (
    make_producer_draft_parallel,
    require_producer_zero_drafts,
)
from hcu_vllm_das.models.hy4_standard.pcp_manager import (
    HYV4PCPProducerManager,
    PCPLocalLayout,
)
from hcu_vllm_das.models.hy4_standard.pcp_scheduler import validate_prefill_only_request
from vllm.config import ParallelConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import Request, RequestStatus


def request(*, max_tokens=1, **sampling):
    params = SamplingParams(
        max_tokens=max_tokens,
        extra_args={
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "transfer_id": "cpu-proof",
            }
        },
        **sampling,
    )
    params.update_from_generation_config({}, eos_token_id=7)
    return Request("pcp-proof", [11, 12, 13], params, None)


@pytest.mark.parametrize(
    "sampling, token",
    [
        ({}, 7),
        ({"stop_token_ids": [8]}, 8),
        ({"stop": ["END"]}, 9),
        ({"min_tokens": 1, "skip_clone": True, "stop": "END"}, 7),
    ],
)
def test_discarded_producer_sample_ends_by_native_length_cap(sampling, token):
    req = request(**sampling)
    original = req.sampling_params
    before = deepcopy(original)
    prepared = validate_prefill_only_request(req)
    assert isinstance(prepared, SamplingParams)
    assert prepared is not original
    assert prepared.ignore_eos and prepared.eos_token_id is None
    assert prepared.stop == [] and prepared.stop_token_ids == []
    assert prepared.all_stop_token_ids == set() and prepared.min_tokens == 0
    assert prepared.output_text_buffer_length == 0
    assert prepared.extra_args == original.extra_args
    assert prepared.extra_args is not original.extra_args
    req.sampling_params = prepared
    req.append_output_token_ids([token])
    assert check_stop(req, max_model_len=128)
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert original == before
    assert original.eos_token_id == 7


def test_real_eos_request_without_normalization_stops_before_length_cap():
    req = request()
    req.append_output_token_ids([7])
    assert check_stop(req, max_model_len=128)
    assert req.status == RequestStatus.FINISHED_STOPPED


@pytest.mark.parametrize(
    "change",
    [
        {"max_tokens": 2},
        {"resumable": True},
        {"kv_transfer_params": {"do_remote_decode": False, "transfer_id": "x"}},
        {
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": True,
                "transfer_id": "x",
            }
        },
        {"kv_transfer_params": {"do_remote_decode": True}},
    ],
)
def test_prefill_only_admission_rejects_invalid_real_request(change):
    req = request()
    for name, value in change.items():
        setattr(req, name, value)
    before = deepcopy(req.sampling_params)
    with pytest.raises(ValueError):
        validate_prefill_only_request(req)
    assert req.sampling_params == before
    assert req.status == RequestStatus.WAITING


def layout_for(rank, counts, prefill, *, shard=False):
    manager = HYV4PCPProducerManager(16, rank, torch.device("cpu"), shard)
    counts = np.asarray(counts, dtype=np.int64)
    prefill = np.asarray(prefill, dtype=np.bool_)
    qsl = np.concatenate(([0], np.cumsum(counts)))
    segments = [manager._get_rank_segments(r, counts, prefill, qsl) for r in range(16)]
    padded = max(sum(s.num_tokens for s in part) for part in segments)
    return PCPLocalLayout.from_segments(
        segments[rank], global_num_tokens=int(qsl[-1]), padded_tokens=padded, epoch=1
    )


def test_stock_zigzag_selects_global_mrope_and_real_embedding_rows():
    layout = layout_for(7, [64, 3, 65], [True, True, True])
    rows = [14, 15, 48, 49, 88, 89, 90]
    assert list(layout.rows[:7]) == rows
    assert list(layout.valid) == [True] * 7 + [False] * 3
    positions = torch.stack([torch.arange(132) + offset for offset in (0, 1000, 2000)])
    selected = layout.select(positions, token_dim=1)
    torch.testing.assert_close(selected[:, :7], positions[:, rows])
    assert torch.count_nonzero(selected[:, 7:]) == 0
    embedding = torch.nn.Embedding.from_pretrained(
        torch.arange(528).reshape(132, 4).float()
    )
    global_embeddings = embedding(torch.arange(132))
    local_embeddings = layout.select(global_embeddings)
    torch.testing.assert_close(local_embeddings[:7], embedding(torch.tensor(rows)))
    assert torch.count_nonzero(local_embeddings[7:]) == 0


def test_stock_decode_reorder_is_shared_by_positions_and_embeddings():
    layout = layout_for(0, [64, 1, 64], [True, False, True], shard=True)
    # Native decode-first reorder must be preserved, not recomputed by request index.
    assert layout.rows[0] == 64
    global_rows = torch.arange(129)
    local = layout.select(global_rows)
    assert local[0] == 64
    assert local[1:5].tolist() == [0, 1, 62, 63]


def test_empty_short_query_owner_has_only_zero_padding():
    layout = layout_for(15, [1], [False], shard=True)
    assert layout.valid == (False,)
    assert layout.select(torch.tensor([[17.0, 18.0]])).tolist() == [[0.0, 0.0]]


def test_producer_dense_draft_has_real_single_rank_world_without_mutating_target():
    target = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=16,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        all2all_backend="deepep_high_throughput",
        distributed_executor_backend="mp",
    )
    before = deepcopy(target)
    draft = make_producer_draft_parallel(target)
    assert target == before
    assert draft.world_size == draft.world_size_across_dp == 1
    assert draft.prefill_context_parallel_size == draft.data_parallel_size == 1
    assert not draft.enable_expert_parallel and not draft.is_moe_model


@pytest.mark.parametrize("k", [None, False, 1, 6])
def test_producer_cannot_silently_execute_pcp_draft_decode(k):
    with pytest.raises(ValueError, match="explicit K=0"):
        require_producer_zero_drafts(
            dummy_run=False, is_profile=False, num_speculative_tokens=k
        )


def test_explicit_zero_and_private_profile_are_distinct_admitted_paths():
    require_producer_zero_drafts(
        dummy_run=False, is_profile=False, num_speculative_tokens=0
    )
    require_producer_zero_drafts(
        dummy_run=True, is_profile=True, num_speculative_tokens=6
    )


@pytest.fixture
def producer_scheduler(tmp_path):
    from hcu_vllm_das.models.hy4_standard.pcp_scheduler import HYV4PCPProducerScheduler
    from hcu_vllm_das.models.hy4_standard.pd_policy import DIRECT_MOONCAKE_MODULE
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        CompilationMode,
        CUDAGraphMode,
        KVTransferConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.transformers_utils.configs.hy_v4 import HYV4Config
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        MLAAttentionSpec,
    )
    from vllm.v1.structured_output import StructuredOutputManager

    # A genuine standard text component carrying the owned VL identity, as
    # produced by init_vllm_registered_model for the outer multimodal model.
    hf = HYV4Config(hy4_standard_vl=True, architectures=["HYV4ForCausalLM"])
    hf.save_pretrained(tmp_path)
    model = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=128,
    )
    parallel = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=16,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        all2all_backend="deepep_high_throughput",
        distributed_executor_backend="mp",
    )
    cache = CacheConfig(block_size=64, enable_prefix_caching=False)
    cache.num_gpu_blocks = 16
    config = VllmConfig(
        model_config=model,
        parallel_config=parallel,
        cache_config=cache,
        scheduler_config=SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            max_num_batched_tokens=128,
            max_num_seqs=2,
            async_scheduling=False,
            scheduler_cls=HYV4PCPProducerScheduler,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
            custom_ops=["none"],
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MooncakeConnector",
            kv_role="kv_producer",
            kv_connector_module_path=DIRECT_MOONCAKE_MODULE,
        ),
    )
    plan = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn.attn"],
                kv_cache_spec=MLAAttentionSpec(
                    block_size=64, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
                ),
            )
        ],
    )
    return HYV4PCPProducerScheduler(
        config, plan, StructuredOutputManager(config), block_size=64
    )


def test_real_scheduler_rejects_multitoken_before_native_queue_mutation(
    producer_scheduler,
):
    scheduler = producer_scheduler
    req = request(max_tokens=2)
    before = (
        dict(scheduler.requests),
        list(scheduler.waiting),
        list(scheduler.running),
    )
    with pytest.raises(ValueError, match="max_tokens=1"):
        scheduler.add_request(req)
    assert (scheduler.requests, list(scheduler.waiting), scheduler.running) == before


def test_real_scheduler_enqueues_private_sampling_copy_and_rejects_duplicate(
    producer_scheduler,
):
    scheduler = producer_scheduler
    req = request(stop_token_ids=[8], stop=["END"])
    original = req.sampling_params
    original_before = deepcopy(original)
    scheduler.add_request(req)
    assert scheduler.requests[req.request_id] is req
    assert list(scheduler.waiting) == [req]
    assert req.sampling_params is not original
    assert req.kv_transfer_params is not original.extra_args["kv_transfer_params"]
    req.append_output_token_ids([7])
    assert check_stop(req, max_model_len=128)
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert original == original_before
    with pytest.raises(ValueError, match="active request"):
        scheduler.add_request(request())
    assert list(scheduler.waiting) == [req]


def test_actual_cache_plan_target_filter_keeps_alias_schema_unchanged():
    from hcu_vllm_das.models.hy4_standard.pcp_runner import target_attention_names
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        MLAAttentionSpec,
    )

    target = [f"model.layers.{i}.self_attn.attn" for i in (61, 65, 69, 73, 77)]
    draft = [f"model.dfly_layers.{i}.self_attn.attn" for i in range(5)]
    plan = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=target + draft,
                kv_cache_spec=MLAAttentionSpec(
                    block_size=64, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
                ),
            )
        ],
    )
    before = deepcopy(plan)
    assert target_attention_names(plan, draft) == set(target)
    assert plan == before


def test_actual_image_mrope_keeps_global_media_coordinates_across_pcp_chunks():
    from hcu_vllm_das.models.hy4_standard.configs import HYV4VLConfig
    from hcu_vllm_das.models.hy4_standard.mrope import build_mrope_positions

    config = HYV4VLConfig()
    tokens = [17, config.image_start_token_id] + [config.image_token_id] * 6
    tokens += [config.image_end_token_id, 18] + [19] * 54
    global_positions, _ = build_mrope_positions(config, tokens, [[1, 4, 4]], [])
    layout = layout_for(1, [64], [True])
    assert layout.rows == (2, 3, 60, 61)
    local = layout.select(global_positions, token_dim=1)
    assert local[:, :2].tolist() == [[2, 2], [2, 2], [2, 3]]
    torch.testing.assert_close(local[:, 2:], global_positions[:, [60, 61]])


def test_manager_rejects_plain_scheduler_before_runner_construction(producer_scheduler):
    from vllm.v1.core.sched.scheduler import Scheduler

    config = producer_scheduler.vllm_config
    HYV4PCPProducerManager.validate_config(config, supports_mm_inputs=True)
    config.scheduler_config.scheduler_cls = Scheduler
    with pytest.raises(ValueError, match="scheduler"):
        HYV4PCPProducerManager.validate_config(config, supports_mm_inputs=True)


def test_request_and_sampling_max_tokens_cannot_diverge():
    req = request()
    req.sampling_params.max_tokens = 2
    with pytest.raises(ValueError, match="max_tokens=1"):
        validate_prefill_only_request(req)
