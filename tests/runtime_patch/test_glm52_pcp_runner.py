# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for GLM-5.2 PCP integration with Model Runner V2."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
import textwrap
from dataclasses import dataclass
from types import FunctionType, ModuleType, SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest
import torch

from vllm_hcu.patch.worker.framework_opt._common import PatchCompatibilityError


def test_unitary_pcp_world_size_is_fullgraph_compilable() -> None:
    """TP/DCP-only GLM execution must not inspect dynamic PCP/MTP state."""

    from vllm_hcu.model_executor.layers.attention.pcp import (
        effective_pcp_world_size,
    )

    def resolve_world_size(value: torch.Tensor) -> torch.Tensor:
        return value + effective_pcp_world_size(1)

    compiled = torch.compile(resolve_world_size, backend="eager", fullgraph=True)

    torch.testing.assert_close(compiled(torch.tensor(1)), torch.tensor(2))


@pytest.fixture
def pcp_runner_module(monkeypatch: pytest.MonkeyPatch):
    """Load the HCU runner over a behavior-recording upstream MRV2 base."""

    events: list[str] = []
    upstream_name = "vllm.v1.worker.gpu.model_runner"
    adapter_name = "vllm_hcu.v1.hcu_model_runner_v2"
    upstream_module = ModuleType(upstream_name)
    pcp_module = ModuleType("vllm_hcu.v1.pcp_manager")
    pcp_module.maybe_build_pcp_manager = lambda *args: None

    class FakeBlockTables:
        def get_dummy_block_tables(self, num_reqs):
            events.append("block_tables.get_dummy_block_tables")
            return ("dummy-blocks", num_reqs)

        def get_dummy_slot_mappings(self, num_tokens):
            events.append("block_tables.get_dummy_slot_mappings")
            return ("global-dummy-slots", num_tokens)

    class UpstreamGPUModelRunner:
        def __init__(self, vllm_config, device):
            events.append("super.__init__")
            self.vllm_config = vllm_config
            self.device = device
            self.req_states = object()
            self.execute_model_state = None

        def initialize_kv_cache(self, kv_cache_config):
            events.append("super.initialize_kv_cache")
            self.kv_cache_config = kv_cache_config
            self.block_tables = FakeBlockTables()

        def prepare_inputs(self, scheduler_output, batch_desc):
            events.append("super.prepare_inputs")
            assert scheduler_output == "scheduler-output"
            assert batch_desc == "batch-desc"
            return self.global_batch

        def prepare_attn(self, input_batch):
            events.append("super.prepare_attn")
            return ("global-blocks", input_batch), "global-slots"

        def prepare_dummy_attn(self, input_batch):
            events.append("super.prepare_dummy_attn")
            return ("global-dummy-blocks", input_batch), "global-dummy-slots"

        def sample_tokens(self, grammar_output):
            events.append("super.sample_tokens")
            assert grammar_output == "grammar"
            assert self.execute_model_state.hidden_states is self.expected_hidden
            assert self.execute_model_state.input_batch is self.expected_batch
            if hasattr(self, "expected_attn_metadata"):
                assert (
                    self.execute_model_state.attn_metadata
                    == self.expected_attn_metadata
                )
                assert (
                    self.execute_model_state.slot_mappings_by_layer
                    == self.expected_slot_mappings_by_layer
                )
                from vllm_hcu.model_executor.layers.attention import pcp

                assert getattr(
                    pcp, "in_replicated_mtp_batch", lambda: False
                )()
            return "sampled"

    upstream_module.GPUModelRunner = UpstreamGPUModelRunner
    monkeypatch.setitem(sys.modules, upstream_name, upstream_module)
    monkeypatch.setitem(sys.modules, pcp_module.__name__, pcp_module)
    monkeypatch.delitem(sys.modules, adapter_name, raising=False)
    adapter_module = importlib.import_module(adapter_name)
    yield adapter_module, events


def _config(pcp_size: int) -> object:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
        )
    )


class _ExecuteModelState(NamedTuple):
    input_batch: object
    hidden_states: object


class _MTPExecuteModelState(NamedTuple):
    input_batch: object
    attn_metadata: object
    slot_mappings_by_layer: object
    hidden_states: object
    aux_hidden_states: object
    finished_req_ids: object


def test_pcp_runner_orders_lifecycle_and_restores_sampling_state(
    pcp_runner_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving partition or restore across its upstream boundary is a bug."""

    runner_module, events = pcp_runner_module
    global_batch = object()
    local_batch = object()
    global_hidden = object()
    local_hidden = object()
    synchronized_batches: list[object] = []

    class Manager:
        def partition_batch(self, input_batch):
            events.append("partition_batch")
            assert input_batch is global_batch
            return local_batch

        def prepare_attn(self, input_batch):
            events.append("pcp.prepare_attn")
            assert input_batch is local_batch
            return "local-blocks", "gathered-slots"

        def restore_for_sampling(self, hidden_states):
            events.append("restore_for_sampling")
            assert hidden_states is local_hidden
            return global_hidden, global_batch

    manager = Manager()

    def build_manager(vllm_config, device, req_states, block_tables):
        events.append("build_pcp_manager")
        assert vllm_config.parallel_config.prefill_context_parallel_size == 2
        assert device == "hcu:0"
        assert req_states is runner.req_states
        assert block_tables is runner.block_tables
        return manager

    def synchronize(model_runner, input_batch):
        assert events[-1] == "super.sample_tokens"
        assert model_runner is runner
        synchronized_batches.append(input_batch)
        return False

    monkeypatch.setattr(runner_module, "maybe_build_pcp_manager", build_manager)
    monkeypatch.setattr(
        runner_module,
        "synchronize_pp_spec_draft_tokens",
        synchronize,
    )

    runner = runner_module.HcuGPUModelRunnerV2(_config(2), "hcu:0")
    assert runner.pcp_manager is None
    runner.global_batch = global_batch
    events.clear()

    runner.initialize_kv_cache(SimpleNamespace(kv_cache_groups=[object()]))
    assert runner.pcp_manager is manager
    prepared = runner.prepare_inputs("scheduler-output", "batch-desc")
    assert prepared is local_batch
    assert runner.prepare_attn(prepared) == ("local-blocks", "gathered-slots")

    state = _ExecuteModelState(local_batch, local_hidden)
    runner.execute_model_state = state
    runner.expected_hidden = global_hidden
    runner.expected_batch = global_batch
    assert runner.sample_tokens("grammar") == "sampled"

    assert state.hidden_states is local_hidden
    assert state.input_batch is local_batch
    assert runner.execute_model_state.hidden_states is global_hidden
    assert runner.execute_model_state.input_batch is global_batch
    assert synchronized_batches == [global_batch]
    assert events == [
        "super.initialize_kv_cache",
        "build_pcp_manager",
        "super.prepare_inputs",
        "partition_batch",
        "pcp.prepare_attn",
        "restore_for_sampling",
        "super.sample_tokens",
    ]


def test_pcp_runner_rejects_multiple_resolved_kv_cache_groups(
    pcp_runner_module,
) -> None:
    """PCP metadata has one block table and cannot address two cache groups."""

    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    runner_module, events = pcp_runner_module
    runner = runner_module.HcuGPUModelRunnerV2(_config(2), "hcu:0")
    events.clear()
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full.layer"], spec),
            KVCacheGroupSpec(["sliding.layer"], spec),
        ],
    )

    with pytest.raises(ValueError, match="exactly one KV cache group"):
        runner.initialize_kv_cache(kv_cache_config)

    assert events == []


def test_pcp_runner_replaces_immutable_execute_model_state(
    pcp_runner_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v0.25.1 stores execution state in an immutable NamedTuple."""

    runner_module, _ = pcp_runner_module
    global_batch = object()
    local_batch = object()
    global_hidden = object()
    local_hidden = object()

    class Manager:
        def restore_for_sampling(self, hidden_states):
            assert hidden_states is local_hidden
            return global_hidden, global_batch

    monkeypatch.setattr(
        runner_module,
        "synchronize_pp_spec_draft_tokens",
        lambda *args: False,
    )
    runner = runner_module.HcuGPUModelRunnerV2(_config(2), "hcu:0")
    runner.pcp_manager = Manager()
    original_state = _ExecuteModelState(local_batch, local_hidden)
    runner.execute_model_state = original_state
    runner.expected_hidden = global_hidden
    runner.expected_batch = global_batch

    assert runner.sample_tokens("grammar") == "sampled"

    assert original_state.input_batch is local_batch
    assert original_state.hidden_states is local_hidden
    assert runner.execute_model_state is not original_state
    assert runner.execute_model_state.input_batch is global_batch
    assert runner.execute_model_state.hidden_states is global_hidden


def test_pcp_mtp_rebuilds_global_drafter_attention_state(
    pcp_runner_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reusing rank-local target metadata corrupts the global MTP proposal."""

    runner_module, events = pcp_runner_module
    global_batch = object()
    local_batch = object()
    global_hidden = object()
    local_hidden = object()

    class Manager:
        def restore_for_sampling(self, hidden_states):
            events.append("restore_for_sampling")
            assert hidden_states is local_hidden
            return global_hidden, global_batch

        def prepare_global_attn(self):
            events.append("pcp.prepare_global_attn")
            return ("cached-global-blocks", global_batch), "cached-global-slots"

    class ModelState:
        def prepare_attn(
            self,
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
        ):
            events.append("model_state.prepare_global_mtp_attn")
            assert input_batch is global_batch
            assert cudagraph_mode.name == "NONE"
            assert block_tables == ("cached-global-blocks", global_batch)
            assert slot_mappings == "cached-global-slots"
            assert attn_groups == "attn-groups"
            assert kv_cache_config == "kv-config"
            return "global-mtp-attn-metadata"

    def build_slots(slot_mappings, kv_cache_config):
        events.append("build_global_slot_mappings_by_layer")
        assert slot_mappings == "cached-global-slots"
        assert kv_cache_config == "kv-config"
        return "global-mtp-slots-by-layer"

    monkeypatch.setattr(
        runner_module,
        "build_slot_mappings_by_layer",
        build_slots,
        raising=False,
    )
    monkeypatch.setattr(
        runner_module,
        "synchronize_pp_spec_draft_tokens",
        lambda *args: False,
    )

    runner = runner_module.HcuGPUModelRunnerV2(_config(2), "hcu:0")
    runner.pcp_manager = Manager()
    runner.speculator = object()
    runner.model_state = ModelState()
    runner.kv_cache_config = "kv-config"
    runner.attn_groups = "attn-groups"
    runner.expected_hidden = global_hidden
    runner.expected_batch = global_batch
    runner.expected_attn_metadata = "global-mtp-attn-metadata"
    runner.expected_slot_mappings_by_layer = "global-mtp-slots-by-layer"
    runner.execute_model_state = _MTPExecuteModelState(
        input_batch=local_batch,
        attn_metadata="local-attn-metadata",
        slot_mappings_by_layer="local-slots-by-layer",
        hidden_states=local_hidden,
        aux_hidden_states=None,
        finished_req_ids=set(),
    )
    events.clear()

    assert runner.sample_tokens("grammar") == "sampled"
    assert events == [
        "restore_for_sampling",
        "pcp.prepare_global_attn",
        "build_global_slot_mappings_by_layer",
        "model_state.prepare_global_mtp_attn",
        "super.sample_tokens",
    ]


def test_pcp_runner_routes_dummy_slots_through_manager(
    pcp_runner_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Using global dummy slots would under-size a PCP-expanded dummy batch."""

    runner_module, events = pcp_runner_module
    input_batch = SimpleNamespace(num_reqs=3, num_tokens=7)

    class Manager:
        def get_dummy_slot_mappings(self, num_tokens):
            events.append("pcp.get_dummy_slot_mappings")
            assert num_tokens == 7
            return "pcp-dummy-slots"

    manager = Manager()
    monkeypatch.setattr(
        runner_module,
        "maybe_build_pcp_manager",
        lambda *args: manager,
    )

    runner = runner_module.HcuGPUModelRunnerV2(_config(2), "hcu:0")
    runner.initialize_kv_cache(SimpleNamespace(kv_cache_groups=[object()]))
    events.clear()

    assert runner.prepare_dummy_attn(input_batch) == (
        ("dummy-blocks", 3),
        "pcp-dummy-slots",
    )
    assert events == [
        "block_tables.get_dummy_block_tables",
        "pcp.get_dummy_slot_mappings",
    ]


def test_pcp_one_preserves_the_existing_runner_event_path(
    pcp_runner_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PCP=1 must not call a manager helper or bypass an upstream method."""

    runner_module, events = pcp_runner_module
    global_batch = object()
    hidden_states = object()

    def unexpected_builder(*args):
        pytest.fail("PCP=1 called maybe_build_pcp_manager")

    def synchronize(model_runner, input_batch):
        events.append("synchronize_pp_spec_draft_tokens")
        assert model_runner is runner
        assert input_batch is global_batch
        return True

    monkeypatch.setattr(
        runner_module,
        "maybe_build_pcp_manager",
        unexpected_builder,
    )
    monkeypatch.setattr(
        runner_module,
        "synchronize_pp_spec_draft_tokens",
        synchronize,
    )

    runner = runner_module.HcuGPUModelRunnerV2(_config(1), "hcu:0")
    assert runner.pcp_manager is None
    runner.global_batch = global_batch
    events.clear()

    runner.initialize_kv_cache("kv-config")
    assert runner.prepare_inputs("scheduler-output", "batch-desc") is global_batch
    assert runner.prepare_attn(global_batch) == (
        ("global-blocks", global_batch),
        "global-slots",
    )
    assert runner.prepare_dummy_attn(global_batch) == (
        ("global-dummy-blocks", global_batch),
        "global-dummy-slots",
    )
    runner.execute_model_state = SimpleNamespace(
        hidden_states=hidden_states,
        input_batch=global_batch,
    )
    runner.expected_hidden = hidden_states
    runner.expected_batch = global_batch
    assert runner.sample_tokens("grammar") == "sampled"

    assert events == [
        "super.initialize_kv_cache",
        "super.prepare_inputs",
        "super.prepare_attn",
        "super.prepare_dummy_attn",
        "super.sample_tokens",
        "synchronize_pp_spec_draft_tokens",
    ]


@dataclass
class _CommonAttentionMetadata:
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    is_prefilling: torch.Tensor | None = None


CommonAttentionMetadata = _CommonAttentionMetadata


def _build_attn_metadata(
    attn_groups,
    num_reqs,
    num_tokens,
    query_start_loc_gpu,
    query_start_loc_cpu,
    max_query_len,
    seq_lens,
    max_seq_len,
    block_tables,
    slot_mappings,
    kv_cache_config,
    seq_lens_cpu_upper_bound=None,
    dcp_local_seq_lens=None,
    positions=None,
    mm_req_doc_ranges=None,
    model_specific_attn_metadata=None,
    for_cudagraph_capture=False,
    causal=True,
    rswa_prefix_lens=None,
):
    del (
        num_tokens,
        max_query_len,
        seq_lens,
        max_seq_len,
        block_tables,
        slot_mappings,
        kv_cache_config,
        seq_lens_cpu_upper_bound,
        dcp_local_seq_lens,
        positions,
        mm_req_doc_ranges,
        for_cudagraph_capture,
        causal,
        rswa_prefix_lens,
    )
    extra = model_specific_attn_metadata.get_extra_common_attn_kwargs(
        0, num_reqs
    )
    common = CommonAttentionMetadata(
        query_start_loc_gpu,
        query_start_loc_cpu,
        **extra,
    )
    builder = attn_groups[0][0].get_metadata_builder(0)
    attn_extra = model_specific_attn_metadata.get_extra_attn_kwargs(
        builder, num_reqs
    )
    return {
        "mla.layer": builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            **attn_extra,
        )
    }


def _build_attn_metadata_without_common_hook(
    attn_groups,
    num_reqs,
    num_tokens,
    query_start_loc_gpu,
    query_start_loc_cpu,
    max_query_len,
    seq_lens,
    max_seq_len,
    block_tables,
    slot_mappings,
    kv_cache_config,
    seq_lens_cpu_upper_bound=None,
    dcp_local_seq_lens=None,
    positions=None,
    mm_req_doc_ranges=None,
    model_specific_attn_metadata=None,
    for_cudagraph_capture=False,
    causal=True,
    rswa_prefix_lens=None,
):
    del (
        attn_groups,
        num_reqs,
        num_tokens,
        max_query_len,
        seq_lens,
        max_seq_len,
        block_tables,
        slot_mappings,
        kv_cache_config,
        seq_lens_cpu_upper_bound,
        dcp_local_seq_lens,
        positions,
        mm_req_doc_ranges,
        model_specific_attn_metadata,
        for_cudagraph_capture,
        causal,
        rswa_prefix_lens,
    )
    return CommonAttentionMetadata(
        query_start_loc_gpu,
        query_start_loc_cpu,
    )


build_attn_metadata = _build_attn_metadata


class CUDAGraphMode:
    FULL = object()
    NONE = object()


def compute_mm_prefix_ranges(**kwargs):
    return kwargs


def _original_prepare_attn(
    self,
    input_batch,
    cudagraph_mode,
    block_tables,
    slot_mappings,
    attn_groups,
    kv_cache_config,
    for_capture=False,
):
    if cudagraph_mode == CUDAGraphMode.FULL:
        num_reqs = input_batch.num_reqs_after_padding
        num_tokens = input_batch.num_tokens_after_padding
    else:
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
    query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
    max_query_len = input_batch.num_scheduled_tokens.max().item()
    seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
    if for_capture:
        max_seq_len = self.max_model_len
    else:
        max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
    req_doc_ranges = None
    if (
        self.supports_mm_inputs
        and self.encoder_cache is not None
        and self.model_config.is_mm_prefix_lm
    ):
        req_doc_ranges = compute_mm_prefix_ranges(
            req_ids=input_batch.req_ids,
            mm_features=self.encoder_cache.mm_features,
            sliding_window=self.model_config.get_sliding_window(),
        )
    return build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=input_batch.query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=max_query_len,
        seq_lens=input_batch.seq_lens,
        max_seq_len=max_seq_len,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
        positions=input_batch.positions,
        mm_req_doc_ranges=req_doc_ranges,
        for_cudagraph_capture=for_capture,
        rswa_prefix_lens=input_batch.prompt_lens,
    )


def _prepare_attn_with_request_sizing_drift(
    self,
    input_batch,
    cudagraph_mode,
    block_tables,
    slot_mappings,
    attn_groups,
    kv_cache_config,
    for_capture=False,
):
    if cudagraph_mode == CUDAGraphMode.FULL:
        num_reqs = input_batch.num_reqs_after_padding
        num_tokens = input_batch.num_tokens_after_padding
    else:
        # Same signature and audited names, but silently changes request sizing.
        num_reqs = input_batch.num_reqs + 1
        num_tokens = input_batch.num_tokens
    query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
    max_query_len = input_batch.num_scheduled_tokens.max().item()
    seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
    if for_capture:
        max_seq_len = self.max_model_len
    else:
        max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
    req_doc_ranges = None
    if (
        self.supports_mm_inputs
        and self.encoder_cache is not None
        and self.model_config.is_mm_prefix_lm
    ):
        req_doc_ranges = compute_mm_prefix_ranges(
            req_ids=input_batch.req_ids,
            mm_features=self.encoder_cache.mm_features,
            sliding_window=self.model_config.get_sliding_window(),
        )
    return build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=input_batch.query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=max_query_len,
        seq_lens=input_batch.seq_lens,
        max_seq_len=max_seq_len,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
        positions=input_batch.positions,
        mm_req_doc_ranges=req_doc_ranges,
        for_cudagraph_capture=for_capture,
        rswa_prefix_lens=input_batch.prompt_lens,
    )


def _fake_default_model_state_module(adapter) -> ModuleType:
    target = ModuleType(adapter.TARGET_MODULE)

    class DefaultModelState:
        pass

    target.CUDAGraphMode = CUDAGraphMode
    target.CommonAttentionMetadata = _CommonAttentionMetadata
    target.torch = torch
    target.compute_mm_prefix_ranges = compute_mm_prefix_ranges
    target.build_attn_metadata = FunctionType(
        _build_attn_metadata.__code__,
        target.__dict__,
        name="build_attn_metadata",
        argdefs=_build_attn_metadata.__defaults__,
    )
    DefaultModelState.prepare_attn = FunctionType(
        _original_prepare_attn.__code__,
        target.__dict__,
        name="prepare_attn",
        argdefs=_original_prepare_attn.__defaults__,
    )
    target.DefaultModelState = DefaultModelState
    return target


def _source_sha256(function) -> str:
    source = textwrap.dedent(inspect.getsource(function))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _accept_synthetic_model_state_sources(
    monkeypatch: pytest.MonkeyPatch,
    adapter,
    target: ModuleType,
) -> None:
    monkeypatch.setattr(
        adapter,
        "_V0251_PREPARE_ATTN_SOURCE_SHA256",
        _source_sha256(target.DefaultModelState.prepare_attn),
        raising=False,
    )
    monkeypatch.setattr(
        adapter,
        "_V0251_BUILD_ATTN_METADATA_SOURCE_SHA256",
        _source_sha256(target.build_attn_metadata),
        raising=False,
    )


def test_pcp_default_model_state_slices_metadata_and_propagates_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trailing virtual rows or missing phase data break MLA classification."""

    adapter = importlib.import_module(
        "vllm_hcu.patch.worker.framework_opt.patch_pcp_model_state"
    )
    target = _fake_default_model_state_module(adapter)
    _accept_synthetic_model_state_sources(monkeypatch, adapter, target)
    assert adapter.apply_to_module(target) is True
    assert adapter.apply_to_module(target) is False

    class Builder:
        supports_pcp_plan = True

        def build(
            self,
            *,
            common_prefix_len,
            common_attn_metadata,
            pcp_plan=None,
        ):
            assert common_prefix_len == 0
            common_attn_metadata.pcp_plan = pcp_plan
            return common_attn_metadata

    class Group:
        def get_metadata_builder(self, index):
            assert index == 0
            return Builder()

    state = target.DefaultModelState()
    state.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2)
    )
    state.supports_mm_inputs = False
    state.max_model_len = 64
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=4,
        num_tokens=3,
        num_tokens_after_padding=8,
        query_start_loc_np=np.array([0, 1, 3, 9, 9], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 3, 9, 9], dtype=torch.int32),
        num_scheduled_tokens=np.array([1, 2], dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor([1, 3], dtype=torch.int32),
        seq_lens=torch.tensor([1, 3], dtype=torch.int32),
        dcp_local_seq_lens=None,
        positions=torch.tensor([0, 0, 1], dtype=torch.int64),
        req_ids=["decode", "prefill"],
        is_prefilling_np=np.array([False, True], dtype=np.bool_),
        prompt_lens=None,
        _vllm_hcu_pcp_plan="gqa-plan",
    )
    metadata = state.prepare_attn(
        input_batch,
        target.CUDAGraphMode.NONE,
        (torch.zeros((2, 1), dtype=torch.int32),),
        torch.zeros((1, 3), dtype=torch.int64),
        [[Group()]],
        SimpleNamespace(kv_cache_groups=[object()]),
    )
    common = metadata["mla.layer"]

    captured: dict[str, object] = {}

    def capture_original(**kwargs):
        captured.update(kwargs)
        return {"path": "original"}

    target.build_attn_metadata = capture_original
    state.vllm_config.parallel_config.prefill_context_parallel_size = 1
    pcp_one = state.prepare_attn(
        input_batch,
        target.CUDAGraphMode.NONE,
        (torch.zeros((2, 1), dtype=torch.int32),),
        torch.zeros((1, 3), dtype=torch.int64),
        [[Group()]],
        SimpleNamespace(kv_cache_groups=[object()]),
    )
    payload = {
        "pcp_query_gpu": common.query_start_loc.tolist(),
        "pcp_query_cpu": common.query_start_loc_cpu.tolist(),
        "pcp_is_prefilling": common.is_prefilling.tolist(),
        "pcp_plan": common.pcp_plan,
        "pcp_one_result": pcp_one,
        "pcp_one_query_gpu": captured["query_start_loc_gpu"].tolist(),
        "pcp_one_query_cpu": captured["query_start_loc_cpu"].tolist(),
        "pcp_one_has_phase_adapter": (
            "model_specific_attn_metadata" in captured
        ),
    }
    assert payload == {
        "pcp_query_gpu": [0, 1, 3],
        "pcp_query_cpu": [0, 1, 3],
        "pcp_is_prefilling": [False, True],
        "pcp_plan": "gqa-plan",
        "pcp_one_result": {"path": "original"},
        "pcp_one_query_gpu": [0, 1, 3, 9, 9],
        "pcp_one_query_cpu": [0, 1, 3, 9, 9],
        "pcp_one_has_phase_adapter": False,
    }


def test_pcp_model_state_routes_plan_only_to_flash_attention_builder() -> None:
    """Passing a GQA plan to MLA metadata would change its established ABI."""

    adapter = importlib.import_module(
        "vllm_hcu.patch.worker.framework_opt.patch_pcp_model_state"
    )
    plan = object()
    request_metadata = adapter._PCPRequestPhaseMetadata(
        torch.tensor([True]),
        pcp_plan=plan,
    )

    flash_builder = SimpleNamespace(supports_pcp_plan=True)
    mla_builder = SimpleNamespace()
    assert request_metadata.get_extra_attn_kwargs(flash_builder, 1) == {
        "pcp_plan": plan
    }
    assert request_metadata.get_extra_attn_kwargs(mla_builder, 1) == {}


def test_pcp_default_model_state_rejects_same_signature_behavior_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed request-sizing body must fail before wrapper installation."""

    adapter = importlib.import_module(
        "vllm_hcu.patch.worker.framework_opt.patch_pcp_model_state"
    )
    target = _fake_default_model_state_module(adapter)
    _accept_synthetic_model_state_sources(monkeypatch, adapter, target)
    drifted = FunctionType(
        _prepare_attn_with_request_sizing_drift.__code__,
        target.__dict__,
        name="prepare_attn",
        argdefs=_prepare_attn_with_request_sizing_drift.__defaults__,
    )
    target.DefaultModelState.prepare_attn = drifted

    with pytest.raises(
        PatchCompatibilityError, match="source fingerprint"
    ) as error:
        adapter.apply_to_module(target)

    message = str(error.value)
    assert adapter.TARGETS[0] in message
    assert "expected sha256=" in message
    assert "actual sha256=" in message
    assert target.DefaultModelState.prepare_attn is drifted
    assert not hasattr(target.DefaultModelState, "_vllm_hcu_original_prepare_attn")
    assert not getattr(target, "_vllm_hcu_pcp_model_state_applied", False)


def test_pcp_default_model_state_rejects_common_hook_behavior_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A helper that drops model-specific common metadata must fail closed."""

    adapter = importlib.import_module(
        "vllm_hcu.patch.worker.framework_opt.patch_pcp_model_state"
    )
    target = _fake_default_model_state_module(adapter)
    _accept_synthetic_model_state_sources(monkeypatch, adapter, target)
    original_prepare_attn = target.DefaultModelState.prepare_attn
    drifted = FunctionType(
        _build_attn_metadata_without_common_hook.__code__,
        target.__dict__,
        name="build_attn_metadata",
        argdefs=_build_attn_metadata_without_common_hook.__defaults__,
    )
    target.build_attn_metadata = drifted

    with pytest.raises(
        PatchCompatibilityError, match="source fingerprint"
    ) as error:
        adapter.apply_to_module(target)

    message = str(error.value)
    assert adapter.TARGETS[1] in message
    assert "expected sha256=" in message
    assert "actual sha256=" in message
    assert target.build_attn_metadata is drifted
    assert target.DefaultModelState.prepare_attn is original_prepare_attn
    assert not hasattr(target.DefaultModelState, "_vllm_hcu_original_prepare_attn")
    assert not getattr(target, "_vllm_hcu_pcp_model_state_applied", False)


def test_pcp_default_model_state_rejects_signature_drift() -> None:
    """A changed upstream prepare_attn contract must fail before installation."""

    adapter = importlib.import_module(
        "vllm_hcu.patch.worker.framework_opt.patch_pcp_model_state"
    )
    target = ModuleType(adapter.TARGET_MODULE)

    class DefaultModelState:
        def prepare_attn(self, input_batch):
            return input_batch

    target.DefaultModelState = DefaultModelState
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        adapter.apply_to_module(target)
