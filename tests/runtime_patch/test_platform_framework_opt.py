# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

# Unit tests activate adapters explicitly and therefore disable plugin discovery.
os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

from vllm_hcu.patch.platform.framework_opt import (
    patch_engine_core,
    patch_kv_cache_utils,
    patch_kv_connector_factory,
    patch_mooncake_connector,
    patch_multiproc_executor,
    patch_output_processor,
    patch_outputs,
    patch_parallel_state,
    patch_scheduler,
)


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def test_clean_vllm_real_factory_and_runtime_contract_smoke():
    # Run before this process imports the heavy Mooncake/model stack so the
    # independent clean-vLLM smoke does not temporarily double memory use.
    _run_clean_vllm_real_factory_and_runtime_contract_smoke()


def test_group_coordinator_all_to_all_delegates_to_device_communicator():
    calls = []

    class GroupCoordinator:
        def reduce_scatter(self, input_, dim=-1):
            return input_

    class Communicator:
        def all_to_all_single(self, output, input):
            calls.append((output, input))
            return "done"

    module = _module(patch_parallel_state.TARGET_MODULE, GroupCoordinator=GroupCoordinator)
    assert patch_parallel_state.apply_to_module(module) is True
    group = GroupCoordinator()
    group.device_communicator = Communicator()
    assert group.all_to_all_single("out", "in") == "done"
    assert calls == [("out", "in")]
    assert patch_parallel_state.apply_to_module(module) is False


@dataclass(frozen=True)
class _Spec:
    block_size: int
    bytes_per_token: int
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self):
        return self.page_size_padded or self.block_size * self.bytes_per_token


def test_kv_page_size_uses_lcm_and_safe_padding():
    from vllm_hcu.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size

    specs = {"a": _Spec(1, 6), "b": _Spec(1, 10)}
    result = unify_kv_cache_spec_page_size(specs)
    assert {spec.page_size_bytes for spec in result.values()} == {30}
    assert result["a"].block_size == 5
    assert result["b"].block_size == 3

    def official(kv_cache_spec):
        return kv_cache_spec

    module = _module(
        patch_kv_cache_utils.TARGET_MODULE,
        unify_kv_cache_spec_page_size=official,
    )
    assert patch_kv_cache_utils.apply_to_module(module) is True
    assert module.unify_kv_cache_spec_page_size is unify_kv_cache_spec_page_size
    assert patch_kv_cache_utils.apply_to_module(module) is False


def _fake_factory_module():
    class KVConnectorFactory:
        _registry = {}

        @classmethod
        def get_connector_class(cls, config):
            return config.connector_cls

        @classmethod
        def create_connector(cls, config, role, kv_cache_config):
            return config.connector_cls(config, role, kv_cache_config)

    return _module(
        patch_kv_connector_factory.TARGET_MODULE,
        KVConnectorFactory=KVConnectorFactory,
        supports_hma=lambda connector: True,
    )


def test_kv_factory_registers_hcu_connectors_lazily(monkeypatch):
    imports = []

    class MooncakeConnector:
        pass

    def fake_import(name):
        imports.append(name)
        return SimpleNamespace(MooncakeConnector=MooncakeConnector)

    monkeypatch.setattr(patch_kv_connector_factory.importlib, "import_module", fake_import)
    module = _fake_factory_module()
    factory = module.KVConnectorFactory
    assert patch_kv_connector_factory.apply_to_module(module) is True
    assert imports == []
    for name, expected in patch_kv_connector_factory._HCU_CONNECTORS.items():
        loader = factory._registry[name]
        assert loader._hcu_connector_target == expected
    assert factory._registry["MooncakeConnector"]() is MooncakeConnector
    assert imports == [patch_kv_connector_factory._HCU_CONNECTORS["MooncakeConnector"][0]]
    assert patch_kv_connector_factory.apply_to_module(module) is False


def test_kv_factory_dp_rank_is_optional_and_compatible(monkeypatch):
    module = _fake_factory_module()
    patch_kv_connector_factory.apply_to_module(module)

    class Connector:
        def __init__(self, config, role, kv_cache_config):
            self.args = (config, role, kv_cache_config)

    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(connector_cls=Connector),
        connector_cls=Connector,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        parallel_config=SimpleNamespace(data_parallel_rank=7),
    )
    monkeypatch.setattr(patch_kv_connector_factory.henvs, "VLLM_HCU_USE_DP_CONNECTOR", False)
    assert "dp_rank" in inspect.signature(
        module.KVConnectorFactory.create_connector
    ).parameters
    result = module.KVConnectorFactory.create_connector(config, "role", "cache")
    assert isinstance(result, Connector)


def test_kv_factory_passes_dp_rank_only_to_supporting_connector(monkeypatch):
    module = _fake_factory_module()
    patch_kv_connector_factory.apply_to_module(module)

    class Connector:
        def __init__(self, config, role, kv_cache_config, dp_rank=-1):
            self.dp_rank = dp_rank

    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(connector_cls=Connector),
        connector_cls=Connector,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        parallel_config=SimpleNamespace(data_parallel_rank=7),
    )
    monkeypatch.setattr(patch_kv_connector_factory.henvs, "VLLM_HCU_USE_DP_CONNECTOR", True)
    result = module.KVConnectorFactory.create_connector(config, "role", "cache")
    assert result.dp_rank == 7


@pytest.fixture(scope="module")
def hcu_mooncake():
    return importlib.import_module(patch_mooncake_connector.TARGET_MODULE)


def test_hcu_mooncake_trace_and_transfer_id_contract(hcu_mooncake):
    request_id = "cmpl-123e4567-e89b-12d3-a456-426614174000-extra"
    assert hcu_mooncake.transfer_id_from_req(request_id) == "xfer-123e4567-e89b-12d3-a456-426614174000"
    assert hcu_mooncake.transfer_id_from_req("plain") is None
    assert hcu_mooncake.transfer_id_from_req("plain", {"transfer_id": "x"}) == "x"


def test_hcu_mooncake_heterogeneous_tp_contracts(hcu_mooncake):
    assert hcu_mooncake._get_tp_ratio(4, 2) == 2
    assert hcu_mooncake._get_tp_ratio(2, 4) == -2
    assert hcu_mooncake._get_head_split_ratio(-4) == 4
    assert hcu_mooncake._validate_block_lens_match([4, 8], [4, 8]) is None
    assert "mismatch" in hcu_mooncake._validate_hetero_block_lens_match([4], [8], -2)


def test_hcu_mooncake_metadata_and_layout_contracts(hcu_mooncake):
    assert patch_mooncake_connector.apply_to_module(hcu_mooncake) is True
    assert patch_mooncake_connector.apply_to_module(hcu_mooncake) is False
    metadata = hcu_mooncake.MooncakeXferMetadata(
        remote_hostname="host",
        remote_port=1,
        remote_tp_size=2,
        remote_tp_rank=0,
        req_blocks={},
        kv_caches_base_addr=[],
        block_lens=[],
    )
    assert metadata.src_layer_offset == 0
    assert metadata.model_layer_start == -1
    assert hcu_mooncake._parse_model_layer_index("model.layers.12.self_attn") == 12
    assert hcu_mooncake._cache_type_sort_key("model.layers.0.indexer") == 0


def test_hcu_mooncake_transfer_planning_contracts(hcu_mooncake):
    src, dst, lengths = [], [], []
    hcu_mooncake._append_homogeneous_fa_layer_transfers(
        src_ptrs=src,
        dst_ptrs=dst,
        lengths=lengths,
        local_layer_addr=100,
        layer_block_len=10,
        remote_layer_addr=200,
        layer_remote_block_len=10,
        group_local_block_ids=[[0, 1]],
        group_remote_block_ids=[[2, 3]],
    )
    assert (src, dst, lengths) == ([100], [220], [20])


def test_hcu_mooncake_pp_and_error_propagation_contracts(hcu_mooncake):
    assert hcu_mooncake._validate_hetero_slot_size_bytes(16, 32, 2) is None
    assert hcu_mooncake._validate_hetero_slot_size_bytes(16, 8, -2) is None
    assert "mismatch" in hcu_mooncake._validate_block_lens_match([1], [2])
    for method in ("_fail_pull_metas", "_iter_fa_layer_cache_pairs", "receive_kv"):
        assert callable(getattr(hcu_mooncake.MooncakeConnectorWorker, method))


def test_hcu_mooncake_bootstrap_contracts(hcu_mooncake, monkeypatch):
    local = SimpleNamespace(
        parallel_config=SimpleNamespace(local_engines_only=True, data_parallel_index=3)
    )
    distributed = SimpleNamespace(
        parallel_config=SimpleNamespace(local_engines_only=False, data_parallel_index=0)
    )
    monkeypatch.setattr(hcu_mooncake, "is_local_first_rank", lambda: True)
    monkeypatch.setattr(hcu_mooncake, "is_global_first_rank", lambda: False)
    assert hcu_mooncake.should_launch_bootstrap_server(local) is True
    assert hcu_mooncake.should_launch_bootstrap_server(distributed) is False


def _fake_scheduler_module():
    class Scheduler:
        def schedule(self, throttle_prefills=False):
            return "official"

        def update_draft_token_ids(self, draft_token_ids):
            return None

        def update_draft_token_ids_in_output(self, draft_token_ids, scheduler_output):
            return None

    for name in (
        "_select_waiting_queue_for_scheduling",
        "_is_blocked_waiting_status",
        "_try_promote_blocked_waiting_request",
        "_try_schedule_encoder_inputs",
        "_mamba_block_aligned_split",
        "_build_kv_connector_meta",
        "_inflight_prefill_reserved_blocks",
        "_make_cached_request_data",
        "_update_after_schedule",
        "_preempt_request",
    ):
        setattr(Scheduler, name, lambda self, *args, **kwargs: None)
    return _module(patch_scheduler.TARGET_MODULE, Scheduler=Scheduler)


def test_scheduler_feature_off_keeps_official_class(monkeypatch):
    monkeypatch.setattr(patch_scheduler, "apply", lambda module=None: True)
    monkeypatch.setattr(patch_scheduler.henvs, "VLLM_HCU_USE_PD_SPLIT", False)
    config = SimpleNamespace(
        additional_config={"hcu": {}},
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(
            scheduler_cls=None,
            async_scheduling=False,
        ),
    )
    assert patch_scheduler.select_hcu_scheduler(config) is False
    assert config.scheduler_config.scheduler_cls is None
    from vllm_hcu.v1.core.sched.scheduler import HcuScheduler

    observed: list[bool] = []

    def official_schedule(self, throttle_prefills: bool = False):
        observed.append(throttle_prefills)
        return "official"

    monkeypatch.setattr(HcuScheduler.__mro__[1], "schedule", official_schedule)
    scheduler = object.__new__(HcuScheduler)
    assert HcuScheduler.schedule(scheduler) == "official"
    assert HcuScheduler.schedule(scheduler, throttle_prefills=True) == "official"
    assert observed == [False, True]


def test_scheduler_selects_hcu_class_through_scheduler_cls(monkeypatch):
    monkeypatch.setattr(
        patch_scheduler,
        "apply",
        lambda module=None: pytest.fail("selector eagerly validated scheduler module"),
    )
    monkeypatch.setattr(patch_scheduler.henvs, "VLLM_HCU_USE_PD_SPLIT", True)
    monkeypatch.setattr(patch_scheduler.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    config = SimpleNamespace(
        additional_config={"hcu": {}},
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(
            scheduler_cls=None,
            async_scheduling=False,
        ),
    )
    assert patch_scheduler.select_hcu_scheduler(config) is True
    assert config.scheduler_config.scheduler_cls == patch_scheduler.HCU_SCHEDULER_PATH
    assert patch_scheduler.select_hcu_scheduler(config) is False
    config.scheduler_config.scheduler_cls = None
    monkeypatch.setattr(patch_scheduler.henvs, "VLLM_HCU_USE_CUSTOM_OPS", False)
    with pytest.raises(RuntimeError, match="requires VLLM_HCU_USE_CUSTOM_OPS"):
        patch_scheduler.select_hcu_scheduler(config)


def test_scheduler_rejects_split_pd_with_async_scheduling(monkeypatch):
    monkeypatch.setattr(patch_scheduler.henvs, "VLLM_HCU_USE_PD_SPLIT", True)
    monkeypatch.setattr(patch_scheduler.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    config = SimpleNamespace(
        additional_config={"hcu": {}},
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(
            scheduler_cls=None,
            async_scheduling=True,
        ),
    )
    with pytest.raises(RuntimeError, match="--no-async-scheduling"):
        patch_scheduler.select_hcu_scheduler(config)
    assert config.scheduler_config.scheduler_cls is None


def test_hcu_split_pd_scheduler_is_waiting_first_and_lora_safe():
    from vllm_hcu.v1.core.sched.scheduler import HcuScheduler, PauseState

    scheduler = object.__new__(HcuScheduler)
    scheduler.lora_config = object()
    scheduler.running = [
        SimpleNamespace(lora_request=SimpleNamespace(lora_int_id=2)),
        SimpleNamespace(lora_request=None),
    ]
    assert scheduler._hcu_initial_scheduled_loras() == {2}
    scheduler.waiting = [object()]
    scheduler.skipped_waiting = []
    scheduler._pause_state = PauseState.PAUSED_NEW
    assert scheduler._hcu_can_schedule_waiting(1) is False
    scheduler._pause_state = PauseState.UNPAUSED
    assert scheduler._hcu_can_schedule_waiting(1) is True
    assert scheduler._hcu_can_schedule_waiting(0) is False
    assert HcuScheduler.schedule_split_pd is not HcuScheduler.__mro__[1].schedule


def test_multi_mtp_uses_existing_draft_token_ids_channel():
    module = _fake_scheduler_module()
    assert patch_scheduler.apply_to_module(module) is True
    assert callable(module.Scheduler.update_draft_token_ids)
    assert callable(module.Scheduler.update_draft_token_ids_in_output)


def test_scheduler_logs_decoder_kv_ready_event():
    from vllm_hcu.v1.core.sched.scheduler import HcuScheduler

    assert "log_ttft_event" in HcuScheduler._update_waiting_for_remote_kv.__code__.co_names


def _fake_engine_core_module(calls):
    class EngineCore:
        def post_step(self, model_executed):
            calls.append(("post", model_executed))
            if not self.async_scheduling and self.use_spec_decode and model_executed:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                if draft_token_ids is not None:
                    self.scheduler.update_draft_token_ids(draft_token_ids)
            return "official"

    class EngineCoreProc:
        def __init__(self, vllm_config=None):
            pass

        def _handle_client_request(self, request_type, request):
            calls.append(("handle", request_type, request))
            return "handled"

    return _module(
        patch_engine_core.TARGET_MODULE,
        EngineCore=EngineCore,
        EngineCoreProc=EngineCoreProc,
        EngineCoreRequestType=SimpleNamespace(ADD="add"),
    )


def test_engine_core_registers_dp_connector_request(monkeypatch):
    calls = []
    module = _fake_engine_core_module(calls)
    patch_engine_core.apply_to_module(module)
    monkeypatch.setattr(patch_engine_core, "_prepare_engine_core_runtime", lambda: None)
    monkeypatch.setattr(patch_engine_core, "_set_engine_core_process_role", lambda: None)
    connector = SimpleNamespace(
        register_req=lambda request_id: calls.append(("register", request_id))
    )
    proc = module.EngineCoreProc()
    proc.scheduler = SimpleNamespace(connector=connector)
    monkeypatch.setattr(patch_engine_core.henvs, "VLLM_HCU_USE_DP_CONNECTOR", True)
    request = (SimpleNamespace(request_id="r1"), 0)
    assert proc._handle_client_request("add", request) == "handled"
    assert calls[0] == ("register", "r1")


def test_engine_core_dp_connector_rejects_malformed_add_requests(monkeypatch):
    calls = []
    module = _fake_engine_core_module(calls)
    patch_engine_core.apply_to_module(module)
    monkeypatch.setattr(patch_engine_core, "_prepare_engine_core_runtime", lambda: None)
    monkeypatch.setattr(patch_engine_core, "_set_engine_core_process_role", lambda: None)
    monkeypatch.setattr(patch_engine_core.henvs, "VLLM_HCU_USE_DP_CONNECTOR", True)
    proc = module.EngineCoreProc()
    proc.scheduler = SimpleNamespace(
        connector=SimpleNamespace(register_req=lambda request_id: None)
    )

    for request in (None, (), [], SimpleNamespace(request_id="r1")):
        with pytest.raises(RuntimeError, match="non-empty tuple"):
            proc._handle_client_request("add", request)
    with pytest.raises(RuntimeError, match="missing request_id"):
        proc._handle_client_request("add", (object(), 0))

    proc.scheduler = SimpleNamespace(connector=None)
    with pytest.raises(RuntimeError, match="has no connector"):
        proc._handle_client_request("add", (SimpleNamespace(request_id="r1"), 0))
    proc.scheduler = SimpleNamespace(connector=object())
    with pytest.raises(RuntimeError, match="required register_req"):
        proc._handle_client_request("add", (SimpleNamespace(request_id="r1"), 0))
    assert calls == []


def test_engine_core_dp_connector_feature_off_delegates_malformed_request(
    monkeypatch,
):
    calls = []
    module = _fake_engine_core_module(calls)
    patch_engine_core.apply_to_module(module)
    monkeypatch.setattr(patch_engine_core, "_prepare_engine_core_runtime", lambda: None)
    monkeypatch.setattr(patch_engine_core, "_set_engine_core_process_role", lambda: None)
    monkeypatch.setattr(patch_engine_core.henvs, "VLLM_HCU_USE_DP_CONNECTOR", False)
    proc = module.EngineCoreProc()

    assert proc._handle_client_request("add", None) == "handled"
    assert calls == [("handle", "add", None)]


def test_engine_core_preloads_hcu_worker_before_worker_patches(monkeypatch):
    events = []
    monkeypatch.setattr(
        patch_engine_core, "_set_engine_core_process_role", lambda: events.append("role")
    )
    monkeypatch.setattr(
        patch_engine_core.importlib,
        "import_module",
        lambda name: events.append(("import", name)),
    )
    from vllm_hcu.patch import worker as worker_dispatcher

    monkeypatch.setattr(
        worker_dispatcher,
        "prepare_worker_patches",
        lambda: events.append("prepare"),
    )

    patch_engine_core._prepare_engine_core_runtime()

    assert events == ["role", ("import", "vllm_hcu.v1.worker"), "prepare"]


def test_engine_core_worker_import_failure_propagates_before_patch_prepare(
    monkeypatch,
):
    events = []
    monkeypatch.setattr(
        patch_engine_core, "_set_engine_core_process_role", lambda: events.append("role")
    )

    def fail_import(name):
        events.append(("import", name))
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(patch_engine_core.importlib, "import_module", fail_import)
    from vllm_hcu.patch import worker as worker_dispatcher

    monkeypatch.setattr(
        worker_dispatcher,
        "prepare_worker_patches",
        lambda: events.append("prepare"),
    )

    with pytest.raises(ModuleNotFoundError, match="vllm_hcu.v1.worker"):
        patch_engine_core._prepare_engine_core_runtime()
    assert events == ["role", ("import", "vllm_hcu.v1.worker")]


def test_engine_core_init_failure_reasserts_process_role(monkeypatch):
    calls = []
    module = _fake_engine_core_module(calls)

    def fail_init(self, vllm_config=None):
        calls.append(("init", vllm_config))
        raise LookupError("engine init failed")

    module.EngineCoreProc.__init__ = fail_init
    patch_engine_core.apply_to_module(module)
    events = []
    monkeypatch.setattr(
        patch_engine_core,
        "_prepare_engine_core_runtime",
        lambda: events.append("prepare"),
    )
    monkeypatch.setattr(
        patch_engine_core,
        "_set_engine_core_process_role",
        lambda: events.append("role"),
    )

    with pytest.raises(LookupError, match="engine init failed"):
        module.EngineCoreProc(vllm_config="config")
    assert calls == [("init", "config")]
    assert events == ["prepare", "role"]


def test_engine_core_preserves_draft_token_channel_for_pp_multi_mtp():
    from vllm.v1.outputs import DraftTokenIds

    calls = []
    module = _fake_engine_core_module(calls)
    official_post_step = module.EngineCore.post_step
    patch_engine_core.apply_to_module(module)
    assert module.EngineCore.post_step is official_post_step

    draft = DraftTokenIds(req_ids=["r1"], draft_token_ids=[[11, 12]])

    def take_draft_token_ids():
        calls.append(("take",))
        return draft

    def update_draft_token_ids(value):
        calls.append(("update", value))

    core = module.EngineCore()
    core.vllm_config = SimpleNamespace(
        additional_config={"hcu": {"enable_multi_layers_mtp": True}},
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )
    core.async_scheduling = False
    core.use_spec_decode = True
    core.model_executor = SimpleNamespace(
        take_draft_token_ids=take_draft_token_ids,
    )
    core.scheduler = SimpleNamespace(
        update_draft_token_ids=update_draft_token_ids,
    )
    assert core.post_step(True) == "official"
    assert calls == [
        ("post", True),
        ("take",),
        ("update", draft),
    ]


def _fake_output_processor_module(calls):
    class OutputProcessor:
        def process_outputs(
            self,
            engine_core_outputs,
            engine_core_timestamp=None,
            iteration_stats=None,
        ):
            calls.append("official")
            for output in engine_core_outputs:
                self.request_states[output.request_id].is_prefilling = False
            return "result"

    return _module(patch_output_processor.TARGET_MODULE, OutputProcessor=OutputProcessor)


def test_output_processor_mooncake_import_is_feature_lazy(monkeypatch):
    calls = []
    module = _fake_output_processor_module(calls)
    patch_output_processor.apply_to_module(module)
    processor = module.OutputProcessor()
    processor.request_states = {"r": SimpleNamespace(is_prefilling=True)}
    output = SimpleNamespace(request_id="r", kv_transfer_params={})
    monkeypatch.setattr(patch_output_processor.henvs, "VLLM_HCU_MOONCAKE_TTFT_TRACE", False)
    assert processor.process_outputs([output]) == "result"
    assert calls == ["official"]


def test_output_processor_logs_first_decoder_token_once(monkeypatch):
    calls = []
    module = _fake_output_processor_module(calls)
    patch_output_processor.apply_to_module(module)
    processor = module.OutputProcessor()
    processor.request_states = {"r": SimpleNamespace(is_prefilling=True)}
    output = SimpleNamespace(request_id="r", kv_transfer_params={"transfer_id": "x"})
    trace_module = _module(
        patch_mooncake_connector.TARGET_MODULE,
        log_ttft_event=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setitem(sys.modules, trace_module.__name__, trace_module)
    monkeypatch.setattr(patch_output_processor.henvs, "VLLM_HCU_MOONCAKE_TTFT_TRACE", True)
    processor.process_outputs([output])
    processor.process_outputs([output])
    trace_calls = [call for call in calls if isinstance(call, tuple)]
    assert len(trace_calls) == 1
    assert trace_calls[0][0] == ("d_first_token",)


def test_hcu_multiproc_executor_sizes_message_queue_from_v0251_config(monkeypatch):
    from vllm_hcu.v1.executor import multiproc_executor as hcu_executor

    records = []

    class FakeMessageQueue:
        def __init__(self, *args, max_chunks=10, **kwargs):
            records.append(max_chunks)

        @classmethod
        def create_from_handle(cls, handle, rank):
            records.append(("handle", handle, rank))
            return cls(1, 1)

    def fake_parent_init(self):
        hcu_executor._upstream.MessageQueue(4, 4, max_chunk_bytes=1)
        hcu_executor._upstream.MessageQueue(
            4, 4, max_chunk_bytes=1, max_chunks=8
        )
        hcu_executor._upstream.MessageQueue(
            4, 4, max_chunk_bytes=1, max_chunks=20
        )
        with pytest.raises(TypeError, match="max_chunks must be an integer"):
            hcu_executor._upstream.MessageQueue(
                4, 4, max_chunk_bytes=1, max_chunks=None
            )
        hcu_executor._upstream.MessageQueue(1, 1)
        hcu_executor._upstream.MessageQueue.create_from_handle("response", 0)

    monkeypatch.setattr(hcu_executor._upstream, "MessageQueue", FakeMessageQueue)
    monkeypatch.setattr(
        hcu_executor._upstream.MultiprocExecutor,
        "_init_executor",
        fake_parent_init,
    )
    executor = object.__new__(hcu_executor.HcuMultiprocExecutor)
    executor.vllm_config = SimpleNamespace(max_concurrent_batches=3)
    executor._init_executor()
    assert records == [12, 12, 20, 10, ("handle", "response", 0), 10]
    assert hcu_executor._upstream.MessageQueue is FakeMessageQueue
    monkeypatch.setattr(patch_multiproc_executor, "apply", lambda module=None: True)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(distributed_executor_backend="mp")
    )
    assert patch_multiproc_executor.select_hcu_multiproc_executor(config) is True
    assert config.parallel_config.distributed_executor_backend == (
        patch_multiproc_executor.HCU_MULTIPROC_EXECUTOR_PATH
    )
    assert patch_multiproc_executor.select_hcu_multiproc_executor(config) is False


def test_hcu_multiproc_executor_restores_queue_after_parent_failure(monkeypatch):
    from vllm_hcu.v1.executor import multiproc_executor as hcu_executor

    class FakeMessageQueue:
        pass

    def fake_parent_init(self):
        raise LookupError("parent init failed")

    monkeypatch.setattr(hcu_executor._upstream, "MessageQueue", FakeMessageQueue)
    monkeypatch.setattr(
        hcu_executor._upstream.MultiprocExecutor,
        "_init_executor",
        fake_parent_init,
    )
    executor = object.__new__(hcu_executor.HcuMultiprocExecutor)
    executor.vllm_config = SimpleNamespace(max_concurrent_batches=3)

    with pytest.raises(LookupError, match="parent init failed"):
        executor._init_executor()
    assert hcu_executor._upstream.MessageQueue is FakeMessageQueue
    assert hcu_executor._FORK_ORIGINAL_MESSAGE_QUEUE is None
    assert hcu_executor._FORK_PROXY_MESSAGE_QUEUE is None


def test_hcu_multiproc_executor_detects_concurrent_queue_replacement(monkeypatch):
    from vllm_hcu.v1.executor import multiproc_executor as hcu_executor

    class FakeMessageQueue:
        pass

    class ThirdPartyMessageQueue:
        pass

    def fake_parent_init(self):
        hcu_executor._upstream.MessageQueue = ThirdPartyMessageQueue

    monkeypatch.setattr(hcu_executor._upstream, "MessageQueue", FakeMessageQueue)
    monkeypatch.setattr(
        hcu_executor._upstream.MultiprocExecutor,
        "_init_executor",
        fake_parent_init,
    )
    executor = object.__new__(hcu_executor.HcuMultiprocExecutor)
    executor.vllm_config = SimpleNamespace(max_concurrent_batches=3)

    with pytest.raises(RuntimeError, match="replaced concurrently"):
        executor._init_executor()
    assert hcu_executor._upstream.MessageQueue is ThirdPartyMessageQueue
    assert hcu_executor._FORK_ORIGINAL_MESSAGE_QUEUE is None
    assert hcu_executor._FORK_PROXY_MESSAGE_QUEUE is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
def test_hcu_multiproc_executor_restores_message_queue_in_fork_child(monkeypatch):
    from vllm_hcu.v1.executor import multiproc_executor as hcu_executor

    class FakeMessageQueue:
        def __init__(self, *args, **kwargs):
            pass

    child_result: list[bytes] = []

    def fake_parent_init(self):
        assert hcu_executor._upstream.MessageQueue is not FakeMessageQueue
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # pragma: no cover - assertions are reported by the parent
            try:
                restored = hcu_executor._upstream.MessageQueue is FakeMessageQueue
                os.write(write_fd, b"restored" if restored else b"leaked")
            finally:
                os._exit(0)
        os.close(write_fd)
        child_result.append(os.read(read_fd, 32))
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0

    monkeypatch.setattr(hcu_executor._upstream, "MessageQueue", FakeMessageQueue)
    monkeypatch.setattr(
        hcu_executor._upstream.MultiprocExecutor,
        "_init_executor",
        fake_parent_init,
    )
    executor = object.__new__(hcu_executor.HcuMultiprocExecutor)
    executor.vllm_config = SimpleNamespace(max_concurrent_batches=3)
    executor._init_executor()

    assert child_result == [b"restored"]
    assert hcu_executor._upstream.MessageQueue is FakeMessageQueue


def test_outputs_keep_model_runner_ipc_stable_and_use_draft_channel():
    @dataclass
    class ModelRunnerOutput:
        req_ids: list[str]
        req_id_to_index: dict[str, int]

    @dataclass
    class DraftTokenIds:
        req_ids: list[str]
        draft_token_ids: list[list[int]]

    module = _module(
        patch_outputs.TARGET_MODULE,
        ModelRunnerOutput=ModelRunnerOutput,
        DraftTokenIds=DraftTokenIds,
        EMPTY_MODEL_RUNNER_OUTPUT=ModelRunnerOutput([], {}),
    )
    assert patch_outputs.apply_to_module(module) is True
    assert not hasattr(ModelRunnerOutput([], {}), "spec_token_ids")
    assert DraftTokenIds(["r"], [[1]]).draft_token_ids == [[1]]
    assert patch_outputs.apply_to_module(module) is False


def test_clean_v0251_model_runner_output_and_hcu_draft_method_contract():
    repo = Path(__file__).resolve().parents[2]
    clean_vllm = Path(
        os.environ.get("VLLM_V0251_SOURCE_ROOT", repo.parent / "vllm_0251")
    )
    script = r'''
import ast
import os
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import vllm
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

assert "spec_token_ids" not in {field.name for field in fields(ModelRunnerOutput)}
output = ModelRunnerOutput(
    req_ids=["r1"],
    req_id_to_index={"r1": 0},
    sampled_token_ids=[[7]],
)
assert output.sampled_token_ids == [[7]]
try:
    ModelRunnerOutput(
        req_ids=["r1"],
        req_id_to_index={"r1": 0},
        spec_token_ids=[[11, 12]],
    )
except TypeError:
    pass
else:
    raise AssertionError("clean ModelRunnerOutput accepted retired IPC field")

runner_source = Path(os.environ["HCU_RUNNER_SOURCE"])
tree = ast.parse(runner_source.read_text(encoding="utf-8"))
runner_class = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner"
)
take_method = next(
    node
    for node in runner_class.body
    if isinstance(node, ast.FunctionDef) and node.name == "take_draft_token_ids"
)
module = ast.Module(body=[take_method], type_ignores=[])
ast.fix_missing_locations(module)
namespace = {"DraftTokenIds": DraftTokenIds}
exec(compile(module, str(runner_source), "exec"), namespace)

runner = SimpleNamespace(
    num_spec_tokens=2,
    _draft_token_req_ids=["r1"],
    _get_draft_token_ids_cpu=lambda: ([[11, 12]], ["r1"]),
)
draft = namespace["take_draft_token_ids"](runner)
assert draft == DraftTokenIds(req_ids=["r1"], draft_token_ids=[[11, 12]])
runner.num_spec_tokens = 0
assert namespace["take_draft_token_ids"](runner) is None

bad_keywords = [
    keyword.arg
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "ModelRunnerOutput"
    for keyword in node.keywords
    if keyword.arg == "spec_token_ids"
]
assert bad_keywords == []
assert "output_spec_token_ids" not in {
    node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
}
print("CLEAN_OUTPUT_DRAFT_CHANNEL_OK", vllm.__file__)
'''
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = "__disabled__"
    env["HCU_RUNNER_SOURCE"] = str(repo / "vllm_hcu/v1/hcu_model_runner.py")
    env["PYTHONPATH"] = os.pathsep.join((str(clean_vllm), str(repo)))
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CLEAN_OUTPUT_DRAFT_CHANNEL_OK" in result.stdout
    assert str(clean_vllm / "vllm/__init__.py") in result.stdout


def _run_clean_vllm_real_factory_and_runtime_contract_smoke():
    repo = Path(__file__).resolve().parents[2]
    clean_vllm = Path(
        os.environ.get("VLLM_V0251_SOURCE_ROOT", repo.parent / "vllm_0251")
    )
    script = """
from vllm_hcu.patch.platform.framework_opt import (
    patch_kv_connector_factory, patch_outputs,
)
for adapter in (
    patch_kv_connector_factory, patch_outputs,
):
    assert adapter.apply() is True
    assert adapter.apply() is False
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
assert getattr(KVConnectorFactory._registry['MooncakeConnector'], '_hcu_connector_target') == (
    'vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector',
    'MooncakeConnector',
)
print('platform-framework-real-smoke-ok')
"""
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = "__disabled__"
    env["PYTHONPATH"] = os.pathsep.join((str(repo), str(clean_vllm)))
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "platform-framework-real-smoke-ok" in result.stdout
