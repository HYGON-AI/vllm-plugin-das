# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
import contextlib
import dataclasses
import enum
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_hcu.patch.config import HcuFeatureConfig
from vllm_hcu.patch.import_coordinator import ExactImportCoordinator
from vllm_hcu.patch.runtime_state import PatchRegistry
from vllm_hcu.v1 import worker_framework_runtime
from vllm_hcu.patch.worker.framework_opt import (
    patch_all2all,
    patch_base_device_communicator,
    patch_cuda_communicator,
    patch_dp_utils,
    patch_eagle_utils,
    patch_forward_context,
    patch_gpu_ubatch_wrapper,
    patch_llm_base_proposer,
    patch_pynccl,
    patch_pynccl_wrapper,
    patch_ubatch_utils,
)


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _config(**updates: object) -> SimpleNamespace:
    values = HcuFeatureConfig(**updates).to_dict()
    return SimpleNamespace(additional_config={"hcu": values})


def test_hcu_downstream_config_uses_sidecar_not_upstream_only_fields():
    paths = (
        "vllm_hcu/v1/hcu_model_runner.py",
        "vllm_hcu/model_executor/layers/sp_utils.py",
        "vllm_hcu/models/deepseek_v2.py",
        "vllm_hcu/models/glm4_moe.py",
        "vllm_hcu/models/hy_v3.py",
    )
    forbidden = (
        ".parallel_config.enable_lightly_cp",
        ".parallel_config.enable_lightly_cplb",
        '"enable_custom_sp", False',
        ".speculative_config.enable_multi_layers_mtp",
    )
    for path in paths:
        source = Path(path).read_text(encoding="utf-8-sig")
        assert "get_hcu_config" in source
        assert not any(fragment in source for fragment in forbidden)


def test_hcu_runner_uses_v0251_routed_experts_contract():
    source = Path("vllm_hcu/v1/hcu_model_runner.py").read_text(
        encoding="utf-8-sig"
    )
    removed_legacy_api = (
        "extract_routed_experts_for_current_batch",
        "free_routing_buffers",
        "get_global_experts_capturer",
        "init_routed_experts_capturer_with_shared_cache",
        "issue_routing_d2h_copy",
        "routed_experts_dict=",
    )
    assert not any(name in source for name in removed_legacy_api)
    required_v0251_api = (
        "RoutedExpertsCapturer",
        "RoutedExpertsLists",
        "RoutedExpertsTensors",
        "self.routed_experts_capturer.clear_buffer()",
        "self.routed_experts_slot_mapping_device",
        "routed_experts=routed_experts_snapshot",
    )
    assert all(name in source for name in required_v0251_api)


def test_hcu_runner_passes_request_phase_to_deepep_auto_selector():
    source = Path("vllm_hcu/v1/hcu_model_runner.py").read_text(
        encoding="utf-8-sig"
    )
    tree = ast.parse(source)

    def context_calls(method_name: str) -> list[ast.Call]:
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
        return [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "set_forward_context"
        ]

    execute_calls = context_calls("execute_model")
    dummy_calls = context_calls("_dummy_run")
    assert execute_calls and dummy_calls
    for call in (*execute_calls, *dummy_calls):
        assert "deepep_auto_is_prefilling" in {
            keyword.arg for keyword in call.keywords
        }


def test_hcu_runner_uses_v0251_kv_block_zeroer_constructor_contract():
    source = Path("vllm_hcu/v1/hcu_model_runner.py").read_text(
        encoding="utf-8-sig"
    )
    tree = ast.parse(source)
    init_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_init_kv_zero_meta"
    )
    constructor = next(
        node
        for node in ast.walk(init_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "KVBlockZeroer"
    )
    assert {keyword.arg for keyword in constructor.keywords} == {
        "pin_memory",
        "attn_groups_iter",
        "kernel_block_sizes",
        "cache_dtype",
        "runner_only_attn_layers",
        "static_forward_context",
    }
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "init_meta"
        for node in ast.walk(init_method)
    )


@pytest.mark.parametrize(
    "path,class_name",
    (
        (
            "vllm_hcu/v1/attention/backends/flash_attn.py",
            "HcuFlashAttentionBackend",
        ),
        (
            "vllm_hcu/v1/attention/backends/mla/flashmla.py",
            "HcuFlashMLABackend",
        ),
    ),
)
def test_hcu_attention_backend_uses_v0251_combination_signature(path, class_name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "supports_combination"
    )
    assert [argument.arg for argument in method.args.args] == [
        "cls",
        "head_size",
        "dtype",
        "kv_cache_dtype",
        "block_size",
        "use_mla",
        "has_sink",
        "use_sparse",
        "use_mm_prefix",
        "device_capability",
    ]


def test_hcu_runner_uses_v0251_input_batch_constructor_contract():
    import ast

    source = Path("vllm_hcu/v1/hcu_model_runner.py").read_text(
        encoding="utf-8-sig"
    )
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "InputBatch"
    ]
    assert len(calls) == 2
    for call in calls:
        assert "pin_memory" not in {keyword.arg for keyword in call.keywords}


def test_hcu_runner_uses_v0251_uniform_kv_cache_contract():
    source = Path("vllm_hcu/v1/hcu_model_runner.py").read_text(
        encoding="utf-8-sig"
    )
    assert "self.use_uniform_kv_cache(self.attn_groups)" in source
    assert "self.use_uniform_kv_cache(self.attn_groups, cache_dtype)" not in source


def _fake_all2all_module() -> ModuleType:
    class DeepEPAll2AllManagerBase:
        def __init__(self, cpu_group, tcp_store_group=None):
            self.cpu_group = cpu_group
            self.tcp_store_group = tcp_store_group
            self.internode = bool(tcp_store_group)
            self.num_sms = 20

    class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
        def _make_all2all_kwargs(self):
            return {"upstream": True}

        def set_num_sms(self, num_sms: int):
            self.applied_sms = min(num_sms, self.num_sms)

    class DeepEPLLAll2AllManager(DeepEPAll2AllManagerBase):
        def __init__(self, cpu_group, tcp_store_group=None):
            super().__init__(cpu_group, tcp_store_group)
            self.support_fault_tolerance = False

        def _make_all2all_kwargs(
            self,
            max_num_tokens_per_dp_rank,
            token_hidden_size,
            num_ep_ranks,
            num_global_experts,
            num_local_experts,
        ):
            return {"upstream": True}

    return _module(
        patch_all2all.TARGET_MODULE,
        DeepEPAll2AllManagerBase=DeepEPAll2AllManagerBase,
        DeepEPHTAll2AllManager=DeepEPHTAll2AllManager,
        DeepEPLLAll2AllManager=DeepEPLLAll2AllManager,
        envs=SimpleNamespace(
            VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE=False,
            VLLM_DEEPEP_BUFFER_SIZE_MB=256,
            VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=False,
        ),
    )


def test_deep_ep_adapter_uses_hcu_buffer_sms_contract_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.platforms import envs as hcu_envs

    module = _fake_all2all_module()
    monkeypatch.setattr(hcu_envs, "VLLM_HCU_DEEPEP_NUM_SMS", 17)
    assert patch_all2all.apply_to_module(module) is True
    assert patch_all2all.apply_to_module(module) is False

    manager = module.DeepEPHTAll2AllManager("group", "tcp")
    assert manager.num_sms == 30
    kwargs = manager._make_all2all_kwargs()
    assert kwargs["num_nvl_bytes"] == 1_000_000_000
    assert kwargs["num_rdma_bytes"] == 500_000_000
    assert kwargs["num_qps_per_rank"] == 30
    manager.set_num_sms(29)
    assert manager.applied_sms == 17
    # The legacy contract replaces the call argument with the environment
    # override first, then preserves vLLM's upper bound by ``self.num_sms``.
    monkeypatch.setattr(hcu_envs, "VLLM_HCU_DEEPEP_NUM_SMS", 40)
    manager.set_num_sms(29)
    assert manager.applied_sms == 30
    intranode = module.DeepEPHTAll2AllManager("group")
    intranode_kwargs = intranode._make_all2all_kwargs()
    assert intranode.num_sms == 60
    assert intranode_kwargs["num_rdma_bytes"] == 0
    assert intranode_kwargs["num_qps_per_rank"] == 1


def test_deep_ep_auto_manager_sizes_one_buffer_for_ht_and_ll(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    class Buffer:
        @staticmethod
        def get_low_latency_rdma_size_hint(**kwargs):
            calls.append(kwargs)
            return kwargs["num_max_dispatch_tokens_per_rank"] * 23_437_500

    monkeypatch.setitem(sys.modules, "deep_ep", SimpleNamespace(Buffer=Buffer))
    module = _fake_all2all_module()
    assert patch_all2all.apply_to_module(module)

    manager = module.DeepEPAutoAll2AllManager("group", "tcp")
    manager.support_fault_tolerance = True
    kwargs = manager._make_all2all_kwargs(
        max_num_tokens_per_dp_rank=32,
        token_hidden_size=7168,
        num_ep_ranks=8,
        num_global_experts=256,
        num_local_experts=32,
    )
    assert manager.is_deepep_auto_manager is True
    assert manager.max_sms_used() == 48
    assert len(calls) == 32
    assert calls[0] == {
        "num_max_dispatch_tokens_per_rank": 1,
        "hidden": 7168,
        "num_ranks": 8,
        "num_experts": 256,
    }
    assert calls[-1] == {
        "num_max_dispatch_tokens_per_rank": 32,
        "hidden": 7168,
        "num_ranks": 8,
        "num_experts": 256,
    }
    assert kwargs == {
        "group": "group",
        "num_nvl_bytes": 1_000_000_000,
        "num_rdma_bytes": 750_000_000,
        "low_latency_mode": True,
        "num_qps_per_rank": 32,
        "allow_nvlink_for_low_latency_mode": True,
        "allow_mnnvl": False,
        "explicitly_destroy": True,
        "enable_shrink": True,
    }
    assert manager._make_all2all_kwargs(
        max_num_tokens_per_dp_rank=32,
        token_hidden_size=7168,
        num_ep_ranks=8,
        num_global_experts=256,
        num_local_experts=32,
    ) == kwargs
    assert len(calls) == 32


def test_deep_ep_low_latency_rejects_first_invalid_model_specific_hint(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    class Buffer:
        @staticmethod
        def get_low_latency_rdma_size_hint(**kwargs):
            calls.append(kwargs)
            tokens = kwargs["num_max_dispatch_tokens_per_rank"]
            required_bytes = (
                tokens
                * kwargs["hidden"]
                * kwargs["num_ranks"]
                * kwargs["num_experts"]
            )
            glm_test_limit = 3 * 6144 * 8 * 256
            if required_bytes > glm_test_limit:
                return sys.maxsize + 1
            return required_bytes

    monkeypatch.setitem(sys.modules, "deep_ep", SimpleNamespace(Buffer=Buffer))
    module = _fake_all2all_module()
    assert patch_all2all.apply_to_module(module)
    manager = module.DeepEPLLAll2AllManager("group", "tcp")

    with pytest.raises(
        ValueError,
        match=(
            r"DeepEP low-latency RDMA size hint overflowed at token capacity 4"
            r".*hidden_size=6144.*num_global_experts=256.*ep_size=8"
        ),
    ):
        manager._make_all2all_kwargs(
            max_num_tokens_per_dp_rank=512,
            token_hidden_size=6144,
            num_ep_ranks=8,
            num_global_experts=256,
            num_local_experts=32,
        )

    assert [
        call["num_max_dispatch_tokens_per_rank"] for call in calls
    ] == [1, 2, 3, 4]
    assert all(call["hidden"] == 6144 for call in calls)
    assert all(call["num_ranks"] == 8 for call in calls)
    assert all(call["num_experts"] == 256 for call in calls)

    calls.clear()
    kwargs = manager._make_all2all_kwargs(
        max_num_tokens_per_dp_rank=512,
        token_hidden_size=256,
        num_ep_ranks=2,
        num_global_experts=16,
        num_local_experts=8,
    )
    assert kwargs["num_rdma_bytes"] == 512 * 256 * 2 * 16
    assert len(calls) == 512
    assert calls[-1] == {
        "num_max_dispatch_tokens_per_rank": 512,
        "hidden": 256,
        "num_ranks": 2,
        "num_experts": 16,
    }
    assert manager._make_all2all_kwargs(
        max_num_tokens_per_dp_rank=512,
        token_hidden_size=256,
        num_ep_ranks=2,
        num_global_experts=16,
        num_local_experts=8,
    ) == kwargs
    assert len(calls) == 512


def test_deep_ep_low_latency_rejects_decreasing_hint_and_wraps_native_error(
    monkeypatch: pytest.MonkeyPatch,
):
    class Buffer:
        @staticmethod
        def get_low_latency_rdma_size_hint(**kwargs):
            tokens = kwargs["num_max_dispatch_tokens_per_rank"]
            if tokens == 3:
                return 50
            return tokens * 100

    monkeypatch.setitem(sys.modules, "deep_ep", SimpleNamespace(Buffer=Buffer))
    module = _fake_all2all_module()
    assert patch_all2all.apply_to_module(module)
    manager = module.DeepEPLLAll2AllManager("group", "tcp")

    with pytest.raises(
        ValueError,
        match=r"overflowed at token capacity 3.*previous_hint=200, current_hint=50",
    ):
        manager._make_all2all_kwargs(5, 6144, 8, 256, 32)

    class FailingBuffer:
        @staticmethod
        def get_low_latency_rdma_size_hint(**kwargs):
            if kwargs["num_max_dispatch_tokens_per_rank"] == 2:
                raise OverflowError("native overflow")
            return 100

    monkeypatch.setitem(
        sys.modules, "deep_ep", SimpleNamespace(Buffer=FailingBuffer)
    )
    second_manager = module.DeepEPLLAll2AllManager("group", "tcp")
    with pytest.raises(
        ValueError,
        match=(
            r"RDMA size hint failed at token capacity 2.*hidden_size=6144"
            r".*num_global_experts=256.*ep_size=8"
        ),
    ) as exc_info:
        second_manager._make_all2all_kwargs(5, 6144, 8, 256, 32)
    assert isinstance(exc_info.value.__cause__, OverflowError)


def _fake_base_communicator_module() -> ModuleType:
    class DeviceCommunicatorBase:
        def __init__(
            self,
            cpu_group,
            device=None,
            device_group=None,
            unique_name="",
            global_ranks=None,
            global_world_size=None,
            use_all2all=False,
        ):
            self.device_group = device_group
            self.is_ep_communicator = unique_name.split(":")[0] == "ep"
            self.use_all2all = use_all2all

        def reduce_scatter(self, input_, dim=-1):
            return input_

    return _module(
        patch_base_device_communicator.TARGET_MODULE,
        DeviceCommunicatorBase=DeviceCommunicatorBase,
    )


def test_base_communicator_reads_custom_sp_sidecar_and_uses_torch_collective(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.config

    config = _config(enable_custom_sp=True)
    config.parallel_config = SimpleNamespace(data_parallel_size=1)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    calls: list[tuple[object, object, object]] = []
    monkeypatch.setattr(
        torch.distributed,
        "all_to_all_single",
        lambda output, input_, group=None: calls.append((output, input_, group)),
    )
    module = _fake_base_communicator_module()
    assert patch_base_device_communicator.apply_to_module(module)
    assert not patch_base_device_communicator.apply_to_module(module)
    communicator = module.DeviceCommunicatorBase(
        object(), device_group="device-group", unique_name="ep:0"
    )
    assert communicator.use_all2all is True
    output, input_ = object(), object()
    assert communicator.all_to_all_single(output, input_) is output
    assert calls == [(output, input_, "device-group")]


def test_cuda_communicator_registers_exact_exchange_and_marks_stale_removal_obsolete():
    class CudaCommunicator:
        def __init__(
            self,
            cpu_group,
            device=None,
            device_group=None,
            unique_name="",
            global_ranks=None,
            global_world_size=None,
            tcp_store_group=None,
            use_all2all=False,
        ):
            self.use_all2all = use_all2all

    module = _module(
        patch_cuda_communicator.TARGET_MODULE, CudaCommunicator=CudaCommunicator
    )
    coordinator = ExactImportCoordinator(registry=PatchRegistry())
    registration = patch_cuda_communicator.register(coordinator)
    assert registration.module_name == patch_cuda_communicator.CUSTOM_ALLREDUCE_MODULE
    assert patch_cuda_communicator.apply_to_module(module) is True
    assert patch_cuda_communicator.apply_to_module(module) is False
    assert "all_to_all_single" not in vars(CudaCommunicator)


def test_cuda_communicator_replaces_normalized_ll_manager_for_auto(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.config
    import vllm.distributed.device_communicators.all2all as all2all

    created: list[tuple[object, object]] = []

    class DeepEPAutoAll2AllManager:
        def __init__(self, cpu_group, tcp_store_group=None):
            created.append((cpu_group, tcp_store_group))

    monkeypatch.setattr(
        all2all,
        "DeepEPAutoAll2AllManager",
        DeepEPAutoAll2AllManager,
        raising=False,
    )
    config = _config(deepep_auto=True)
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config_or_none", lambda: config
    )

    class CudaCommunicator:
        def __init__(
            self,
            cpu_group,
            device=None,
            device_group=None,
            unique_name="",
            global_ranks=None,
            global_world_size=None,
            tcp_store_group=None,
            use_all2all=False,
        ):
            del device, device_group, unique_name, global_ranks, global_world_size
            self.cpu_group = cpu_group
            self.use_all2all = use_all2all
            self.all2all_manager = "normalized-low-latency"

    module = _module(
        patch_cuda_communicator.TARGET_MODULE,
        CudaCommunicator=CudaCommunicator,
    )
    assert patch_cuda_communicator.apply_to_module(module)
    communicator = module.CudaCommunicator(
        "cpu-group", tcp_store_group="tcp", use_all2all=True
    )
    assert isinstance(communicator.all2all_manager, DeepEPAutoAll2AllManager)
    assert created == [("cpu-group", "tcp")]


def test_cuda_communicator_auto_ignores_non_ep_collective_groups(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.config

    config = _config(deepep_auto=True)
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config_or_none", lambda: config
    )

    class CudaCommunicator:
        def __init__(
            self,
            cpu_group,
            device=None,
            device_group=None,
            unique_name="",
            global_ranks=None,
            global_world_size=None,
            tcp_store_group=None,
            use_all2all=False,
        ):
            del (
                device,
                device_group,
                global_ranks,
                global_world_size,
                tcp_store_group,
            )
            self.cpu_group = cpu_group
            self.use_all2all = use_all2all
            self.all2all_manager = "official-collective"

    module = _module(
        patch_cuda_communicator.TARGET_MODULE,
        CudaCommunicator=CudaCommunicator,
    )
    assert patch_cuda_communicator.apply_to_module(module)
    communicator = module.CudaCommunicator("cpu-group", unique_name="dp:0")
    assert communicator.all2all_manager == "official-collective"


@dataclasses.dataclass
class _Function:
    name: str
    restype: object
    argtypes: list[object]


def _fake_pynccl_wrapper_module(*, cached: bool = False) -> ModuleType:
    class NCCLLibrary:
        exported_functions: list[_Function] = []
        path_to_library_cache = {"lib": object()} if cached else {}
        path_to_dict_mapping = {}

        def __init__(self, so_file=None):
            self._funcs = {}

        def ncclSend(self, sendbuff, count, datatype, dest, comm, stream):
            return None

    return _module(
        patch_pynccl_wrapper.TARGET_MODULE,
        NCCLLibrary=NCCLLibrary,
        Function=_Function,
        ncclResult_t=object(),
        buffer_type=lambda value: value,
        ncclDataType_t=object(),
        ncclComm_t=object(),
        cudaStream_t=lambda value: value,
        find_nccl_library=lambda: "librccl.so",
    )


def test_pynccl_wrapper_capability_registration_precedes_library_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _fake_pynccl_wrapper_module()
    monkeypatch.setattr(
        patch_pynccl_wrapper,
        "_probe_rccl_symbol",
        lambda target: (True, "librccl.so"),
    )
    assert patch_pynccl_wrapper.apply_to_module(module, required=True)
    assert not patch_pynccl_wrapper.apply_to_module(module, required=True)
    assert [item.name for item in module.NCCLLibrary.exported_functions] == [
        "ncclAllToAll"
    ]
    calls: list[tuple[object, ...]] = []
    library = object.__new__(module.NCCLLibrary)
    library._funcs = {"ncclAllToAll": lambda *args: calls.append(args) or 0}
    library.NCCL_CHECK = lambda result: None
    library.ncclAllToAll(1, 2, 3, 4, 5, 6)
    assert calls == [(1, 2, 3, 4, 5, 6)]

    cached = _fake_pynccl_wrapper_module(cached=True)
    assert not patch_pynccl_wrapper.apply_to_module(cached)
    with pytest.raises(RuntimeError, match="explicitly requested"):
        patch_pynccl_wrapper.apply_to_module(cached, required=True)


def test_pynccl_communicator_method_is_capability_gated(
    monkeypatch: pytest.MonkeyPatch,
):
    class ReduceOp(enum.Enum):
        SUM = "sum"

    class PyNcclCommunicator:
        def reduce_scatter(
            self,
            output_tensor,
            input_tensor,
            op=ReduceOp.SUM,
            stream=None,
        ):
            return None

    module = _module(
        patch_pynccl.TARGET_MODULE,
        PyNcclCommunicator=PyNcclCommunicator,
        ReduceOp=ReduceOp,
        current_stream=lambda: SimpleNamespace(cuda_stream=99),
        buffer_type=lambda value: value,
        cudaStream_t=lambda value: value,
        ncclDataTypeEnum=SimpleNamespace(from_torch=lambda dtype: dtype),
    )
    wrapper = _fake_pynccl_wrapper_module()
    wrapper._vllm_hcu_pynccl_all_to_all_applied = True

    def nccl_all_to_all(*args):
        return None

    nccl_all_to_all._vllm_hcu_pynccl_all_to_all_wrapper = True
    wrapper.NCCLLibrary.ncclAllToAll = nccl_all_to_all
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.pynccl_wrapper",
        wrapper,
    )
    assert patch_pynccl.apply_to_module(module, required=True)
    assert not patch_pynccl.apply_to_module(module, required=True)

    class Tensor:
        def __init__(self, device="cuda:0", size=4):
            self.device = device
            self.dtype = "int8"
            self._size = size

        def numel(self):
            return self._size

        def data_ptr(self):
            return id(self)

    calls: list[tuple[object, ...]] = []
    communicator = object.__new__(PyNcclCommunicator)
    communicator.disabled = False
    communicator.device = "cuda:0"
    communicator.world_size = 2
    communicator.comm = "comm"
    communicator.nccl = SimpleNamespace(
        ncclAllToAll=lambda *args: calls.append(args)
    )
    input_tensor, output_tensor = Tensor(), Tensor()
    assert (
        communicator.all_to_all_single(output_tensor, input_tensor)
        is output_tensor
    )
    assert calls[0][2] == 2
    with pytest.raises(AssertionError, match="input tensor"):
        communicator.all_to_all_single(output_tensor, Tensor(device="cuda:1"))


class _Mode(enum.Enum):
    NONE = 0


def _fake_forward_context_module() -> ModuleType:
    @dataclasses.dataclass
    class ForwardContext:
        value: object

    def create_forward_context(
        attn_metadata,
        vllm_config,
        dp_metadata=None,
        cudagraph_runtime_mode=_Mode.NONE,
        batch_descriptor=None,
        ubatch_slices=None,
        slot_mapping=None,
        additional_kwargs=None,
        skip_compiled=False,
        is_padding=None,
    ):
        return ForwardContext(attn_metadata)

    @contextlib.contextmanager
    def set_forward_context(
        attn_metadata,
        vllm_config,
        num_tokens=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=_Mode.NONE,
        batch_descriptor=None,
        ubatch_slices=None,
        slot_mapping=None,
        skip_compiled=False,
        is_padding=None,
    ):
        yield "official"

    return _module(
        patch_forward_context.TARGET_MODULE,
        ForwardContext=ForwardContext,
        CUDAGraphMode=_Mode,
        create_forward_context=create_forward_context,
        set_forward_context=set_forward_context,
    )


def test_forward_context_keeps_dataclass_and_attaches_runtime_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _fake_forward_context_module()
    original_class = module.ForwardContext
    assert patch_forward_context.apply_to_module(module)
    assert not patch_forward_context.apply_to_module(module)
    config = SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(all2all_backend="naive")
    )
    context = module.create_forward_context(
        "metadata",
        config,
        scatter_indexes_tensor="scatter",
        gather_indexes_tensor="gather",
        enable_lightly_cp=True,
        enable_lightly_cplb=True,
    )
    assert type(context) is original_class
    assert dataclasses.is_dataclass(context)
    assert context.scatter_indexes_tensor == "scatter"
    assert context.gather_indexes_tensor == "gather"
    assert context.enable_lightly_cp is True

    sentinel = contextlib.nullcontext("hcu")
    monkeypatch.setattr(
        "vllm_hcu.forward_context_runtime.set_forward_context",
        lambda *args, **kwargs: sentinel,
    )
    config.parallel_config.all2all_backend = "deepep_low_latency"
    with module.set_forward_context(None, config) as value:
        assert value == "hcu"


def test_deepep_auto_forward_mode_requires_decode_phase_evidence():
    from vllm_hcu.forward_context_runtime import (
        choose_deepep_auto_low_latency,
    )

    config = _config(deepep_auto=True)
    config.scheduler_config = SimpleNamespace(max_num_seqs=8)
    config.speculative_config = SimpleNamespace(num_speculative_tokens=3)

    assert not choose_deepep_auto_low_latency(
        config, 512, None, SimpleNamespace(uniform=True)
    )
    from vllm_hcu.forward_context_runtime import (
        deepep_auto_request_phase_scope,
        set_deepep_auto_request_phase,
    )

    with deepep_auto_request_phase_scope():
        set_deepep_auto_request_phase(torch.tensor([False]))
        assert choose_deepep_auto_low_latency(
            config,
            1,
            None,
            SimpleNamespace(uniform=True),
            SimpleNamespace(max_query_len=1, max_seq_len=100),
        )
    assert choose_deepep_auto_low_latency(
        config,
        1,
        None,
        SimpleNamespace(uniform=True),
        SimpleNamespace(max_query_len=1, max_seq_len=100),
        torch.tensor([False]),
    )
    assert not choose_deepep_auto_low_latency(
        config, 1, None, SimpleNamespace(uniform=False)
    )
    assert not choose_deepep_auto_low_latency(config, 32, None, None)
    assert not choose_deepep_auto_low_latency(config, 33, None, None)
    assert not choose_deepep_auto_low_latency(
        config, None, torch.tensor([4, 32, 16]), None
    )
    assert not choose_deepep_auto_low_latency(
        config, None, torch.tensor([4, 33, 16]), None
    )


def test_dspark_deepep_auto_uses_ht_for_prefill_and_ll_for_uniform_decode():
    from vllm_hcu.forward_context_runtime import choose_deepep_auto_low_latency

    config = _config(deepep_auto=True)
    config.scheduler_config = SimpleNamespace(max_num_seqs=8)
    config.speculative_config = SimpleNamespace(
        method="dspark",
        num_speculative_tokens=7,
    )

    assert not choose_deepep_auto_low_latency(
        config,
        512,
        None,
        SimpleNamespace(uniform=False),
    )
    assert not choose_deepep_auto_low_latency(
        config,
        8,
        None,
        SimpleNamespace(uniform=True),
    )
    assert not choose_deepep_auto_low_latency(
        config,
        2,
        None,
        SimpleNamespace(uniform=True),
        SimpleNamespace(max_query_len=2, max_seq_len=100),
        torch.tensor([True]),
    )
    assert not choose_deepep_auto_low_latency(config, 64, None, None)
    assert not choose_deepep_auto_low_latency(config, 65, None, None)
    decode_metadata = {
        "model.layers.0.self_attn": SimpleNamespace(
            max_query_len=8,
            max_seq_len=128,
            is_prefilling=torch.tensor([False]),
        )
    }
    assert choose_deepep_auto_low_latency(
        config,
        256,
        None,
        SimpleNamespace(uniform=False),
        decode_metadata,
        torch.tensor([False]),
    )
    prefill_metadata = SimpleNamespace(
        max_query_len=8,
        max_seq_len=8,
        is_prefilling=torch.tensor([True]),
    )
    assert not choose_deepep_auto_low_latency(
        config,
        8,
        None,
        SimpleNamespace(uniform=False),
        prefill_metadata,
        torch.tensor([True]),
    )

    mixed_metadata = {
        "prefill": SimpleNamespace(
            max_query_len=2,
            max_seq_len=100,
            is_prefilling=torch.tensor([True]),
        ),
        "decode": SimpleNamespace(
            max_query_len=1,
            max_seq_len=128,
            is_prefilling=torch.tensor([False]),
        ),
    }
    assert not choose_deepep_auto_low_latency(
        config,
        16,
        None,
        SimpleNamespace(uniform=False),
        mixed_metadata,
        torch.tensor([True, False]),
    )
    assert not choose_deepep_auto_low_latency(
        config,
        2,
        None,
        SimpleNamespace(uniform=False),
        SimpleNamespace(max_query_len=2, max_seq_len=100),
        torch.tensor([True]),
    )
    assert not choose_deepep_auto_low_latency(
        config,
        2,
        None,
        SimpleNamespace(uniform=False),
        SimpleNamespace(max_query_len=2, max_seq_len=100),
    )
    assert choose_deepep_auto_low_latency(
        config,
        1,
        None,
        SimpleNamespace(uniform=False),
        SimpleNamespace(max_query_len=1, max_seq_len=100),
        torch.tensor([False]),
    )


def test_dspark_deepep_auto_uses_dp_global_phase_on_empty_ranks(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm_hcu.forward_context_runtime as runtime

    choose_deepep_auto_low_latency = runtime.choose_deepep_auto_low_latency

    config = _config(deepep_auto=True)
    config.parallel_config = SimpleNamespace(data_parallel_size=8)
    config.scheduler_config = SimpleNamespace(max_num_seqs=8)
    config.speculative_config = SimpleNamespace(
        method="dspark",
        num_speculative_tokens=7,
    )
    nonuniform = SimpleNamespace(uniform=False)
    decode_tokens = torch.tensor([8, 0, 0, 0, 0, 0, 0, 0])

    synchronized_phase = True
    local_evidence: list[tuple[bool, bool]] = []

    def synchronize_phase(
        _config: object,
        *,
        local_active: bool,
        local_decode: bool,
    ) -> bool:
        local_evidence.append((local_active, local_decode))
        return synchronized_phase

    monkeypatch.setattr(
        runtime,
        "_synchronize_deepep_auto_phase",
        synchronize_phase,
        raising=False,
    )

    # The active and seven empty ranks must select the same DeepEP collective.
    assert choose_deepep_auto_low_latency(
        config,
        8,
        decode_tokens,
        nonuniform,
        SimpleNamespace(max_query_len=8, max_seq_len=128),
        torch.tensor([False]),
    )
    assert choose_deepep_auto_low_latency(
        config,
        0,
        decode_tokens,
        nonuniform,
        None,
    )
    assert local_evidence == [(True, True), (False, False)]

    synchronized_phase = False
    local_evidence.clear()
    short_prefill_tokens = torch.tensor([8, 0, 0, 0, 0, 0, 0, 0])
    assert not choose_deepep_auto_low_latency(
        config,
        8,
        short_prefill_tokens,
        nonuniform,
        SimpleNamespace(max_query_len=8, max_seq_len=8),
        torch.tensor([True]),
    )
    assert not choose_deepep_auto_low_latency(
        config,
        0,
        short_prefill_tokens,
        nonuniform,
        None,
    )
    assert local_evidence == [(True, False), (False, False)]


@pytest.mark.parametrize(
    ("kv_role", "expected"),
    [("kv_producer", False), ("kv_consumer", True)],
)
def test_dspark_mooncake_pd_role_skips_dynamic_phase_collective(
    monkeypatch: pytest.MonkeyPatch,
    kv_role: str,
    expected: bool,
):
    import vllm_hcu.forward_context_runtime as runtime

    config = _config(deepep_auto=True)
    config.parallel_config = SimpleNamespace(data_parallel_size=4)
    config.speculative_config = SimpleNamespace(
        method="dspark",
        num_speculative_tokens=7,
    )
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="MooncakeConnector",
        kv_role=kv_role,
    )
    config.model_config = SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"]
    )
    monkeypatch.setattr(
        runtime,
        "_synchronize_deepep_auto_phase",
        lambda *args, **kwargs: pytest.fail(
            "role-fixed Mooncake P/D must not add a DP collective"
        ),
    )

    assert (
        runtime.choose_deepep_auto_low_latency(
            config,
            8,
            torch.tensor([8, 0, 0, 0]),
            SimpleNamespace(uniform=expected),
        )
        is expected
    )


def test_dp_coordination_deepep_low_latency_and_feature_off_delegation():
    calls: list[tuple[object, ...]] = []

    def coordinate_batch_across_dp(
        num_tokens_unpadded,
        allow_microbatching,
        parallel_config,
        num_tokens_padded=None,
        uniform_decode=None,
        cudagraph_mode=0,
    ):
        calls.append((num_tokens_unpadded, parallel_config))
        return True, "tokens", 2

    module = _module(
        patch_dp_utils.TARGET_MODULE,
        coordinate_batch_across_dp=coordinate_batch_across_dp,
    )
    patch_dp_utils.apply_to_module(module)
    low_latency = SimpleNamespace(
        data_parallel_size=4, all2all_backend="deepep_low_latency"
    )
    assert module.coordinate_batch_across_dp(4, False, low_latency) == (
        False,
        None,
        0,
    )
    normal = SimpleNamespace(data_parallel_size=4, all2all_backend="naive")
    assert module.coordinate_batch_across_dp(4, False, normal) == (
        True,
        "tokens",
        2,
    )
    assert calls == [(4, normal)]


class _Buffer:
    def __init__(self, size, **kwargs):
        self.size = size


def _fake_proposer_module() -> ModuleType:
    class SpecDecodeBaseProposer:
        def __init__(
            self,
            vllm_config,
            device,
            pass_hidden_states_to_model,
            runner=None,
        ):
            self.vllm_config = vllm_config
            self.compilation_config = vllm_config.compilation_config
            self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
            self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self.backup_next_token_ids = _Buffer(self.max_batch_size)
            self.arange = torch.arange(max(self.max_batch_size + 1, self.max_num_tokens))
            self.rocm_branch_initialized = True

        def propose(
            self,
            num_speculative_tokens,
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_token_ids,
            token_indices_to_sample,
            common_attn_metadata,
            sampling_metadata,
            mm_embed_inputs=None,
            num_rejected_tokens_gpu=None,
            slot_mappings=None,
        ):
            return "official"

        def prepare_inputs_padded(
            self,
            common_attn_metadata,
            spec_decode_metadata,
            valid_sampled_tokens_count,
        ):
            return SimpleNamespace(num_actual_tokens=7), None, None

        def _maybe_share_lm_head(self, target_language_model):
            return "official-share"

        def _determine_batch_execution_and_padding(
            self, num_tokens, use_cudagraphs=True
        ):
            return num_tokens, use_cudagraphs

    return _module(
        patch_llm_base_proposer.TARGET_MODULE,
        SpecDecodeBaseProposer=SpecDecodeBaseProposer,
        CpuGpuBuffer=_Buffer,
        is_pin_memory_available=lambda: False,
        torch=torch,
    )


def _proposer_config(**hcu: object) -> SimpleNamespace:
    config = _config(**hcu)
    config.scheduler_config = SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=6
    )
    config.parallel_config = SimpleNamespace(tensor_parallel_size=4)
    config.compilation_config = SimpleNamespace(
        pass_config=SimpleNamespace(enable_sp=False)
    )
    return config


def test_proposer_sidecar_init_cplb_fix_rocm_preservation_and_custom_sp_padding():
    module = _fake_proposer_module()
    assert patch_llm_base_proposer.apply_to_module(module)
    assert not patch_llm_base_proposer.apply_to_module(module)
    runner = SimpleNamespace(lightly_cp_threshold=8)
    config = _proposer_config(
        enable_lightly_cp=True,
        enable_lightly_cplb=True,
        enable_custom_sp=True,
        enable_multi_layers_mtp=True,
    )
    proposer = module.SpecDecodeBaseProposer(config, "cpu", True, runner)
    assert proposer.rocm_branch_initialized is True
    assert proposer.max_batch_size == 8
    assert proposer.backup_next_token_ids.size == 8
    assert proposer.query_start_loc.size == proposer.max_batch_size + 1
    assert proposer.seq_lens.size == proposer.max_batch_size
    assert proposer.enable_multi_layers_mtp is True
    assert proposer._pad_for_sequence_parallelism(5) == 8
    assert proposer._determine_batch_execution_and_padding(5) == (8, True)

    off = module.SpecDecodeBaseProposer(_proposer_config(), "cpu", True)
    assert off.propose(1, None, None, None, None, None, None, None) == "official"

    common = SimpleNamespace(num_kv_actual_tokens=5)
    prepared, _, _ = off.prepare_inputs_padded(common, None, None)
    assert prepared.num_actual_tokens == 7
    assert prepared.num_kv_actual_tokens == 5


def test_proposer_registers_hcu_spec_decode_metadata_once() -> None:
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseMetadata,
    )
    from vllm_hcu.v1.spec_decode import proposer_runtime
    from vllm_hcu.v1.attention.backends.flash_attn import (
        FlashAttentionMetadata,
    )

    proposer = SimpleNamespace(allowed_attn_types=(str,))
    config = _proposer_config()

    proposer_runtime.initialize_proposer(
        SimpleNamespace(), proposer, config, "cpu", None
    )
    proposer_runtime.initialize_proposer(
        SimpleNamespace(), proposer, config, "cpu", None
    )

    assert proposer.allowed_attn_types == (
        str,
        FlashMLASparseMetadata,
        FlashAttentionMetadata,
    )


def test_proposer_without_allowlist_does_not_import_optional_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from vllm_hcu.v1.spec_decode import proposer_runtime

    original_import = builtins.__import__

    def import_without_flash_attention(name, *args, **kwargs):
        if name in {
            "vllm.v1.attention.backends.mla.flashmla_sparse",
            "vllm_hcu.v1.attention.backends.flash_attn",
        }:
            raise AssertionError("optional attention metadata was imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_flash_attention)
    proposer = SimpleNamespace(allowed_attn_types=None)

    proposer_runtime.initialize_proposer(
        SimpleNamespace(), proposer, _proposer_config(), "cpu", None
    )

    assert proposer.allowed_attn_types is None


def test_proposer_allows_triton_fallback_when_flash_attn_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from vllm_hcu.v1.spec_decode import proposer_runtime

    original_import = builtins.__import__

    def import_without_flash_attention(name, *args, **kwargs):
        if name == "vllm_hcu.v1.attention.backends.flash_attn":
            error = ModuleNotFoundError("No module named 'flash_attn'")
            error.name = "flash_attn"
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_flash_attention)
    proposer = SimpleNamespace(allowed_attn_types=(str,))

    proposer_runtime.initialize_proposer(
        SimpleNamespace(), proposer, _proposer_config(), "cpu", None
    )

    assert proposer.allowed_attn_types[0] is str


def test_proposer_propagates_flash_attention_symbol_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from vllm_hcu.v1.spec_decode import proposer_runtime

    original_import = builtins.__import__

    def import_with_incompatible_flash_attention(name, *args, **kwargs):
        if name == "vllm_hcu.v1.attention.backends.flash_attn":
            raise ImportError("cannot import name 'hg_flash_attn_varlen_func'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(
        builtins,
        "__import__",
        import_with_incompatible_flash_attention,
    )

    with pytest.raises(ImportError, match="hg_flash_attn_varlen_func"):
        proposer_runtime.initialize_proposer(
            SimpleNamespace(),
            SimpleNamespace(allowed_attn_types=(str,)),
            _proposer_config(),
            "cpu",
            None,
        )


def test_proposer_lightly_cp_atomic_metadata_and_forward_context_chain(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.v1.spec_decode import proposer_runtime

    canonical = SimpleNamespace(
        num_actual_tokens=5,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
        _num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([1], dtype=torch.int32),
    )
    cp_metadata = SimpleNamespace(
        num_reqs=1,
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        scatter_indexes_tensor=torch.tensor([0]),
        gather_indexes_tensor=torch.tensor([0]),
        cp_common_metadata=canonical,
    )
    common = SimpleNamespace(
        num_reqs=1,
        max_query_len=5,
        seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 5], dtype=torch.int32),
        seq_lens=torch.tensor([5], dtype=torch.int32),
        _num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
        batch_size=lambda: 1,
    )
    events: list[tuple[str, object]] = []
    lightly_cp = _module(
        "vllm_hcu.v1.attention.lightly_cp_utils",
        pad_for_mla_cp=lambda value: 8,
        prepare_cp_metadata=lambda **kwargs: (
            events.append(("prepare", kwargs["num_tokens"])) or cp_metadata
        ),
    )
    monkeypatch.setitem(
        sys.modules, "vllm_hcu.v1.attention.lightly_cp_utils", lightly_cp
    )

    @contextlib.contextmanager
    def set_forward_context(*args, **kwargs):
        events.append(("context", kwargs))
        yield

    module = SimpleNamespace(torch=torch, set_forward_context=set_forward_context)

    class Model:
        def __call__(self, **kwargs):
            events.append(("model", kwargs))
            return torch.ones(1, 2)

    def build_metadata(metadata, draft_index=None):
        events.append(("metadata", metadata))
        return [metadata], {"layer": metadata}

    proposer = SimpleNamespace(
        method="mtp",
        model=Model(),
        hidden_size=2,
        set_inputs_first_pass=lambda **kwargs: (
            5,
            torch.tensor([0]),
            common,
        ),
        runner=SimpleNamespace(lightly_cp_threshold=1),
        enable_lightly_cp=True,
        enable_lightly_cplb=False,
        query_start_loc=object(),
        seq_lens=object(),
        build_per_group_and_layer_attn_metadata=build_metadata,
        _determine_batch_execution_and_padding=lambda value: (
            "mode",
            value,
            None,
        ),
        build_model_inputs_first_pass=lambda num_tokens, num_input_tokens, mm: (
            {},
            1,
        ),
        vllm_config=object(),
        _get_slot_mapping=lambda *args: {"slot": args},
        model_returns_tuple=lambda: False,
        _greedy_sample=lambda hidden: torch.tensor([42]),
        num_speculative_tokens=1,
        parallel_drafting=False,
    )
    result = proposer_runtime.propose(
        module,
        proposer,
        1,
        torch.arange(5),
        torch.arange(5),
        torch.ones(5, 2),
        torch.tensor([1]),
        torch.tensor([0]),
        common,
        object(),
    )
    assert result.tolist() == [[42]]
    assert events[0] == ("prepare", 5)
    assert events[1] == ("metadata", cp_metadata)
    context_kwargs = next(value for name, value in events if name == "context")
    assert context_kwargs["scatter_indexes_tensor"] is cp_metadata.scatter_indexes_tensor
    assert context_kwargs["gather_indexes_tensor"] is cp_metadata.gather_indexes_tensor
    assert context_kwargs["enable_lightly_cp"] is True

    # A second draft step must switch back from the rank-local CP view to the
    # canonical metadata carried by cp_common_metadata.
    events.clear()
    proposer.num_speculative_tokens = 2
    proposer.allowed_attn_types = None
    proposer.uses_mrope = False
    proposer.positions = torch.zeros(8, dtype=torch.int64)
    proposer.constant_draft_positions = False
    proposer.block_size = 1
    proposer.input_ids = torch.zeros(8, dtype=torch.int32)
    proposer.hidden_states = torch.zeros(8, 2)
    proposer.supports_mm_inputs = False
    proposer.pass_hidden_states_to_model = False
    proposer.arange = torch.arange(9, dtype=torch.int32)
    proposer.token_arange_np = np.arange(9)
    proposer._get_positions = lambda size: proposer.positions[:size]
    proposer._update_positions_dependent_metadata = (
        lambda positions, metadata, *args: (
            events.append(("canonical", metadata)) or positions
        )
    )
    result = proposer_runtime.propose(
        module,
        proposer,
        2,
        torch.arange(5),
        torch.arange(5),
        torch.ones(5, 2),
        torch.tensor([1]),
        torch.tensor([0]),
        common,
        object(),
    )
    assert result.tolist() == [[42, 42]]
    assert ("canonical", canonical) in events
    assert ("metadata", canonical) in events


def test_multi_layer_mtp_preserves_distinct_trained_head():
    from vllm_hcu.v1.spec_decode.proposer_runtime import (
        preserve_multi_layer_mtp_heads,
    )

    trained = SimpleNamespace(weight=torch.tensor([[2.0]]))
    shared = SimpleNamespace(head=trained)
    layer = SimpleNamespace(shared_head=shared)
    proposer = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    )
    target = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.tensor([[1.0]])))

    def official(self, target_language_model):
        self.model.model.layers[0].shared_head.head = target_language_model.lm_head

    preserve_multi_layer_mtp_heads(proposer, target, official)
    assert shared.head is trained


def test_eagle_topk_buffer_sharing_is_multi_mtp_gated():
    target_buffer = object()

    class DraftInner:
        def __init__(self):
            self.child = SimpleNamespace(topk_indices_buffer=None)

        def named_modules(self):
            return [("", self), ("child", self.child)]

    models: list[object] = []

    def load_eagle_model(target_model, vllm_config):
        model = SimpleNamespace(model=DraftInner())
        models.append(model)
        return model

    module = _module(
        patch_eagle_utils.TARGET_MODULE, load_eagle_model=load_eagle_model
    )
    patch_eagle_utils.apply_to_module(module)
    target = SimpleNamespace(model=SimpleNamespace(topk_indices_buffer=target_buffer))
    off_model = module.load_eagle_model(target, _config())
    assert off_model.model.child.topk_indices_buffer is None
    on_model = module.load_eagle_model(
        target, _config(enable_multi_layers_mtp=True)
    )
    assert on_model.model.child.topk_indices_buffer is target_buffer


def test_ubatch_sms_guard_disables_only_missing_compute_control(
    monkeypatch: pytest.MonkeyPatch,
):
    class UBatchWrapper:
        @staticmethod
        def _create_sm_control_context(vllm_config):
            return "official"

    module = _module(
        patch_gpu_ubatch_wrapper.TARGET_MODULE,
        UBatchWrapper=UBatchWrapper,
        deep_gemm_set_num_sms=lambda value: None,
    )
    monkeypatch.setattr(
        "vllm_hcu.v1.worker_framework_runtime.deep_gemm_has_sms_api",
        lambda target: False,
    )
    monkeypatch.setattr(
        "vllm_hcu.v1.worker_framework_runtime.create_sm_control_context_without_compute",
        lambda target, config: "hcu-no-compute-sms",
    )
    patch_gpu_ubatch_wrapper.apply_to_module(module)
    assert UBatchWrapper._create_sm_control_context(object()) == "hcu-no-compute-sms"


def test_split_group_compat_drops_removed_backend_keyword() -> None:
    calls: list[dict[str, object]] = []

    def torch_211_split_group(
        parent_pg=None,
        split_ranks=None,
        timeout=None,
        pg_options=None,
        group_desc=None,
    ):
        calls.append(
            {
                "parent_pg": parent_pg,
                "split_ranks": split_ranks,
                "timeout": timeout,
                "pg_options": pg_options,
                "group_desc": group_desc,
            }
        )
        return "group"

    distributed = SimpleNamespace(split_group=torch_211_split_group)

    assert worker_framework_runtime.install_split_group_backend_compat(
        distributed
    )
    assert not worker_framework_runtime.install_split_group_backend_compat(
        distributed
    )
    assert (
        distributed.split_group(
            split_ranks=[[0, 1]],
            group_desc="pp:device",
            backend="cuda:nccl",
        )
        == "group"
    )
    assert calls == [
        {
            "parent_pg": None,
            "split_ranks": [[0, 1]],
            "timeout": None,
            "pg_options": None,
            "group_desc": "pp:device",
        }
    ]


def test_split_group_compat_preserves_native_backend_signature() -> None:
    def split_group(*, backend=None):
        return backend

    distributed = SimpleNamespace(split_group=split_group)

    assert not worker_framework_runtime.install_split_group_backend_compat(
        distributed
    )
    assert distributed.split_group(backend="cuda:nccl") == "cuda:nccl"


def test_pp_v2_spec_warmup_suppresses_and_restores_sample_broadcast() -> None:
    calls: list[tuple[str, object]] = []

    class PPHandler:
        def receive(self, input_batch: object) -> bool:
            calls.append(("receive", input_batch))
            return True

        def broadcast(self, sampled_tokens: object) -> None:
            calls.append(("broadcast", sampled_tokens))

    handler = PPHandler()
    receive = handler.receive
    broadcast = handler.broadcast
    model_runner = SimpleNamespace(pp_handler=handler)

    with worker_framework_runtime.suppress_pp_v2_warmup_sample_broadcast(
        model_runner
    ):
        assert model_runner._vllm_hcu_suppress_pp_spec_draft_sync is True
        assert handler.receive("input") is False
        assert handler.broadcast("tokens") is None
        assert calls == []

    assert not hasattr(model_runner, "_vllm_hcu_suppress_pp_spec_draft_sync")
    assert handler.receive == receive
    assert handler.broadcast == broadcast
    assert handler.receive("input") is True
    handler.broadcast("tokens")
    assert calls == [("receive", "input"), ("broadcast", "tokens")]


def test_pp_v2_spec_broadcast_pads_initial_sample_to_receiver_width() -> None:
    from vllm_hcu.v1 import hcu_model_runner_v2

    install_fixed_width_pp_sample_broadcast = (
        hcu_model_runner_v2.install_fixed_width_pp_sample_broadcast
    )

    captured: list[torch.Tensor] = []

    class PPHandler:
        is_last_rank = True
        max_sample_len = 6

        def broadcast(self, sampled_token_ids: torch.Tensor, *args, **kwargs):
            captured.append(sampled_token_ids)

    handler = PPHandler()
    runner = SimpleNamespace(pp_handler=handler)

    assert install_fixed_width_pp_sample_broadcast(runner)
    assert not install_fixed_width_pp_sample_broadcast(runner)
    handler.broadcast(
        torch.tensor([[42]], dtype=torch.int64),
        object(),
        object(),
        object(),
    )

    assert len(captured) == 1
    assert captured[0].shape == (1, 6)
    assert captured[0].dtype == torch.int64
    assert captured[0][0, 0].item() == 42
    assert captured[0][0, 1:].tolist() == [0, 0, 0, 0, 0]


def test_pp_v2_spec_drafts_are_broadcast_to_non_last_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.v1 import hcu_model_runner_v2

    calls: list[tuple[str, object]] = []
    payload = torch.tensor([[101, 102, 103]], dtype=torch.int64)

    class Stream:
        def wait_stream(self, stream: object) -> None:
            calls.append(("wait", stream))

        def synchronize(self) -> None:
            calls.append(("sync", self))

    main_stream = object()
    broadcast_stream = Stream()
    handler = SimpleNamespace(
        is_last_rank=False,
        last_rank=7,
        broadcast_group="draft-group",
        broadcast_stream=broadcast_stream,
        main_stream=main_stream,
    )
    runner = SimpleNamespace(
        pp_handler=handler,
        num_speculative_steps=3,
        device=torch.device("cpu"),
        req_states=SimpleNamespace(
            draft_tokens=torch.zeros((4, 3), dtype=torch.int64)
        ),
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([2], dtype=torch.int64),
    )

    monkeypatch.setattr(
        hcu_model_runner_v2.torch.cuda,
        "stream",
        lambda stream: contextlib.nullcontext(),
    )

    def broadcast(tensor, *, src, group):
        calls.append(("broadcast", (src, group)))
        tensor.copy_(payload)

    monkeypatch.setattr(
        hcu_model_runner_v2.torch.distributed, "broadcast", broadcast
    )

    assert hcu_model_runner_v2.synchronize_pp_spec_draft_tokens(
        runner, input_batch
    )
    assert runner.req_states.draft_tokens[2].tolist() == [101, 102, 103]
    assert calls == [
        ("wait", main_stream),
        ("broadcast", (7, "draft-group")),
        ("sync", broadcast_stream),
    ]


def _fake_ubatch_module() -> ModuleType:
    @dataclasses.dataclass
    class UBatchSlice:
        request_slice: slice
        token_slice: slice

        @property
        def num_tokens(self):
            return self.token_slice.stop - self.token_slice.start

    def _pad_out_ubatch_slices(ubatch_slices, num_total_tokens, num_reqs_padded):
        last = ubatch_slices[-1]
        return ubatch_slices[:-1] + [
            UBatchSlice(
                slice(last.request_slice.start, num_reqs_padded),
                slice(last.token_slice.start, num_total_tokens),
            )
        ]

    def maybe_create_ubatch_slices(
        should_ubatch,
        num_scheduled_tokens,
        num_tokens_padded,
        num_reqs_padded,
        num_ubatches,
        split_point=None,
    ):
        raise AssertionError("HCU implementation must own list split points")

    def _make_metadata_with_slice(ubatch_slice, attn_metadata):
        return SimpleNamespace()

    return _module(
        patch_ubatch_utils.TARGET_MODULE,
        UBatchSlice=UBatchSlice,
        np=np,
        _pad_out_ubatch_slices=_pad_out_ubatch_slices,
        maybe_create_ubatch_slices=maybe_create_ubatch_slices,
        _make_metadata_with_slice=_make_metadata_with_slice,
    )


def test_ubatch_list_splits_use_python_ints_and_preserve_attention_metadata():
    module = _fake_ubatch_module()
    patch_ubatch_utils.apply_to_module(module)
    slices, padded = module.maybe_create_ubatch_slices(
        True,
        np.array([3, 3], dtype=np.int32),
        np.int32(8),
        np.int32(2),
        np.int32(2),
        [np.int32(3)],
    )
    assert all(
        type(boundary) is int
        for item in (*slices, *padded)
        for boundary in (
            item.request_slice.start,
            item.request_slice.stop,
            item.token_slice.start,
            item.token_slice.stop,
        )
    )
    metadata = SimpleNamespace(
        positions=torch.arange(6), is_prefilling=torch.tensor([True, False])
    )
    result = module._make_metadata_with_slice(slices[0], metadata)
    torch.testing.assert_close(result.positions, torch.arange(3))
    torch.testing.assert_close(result.is_prefilling, torch.tensor([True]))


@pytest.mark.hcu
def test_clean_vllm_modules_import_apply_and_second_apply_is_idempotent():
    script = r'''
import os
from pathlib import Path
import vllm
target_root = Path(os.environ["VLLM_V028_SOURCE_ROOT"]).resolve()
target_file = Path(vllm.__file__).resolve()
assert target_file.is_relative_to(target_root), (
    f"vllm resolved outside target root: {target_file} not under {target_root}"
)
print('VLLM_SOURCE', vllm.__file__)
from vllm_hcu.patch.worker.framework_opt import (
    patch_all2all, patch_base_device_communicator, patch_cuda_communicator,
    patch_dp_utils, patch_eagle_utils, patch_forward_context,
    patch_gpu_ubatch_wrapper, patch_llm_base_proposer, patch_pynccl,
    patch_pynccl_wrapper, patch_ubatch_utils,
)
adapters = (
    patch_all2all, patch_base_device_communicator, patch_forward_context,
    patch_llm_base_proposer, patch_dp_utils, patch_eagle_utils,
    patch_gpu_ubatch_wrapper, patch_ubatch_utils,
)
for adapter in adapters:
    assert adapter.apply() is True, adapter.__name__
    assert adapter.apply() is False, adapter.__name__
assert patch_cuda_communicator.apply_to_module(
    __import__(patch_cuda_communicator.TARGET_MODULE, fromlist=['CudaCommunicator'])
) is True
assert patch_cuda_communicator.apply_to_module(
    __import__(patch_cuda_communicator.TARGET_MODULE, fromlist=['CudaCommunicator'])
) is False
wrapper_applied = patch_pynccl_wrapper.apply(required=False)
assert patch_pynccl_wrapper.apply(required=False) is False
pynccl_applied = patch_pynccl.apply(required=False)
assert patch_pynccl.apply(required=False) is False
print('REAL_WORKER_FRAMEWORK_OK', wrapper_applied, pynccl_applied)
'''
    env = dict(os.environ)
    # Apply adapters explicitly so this subprocess is independent of ambient
    # plugin discovery settings.
    env["VLLM_PLUGINS"] = "__disabled__"
    repository = Path(__file__).resolve().parents[2]
    import importlib.util

    vllm_spec = importlib.util.find_spec("vllm")
    assert vllm_spec is not None and vllm_spec.origin is not None
    target_vllm = Path(vllm_spec.origin).resolve().parent.parent
    if not (target_vllm / "vllm" / "__init__.py").is_file():
        raise RuntimeError(
            f"VLLM_V028_SOURCE_ROOT does not contain vllm: {target_vllm}"
        )
    env["VLLM_V028_SOURCE_ROOT"] = str(target_vllm)
    env["PYTHONPATH"] = os.pathsep.join((str(target_vllm), str(repository)))
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REAL_WORKER_FRAMEWORK_OK" in result.stdout
    assert f"VLLM_SOURCE {target_vllm / 'vllm' / '__init__.py'}" in result.stdout
