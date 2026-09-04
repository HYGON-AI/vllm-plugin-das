# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""CPU-safe eligibility tests for the HCU CustomAllreduce PCIe topology gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vllm_hcu.platforms import envs as hcu_envs


REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_ALLREDUCE_SOURCE = (
    REPO_ROOT
    / "vllm_hcu"
    / "distributed"
    / "device_communicators"
    / "custom_all_reduce.py"
)
PCIE_CUSTOM_ALLREDUCE_ENV = "VLLM_HCU_ENABLE_PCIE_CUSTOM_ALLREDUCE"


class _FakeDevice:
    def __init__(self, spec: int | str) -> None:
        if isinstance(spec, int):
            self.type = "cuda"
            self.index = spec
            return
        text = str(spec)
        if ":" in text:
            kind, index = text.split(":", 1)
            self.type = kind
            self.index = int(index)
            return
        self.type = text
        self.index = 0


class _FakeTensor:
    def __init__(self, data, dtype=None, device="cpu") -> None:
        self._values = list(data) if isinstance(data, (list, tuple)) else [data]
        self.dtype = dtype
        self.device = device

    def item(self):
        return self._values[0]

    def fill_(self, value):
        self._values = [value for _ in self._values]
        return self

    def numel(self) -> int:
        return len(self._values)

    def element_size(self) -> int:
        return 4

    def is_contiguous(self) -> bool:
        return True


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _warning_sink(bucket: list[str]):
    def warning(message: str, *args: object, **_kwargs: object) -> None:
        bucket.append(message % args if args else message)

    return warning


def _load_custom_allreduce_source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    world_size: int,
    fully_connected: bool,
) -> tuple[ModuleType, list[str], list[tuple[object, ...]]]:
    warnings: list[str] = []
    init_calls: list[tuple[object, ...]] = []
    hcu_ops = SimpleNamespace(
        meta_size=lambda: 8,
        init_custom_ar=lambda *args, **_kwargs: (
            init_calls.append(args) or 99
        ),
        register_buffer=lambda *_args, **_kwargs: None,
        allocate_shared_buffer_and_handle=lambda _size: (1, b"handle"),
        open_mem_handle=lambda _handle: 2,
        free_shared_buffer=lambda _pointer: None,
        dispose=lambda _pointer: None,
    )

    dist = _module("torch.distributed")
    dist.Backend = SimpleNamespace(NCCL="nccl")
    dist.ProcessGroup = object
    dist.get_backend = lambda _group: "gloo"
    dist.get_rank = lambda group=None: 0
    dist.get_world_size = lambda group=None: world_size

    def all_gather(gather_list, tensor, group=None) -> None:
        del group
        value = tensor.item()
        for item in gather_list:
            item.fill_(value)

    def all_gather_object(handles, handle, group=None) -> None:
        del group
        for index in range(len(handles)):
            handles[index] = handle

    dist.all_gather = all_gather
    dist.all_gather_object = all_gather_object

    torch = _module("torch")
    torch.distributed = dist
    torch.device = _FakeDevice
    torch.tensor = lambda data, dtype=None, device="cpu": _FakeTensor(
        data, dtype=dtype, device=device
    )
    torch.Tensor = _FakeTensor
    torch.int = "int"
    torch.uint8 = "uint8"
    torch.empty = lambda *_args, **_kwargs: SimpleNamespace()
    torch.ops = SimpleNamespace(hcu_ops=hcu_ops)
    torch.cuda = SimpleNamespace(can_device_access_peer=lambda *_args: True)

    vllm = _module("vllm")
    vllm.__path__ = []
    distributed = _module("vllm.distributed")
    distributed.__path__ = []
    communicators = _module("vllm.distributed.device_communicators")
    communicators.__path__ = []
    envs = _module(
        "vllm.envs",
        VLLM_SKIP_P2P_CHECK=False,
        CUDA_VISIBLE_DEVICES=",".join(str(index) for index in range(world_size)),
    )
    all_reduce_utils = _module(
        "vllm.distributed.device_communicators.all_reduce_utils",
        CUSTOM_ALL_REDUCE_MAX_SIZES={},
        gpu_p2p_access_check=lambda _rank, _peer: True,
    )
    parallel_state = _module(
        "vllm.distributed.parallel_state",
        in_the_same_node_as=lambda _group, source_rank=0: [True] * world_size,
    )
    logger = _module(
        "vllm.logger",
        init_logger=lambda _name: SimpleNamespace(
            info=lambda *_args, **_kwargs: None,
            debug=lambda *_args, **_kwargs: None,
            warning=_warning_sink(warnings),
        ),
    )
    platform = SimpleNamespace(
        get_device_capability=lambda: None,
        is_cuda=lambda: False,
        device_count=lambda: world_size,
        is_cuda_alike=lambda: True,
        is_fully_connected=lambda _ids: fully_connected,
        is_rocm=lambda: True,
    )
    platforms = _module("vllm.platforms", current_platform=platform)
    hcu_ops_module = _module("vllm_hcu.hcu_ops")

    stubs = {
        "torch": torch,
        "torch.distributed": dist,
        "vllm": vllm,
        "vllm.envs": envs,
        "vllm.distributed": distributed,
        "vllm.distributed.device_communicators": communicators,
        "vllm.distributed.device_communicators.all_reduce_utils": (
            all_reduce_utils
        ),
        "vllm.distributed.parallel_state": parallel_state,
        "vllm.logger": logger,
        "vllm.platforms": platforms,
        "vllm_hcu.hcu_ops": hcu_ops_module,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "_vllm_hcu_custom_all_reduce_pcie_gate_test",
        CUSTOM_ALLREDUCE_SOURCE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, warnings, init_calls


def _construct(module: ModuleType):
    return module.CustomAllreduce(group=object(), device="cuda:0")


def _aligned_input() -> _FakeTensor:
    return _FakeTensor([0] * 16)


COMMUNICATOR_PATCH_SOURCE = (
    REPO_ROOT
    / "vllm_hcu"
    / "patch"
    / "worker"
    / "framework_opt"
    / "patch_cuda_communicator.py"
)


def test_source_fails_closed_on_pcie_instead_of_warning_and_continuing() -> None:
    source = CUSTOM_ALLREDUCE_SOURCE.read_text(encoding="utf-8")

    assert "We are using PCIe's custom allreduce" not in source
    assert "allow_custom_allreduce_for_topology" in source
    assert "custom_allreduce_topology_decision" in source
    assert PCIE_CUSTOM_ALLREDUCE_ENV in source


def test_cuda_communicator_patch_gates_aiter_before_backend_init() -> None:
    source = COMMUNICATOR_PATCH_SOURCE.read_text(encoding="utf-8")
    wrapper = source[source.index("def hcu_init(") :]

    assert "suppress_aiter_custom_allreduce_unless_topology_allows" in wrapper
    assert "is_custom_all_reduce_enabled" in source
    gate_index = wrapper.index("suppress_aiter_custom_allreduce_unless_topology_allows")
    init_index = wrapper.index("init(\n                self,")
    assert gate_index < init_index


def test_pcie_custom_allreduce_env_defaults_to_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PCIE_CUSTOM_ALLREDUCE_ENV, raising=False)
    hcu_envs.__dict__.pop(PCIE_CUSTOM_ALLREDUCE_ENV, None)

    assert PCIE_CUSTOM_ALLREDUCE_ENV in hcu_envs.hcu_vllm_environment_variables
    assert hcu_envs.VLLM_HCU_ENABLE_PCIE_CUSTOM_ALLREDUCE is False
    assert hcu_envs.is_set(PCIE_CUSTOM_ALLREDUCE_ENV) is False


@pytest.mark.parametrize("value", ("1", "true", "TRUE"))
def test_pcie_custom_allreduce_env_opt_in_parses_true(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(PCIE_CUSTOM_ALLREDUCE_ENV, value)
    hcu_envs.__dict__.pop(PCIE_CUSTOM_ALLREDUCE_ENV, None)

    assert hcu_envs.VLLM_HCU_ENABLE_PCIE_CUSTOM_ALLREDUCE is True


def test_pcie_tp2_disables_custom_allreduce_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PCIE_CUSTOM_ALLREDUCE_ENV, raising=False)
    module, warnings, init_calls = _load_custom_allreduce_source(
        monkeypatch, world_size=2, fully_connected=False
    )

    assert module.allow_custom_allreduce_for_topology(False) is False
    communicator = _construct(module)

    assert communicator.disabled is True
    assert communicator._ptr == 0
    assert communicator.meta_ptrs == []
    assert communicator.buffer_ptrs == []
    assert init_calls == []
    assert communicator.should_custom_ar(_aligned_input()) is False
    assert any("HCCL will be used instead" in message for message in warnings)
    assert any(PCIE_CUSTOM_ALLREDUCE_ENV in message for message in warnings)


def test_fully_connected_topology_still_enables_custom_allreduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PCIE_CUSTOM_ALLREDUCE_ENV, raising=False)
    module, warnings, init_calls = _load_custom_allreduce_source(
        monkeypatch, world_size=2, fully_connected=True
    )

    assert module.allow_custom_allreduce_for_topology(True) is True
    communicator = _construct(module)

    assert communicator.disabled is False
    assert communicator.fully_connected is True
    assert communicator.world_size == 2
    assert communicator._ptr == 99
    assert init_calls
    assert communicator.should_custom_ar(_aligned_input()) is True
    assert not any("HCCL will be used instead" in message for message in warnings)
    assert not any("PCIe topology" in message for message in warnings)


def test_pcie_opt_in_re_enables_tp2_custom_allreduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PCIE_CUSTOM_ALLREDUCE_ENV, "1")
    module, warnings, init_calls = _load_custom_allreduce_source(
        monkeypatch, world_size=2, fully_connected=False
    )

    assert module.allow_custom_allreduce_for_topology(False) is True
    communicator = _construct(module)

    assert communicator.disabled is False
    assert communicator.fully_connected is False
    assert communicator.world_size == 2
    assert communicator.max_size == 32 * 8192 * 2
    assert communicator._ptr == 99
    assert init_calls
    assert communicator.should_custom_ar(_aligned_input()) is True
    assert any(PCIE_CUSTOM_ALLREDUCE_ENV in message for message in warnings)
    assert any("PCIe topology" in message for message in warnings)
    assert not any("HCCL will be used instead" in message for message in warnings)


def test_fully_connected_fast_path_ignores_pcie_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PCIE_CUSTOM_ALLREDUCE_ENV, "1")
    module, warnings, init_calls = _load_custom_allreduce_source(
        monkeypatch, world_size=4, fully_connected=True
    )

    communicator = _construct(module)

    assert communicator.disabled is False
    assert communicator.fully_connected is True
    assert communicator.world_size == 4
    assert init_calls
    assert communicator.should_custom_ar(_aligned_input()) is True
    assert not any("PCIe topology" in message for message in warnings)


def test_topology_decision_fails_closed_on_pcie_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PCIE_CUSTOM_ALLREDUCE_ENV, raising=False)
    hcu_envs.__dict__.pop(PCIE_CUSTOM_ALLREDUCE_ENV, None)
    module, _warnings, _init_calls = _load_custom_allreduce_source(
        monkeypatch, world_size=2, fully_connected=False
    )

    fully_connected, allowed = module.custom_allreduce_topology_decision(
        object(), "cuda:0"
    )

    assert fully_connected is False
    assert allowed is False


def test_topology_decision_allows_fully_connected_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PCIE_CUSTOM_ALLREDUCE_ENV, raising=False)
    hcu_envs.__dict__.pop(PCIE_CUSTOM_ALLREDUCE_ENV, None)
    module, _warnings, _init_calls = _load_custom_allreduce_source(
        monkeypatch, world_size=2, fully_connected=True
    )

    fully_connected, allowed = module.custom_allreduce_topology_decision(
        object(), "cuda:0"
    )

    assert fully_connected is True
    assert allowed is True


def test_topology_decision_allows_pcie_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PCIE_CUSTOM_ALLREDUCE_ENV, "1")
    hcu_envs.__dict__.pop(PCIE_CUSTOM_ALLREDUCE_ENV, None)
    module, _warnings, _init_calls = _load_custom_allreduce_source(
        monkeypatch, world_size=2, fully_connected=False
    )

    fully_connected, allowed = module.custom_allreduce_topology_decision(
        object(), "cuda:0"
    )

    assert fully_connected is False
    assert allowed is True


def _install_vllm_config_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm = sys.modules.get("vllm")
    if vllm is None:
        vllm = _module("vllm")
        vllm.__path__ = []
        monkeypatch.setitem(sys.modules, "vllm", vllm)
    config = _module(
        "vllm.config",
        get_current_vllm_config_or_none=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    monkeypatch.setattr(vllm, "config", config, raising=False)


def _install_aiter_ops(monkeypatch: pytest.MonkeyPatch, *, enabled: bool):
    class rocm_aiter_ops:
        @classmethod
        def is_custom_all_reduce_enabled(cls) -> bool:
            return enabled

    aiter_module = _module("vllm._aiter_ops", rocm_aiter_ops=rocm_aiter_ops)
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", aiter_module)
    vllm = sys.modules.get("vllm")
    if vllm is not None:
        monkeypatch.setattr(vllm, "_aiter_ops", aiter_module, raising=False)
    return rocm_aiter_ops


def _patched_cuda_communicator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    aiter_enabled: bool,
    fully_connected: bool,
    allowed: bool,
):
    from vllm_hcu.patch.worker.framework_opt import patch_cuda_communicator

    _install_vllm_config_stub(monkeypatch)
    aiter_ops = _install_aiter_ops(monkeypatch, enabled=aiter_enabled)
    topology_calls: list[tuple[object, object]] = []

    def _decision(cpu_group, device):
        topology_calls.append((cpu_group, device))
        return fully_connected, allowed

    monkeypatch.setattr(patch_cuda_communicator, "_topology_decision", _decision)
    monkeypatch.setattr(
        patch_cuda_communicator,
        "_log_aiter_topology_decision",
        lambda **_kwargs: None,
    )

    class UpstreamCudaCommunicator:
        """Mirrors v0.25.1 CudaCommunicator custom-AR backend selection."""

        def __init__(
            self,
            cpu_group,
            device=None,
            device_group=None,
            unique_name="",
            global_ranks=None,
            global_world_size=None,
            tcp_store_group=None,
        ):
            del device_group, global_ranks, global_world_size, tcp_store_group
            from vllm._aiter_ops import rocm_aiter_ops

            self.cpu_group = cpu_group
            self.device = device
            self.unique_name = unique_name
            self.use_all2all = False
            self.probe_during_init = bool(
                rocm_aiter_ops.is_custom_all_reduce_enabled()
            )
            use_custom_allreduce = "tp" in unique_name
            use_aiter_allreduce = use_custom_allreduce and self.probe_during_init
            self.use_custom_allreduce = use_custom_allreduce
            self.use_aiter_allreduce = use_aiter_allreduce
            self.aiter_ar_comm = object() if use_aiter_allreduce else None
            self.ca_comm = None
            if use_custom_allreduce and self.aiter_ar_comm is None:
                self.ca_comm = object()

    module = _module(
        patch_cuda_communicator.TARGET_MODULE,
        CudaCommunicator=UpstreamCudaCommunicator,
    )
    assert patch_cuda_communicator.apply_to_module(module) is True
    return module, aiter_ops, topology_calls


def test_aiter_custom_allreduce_not_selected_on_pcie_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, aiter_ops, topology_calls = _patched_cuda_communicator(
        monkeypatch,
        aiter_enabled=True,
        fully_connected=False,
        allowed=False,
    )

    communicator = module.CudaCommunicator(
        "cpu-group", device="cuda:0", unique_name="tp:0"
    )

    assert topology_calls == [("cpu-group", "cuda:0")]
    assert communicator.probe_during_init is False
    assert communicator.use_aiter_allreduce is False
    assert communicator.aiter_ar_comm is None
    assert communicator.ca_comm is not None
    assert aiter_ops.is_custom_all_reduce_enabled() is True


def test_aiter_custom_allreduce_selected_on_pcie_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, aiter_ops, topology_calls = _patched_cuda_communicator(
        monkeypatch,
        aiter_enabled=True,
        fully_connected=False,
        allowed=True,
    )

    communicator = module.CudaCommunicator(
        "cpu-group", device="cuda:0", unique_name="tp:0"
    )

    assert topology_calls == [("cpu-group", "cuda:0")]
    assert communicator.probe_during_init is True
    assert communicator.use_aiter_allreduce is True
    assert communicator.aiter_ar_comm is not None
    assert communicator.ca_comm is None
    assert aiter_ops.is_custom_all_reduce_enabled() is True


def test_aiter_custom_allreduce_selected_on_fully_connected_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, aiter_ops, topology_calls = _patched_cuda_communicator(
        monkeypatch,
        aiter_enabled=True,
        fully_connected=True,
        allowed=True,
    )

    communicator = module.CudaCommunicator(
        "cpu-group", device="cuda:0", unique_name="tp:0"
    )

    assert topology_calls == [("cpu-group", "cuda:0")]
    assert communicator.use_aiter_allreduce is True
    assert communicator.aiter_ar_comm is not None
    assert communicator.ca_comm is None
    assert aiter_ops.is_custom_all_reduce_enabled() is True


def test_aiter_disabled_skips_topology_gate_and_native_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, aiter_ops, topology_calls = _patched_cuda_communicator(
        monkeypatch,
        aiter_enabled=False,
        fully_connected=False,
        allowed=False,
    )

    communicator = module.CudaCommunicator(
        "cpu-group", device="cuda:0", unique_name="tp:0"
    )

    assert topology_calls == []
    assert communicator.probe_during_init is False
    assert communicator.use_aiter_allreduce is False
    assert communicator.aiter_ar_comm is None
    assert communicator.ca_comm is not None
    assert aiter_ops.is_custom_all_reduce_enabled() is False


def test_non_tp_communicator_does_not_consult_aiter_topology_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, aiter_ops, topology_calls = _patched_cuda_communicator(
        monkeypatch,
        aiter_enabled=True,
        fully_connected=False,
        allowed=False,
    )

    communicator = module.CudaCommunicator(
        "cpu-group", device="cuda:0", unique_name="dp:0"
    )

    assert topology_calls == []
    assert communicator.use_aiter_allreduce is False
    assert communicator.aiter_ar_comm is None
    assert communicator.ca_comm is None
    assert aiter_ops.is_custom_all_reduce_enabled() is True


def test_missing_device_fails_closed_before_aiter_construction() -> None:
    from vllm_hcu.patch.worker.framework_opt import patch_cuda_communicator

    assert patch_cuda_communicator._topology_decision(object(), None) == (
        False,
        False,
    )


def test_aiter_custom_allreduce_fails_closed_when_topology_probe_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.patch.worker.framework_opt import patch_cuda_communicator

    module, aiter_ops, _topology_calls = _patched_cuda_communicator(
        monkeypatch,
        aiter_enabled=True,
        fully_connected=False,
        allowed=True,
    )

    def _boom(_cpu_group, _device):
        raise RuntimeError("topology probe failed")

    monkeypatch.setattr(patch_cuda_communicator, "_topology_decision", _boom)

    communicator = module.CudaCommunicator(
        "cpu-group", device="cuda:0", unique_name="tp:0"
    )

    assert communicator.probe_during_init is False
    assert communicator.aiter_ar_comm is None
    assert communicator.ca_comm is not None
    assert aiter_ops.is_custom_all_reduce_enabled() is True
