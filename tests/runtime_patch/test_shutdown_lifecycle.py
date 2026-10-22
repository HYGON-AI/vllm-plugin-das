# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vllm_hcu.patch.worker.framework_opt import patch_gpu_worker_shutdown


REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_ALLREDUCE_SOURCE = (
    REPO_ROOT
    / "vllm_hcu"
    / "distributed"
    / "device_communicators"
    / "custom_all_reduce.py"
)


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _fake_gpu_worker_module(
    *,
    rocm: bool = True,
    cudart_message: str = patch_gpu_worker_shutdown._MISSING_CUDART_MESSAGE,
) -> tuple[ModuleType, type]:
    platform = SimpleNamespace(is_rocm=lambda: rocm, is_cuda_alike=lambda: True)

    cuda_globals = {
        "__name__": patch_gpu_worker_shutdown._CUDA_WRAPPER_MODULE,
        "MESSAGE": cudart_message,
    }
    exec(
        "class CudaRTLibrary:\n"
        "    def __init__(self):\n"
        "        raise AssertionError(MESSAGE)\n",
        cuda_globals,
    )
    cumem_globals = {
        "__name__": patch_gpu_worker_shutdown._CUMEM_MODULE,
        "CudaRTLibrary": cuda_globals["CudaRTLibrary"],
    }
    cumem_code = compile(
        "libcudart = CudaRTLibrary()",
        "audited_fake_cumem.py",
        "exec",
    )
    shutdown_globals = {
        "__name__": patch_gpu_worker_shutdown.TARGET_MODULE,
        "current_platform": platform,
        "CUMEM_CODE": cumem_code,
        "CUMEM_GLOBALS": cumem_globals,
    }
    # ``use_real_import`` keeps the audited v0.25.1 names in the code object.
    # The synthetic exec branch produces the same three relevant traceback
    # frames without importing vLLM or touching a device.
    exec(
        "def shutdown(self):\n"
        "    self.events.append('target-start')\n"
        "    if self.use_real_import:\n"
        "        if current_platform.is_cuda_alike():\n"
        "            from vllm.device_allocator.cumem import CuMemAllocator\n"
        "            if CuMemAllocator.instance is not None:\n"
        "                CuMemAllocator.instance.release_pools()\n"
        "    if self.raise_probe:\n"
        "        exec(CUMEM_CODE, CUMEM_GLOBALS)\n"
        "    return 'target-result'\n",
        shutdown_globals,
    )

    class Worker:
        pass

    Worker.shutdown = shutdown_globals["shutdown"]
    module = _module(
        patch_gpu_worker_shutdown.TARGET_MODULE,
        Worker=Worker,
        current_platform=platform,
    )
    return module, Worker


def _worker(worker_class: type, *, raise_probe: bool) -> object:
    worker = worker_class()
    worker.events = []
    worker.use_real_import = False
    worker.raise_probe = raise_probe
    return worker


def test_gpu_worker_shutdown_preserves_target_behavior_and_is_idempotent():
    module, worker_class = _fake_gpu_worker_module()
    assert patch_gpu_worker_shutdown.apply_to_module(module) is True
    assert patch_gpu_worker_shutdown.apply_to_module(module) is False

    worker = _worker(worker_class, raise_probe=False)
    assert worker.shutdown() == "target-result"
    assert worker.events == ["target-start"]


def test_gpu_worker_shutdown_skips_only_absent_unused_cumem_probe(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delitem(
        sys.modules, patch_gpu_worker_shutdown._CUMEM_MODULE, raising=False
    )
    module, worker_class = _fake_gpu_worker_module()
    assert patch_gpu_worker_shutdown.apply_to_module(module) is True

    worker = _worker(worker_class, raise_probe=True)
    assert worker.shutdown() is None
    assert worker.events == ["target-start"]


@pytest.mark.parametrize("rocm,preloaded,wrong_message", [
    (False, False, False),
    (True, True, False),
    (True, False, True),
])
def test_gpu_worker_shutdown_rethrows_every_unapproved_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
    rocm: bool,
    preloaded: bool,
    wrong_message: bool,
):
    if preloaded:
        monkeypatch.setitem(
            sys.modules,
            patch_gpu_worker_shutdown._CUMEM_MODULE,
            ModuleType(patch_gpu_worker_shutdown._CUMEM_MODULE),
        )
    else:
        monkeypatch.delitem(
            sys.modules, patch_gpu_worker_shutdown._CUMEM_MODULE, raising=False
        )
    message = (
        "different allocator assertion"
        if wrong_message
        else patch_gpu_worker_shutdown._MISSING_CUDART_MESSAGE
    )
    module, worker_class = _fake_gpu_worker_module(
        rocm=rocm,
        cudart_message=message,
    )
    assert patch_gpu_worker_shutdown.apply_to_module(module) is True

    worker = _worker(worker_class, raise_probe=True)
    with pytest.raises(AssertionError, match=message):
        worker.shutdown()


def test_gpu_worker_shutdown_rejects_target_without_v0251_probe():
    class Worker:
        def shutdown(self):
            return None

    module = _module(
        patch_gpu_worker_shutdown.TARGET_MODULE,
        Worker=Worker,
        current_platform=SimpleNamespace(is_rocm=lambda: True),
    )
    with pytest.raises(
        RuntimeError,
        match="no longer contains the audited v0.25.1 cumem shutdown probe",
    ):
        patch_gpu_worker_shutdown.apply_to_module(module)


def _load_custom_allreduce_source(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    vllm = _module("vllm")
    vllm.__path__ = []
    distributed = _module("vllm.distributed")
    distributed.__path__ = []
    communicators = _module("vllm.distributed.device_communicators")
    communicators.__path__ = []
    envs = _module(
        "vllm.envs",
        VLLM_SKIP_P2P_CHECK=False,
        CUDA_VISIBLE_DEVICES=None,
    )
    all_reduce_utils = _module(
        "vllm.distributed.device_communicators.all_reduce_utils",
        CUSTOM_ALL_REDUCE_MAX_SIZES={},
        gpu_p2p_access_check=lambda rank, peer: True,
    )
    parallel_state = _module(
        "vllm.distributed.parallel_state",
        in_the_same_node_as=lambda group, source_rank=0: [True],
    )
    logger = _module(
        "vllm.logger",
        init_logger=lambda name: SimpleNamespace(
            info=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        ),
    )
    platforms = _module(
        "vllm.platforms",
        current_platform=SimpleNamespace(),
    )
    hcu_ops_module = _module("vllm_hcu.hcu_ops")

    dist = _module("torch.distributed")
    dist.Backend = SimpleNamespace(NCCL="nccl")
    dist.ProcessGroup = object

    class _FakeDevice:
        def __init__(self, spec: int | str) -> None:
            del spec
            self.type = "cuda"
            self.index = 0

    torch = _module("torch")
    torch.distributed = dist
    torch.device = _FakeDevice
    torch.ops = SimpleNamespace()

    modules = {
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
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "_vllm_hcu_custom_all_reduce_lifecycle_test",
        CUSTOM_ALLREDUCE_SOURCE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_custom_allreduce_initializes_native_ownership_inert(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_custom_allreduce_source(monkeypatch)
    module.custom_ar = False
    communicator = module.CustomAllreduce(None, "cuda:0")
    assert communicator.disabled is True
    assert communicator._ptr == 0
    assert communicator.meta_ptrs == []
    assert communicator.buffer_ptrs == []
    assert communicator.rank == -1
    communicator.close()


def test_custom_allreduce_close_is_idempotent_and_releases_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_custom_allreduce_source(monkeypatch)
    calls: list[tuple[str, int]] = []
    hcu_ops = SimpleNamespace(
        dispose=lambda pointer: calls.append(("dispose", pointer)),
        free_shared_buffer=lambda pointer: calls.append(("free", pointer)),
    )
    module.torch = SimpleNamespace(ops=SimpleNamespace(hcu_ops=hcu_ops))

    communicator = object.__new__(module.CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 7
    communicator.meta_ptrs = [11, 12]
    communicator.buffer_ptrs = [21, 22]
    communicator.rank = 1
    communicator.close()
    communicator.close()

    assert calls == [("dispose", 7), ("free", 12), ("free", 22)]
    assert communicator._ptr == 0
    assert communicator.meta_ptrs == []
    assert communicator.buffer_ptrs == []
    assert communicator.disabled is True


def test_custom_allreduce_finalizer_does_not_access_cleared_torch_global(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_custom_allreduce_source(monkeypatch)
    communicator = object.__new__(module.CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 7
    communicator.meta_ptrs = [11]
    communicator.buffer_ptrs = [21]
    communicator.rank = 0
    module.torch = None

    communicator.__del__()
    assert communicator._ptr == 7
    assert communicator.meta_ptrs == [11]
    assert communicator.buffer_ptrs == [21]

    # Keep the real test-process finalizer inert after proving the guarded path.
    communicator._ptr = 0
    communicator.meta_ptrs = []
    communicator.buffer_ptrs = []
