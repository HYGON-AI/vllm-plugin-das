from enum import Enum
from types import ModuleType

import pytest

from vllm_hcu.patch.worker.op_opt.moe import patch_wna16_oracle


def _fake_oracle() -> tuple[ModuleType, list[str]]:
    calls: list[str] = []

    class WNA16MoEBackend(Enum):
        TRITON = "triton"
        HUMMING = "humming"

    def map_wna16_backend(runner_backend):
        calls.append(runner_backend)
        if runner_backend == "triton":
            return WNA16MoEBackend.TRITON
        raise ValueError(runner_backend)

    module = ModuleType(patch_wna16_oracle.TARGET_MODULE)
    module.WNA16MoEBackend = WNA16MoEBackend
    module.map_wna16_backend = map_wna16_backend
    return module, calls


def test_explicit_aiter_uses_triton_container_for_hcu_runtime_dispatch():
    module, calls = _fake_oracle()

    assert patch_wna16_oracle.apply_to_module(module) is True
    assert patch_wna16_oracle.apply_to_module(module) is False
    assert module.map_wna16_backend("aiter") is module.WNA16MoEBackend.TRITON
    assert calls == []

    assert module.map_wna16_backend("triton") is module.WNA16MoEBackend.TRITON
    assert calls == ["triton"]


def test_wna16_oracle_rejects_incompatible_mapper_signature():
    module, _ = _fake_oracle()
    module.map_wna16_backend = lambda backend, extra: (backend, extra)

    with pytest.raises(Exception, match="map_wna16_backend"):
        patch_wna16_oracle.apply_to_module(module)
