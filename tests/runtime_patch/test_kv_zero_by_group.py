# SPDX-License-Identifier: Apache-2.0
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.utils import KVBlockZeroer
from vllm_hcu.v1.hcu_model_runner_v2 import (
    HcuGPUModelRunnerV2,
    _GroupKVBlockZeroer,
    _kv_block_owners,
)


class RecordingZeroer:
    def __init__(self, calls, name, meta=True):
        self.calls, self.name = calls, name
        self._meta = object() if meta else None

    def zero_block_ids(self, block_ids):
        self.calls.append((self.name, list(block_ids)))

    def warmup(self, num_kv_blocks):
        self.calls.append((f"{self.name}.warmup", num_kv_blocks))


def scheduler_output(to_zero):
    return SimpleNamespace(
        new_block_ids_to_zero=to_zero,
        scheduled_new_reqs=[SimpleNamespace(block_ids=([1, 2], [3], [4, 5]))],
        scheduled_cached_reqs=SimpleNamespace(new_block_ids=[None, ([6], [], [7])]),
    )


def test_block_owners_cover_new_and_cached_requests():
    assert _kv_block_owners(scheduler_output(None)) == {
        1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 0, 7: 2,
    }


def test_conflicting_owner_uses_fallback():
    output = scheduler_output(None)
    output.scheduled_cached_reqs.new_block_ids.append(([], [1], []))
    assert 1 not in _kv_block_owners(output)


def test_dispatches_to_owner_and_falls_back(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.delenv("VLLM_HCU_KV_ZERO_BY_GROUP", raising=False)
    calls = []
    dispatcher = _GroupKVBlockZeroer(
        RecordingZeroer(calls, "all"),
        {0: RecordingZeroer(calls, "g0"), 2: RecordingZeroer(calls, "g2")},
    )
    dispatcher.owners = _kv_block_owners(scheduler_output(None))
    dispatcher.zero_block_ids([1, 4, 3, 6, 9, 7])
    assert calls == [("g0", [1, 6]), ("g2", [4, 7]), ("all", [3, 9])]


def test_disabled_dispatcher_uses_upstream_zeroer(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_KV_ZERO_BY_GROUP", "0")
    calls = []
    dispatcher = _GroupKVBlockZeroer(
        RecordingZeroer(calls, "all"), {0: RecordingZeroer(calls, "g0")}
    )
    dispatcher.owners = {1: 0}
    dispatcher.zero_block_ids([1])
    assert calls == [("all", [1])]


def test_master_switch_disables_dispatcher(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("VLLM_HCU_KV_ZERO_BY_GROUP", "1")
    calls = []
    dispatcher = _GroupKVBlockZeroer(
        RecordingZeroer(calls, "all"), {0: RecordingZeroer(calls, "g0")}
    )
    dispatcher.owners = {1: 0}
    dispatcher.zero_block_ids([1])
    assert calls == [("all", [1])]


def test_group_zeroer_preserves_native_warmup_contract(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.delenv("VLLM_HCU_KV_ZERO_BY_GROUP", raising=False)
    calls = []

    def native_zeroer(name):
        zeroer = object.__new__(KVBlockZeroer)

        def zero_block_ids(self, block_ids):
            calls.append((name, block_ids))

        zeroer.zero_block_ids = MethodType(zero_block_ids, zeroer)
        return zeroer

    fallback = native_zeroer("all")
    group0 = native_zeroer("g0")
    group1 = native_zeroer("g1")

    _GroupKVBlockZeroer(fallback, {0: group0, 1: group1}).warmup(32)

    # Exercise the installed KVBlockZeroer.warmup implementation; only the
    # kernel launch is replaced so this CPU test retains the real lifecycle.
    assert calls == [("all", [0]), ("g0", [0]), ("g1", [0])]


def test_update_uses_upstream_order_without_mutating_output(monkeypatch):
    calls = []

    def upstream_update(self, output):
        calls.append(("update", output.new_block_ids_to_zero))
        self.kv_block_zeroer.zero_block_ids(output.new_block_ids_to_zero)

    monkeypatch.setattr(GPUModelRunner, "update_requests", upstream_update)
    runner = object.__new__(HcuGPUModelRunnerV2)
    runner.kv_block_zeroer = _GroupKVBlockZeroer(
        RecordingZeroer(calls, "all"), {0: RecordingZeroer(calls, "g0")}
    )
    output = scheduler_output([1, 3])
    runner.update_requests(output)
    assert calls == [("update", [1, 3]), ("g0", [1]), ("all", [3])]
    assert output.new_block_ids_to_zero == [1, 3]
    assert runner.kv_block_zeroer.owners is None


def test_init_builds_one_upstream_zeroer_per_group(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.delenv("VLLM_HCU_KV_ZERO_BY_GROUP", raising=False)
    built = []

    class Zeroer(RecordingZeroer):
        def __init__(self, device, groups, kernel_block_sizes,
                     static_forward_context, num_blocks):
            groups = list(groups)
            built.append(sorted({g.kv_cache_group_id for g in groups}))
            super().__init__([], "z", meta=groups[0].kv_cache_group_id != 1)

    monkeypatch.setattr(
        GPUModelRunner,
        "_init_kv_zero_meta",
        lambda self: setattr(
            self,
            "kv_block_zeroer",
            Zeroer(None, [SimpleNamespace(kv_cache_group_id=0)], None, None, None),
        ),
    )
    runner = object.__new__(HcuGPUModelRunnerV2)
    runner.device = torch.device("cpu")
    runner.attn_groups = [
        [SimpleNamespace(kv_cache_group_id=0), SimpleNamespace(kv_cache_group_id=0)],
        [SimpleNamespace(kv_cache_group_id=1)],
        [SimpleNamespace(kv_cache_group_id=2)],
    ]
    runner.kernel_block_sizes = [64, 64, 4]
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.kv_cache_config = SimpleNamespace(num_blocks=8)
    runner._init_kv_zero_meta()
    assert built == [[0], [0], [1], [2]]
    assert sorted(runner.kv_block_zeroer.zeroers) == [0, 2]


@pytest.mark.parametrize("custom_ops", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("num_groups", [1, 3])
def test_init_and_startup_warmup_for_single_and_multi_group(
    monkeypatch, custom_ops, enabled, num_groups
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1" if custom_ops else "0")
    monkeypatch.setenv("VLLM_HCU_KV_ZERO_BY_GROUP", "1" if enabled else "0")
    calls = []

    class Zeroer(RecordingZeroer):
        def __init__(self, device, groups, kernel_block_sizes,
                     static_forward_context, num_blocks):
            groups = list(groups)
            super().__init__(calls, f"g{groups[0].kv_cache_group_id}")

    fallback = Zeroer(None, [SimpleNamespace(kv_cache_group_id=0)], None, None, None)
    monkeypatch.setattr(
        GPUModelRunner,
        "_init_kv_zero_meta",
        lambda self: setattr(self, "kv_block_zeroer", fallback),
    )
    runner = object.__new__(HcuGPUModelRunnerV2)
    runner.device = torch.device("cpu")
    runner.attn_groups = [
        [SimpleNamespace(kv_cache_group_id=group_id)]
        for group_id in range(num_groups)
    ]
    runner.kernel_block_sizes = [64] * num_groups
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.kv_cache_config = SimpleNamespace(num_blocks=32)

    runner._init_kv_zero_meta()
    # This is the exact interface called by vllm.v1.worker.gpu.warmup.
    runner.kv_block_zeroer.warmup(runner.kv_cache_config.num_blocks)

    wrapped = custom_ops and enabled and num_groups > 1
    assert isinstance(runner.kv_block_zeroer, _GroupKVBlockZeroer) is wrapped
    expected = [("g0.warmup", 32)]
    if wrapped:
        expected += [(f"g{i}.warmup", 32) for i in range(num_groups)]
    assert calls == expected
