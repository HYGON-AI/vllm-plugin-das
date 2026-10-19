# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
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
    monkeypatch.setenv("VLLM_HCU_KV_ZERO_BY_GROUP", "0")
    calls = []
    dispatcher = _GroupKVBlockZeroer(
        RecordingZeroer(calls, "all"), {0: RecordingZeroer(calls, "g0")}
    )
    dispatcher.owners = {1: 0}
    dispatcher.zero_block_ids([1])
    assert calls == [("all", [1])]


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
