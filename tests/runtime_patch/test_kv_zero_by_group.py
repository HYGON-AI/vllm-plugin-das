# SPDX-License-Identifier: Apache-2.0
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.framework_opt import patch_kv_zero_by_group as patch
from vllm_hcu.patch.worker.framework_opt._common import PatchCompatibilityError


class RecordingZeroer:
    def __init__(self, calls, name, meta=True):
        self.calls, self.name = calls, name
        self._meta = object() if meta else None

    def zero_block_ids(self, block_ids):
        self.calls.append((self.name, list(block_ids)))


def target(calls):
    class GPUModelRunner:
        def _init_kv_zero_meta(self):
            self.kv_block_zeroer = RecordingZeroer(calls, "all")

        def update_requests(self, scheduler_output):
            calls.append(("update", scheduler_output.new_block_ids_to_zero))
            if scheduler_output.new_block_ids_to_zero:
                self.kv_block_zeroer.zero_block_ids(scheduler_output.new_block_ids_to_zero)

    module = ModuleType(patch.TARGET_MODULE)
    module.GPUModelRunner = GPUModelRunner
    return module, GPUModelRunner


def scheduler_output(to_zero):
    return SimpleNamespace(
        new_block_ids_to_zero=to_zero,
        scheduled_new_reqs=[SimpleNamespace(block_ids=([1, 2], [3], [4, 5]))],
        scheduled_cached_reqs=SimpleNamespace(new_block_ids=[None, ([6], [], [7])]),
    )


def test_apply_is_idempotent_and_rejects_drift():
    module, cls = target([])
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    cls.update_requests = lambda self, scheduler_output: None
    with pytest.raises(PatchCompatibilityError, match="stale"):
        patch.apply_to_module(module)
    module, cls = target([])
    cls.update_requests = lambda self: None
    with pytest.raises(PatchCompatibilityError, match="signature"):
        patch.apply_to_module(module)


def test_block_owners_cover_new_and_cached_requests():
    assert patch.block_owners(scheduler_output(None)) == {
        1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 0, 7: 2,
    }


def test_update_zeroes_each_block_with_its_owner_before_original(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_KV_ZERO_BY_GROUP", raising=False)
    calls = []
    module, cls = target(calls)
    patch.apply_to_module(module)
    runner = cls()
    runner.kv_block_zeroer = RecordingZeroer(calls, "all")
    # Group 1 has no zeroable segments: its ids keep upstream zeroing.
    setattr(runner, patch._ZEROERS, {0: RecordingZeroer(calls, "g0"),
                                     2: RecordingZeroer(calls, "g2")})
    output = scheduler_output([1, 4, 3, 6, 9, 7])
    runner.update_requests(output)
    assert calls == [
        ("g0", [1, 6]), ("g2", [4, 7]), ("all", [3, 9]), ("update", None),
    ]
    assert output.new_block_ids_to_zero == [1, 4, 3, 6, 9, 7]


@pytest.mark.parametrize("case", ["disabled", "no_zeroers", "nothing_to_zero"])
def test_update_keeps_upstream_path(monkeypatch, case):
    monkeypatch.setenv("VLLM_HCU_KV_ZERO_BY_GROUP", "0" if case == "disabled" else "1")
    calls = []
    module, cls = target(calls)
    patch.apply_to_module(module)
    runner = cls()
    runner.kv_block_zeroer = RecordingZeroer(calls, "all")
    setattr(runner, patch._ZEROERS,
            {} if case == "no_zeroers" else {0: RecordingZeroer(calls, "g0")})
    to_zero = None if case == "nothing_to_zero" else [1, 3]
    runner.update_requests(scheduler_output(to_zero))
    expected = [("update", to_zero)]
    if to_zero:
        expected.append(("all", to_zero))
    assert calls == expected


def test_init_builds_one_upstream_zeroer_per_group():
    built = []

    class Zeroer(RecordingZeroer):
        def __init__(self, device, groups, kernel_block_sizes,
                     static_forward_context, num_blocks):
            groups = list(groups)
            built.append(sorted({g.kv_cache_group_id for g in groups}))
            super().__init__([], "z", meta=groups[0].kv_cache_group_id != 1)

    module, cls = target([])
    cls._init_kv_zero_meta = lambda self: setattr(
        self, "kv_block_zeroer", Zeroer(None, [SimpleNamespace(kv_cache_group_id=0)],
                                        None, None, None))
    patch.apply_to_module(module)
    runner = cls()
    runner.device = torch.device("cpu")
    runner.attn_groups = [[SimpleNamespace(kv_cache_group_id=0),
                           SimpleNamespace(kv_cache_group_id=0)],
                          [SimpleNamespace(kv_cache_group_id=1)],
                          [SimpleNamespace(kv_cache_group_id=2)]]
    runner.kernel_block_sizes = [64, 64, 4]
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.kv_cache_config = SimpleNamespace(num_blocks=8)
    runner._init_kv_zero_meta()
    assert built == [[0], [0], [1], [2]]
    assert sorted(getattr(runner, patch._ZEROERS)) == [0, 2]
