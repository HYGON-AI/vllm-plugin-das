# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Exercise EPLB adapter composition in the actual worker inventory order."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


_OFFLINE_NAMES = (
    "patch_model_loader_static_eplb",
    "patch_static_expert_mapping",
    "patch_offline_eplb",
    "patch_eplb_communicator",
)


def test_e22_config_and_baseline_adapter_order_are_unchanged():
    from vllm_hcu.patch.config import HcuFeatureConfig
    from vllm_hcu.patch import worker

    features = worker._feature_states(
        SimpleNamespace(parallel_config=SimpleNamespace(all2all_backend=None)),
        HcuFeatureConfig(),
    )
    assert features["offline_eplb"] is False
    assert features["eplb_control"] is False
    current = worker._FRAMEWORK_CALLBACKS
    moved = tuple(s for s in current if s.adapter.rsplit(".", 1)[-1] in _OFFLINE_NAMES)
    rest = tuple(s for s in current if s not in moved)
    assert tuple(s.adapter.rsplit(".", 1)[-1] for s in moved) == _OFFLINE_NAMES
    assert tuple(s.feature for s in moved) == (
        "offline_eplb", "offline_eplb", "eplb_control", "offline_eplb"
    )
    assert current == rest + moved
    # Feature flags are diagnostic metadata, not an application gate. Prove
    # only the exact baseline ordering here; actual delegation is checked below.
    assert tuple(s for s in moved + rest if s not in moved) == rest
    assert all(s.feature == "always" for s in rest
               if "patch_step5_eplb_" in s.adapter)


def test_real_coordinator_keeps_static_guards_outside_step5_baseline():
    # A separate process prevents native-module/adapter mutations leaking into
    # other tests and exercises source-signature checks against the installed ABI.
    code = r'''
from types import SimpleNamespace
import sys
import torch
import vllm.distributed.eplb.eplb_state as target
from vllm_hcu.patch import worker
from vllm_hcu.patch.import_coordinator import ExactImportCoordinator
from vllm_hcu.patch.runtime_state import PatchRegistry, PatchStatus
from vllm_hcu.patch.worker.framework_opt import patch_offline_eplb as offline

names = {"patch_step5_eplb_state", "patch_offline_eplb"}
specs = tuple(
    spec for spec in worker._FRAMEWORK_CALLBACKS
    if spec.adapter.rsplit(".", 1)[-1] in names
)
assert len(specs) == 2
registry = PatchRegistry()
coordinator = ExactImportCoordinator(registry=registry)
for spec in specs:
    adapter = worker._load_adapter(spec.adapter)
    coordinator.register_callback(
        adapter.PATCH_ID, adapter.TARGET_MODULE,
        worker._callback_for(spec.adapter, spec.optional_capability,
                             coordinator, adapter.PATCH_ID),
        targets=adapter.TARGETS, feature_enabled=True,
    )
    assert registry.get(adapter.PATCH_ID).status is PatchStatus.APPLIED

from vllm_hcu.distributed import eplb_runtime
assert target.EplbState.step.__wrapped__ is eplb_runtime.step
assert target.EplbState.rearrange.__wrapped__ is eplb_runtime.rearrange
installed = getattr(target, offline._MARKER)
assert installed[1] is target.EplbState.step
assert installed[2] is target.EplbState.rearrange

# Real class/methods, CPU state only: these outer guards must never reach HCU
# policy, stream/event creation, communicator, recording, or weight movement.
state = target.EplbState.__new__(target.EplbState)
state.parallel_config = SimpleNamespace(
    _vllm_hcu_expert_map_path="/already-committed-static-map.json",
    _vllm_hcu_expert_map_record_path=None,
    _vllm_hcu_eplb_disable_rearrange=True,
)
live = SimpleNamespace(
    physical_to_logical_map=torch.tensor([[0, 1, 2, 1], [2, 0, 1, 2]]),
    logical_to_physical_map=torch.tensor([[[0, -1], [1, 3], [2, -1]]]),
    logical_replica_count=torch.tensor([[1, 2, 1]]),
    expert_load_window=torch.tensor([[[2, 3, 5, 7]]]),
    expert_weights=[torch.arange(16).reshape(2, 4, 2)],
)
state.model_states = {"static": live}
state.expert_rearrangement_step = 13
state.expert_load_window_step = 17
state._vllm_hcu_offline_proposal_events = 0
state._vllm_hcu_offline_transfer_events = 0
state._vllm_hcu_offline_rearrangement_events = 0
recording = torch.tensor(False)
state.is_recording = recording
before = {name: value.clone() for name, value in vars(live).items()
          if isinstance(value, torch.Tensor)}
weights = [value.clone() for value in live.expert_weights]
for kwargs in ({}, {"is_dummy": True}, {"is_profile": True}, {"log_stats": True}):
    assert state.step(**kwargs) is None
for kwargs in ({}, {"is_profile": True}, {"rank_mapping": {0: 1, 1: 0}}):
    assert state.rearrange(**kwargs) is None
for name, value in before.items():
    assert torch.equal(getattr(live, name), value), name
assert all(torch.equal(a, b) for a, b in zip(live.expert_weights, weights))
assert state.expert_rearrangement_step == 13
assert state.expert_load_window_step == 17
assert state.is_recording is recording and not recording.item()
assert (state._vllm_hcu_offline_proposal_events,
        state._vllm_hcu_offline_transfer_events,
        state._vllm_hcu_offline_rearrangement_events) == (0, 0, 0)

# E22 has no EPLB state in the engine (enable_eplb=False). Even if called
# explicitly, disabled offline controls must delegate the exact arguments to
# the same HCU implementation. Intercept function entry before any native/GPU
# work; this observes actual wrappers without replacing either implementation.
state.parallel_config = SimpleNamespace(
    _vllm_hcu_expert_map_path=None,
    _vllm_hcu_expert_map_record_path=None,
    _vllm_hcu_eplb_disable_rearrange=False,
)
class ReachedBaseline(Exception):
    pass
def baseline_entry(call, baseline, kwargs):
    found = []
    def observe(frame, event, arg):
        if event == "call" and frame.f_code is baseline.__code__:
            found.append(dict(frame.f_locals))
            raise ReachedBaseline()
    sys.setprofile(observe)
    try:
        try:
            call(state, **kwargs)
        except ReachedBaseline:
            pass
        else:
            raise AssertionError("offline wrapper did not reach HCU baseline")
    finally:
        sys.setprofile(None)
    assert len(found) == 1
    return found[0]
for wrapped, baseline, cases in (
    (target.EplbState.step, eplb_runtime.step,
     ({}, {"is_dummy": True}, {"is_profile": True},
      {"log_stats": True}, {"is_dummy": True, "log_stats": True})),
    (target.EplbState.rearrange, eplb_runtime.rearrange,
     ({}, {"is_profile": True}, {"rank_mapping": {0: 1, 1: 0}})),
):
    for kwargs in cases:
        direct = baseline_entry(baseline, baseline, kwargs)
        delegated = baseline_entry(wrapped, baseline, kwargs)
        assert delegated == direct
print("real coordinator static lifecycle guards preserved CPU maps/weights/counters")
print("disabled offline controls delegate exact arguments to HCU baseline")
'''
    env = dict(os.environ)
    env["VLLM_PLUGINS"] = "__disabled__"
    env["HCU_STATIC_EPLB_LOAD_DIAGNOSTICS"] = "0"
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join((str(root), env.get("PYTHONPATH", "")))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env,
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
