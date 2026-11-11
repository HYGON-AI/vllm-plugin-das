# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm_hcu.forward_context_runtime import (
    deepep_auto_request_phase_scope,
    get_deepep_auto_request_phase,
)
from vllm_hcu.patch.config import HcuFeatureConfig
from vllm_hcu.v1 import hcu_model_runner_v2 as runner_module


def _runner(*, deepep_auto: bool):
    runner = object.__new__(runner_module.HcuGPUModelRunnerV2)
    runner.vllm_config = SimpleNamespace(
        additional_config={
            "hcu": HcuFeatureConfig(deepep_auto=deepep_auto).to_dict()
        },
        model_config=SimpleNamespace(architectures=[]),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )
    runner.pcp_manager = None
    runner._hsl_enabled = False
    runner._hcu_profile_mtp_full_batch = False
    return runner


@pytest.mark.parametrize(
    ("uniform_decode", "expected_phase"),
    [(True, False), (False, True)],
)
def test_dummy_run_phase_reaches_nested_execute_model(
    monkeypatch: pytest.MonkeyPatch,
    uniform_decode: bool,
    expected_phase: bool,
):
    runner = _runner(deepep_auto=True)
    observed: list[object | None] = []
    monkeypatch.setattr(runner_module, "is_step5_config", lambda config: False)

    def execute_model(self, *args, **kwargs):
        observed.append(get_deepep_auto_request_phase())
        return "executed"

    def dummy_run(self, num_tokens, *args, **kwargs):
        return self.execute_model("dummy", dummy_run=True)

    monkeypatch.setattr(GPUModelRunner, "execute_model", execute_model)
    monkeypatch.setattr(GPUModelRunner, "_dummy_run", dummy_run)

    assert runner._dummy_run(4, uniform_decode=uniform_decode) == "executed"
    assert observed == [expected_phase]
    assert runner_module._DEEPEP_AUTO_DUMMY_PHASE.get() is None


def test_dummy_run_phase_is_restored_after_exception(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner(deepep_auto=True)
    monkeypatch.setattr(runner_module, "is_step5_config", lambda config: False)

    def execute_model(self, *args, **kwargs):
        assert get_deepep_auto_request_phase() is False
        raise RuntimeError("dummy failure")

    def dummy_run(self, num_tokens, *args, **kwargs):
        return self.execute_model("dummy", dummy_run=True)

    monkeypatch.setattr(GPUModelRunner, "execute_model", execute_model)
    monkeypatch.setattr(GPUModelRunner, "_dummy_run", dummy_run)

    with pytest.raises(RuntimeError, match="dummy failure"):
        runner._dummy_run(4, uniform_decode=True)

    assert runner_module._DEEPEP_AUTO_DUMMY_PHASE.get() is None


def test_dummy_phase_scope_restores_nested_values():
    assert runner_module._DEEPEP_AUTO_DUMMY_PHASE.get() is None

    with runner_module._deepep_auto_dummy_phase_scope(False):
        assert runner_module._DEEPEP_AUTO_DUMMY_PHASE.get() is False
        with runner_module._deepep_auto_dummy_phase_scope(True):
            assert runner_module._DEEPEP_AUTO_DUMMY_PHASE.get() is True
        assert runner_module._DEEPEP_AUTO_DUMMY_PHASE.get() is False

    assert runner_module._DEEPEP_AUTO_DUMMY_PHASE.get() is None


def test_dummy_run_phase_is_not_published_when_feature_is_off(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner(deepep_auto=False)
    observed: list[object | None] = []
    monkeypatch.setattr(runner_module, "is_step5_config", lambda config: False)

    def execute_model(self, *args, **kwargs):
        observed.append(get_deepep_auto_request_phase())
        return "executed"

    def dummy_run(self, num_tokens, *args, **kwargs):
        return self.execute_model("dummy", dummy_run=True)

    monkeypatch.setattr(GPUModelRunner, "execute_model", execute_model)
    monkeypatch.setattr(GPUModelRunner, "_dummy_run", dummy_run)

    assert runner._dummy_run(4, uniform_decode=True) == "executed"
    assert observed == [None]


def test_real_batch_phase_remains_authoritative_without_dummy_scope(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner(deepep_auto=True)
    phase = np.array([True, False], dtype=np.bool_)
    input_batch = SimpleNamespace(is_prefilling_np=phase)
    monkeypatch.setattr(
        GPUModelRunner,
        "prepare_inputs",
        lambda self, scheduler_output, batch_req_state, batch_desc: input_batch,
    )

    with deepep_auto_request_phase_scope():
        result = runner.prepare_inputs("scheduler", "state", "descriptor")
        assert result is input_batch
        assert get_deepep_auto_request_phase() is phase

    assert runner_module._DEEPEP_AUTO_DUMMY_PHASE.get() is None
