from types import SimpleNamespace

import numpy as np
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_hcu.v1.hcu_model_runner_v2 import HcuGPUModelRunnerV2


def test_glm5next_prefill_runs_eager_but_decode_keeps_graph(monkeypatch):
    graph = object()
    calls = []

    def dispatch(num_reqs, num_tokens, uniform_token_count, **kwargs):
        calls.append((num_reqs, num_tokens, uniform_token_count))
        return graph

    manager = SimpleNamespace(dispatch=dispatch)
    runner = object.__new__(HcuGPUModelRunnerV2)
    runner.cudagraph_manager = manager
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Glm5NextForConditionalGeneration"])
    )

    def parent_gather(self, is_prefilling):
        state = SimpleNamespace(is_prefilling_np=np.array([is_prefilling]))
        return state, None

    def parent_execute(self, is_prefilling):
        self.gather_batch_req_state(is_prefilling)
        return self.cudagraph_manager.dispatch(1, 19, 19)

    monkeypatch.setattr(GPUModelRunner, "gather_batch_req_state", parent_gather)
    monkeypatch.setattr(GPUModelRunner, "execute_model", parent_execute)

    prefill = runner.execute_model(True)
    assert prefill.cg_mode == CUDAGraphMode.NONE
    assert prefill.num_tokens == 19
    assert manager.dispatch is dispatch
    assert calls == []

    decode = runner.execute_model(False)
    assert decode is graph
    assert calls == [(1, 19, 19)]
    assert manager.dispatch is dispatch

    class MethodManager:
        def dispatch(self, num_reqs, num_tokens, uniform_token_count, **kwargs):
            return graph

    runner.cudagraph_manager = MethodManager()
    assert runner.execute_model(True).cg_mode == CUDAGraphMode.NONE
    assert "dispatch" not in vars(runner.cudagraph_manager)
    assert runner.execute_model(False) is graph
