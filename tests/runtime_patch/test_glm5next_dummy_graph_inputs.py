from types import SimpleNamespace

from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_hcu.v1.hcu_model_runner_v2 import HcuGPUModelRunnerV2


class InputState:
    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        return {"inputs_embeds": "embedding-buffer"}


def test_glm5next_dummy_graph_uses_real_text_input_schema(monkeypatch):
    def inspect_parent(self, *args, **kwargs):
        return self.model_state.prepare_dummy_inputs(1, 24)

    monkeypatch.setattr(GPUModelRunner, "capture_model", inspect_parent)
    runner = object.__new__(HcuGPUModelRunnerV2)
    runner.model_state = InputState()
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Glm5NextForConditionalGeneration"])
    )

    dummy_inputs = runner.capture_model()
    assert dummy_inputs == {"inputs_embeds": "embedding-buffer", "input_ids": None}
    assert runner.model_state.prepare_dummy_inputs(1, 24) == {
        "inputs_embeds": "embedding-buffer"
    }
