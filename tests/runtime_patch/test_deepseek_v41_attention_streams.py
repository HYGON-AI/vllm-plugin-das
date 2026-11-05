from types import ModuleType, SimpleNamespace

import pytest

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v41_attention


@pytest.mark.parametrize(
    ("setting", "multi_stream"),
    [(None, False), ("0", False), ("1", True)],
)
def test_attention_stream_gate_applies_to_constructor_and_all_consumers(
    monkeypatch, setting, multi_stream
):
    class DeepseekV4Attention:
        def __init__(
            self,
            vllm_config,
            prefix,
            topk_indices_buffer=None,
            aux_stream_list=None,
            candidate_block_buffer=None,
        ):
            self.aux_stream_list = aux_stream_list
            self.arguments = (
                vllm_config, prefix, topk_indices_buffer, candidate_block_buffer
            )

        def _run_parallel_input_projections(self, hidden_states):
            raise AssertionError("Original projection must be replaced")

    module = ModuleType(patch_deepseek_v41_attention.TARGET_MODULE)
    module.DeepseekV4Attention = DeepseekV4Attention
    module.execute_in_parallel = lambda *args, **kwargs: None
    module.envs = SimpleNamespace(VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=0)
    flag = "VLLM_HCU_ENABLE_DEEPSEEK_V4_MULTI_STREAM"
    monkeypatch.delitem(patch_deepseek_v41_attention.henvs.__dict__, flag, raising=False)
    if setting is None:
        monkeypatch.delenv(flag, raising=False)
    else:
        monkeypatch.setenv(flag, setting)
    config, topk_buffer, candidate_buffer = object(), object(), object()
    streams = [object(), object(), object()]

    assert patch_deepseek_v41_attention.apply_to_module(module) is True
    positional = DeepseekV4Attention(
        config, "layers.0", topk_buffer, streams, candidate_buffer
    )
    keyword = DeepseekV4Attention(
        vllm_config=config,
        prefix="layers.0",
        topk_indices_buffer=topk_buffer,
        aux_stream_list=streams,
        candidate_block_buffer=candidate_buffer,
    )
    for attention in (positional, keyword):
        assert attention.aux_stream_list is (streams if multi_stream else None)
        assert attention.arguments == (
            config, "layers.0", topk_buffer, candidate_buffer
        )
    assert patch_deepseek_v41_attention.apply_to_module(module) is False
