"""Real Core function admission retains aliases and non-draft semantics."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from vllm.config.speculative import SpeculativeConfig
from hcu_vllm_das.parallel import minimax31_dspark_admission as admission


def test_minimax_dspark_is_not_rewritten_to_mtp_and_alias_is_preserved():
    alias=SpeculativeConfig.hf_config_override
    admission.install()
    config=SimpleNamespace(architectures=[admission.ARCH],model_type='minimax_m3_vl',
        dspark_block_size=7,dspark_target_layer_ids=[3,17,31,45,59],
        num_hidden_layers=5,hidden_size=6144)
    assert alias is SpeculativeConfig.hf_config_override
    assert alias(config) is config
    bad=deepcopy(config);bad.num_hidden_layers=4
    with pytest.raises(ValueError,match='unsupported'):alias(bad)
    # A normal non-MTP configuration is still delegated to Core unchanged.
    from transformers import PretrainedConfig
    ordinary=PretrainedConfig(architectures=['Qwen3ForCausalLM'],model_type='qwen3')
    assert alias(ordinary) is ordinary
    mtp=PretrainedConfig(architectures=['MiniMaxM3SparseForConditionalGeneration'],
        model_type='minimax_m3_vl',text_config=PretrainedConfig(num_mtp_modules=1))
    assert alias(mtp).architectures==['MiniMaxM3MTP']
    admission.check_health()


def test_source_drift_rejected():
    def unrelated():pass
    with pytest.raises(RuntimeError,match='source changed'):
        admission._prepare(unrelated,'__post_init__')
