# SPDX-License-Identifier: Apache-2.0
"""Run against a vLLM core with patches/vllm-0.25.1-...61023.patch applied."""
import dataclasses
import json
import pickle
from types import SimpleNamespace

import pytest

from vllm.config import AttentionConfig

pytestmark = pytest.mark.skipif(
    'sparse_mla_prefill_workspace_factor' not in AttentionConfig.__dataclass_fields__,
    reason='requires the official vLLM core source backport',
)


@pytest.mark.parametrize('factor', [2, 3, 10])
def test_official_field_roundtrip(factor):
    config = AttentionConfig(sparse_mla_prefill_workspace_factor=factor)
    assert dataclasses.asdict(config)['sparse_mla_prefill_workspace_factor'] == factor
    assert pickle.loads(pickle.dumps(config)).sparse_mla_prefill_workspace_factor == factor


def test_official_cli():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(['--attention-config', '{"sparse_mla_prefill_workspace_factor":10}'])
    assert args.attention_config.sparse_mla_prefill_workspace_factor == 10
    with pytest.raises(SystemExit):
        parser.parse_args(['--attention-config', '{"sparse_mla_prefill_workspace_factor":1}'])
    with pytest.raises(SystemExit):
        parser.parse_args(['--attention-config', '{"sparse_mla_prefill_workspace_factor":10,"bad_key":1}'])


@pytest.mark.parametrize('length,factor', [(89088,5),(262144,5),(262145,2),(1048576,2)])
def test_real_vllm_config_default(tmp_path, length, factor):
    from vllm.config import VllmConfig, ModelConfig, DeviceConfig
    (tmp_path/'config.json').write_text(json.dumps(dict(
        model_type='qwen2', architectures=['Qwen2ForCausalLM'], hidden_size=64,
        intermediate_size=128, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, vocab_size=256, max_position_embeddings=1048576,
        torch_dtype='float32')))
    model = ModelConfig(model=str(tmp_path), tokenizer=str(tmp_path),
                        max_model_len=length, dtype='float32', enforce_eager=True)
    config = VllmConfig(model_config=model, device_config=DeviceConfig('cpu'))
    assert config.attention_config.sparse_mla_prefill_workspace_factor == factor
    model.max_model_len = 88000
    assert pickle.loads(pickle.dumps(config.attention_config)).sparse_mla_prefill_workspace_factor == factor


@pytest.mark.parametrize('factor', [2, 5, 10])
def test_actual_core_sizing(factor):
    from vllm.v1.attention.backends.mla.flashmla_sparse import get_prefill_workspace_size
    from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size
    cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=89088),
                          attention_config=AttentionConfig(sparse_mla_prefill_workspace_factor=factor))
    assert get_prefill_workspace_size(cfg) == 89088 * factor
    assert get_max_prefill_buffer_size(cfg) == 89088 * factor * 8
