# SPDX-License-Identifier: Apache-2.0
"""CPU selection contracts for serialized groups; no kernel accuracy claims."""
from types import SimpleNamespace

import pytest

from tests.vllm033.test_slimquant import checkpoint_config


EXPERT = 'model.layers.0.mlp.experts'
PLE = 'model.layers.0.ple.ple_embedding.ngram_embedding'


def ple_config(monkeypatch, ignores, parts=4, declarations=None):
    import hcu_vllm_das.slimquant as implementation

    raw = checkpoint_config()
    raw['ignore'] = ignores
    if declarations is not None:
        raw['mixed_precision']['ngram_embedding']['modules'] = declarations
    context = SimpleNamespace(model_config=SimpleNamespace(
        hf_text_config=SimpleNamespace(split_ngram_parts=parts)))
    # Only the public metadata boundary is doubled. No hardware or Core method
    # is replaced; these are selector tests, separate from native PLE tests.
    monkeypatch.setattr(implementation, 'get_current_vllm_config_or_none',
                        lambda: context, raising=False)
    return implementation.SlimQuantConfig.from_config(raw)


@pytest.mark.parametrize('ignore', [PLE, PLE + '.shard_*', r're:.*[0-9]$'])
def test_uniform_ple_ignore_selects_bf16(monkeypatch, ignore):
    config = ple_config(monkeypatch, [ignore])
    assert type(config.get_embedding_method(PLE, object)).__name__ == 'Qwen4ExpPLEUnquantizedEmbeddingMethod'


@pytest.mark.parametrize('ignore', [PLE + '.shard_0', PLE + '.shard_3', r're:.*shard_[01]$'])
def test_partial_ple_ignore_refuses_single_storage_method(monkeypatch, ignore):
    config = ple_config(monkeypatch, [ignore])
    with pytest.raises(ValueError, match='partially ignored PLE'):
        config.get_embedding_method(PLE, object)


def test_partial_ple_declaration_refuses_single_storage_method(monkeypatch):
    config = ple_config(monkeypatch, [], declarations=[PLE + '.shard_0'])
    with pytest.raises(ValueError, match='partially declared PLE'):
        config.get_embedding_method(PLE, object)


@pytest.mark.parametrize('parts', [None, 0, -1, True])
def test_ple_requires_actual_positive_shard_metadata(monkeypatch, parts):
    config = ple_config(monkeypatch, [], parts=parts)
    with pytest.raises(ValueError, match='split_ngram_parts'):
        config.get_embedding_method(PLE, object)


def test_ple_does_not_cache_shard_count_across_contexts(monkeypatch):
    import hcu_vllm_das.slimquant as implementation

    config = ple_config(monkeypatch, [PLE + '.shard_0'], parts=1)
    assert type(config.get_embedding_method(PLE, object)).__name__ == 'Qwen4ExpPLEUnquantizedEmbeddingMethod'
    context = SimpleNamespace(model_config=SimpleNamespace(
        hf_text_config=SimpleNamespace(split_ngram_parts=2)))
    monkeypatch.setattr(implementation, 'get_current_vllm_config_or_none', lambda: context)
    with pytest.raises(ValueError, match='partially ignored PLE'):
        config.get_embedding_method(PLE, object)


def test_ple_none_declared_keeps_bf16(monkeypatch):
    config = ple_config(monkeypatch, [], declarations=['other.ple.shard_*'])
    assert type(config.get_embedding_method(PLE, object)).__name__ == 'Qwen4ExpPLEUnquantizedEmbeddingMethod'


def test_fully_ignored_ple_does_not_require_quantized_declarations(monkeypatch):
    config = ple_config(monkeypatch, [PLE + '.shard_*'], declarations=[PLE + '.shard_0'])
    assert type(config.get_embedding_method(PLE, object)).__name__ == 'Qwen4ExpPLEUnquantizedEmbeddingMethod'


@pytest.mark.parametrize('ignores, expected', [
    ([EXPERT], 'bf16'),
    ([EXPERT + '.gate_up_proj', EXPERT + '.down_proj'], 'bf16'),
    ([], 'quantized'),
])
def test_uniform_expert_ignore_selects_complete_method(monkeypatch, ignores, expected):
    import torch
    import hcu_vllm_das.slimquant as implementation

    raw = checkpoint_config()
    raw['ignore'] = ignores
    layer = object.__new__(implementation.RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = object()
    monkeypatch.setattr(implementation, 'UnquantizedFusedMoEMethod', lambda config: ('bf16', config))
    monkeypatch.setattr(implementation, 'SlimQuantInt8Experts', lambda *args: ('quantized', args[-1]))
    result = implementation.SlimQuantConfig.from_config(raw).get_quant_method(layer, EXPERT)
    assert result == (expected, layer.moe_config)


@pytest.mark.parametrize('projection', ['gate_up_proj', 'down_proj'])
def test_partial_expert_ignore_refuses_grouped_method(monkeypatch, projection):
    import torch
    import hcu_vllm_das.slimquant as implementation

    raw = checkpoint_config()
    raw['ignore'] = [EXPERT + '.' + projection]
    layer = object.__new__(implementation.RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = object()
    monkeypatch.setattr(implementation, 'SlimQuantInt8Experts', lambda *args: ('quantized', args[-1]))
    with pytest.raises(ValueError, match='partially ignored expert'):
        implementation.SlimQuantConfig.from_config(raw).get_quant_method(layer, EXPERT)
