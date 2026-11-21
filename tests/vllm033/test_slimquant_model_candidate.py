# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import pytest
from vllm.models.kimi_k3.nvidia import model as upstream
from vllm.models.kimi_k3.nvidia import kda
from hcu_vllm_das.quantization import kimi31_load_model as candidate
from hcu_vllm_das.quantization.slimquant_load_only import Kimi31SlimQuantLoadOnlyConfig


def test_private_globals_and_unregistered_model():
    assert upstream.KimiLinearModel.__init__.__globals__["KimiDecoderLayer"] is upstream.KimiDecoderLayer
    assert kda.KimiK3DeltaAttention.__init__.__globals__["_KimiGDNMergedColumnParallelLinear"] is kda._KimiGDNMergedColumnParallelLinear
    assert candidate._Model.__init__.__globals__["KimiDecoderLayer"] is candidate._Decoder
    assert candidate._Delta.__init__.__globals__["_KimiGDNMergedColumnParallelLinear"] is candidate._mixed_projection
    assert candidate.Kimi31SlimQuantLoadOnly.__init__.__globals__["init_vllm_registered_model"] is candidate._registered_inner
    from vllm.model_executor.models.registry import ModelRegistry
    assert candidate.OUTER_ARCH not in ModelRegistry.models and candidate.INNER_ARCH not in ModelRegistry.models


def test_inner_admission_and_registry_delegation(monkeypatch):
    config=Kimi31SlimQuantLoadOnlyConfig({"quant_method":"slimquant_w4a8", "activation_scheme":"dynamic", "modules_to_not_convert":list(Kimi31SlimQuantLoadOnlyConfig._EXPECTED_IGNORES)})
    text=SimpleNamespace(min_moe_intermediate_per_partition=192)
    outer=SimpleNamespace(model_type="kimi_k31",text_config=text)
    vc=SimpleNamespace(model_config=SimpleNamespace(hf_config=outer),parallel_config=SimpleNamespace(tensor_parallel_size=16),quant_config=config)
    monkeypatch.setattr(candidate,"_original_inner_factory",lambda **kw:kw)
    result=candidate._registered_inner(vllm_config=vc,hf_config=text,architectures=["KimiLinearForCausalLM"])
    assert result["architectures"] == [candidate.INNER_ARCH]
    text.min_moe_intermediate_per_partition=256
    with pytest.raises(ValueError,match="unpadded"):
        candidate._registered_inner(vllm_config=vc,hf_config=text,architectures=["KimiLinearForCausalLM"])
    text.min_moe_intermediate_per_partition=192; vc.parallel_config.tensor_parallel_size=8
    with pytest.raises(ValueError,match="TP16"):
        candidate._registered_inner(vllm_config=vc,hf_config=text,architectures=["KimiLinearForCausalLM"])


def test_constructor_mismatch_and_forward_fail_closed():
    with pytest.raises(RuntimeError, match="binding changed"):
        candidate._clone(upstream.KimiLinearModel.__init__, [("KimiDecoderLayer", object(), object())])
    for cls in (candidate.Kimi31SlimQuantLoadOnly, candidate.Kimi31SlimQuantLoadOnlyInner):
        instance=cls.__new__(cls)
        with pytest.raises(RuntimeError, match="refuses forward"):
            cls.forward(instance)
