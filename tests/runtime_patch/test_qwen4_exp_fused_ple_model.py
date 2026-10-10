# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

import pytest
import torch
from vllm.model_executor.models.utils import WeightsMapper

from vllm_hcu.patch.worker.core_fix import patch_qwen4_exp_fused_ple as patch
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


def _target_module() -> tuple[ModuleType, type, type]:
    class Qwen4ExpPLELayer:
        pass

    Qwen4ExpPLELayer.__module__ = "vllm_hcu.models.qwen4_exp.amd.ple_layer"

    class Qwen4ExpDecoderLayer:
        def forward(
            self,
            hidden_states,
            prev_block_output,
            prev_injection,
            positions,
            *,
            input_ids,
            query_start_loc,
            ngram_context,
        ):
            del (
                prev_block_output,
                prev_injection,
                positions,
                input_ids,
                query_start_loc,
                ngram_context,
            )
            return hidden_states + 100, hidden_states, hidden_states

    class Qwen4ExpModel:
        hf_to_vllm_mapper = WeightsMapper()

    class Qwen4ExpForCausalLM:
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    class Qwen4ExpForConditionalGeneration:
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    class Qwen4ExpMTP:
        def forward(self, hidden_states):
            return hidden_states

    module = ModuleType(patch.TARGET_MODULE)
    module.Qwen4ExpPLELayer = Qwen4ExpPLELayer
    module.Qwen4ExpDecoderLayer = Qwen4ExpDecoderLayer
    module.Qwen4ExpModel = Qwen4ExpModel
    module.Qwen4ExpForCausalLM = Qwen4ExpForCausalLM
    module.Qwen4ExpForConditionalGeneration = Qwen4ExpForConditionalGeneration
    module.Qwen4ExpMTP = Qwen4ExpMTP
    return module, Qwen4ExpDecoderLayer, Qwen4ExpMTP


def test_fused_ple_uses_one_replicated_kv_projection() -> None:
    ple_layer = importlib.import_module("vllm_hcu.models.qwen4_exp.amd.ple_layer")
    source = inspect.getsource(ple_layer.Qwen4ExpPLELayer.__init__)

    assert "self.kv_proj = MergedColumnParallelLinear(" in source
    assert "disable_tp=True" in source
    assert "self.key_proj =" not in source
    assert "self.value_proj =" not in source


def test_key_and_value_checkpoint_weights_map_to_kv_shards() -> None:
    module, _, _ = _target_module()
    assert patch.apply_to_module(module) is True
    mapper = module.Qwen4ExpModel.hf_to_vllm_mapper

    assert mapper._map_name_with_shard("model.layers.2.ple.key_proj.weight") == (
        "model.layers.2.ple.kv_proj.weight",
        0,
    )
    assert mapper._map_name_with_shard("model.layers.2.ple.value_proj.weight") == (
        "model.layers.2.ple.kv_proj.weight",
        1,
    )


def test_quantization_excludes_match_original_projection_names() -> None:
    module, _, _ = _target_module()
    patch.apply_to_module(module)

    expected = ["key_proj", "value_proj"]
    assert module.Qwen4ExpForCausalLM.packed_modules_mapping["kv_proj"] == expected
    assert (
        module.Qwen4ExpForConditionalGeneration.packed_modules_mapping["kv_proj"]
        == expected
    )


class _HyperConnection:
    def combine(self, hidden_states, block_output, injection):
        return hidden_states + block_output + injection

    def mix(self, hidden_states):
        return hidden_states, hidden_states, torch.zeros_like(hidden_states)

    def combine_and_mix(self, hidden_states, block_output, injection):
        del block_output
        return hidden_states, hidden_states, injection


class _PLE:
    def __init__(self):
        self.inputs = []

    def __call__(self, hidden_states, input_ids, query_start_loc, ngram_context):
        self.inputs.append(
            (hidden_states.clone(), input_ids, query_start_loc, ngram_context)
        )
        return hidden_states + 3


def test_decoder_adds_residual_exactly_once() -> None:
    module, decoder_cls, _ = _target_module()
    patch.apply_to_module(module)
    decoder = decoder_cls()
    decoder.ple = _PLE()
    decoder.attn_hyper_connection = _HyperConnection()
    decoder.mlp_hyper_connection = _HyperConnection()
    decoder.layer_type = "full_attention"
    decoder.self_attn = lambda **kwargs: kwargs["hidden_states"]
    decoder.mlp = lambda hidden_states: hidden_states
    hidden_states = torch.full((2, 4), 2.0)

    materialized, _, _ = decoder.forward(
        hidden_states,
        None,
        None,
        torch.arange(2),
        input_ids=torch.tensor([1, 2]),
        query_start_loc=torch.tensor([0, 2]),
        ngram_context=torch.tensor([[3, 4]]),
    )

    assert torch.equal(materialized, torch.full_like(hidden_states, 5.0))
    assert torch.equal(decoder.ple.inputs[0][0], hidden_states)


def test_fused_ple_does_not_wrap_mtp_forward() -> None:
    module, _, mtp_cls = _target_module()
    original = mtp_cls.forward

    assert patch.apply_to_module(module) is True
    assert patch.apply_to_module(module) is False
    assert mtp_cls.forward is original


def test_fused_ple_patch_fails_closed_on_signature_drift() -> None:
    module, decoder_cls, _ = _target_module()

    def drifted_forward(
        self,
        hidden_states,
        prev_block_output,
        prev_injection,
        positions,
        *,
        input_ids,
        query_start_loc,
    ):
        del self, prev_block_output, prev_injection, positions
        del input_ids, query_start_loc
        return hidden_states, hidden_states, hidden_states

    decoder_cls.forward = drifted_forward
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        patch.apply_to_module(module)
