# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import copy
import importlib
from types import ModuleType, SimpleNamespace
from typing import get_args

import pytest
import torch

from vllm_hcu.models.hy_v4.config import HYV4Config


def _module(name):
    assert importlib.util.find_spec(name) is not None, f"Missing native MTP module: {name}"
    return importlib.import_module(name)


def _mtp():
    return _module("vllm_hcu.models.hy_v4.mtp")


@pytest.mark.parametrize("scheme,block", [("static", None), ("dynamic", [128, 128])])
def test_mtp_config_extends_one_layer_and_preserves_activation_scheme(scheme, block):
    mtp = _mtp()
    target = HYV4Config(num_hidden_layers=2, layer_types=["full_attention", "sparse"])
    target.quantization_config = {"activation_scheme": scheme, "weight_block_size": block}
    before = copy.deepcopy(target.to_dict())
    draft = mtp._make_mtp_layer_config(target, 2)
    assert draft.layer_types == ["full_attention", "deepseek_sparse_attention", "deepseek_sparse_attention"]
    assert draft.mlp_layer_types == ["dense", "sparse", "sparse"]
    assert draft.num_hidden_layers == 2
    assert draft.indexer_types == ["full", "full"]
    assert draft.enable_ihc is False
    assert draft.quantization_config == {"activation_scheme": scheme, "weight_block_size": block}
    assert draft.head_dtype == "float32"
    assert target.to_dict() == before


@pytest.mark.parametrize("algo", [None, "NONE", "FP8"])
def test_mtp_channel_quantization_inherits_current_owner(algo):
    mtp = _mtp()
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig

    backbone = CompressedTensorsConfig.from_config({
        "format": "float-quantized", "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "symmetric": True},
            "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "symmetric": True, "dynamic": True}}},
        "ignore": ["lm_head", "model.mtp_layers.0.self_attn.linear_gate"],
    })
    config = SimpleNamespace(mtp_quant_algo=algo, quantization_config={"quant_method": "compressed-tensors"})
    assert mtp._create_mtp_quant_config(config, backbone) is backbone
    draft = mtp._remap_mtp_quant_exclusions(backbone, 2, 1)
    assert draft is not backbone
    assert "model.layers.2.self_attn.linear_gate" in draft.ignore
    assert "model.layers.2.self_attn.linear_gate" not in backbone.ignore
    assert draft.target_scheme_map == backbone.target_scheme_map


@pytest.mark.parametrize("scheme,block", [("static", None), ("dynamic", [128, 128])])
def test_mtp_native_fp8_preserves_checkpoint_quantization(scheme, block):
    mtp = _mtp()
    config = SimpleNamespace(mtp_quant_algo="FP8", quantization_config={
        "activation_scheme": scheme, "weight_block_size": block, "scale_fmt": "ue8m0"})
    result = mtp._create_mtp_quant_config(config)
    assert result.activation_scheme == scheme
    assert result.weight_block_size == block
    assert result.is_scale_e8m0 is True


@pytest.mark.parametrize("algo", ["BF16", "FP16"])
def test_mtp_explicit_unquantized_draft(algo):
    assert _mtp()._create_mtp_quant_config(SimpleNamespace(mtp_quant_algo=algo), object()) is None


def test_mtp_unknown_quantization_fails_closed():
    with pytest.raises(ValueError, match="Unsupported HYV4"):
        _mtp()._create_mtp_quant_config(SimpleNamespace(mtp_quant_algo="INT3"), object())


@pytest.mark.parametrize("layer_idx,length", [(3, 2), (2, 4)])
def test_mtp_rejects_non_single_layer_extension(layer_idx, length):
    mtp = _mtp()
    config = HYV4Config(num_hidden_layers=2)
    config.layer_types = ["full_attention"] * length
    with pytest.raises(ValueError, match="exactly one"):
        mtp._make_mtp_layer_config(config, layer_idx)


def test_mtp_config_adapter_uses_current_speculative_owner_and_registers_callback():
    adapter = _module("vllm_hcu.patch.platform.core_fix.patch_hy_v4_mtp_config")
    import vllm.config.speculative as speculative
    from vllm_hcu.patch.platform.core_fix import platform_core_callback_names

    adapter.apply_to_module(speculative)
    target = HYV4Config(num_hidden_layers=2, architectures=["HYV4ForCausalLM"])
    target.quantization_config = {"activation_scheme": "static"}
    draft = speculative.SpeculativeConfig.hf_config_override(copy.deepcopy(target))
    assert draft.model_type == "hy_v4_mtp"
    assert draft.architectures == ["HYV4MTPModel"]
    assert draft.n_predict == 1 and draft.hc_mult == 1
    assert draft.quantization_config == target.quantization_config
    assert target.hc_mult == 4 and target.model_type == "hy_v4"
    assert "hy_v4_mtp" in get_args(speculative.MTPModelTypes)
    assert (adapter.PATCH_ID, adapter.TARGET_MODULE) in platform_core_callback_names()
    assert adapter.apply_to_module(speculative) is False
    unrelated = SimpleNamespace(model_type="unrelated", architectures=["UnrelatedModel"])
    assert speculative.SpeculativeConfig.hf_config_override(unrelated) is unrelated


@pytest.mark.parametrize("count", [0, 2])
def test_mtp_config_rejects_checkpoint_without_exactly_one_draft_layer(count):
    adapter = _module("vllm_hcu.patch.platform.core_fix.patch_hy_v4_mtp_config")
    import vllm.config.speculative as speculative
    adapter.apply_to_module(speculative)
    with pytest.raises(ValueError, match="exactly one"):
        speculative.SpeculativeConfig.hf_config_override(HYV4Config(
            num_nextn_predict_layers=count, architectures=["HYV4ForCausalLM"]))


def test_mtp_adapter_rejects_wrong_owner_and_api():
    adapter = _module("vllm_hcu.patch.platform.core_fix.patch_hy_v4_mtp_config")
    from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError
    for name in ("wrong.owner", adapter.TARGET_MODULE):
        with pytest.raises(PatchCompatibilityError):
            adapter.apply_to_module(ModuleType(name))


def test_mtp_model_config_honors_head_dtype():
    import vllm.config.model as module
    from vllm_hcu.patch.platform.core_fix import patch_hy_v4_model_head_dtype
    patch_hy_v4_model_head_dtype.apply_to_module(module)
    config = object.__new__(module.ModelConfig)
    config.hf_config = SimpleNamespace(model_type="hy_v4_mtp", head_dtype="float32")
    config.dtype = torch.bfloat16
    config.runner_type = "generate"
    assert config.head_dtype is torch.float32
