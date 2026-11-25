# SPDX-License-Identifier: Apache-2.0
"""Installed model-local CT admission; no device or provider is simulated.

The first regression catches ignoring explicit AITER selection for Flash routed
experts. Real Core CT parsing is retained; only the active configuration context
is supplied by the test. Native arithmetic/Graph qualification is separate.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest


def _recipe():
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
        "config_groups": {
            "fp8": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "channel",
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                },
            }
        },
    }


def _current(backend):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            enable_expert_parallel=False,
            enable_elastic_ep=False,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
            use_sequence_parallel_moe=False,
            enable_eplb=False,
            enable_dbo=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
        lora_config=None,
        kernel_config=SimpleNamespace(moe_backend=backend),
        model_config=SimpleNamespace(enforce_eager=False),
    )


@pytest.mark.parametrize("for_moe", [False, True], ids=["dense", "experts"])
@pytest.mark.parametrize("backend", ["auto", "triton", "aiter"])
def test_explicit_aiter_is_selected_only_for_flash_experts(
    monkeypatch, backend, for_moe
):
    import vllm.config
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    context = _current(backend)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: context)
    source = CompressedTensorsConfig.from_config(_recipe())
    before = deepcopy(source.config)
    targets = deepcopy(source.target_scheme_map)
    selected = prepare_flash_quant_config(
        SimpleNamespace(flash_w4a8_backend="disabled"),
        source,
        for_moe=for_moe,
    )
    if backend == "aiter" and for_moe:
        assert selected is not source, (
            "Explicit AITER selection was ignored for routed experts"
        )
        assert isinstance(selected, CompressedTensorsConfig)
        # Actual per-model copy, retaining Core loader configuration and leaving
        # dense/DFly owners untouched. Re-running preparation is idempotent.
        assert selected.config == before
        assert selected.packed_modules_mapping == source.packed_modules_mapping
        assert (
            prepare_flash_quant_config(
                SimpleNamespace(flash_w4a8_backend="disabled"), selected, for_moe=True
            )
            is selected
        )
    else:
        assert selected is source
    assert source.config == before and source.target_scheme_map == targets


def _expert(torch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.layer import RoutedExperts

    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(
        num_logical_experts=2,
        is_act_and_mul=True,
        activation=MoEActivation.SILU,
        moe_backend="aiter",
    )
    layer.ckpt_gate_proj_name = "gate_proj"
    layer.ckpt_up_proj_name = "up_proj"
    layer.ckpt_down_proj_name = "down_proj"
    return layer


def test_real_ct_expert_parameters_retain_loader_and_channel_scales(monkeypatch):
    import torch
    from hcu_vllm_das.models.hy_v4_flash import aiter_fp8_moe as provider
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    current = _current("aiter")
    monkeypatch.setattr(provider, "get_current_vllm_config", lambda: current)
    source = CompressedTensorsConfig.from_config(_recipe())
    config = provider.HYV4FlashAiterFP8Config.for_moe(source)
    layer = _expert(torch)
    method = config.get_quant_method(layer, "model.layers.1.mlp.experts")
    # Exercise the inherited Core allocation/parameter ABI on CPU, without
    # pretending this tiny loader fixture is an admitted native MoE geometry.
    loader = object()
    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=2,
        params_dtype=torch.bfloat16,
        weight_loader=loader,
    )
    assert layer.w13_weight.shape == (2, 4, 4)
    assert layer.w2_weight.shape == (2, 4, 2)
    assert layer.w13_weight.dtype == layer.w2_weight.dtype == torch.float8_e4m3fn
    assert layer.w13_weight_scale.shape == (2, 4, 1)
    assert layer.w2_weight_scale.shape == (2, 4, 1)
    assert layer.w13_weight_scale.dtype == layer.w2_weight_scale.dtype == torch.float32
    assert layer.w13_weight.weight_loader is loader
    assert layer.w2_weight.weight_loader is loader
    assert layer.w13_input_scale is layer.w2_input_scale is None
    assert not method.supports_eplb
    dense = torch.nn.Linear(4, 4)
    assert config.get_quant_method(dense, "dense") is source.get_quant_method(
        dense, "dense"
    )


def test_foreign_device_refuses_before_loaded_tensor_access(monkeypatch):
    import torch
    from compressed_tensors.quantization import QuantizationArgs
    from hcu_vllm_das.models.hy_v4_flash import aiter_fp8_moe as provider

    recipe = _recipe()["config_groups"]["fp8"]
    method = provider.HYV4FlashAiterFP8Method(
        QuantizationArgs(**recipe["weights"]),
        QuantizationArgs(**recipe["input_activations"]),
        SimpleNamespace(is_act_and_mul=True),
    )
    monkeypatch.setattr(provider, "get_current_vllm_config", lambda: _current("aiter"))
    monkeypatch.setattr(provider, "_is_gfx938", lambda: False)
    with pytest.raises(NotImplementedError, match="gfx938"):
        method.process_weights_after_loading(torch.nn.Module())


def test_postload_queries_public_vendor_config_before_selecting_layout(monkeypatch):
    import sys
    from types import ModuleType

    import torch
    from compressed_tensors.quantization import QuantizationArgs
    from hcu_vllm_das.models.hy_v4_flash import aiter_fp8_moe as provider
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        get_routing_method_type,
    )

    # Configuration lookup and native packing are external GPU-provider leaves.
    # Keep the real method, Core quant config and expert constructor, replacing
    # only vendor leaves and group/workspace construction for this CPU boundary.
    calls = []
    vendor = ModuleType("aiter.moe")
    vendor.AiterMoeConfig = SimpleNamespace
    vendor.MoeQuantType = SimpleNamespace(FP8_W8A8="fp8_w8a8")
    vendor.MoeSolutionType = SimpleNamespace(ASM="asm")

    def get_config(*args, **kwargs):
        calls.append((args, kwargs))
        return True, SimpleNamespace(
            quant_type="fp8_w8a8",
            solution_type="asm",
            config={"SOL_ID1": 11007, "SOL_ID2": 21000, "BLOCK_SIZE_M": 32},
            need_shuffle=bool(kwargs.get("use_shuffle", 0)),
            need_shuffle_scale=False,
            dtype=torch.bfloat16,
        )

    vendor.get_aiter_moe_config = get_config
    vendor.aiter_moe_shfl_weight = lambda a, b, cfg: (a.clone(), b.clone())
    parent = ModuleType("aiter")
    parent.__path__ = []
    parent.moe = vendor
    monkeypatch.setitem(sys.modules, "aiter", parent)
    monkeypatch.setitem(sys.modules, "aiter.moe", vendor)
    monkeypatch.setattr(provider, "get_current_vllm_config", lambda: _current("aiter"))
    monkeypatch.setattr(provider, "_is_gfx938", lambda: True)

    def factory(q, moe, expert_class, backend, routing_tables=None):
        return SimpleNamespace(fused_experts=expert_class(moe, q))

    monkeypatch.setattr(provider, "make_fp8_moe_kernel", factory)
    recipe = _recipe()["config_groups"]["fp8"]
    parallel = FusedMoEParallelConfig(
        tp_size=4,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="naive",
        enable_eplb=False,
    )
    moe = FusedMoEConfig(
        num_experts=512,
        experts_per_token=16,
        hidden_dim=1536,
        intermediate_size=1536,
        num_local_experts=512,
        num_logical_experts=512,
        activation=MoEActivation.SILU,
        device="cpu",
        in_dtype=torch.bfloat16,
        moe_backend="aiter",
        moe_parallel_config=parallel,
        swiglu_limit=10.0,
        routing_method=get_routing_method_type(
            scoring_func="sigmoid",
            top_k=16,
            renormalize=True,
            num_expert_group=1,
            has_e_score_bias=True,
        ),
    )
    method = provider.HYV4FlashAiterFP8Method(
        QuantizationArgs(**recipe["weights"]),
        QuantizationArgs(**recipe["input_activations"]),
        moe,
    )
    fields = {
        "activation": MoEActivation.SILU,
        "swiglu_limit": 10.0,
        "apply_router_weight_on_input": False,
        "expert_map": None,
        "w13_weight": torch.empty(
            512, 768, 1536, dtype=torch.float8_e4m3fn, device="meta"
        ),
        "w2_weight": torch.empty(
            512, 1536, 384, dtype=torch.float8_e4m3fn, device="meta"
        ),
        "w13_weight_scale": torch.empty(512, 768, 1, device="meta"),
        "w2_weight_scale": torch.empty(512, 1536, 1, device="meta"),
        "_expert_routing_tables": lambda: None,
    }
    layer = torch.nn.Module()
    for name, value in fields.items():
        if isinstance(value, torch.Tensor):
            value = torch.nn.Parameter(value, requires_grad=False)
        setattr(layer, name, value)
    method.process_weights_after_loading(layer)
    assert calls, "MoE postload bypassed the public AITER config query"
    args, kwargs = calls[0]
    assert args[:6] == (1, 512, 768, 1536, 1536, 16)
    assert kwargs["activation"] == "silu"


def test_shuffle_policy_is_frozen_and_official_model_hash_separates_layout(tmp_path):
    from hcu_vllm_das.models.hy_v4_flash.aiter_fp8_moe import HYV4FlashAiterFP8Config
    from transformers import LlamaConfig
    from vllm.config import ModelConfig
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    source = CompressedTensorsConfig.from_config(_recipe())
    original_recipe = deepcopy(source.config)
    raw = HYV4FlashAiterFP8Config.for_moe(source, use_shuffle=False)
    shuffled = HYV4FlashAiterFP8Config.for_moe(source, use_shuffle=True)
    assert source.config == original_recipe
    assert HYV4FlashAiterFP8Config.for_moe(raw, use_shuffle=False) is raw
    assert HYV4FlashAiterFP8Config.for_moe(shuffled, use_shuffle=True) is shuffled
    with pytest.raises(ValueError, match="frozen"):
        HYV4FlashAiterFP8Config.for_moe(raw, use_shuffle=True)
    with pytest.raises(TypeError, match="boolean"):
        HYV4FlashAiterFP8Config.for_moe(source, use_shuffle="1")
    model = tmp_path / "tiny-model"
    model.mkdir()
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=128,
    ).save_pretrained(model)
    configs = [
        ModelConfig(
            model=str(model),
            tokenizer=str(model),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=128,
            hf_overrides={"flash_aiter_moe_use_shuffle": flag},
        )
        for flag in (False, True)
    ]
    assert configs[0].hf_config.flash_aiter_moe_use_shuffle is False
    assert configs[1].hf_config.flash_aiter_moe_use_shuffle is True
    assert configs[0].compute_hash() != configs[1].compute_hash()
