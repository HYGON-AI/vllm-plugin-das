"""Exercise upstream metadata/materialization/replay with real expert loaders."""

import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from compressed_tensors.quantization import QuantizationArgs

from tests.models.static_eplb_test_utils import GenericMoE
from vllm.model_executor.model_loader.reload import layerwise, meta
from vllm_hcu.model_executor.layers.fused_moe.static_eplb import bind_static_eplb_plan
from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8_moe import (
    HcuCompressedTensorsW4A8Int8MoEMethod,
    repack_compressed_tensors_int4,
)

PARAMETERS = (
    "w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
)


def make_model(monkeypatch, tmp_path, static):
    rows = [[1, 0, 2, 3], [0, 1, 3, 2]]
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"model_maps": {
        "GenericMoE": {"physical_to_logical_map": rows},
    }}))
    parallel = SimpleNamespace(
        enable_eplb=static, enable_expert_parallel=True,
        tensor_parallel_size=1, pipeline_parallel_size=1,
        enable_ep_weight_filter=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        _vllm_hcu_expert_map_path=str(path) if static else None,
    )
    config = SimpleNamespace(parallel_config=parallel)
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(
        "vllm_hcu.model_executor.layers.fused_moe.deepep_runtime."
        "slimquant_w4a8_uses_deepep_ht", lambda _: True,
    )
    model = GenericMoE(0)
    model.num_logical_experts = model.num_routed_experts = 4
    model.num_redundant_experts = 0
    for layer in model.moe_layers:
        owner = layer.routed_experts
        moe = SimpleNamespace(
            moe_backend="deep_gemm", has_bias=False,
            hidden_dim_unpadded=8, intermediate_size_per_partition_unpadded=8,
            num_logical_experts=4, num_experts=4, tp_rank=0, tp_size=1,
            is_act_and_mul=True,
            moe_parallel_config=SimpleNamespace(
                tp_size=1, pcp_size=16, dp_size=1, ep_size=16,
                all2all_backend="deepep_high_throughput", use_ep=True,
            ),
        )
        quant = HcuCompressedTensorsW4A8Int8MoEMethod(
            QuantizationArgs(
                num_bits=4, type="int", strategy="channel", symmetric=True,
                dynamic=False, group_size=-1,
            ),
            QuantizationArgs(
                num_bits=8, type="int", strategy="token", symmetric=True,
                dynamic=True,
            ),
            moe,
        )
        del owner.quant_method
        owner.quant_method = quant
        owner.moe_config = moe
        for name in PARAMETERS:
            delattr(owner, name)
        quant.create_weights(
            owner, num_experts=2, hidden_size=8,
            intermediate_size_per_partition=8, params_dtype=torch.bfloat16,
            weight_loader=owner.weight_loader,
        )
    if static:
        bind_static_eplb_plan(config, model)
    return model, rows[0][:2] if static else [0, 1]


@pytest.mark.parametrize("style", ["positional", "keyword"])
@pytest.mark.parametrize("static", [False, True], ids=["eplb_disabled", "static_map"])
def test_layerwise_reload_retargets_all_weights_and_scales(
    monkeypatch, tmp_path, style, static,
):
    with torch.device("cpu"):
        model, local_logical = make_model(monkeypatch, tmp_path, static)
        owner = model.moe_layers[0].routed_experts
        if not static:
            assert owner.quant_method.supports_eplb is False
        kernel_parameters = {name: getattr(owner, name) for name in PARAMETERS}
        with torch.no_grad():
            for parameter in kernel_parameters.values():
                parameter.fill_(19)
        storage = {name: param.data_ptr() for name, param in kernel_parameters.items()}

        # Only GPU-specific postprocessing is isolated; upstream capture,
        # sanitation, buffering, materialization, replay and storage copy run.
        processed = []
        monkeypatch.setattr(owner.quant_method, "process_weights_after_loading",
                            lambda layer: processed.append({
                                name: getattr(layer, name).detach().clone()
                                for name in PARAMETERS
                            }))
        materialize = meta.materialize_meta_tensor
        materialized = []

        def materialize_with_sentinel(tensor):
            result = materialize(tensor)
            with torch.no_grad():
                result.fill_(313 if result.is_floating_point() else -103)
            materialized.append(result)
            return result

        monkeypatch.setattr(meta, "materialize_meta_tensor", materialize_with_sentinel)
        layerwise.record_metadata_for_reloading(owner)
        info = layerwise.get_layerwise_info(owner)
        assert all(param.is_meta for param in info.restore_metadata[0].values())

        for round_index in range(2):
            layerwise.initialize_layerwise_reload(owner)
            assert all(getattr(owner, name).is_meta for name in PARAMETERS)
            expected = {name: torch.empty_like(param)
                        for name, param in kernel_parameters.items()}
            for logical in range(2):
                for shard, name in [
                    ("w1", "w13_weight"), ("w3", "w13_weight"),
                    ("w2", "w2_weight"),
                ]:
                    value = logical + round_index - (3 if shard == "w3" else 1)
                    packed = pack_to_int32(torch.full((8, 8), value, dtype=torch.int8), 4)
                    scale = torch.full((8, 1), float(logical + round_index + 2))
                    local = local_logical.index(logical)
                    target = expected[name][local]
                    target_scale = expected[name + "_scale"][local]
                    if name == "w13_weight":
                        offset = 0 if shard == "w1" else 8
                        target = target[offset:offset + 8]
                        target_scale = target_scale[offset:offset + 8]
                    target.copy_(repack_compressed_tensors_int4(packed))
                    target_scale.copy_(scale / 16)
                    for param_name, checkpoint in [(name, packed), (name + "_scale", scale)]:
                        param = getattr(owner, param_name)
                        if style == "positional":
                            param.weight_loader(param, checkpoint, param_name, shard,
                                                logical, True)
                        else:
                            param.weight_loader(
                                param=param, loaded_weight=checkpoint,
                                weight_name=param_name, shard_id=shard,
                                expert_id=logical, return_success=True,
                            )
            layerwise.finalize_layerwise_reload(owner, SimpleNamespace())
            assert len(processed) == round_index + 1
            assert len(materialized) == 4 * (round_index + 1)
            mismatches = [name for name in PARAMETERS if not torch.equal(
                getattr(owner, name), expected[name])]
            assert not mismatches, f"Reload replay left wrong materialized data: {mismatches}"
            for name in PARAMETERS:
                torch.testing.assert_close(processed[-1][name], expected[name])
                assert getattr(owner, name) is kernel_parameters[name]
                assert getattr(owner, name).data_ptr() == storage[name]
            assert info.can_load() is False


def test_composed_bound_loader_exposes_reload_parameter_names(monkeypatch, tmp_path):
    with torch.device("cpu"):
        model, _ = make_model(monkeypatch, tmp_path, static=False)
    owner = model.moe_layers[0].routed_experts
    for name in PARAMETERS:
        loader = getattr(owner, name).weight_loader
        assert loader.__self__ is owner
        assert callable(loader.__func__)
        assert list(inspect.signature(loader).parameters)[:2] == ["param", "loaded_weight"]
