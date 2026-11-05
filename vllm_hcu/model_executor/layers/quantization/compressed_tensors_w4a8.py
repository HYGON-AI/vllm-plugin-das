# SPDX-License-Identifier: Apache-2.0
"""HCU execution adapter for compressed-tensors channel INT4 / token INT8."""

from copy import copy

import torch

from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
    SlimQuantW4A8Int8AiterMoEMethod,
)


def convert_packed_int4(weight):
    if weight.dtype != torch.int32 or weight.ndim < 1:
        raise ValueError("Expected compressed-tensors int32 packed weights")
    raw = weight.contiguous().view(torch.uint8)
    return ((((raw & 15) << 4) | (raw >> 4)) ^ 0x88).view(torch.int8)


def checkpoint_weights(model, weights):
    """Rename packed tensors only for owners using this adapter."""
    owners = {
        name: module
        for name, module in model.named_modules()
        if isinstance(getattr(module, "quant_method", None), HcuCompressedW4A8)
    }
    for name, weight in weights:
        if ".experts." in name:
            owner = name.split(".experts.", 1)[0] + ".experts"
            # The MTP loader inserts .mtp_block. after reading the checkpoint.
            module = owners.get(owner) or owners.get(
                owner.replace(".mlp.experts", ".mtp_block.mlp.experts")
            )
            if module is not None:
                if name.endswith(".weight_shape"):
                    continue
                if name.endswith(".weight_packed"):
                    name = name.removesuffix("_packed")
        yield name, weight


def mtp_quant_config(config, layer_prefix):
    """Alias GLM decoder rules without changing attention/layer identities."""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        find_matched_target,
        should_ignore_layer,
    )

    if not isinstance(config, CompressedTensorsConfig):
        return config
    result = copy(config)
    original = result.get_scheme_dict

    def get_scheme_dict(layer, layer_name=None):
        if layer_name is not None and layer_name.startswith(layer_prefix + "."):
            if should_ignore_layer(
                layer_name, result.ignore, result.packed_modules_mapping
            ):
                return None
            targets = result.target_scheme_map.keys()
            physical = find_matched_target(
                layer_name, layer, targets, result.packed_modules_mapping
            )
            class_default = find_matched_target(None, layer, targets)
            # Explicit physical rules take priority over the decoder alias;
            # a generic class fallback must not hide canonical INT4 rules.
            if physical == class_default:
                layer_name = "model.decoder" + layer_name[len(layer_prefix) :]
        return original(layer, layer_name)

    result.get_scheme_dict = get_scheme_dict
    return result


def select_method(config, layer, prefix):
    """Use native parsed schemes, priority, formats and ignore rules."""
    config._add_fused_moe_to_target_scheme_map()
    count = layer.moe_config.num_logical_experts
    schemes = [
        config.get_scheme_dict(layer, f"{prefix}.{expert}.{projection}_proj")
        for expert in range(count)
        for projection in ("gate", "up", "down")
    ]

    def supported(scheme):
        if scheme is None:
            return False
        w, a = scheme.get("weights"), scheme.get("input_activations")
        return bool(
            w is not None
            and a is not None
            and scheme.get("format") == "pack-quantized"
            and w.num_bits == 4
            and w.type == "int"
            and w.strategy == "channel"
            and w.symmetric
            and not w.dynamic
            and w.actorder is None
            and a.num_bits == 8
            and a.type == "int"
            and a.strategy == "token"
            and a.symmetric
            and a.dynamic is True
        )

    selected = [supported(s) for s in schemes]
    if not any(selected):
        return None
    if not all(selected) or any(s != schemes[0] for s in schemes):
        raise ValueError(
            "HCU W4A8 requires the same scheme for every expert projection"
        )
    if schemes[0].get("output_activations") is not None:
        raise ValueError("HCU W4A8 does not support output activation quantization")
    return HcuCompressedW4A8(config, layer.moe_config)


class HcuCompressedW4A8(SlimQuantW4A8Int8AiterMoEMethod):
    """Keep native config selection and reuse the packed SlimQuant Triton path.

    AITER gemm2 installation and online weight replacement are deliberately
    outside this initial adapter: neither can safely reuse arbitrary packed
    weights without a separately validated lifecycle contract.
    """

    @property
    def supports_eplb(self):
        return False

    def __init__(self, config, moe):
        if moe.moe_backend != "triton":
            raise ValueError(
                "compressed-tensors channel W4A8 requires --moe-backend triton"
            )
        parallel = moe.moe_parallel_config
        if parallel.ep_size != 1:
            raise ValueError("compressed-tensors channel W4A8 currently requires EP=1")
        if getattr(moe, "aiter_fmoe_shared_expert_enabled", False):
            raise ValueError(
                "compressed-tensors channel W4A8 requires shared-expert fusion disabled"
            )
        if moe.has_bias:
            raise ValueError(
                "compressed-tensors channel W4A8 does not support expert bias"
            )
        super().__init__(config, moe)

    def create_weights(self, layer, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            param = getattr(layer, name)
            original = param.weight_loader
            is_scale = name.endswith("scale")

            def loader(
                param, loaded_weight, *args, _loader=original, _scale=is_scale, **kwargs
            ):
                value = (
                    loaded_weight.to(param.dtype) / 16
                    if _scale
                    else convert_packed_int4(loaded_weight)
                )
                return _loader(param, value, *args, **kwargs)

            param.weight_loader = loader
