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


def repack_aiter_gemm2(weight):
    """Regroup eight signed nibbles, then install AITER's per-expert layout."""
    from aiter.ops.shuffle import w4a8_moe_layout_shuffle_gemm2

    if weight.dtype != torch.int8 or weight.ndim != 3 or weight.shape[-1] % 4:
        raise ValueError("AITER W4A8 gemm2 requires rank-3 INT8 and K divisible by 8")
    raw = weight.contiguous().view(torch.uint8)
    nibbles = torch.stack((raw >> 4, raw & 15), dim=-1).reshape(*raw.shape[:-1], -1, 8)
    regrouped = ((nibbles[..., :4] << 4) | nibbles[..., 4:]).reshape_as(raw)
    return (
        torch.stack([w4a8_moe_layout_shuffle_gemm2(expert) for expert in regrouped])
        .reshape_as(weight)
        .view(torch.int8)
    )


def checkpoint_weights(model, weights):
    """Rename packed tensors only for owners using this adapter."""
    owners = {
        # MoERunner owns the RoutedExperts child; checkpoint names address
        # the enclosing .experts owner instead of .experts.routed_experts.
        name.removesuffix(".routed_experts"): module
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

    def ignored(name):
        candidates = [name]
        if name is not None and ".experts." in name:
            candidates.append(name.split(".experts.", 1)[0] + ".experts")
        return any(
            should_ignore_layer(n, result.ignore, result.packed_modules_mapping)
            for n in candidates
        )

    def get_scheme_dict(layer, layer_name=None):
        if layer_name is not None and layer_name.startswith(layer_prefix + "."):
            if ignored(layer_name):
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
        if ignored(layer_name):
            return None
        return original(layer, layer_name)

    result.get_scheme_dict = get_scheme_dict
    return result


def select_method(config, layer, prefix):
    """Use native parsed schemes, priority, formats and ignore rules."""
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

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
    if should_ignore_layer(prefix, config.ignore, config.packed_modules_mapping):
        from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod

        # Native MoE selection only checks projection names. Returning None
        # here would let it quantize an explicitly ignored fused owner.
        return UnquantizedFusedMoEMethod(layer.moe_config)
    if not all(selected) or any(s != schemes[0] for s in schemes):
        raise ValueError(
            "HCU W4A8 requires the same scheme for every expert projection"
        )
    if schemes[0].get("output_activations") is not None:
        raise ValueError("HCU W4A8 does not support output activation quantization")
    return HcuCompressedW4A8(config, layer.moe_config)


class HcuCompressedW4A8(SlimQuantW4A8Int8AiterMoEMethod):
    """Native config selection with packed Triton or AITER gemm2 execution."""

    @property
    def supports_eplb(self):
        return False

    def __init__(self, config, moe):
        if moe.moe_backend not in ("auto", "aiter", "triton"):
            raise ValueError(
                "compressed-tensors channel W4A8 requires auto, aiter or triton"
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
        self._aiter_generation = None

    @staticmethod
    def _generation(layer):
        return tuple(
            (id(t), t.data_ptr(), t._version, tuple(t.shape), tuple(t.stride()))
            for name in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            )
            for t in (getattr(layer, name),)
        )

    def prepare_aiter_w4a8_config(self, layer, config):
        """Keep P/D token-count selection from re-shuffling installed gemm2."""
        if self._aiter_generation != self._generation(layer):
            raise RuntimeError("W4A8 AITER weights/scales changed; rebuild the model")
        if config is None:
            raise RuntimeError("AITER has no solution for installed W4A8 gemm2 weights")
        # AITER's selection cache can be shared. Never mutate its config.
        config = copy(config)
        config.need_shuffle = False
        config.need_shuffle_scale = False
        return config

    def process_weights_after_loading(self, layer):
        if self.moe.moe_backend == "triton":
            return super().process_weights_after_loading(layer)
        if self._aiter_generation is not None:
            self.prepare_aiter_w4a8_config(layer, self._aiter_config)
            return
        from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import (
            install_aiter_moe_scale_layout,
            install_aiter_moe_weight_layout,
            prewarm_aiter_w4a8_moe,
        )

        config = prewarm_aiter_w4a8_moe(self, layer)
        if config is None:
            raise RuntimeError("No AITER W4A8 solution; use --moe-backend triton")
        # Disable generic shuffle on a local copy before recording its layout.
        config = copy(config)
        config.need_shuffle = False
        config.need_shuffle_scale = False
        for name in ("w13_weight", "w2_weight"):
            param = getattr(layer, name)
            param.data.copy_(repack_aiter_gemm2(param.data))
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        install_aiter_moe_weight_layout(
            layer,
            config,
            logical_shape=(
                layer.w13_weight.shape[0],
                layer.w13_weight.shape[1],
                layer.w2_weight.shape[1],
                layer.w13_weight.shape[2] * 2,
            ),
        )
        install_aiter_moe_scale_layout(layer, self.moe_quant_config, config)
        self._supports_int8_allgatherv_input = True
        self._aiter_config = config
        self._aiter_generation = self._generation(layer)

    def create_weights(self, layer, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            param = getattr(layer, name)
            original = param.weight_loader
            is_scale = name.endswith("scale")

            def loader(
                param, loaded_weight, *args, _loader=original, _scale=is_scale, **kwargs
            ):
                if self._aiter_generation is not None:
                    raise RuntimeError(
                        "W4A8 AITER checkpoint reload requires a fresh model"
                    )
                value = (
                    loaded_weight.to(param.dtype) / 16
                    if _scale
                    else convert_packed_int4(loaded_weight)
                )
                return _loader(param, value, *args, **kwargs)

            param.weight_loader = loader
