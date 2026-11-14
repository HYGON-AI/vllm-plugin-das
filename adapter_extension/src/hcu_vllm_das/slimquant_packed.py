# SPDX-License-Identifier: Apache-2.0
"""Optional raw packed SlimQuant method on Core's public routed-expert seam."""

from __future__ import annotations

from collections.abc import Mapping
import torch
from vllm.model_executor import utils as runtime_utils
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

from . import slimquant_activation as lifecycle


def select_moe_provider(context):
    additional = getattr(context, "additional_config", None)
    # Core also accepts opaque SupportsHash platform objects. They do not
    # declare this optional namespace and must retain the expanded default.
    if not isinstance(additional, Mapping):
        return "expanded-int8"
    option = additional.get("hcu_slimquant", {})
    if not isinstance(option, dict) or set(option) - {"moe_provider"}:
        raise ValueError("SlimQuant additional_config requires a moe_provider mapping")
    provider = option.get("moe_provider", "expanded-int8")
    if provider not in ("expanded-int8", "boltops"):
        raise ValueError("SlimQuant moe_provider must be expanded-int8 or boltops")
    return provider


def prepare_packed_provider_inputs(x, topk_weights, topk_ids):
    """Translate Core's padding slots to safe, zero-contribution vendor routes.

    Core uses -1 expert ids for padded tokens. The packed vendor sorter indexes
    expert counters directly and stores one slot per expert/token pair. Give
    padding slots distinct unused ids instead of duplicating an active expert.
    Unused hidden rows may contain NaNs; zero them before zero-weight routes
    are evaluated. Tensor predicates remain dynamic during Graph replay.
    """
    valid = topk_ids >= 0
    # Core guarantees top-k <= expert count. Among the first top-k experts,
    # at least as many ids are unused as there are padding slots in this row.
    candidates = torch.arange(topk_ids.shape[-1], device=topk_ids.device)
    occupied = (topk_ids.unsqueeze(-1) == candidates).any(dim=1)
    unused_first = occupied.to(torch.int32).argsort(dim=-1)
    padding_position = ((~valid).cumsum(-1) - 1).clamp_min(0)
    padding_ids = unused_first.gather(-1, padding_position).to(topk_ids.dtype)
    return (
        torch.where(valid.any(-1, keepdim=True), x, 0),
        torch.where(valid, topk_weights, 0),
        torch.where(valid, topk_ids, padding_ids),
    )


class SlimQuantPackedExperts(FusedMoEMethodBase):
    """Keep raw bytes/scales; Core retains loading, routing and final reductions.

    This initial provider refuses unverified modes. Derived scales are buffers,
    not serialized parameters. Preprocessed-weight transfer remains unsupported.
    """

    def __init__(self, moe):
        super().__init__(moe)
        if not callable(
            getattr(FusedMoEMethodBase, "validate_weight_update", None)
        ) or not all(
            callable(getattr(runtime_utils, name, None))
            for name in ("register_derived_buffer", "set_derived_buffer")
        ):
            raise ValueError(
                "packed SlimQuant requires Core derived-buffer reload APIs and weight-update preflight"
            )
        parallel = moe.moe_parallel_config
        unsupported = any(
            getattr(parallel, name, 1) != 1
            for name in ("pcp_size", "dp_size", "ep_size", "sp_size")
        )
        unsupported |= bool(parallel.use_ep or parallel.enable_eplb)
        unsupported |= any(
            getattr(moe, name, False)
            for name in ("has_bias", "is_lora_enabled", "has_hash_routing")
        )
        unsupported |= any(
            getattr(moe, name, None) is not None
            for name in (
                "swiglu_limit",
                "swiglu_alpha",
                "swiglu_beta",
                "activation_situ_beta",
                "activation_situ_linear_beta",
            )
        )
        unsupported |= getattr(moe.activation, "value", moe.activation) != "silu"
        unsupported |= not moe.is_act_and_mul or moe.moe_backend != "auto"
        unsupported |= moe.num_local_experts != moe.num_experts
        if unsupported:
            raise ValueError(
                "Unsupported mode for packed SlimQuant; initial provider requires plain TP SiLU without bias/clamp/LoRA/EP/DP/PCP/SP/EPLB"
            )
        self._provider = None

    def prepare(self):
        self._provider = lifecycle.prepare_packed_provider()

    def validate_weight_update(self, layer, *, operation):
        lifecycle.require_usable()
        if operation != "checkpoint_reload":
            raise NotImplementedError(
                "packed SlimQuant kernel-format reload and online weight transfer "
                "are not yet supported; use complete raw checkpoint reload"
            )

    @property
    def topk_indices_dtype(self):
        return torch.int32

    def get_fused_moe_quant_config(self, layer):
        # Public non-internal-MK execution: apply consumes the selected routes.
        return None

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        lifecycle.require_usable()
        width = intermediate_size_per_partition
        if hidden_size <= 0 or width <= 0 or hidden_size % 2 or width % 2:
            raise ValueError(
                "packed SlimQuant requires positive even K and intermediate dimensions"
            )
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("packed SlimQuant requires BF16 or FP16 activations")
        if getattr(layer, "apply_router_weight_on_input", False):
            raise ValueError(
                "packed SlimQuant router-weight-on-input is not yet supported"
            )
        if (
            getattr(self.moe, "hidden_dim_unpadded", hidden_size) != hidden_size
            or getattr(self.moe, "intermediate_size_per_partition_unpadded", width)
            != width
        ):
            raise ValueError("packed SlimQuant padded geometry is not yet supported")
        original_loader = extra_weight_attrs["weight_loader"]
        layer._slimquant_ready = False
        layer._slimquant_loaded = {
            name: set()
            for name in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            )
        }
        layer._slimquant_geometry = (num_experts, hidden_size, width)
        tp = self.moe.moe_parallel_config.tp_size
        for name, shape, dtype in [
            ("w13_weight", (num_experts, 2 * width, hidden_size // 2), torch.int8),
            ("w2_weight", (num_experts, hidden_size, width // 2), torch.int8),
            ("w13_weight_scale", (num_experts, 2 * width, 1), torch.float32),
            ("w2_weight_scale", (num_experts, hidden_size, 1), torch.float32),
        ]:
            parameter = torch.nn.Parameter(
                torch.empty(shape, dtype=dtype), requires_grad=False
            )
            attrs = dict(extra_weight_attrs)
            if name.endswith("_scale"):
                attrs["quant_method"] = FusedMoeWeightScaleSupported.CHANNEL.value

            def load(
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
                return_success=False,
                *,
                _name=name,
                _dtype=dtype,
            ):
                lifecycle.require_usable()
                if loaded_weight.dtype != _dtype:
                    raise ValueError(f"packed SlimQuant {_name} dtype must be {_dtype}")
                if not 0 <= expert_id < num_experts:
                    raise ValueError(
                        "packed SlimQuant expert id outside local TP group"
                    )
                shards = ("w2",) if _name.startswith("w2_") else ("w1", "w3")
                if shard_id not in shards:
                    raise ValueError("packed SlimQuant unexpected checkpoint shard")
                if _name.endswith("_scale"):
                    expected = (hidden_size, 1) if shard_id == "w2" else (width * tp, 1)
                    if not bool(
                        torch.isfinite(loaded_weight).all()
                        and (loaded_weight > 0).all()
                        and (loaded_weight <= torch.finfo(torch.float32).max / 16).all()
                    ):
                        raise ValueError(
                            "packed SlimQuant scales must be positive finite FP32 after x16"
                        )
                else:
                    expected = (
                        (hidden_size, width * tp // 2)
                        if shard_id == "w2"
                        else (width * tp, hidden_size // 2)
                    )
                if tuple(loaded_weight.shape) != expected:
                    raise ValueError(
                        f"packed SlimQuant {_name} shape requires {expected}"
                    )
                if layer._slimquant_ready:
                    layer._slimquant_ready = False
                    for entries in layer._slimquant_loaded.values():
                        entries.clear()
                loaded = original_loader(
                    param,
                    loaded_weight,
                    weight_name,
                    shard_id,
                    expert_id,
                    return_success,
                )
                if loaded is not False:
                    layer._slimquant_loaded[_name].add((expert_id, shard_id))
                return loaded

            attrs["weight_loader"] = load
            set_weight_attrs(parameter, attrs)
            layer.register_parameter(name, parameter)
        for stem in ("w13", "w2"):
            name = f"{stem}_scale_for_kernel"
            runtime_utils.register_derived_buffer(layer, name)
            runtime_utils.set_derived_buffer(
                layer,
                name,
                torch.empty_like(getattr(layer, f"{stem}_weight_scale")),
            )

    def process_weights_after_loading(self, layer):
        lifecycle.require_usable()
        layer._slimquant_ready = False
        experts, _, _ = layer._slimquant_geometry
        for name, seen in layer._slimquant_loaded.items():
            shards = ("w2",) if name.startswith("w2_") else ("w1", "w3")
            required = {
                (expert, shard) for expert in range(experts) for shard in shards
            }
            if seen != required:
                raise ValueError(f"incomplete packed SlimQuant generation for {name}")
        for stem in ("w13", "w2"):
            scale = getattr(layer, f"{stem}_weight_scale")
            if not bool(
                torch.isfinite(scale).all()
                and (scale > 0).all()
                and (scale <= torch.finfo(torch.float32).max / 16).all()
            ):
                raise ValueError(
                    "packed SlimQuant scales must be positive finite FP32 after x16"
                )
            runtime_utils.set_derived_buffer(
                layer, f"{stem}_scale_for_kernel", scale * 16.0
            )
        layer._slimquant_ready = True

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    ):
        lifecycle.require_usable()
        if self._provider is None or not layer._slimquant_ready:
            raise RuntimeError(
                "packed SlimQuant requires prepared provider and complete raw weights"
            )
        _, hidden, _ = layer._slimquant_geometry
        if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("packed SlimQuant execution requires HCU BF16/FP16 inputs")
        if x.ndim != 2 or x.shape[-1] != hidden or not x.is_contiguous():
            raise ValueError(
                "packed SlimQuant requires contiguous token-by-hidden inputs"
            )
        if (
            topk_ids.dtype != torch.int32
            or topk_weights.dtype != torch.float32
            or topk_ids.shape != topk_weights.shape
            or topk_ids.shape[0] != x.shape[0]
        ):
            raise ValueError("packed SlimQuant route metadata is incompatible")
        # Core has already scheduled shared experts and merges/reduces its
        # output after this routed-only return. Do not execute them twice.
        x, topk_weights, topk_ids = prepare_packed_provider_inputs(
            x, topk_weights, topk_ids
        )
        return self._provider(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            inplace=False,
            activation="silu",
            use_int4_w4a8=True,
            per_channel_quant=True,
            global_num_experts=self.moe.num_experts,
            w1_scale=layer.w13_scale_for_kernel,
            w2_scale=layer.w2_scale_for_kernel,
            output_dtype=x.dtype,
        )
