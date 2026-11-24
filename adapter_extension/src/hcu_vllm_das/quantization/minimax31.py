# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3.1 channel W4A8/W8A8 checkpoint contract.

Uses the existing DAS numerical implementation and official vLLM model loaders.
Importing this module does not activate the legacy platform/general plugins.
"""
from __future__ import annotations

from copy import deepcopy
import re
import torch
from vllm.model_executor.utils import is_weights_pre_processed, replace_parameter

from .slimquant_w4a8 import (
    SlimQuantW4A8Int8Config, SlimQuantW4A8Int8LinearMethod,
    SlimQuantW4A8Int8AiterMoEMethod,
)

_EXPERT = re.compile(r"language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w[123]")
_LINEAR = re.compile(
    r"language_model\.model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj|index_q_proj|index_k_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj)"
    r"|block_sparse_moe\.shared_experts\.(?:gate_proj|up_proj|down_proj))"
)
_IGNORES = frozenset({
    r"re:.*lm_head(?:\.|$)", r"re:.*embed_tokens(?:\.|$)",
    r"re:.*(?:block_sparse_moe|mlp)\.gate(?:\.|$)",
    r"re:.*vision_tower.*", r"re:.*multi_modal_projector.*",
    r"re:.*patch_merge_mlp.*",
})


def _modules(entry, *, expected, pattern, label):
    if not isinstance(entry, dict):
        raise ValueError(f"missing {label} quantization contract")
    for name, value in expected.items():
        if entry.get(name) != value:
            raise ValueError(f"unsupported {label}.{name}: {entry.get(name)!r}")
    modules = entry.get("modules")
    if (not isinstance(modules, list) or not modules
            or any(not isinstance(x, str) or pattern.fullmatch(x) is None for x in modules)
            or len(modules) != len(set(modules))):
        raise ValueError(f"invalid or duplicate {label} module names")
    return list(modules)


class MiniMax31SlimQuantConfig(SlimQuantW4A8Int8Config):
    """Preserve target checkpoint precision and retain explicit cache identity."""

    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if isinstance(method, SlimQuantW4A8Int8LinearMethod):
            return MiniMax31W8A8LinearMethod(self)
        if isinstance(method, SlimQuantW4A8Int8AiterMoEMethod):
            return MiniMax31W4A8MoEMethod(self, layer.moe_config)
        return method

    @classmethod
    def from_config(cls, config):
        if (config.get("quant_method") != "slimquant_w4a8"
                or config.get("activation_scheme") != "dynamic"):
            raise ValueError("MiniMax M3.1 requires dynamic slimquant_w4a8")
        ignores = config.get("ignore")
        if (not isinstance(ignores, list) or any(not isinstance(x, str) for x in ignores)
                or len(ignores) != len(_IGNORES) or set(ignores) != _IGNORES):
            raise ValueError("unsupported MiniMax M3.1 precision ignore set")
        mixed = config.get("mixed_precision")
        if not isinstance(mixed, dict) or set(mixed) != {"experts", "w8a8_include"}:
            raise ValueError("expected MiniMax W4 experts and W8 linear precision map")
        experts = _modules(mixed["experts"], label="experts", pattern=_EXPERT,
                           expected={"weight_bits": 4, "format": "pack-int4-nibble", "scope": "per_channel"})
        linears = _modules(mixed["w8a8_include"], label="w8a8_include", pattern=_LINEAR,
                           expected={"weight_bits": 8, "activation_bits": 8,
                                     "format": "int8-per-channel", "scope": "per_channel",
                                     "activation_scheme": "dynamic_per_token"})
        result = cls(ignore=list(ignores), w8a8_include=linears)
        backend = config.get("hcu_minimax_routed_backend", "deepgemm_hipc_v1")
        if backend not in ("bolt", "deepgemm_hipc_v1"):
            raise ValueError("unknown MiniMax routed backend in weight configuration")
        result.routed_backend = backend
        w8_backend = config.get("hcu_minimax_w8a8_backend", "hipblaslt_v1")
        w8_fallback = config.get("hcu_minimax_w8a8_fallback", "error")
        if w8_backend not in ("triton", "hipblaslt_v1") or w8_fallback not in ("error", "triton"):
            raise ValueError("unknown MiniMax W8A8 backend/fallback")
        result.w8a8_backend, result.w8a8_fallback = w8_backend, w8_fallback
        result.expert_modules = tuple(experts)
        result.hcu_weight_cache_identity = deepcopy(config.get("hcu_weight_cache_identity"))
        return result


class MiniMax31W8A8LinearMethod(SlimQuantW4A8Int8LinearMethod):
    """Restore the official dynamic-channel runtime layout without transposing twice."""
    supports_pre_processed_weights = True

    def process_weights_after_loading(self, layer):
        kernel = layer.scheme.kernel
        supported = {
            "TritonInt8ScaledMMLinearKernel", "AiterInt8ScaledMMLinearKernel",
            "CutlassInt8ScaledMMLinearKernel", "MiniMax31HipBlasLtInt8Kernel",
        }
        if type(kernel).__name__ not in supported:
            raise RuntimeError(f"unaudited MiniMax INT8 cache kernel: {type(kernel).__name__}")
        if not is_weights_pre_processed():
            super().process_weights_after_loading(layer)
        # create_weights has already rebuilt the scheme/kernel Python objects.
        # Only the exported Parameter storage is restored by the IPC loader.
        n = sum(layer.logical_widths)
        k = layer.input_size_per_partition
        if (layer.weight.dtype != torch.int8 or tuple(layer.weight.shape) != (k, n)
                or tuple(layer.weight.stride()) != (1, k)
                or layer.weight_scale.dtype != torch.float32
                or tuple(layer.weight_scale.shape) != (n, 1)
                or not layer.weight_scale.is_contiguous()):
            raise RuntimeError("invalid MiniMax preprocessed INT8 weight/scale layout")
        if any(getattr(layer, name, None) is not None for name in ("input_scale", "input_zero_point", "azp_adj")):
            raise RuntimeError("MiniMax INT8 requires symmetric dynamic token quantization")
        backend = self.quantization_config.w8a8_backend
        if backend == "hipblaslt_v1":
            from .minimax31_w8a8 import MiniMax31HipBlasLtInt8Kernel
            layer.scheme.kernel = MiniMax31HipBlasLtInt8Kernel(
                layer.scheme.kernel.config, layer.scheme.kernel.layer_param_names,
                fallback=self.quantization_config.w8a8_fallback)
        elif backend == "triton":
            # Explicit retained comparison backend, using the same processed
            # weight storage and input quantization ABI.
            from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonInt8ScaledMMLinearKernel
            layer.scheme.kernel = TritonInt8ScaledMMLinearKernel(
                layer.scheme.kernel.config, layer.scheme.kernel.layer_param_names)
        else:
            raise RuntimeError("unknown MiniMax W8A8 backend")


class MiniMax31W4A8MoEMethod(SlimQuantW4A8Int8AiterMoEMethod):
    """Packed expert layout shared by preload and the BoltOPs Triton path."""
    supports_pre_processed_weights = True
    supports_int8_allgatherv_input = False
    @property
    def requires_device_loading(self):
        # The native packer requires device weights on fresh load. IPC restore
        # only rebuilds metadata and must not move or rewrite shared storage.
        return (self.quant_config.routed_backend == "deepgemm_hipc_v1"
                and not is_weights_pre_processed())

    def process_weights_after_loading(self, layer):
        activation = getattr(self.moe.activation, "value", self.moe.activation)
        if activation != "swigluoai_uninterleave":
            raise RuntimeError(f"unexpected MiniMax expert activation: {activation}")
        e, twice_i, half_h = layer.w13_weight.shape
        h = self.moe.hidden_dim
        i = self.moe.intermediate_size_per_partition
        if (twice_i != 2*i or half_h*2 != h
                or tuple(layer.w2_weight.shape) != (e, h, i//2)):
            raise RuntimeError("invalid MiniMax raw packed expert dimensions")
        for name, shape, dtype in (
            ("w13_weight", (e,2*i,h//2), torch.int8),
            ("w2_weight", (e,h,i//2), torch.int8),
            ("w13_weight_scale", (e,2*i,1), torch.float32),
            ("w2_weight_scale", (e,h,1), torch.float32),
        ):
            tensor = getattr(layer, name)
            if tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():
                raise RuntimeError(f"invalid MiniMax packed expert parameter: {name}")
            tensor.requires_grad_(False)
        backend = self.quant_config.routed_backend
        if backend == "deepgemm_hipc_v1":
            from triton_tmp_dir.minimax31_hipc_moe import prepare_hipc_weights, LAYOUT
            if not is_weights_pre_processed():
                # Checkpoint gate/up are already split. Pack once before export;
                # processed IPC tensors must never enter this transform again.
                w13, w2 = prepare_hipc_weights(layer.w13_weight, layer.w2_weight)
                replace_parameter(layer, "w13_weight", torch.nn.Parameter(w13, requires_grad=False))
                replace_parameter(layer, "w2_weight", torch.nn.Parameter(w2, requires_grad=False))
            layer._minimax31_expert_layout = LAYOUT
        else:
            if not is_weights_pre_processed():
                from triton_tmp_dir.minimax31_w4a8_moe import prepare_minimax31_w13
                weight, scale = prepare_minimax31_w13(layer.w13_weight, layer.w13_weight_scale)
                replace_parameter(layer, "w13_weight", torch.nn.Parameter(weight, requires_grad=False))
                replace_parameter(layer, "w13_weight_scale", torch.nn.Parameter(scale, requires_grad=False))
            layer._minimax31_expert_layout = "signed-high-nibble-first-interleaved-v1"
        # A versioned producer/config must bind backend choice and source hash.
        # IPC restores processed storage; only Python metadata is rebuilt here.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        from triton_tmp_dir.minimax31_w4a8_moe import apply_minimax31_w4a8
        if getattr(layer, "_minimax31_expert_layout", None) == "signed-high-nibble-first-hipc-split-v1":
            raise RuntimeError("HIPC weights require the fixed-count DeepEP local runner")
        if getattr(layer, "_minimax31_expert_layout", None) != "signed-high-nibble-first-interleaved-v1":
            raise RuntimeError("MiniMax expert post-load processing has not completed")
        # PCP padding is marked -1 by the upstream router; Bolt's sorting
        # indexes its workspace directly and requires every id to be in range.
        # Routing weight zero makes a substituted legal id contribute nothing.
        parallel = self.moe.moe_parallel_config
        if parallel.pcp_size > 1:
            if (parallel.pcp_size != 8 or parallel.ep_size != 8 or not parallel.use_ep
                    or parallel.all2all_backend != "allgather_reducescatter"):
                raise RuntimeError("unsupported MiniMax PCP expert routing")
            if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != x.shape[0]:
                raise RuntimeError("invalid gathered MiniMax PCP routes")
            torch._assert_async(((topk_ids >= -1) & (topk_ids < 128)).all(),
                                "MiniMax PCP expert id is outside [-1, 128)")
            padding = topk_ids == -1
            topk_ids = topk_ids.masked_fill(padding, 0)
            topk_weights = topk_weights.masked_fill(padding, 0.0)
        # The vLLM runner evaluates shared experts and the outer MiniMax block
        # applies routed_scaling_factor. Neither belongs inside this kernel.
        return apply_minimax31_w4a8(
            x, layer.w13_weight, layer.w2_weight,
            layer.w13_weight_scale, layer.w2_weight_scale, topk_weights, topk_ids,
            alpha=self.moe.swiglu_alpha, beta=self.moe.swiglu_beta,
            limit=self.moe.swiglu_limit, expert_map=getattr(layer, "expert_map", None),
            w13_layout="interleaved")


def restore_minimax31_router_aliases(model):
    """Bind unregistered router references to the restored full expert bias.

    IPC replaces Parameter objects. The official router holds a plain Python
    reference created while the model was on meta; it is not an exported
    parameter owner. Keep the full vector on every TP rank, without slicing.
    """
    repaired = 0
    for module in model.modules():
        if type(module).__name__ != "MiniMaxM3MoE":
            continue
        bias = module.e_score_correction_bias
        if bias is None:
            continue
        router = module.experts.router
        if (bias.is_meta or bias.dtype != torch.float32 or bias.ndim != 1
                or bias.numel() != module.gate.weight.shape[0]):
            raise RuntimeError("MiniMax routing bias must be a loaded full FP32 expert vector")
        if not hasattr(router, "e_score_correction_bias"):
            raise RuntimeError("MiniMax router bias interface changed")
        if router.e_score_correction_bias is not bias:
            router.e_score_correction_bias = bias
            repaired += 1
    return repaired


def restore_minimax31_scratch(model, device):
    """Rebuild this engine's plain meta top-k scratch and local counters."""
    roots = [m for m in model.modules()
             if type(m).__name__ == "MiniMaxM3Model"
             and type(m).__module__ == "vllm.models.minimax_m3.amd.model"]
    if len(roots) != 1:
        raise RuntimeError("expected exactly one AMD MiniMax M3 model scratch owner")
    root = roots[0]
    old = root.topk_indices_buffer
    if old is None or not old.is_meta:
        return 0
    if old.dtype != torch.int32 or old.ndim != 3 or not old.is_contiguous():
        raise RuntimeError("unexpected MiniMax top-k scratch contract")
    if root.sparse_bt_buffer is not None or root.sparse_ctx_buffer is not None:
        raise RuntimeError("MSA shared scratch restoration is not validated")
    bindings = []
    for layer in root.layers:
        attention = getattr(layer, "self_attn", None)
        attention_type = type(attention)
        if (attention_type.__name__, attention_type.__module__) == (
                "MiniMax31PCPSparseAttention", "hcu_vllm_das.parallel.minimax31_pcp"):
            from hcu_vllm_das.parallel.minimax31_pcp import (
                MiniMax31PCPSparseAttention, _original_sparse)
            if attention_type is not MiniMax31PCPSparseAttention or not isinstance(attention, _original_sparse):
                raise RuntimeError("unexpected MiniMax PCP sparse attention type")
        elif attention_type.__name__ == "MiniMaxM3SparseAttention":
            if attention_type.__module__ != "vllm.models.minimax_m3.amd.model":
                raise RuntimeError("unexpected MiniMax sparse attention type")
        else:
            continue
        impl = attention.indexer.impl
        if (attention.topk_indices_buffer is not old or impl.topk_indices_buffer is not old
                or attention.sparse_bt_buffer is not None or attention.sparse_ctx_buffer is not None):
            raise RuntimeError("MiniMax scratch alias or MSA contract changed")
        if "topk_completion_counter" not in impl._buffers:
            raise RuntimeError("MiniMax indexer completion counter is not registered")
        counter = impl._buffers["topk_completion_counter"]
        # Native IPC maps registered non-persistent buffers from the producer.
        # Their token capacity can differ from this engine's meta scratch.
        if counter is not None and (counter.dtype != torch.int32
                                    or counter.ndim != 2 or not counter.is_contiguous()
                                    or counter.shape[0] != old.shape[0]
                                    or counter.shape[1] <= 0):
            raise RuntimeError("unexpected MiniMax top-k completion counter")
        bindings.append((attention, impl, counter))
    if not bindings:
        raise RuntimeError("MiniMax sparse scratch has no consumers")
    # All validation and allocations precede publication; weights and fixed
    # buffers remain untouched. Each indexer gets a private zero counter.
    replacement = torch.empty_like(old, device=device)
    counters = [None if counter is None else
                torch.zeros(old.shape[:2], dtype=torch.int32, device=device)
                for _, _, counter in bindings]
    root.topk_indices_buffer = replacement
    for (attention, impl, _), counter in zip(bindings, counters):
        attention.topk_indices_buffer = replacement
        impl.topk_indices_buffer = replacement
        impl._buffers["topk_completion_counter"] = counter
        # IPC's register_buffer defaults to persistent=True; restore the
        # upstream execution-state semantics after replacing its buffer.
        impl._non_persistent_buffers_set.add("topk_completion_counter")
    return len(bindings)


def after_minimax31_model_load(context):
    if context.vllm_config.model_config.quantization != "slimquant_w4a8":
        return
    model = context.worker.model_runner.model
    restore_minimax31_router_aliases(model)
    if getattr(context.vllm_config.model_config.hf_config,"hcu_minimax_decode",None) is not None:
        from ..parallel.minimax31_decode import validate_scope,restore_execution_state
        validate_scope(context.vllm_config)
        restore_execution_state(model,context.worker.device)
        return
    restore_minimax31_scratch(model, context.worker.device)
