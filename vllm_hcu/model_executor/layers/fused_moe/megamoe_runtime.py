# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opt-in HCU MegaMoE; keep vLLM's expert loader, router and shared experts."""

from __future__ import annotations

import os

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)


_K1_ROUTE_TILE_M = 256
_K1_LL_ROW_TILE = 64
_UINT32_MAX = (1 << 32) - 1


def _align(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _k1_rows(backend, tokens, num_ranks, topk, local_experts) -> int:
    """Upper bound of K1/K3 launch rows; mirrors megamoe k1_fused_ext.cu."""
    if backend == "ll":
        expected = -(-tokens * topk // local_experts)
        per_expert = max(
            _align(num_ranks * tokens, _K1_LL_ROW_TILE),
            _align(expected + 128, _K1_LL_ROW_TILE),
        )
        return local_experts * per_expert
    # skew_safe_compact_capacity_tiles
    routes = num_ranks * tokens * topk
    max_tiles = local_experts * -(-num_ranks * tokens // _K1_ROUTE_TILE_M)
    active = min(local_experts, routes)
    fragmented = active + max(0, routes - active) // _K1_ROUTE_TILE_M
    return max(local_experts, min(max_tiles, fragmented)) * _K1_ROUTE_TILE_M


def max_chunk_tokens(moe, backend) -> int:
    """Largest per-rank token count whose K1/K3 asm strides fit in uint32."""
    width = max(2 * moe.intermediate_size_per_partition, moe.hidden_dim)
    shape = (moe.ep_size, moe.experts_per_token, moe.num_local_experts)
    lo, hi = 1, max(1, int(moe.max_num_tokens))
    if _k1_rows(backend, hi, *shape) * width <= _UINT32_MAX:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _k1_rows(backend, mid, *shape) * width <= _UINT32_MAX:
            lo = mid
        else:
            hi = mid - 1
    return lo


def validate_config(config, moe, layer, quant_mode=None) -> None:
    parallel = config.parallel_config
    hf = config.model_config.hf_config
    if getattr(hf, "model_type", None) not in ("deepseek_v4", "deepseek_v4_dspark"):
        raise ValueError("HCU MegaMoE supports DeepSeek-V4 target/DSpark models only")
    if not parallel.enable_expert_parallel or moe.ep_size != 8:
        raise ValueError("HCU MegaMoE requires --enable-expert-parallel and EP8")
    # EP rewrites the MoE-local TP size to one, but the model's original TP
    # ranks still receive replicated (non-sequence-parallel) tokens.  MegaMoE
    # registers routes per source rank and cannot coalesce those replicas, so
    # accepting model TP here would execute every routed expert TP times.
    if parallel.tensor_parallel_size > 1:
        raise ValueError(
            "HCU MegaMoE does not support tensor parallelism; use "
            "--tensor-parallel-size 1 with data parallel expert parallelism"
        )
    if quant_mode == "int8" and not config.model_config.enforce_eager:
        raise ValueError("HCU MegaMoE PACK5 adapter requires --enforce-eager")
    if moe.moe_backend != "deep_gemm_mega_moe":
        raise ValueError(
            "HCU MegaMoE requires --moe-backend deep_gemm_mega_moe"
        )
    if parallel.enable_dbo:
        raise ValueError("HCU MegaMoE PACK5 does not support DBO")
    if (
        moe.dp_size > 1
        and getattr(parallel, "all2all_backend", None) == "deepep_low_latency"
        and not getattr(parallel, "_vllm_hcu_deepep_auto", False)
    ):
        raise ValueError(
            "HCU MegaMoE DP execution requires coordinated token metadata; "
            "fixed deepep_low_latency suppresses it"
        )
    offload = getattr(config, "offload_config", None)
    if offload is not None and (
        offload.uva.cpu_offload_gb or offload.prefetch.offload_group_size
    ):
        raise ValueError("HCU MegaMoE does not support model weight offloading")
    if getattr(config.model_config, "enable_sleep_mode", False):
        raise ValueError("HCU MegaMoE IPC buffers do not support sleep mode")
    if moe.in_dtype != torch.bfloat16:
        raise ValueError("HCU MegaMoE requires --dtype bfloat16")
    if moe.is_sequence_parallel:
        raise ValueError("HCU MegaMoE does not support sequence parallelism")
    if moe.pcp_size > 1 and (
        moe.dp_size > 1 or not config.model_config.enforce_eager
    ):
        raise ValueError("HCU MegaMoE PCP requires DP1 and eager execution")
    shape = (
        moe.num_experts,
        moe.num_local_experts,
        moe.hidden_dim,
        moe.intermediate_size_per_partition,
        moe.experts_per_token,
    )
    if shape != (256, 32, 4096, 2048, 6):
        raise ValueError("HCU MegaMoE PACK5 supports DSV4-Flash EP8 shapes only")
    if (
        parallel.enable_eplb
        or layer.expert_map_manager.num_fused_shared_experts
        or moe.has_bias
        or moe.is_lora_enabled
    ):
        raise ValueError(
            "HCU MegaMoE does not support EPLB, fused shared experts, bias or LoRA"
        )
    activation = getattr(moe.activation, "value", moe.activation)
    if layer.scoring_func != "sqrtsoftplus" or activation != "silu":
        raise ValueError("HCU MegaMoE requires sqrtsoftplus routing and SwiGLU")
    if moe.skip_final_all_reduce:
        raise ValueError("HCU MegaMoE cannot defer the final all-reduce")
    if getattr(layer, "apply_router_weight_on_input", False):
        raise ValueError("HCU MegaMoE applies router weights after GEMM2 only")
    if (
        getattr(layer, "swiglu_alpha", None) is not None
        or getattr(layer, "swiglu_beta", None) is not None
    ):
        raise ValueError("HCU MegaMoE does not support parameterized SwiGLU")
    if str(layer.expert_placement_strategy).split(".")[-1].lower() != "linear":
        raise ValueError("HCU MegaMoE requires linear expert placement")


def select_method(layer, quant_config, moe):
    """Select before fallback kernel construction or weight shuffling."""
    config = get_current_vllm_config()
    if getattr(config.model_config.hf_config, "model_type", None) not in (
        "deepseek_v4", "deepseek_v4_dspark",
    ):
        return None
    validate_config(config, moe, layer)
    name = quant_config.get_name() if quant_config is not None else None
    if name not in (
        "slimquant_w4a8", "compressed-tensors", "slimquant_compressed_tensors_marlin",
    ):
        raise ValueError("HCU MegaMoE requires SlimQuant W4A8 or Channel-FP8 experts")
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

    # Selection bypasses get_quant_method, so honor its unquantized-layer guard
    # before looking up projection schemes or constructing quantized weights.
    if should_ignore_layer(
        layer.layer_name,
        ignore=quant_config.ignore,
        fused_mapping=quant_config.packed_modules_mapping,
        use_fnmatch=True,
    ):
        raise ValueError("HCU MegaMoE cannot run ignored/unquantized experts")
    if name == "slimquant_w4a8":
        validate_config(config, moe, layer, "int8")
        return HcuMegaMoEMethod(moe, "int8")
    quant_config._add_fused_moe_to_target_scheme_map()
    schemes = [
        quant_config.get_scheme_dict(layer, layer.layer_name + f".0.{proj}_proj")
        for proj in ("gate", "up", "down")
    ]
    if schemes[0] is None or any(s != schemes[0] for s in schemes[1:]):
        raise ValueError("HCU MegaMoE requires matching quantized expert projections")
    weight = schemes[0].get("weights")
    activation = schemes[0].get("input_activations")
    from compressed_tensors.quantization import QuantizationStrategy

    if not (
        quant_config._is_fp8_w8a8(weight, activation)
        and weight.strategy == QuantizationStrategy.CHANNEL
        and activation.strategy == QuantizationStrategy.TOKEN
        and activation.dynamic
        and weight.symmetric
        and not weight.dynamic
        and activation.symmetric
    ):
        raise ValueError("HCU MegaMoE requires symmetric Channel-FP8/dynamic-token FP8")
    kv_transfer = getattr(config, "kv_transfer_config", None)
    prefill_only = kv_transfer is not None and kv_transfer.kv_role == "kv_producer"
    graph_tokens = 0
    if not config.model_config.enforce_eager:
        graph_tokens = min(moe.max_num_tokens, moe.max_capture_size)
    return HcuMegaMoEMethod(
        moe, "fp8", weight_quant=weight,
        graph_tokens=graph_tokens, force_normal=prefill_only or moe.pcp_size > 1,
    )


def release_buffers() -> None:
    """Release IPC resources collectively, before worker process-group teardown."""
    try:
        ep = get_ep_group()
    except AssertionError:
        # No buffer can exist if model-parallel initialization never finished.
        return
    buffers = getattr(ep, "_hcu_megamoe_buffers", {})
    # Native destruction is fail-stop. Never retry a partially freed handle.
    values = list(buffers.values())
    buffers.clear()
    for buf in values:
        buf.destroy()
        getattr(buf, "_int4_unpack_workspaces", {}).clear()


class HcuMegaMoEMethod(FusedMoEMethodBase):
    # MegaMoE combines global routed experts itself; shared experts remain in
    # the standard runner, which reduces their TP partial output separately.
    hcu_owns_dispatch = True
    hcu_output_is_reduced = True

    def __init__(
        self, moe, quant_mode, *, weight_quant=None,
        graph_tokens=0, force_normal=False,
    ):
        super().__init__(moe)
        self.quant_mode = quant_mode
        self.weight_quant = weight_quant
        self.block_quant = False
        self.static_input_scales = False
        self.weight_block_size = None
        self._buffer = None
        self._int4_scales = None
        self._l1 = self._l2 = None
        self._graph_tokens = graph_tokens
        mode = os.environ.get("MEGAMOE_HCU_BACKEND", "auto").strip().lower()
        if mode not in ("auto", "ll", "normal"):
            raise ValueError("MEGAMOE_HCU_BACKEND must be auto, ll or normal")
        self._mode = "normal" if quant_mode == "int8" or force_normal else mode
        self._ll_threshold = int(os.environ.get("MEGAMOE_HCU_NORMAL_LL_TOKEN_THRESHOLD", "496"))
        if self._ll_threshold < 0:
            raise ValueError("MEGAMOE_HCU_NORMAL_LL_TOKEN_THRESHOLD must be non-negative")
        self._layout = "normal" if self._mode == "normal" else "unified"
        # Eager batches above this are split into several MegaMoE calls.
        self._max_chunk_tokens = max_chunk_tokens(
            moe, "ll" if self._mode == "ll" else "normal"
        )

    def _ll_capacity(self):
        # The wheel sizes LL scratch with its own threshold, which may differ
        # from the HCU selector. The actual allocation and asm stride bound
        # take precedence over either tuning threshold.
        return min(
            max_chunk_tokens(self.moe, "ll"),
            int(getattr(self._buffer, "ll_scratch_capacity_tokens_per_rank",
                        self.moe.max_num_tokens)),
        )

    def _backend(self, capacity, *, launch_capacity=None):
        if self._mode == "auto":
            required = capacity if launch_capacity is None else launch_capacity
            return "ll" if (
                capacity <= self._ll_threshold and required <= self._ll_capacity()
            ) else "normal"
        return self._mode

    @property
    def topk_indices_dtype(self):
        return torch.int64

    def maybe_roundup_sizes(
        self, hidden_size, intermediate_size_per_partition,
        act_dtype, moe_parallel_config,
    ):
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self, layer, num_experts, hidden_size,
        intermediate_size_per_partition, params_dtype, **attrs,
    ):
        if self.quant_mode == "int8":
            from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
                SlimQuantW4A8Int8AiterMoEMethod,
            )

            create = SlimQuantW4A8Int8AiterMoEMethod.create_weights
        else:
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import (  # noqa: E501
                CompressedTensorsW8A8Fp8MoEMethod,
            )

            create = CompressedTensorsW8A8Fp8MoEMethod.create_weights
        create(
            self, layer, num_experts, hidden_size,
            intermediate_size_per_partition, params_dtype, **attrs,
        )

    def get_fused_moe_quant_config(self, layer):
        from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

        return FusedMoEQuantConfig.make(
            torch.int8 if self.quant_mode == "int8" else torch.float8_e4m3fn,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            per_act_token_quant=True,
            weight_dtype="int4" if self.quant_mode == "int8" else None,
            gemm1_clamp_limit=layer.swiglu_limit,
        )

    @torch.inference_mode()
    def process_weights_after_loading(self, layer):
        # Loaders write through param.data without advancing Parameter._version.
        # Every post-load call is a refresh boundary, including in-place reloads.
        # No MegaMoE dependency is imported unless the feature was selected.
        os.environ.setdefault("K3_USE_ASM_TAIL_REDUCE", "0")
        try:
            import megamoe
        except ModuleNotFoundError as exc:
            if exc.name != "megamoe":
                raise
            raise RuntimeError(
                "--moe-backend deep_gemm_mega_moe on HCU requires the optional "
                "standalone megamoe package. Build and install a compatible wheel "
                "in the serving environment using the DCU/HIP instructions at "
                "https://github.com/Sherlolo/DeepGEMM-megamoe-w8a8. "
                "The standard image does not install this package automatically."
            ) from exc

        required = ["get_symm_buffer_for_mega_moe"]
        if self.quant_mode == "fp8":
            required += [
                ("flatten_pack5_weight_asm_normal"
                 if self._layout == "normal" else "flatten_pack5_weight"),
                "cast_to_fp8_channelwise", "fp8_w8a8_mega_moe",
            ]
        missing = [
            name for name in required
            if not callable(getattr(megamoe, name, None))
        ]
        if missing:
            raise RuntimeError(
                "Incompatible HCU MegaMoE wheel; missing APIs: " + ", ".join(missing)
            )
        w1, w2 = layer.w13_weight, layer.w2_weight
        if self.quant_mode == "int8":
            from .hcu_int4_megamoe import (
                validate_megamoe_int8_runtime,
                transform_int4_weights_for_mega_moe_normal,
            )

            validate_megamoe_int8_runtime()
            l1, l2 = transform_int4_weights_for_mega_moe_normal(
                w1, w2,
                l1_scale=layer.w13_weight_scale,
                l2_scale=layer.w2_weight_scale,
                scale_multiplier=16.0, prepack=False,
            )
        else:
            if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
                raise ValueError("HCU MegaMoE requires E4M3FN checkpoint weights")
            pack = (
                megamoe.flatten_pack5_weight_asm_normal
                if self._layout == "normal" else megamoe.flatten_pack5_weight
            )
            l1, l2 = tuple(
                {self._layout: (
                    pack(w.contiguous()),
                    s.reshape(w.shape[0], w.shape[1]).float().contiguous(),
                )}
                for w, s in (
                    (w1, layer.w13_weight_scale), (w2, layer.w2_weight_scale)
                )
            )
        max_tokens = int(self.moe.max_num_tokens)
        ep = get_ep_group()
        group = ep.device_group
        if hasattr(group, "size") and group.size() != self.moe.ep_size:
            raise ValueError("MegaMoE process group does not match expert placement")
        if hasattr(group, "rank") and group.rank() != self.moe.ep_rank:
            raise ValueError("MegaMoE process group rank does not match expert placement")
        cache = getattr(ep, "_hcu_megamoe_buffers", None)
        if cache is None:
            cache = ep._hcu_megamoe_buffers = {}
        key = (
            w1.device, self.quant_mode, max_tokens, self._graph_tokens,
            self.moe.num_experts, self.moe.experts_per_token,
            self.moe.hidden_dim, self.moe.intermediate_size_per_partition,
        )
        if key not in cache:
            kwargs = {"quant_mode": "int8"} if self.quant_mode == "int8" else {}
            if self._graph_tokens:
                kwargs["cuda_graph_max_tokens_per_rank"] = self._graph_tokens
            cache[key] = megamoe.get_symm_buffer_for_mega_moe(
                group, self.moe.num_experts, max_tokens,
                self.moe.experts_per_token, self.moe.hidden_dim,
                self.moe.intermediate_size_per_partition,
                use_fp8_dispatch=True, activation="swiglu",
                **kwargs,
            )
        self._buffer = cache[key]
        if self.quant_mode == "int8" and getattr(self._buffer, "quant_mode", None) != "int8":
            raise RuntimeError("MegaMoE factory did not create an INT8 symmetric buffer")
        if self._graph_tokens and getattr(self._buffer, "cuda_graph_num_tokens", None) is None:
            raise RuntimeError("MegaMoE wheel does not provide graph token buffers")
        if self.quant_mode == "fp8" and self._l1 is not None:
            # Graphs hold addresses, not Python attributes. Refresh packed
            # weights/scales in place so reload cannot leave replay on stale
            # copies while eager uses the newly assigned dictionaries.
            for old, new in ((self._l1, l1), (self._l2, l2)):
                for dst, src in zip(old[self._layout], new[self._layout]):
                    if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
                        raise ValueError("MegaMoE weight reload changed packed tensor geometry")
            with torch.no_grad():
                for old, new in ((self._l1, l1), (self._l2, l2)):
                    for dst, src in zip(old[self._layout], new[self._layout]):
                        dst.copy_(src)
            l1, l2 = self._l1, self._l2
        if self.quant_mode == "int8":
            # Native layerwise reload restores canonical Parameters AFTER this
            # callback. Retaining raw weights here would pin temporary storage.
            self._int4_scales = (l1["raw_int4"][1], l2["raw_int4"][1])
        else:
            self._l1, self._l2 = l1, l2
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self._graph_tokens:
            self._warmup_graph(layer)

    def _warmup_graph(self, layer):
        """Compile graph-specific native launches before CUDA capture starts."""
        import megamoe

        buf = self._buffer
        if self._graph_tokens > self._max_chunk_tokens:
            raise ValueError("MegaMoE CUDA graph bucket exceeds K1/K3 stride capacity")
        if self._mode == "ll" and self._graph_tokens > self._ll_capacity():
            raise ValueError("MegaMoE CUDA graph bucket exceeds LL scratch/stride capacity")
        key = (self._layout, self._mode, self._graph_tokens, layer.swiglu_limit)
        warmed = getattr(buf, "_hcu_megamoe_graph_warmups", None)
        if warmed is None:
            warmed = buf._hcu_megamoe_graph_warmups = set()
        if key in warmed:
            return
        buf.x.zero_()
        buf.x_sf.fill_(1)
        buf.topk_idx.fill_(-1)
        buf.topk_weights.zero_()
        buf.cuda_graph_num_tokens.zero_()
        y = torch.empty(
            (self._graph_tokens, self.moe.hidden_dim),
            dtype=torch.bfloat16, device=buf.x.device,
        )
        if self._mode == "auto":
            backends = (
                ("ll", "normal")
                if self._graph_tokens <= self._ll_capacity() else ("normal",)
            )
        else:
            backends = (self._mode,)
        for backend in backends:
            megamoe.fp8_w8a8_mega_moe(
                y, self._l1, self._l2, buf,
                megamoe_backend=backend, graph=True,
                capacity_num_tokens=self._graph_tokens,
                activation_clamp=layer.swiglu_limit,
                cumulative_local_expert_recv_stats=None, fast_math=True,
            )
        warmed.add(key)

    def apply(
        self, layer, x, topk_weights, topk_ids,
        shared_experts=None, shared_experts_input=None,
    ):
        import megamoe

        if self._buffer is None:
            raise RuntimeError("HCU MegaMoE weights/buffer were not initialized")
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and self.quant_mode == "int8":
            raise RuntimeError("HCU MegaMoE PACK5 adapter requires eager execution")
        if x.ndim != 2 or x.dtype != torch.bfloat16:
            raise ValueError("HCU MegaMoE requires BF16 hidden states")
        if (
            topk_ids.shape != topk_weights.shape
            or topk_ids.shape != (x.shape[0], self.moe.experts_per_token)
        ):
            raise ValueError("HCU MegaMoE requires matching [tokens, topk] routing tensors")
        n = x.shape[0]
        # DP ranks must make the same capacity choice, including empty ranks.
        context = get_forward_context()
        mode = getattr(context, "cudagraph_runtime_mode", None)
        graph = capturing or (
            self._graph_tokens > 0 and mode is not None
            and getattr(mode, "name", "NONE") != "NONE"
        )
        metadata = context.dp_metadata
        if graph:
            capacity = self._graph_tokens
            if capacity <= 0 or n > capacity:
                raise ValueError("MegaMoE CUDA graph bucket exceeds configured capacity")
        elif metadata is None:
            if getattr(self.moe, "dp_size", 1) > 1:
                raise ValueError("MegaMoE DP execution requires coordinated token metadata")
            capacity = n
        else:
            sizes = metadata.num_tokens_across_dp_cpu
            capacity = int(sizes.max())
            rank = self.moe.dp_rank
            if int(sizes[rank]) != n:
                raise ValueError("MegaMoE local tokens disagree with coordinated DP metadata")
        if capacity > self.moe.max_num_tokens or n > self.moe.max_num_tokens:
            raise ValueError(
                "HCU MegaMoE token count exceeds the symmetric buffer capacity"
            )
        y = (
            x.new_empty((capacity, x.shape[1])) if graph else torch.empty_like(x)
        )
        if capacity == 0:
            return y
        buf = self._buffer
        if graph and getattr(buf, "cuda_graph_num_tokens", None) is None:
            raise RuntimeError("MegaMoE wheel does not provide graph token buffers")
        l1, l2 = self._l1, self._l2
        if self.quant_mode == "int8":
            from .hcu_int4_megamoe import int4_w4a8_mega_moe

            # Resolve after reload finalization has restored the layer's weights.
            s1, s2 = self._int4_scales
            l1 = {"raw_int4": (layer.w13_weight, s1)}
            l2 = {"raw_int4": (layer.w2_weight, s2)}
            run = int4_w4a8_mega_moe
        else:
            run = megamoe.fp8_w8a8_mega_moe
        # K1/K3 asm strides are uint32, so large eager batches run as several
        # MegaMoE calls. Chunking follows the shared capacity, so every EP/DP
        # rank issues the same number of collective calls.
        chunk = self._max_chunk_tokens
        if self._mode == "ll":
            chunk = min(chunk, self._ll_capacity())
        if graph:
            if capacity > chunk:
                raise ValueError(
                    "MegaMoE CUDA graph bucket exceeds the K1/K3 uint32 stride limit"
                )
            chunk = capacity
        for start in range(0, capacity, chunk):
            chunk_capacity = min(chunk, capacity - start)
            lo, hi = min(start, n), min(start + chunk_capacity, n)
            m = hi - lo
            # Clear stale routes, particularly when an idle DP rank follows a
            # non-empty step. -1 is MegaMoE's unused-expert sentinel.
            buf.topk_idx.fill_(-1)
            buf.topk_weights.zero_()
            if graph:
                # Capture the padded bucket size as a GPU operation. Every replay
                # executes this update; another target/draft bucket cannot leave a
                # stale host-provided count in the shared symmetric buffer.
                buf.cuda_graph_num_tokens.fill_(n)
            if m:
                xs, ids, ws = x[lo:hi], topk_ids[lo:hi], topk_weights[lo:hi]
                if self.quant_mode == "int8":
                    megamoe.mega_moe_pre_dispatch_int8(
                        xs.contiguous(), ids.contiguous(), ws.contiguous(),
                        buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights, m,
                    )
                else:
                    q, scale = megamoe.cast_to_fp8_channelwise(xs.contiguous())
                    buf.x[:m].copy_(q)
                    buf.x_sf[:m].copy_(scale.reshape(-1))
                    buf.topk_idx[:m].copy_(ids)
                    buf.topk_weights[:m].copy_(ws)
            descriptor = getattr(context, "batch_descriptor", None)
            selector = (
                getattr(descriptor, "num_tokens", capacity) if graph
                else chunk_capacity
            )
            if graph and self.moe.dp_size > 1:
                # Local graph buckets can differ across DP ranks. Pin selection
                # to the shared maximum graph capacity so captured collectives
                # agree regardless of which local buckets are replayed together.
                selector = capacity
            run(
                y if graph else y[lo:hi], l1, l2, buf,
                megamoe_backend=self._backend(selector, launch_capacity=chunk_capacity),
                graph=graph, capacity_num_tokens=chunk_capacity,
                activation_clamp=layer.swiglu_limit,
                cumulative_local_expert_recv_stats=None, fast_math=True,
            )
        return y[:n] if graph else y
