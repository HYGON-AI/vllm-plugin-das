# SPDX-License-Identifier: Apache-2.0
"""Explicit MiniMax DP8 model integration; PCP execution remains independent."""
from __future__ import annotations
from dataclasses import asdict
import json

import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder
from .minimax31_config import decode_options
from .minimax31_decode_contract import validate_parallel
from .minimax31_decode_placement import configured_placement, placement_for_model

_factory = None
_original_factory = None
_executor = None
_executor_identity = None


def validate_scope(config):
    validate_parallel(config.parallel_config)
    if not config.parallel_config.enable_expert_parallel or not config.use_v2_model_runner:
        raise ValueError("MiniMax decode requires EP8 and V2 model runner")
    if config.lora_config is not None or config.parallel_config.enable_dbo:
        raise ValueError("MiniMax decode does not support LoRA/DBO")
    arch = getattr(config.model_config.hf_config, "architectures", ()) or ()
    if not set(arch).intersection({"MiniMaxM3SparseForConditionalGeneration", "MiniMaxM3SparseForCausalLM"}):
        raise ValueError("MiniMax decode capability only supports the real MiniMax model")
    if config.model_config.quantization != "slimquant_w4a8":
        raise ValueError("MiniMax decode requires the checkpoint's mixed SlimQuant format")
    if not config.model_config.enforce_eager:
        raise ValueError('MiniMax model-level graph is not qualified; use --enforce-eager')
    # Core's VLM wrapper constructs the text model under a ModelConfig whose
    # HF config is text_config. Quantization remains on VllmConfig.quant_config.
    q = getattr(config.model_config.hf_config,"quantization_config",{})
    if q.get("hcu_minimax_static_placement") is not None:
        raise ValueError("decode static placement requires its independent ownership contract")
    if (getattr(config.parallel_config, "enable_ep_weight_filter", False)
            and configured_placement(config.model_config) is not None):
        raise ValueError("checkpoint expert filtering uses unpermuted logical owners")
    if getattr(config.quant_config,"routed_backend",None) != "deepgemm_hipc_v1":
        raise ValueError("MiniMax low-latency requires HIPC processed weights")


def _get_executor(layer, hidden, config, *, options=None):
    global _executor, _executor_identity
    from vllm.distributed.parallel_state import get_ep_group
    from hcu_vllm_das.moe.minimax31_ll import LowLatencyMoE
    group = get_ep_group()
    validate_scope(config)
    if getattr(layer, "_minimax31_expert_layout", None) != "signed-high-nibble-first-hipc-split-v1":
        raise ValueError("LL runner requires producer-prepared HIPC weight layout")
    max_tokens = int(getattr(config.model_config.hf_config, "hcu_minimax_decode", {}).get("max_dispatch_tokens",256))
    if options is None:
        options = decode_options(config)
    identity = (id(group.device_group),hidden.device,hidden.shape[-1],layer.w13_weight.shape,
                torch.cuda.current_stream(hidden.device).cuda_stream,max_tokens,options)
    if _executor is None:
        _executor = LowLatencyMoE(group.device_group, device=hidden.device,
            hidden=hidden.shape[-1], intermediate=layer.w13_weight.shape[1]//2,
            physical_experts=layer.global_num_experts,max_tokens=max_tokens,
            hipc_verify_tuned=options.hipc_verify_tuned,
            lightop_moe_tuned=options.lightop_moe_tuned)
        _executor_identity = identity
        print("MINIMAX_DECODE_LL "+json.dumps({"transport":"int8+fp32_scale",
            "mode":"low_latency","gemm":"masked_hipc","max_tokens":max_tokens,
            "physical_experts":layer.global_num_experts,"options":asdict(options)}),flush=True)
    elif identity != _executor_identity:
        raise RuntimeError("LL executor group/device/geometry/stream changed")
    return _executor


class PermutedDecodeExperts(RoutedExperts):
    """Use the stock checkpoint loader with the reference physical IDs."""

    def __init__(self, *args, **kwargs):
        from vllm.config import get_current_vllm_config
        placement = placement_for_model(get_current_vllm_config().model_config)
        if placement is None:
            raise ValueError("decode permuted experts require a frozen placement")
        super().__init__(*args, **kwargs)
        if (self.global_num_experts, self.local_num_experts) != (128, 16):
            raise ValueError("decode permutation needs 16 existing slots per rank")
        self.decode_layer_id, self.decode_permutation = placement.for_layer(self.layer_name)
        self.decode_map_sha256 = placement.map_sha256
        self.decode_loaded_shards = set()
        self.decode_gpu_permutation = None

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        from vllm.model_executor.utils import is_weights_pre_processed
        if is_weights_pre_processed():
            raise RuntimeError("decode imported storage cannot load checkpoint shards")
        if type(expert_id) is not int or not 0 <= expert_id < 128:
            raise ValueError("decode checkpoint expert ID must be logical")
        physical = self.decode_permutation[expert_id]
        name = weight_name.rsplit(".", 1)[-1]
        expected = {
            "w13_weight": ("w1", "w3"), "w13_weight_scale": ("w1", "w3"),
            "w2_weight": ("w2",), "w2_weight_scale": ("w2",),
        }
        if name not in expected or shard_id not in expected[name] or getattr(self, name) is not param:
            raise ValueError("unexpected permuted decode checkpoint shard")
        key = (physical, name, shard_id)
        if key in self.decode_loaded_shards:
            raise RuntimeError("duplicate permuted decode checkpoint shard")
        loaded = super().weight_loader(param, loaded_weight, weight_name, shard_id, physical,
                                       return_success=True)
        if loaded != (physical // 16 == self.expert_map_manager.ep_rank):
            raise RuntimeError("checkpoint shard did not load on its physical DP owner")
        if loaded:
            self.decode_loaded_shards.add(key)
        return loaded if return_success else None


def verify_permuted_model(model, placement, rank, *, checkpoint):
    layers = {m.decode_layer_id: m for m in model.modules() if isinstance(m, PermutedDecodeExperts)}
    if set(layers) != set(range(3, 60)):
        raise RuntimeError("producer did not construct all permuted experts")
    expected = {
        (slot, name, shard)
        for slot in range(16 * rank, 16 * (rank + 1))
        for name, shards in (("w13_weight", ("w1", "w3")), ("w13_weight_scale", ("w1", "w3")),
                             ("w2_weight", ("w2",)), ("w2_weight_scale", ("w2",)))
        for shard in shards
    }
    for layer in layers.values():
        if (layer.decode_map_sha256 != placement.map_sha256
                or layer.decode_permutation != placement.logical_to_physical[layer.decode_layer_id - 3]):
            raise RuntimeError(f"decode map is incorrect at layer {layer.decode_layer_id}")
        if (checkpoint and layer.decode_loaded_shards != expected
                or not checkpoint and layer.decode_loaded_shards):
            raise RuntimeError(f"decode checkpoint load state differs at layer {layer.decode_layer_id}")
    print("MINIMAX_DECODE_PLACEMENT_VERIFIED " + json.dumps({
        "rank": rank, "layers": len(layers), "checkpoint": checkpoint,
        "loaded_local_shards": len(layers) * len(expected) if checkpoint else 0,
        "map_sha256": placement.map_sha256,
    }), flush=True)


def _shared_expert_down(mlp, gate_up):
    from vllm.models.minimax_m3.amd.model import swiglu_oai_split

    activation = swiglu_oai_split(
        gate_up,
        alpha=mlp.swiglu_alpha,
        beta=mlp.swiglu_beta,
        limit=mlp.swiglu_limit,
    )
    output, _ = mlp.down_proj(activation)
    return output


class MiniMax31DecodeMoERunner(MoERunner):
    def __init__(self,*args,**kwargs):
        from vllm.config import get_current_vllm_config
        config=get_current_vllm_config()
        validate_scope(config)
        super().__init__(*args,**kwargs)
        # Core's execute_dummy_batch RPC has no current-config context. Keep
        # construction-time ownership rather than consulting ambient state.
        self._minimax_decode_config=config
        self._minimax_decode_options = decode_options(config)
        schedule = self._minimax_decode_options.shared_expert_schedule
        self._shared_auto_shapes = schedule == "auto"
        if self._shared_experts is None:
            schedule = "stock"
        elif schedule != "stock":
            from vllm.models.minimax_m3.amd.model import MiniMaxM3MLP

            stock_mlp = type(self._shared_experts._layer) is MiniMaxM3MLP
            if self._shared_auto_shapes:
                schedule = "stock"
                if stock_mlp:
                    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
                    arch = getattr(gpu, "gcnArchName", "").split(":", 1)[0]
                    if arch == "gfx938" and gpu.multi_processor_count == 64:
                        schedule = "dual_window"
            elif not stock_mlp:
                raise ValueError("explicit shared overlap requires the stock MiniMax shared MLP")
        self._shared_dispatch_overlap = schedule == "dual_window"
        self._shared_combine_overlap = schedule in ("combine", "dual_window")
        self._shared_aux_overlap = schedule == "aux"
        self._shared_aux_stream = None
        self._shared_w8a8_fused = self._minimax_decode_options.shared_w8a8_fused

    @property
    def _fused_output_is_reduced(self):
        # LL combine has already returned the weighted sum for each local
        # query. Core must not run its generic DP gather/reduce again.
        return True

    def _forward_impl(self, hidden_states, router_logits, shared_experts_input, input_ids=None):
        if self.gate is not None or input_ids is not None or self.routed_experts.quant_method.is_monolithic:
            raise RuntimeError("unsupported MiniMax decode MoE invocation")
        self.routed_experts._ensure_moe_quant_config_init()
        shared = self._shared_experts
        overlap = False
        before_dispatch_recv = None
        before_combine_recv = None
        aux_launch = None
        shared_output = None
        if shared is not None:
            assert shared_experts_input is not None
            if shared_experts_input.numel() == 0:
                # There is no local shared work, but the rank still runs DeepEP.
                shared_output = torch.empty_like(shared_experts_input)
            elif (
                (self._shared_combine_overlap or getattr(self, "_shared_aux_overlap", False))
                and (not self._shared_auto_shapes or shared_experts_input.shape[0] in (64, 80, 96, 128, 144))
            ):
                mlp = shared._layer
                gate_up = None
                fused_kernels = None
                if getattr(self, "_shared_w8a8_fused", False):
                    from .minimax31_w8a8_fused import (
                        project_shared_down,
                        project_shared_gate_up,
                        shared_w8a8_kernels,
                    )

                    fused_kernels = shared_w8a8_kernels(mlp, shared_experts_input)

                def project_gate_up():
                    nonlocal gate_up
                    if fused_kernels is None:
                        gate_up, _ = mlp.gate_up_proj(shared_experts_input)
                    else:
                        gate_up = project_shared_gate_up(mlp, shared_experts_input, fused_kernels[0])

                def project_down():
                    nonlocal shared_output
                    if gate_up is None:
                        raise RuntimeError("DeepEP did not execute the shared expert gate/up projection")
                    if fused_kernels is None:
                        shared_output = _shared_expert_down(mlp, gate_up)
                    else:
                        shared_output = project_shared_down(mlp, gate_up, fused_kernels[1])

                if getattr(self, "_shared_aux_overlap", False):
                    def run_shared():
                        project_gate_up()
                        project_down()

                    aux_launch = run_shared
                else:
                    before_combine_recv = project_down
                    if self._shared_dispatch_overlap:
                        before_dispatch_recv = project_gate_up
                    else:
                        project_gate_up()
            else:
                overlap = shared.maybe_forward_async(shared_experts_input)
                self._maybe_apply_shared_experts(shared_experts_input, SharedExpertsOrder.NO_OVERLAP)
        weights, ids = self.router.select_experts(hidden_states=hidden_states,
            router_logits=router_logits,topk_indices_dtype=torch.int64,input_ids=None)
        ids = ids.to(torch.int64).contiguous()
        weights = weights.masked_fill(ids<0,0).contiguous()
        layer = self.routed_experts
        if isinstance(layer, PermutedDecodeExperts):
            from triton_tmp_dir.minimax31_decode_remap import map_experts
            if layer.decode_gpu_permutation is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm the static expert remap before graph capture")
                layer.decode_gpu_permutation = torch.tensor(
                    layer.decode_permutation, dtype=torch.int64, device=ids.device)
            ids = map_experts(ids, layer.decode_gpu_permutation,
                              counters=getattr(layer, "decode_eplb_counters", None))
        executor = _get_executor(
            layer, hidden_states, self._minimax_decode_config, options=self._minimax_decode_options
        )
        aux_stream = None
        main_stream = None
        try:
            if aux_launch is not None:
                main_stream = torch.cuda.current_stream(hidden_states.device)
                aux_stream = self._shared_aux_stream
                if aux_stream is None:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError("warm up MiniMax shared auxiliary stream before graph capture")
                    aux_stream = self._shared_aux_stream = torch.cuda.Stream(device=hidden_states.device)
                aux_stream.wait_stream(main_stream)
                with torch.cuda.stream(aux_stream):
                    aux_launch()
            routed = executor(hidden_states.contiguous(),ids,weights,
                (layer.w13_weight,layer.w2_weight,layer.w13_weight_scale,layer.w2_weight_scale),
                before_dispatch_recv=before_dispatch_recv, before_combine_recv=before_combine_recv)
        finally:
            if aux_stream is not None:
                main_stream.wait_stream(aux_stream)
            if overlap:
                shared.wait()
        if shared_output is not None:
            if aux_stream is not None:
                shared_output.record_stream(main_stream)
            return shared_output, routed
        if before_combine_recv is not None:
            raise RuntimeError("DeepEP did not execute the shared expert down projection")
        if shared is not None:
            return shared.output, routed
        return routed


def restore_execution_state(model, device):
    """Replace explicit per-consumer MSA scratch and all of its plain aliases."""
    modules = list(model.modules())
    roots = [m for m in modules if type(m).__name__ == "MiniMaxM3Model"]
    if len(roots) != 1:
        raise ValueError("decode restore requires exactly one MiniMax text root")
    root = roots[0]
    replacements = []
    for field in ("topk_indices_buffer","sparse_bt_buffer","sparse_ctx_buffer"):
        old = getattr(root,field,None)
        if old is not None:
            if not isinstance(old,torch.Tensor):
                raise ValueError("unknown MiniMax execution scratch")
            replacements.append((old,torch.empty_like(old,device=device)))
    for module in modules:
        for field,value in list(vars(module).items()):
            for old,new in replacements:
                if value is old:
                    setattr(module,field,new)
        if "topk_completion_counter" in module._buffers:
            old = module._buffers["topk_completion_counter"]
            if old is not None:
                module._buffers["topk_completion_counter"] = torch.zeros_like(old,device=device)
                module._non_persistent_buffers_set.add("topk_completion_counter")


def install():
    global _factory,_original_factory
    from vllm.models.minimax_m3.amd import model as stock
    if _factory is not None:
        if stock.FusedMoEFactory is not _factory:
            raise RuntimeError("MiniMax decode factory ownership drift")
        return
    original = stock.FusedMoEFactory
    if getattr(original,"__module__",None) != "vllm.model_executor.layers.fused_moe.layer":
        raise RuntimeError("foreign MiniMax MoE factory override")
    def factory(*args,**kwargs):
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        validate_scope(cfg)
        if kwargs.get("runner_cls") is not None or kwargs.get("fuse_shared_experts"):
            raise ValueError("MiniMax decode requires unfused shared experts and its own runner")
        kwargs["runner_cls"] = MiniMax31DecodeMoERunner
        if configured_placement(cfg.model_config) is not None:
            if kwargs.get("routed_experts_cls") is not None:
                raise ValueError("decode placement owns the routed-expert loader")
            kwargs["routed_experts_cls"] = PermutedDecodeExperts
        return original(*args,**kwargs)
    _original_factory,_factory = original,factory
    stock.FusedMoEFactory = factory


def check_health():
    from vllm.models.minimax_m3.amd import model as stock
    if _factory is None or stock.FusedMoEFactory is not _factory:
        raise RuntimeError("MiniMax decode factory is not the active owner")
