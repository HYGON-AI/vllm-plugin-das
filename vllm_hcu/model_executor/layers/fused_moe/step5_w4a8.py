"""Opt-in Step5 W4A8 backend for the existing modular EP/DP execution path.

Selected by the normal SlimQuant method only for the explicit Step5 profile.
"""

import torch
import triton
from vllm.logger import init_logger

logger = init_logger(__name__)
from functools import lru_cache
from .step5_w4a8_policy import padded_route_rows
from .step5_w4a8_kernels import (
    _activation,
    _weighted_output,
    activation_persistent,
    small_permute,
)
from .step5_w4a8_routing import permute64


def _generation(layer):
    return tuple(
        (id(t), t.data_ptr(), t._version, tuple(t.shape))
        for t in (
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
    )


def _no_reload(*args, **kwargs):
    raise RuntimeError("W4A8 packed layout: restart from checkpoint for reload or EPLB")


@lru_cache(maxsize=2)
def _make_experts_class(phase):
    is_prefill = phase == "prefill"
    # Initialize the public package before HCU replacement modules, avoiding
    # the partial deep_gemm_utils import cycle seen in the original probe.
    import vllm.model_executor.layers.fused_moe  # noqa: F401 - initialize before HCU adapters
    from vllm_hcu.model_executor.layers.quantization import (
        slimquant_w4a8_deepgemm_runtime as dg,
    )
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_hcu.ops.step5.w4a8 import require_native_op

    op = require_native_op()

    def native_gemm(a, b, output, indices, config):
        op(
            a[0],
            b[0],
            output,
            a[1].contiguous(),
            b[1].contiguous(),
            indices.contiguous(),
        )
        return output

    class Step5W4A8Experts(dg.DeepEPDeepGemmW4A8ContiguousExperts):
        _hcu_step5_owner = True
        _hcu_step5_w4a8_native = True

        @staticmethod
        def _supports_activation(activation):
            return activation == MoEActivation.SWIGLUSTEP

        def process_weights_after_loading(self, layer):
            # Reuse the common validator and packer, but retain the Parameters
            # as owners here: the base method deliberately packs clones for
            # reload support. Changing that policy for other models would be
            # unsafe, and a second packed owner exceeds this profile's budget.
            dg._validate_w4a8_channel_weights(layer)
            old = getattr(layer, "_hcu_step5_w4a8_owner", None)
            if old is not None:
                if old != _generation(layer):
                    raise RuntimeError(
                        "Packed W4 owner changed; reload/EPLB unsupported"
                    )
            else:
                for parameter in (layer.w13_weight, layer.w2_weight):
                    before = (
                        id(parameter),
                        parameter.data_ptr(),
                        parameter.untyped_storage().nbytes(),
                        tuple(parameter.shape),
                    )
                    if not parameter.is_contiguous() or parameter.storage_offset():
                        raise ValueError(
                            "Single-owner W4 packing requires contiguous complete storage"
                        )
                    with torch.no_grad():
                        packed = dg.pack_w4a8_moe_hipc_weight(parameter)
                    after = (
                        id(parameter),
                        packed.data_ptr(),
                        packed.untyped_storage().nbytes(),
                        tuple(packed.shape),
                    )
                    if before != after:
                        raise RuntimeError(
                            "Native repack allocated a second weight owner"
                        )
                layer._hcu_step5_w4a8_owner = _generation(layer)
                for parameter in (
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                ):
                    parameter.requires_grad_(False)
                    if callable(getattr(parameter, "weight_loader", None)):
                        parameter.weight_loader = _no_reload
            self._deepgemm_w13, self._deepgemm_w2 = layer.w13_weight, layer.w2_weight

        def workspace_shapes(
            self,
            M,
            N,
            K,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        ):
            # This private adapter stores only INT8 gathered/activation data in
            # workspace13; gateup/down BF16 data lives in workspace2. Keep the
            # framework's BF16 dtype but reserve the exact byte capacity.
            base = super().workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )
            if not is_prefill or (N, K) != (3072, 4096):
                return base
            if expert_tokens_meta is None:
                aligned = padded_route_rows(M, topk, local_num_experts)
            else:
                from vllm_hcu.model_executor.layers.quantization import (
                    slimquant_w4a8_deepgemm_runtime as dg,
                )

                aligned = dg.compute_aligned_M(
                    M=M,
                    num_topk=topk,
                    local_num_experts=local_num_experts,
                    alignment=self.ALIGNMENT,
                    expert_tokens_meta=expert_tokens_meta,
                )
            # The final output can alias workspace13 after GEMM2, so also
            # retain M*K BF16 elements when the allocator reuses this buffer.
            first_elements = max(triton.cdiv(aligned * max(K, N // 2), 2), M * K)
            return ((triton.cdiv(first_elements, K), K), (aligned, max(N, K)), (M, K))

        def apply(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        ):
            if apply_router_weight_on_input or not self._supports_activation(
                activation
            ):
                raise ValueError(
                    "Step5 W4 requires output-side weighting and SwigluStep7"
                )
            if hidden_states.dtype != torch.int8 or a1q_scale is None:
                raise ValueError(
                    "Modular prepare must dispatch paired INT8 activations and FP32 scales"
                )
            if hidden_states.shape[0] == 0:
                return
            E, _, N, K, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
            use_small = (
                0 < hidden_states.shape[0] <= 256
                and expert_map is not None
                and topk_ids.is_contiguous()
                and expert_map.is_contiguous()
                and a1q_scale.is_contiguous()
                and expert_tokens_meta is None
            )
            use_large64 = (
                is_prefill
                and not use_small
                and expert_tokens_meta is None
                and expert_map is not None
                and a1q_scale.is_contiguous()
            )
            aligned = dg.compute_aligned_M(
                M=topk_ids.shape[0],
                num_topk=topk,
                local_num_experts=E,
                alignment=self.ALIGNMENT,
                expert_tokens_meta=expert_tokens_meta,
            )
            if use_small:
                aligned = padded_route_rows(hidden_states.shape[0], topk, E)
            elif use_large64:
                aligned = padded_route_rows(hidden_states.shape[0], topk, E)
            gathered = _resize_cache(workspace13.view(dtype=torch.int8), (aligned, K))
            if use_small:
                aq, scales, indices, inverse, _ = small_permute(
                    hidden_states,
                    self._ensure_2d_scale(a1q_scale),
                    topk_ids,
                    expert_map,
                    E,
                    gathered,
                    global_num_experts,
                )
            elif use_large64:
                aq, scales, indices, inverse, _ = permute64(
                    hidden_states,
                    self._ensure_2d_scale(a1q_scale),
                    topk_ids,
                    expert_map,
                    E,
                    gathered,
                )
            else:
                aq, scales, indices, inverse, _ = dg.deepgemm_moe_permute(
                    aq=hidden_states,
                    aq_scale=self._ensure_2d_scale(a1q_scale),
                    topk_ids=topk_ids,
                    local_num_experts=E,
                    expert_map=expert_map,
                    expert_tokens_meta=expert_tokens_meta,
                    aq_out=gathered,
                )
            indices = indices.to(torch.int32).contiguous()
            gateup = _resize_cache(workspace2, (aligned, N))
            native_gemm(
                (aq, scales),
                (self._deepgemm_w13, self.w1_scale),
                gateup,
                indices,
                config={"MODE": 1002},
            )
            intermediate = _resize_cache(
                workspace13.view(dtype=torch.int8), (aligned, N // 2)
            )
            a_scale = torch.empty(
                (aligned, 1), device=hidden_states.device, dtype=torch.float32
            )
            if hidden_states.shape[0] >= 1024:
                activation_persistent[(min(triton.cdiv(aligned, 8), 1024),)](
                    gateup,
                    intermediate,
                    a_scale,
                    indices,
                    aligned,
                    N // 2,
                    8,
                    512,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                _activation[(aligned,)](
                    gateup,
                    intermediate,
                    a_scale,
                    indices,
                    N // 2,
                    triton.next_power_of_2(N // 2),
                    num_warps=4,
                )
            down = _resize_cache(workspace2, (aligned, K))
            native_gemm(
                (intermediate, a_scale),
                (self._deepgemm_w2, self.w2_scale),
                down,
                indices,
                config={"MODE": 1002},
            )
            # Same provider/state protocol as FP8 fused experts. It declines
            # while DP-pipe owns combine or shared output is not ready. The
            # existing finalize hook then avoids a second shared addition.
            from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe

            provider = getattr(ep_fused_moe, "shared_add_provider", None)
            addition = provider(output) if provider is not None else None
            shared, start = addition if addition is not None else (output, 0)
            _weighted_output[(output.shape[0], triton.cdiv(K, 2048))](
                down,
                inverse,
                topk_weights,
                expert_map if expert_map is not None else topk_ids,
                topk_ids,
                output,
                shared,
                output.shape[0],
                K,
                topk,
                global_num_experts,
                expert_map is not None,
                addition is not None,
                start,
                shared.shape[0] if addition is not None else 0,
                2048,
                num_warps=4,
                enable_fp_fusion=False,
            )

    return Step5W4A8Experts


def prepare_step5_w4a8(method, layer, phase):
    """Select modular W4 before model load; no legacy monolithic hook allowed."""
    import vllm.model_executor.layers.fused_moe  # noqa: F401 - initialize before HCU adapters
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk

    Experts = _make_experts_class(phase)

    activation = getattr(method.moe.activation, "value", method.moe.activation)
    if activation != "swiglustep":
        raise ValueError("Native Step5 experts require SwigluStep activation")
    parallel = method.moe.moe_parallel_config
    if parallel.enable_eplb or parallel.all2all_backend != "allgather_reducescatter":
        raise ValueError(
            "This W4 profile supports static EP8 AG/RS, not EPLB or DeepEP"
        )
    if tuple(layer.w13_weight.shape) != (44, 3072, 2048) or tuple(
        layer.w2_weight.shape
    ) != (44, 4096, 768):
        raise ValueError("Expected Step5 EP8 packed W4 dimensions")
    if getattr(layer, "swiglu_limit", 7.0) not in (None, 7.0, 7):
        raise ValueError("Step5 W4 activation clamp must remain7")
    if getattr(layer, "_hcu_step5_w4a8_owner", None) is not None:
        if layer._hcu_step5_w4a8_owner != _generation(layer):
            raise RuntimeError("Packed W4 generation changed")
        if method.moe_kernel is not None:
            return
    quant = method.get_fused_moe_quant_config(layer)
    prepare = maybe_make_prepare_finalize(
        moe=method.moe,
        quant_config=quant,
        routing_tables=None,
        allow_new_interface=True,
        use_monolithic=False,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPModular,
    )

    if type(prepare) is not MoEPrepareAndFinalizeNaiveDPEPModular:
        raise RuntimeError(
            "Expected modular naive DP/EP; refuse silent loss of DP pipeline"
        )
    experts = Experts(method.moe, quant)
    experts.process_weights_after_loading(layer)
    method.moe_quant_config = quant
    method.moe_kernel = mk.FusedMoEKernel(prepare, experts)
    logger.info_once(
        "[step5-w4a8-native] built-in hcu_ops, profile=%s, static EP8", phase
    )
    # The modular prepare owns quantize/dispatch and the modular kernel
    # owns pipe/DBO/combine. Outer naive dispatch must therefore be off.
    if not method.supports_internal_mk:
        raise RuntimeError("Framework did not recognize internal modular ownership")
    layer._hcu_step5_w4a8_backend = {
        "experts": type(experts).__name__,
        "prepare": type(prepare).__name__,
        "activation_dtype": "int8",
        "weight_bits": 4,
        "gemm_mode": 1002,
        "reduction_block": 2048,
        "small_route_alignment": 64,
        "prefill_activation": "persistent8x512",
        "profile": phase,
        "shared_add_protocol": "EP_COMBINE provider/finalize",
        "weight_scale": "canonical checkpoint s8, no x16 copy",
    }


def validate_loaded_model(model):
    layers = [
        module
        for module in model.modules()
        if getattr(module, "_hcu_step5_w4a8_owner", None) is not None
    ]
    storage = {}
    for layer in layers:
        for weight in (layer.w13_weight, layer.w2_weight):
            storage[(weight.device.index, weight.untyped_storage().data_ptr())] = (
                weight.untyped_storage().nbytes()
            )
    if (
        len(layers) != 88
        or len(storage) != 176
        or sum(storage.values()) != 36_540_776_448
    ):
        raise RuntimeError(
            "Step5 native W4A8 must own88EP8 expert layers with one packed owner each"
        )
    logger.info(
        "[step5-w4a8-native] verified88layers, packed bytes=%d", sum(storage.values())
    )
