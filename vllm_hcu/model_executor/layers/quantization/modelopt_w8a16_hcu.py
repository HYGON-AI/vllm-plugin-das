# SPDX-License-Identifier: Apache-2.0
# vllm-hcu: ModelOpt W8A16 (int8 weight-only, fp16/bf16 activation) linear
# and MoE methods for vLLM 0.28. Ported from the 0.25 modelopt overlay
# (which replaced the whole module); 0.28 keeps upstream's modelopt module
# and routes W8A16 here through LINEAR_METHOD_BUILDERS (see
# patch/worker/op_opt/patch_modelopt_w8a16.py).

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter

from vllm_hcu.model_executor.layers.quantization.w8a16_gemm import (
    W8A16_LAYOUT_MARLIN,  # noqa: F401
    W8A16_LAYOUT_RAW,  # noqa: F401
    _deepgemm_version,  # noqa: F401
    w8a16_should_preshuffle,  # noqa: F401
)

try:
    import aiter
    from aiter.moe import (
        get_aiter_moe_config,
        aiter_moe,
        MoeSolutionType,  # noqa: F401
        MoeQuantType,
    )
    from aiter.ops.shuffle import w8a16_marlin_weight_1, w8a16_marlin_weight_2
    _HAS_AITER = True
except ImportError:
    _HAS_AITER = False

logger = init_logger(__name__)


class ModelOptW8A16LinearMethod(LinearMethodBase):
    """Linear method for ModelOpt W8A16 (int8 weight, fp16/bf16 activation).

    Expected checkpoint structure:
      - weight: int8, shape [out, in]
      - weight_scale: float16/float32, shape [out] (per-output-channel scale)
      - NO input_scale (activations are unquantized fp16/bf16)
    """

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # W8A16 uses int8 weight (not fp8)
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Per-channel weight scale (one scale per output channel)
        weight_scale = ChannelQuantScaleParameter(
            output_dim=0,
            data=torch.empty(output_size_per_partition, dtype=params_dtype),
            weight_loader=weight_loader,
        )
        weight_scale[:] = torch.finfo(params_dtype).min
        layer.register_parameter("weight_scale", weight_scale)

    _LOGGED_PRESHUFFLE_SHAPES: set = set()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.detach()
        layer.weight_layout = W8A16_LAYOUT_RAW
        if w8a16_should_preshuffle(*weight.shape):
            import deepgemm

            shape = tuple(weight.shape)
            if shape not in self._LOGGED_PRESHUFFLE_SHAPES:
                self._LOGGED_PRESHUFFLE_SHAPES.add(shape)
                # warning, not info: vllm_hcu.* loggers run at WARNING in
                # EngineCore, an info line here would be silently dropped.
                logger.warning(
                    "W8A16: pre-shuffling linear weights into DeepGEMM "
                    "marlin layout (deepgemm %s), (N, K)=%s",
                    _deepgemm_version(), shape)
            weight = deepgemm.w8a16_marlin_weight(weight)
            layer.weight_layout = W8A16_LAYOUT_MARLIN
        else:
            weight = weight.clone()
        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.detach().clone(), requires_grad=False
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops.vllm.w8a16_gemm(
            x, layer.weight, layer.weight_scale, bias,
            getattr(layer, "weight_layout", W8A16_LAYOUT_RAW))


# ===========================================================================
# W8A16 MoE Method (aiter-based)
# ===========================================================================

class ModelOptW8A16MoEMethod(FusedMoEMethodBase):
    """MoE method for ModelOpt W8A16 (int8 weight, fp16/bf16 activation).

    Uses aiter MoE backend for DCU/HCU when available.
    """

    def __init__(
        self,
        quant_config: ModelOptFp8Config,
        moe_config: FusedMoEConfig,
    ) -> None:
        super().__init__(moe_config)
        self.quant_config = quant_config
        self.moe_quant_config: FusedMoEQuantConfig | None = None

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        # W8A16 runs its own MoE kernel (aiter), so it must never be swapped to
        # a FusedMoEModularMethod. Return None so
        # FusedMoE.maybe_init_modular_kernel (invoked under TP>1 via the EP
        # communicator) leaves this quant method in place.
        return None

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: RoutedExperts,
    ) -> mk.FusedMoEExpertsModular:
        raise ValueError(
            f"{self.__class__.__name__} uses W8A16 aiter MoE path. "
            "This function should not be called."
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.orig_dtype = params_dtype
        layer.num_experts = num_experts

        weight_loader = extra_weight_attrs.get("weight_loader")

        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        w13_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight", w13_weight)

        w2_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight", w2_weight)

        # WEIGHT SCALES - Per-tensor scaling for W8A16 MoE
        # ModelOpt W8A16 stores one scale per output channel along the sharded
        # intermediate dim, not per-tensor scalars. The shared
        # FusedMoE.weight_loader routes every "*weight_scale" name through
        # _load_per_tensor_weight_scale, which assigns the full (unsharded)
        # scale and breaks under TP>1. Give w13_weight_scale a loader that
        # narrows the scale to this rank's intermediate slice.
        w13_weight_scale = PerTensorScaleParameter(
            data=torch.full(
                (num_experts, w13_num_shards, intermediate_size_per_partition),
                1.0,
                dtype=torch.float16,
            ),
            weight_loader=self._make_w8a16_channel_scale_loader(
                layer, weight_loader),
        )
        w2_weight_scale = PerTensorScaleParameter(
            data=torch.full((num_experts, hidden_size), 1.0, dtype=torch.float16),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

    @staticmethod
    def _make_w8a16_channel_scale_loader(layer, default_loader):
        """Per-channel W8A16 MoE scale loader for the w13 (w1/w3) weights.

        The checkpoint scale for w1/w3 is per-output-channel with shape
        [num_experts, shards, intermediate_size]. The shared
        FusedMoE.weight_loader treats every "*weight_scale" name as a
        per-tensor scalar and copies it unsharded, which breaks tensor
        parallelism. This loader narrows the loaded scale to this rank's
        intermediate slice before copying. w2's scale runs along hidden_size,
        which is not sharded, so w2_weight_scale keeps the default loader.
        """

        def loader(param, loaded_weight, weight_name, shard_id, expert_id,
                   return_success=False):
            if "weight_scale" not in weight_name:
                return default_loader(param, loaded_weight, weight_name,
                                      shard_id, expert_id,
                                      return_success=return_success)
            local_expert_id = layer._map_global_expert_id_to_local_expert_id(
                expert_id)
            if local_expert_id == -1:
                return False if return_success else None
            idx = 0 if shard_id == "w1" else 1
            target = param.data[local_expert_id][idx]
            if loaded_weight.shape != target.shape:
                # v0.25 RoutedExperts does not expose tp_rank as an attribute;
                # resolve it from the distributed environment instead.
                from vllm.distributed import (
                    get_tensor_model_parallel_rank,
                )

                shard_size = target.shape[0]
                loaded_weight = loaded_weight.narrow(
                    0, shard_size * get_tensor_model_parallel_rank(),
                    shard_size)
            target.copy_(loaded_weight)
            return True if return_success else None

        return loader

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
    ):
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(
            layer, "w13_weight_scale", w13_scale.data.flatten(start_dim=1)
        )
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        # Setup the W8A16 MoE quant config. Mirrors the upstream FP8 MoE
        # method, which populates moe_quant_config itself during setup, so
        # apply() never sees an unset config regardless of the caller.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        # Convert scales to model dtype so aiter MoE kernel receives matching dtypes
        dtype = torch.get_default_dtype()
        w13_scale = w13_scale.to(dtype)
        w2_scale = w2_scale.to(dtype)

        if _HAS_AITER and envs.VLLM_ROCM_USE_AITER_MOE:
            w13 = w8a16_marlin_weight_1(layer.w13_weight)
            w2 = w8a16_marlin_weight_2(layer.w2_weight)
        else:
            w13 = layer.w13_weight
            w2 = layer.w2_weight

        # Shuffle weights to runtime format and setup kernel.
        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale)

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return int8_w8a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
        )

    @property
    def is_monolithic(self) -> bool:
        return False

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if _HAS_AITER and envs.VLLM_ROCM_USE_AITER_MOE:
            from vllm.platforms import current_platform

            # The inplace fast-path below overwrites `x` in place, which is only
            # safe because on HCU the sequential runner applies shared experts
            # BEFORE the routed matmul (upstream default_moe_runner.py:645-654),
            # so shared_experts never reads the overwritten tensor. That ordering
            # invariant relies on two constraints, asserted here:
            #   * not CUDA: on CUDA `use_shared_experts_stream` can route shared
            #     experts onto a separate stream that races with the routed apply.
            #   * dp_size == 1: the DP-chunked runner reverses the order (apply
            #     first, shared second, default_moe_runner.py:525-535) and would
            #     read `x` after the inplace overwrite. This quant is single-card.
            assert not current_platform.is_cuda(), (
                "W8A16 AITER MoE inplace fast-path is HCU-only: on CUDA the "
                "shared experts stream can race with the routed apply. Do not "
                "reuse this code path on CUDA without restoring x.clone().")
            assert layer.moe_config.dp_size == 1, (
                "W8A16 AITER MoE inplace fast-path assumes single-card "
                "(dp_size=1). With DP the chunked runner reverses shared/apply "
                "order and would read x after inplace overwrite.")
            m = x.shape[0]
            K = x.shape[1]
            n = layer.w2_weight.shape[2]
            e = layer.w13_weight.shape[0]
            topk = topk_ids.shape[1]
            dtype = x.dtype
            N1 = 2 * n
            N2 = n
            status, moe_cfg = get_aiter_moe_config(
                M=m, E=e, N1=N1, N2=N2, K=K,
                top_k=topk, block_size=0, dtype=dtype,
                quant_type=MoeQuantType.INT8_W8A16,
            )

            if not status:
                aiter.logger.info(
                    f"[aiter_moe_w8a16] SKIP {m=}, {N1=}, {N2=}, {K=}, {e=}, {topk=}: "
                    f"no backend available"
                )
                return None

            inplace = True # not getattr(layer.moe_config, "disable_inplace", True)
            # mortal_input = x if not inplace else x.clone()

            return aiter_moe(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                moe_config=moe_cfg,
                inplace=inplace,
                activation=layer.activation.value,
                w1_scale=self.moe_quant_config.w1_scale.unsqueeze(dim=-1),
                w2_scale=self.moe_quant_config.w2_scale.unsqueeze(dim=-1),
                w1_zp=self.moe_quant_config.w1_zp,
                w2_zp=self.moe_quant_config.w2_zp,
                a1_scale=self.moe_quant_config.a1_scale,
                a2_scale=self.moe_quant_config.a2_scale,
                block_shape=self.moe_quant_config.block_shape,
                global_num_experts=layer.global_num_experts,
                expert_map=None,
                routed_scaling_factor=1.0,
            )
        else:
            # 0.28 fused_experts dropped the inplace kwarg (activation is a
            # positional arg there); output is always a fresh tensor.
            from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
            return fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation.value,
                quant_config=self.moe_quant_config,
            )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("W8A16 MoE does not support monolithic path.")


