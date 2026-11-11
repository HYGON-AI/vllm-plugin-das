# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import final

import torch

import vllm.envs as envs
import vllm_hcu.platforms.envs as henvs
from vllm_hcu.model_executor.layers.fused_moe.workspace_growth import get_moe_workspaces
from vllm_hcu.model_executor.layers.fused_moe.combine_runtime import (
    suspend_ep_shared_add,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_maybe_run_recv_hook,
    dbo_register_recv_hook,
    dbo_yield,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

# Prefill (>= this many tokens summed over DP ranks; 0 disables): naive DP/EP
# dispatch -> experts -> combine pipelined over two halves of the source DP
# ranks.  The all-gather of half 1 runs under the experts of half 0 and the
# reduce of half 0 under the experts of half 1, all comm on a side stream with
# grouped NCCL broadcast/reduce. Newer vLLM uses native uniform collectives
# for equal sizes, so BF16 reduction association and performance may differ;
# validate numerical tolerance and the complete MoE (including shared overlap).
# The caller
# joins the side stream with join_dp_moe_pipe() (after launching a late shared
# expert, which then overlaps the last reduce).  Not VLLM_HCU_*: eager-only
# inside the MoE custom op, must not change the compile hash.
_DP_MOE_PIPE_MIN_TOKENS = (
    int(os.environ.get("HCU_DP_MOE_PIPE_MIN_TOKENS", "0"))
    if henvs.custom_ops_enabled() else 0
)
_EP_DIRECT_SUM = (
    int(os.environ.get("HCU_EP_DIRECT_SUM", "0"))
    if henvs.custom_ops_enabled() else 0
)
_DP_PIPE_UNIFORM = henvs.optional_custom_op_enabled(
    os.environ.get("HCU_DP_MOE_PIPE_UNIFORM", "0") == "1"
)
_dp_pipe_stream: torch.cuda.Stream | None = None
# (side stream, tensors it still reads/writes) until the main stream joins it.
_dp_pipe_pending: tuple | None = None
_dp_pipe_announced = False
_step5_ep_workspace_announced = False


def _step5_ep_chunk_provider(experts, hidden_states, w1, w2):
    """Use the implementation selected by the official runtime dispatcher."""
    if not henvs.optional_custom_op_enabled(
        os.environ.get("HCU_STEP5_BOUNDED_EP_WORKSPACE", "0") == "1"
    ):
        return None
    provider = getattr(experts, "_hcu_step5_ep_chunk_size", None)
    if provider is not None:
        return provider if getattr(experts, "_hcu_step5_owner", False) else None
    from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
        TritonOrDeepGemmExperts,
    )

    if type(experts) is TritonOrDeepGemmExperts:
        selected = experts._select_experts_impl(hidden_states, w1, w2)
        return (getattr(selected, "_hcu_step5_ep_chunk_size", None)
                if getattr(selected, "_hcu_step5_owner", False) else None)
    return None


def _step5_ep_private_output(
    workspace13: torch.Tensor, fused_out: torch.Tensor
) -> torch.Tensor:
    """Keep the native EP result alive while GEMM2 reuses scratch storage.

    WorkspaceManager can return disjoint views of one allocation, so compare
    actual contiguous byte intervals rather than their storage identities.
    """
    if (
        workspace13.device != fused_out.device
        or workspace13.numel() == 0
        or fused_out.numel() == 0
        or not workspace13.is_contiguous()
        or not fused_out.is_contiguous()
    ):
        return fused_out
    start = workspace13.data_ptr()
    other = fused_out.data_ptr()
    if (
        start < other + fused_out.numel() * fused_out.element_size()
        and other < start + workspace13.numel() * workspace13.element_size()
    ):
        return torch.empty_like(fused_out)
    return fused_out


def join_dp_moe_pipe() -> None:
    global _dp_pipe_pending
    if _dp_pipe_pending is not None:
        torch.cuda.current_stream().wait_stream(_dp_pipe_pending[0])
        _dp_pipe_pending = None


def _nccl_dtype(t: torch.Tensor):
    from vllm.distributed.device_communicators.pynccl_wrapper import ncclDataTypeEnum

    if t.element_size() == 1:
        t = t.view(torch.uint8)
    return ncclDataTypeEnum.from_torch(t.dtype)

#
# This file defines a set of base classes used to make MoE kernels more modular.
# The goal is to be able to utilize different communication mechanisms with
# any fused MoE kernel without needing to have combinatoric implementations.
#
# The fused moe kernels are broken down into the following components:
#
# [Router] → [Quantize-Dispatch] → [Permute-Experts-Unpermute] → [Combine]
#
# Each component will be independent of (but may inform) the others except for
# [Quantize-Dispatch] and `[Combine] (see below). The components can then be
# mixed and matched with so that DP+EP can be supported easily for multiple
# MoE kernel implementations.
#
# The following main classes are defined:
# * FusedMoEPrepareAndFinalizeModular - an abstract base class for preparation of MoE
#   inputs (e.g. quantization, distribution) and finalization of Moe outputs.
#   The prepare method must take care of any needed quantization and the
#   finalize method, informed by the FusedMoEExpertsModular method,
#   may apply weights and/or do the final reduction of the output.
# * FusedMoEExpertsModular - an abstract base class for the main fused
#   MoE operation, i.e matmul + act_mul + optionally quant + matmul.
#   Some FusedMoEExpertsModular implementations may choose to do
#   the weight application and/or reduction. The class communicates this
#   to [Finalize] via a TopKWeightAndReduce object.
# * FusedMoEModularKernel - an interface class that combines a
#   FusedMoEPrepareAndFinalizeModular and a FusedMoEExpertsModular to
#   provide the standard fused MoE kernel interface.
# * TopKWeightAndReduce - A TopKWeightAndReduce implementation chosen
#   by the FusedMoEExpertsModular implementation that is passed
#   on to [Finalize].
#
# [Quantize-Prepare] and [Finalize] functionality are bundled into a single
# class `FusedMoEPrepareAndFinalizeModular` since they could use collective
# communication mechanisms that need to be consistent.
#


class FusedMoEActivationFormat(Enum):
    """
    The standard activation format (num_tokens, hidden dim).
    """

    Standard = ("standard",)
    """
    The batched experts format (num experts, max tokens per expert, hidden dim)
    """
    BatchedExperts = ("batched_experts",)


@dataclass
class ExpertTokensMetadata:
    """
    Metadata regarding expert-token routing.
    """

    expert_num_tokens: torch.Tensor
    expert_num_tokens_cpu: torch.Tensor | None

    @staticmethod
    def make_from_list(
        expert_num_tokens_list: list[int], device: str
    ) -> "ExpertTokensMetadata":
        expert_num_tokens_cpu = torch.tensor(
            expert_num_tokens_list,
            device="cpu",
            dtype=torch.int32,
            pin_memory=True,
        )
        return ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens_cpu.to(device, non_blocking=True),
            expert_num_tokens_cpu=expert_num_tokens_cpu,
        )


class TopKWeightAndReduce(ABC):
    """
    An abstract base class for weight application and reduction implementations.
    """

    @abstractmethod
    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        """
        Apply topk_weights to the fused_experts_outputs and/or reduce.
        If an output tensor is not passed, it will be created in the
        function.
        """
        raise NotImplementedError


#
# PrepareResultType is a tuple of:
# - quantized + dispatched a.
# - quantized + dispatched a1_scales.
# - Optional ExpertTokensMetadata containing gpu/cpu tensors
#   as big as the number of local experts with the information about the
#   number of tokens assigned to each local expert.
# - Optional dispatched expert topk IDs
# - Optional dispatched expert topk weight
#
# See `prepare` method below.
#
PrepareResultType = tuple[
    torch.Tensor,
    torch.Tensor | None,
    ExpertTokensMetadata | None,
    torch.Tensor | None,
    torch.Tensor | None,
]

#
# PrepareResultType is a tuple of:
# - quantized + dispatched a.
# - quantized + dispatched a1_scales.
# - dispatched router logits.
#
# See `prepare_monolithic` method below.
#
PrepareMonolithicResultType = tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]

ReceiverType = Callable[[], PrepareResultType]

################################################################################
# Prepare/Finalize
################################################################################


class FusedMoEPrepareAndFinalize(ABC):
    """
    An abstract base class for the [Quantize-Prepare] and [Finalize] steps
    described above.

    There are two variants of this class:
    * FusedMoEPrepareAndFinalizeModular - this operates on topk ids and weights
    * FusedMoEPrepareAndFinalizeMonolithic - the operates on router_logits
    """

    def post_init_setup(self, fused_experts: "FusedMoEExperts"):
        """
        Initialize FusedMoEPrepareAndFinalizeModular settings that depend on
        FusedMoEExpertsModular experts object.
        The FusedMoEPrepareAndFinalizeModular implementations that have such
        dependencies may choose to override this function.
        """
        return

    @property
    @abstractmethod
    def activation_format(self) -> FusedMoEActivationFormat:
        """
        A property indicating the output format of the activations for the
        'prepare' method.
        """
        raise NotImplementedError

    @abstractmethod
    def topk_indices_dtype(self) -> torch.dtype | None:
        """
        The PrepareFinalize All2All implementations generally constrain the
        dtype of the topk_ids they support. This function returns the
        required topk indices dtype so it can be respected.
        Return None if there are no such restrictions.
        """
        raise NotImplementedError

    @abstractmethod
    def max_num_tokens_per_rank(self) -> int | None:
        """
        Some PrepareFinalize All2All implementations are batched. Meaning,
        they can process only as set of tokens at a time. This
        function returns the batch size i.e the maximum number of tokens
        the implementation can process at a time.
        Return None if there are no such restrictions.
        """
        raise NotImplementedError

    @abstractmethod
    def num_dispatchers(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def output_is_reduced(self) -> bool:
        """
        Indicates whether or not the output of finalize is reduced across all
        ranks.
        """
        raise NotImplementedError

    def supports_async(self) -> bool:
        """
        Indicates whether or not this class implements prepare_async and
        finalize_async.
        """
        return False

    def on_commit(self) -> None:
        """
        Runs after this prepare/finalize has been committed to the active
        MoE kernel.
        """
        return


# TODO: pass FusedMoEParallelConfig in as ctor parameter?
class FusedMoEPrepareAndFinalizeModular(FusedMoEPrepareAndFinalize):
    """
    An abstract base class for the [Quantize-Prepare] and [Finalize] steps
    described above for the Modular case.
    """

    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> PrepareResultType:
        """
        Perform any quantization (and/or) dispatching needed for this kernel.
        - a1: The (unquantized) input to the MoE layer.
        - topk_ids: The topk ids.
        - topk_weights: The topk weights.
        - num_experts: The total number of experts in the global expert space.
        - expert_map: A tensor mapping expert indices from the global expert
          space to the local expert space of the expert parallel shard.
        - apply_router_weight_on_input: When True, apply the weights to the
          activations, before quantization + dispatching.
        - quant_config: Quantization info provided by the fused experts.
        - defer_input_quant: Runtime parameter indicating whether or not to
          defer input quantization to the FusedMoEExpertsModular
          in cases where the compute kernel expects unquantized inputs

        Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        - Optional ExpertTokensMetadata containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        raise NotImplementedError

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> tuple[Callable, ReceiverType] | ReceiverType:
        """
        Perform any quantization (and/or) dispatching needed for this kernel
        but do not wait for results from other workers.
        - a1: The (unquantized) input to the MoE layer.
        - a1_scale: Optional scales for a1
        - a2_scale: Optional scales for the second MoE gemm.  Required to make
          sure the quantization is consistent for both gemms.
        - topk_ids: The topk ids.
        - topk_weights: The topk weights.
        - num_experts: The total number of experts in the global expert space.
        - expert_map: A tensor mapping expert indices from the global expert
          space to the local expert space of the expert parallel shard.
        - apply_router_weight_on_input: When True, apply the weights to the
          activations, before quantization + dispatching.
        - defer_input_quant: Runtime parameter indicating whether or not to
          defer input quantization to the FusedMoEExpertsModular
          in cases where the compute kernel expects unquantized inputs

        Returns a callback or a hook callback pair that when invoked waits for
        results from other workers and has the same return signature as
        `prepare`, if a hook is returned this is more lightweight check that
        the recv is complete without doing extra work (used by DBO, will be
        refactored in the very near future)

        e.g.

        ret = obj.prepare_async(...)

        if isinstance(ret, tuple):
            hook, receiver = ret
            hook()

        if hook is not None:
        a, a_scales, expert_meta, topk_ids, topk_weights = receiver()

        is equivalent to:

        a, a_scales, expert_meta, topk_ids, topk_weights = obj.prepare(...)
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
    ) -> torch.Tensor | None:
        """
        Perform any combine plus apply weights and perform a reduction on the
        fused experts output.
        - output: The output tensor, written in place.  Must be (M, K) shape.
          An implementation may instead return an (M, K) result tensor, which
          the caller then uses in place of ``output``.
        - fused_expert_output: The unweighted, unreduced output of the fused
          experts, it will have (M, topk, K) shape.
        - topk_weights: The weights to be applied to the fused_experts_output.
        - topk_ids: The topk_ids.
        - apply_router_weight_on_input: When False, apply the weights to
          fused_expert_output.
        - weight_and_reduce_impl: An optional TopKWeightAndReduce
          implementation.
        """
        raise NotImplementedError

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
    ) -> tuple[Callable, Callable] | Callable:
        """
        Perform any combine plus apply weights and perform a reduction on the
        fused experts output but do not wait for results from other workers.
        - output: The output tensor, written in place.  Must be (M, K) shape.
        - fused_expert_output: The unweighted, unreduced output of the fused
          experts, it will have (M, topk, K) shape.
        - topk_weights: The weights to be applied to the fused_experts_output.
        - topk_ids: The topk_ids.
        - apply_router_weight_on_input: When False, apply the weights to
          fused_expert_output.
        - weight_and_reduce_impl: An optional TopKWeightAndReduce
          implementation.

        Returns a callback or a hook callback pair that when invoked waits for
        results from other workers and has the same return signature as
        `finalize`, if a hook is returned this is more lightweight check that
        the recv is complete without doing extra work (used by DBO, will be
        refactored in the very near future)

        ret = obj.finalize_async(output, ...)
        ... output not valid yet ...
        if isinstance(ret, tuple):
            hook, receiver = ret
            hook()
        receiver()
        ... output valid here ...

        is equivalent to:

        obj.finalize(output, ...)
        """
        raise NotImplementedError


class FusedMoEPrepareAndFinalizeMonolithic(FusedMoEPrepareAndFinalize):
    """
    An abstract base class for the [Quantize-Prepare] and [Finalize] steps
    described above for the monolithic case.
    """

    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> PrepareMonolithicResultType:
        """
        Optional method for subclasses compatible with monolithic
        FusedMoEExpertsModular kernels.

        Perform any quantization (and/or) dispatching needed for this kernel.
        - a1: The (unquantized) input to the MoE layer.
        - quant_config: Quantization info provided by the fused experts.
        - defer_input_quant: Runtime parameter indicating whether or not to
            defer input quantization to the FusedMoEExpertsModular

        Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self, fused_expert_output: torch.Tensor) -> torch.Tensor:
        """
        Optional method for subclasses compatible with monolithic
        FusedMoEExpertsModular kernels.

        Perform any combine plus apply weights and perform a reduction on the
        fused experts output.
        - fused_expert_output: The unweighted, unreduced output of the fused
          experts, it will have (M, topk, K) shape.
        """
        raise NotImplementedError


################################################################################
# Experts
################################################################################


# TODO: add supported activations method (return string)
class FusedMoEExperts(ABC):
    # Only AITER experts consume the 0/1 expert mask. Keep the vLLM 0.28
    # capability contract so RoutedExperts.expert_map can safely select the
    # canonical map for HCU Triton and DeepGEMM experts.
    consumes_expert_mask: bool = False

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        """
        moe_config: MoE layer configuration.
        quant_config: Quantization parameters for this experts instance.
        """
        if self.activation_format() == FusedMoEActivationFormat.Standard and (
            max_num_tokens is not None or num_dispatchers is not None
        ):
            raise ValueError(
                "max_num_tokens and num_dispatchers should only be set for "
                "BatchedExperts activation format."
            )
        elif self.activation_format() == FusedMoEActivationFormat.BatchedExperts and (
            max_num_tokens is None or num_dispatchers is None
        ):
            raise ValueError(
                "max_num_tokens and num_dispatchers must be set for "
                "BatchedExperts activation format."
            )

        self.moe_config = moe_config
        self.quant_config = quant_config
        self.activation_config = ApplyMoEActivationConfig.from_configs(
            moe_config, quant_config
        )
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers
        self.expected_m = max_num_tokens

    def set_expected_m(self, expected_m: int) -> None:
        self.expected_m = expected_m

    def get_expected_m(self) -> int | None:
        return self.expected_m

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # noqa: B027
        pass

    @staticmethod
    def is_monolithic() -> bool:
        raise NotImplementedError("Implemented by subclasses.")

    @property
    def expects_unquantized_inputs(self) -> bool:
        """
        Whether or not the PrepareFinalize should defer input quantization
        in the prepare step. If True, then the Experts kernel will
        execute the input quantization itself.

        Sample subclasses that override are AITER and FlashInfer CUTLASS.
        """
        return False

    @staticmethod
    @abstractmethod
    def activation_format() -> FusedMoEActivationFormat:
        """
        A property which is a tuple of the input and output activation formats
        for the 'apply' method.
        """
        raise NotImplementedError

    #
    # Various helpers for registering support for various features.
    # Used by the oracle to select a particular kernel for a deployment.
    #

    @staticmethod
    def is_supported_config(
        cls: type["FusedMoEExperts"],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        def _make_reason(reason: str) -> str:
            return f"kernel does not support {reason}"

        if not cls._supports_current_device():
            return False, _make_reason(f"current device {current_platform.device_name}")
        elif not (moe_config.is_act_and_mul or cls._supports_no_act_and_mul()):
            return False, _make_reason("no act_and_mul MLP layer")
        elif not cls._supports_activation(moe_config.activation):
            return False, _make_reason(f"{moe_config.activation} activation")
        elif not cls._supports_quant_scheme(weight_key, activation_key):
            return False, _make_reason(
                f"quantization scheme {weight_key}x{activation_key}"
            )
        elif not cls._supports_parallel_config(moe_config.moe_parallel_config):
            return False, _make_reason(
                f"parallel config {moe_config.moe_parallel_config}"
            )
        elif not cls._supports_routing_method(
            moe_config.routing_method, weight_key, activation_key
        ):
            return False, _make_reason(f"routing method {moe_config.routing_method}")
        elif not cls._supports_router_logits_dtype(
            moe_config.router_logits_dtype,
            moe_config.routing_method,
        ):
            return False, _make_reason(
                f"router logits dtype {moe_config.router_logits_dtype}"
            )
        elif not cls._supports_shape(moe_config.hidden_dim):
            return False, _make_reason(
                f"{moe_config.hidden_dim} hidden dim is not supported"
            )
        elif activation_format != cls.activation_format():
            return False, _make_reason(f"{activation_format.value} activation format")
        elif envs.VLLM_BATCH_INVARIANT and not cls._supports_batch_invariance():
            return False, _make_reason("batch invariance")
        elif moe_config.is_lora_enabled and not cls.supports_lora():
            return False, _make_reason("LoRA")
        return True, None

    @staticmethod
    @abstractmethod
    def _supports_current_device() -> bool:
        """
        Whether the kernel supports the current device type
        (compute cability and current platform).
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _supports_no_act_and_mul() -> bool:
        """
        Whether the kernel supports act_and_mul=False, i.e.
        non-gated MoE models like Nemotron-Nano.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        """
        Whether the kernel supports a particular act function.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        """
        Whether the kernel supports deployment in particular parallel config.

        Can be overridden if a kernel does not support EP, SP or some other
        configuration.
        """
        raise NotImplementedError

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """
        Whether the kernel supports a routing method (e.g. GroupedTopK).

        Can be overridden by monolithic kernels that execute the router
        in addition to the experts if certain routers are not supported.
        """
        return True

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        """
        Whether a kernel supports a particular dtype for router logits input.

        Can be overridden by monolithic kernels that execute the router
        in addition to the experts if certain dtypes are not supported.
        """
        return True

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        """
        Whether a kernel supports a particular shape. Can be overridden if a kernel
        has specific shape requirements.
        """
        return True

    @staticmethod
    def _supports_batch_invariance() -> bool:
        """
        Whether the kernel supports batch invariance, i.e. the output does not
        depend on the order of the tokens in the input batch. This is useful
        for determining if the kernel can used with VLLM_BATCH_INVARIANT=1.
        """
        return False

    #
    # Various helpers for accessing quantization parameters from the
    # quant_config.
    #

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return self.quant_config.quant_dtype

    @property
    def weight_quant_dtype(self) -> torch.dtype | str | None:
        return self.quant_config.weight_quant_dtype

    @property
    def block_shape(self) -> list[int] | None:
        return self.quant_config.block_shape

    @property
    def per_act_token_quant(self) -> bool:
        return self.quant_config.per_act_token_quant

    @property
    def per_out_ch_quant(self) -> bool:
        return self.quant_config.per_out_ch_quant

    @property
    def a1_scale(self) -> torch.Tensor | None:
        return self.quant_config.a1_scale

    @property
    def a2_scale(self) -> torch.Tensor | None:
        return self.quant_config.a2_scale

    @property
    def a1_gscale(self) -> torch.Tensor | None:
        return self.quant_config.a1_gscale

    @property
    def a2_gscale(self) -> torch.Tensor | None:
        return self.quant_config.a2_gscale

    @property
    def w1_scale(self) -> torch.Tensor | None:
        return self.quant_config.w1_scale

    @property
    def w2_scale(self) -> torch.Tensor | None:
        return self.quant_config.w2_scale

    @property
    def w1_zp(self) -> torch.Tensor | None:
        return self.quant_config.w1_zp

    @property
    def w2_zp(self) -> torch.Tensor | None:
        return self.quant_config.w2_zp

    @property
    def w1_bias(self) -> torch.Tensor | None:
        return self.quant_config.w1_bias

    @property
    def w2_bias(self) -> torch.Tensor | None:
        return self.quant_config.w2_bias

    @property
    def g1_alphas(self) -> torch.Tensor | None:
        return self.quant_config.g1_alphas

    @property
    def g2_alphas(self) -> torch.Tensor | None:
        return self.quant_config.g2_alphas

    @staticmethod
    def supports_lora() -> bool:
        """Return True if this expert impl natively handles LoRA.

        LoRA-aware experts should mix in LoRAExpertsMixin, which flips this
        to True and provides the per-forward LoRA state plumbing.
        """
        return False

    def supports_packed_ue8m0_act_scales(self) -> bool:
        """
        A flag indicating whether or not this class can process packed ue8m0
        activation scales.
        """
        return False


class FusedMoEExpertsModular(FusedMoEExperts):
    """
    An abstract base class for the [Permute-Experts-Unpermute] step described
        above.
    """

    @staticmethod
    def is_monolithic() -> bool:
        return False

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """
        Extract the MoE problem size from the given tensor arguments:
        - a: The hidden states, input to the MoE layer.
        - w1: The first set of expert weights.
        - w2: The second set of expert weights.
        - topk_ids: The topk ids.

        Note: extracting the problem shape from the weight and activation
        tensors is not obvious.  It needs to be done this way specifically
        due to subtle issues with particular kernels, e.g. the int4 kernels
        divide the trailing dimension by two, so it's not "correct" to
        extract N or K from the trailing dimension of w1 or w2.  Similarly,
        some kernels transpose the weights, so this needs to be kept in mind.

        Note: This implementation covers most cases. However, if experts
        require a specialized implementation, like MarlinExperts, they are free
        to override this function.
        """
        if w1.dim() == 7:
            assert w2.dim() == 7 and w1.size(0) == w2.size(0)
            E = w1.size(0)
            # gfx938 pair32: [E, K/64, N/32, 2, 4, 16, 16].
            N = w1.size(2) * w1.size(3) * w1.size(5)
            K = w1.size(1) * w1.size(4) * w1.size(6)
            assert w2.size(2) * w2.size(3) * w2.size(5) == K
        elif w1.dim() == 6:
            assert w2.dim() == 6
            E = w1.size(0)
            # HCU FP8 DeepEP packed layout: [E, K/64, N/16, 4, 16, 16].
            N = w1.size(2) * w1.size(4)
            K = w1.size(1) * w1.size(3) * w1.size(5)
        else:
            assert len(w1.shape) == 3 and len(w2.shape) == 3
            E, N, _ = w1.shape
            K = a1.size(-1)

        if a1.dim() == 2:
            # Make sure we are using the correct a1 (pre-permute).
            assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
            M = a1.size(0)
        else:
            assert a1.dim() == 3
            assert a1.size(0) == E, f"{a1.size(0)} == {E}"
            M = a1.size(1)  # This is max_num_tokens

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)

        return E, M, N, K, topk

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        """
        Workspace type: The dtype to use for the workspace tensors.
        """
        return act_dtype

    @abstractmethod
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """
        Compute the shapes for the temporary and final outputs of the two gemms
        and activation in the fused expert function.  Since the gemms are
        independent, the workspace for the first gemm can be shared with the
        workspace for the last gemm.

        Inputs:
        - M: number of tokens.
        - N: Row (or column) dimension of expert weights.
        - K: hidden dimension
        - topk: The number of top-k experts to select.
        - global_num_experts: global number of experts.
        - local_num_experts: local number of experts due to DP/EP.
        - expert_tokens_meta: number of tokens per expert metadata for batched
                              format.

        Returns a tuple of:
        - workspace13 shape tuple: must be large enough to hold the
          result of either expert gemm.
        - workspace2 shape tuple: must be large enough to hold the
          result of the activation function.
        - output shape tuple: must be exact size of the final gemm output.
        - Note: workspace shapes can be 0 if the workspace is not needed.
          But in order for activation chunking to work, the first dimension
          of each tuple must be the number of tokens when the shape is
          not 0.
        """
        raise NotImplementedError

    @staticmethod
    def adjust_N_for_activation(N: int, activation: MoEActivation) -> int:
        """
        Calculate the output dimension for the activation function.

        For *_no_mul activations (e.g. relu2_no_mul),
        there's no gate/up split, so output size equals input size (N).

        For regular gated activations (e.g., silu, gelu, swigluoai),
        output size is N // 2 due to gate × activation(up) multiplication.

        Args:
            N: The intermediate size (width of w1/w3 weights).
            activation: The activation function enum.

        Returns:
            The output dimension after activation.
        """
        return N if not activation.is_gated else N // 2

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        *,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
        valid_token_counts: torch.Tensor | None = None,
    ) -> None:
        apply_moe_activation(
            activation,
            output,
            input,
            activation_config=self.activation_config,
            topk_ids=topk_ids,
            expert_map=expert_map,
            valid_token_counts=valid_token_counts,
        )

    @abstractmethod
    def finalize_weight_and_reduce_impl(self) -> TopKWeightAndReduce:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        """
        This function computes the intermediate result of a Mixture of Experts
        (MoE) layer using two sets of weights, w1 and w2.

        Parameters:
        - output: (torch.Tensor): The unweighted, unreduced output tensor.
        - hidden_states: (torch.Tensor): The (quantized) input tensor to the MoE
          layer.
        - w1 (torch.Tensor): The first set of expert weights.
        - w2 (torch.Tensor): The second set of expert weights.
        - topk_weights: A map of row to expert weights. Some implementations
          choose to do weight application.
        - topk_ids (torch.Tensor): A map of row to expert id.
        - activation (str): The activation function to apply after the first
          MoE layer.
        - global_num_experts (int): The total number of experts in the global
          expert space.
        - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
          from the global expert space to the local expert space of the expert
          parallel shard.
        - a1q_scale (Optional[torch.Tensor]): Optional quantized scale to be
          used for a1.  Result of quantization from prepare/finalize and not
          from the FusedMoEQuantConfig.
        - workspace13 (torch.Tensor): A scratch tensor used for gemm outputs
          must be large enough to hold output of either MoE gemm.
        - workspace2 (torch.Tensor): A scratch tensor used for the activation
          function.
        - expert_tokens_meta (Optional[ExpertTokensMetadata]) - An optional
          ExpertTokensMetadata object containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - apply_router_weight_on_input: True if router weights are already
          applied on the input. This is relevant if the implementation
          chooses to do weight application.
        """
        raise NotImplementedError


class FusedMoEExpertsMonolithic(FusedMoEExperts):
    """
    An abstract base class for the [Permute-Experts-Unpermute] step described
        above, but with the monolithic interface (accepts router logits
        rather than topk ids and weights).
    """

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """
        Whether the kernel supports a routing method (e.g. GroupedTopK).

        Monolithic kernels should explicitly opt-in to support.
        """
        raise NotImplementedError

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        """
        Whether the kernel supports a dtype for router logits.

        Modular kernels should opt-in to support.
        """
        raise NotImplementedError

    @staticmethod
    def is_monolithic() -> bool:
        return True

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        """
        Same as apply(), except uses router_logits as opposed
        to the topk_ids and topk_weights. This is useful for kernels
        with fused router and fused_experts (e.g. FLASHINFER_TRTLLM).
        """
        raise NotImplementedError


################################################################################
# Kernel
################################################################################


@final
class FusedMoEKernelModularImpl:
    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        fused_experts: FusedMoEExpertsModular,
        N: int = -1,
        K: int = -1,
    ):
        from vllm_hcu.runtime_compat.step5 import is_step5_owner

        self._hcu_step5_owner = is_step5_owner(fused_experts.moe_config)
        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts
        self.shared_experts: SharedExperts | None = None
        moe_parallel_config = fused_experts.moe_config.moe_parallel_config
        self.moe_parallel_config = moe_parallel_config
        self.is_dp_ep = (
            moe_parallel_config is not None
            and moe_parallel_config.dp_size > 1
            and moe_parallel_config.use_ep
        )
        self.N = N
        self.K = K

    def _allocate_buffers(
        self,
        out_dtype: torch.dtype,
        device: torch.device,
        M_chunk: int,
        M_full: int,
        N: int,
        K: int,
        top_k: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Allocate temporary and output buffers for the fused experts op.
        Inputs:
        - out_dtype: output type of workspace and output tensors.
        - device: the device of the workspace and output tensors.
        See `workspace_shapes` for a description of the remainder of arguments.
        Returns a tuple of (workspace13, workspace2, output) tensors.
        """
        assert M_full > 0 and M_chunk > 0

        workspace_dtype = self.fused_experts.workspace_dtype(out_dtype)

        # Get intermediate workspace shapes based off the chunked M size.
        workspace13_shape, workspace2_shape, _ = self.fused_experts.workspace_shapes(
            M_chunk,
            N,
            K,
            top_k,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

        # Get final output shape based on the full M size.
        _, _, fused_out_shape = self.fused_experts.workspace_shapes(
            M_full,
            N,
            K,
            top_k,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

        if M_chunk < M_full:
            # The inner EP loop reuses both scratch buffers across chunks.
            # Its full output must survive every later GEMM1 and GEMM2.
            workspace13, workspace2, fused_out = (
                current_workspace_manager().get_simultaneous(
                    (workspace13_shape, workspace_dtype),
                    (workspace2_shape, workspace_dtype),
                    (fused_out_shape, out_dtype),
                )
            )
            global _step5_ep_workspace_announced
            if not _step5_ep_workspace_announced:
                _step5_ep_workspace_announced = True
                print(
                    f"[step5_ep_workspace] active: full_M={M_full} "
                    f"chunk_M={M_chunk} separate_output=True", flush=True,
                )
            return workspace13, workspace2, fused_out

        # We can reuse the memory between cache1 and cache3 because by the
        # time we need cache3, we're done with cache1.
        # Reuse workspace13 for the output since there is only one chunk.
        max_shape_size = max(prod(workspace13_shape), prod(fused_out_shape))
        growth_mb = 0
        if (
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and not dbo_enabled()
            and self.moe_parallel_config is not None
            and self.moe_parallel_config.all2all_backend
            in ("deepep_auto", "deepep_high_throughput")
            and self.fused_experts.activation_format()
            == FusedMoEActivationFormat.Standard
        ):
            growth_mb = henvs.VLLM_HCU_MOE_WORKSPACE_GROWTH_MB
        common_workspace, workspace2 = get_moe_workspaces(
            current_workspace_manager(),
            (((max_shape_size,), workspace_dtype),
             (workspace2_shape, workspace_dtype)),
            growth_mb * 1024 * 1024,
        )
        workspace13 = _resize_cache(common_workspace, workspace13_shape)
        fused_out = _resize_cache(common_workspace, fused_out_shape)

        return workspace13, workspace2, fused_out

    def _maybe_apply_shared_experts(
        self,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        if shared_experts is not None:
            assert self.prepare_finalize.supports_async()
            assert shared_experts_input is not None
            shared_experts(
                shared_experts_input,
                SharedExpertsOrder.MK_INTERNAL_OVERLAPPED,
                x_and_scale_quanted=x_and_scale_quanted,
            )

    def _prepare(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        prequanted_a1_scale: torch.Tensor | None = None,
        prequanted_a1: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        ExpertTokensMetadata | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        The _prepare method is a wrapper around self.prepare_finalize.prepare
        that handles DBO and async.
        """
        begin_moe_call = getattr(
            self.prepare_finalize,
            "begin_moe_call",
            None,
        )
        if begin_moe_call is not None:
            begin_moe_call()

        if self.fused_experts.activation_format() == FusedMoEActivationFormat.BatchedExperts:
            num_tokens = hidden_states.shape[0]
            num_dispatchers = self.fused_experts.num_dispatchers
            topk = topk_ids.shape[1]
            if num_dispatchers is not None and global_num_experts > 0:
                expected_m = (
                    num_tokens * num_dispatchers * topk + global_num_experts
                ) // global_num_experts
                self.fused_experts.set_expected_m(expected_m)

        use_all2all_kernels = (
            self.moe_parallel_config is not None
            and self.moe_parallel_config.use_all2all_kernels
        )
        if prequanted_a1 and not use_all2all_kernels:
            return (
                hidden_states,
                prequanted_a1_scale,
                None,
                topk_ids,
                topk_weights,
            )

        if not self.prepare_finalize.supports_async():
            # We shouldn't be running an a2a kernel that doesn't
            # support async prepare/finalize
            # TODO(lucas): enable in follow-up
            assert not dbo_enabled()

            (
                a1q,
                a1q_scale,
                expert_tokens_meta,
                _expert_topk_ids,
                _expert_topk_weights,
            ) = self.prepare_finalize.prepare(
                hidden_states,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                self.fused_experts.quant_config,
                defer_input_quant=self.fused_experts.expects_unquantized_inputs,
            )
        else:
            # Overlap shared expert compute with all2all dispatch.
            dbo_maybe_run_recv_hook()
            prepare_ret = self.prepare_finalize.prepare_async(
                hidden_states,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                self.fused_experts.quant_config,
                defer_input_quant=self.fused_experts.expects_unquantized_inputs,
            )

            # TODO(lucas): refactor this in the alternative schedules followup
            # currently unpack if we have hook + receiver pair or just
            # receiver (see finalize_async docstring)
            hook, receiver = (
                prepare_ret if isinstance(prepare_ret, tuple) else (None, prepare_ret)
            )

            if hook is not None:
                if dbo_enabled():
                    # If DBO is being used, register the hook with the ubatch
                    # context and call it in dbo_maybe_run_recv_hook instead of
                    #  passing it to the receiver.
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    hook()

            (
                a1q,
                a1q_scale,
                expert_tokens_meta,
                _expert_topk_ids,
                _expert_topk_weights,
            ) = receiver()

        # Maybe prepare gathered topk_ids and topk_weights from other EP ranks.
        topk_ids = topk_ids if _expert_topk_ids is None else _expert_topk_ids
        topk_weights = (
            topk_weights if _expert_topk_weights is None else _expert_topk_weights
        )

        return a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights

    def _fused_experts(
        self,
        in_dtype: torch.dtype,
        a1q: torch.Tensor,
        a1q_scale: torch.Tensor | None,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        local_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        expert_tokens_meta: ExpertTokensMetadata | None,
        output_alias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, M_full, N, K, top_k = self.fused_experts.moe_problem_size(
            a1q, w1, w2, topk_ids
        )
        if self.N > 0:
            N = self.N
            K = self.K

        # This happens when none of the tokens from the all2all reach this
        # EP rank. Also, note that this is only relevant for CUDAGraph
        # incompatible all2all kernels like the DeepEP high-throughput
        # kernels. CUDAGraph compatible all2all kernels like the DeepEP
        # low-latency kernels are always batched and can never run into
        # the tensor.numel() == 0 case.
        if M_full == 0:
            return torch.empty_like(a1q, dtype=in_dtype)

        M_chunk = M_full
        private_native_ep_output = False
        chunk_provider = _step5_ep_chunk_provider(self.fused_experts, a1q, w1, w2)
        if chunk_provider is not None and not dbo_enabled():
            from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep

            pf = self.prepare_finalize
            if (
                type(pf) is naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular
                and not pf.is_sequence_parallel
                and pf._lora_context is None
            ):
                ep_chunk = chunk_provider(
                    w1, w2, activation, global_num_experts, expert_map,
                    top_k, apply_router_weight_on_input,
                )
                if ep_chunk is not None:
                    M_chunk = min(M_full, ep_chunk)
                    private_native_ep_output = True

        workspace13, workspace2, fused_out = self._allocate_buffers(
            in_dtype,
            a1q.device,
            M_chunk,
            M_full,
            N,
            K,
            top_k,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

        # If caller's output buffer already matches fused_out shape/dtype, alias
        # to skip the redundant copy in TopKWeightAndReduceNoOP.apply downstream.
        # This eliminates most redundant copy events in the double-copy MoE
        # write-back path.
        if current_platform.is_rocm():
            from vllm._aiter_ops import rocm_aiter_ops

            if (
                rocm_aiter_ops.is_fused_moe_enabled()
                and output_alias is not None
                and output_alias.shape == fused_out.shape
                and output_alias.dtype == fused_out.dtype
                and output_alias.device == fused_out.device
                and output_alias.is_contiguous()
            ):
                fused_out = output_alias

        if getattr(self.prepare_finalize, "_hcu_dbo_private_output", False):
            # HT yields before combine. Even disjoint views in this step's
            # shared workspace can be overwritten by the next microbatch.
            fused_out = torch.empty_like(fused_out)
        elif private_native_ep_output:
            # A mixed-size native step needs scratch reuse to avoid a second
            # M*topk*K allocation instead of this much smaller M*K result.
            fused_out = _step5_ep_private_output(workspace13, fused_out)

        self.fused_experts.apply(
            output=fused_out,
            hidden_states=a1q,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=self.fused_experts.a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        return fused_out

    def _finalize(
        self,
        output: torch.Tensor,
        fused_out: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        The _finalize method is a wrapper around self.prepare_finalize.finalize
        that handles DBO, async and shared expert overlap.

        Args:
            shared_experts: SharedExperts | None. The shared experts if any.
            shared_experts_input: Optional separate input for shared experts.
                When latent MoE is used, hidden_states is the latent-projected
                tensor (smaller dimension) used by routed experts, while
                shared_experts_input is the original hidden_states (full
                dimension) needed by the shared expert MLP.
        """
        if not self.prepare_finalize.supports_async():
            assert not dbo_enabled()

            # A prepare/finalize may hand back the combined result instead of
            # writing it into ``output`` (naive DP/EP with a combine hook).
            combined = self.prepare_finalize.finalize(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                self.fused_experts.finalize_weight_and_reduce_impl(),
            )
            if combined is not None:
                return combined
        else:
            finalize_ret = self.prepare_finalize.finalize_async(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                self.fused_experts.finalize_weight_and_reduce_impl(),
            )
            self._maybe_apply_shared_experts(
                shared_experts,
                shared_experts_input,
                x_and_scale_quanted=x_and_scale_quanted,
            )

            # TODO(lucas): refactor this in the alternative schedules followup
            # currently unpack if we have hook + receiver pair or just
            # receiver (see finalize_async docstring)
            hook, receiver = (
                finalize_ret
                if isinstance(finalize_ret, tuple)
                else (None, finalize_ret)
            )

            if hook is not None:
                if dbo_enabled():
                    # If DBO is being used, register the hook with the ubatch
                    # context and call it in dbo_maybe_run_recv_hook instead of
                    #  passing it to the receiver.
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    hook()

            receiver()

        return output

    def _dp_pipe_apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        local_num_experts: int,
        expert_map: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """See _DP_MOE_PIPE_MIN_TOKENS.  None: not applicable, nothing done."""
        if not henvs.custom_ops_enabled():
            return None
        global _dp_pipe_stream, _dp_pipe_pending
        from vllm.distributed import get_dp_group
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            buffer_type,
            cudaStream_t,
            ncclRedOpTypeEnum,
        )
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceNoOP,
        )

        pf = self.prepare_finalize
        fe = self.fused_experts
        # Microbatched (HCU_DBO_MIN_TOKENS, vllm_hcu.v1.hcu_dbo): one group, its
        # dispatch and combine on the ubatch comm stream overlapping the other
        # microbatch's compute instead of this one's.
        dbo = dbo_enabled()
        if not (
            type(pf) is naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular
            and not pf.is_sequence_parallel
            and pf._lora_context is None
            and getattr(fe, "_lora_context", None) is None
            and fe.activation_format() == FusedMoEActivationFormat.Standard
            and not fe.expects_unquantized_inputs
            and isinstance(fe.finalize_weight_and_reduce_impl(), TopKWeightAndReduceNoOP)
        ):
            return None
        dp_meta = get_forward_context().dp_metadata
        if dp_meta is None:
            return None
        sizes = dp_meta.get_chunk_sizes_across_dp_rank()
        if sizes is None or (not dbo and sum(sizes) < _DP_MOE_PIPE_MIN_TOKENS):
            return None
        dp = get_dp_group()
        comm = getattr(dp.device_communicator, "pynccl_comm", None)
        if comm is None or comm.disabled or len(sizes) < 2:
            return None
        assert sizes[dp.rank_in_group] == hidden_states.shape[0]
        join_dp_moe_pipe()
        global _dp_pipe_announced
        if not _dp_pipe_announced:
            _dp_pipe_announced = True
            print(f"[dp_moe_pipe] active: sizes={sizes} dbo={dbo}", flush=True)

        a1q, scales, a1q_scale = naive_dp_ep._quantize_and_setup_dispatch(
            hidden_states, fe.quant_config
        )
        local = [a1q, topk_weights, topk_ids] + (scales or [])
        W = len(sizes)
        offs = [0]
        for n in sizes:
            offs.append(offs[-1] + n)
        M = offs[-1]
        gath = [
            torch.empty((M,) + t.shape[1:], dtype=t.dtype, device=t.device)
            for t in local
        ]
        if scales:
            a1q_scale = gath[3]
        K = hidden_states.shape[1]
        groups = [range(W)] if dbo else [range(0, W // 2), range(W // 2, W)]
        # HCU_EP_DIRECT_SUM: spare rows right after moe_out serve as the direct
        # combine's scratch (see ep_fused_moe.DIRECT_MIN); one half's MoE at a time.
        S = (
            max(offs[g[-1] + 1] - offs[g[0]] for g in groups if len(g))
            if _EP_DIRECT_SUM
            else 0
        )
        if S:
            from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe as _ep_mod
        moe_full = torch.empty((M + S, K), dtype=hidden_states.dtype, device=a1q.device)
        moe_out = moe_full[:M]
        result = torch.empty_like(hidden_states)

        nccl, ncomm = comm.nccl, comm.comm
        sum_op = ncclRedOpTypeEnum.from_torch(torch.distributed.ReduceOp.SUM)

        # The synchronized DP metadata makes this decision identical on peers.
        # Half-DP pipelines and nonuniform/empty shards retain the grouped path.
        uniform = (
            _DP_PIPE_UNIFORM and dbo and len(sizes) == W
            and sizes[0] > 0 and all(n == sizes[0] for n in sizes)
        )
        if uniform and not getattr(self, "_step5_uniform_announced", False):
            self._step5_uniform_announced = True
            print(f"[dp_moe_pipe] uniform collectives: sizes={sizes}", flush=True)

        def broadcast(grp, stream) -> None:
            st = cudaStream_t(stream.cuda_stream)
            if uniform:
                assert len(grp) == W and grp[0] == 0 and grp[-1] == W - 1
                nccl.ncclGroupStart()
                for src, dst in zip(local, gath):
                    nccl.ncclAllGather(
                        buffer_type(src.data_ptr()), buffer_type(dst.data_ptr()),
                        src.numel(), _nccl_dtype(src), ncomm, st,
                    )
                nccl.ncclGroupEnd()
                return
            nccl.ncclGroupStart()
            for src, dst in zip(local, gath):
                for root in grp:
                    sl = dst[offs[root] : offs[root + 1]]
                    nccl.ncclBroadcast(
                        buffer_type(src.data_ptr()),
                        buffer_type(sl.data_ptr()),
                        sl.numel(),
                        _nccl_dtype(src),
                        root,
                        ncomm,
                        st,
                    )
            nccl.ncclGroupEnd()

        def experts(grp) -> None:
            a, b = offs[grp[0]], offs[grp[-1] + 1]
            if b <= a:
                return
            _, _, N, Kw, top_k = fe.moe_problem_size(
                gath[0][a:b], w1, w2, gath[2][a:b]
            )
            if self.N > 0:
                N, Kw = self.N, self.K
            workspace13, workspace2, _ = self._allocate_buffers(
                hidden_states.dtype,
                a1q.device,
                b - a,
                b - a,
                N,
                Kw,
                top_k,
                global_num_experts,
                local_num_experts,
                None,
                activation,
            )
            if S:
                _ep_mod.direct_scratch = moe_full[M:]
            # This path owns its NCCL combine and never calls finalize_hook.
            # Keep the shared result separate for the runner's final add. The
            # scope is local to this microbatch's compute, not the global hook.
            with suspend_ep_shared_add():
                fe.apply(
                    output=moe_out[a:b],
                    hidden_states=gath[0][a:b],
                    w1=w1,
                    w2=w2,
                    topk_weights=gath[1][a:b],
                    topk_ids=gath[2][a:b],
                    activation=activation,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                    a1q_scale=(
                        a1q_scale[a:b]
                        if a1q_scale is not None and a1q_scale.numel() > 1
                        else a1q_scale
                    ),
                    a2_scale=fe.a2_scale,
                    workspace13=workspace13,
                    workspace2=workspace2,
                    expert_tokens_meta=None,
                    apply_router_weight_on_input=False,
                )
            if S:
                _ep_mod.direct_scratch = None

        def reduce(grp, stream) -> None:
            st = cudaStream_t(stream.cuda_stream)
            if uniform:
                assert len(grp) == W and grp[0] == 0 and grp[-1] == W - 1
                nccl.ncclReduceScatter(
                    buffer_type(moe_out.data_ptr()), buffer_type(result.data_ptr()),
                    result.numel(), _nccl_dtype(result), sum_op, ncomm, st,
                )
                return
            nccl.ncclGroupStart()
            for root in grp:
                ch = moe_out[offs[root] : offs[root + 1]]
                nccl.ncclReduce(
                    buffer_type(ch.data_ptr()),
                    buffer_type(result.data_ptr()),
                    ch.numel(),
                    _nccl_dtype(ch),
                    sum_op,
                    root,
                    ncomm,
                    st,
                )
            nccl.ncclGroupEnd()

        if dbo:
            # Every buffer here was allocated on the compute stream, which waits
            # for the combine before this returns, so none is freed early.
            dbo_yield_and_switch_from_compute_to_comm()
            broadcast(groups[0], torch.cuda.current_stream())
            dbo_yield_and_switch_from_comm_to_compute()
            experts(groups[0])
            dbo_yield_and_switch_from_compute_to_comm()
            reduce(groups[0], torch.cuda.current_stream())
            dbo_yield_and_switch_from_comm_to_compute()
            return result

        if _dp_pipe_stream is None:
            _dp_pipe_stream = torch.cuda.Stream(device=a1q.device)
        side = _dp_pipe_stream
        cur = torch.cuda.current_stream()
        side.wait_stream(cur)

        ag_ev = []
        for grp in groups:
            broadcast(grp, side)
            ev = torch.cuda.Event()
            ev.record(side)
            ag_ev.append(ev)

        moe_ev = []
        for grp, ev in zip(groups, ag_ev):
            cur.wait_event(ev)
            experts(grp)
            ev = torch.cuda.Event()
            ev.record(cur)
            moe_ev.append(ev)

        for grp, ev in zip(groups, moe_ev):
            side.wait_event(ev)
            reduce(grp, side)
        _dp_pipe_pending = (side, local, gath, moe_out, result)
        return result

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        activation: MoEActivation = MoEActivation.SILU,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        shared_experts: SharedExperts | None = None,
        shared_experts_input: torch.Tensor | None = None,
        quanted_hidden_states: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        This function computes a Mixture of Experts (MoE) layer using two sets
        of weights, w1 and w2, and top-k gating mechanism.

        Parameters:
        - hidden_states: (torch.Tensor): The input tensor to the MoE layer.
        - w1 (torch.Tensor): The first set of expert weights.
        - w2 (torch.Tensor): The second set of expert weights.
        - topk_weights (torch.Tensor): The topk weights applied at the end of the layer.
        - topk_ids (torch.Tensor): A map of row to expert id.
        - activation (MoEActivation): The activation function to apply after the first
          MoE layer.
        - global_num_experts (int): The total number of experts in the global
          expert space.
        - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
          from the global expert space to the local expert space of the expert
          parallel shard.
        - apply_router_weight_on_input (bool): When true, the topk weights are
          applied directly on the inputs. This is only applicable when topk is
          1.
        - shared_experts: SharedExperts | None. The shared experts if any.
        - shared_experts_input (Optional[torch.Tensor]): Optional separate
          input for shared experts. For latent MoE, this is the original
          hidden_states before latent projection.

        Returns:
        - torch.Tensor: The output tensor after applying the MoE layer.
        """
        local_num_experts = w1.shape[0]
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        if (
            henvs.custom_ops_enabled()
            and _DP_MOE_PIPE_MIN_TOKENS > 0
            and self._hcu_step5_owner
            and quanted_hidden_states is None
            and not apply_router_weight_on_input
        ):
            piped = self._dp_pipe_apply(
                hidden_states,
                w1,
                w2,
                topk_ids,
                topk_weights,
                activation,
                global_num_experts,
                local_num_experts,
                expert_map,
            )
            if piped is not None:
                return piped

        output = torch.empty_like(hidden_states)

        a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights = self._prepare(
            quanted_hidden_states
            if quanted_hidden_states is not None
            else hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
            prequanted_a1_scale=scale,
            prequanted_a1=quanted_hidden_states is not None,
        )

        # Stash the original unquantized hidden states on the LoRA context
        # so apply_w13_lora sees correct-magnitude activations instead of
        # the potentially quantized values produced by _prepare().
        lora_ctx = getattr(self.fused_experts, "_lora_context", None)
        if lora_ctx is not None:
            lora_ctx.original_hidden_states = hidden_states

        fused_out = self._fused_experts(
            in_dtype=hidden_states.dtype,
            a1q=a1q,
            a1q_scale=a1q_scale,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_tokens_meta=expert_tokens_meta,
            output_alias=output,
        )

        if lora_ctx is not None:
            lora_ctx.original_hidden_states = None

        return self._finalize(
            output,
            fused_out,
            hidden_states,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
            x_and_scale_quanted=(
                (quanted_hidden_states, scale)
                if quanted_hidden_states is not None and scale is not None
                else None
            ),
        )


@final
class FusedMoEKernelMonolithicImpl:
    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeMonolithic,
        fused_experts: FusedMoEExpertsMonolithic,
    ):
        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        """
        Same as forward(), except uses router_logits as opposed
        to the topk_ids and topk_weights. This is used for kernels
        that have fused router + experts (e.g. FLASHINFER_TRTLLM).
        """

        a1q, a1q_scale, router_logits = self.prepare_finalize.prepare(
            hidden_states,
            router_logits=router_logits,
            quant_config=self.fused_experts.quant_config,
            defer_input_quant=self.fused_experts.expects_unquantized_inputs,
        )

        fused_out = self.fused_experts.apply(
            hidden_states=a1q,
            w1=w1,
            w2=w2,
            router_logits=router_logits,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            a1q_scale=a1q_scale,
            # grouped topk + fused topk bias parameters
            num_expert_group=num_expert_group,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            topk_group=topk_group,
        )

        output = self.prepare_finalize.finalize(fused_out)

        return output


@final
class FusedMoEKernel:
    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        fused_experts: FusedMoEExperts,
        N: int = -1,
        K: int = -1,
    ):
        super().__init__()

        # Initialize the implementation (monolithic or modular).
        self.impl: FusedMoEKernelModularImpl | FusedMoEKernelMonolithicImpl
        if isinstance(
            prepare_finalize, FusedMoEPrepareAndFinalizeModular
        ) and isinstance(fused_experts, FusedMoEExpertsModular):
            self.impl = FusedMoEKernelModularImpl(
                prepare_finalize,
                fused_experts,
                N=N,
                K=K,
            )

        elif isinstance(
            prepare_finalize, FusedMoEPrepareAndFinalizeMonolithic
        ) and isinstance(fused_experts, FusedMoEExpertsMonolithic):
            self.impl = FusedMoEKernelMonolithicImpl(
                prepare_finalize,
                fused_experts,
            )

        else:
            raise ValueError(
                "prepare_finalize and fused_experts must both be either monolithic "
                f"or non-monolithic but got {prepare_finalize.__class__.__name__} "
                f"and {fused_experts.__class__.__name__}"
            )

        self._post_init_setup()

    @property
    def can_overlap_shared_experts(self) -> bool:
        if isinstance(self.impl, FusedMoEKernelModularImpl):
            return self.impl.prepare_finalize.supports_async()
        else:
            return False

    @property
    def is_monolithic(self) -> bool:
        return isinstance(self.impl, FusedMoEKernelMonolithicImpl)

    @property
    def prepare_finalize(self) -> FusedMoEPrepareAndFinalize:
        return self.impl.prepare_finalize

    @property
    def fused_experts(self) -> FusedMoEExperts:
        return self.impl.fused_experts

    @property
    def moe_config(self) -> FusedMoEConfig:
        return self.fused_experts.moe_config

    def supports_lora(self) -> bool:
        return self.fused_experts.supports_lora()

    def _post_init_setup(self):
        """
        Resolve any leftover setup dependencies between self.prepare_finalize
        and self.fused_experts here.
        """
        self.prepare_finalize.post_init_setup(self.impl.fused_experts)
        assert (
            self.prepare_finalize.activation_format
            == self.fused_experts.activation_format()
        )

    def output_is_reduced(self) -> bool:
        """
        Indicates whether or not the output of fused MoE kernel
        is reduced across all ranks.
        """
        return self.prepare_finalize.output_is_reduced()

    def apply_monolithic(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        assert isinstance(self.impl, FusedMoEKernelMonolithicImpl)
        return self.impl.apply(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            router_logits=router_logits,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            num_expert_group=num_expert_group,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            topk_group=topk_group,
        )

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        shared_experts: SharedExperts | None = None,
        shared_experts_input: torch.Tensor | None = None,
        quanted_hidden_states: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert isinstance(self.impl, FusedMoEKernelModularImpl)
        return self.impl.apply(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
            quanted_hidden_states=quanted_hidden_states,
            scale=scale,
        )
