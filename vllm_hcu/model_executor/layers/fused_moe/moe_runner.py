# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""HCU-owned v0.25.1 MoE runner with pre-quantized input support."""

import inspect
import os
from collections.abc import Callable, Iterable
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.routed_experts import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerName,
    direct_register_custom_op,
)
from vllm_hcu.platforms import envs as henvs

logger = init_logger(__name__)

# Fold the naive DP combine (reduce-scatter over DP), the shared-expert add and
# the TP all-reduce into one custom all-reduce over the EP group (see
# _ep_fused_all_reduce).  Requires the EP group's custom all-reduce, which
# CudaCommunicator creates under the same flag.
_EP_FUSED_COMBINE = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_EP_FUSED_COMBINE", "0") == "1"
)


def _ep_fused_rows(hidden_states: torch.Tensor) -> tuple[int, int] | None:
    """This DP rank's rows of ``hidden_states`` if the EP custom all-reduce
    can combine it (see _ep_fused_all_reduce), else None."""
    ep = get_ep_group()
    dp = get_dp_group()
    ca = getattr(ep.device_communicator, "ca_comm", None)
    if (
        ca is None
        or ca.disabled
        or ep.world_size != dp.world_size * get_tensor_model_parallel_world_size()
    ):
        return None
    sizes = get_forward_context().dp_metadata.get_chunk_sizes_across_dp_rank()
    if hidden_states.shape[0] != sum(sizes) or not ca.should_custom_ar(hidden_states):
        return None
    start = sum(sizes[: dp.rank_in_group])
    return start, start + sizes[dp.rank_in_group]


def _ep_fused_all_reduce(
    hidden_states: torch.Tensor,
    shared_output: torch.Tensor | None,
) -> torch.Tensor | None:
    """Combine + shared add + TP all-reduce as one EP custom all-reduce.

    With naive DP dispatch every EP rank holds routed partial sums for the
    tokens of all DP ranks, laid out by DP rank.  Reduce-scatter over DP
    followed by an all-reduce over TP equals an all-reduce over the whole EP
    group (DP x TP) sliced to this DP rank's rows.  Adding the TP-partial
    shared output into the own rows first folds the shared add into the same
    collective: one custom all-reduce instead of NCCL reduce-scatter + add +
    custom all-reduce.  Returns None when the custom all-reduce cannot take
    the tensor (e.g. large prefill chunks); ``hidden_states`` is then intact.
    ``shared_output=None`` means it was already added into the own rows.
    """
    rows = _ep_fused_rows(hidden_states)
    if rows is None:
        return None
    start, end = rows
    if shared_output is not None:
        hidden_states[start:end] += shared_output
    ca = get_ep_group().device_communicator.ca_comm
    return ca.custom_all_reduce(hidden_states)[start:end]


class _EpFusedCombineState:
    """Hands the shared-expert output of the running MoE layer to the naive
    DP/EP modular-kernel finalize, which does the combine there."""

    def __init__(self) -> None:
        self.shared: torch.Tensor | None = None
        self.done = False
        # (data_ptr, shape) of the expert output the shared output was
        # already added into by the EP fused experts (shared_add below).
        self.added_to: tuple[int, torch.Size] | None = None

    def arm(self, shared_experts: SharedExperts | None, shared_input) -> None:
        self.done = False
        self.shared = None
        self.added_to = None
        # Only when the shared expert already ran inline on this stream; an
        # aux-stream output is not ready yet at finalize time.
        if (
            shared_experts is not None
            and shared_experts._determine_shared_experts_order(shared_input)
            == SharedExpertsOrder.NO_OVERLAP
        ):
            # Peek without consuming; the runner still pops it as usual.
            self.shared = shared_experts._output[shared_experts._output_idx]

    def shared_add(self, out: torch.Tensor) -> tuple[torch.Tensor, int] | None:
        """For the EP fused experts: (shared_output, row_start) to add into
        the rows of ``out`` the finalize hook would add it to, else None.
        Only when the hook is then guaranteed to take the fused path."""
        if self.shared is None or self.added_to is not None:
            return None
        rows = _ep_fused_rows(out)
        if rows is None:
            return None
        self.added_to = (out.data_ptr(), out.shape)
        return self.shared, rows[0]

    def finalize_hook(self, out: torch.Tensor) -> torch.Tensor | None:
        shared, self.shared = self.shared, None
        if shared is None:
            return None
        if self.added_to is not None:
            assert self.added_to == (out.data_ptr(), out.shape)
            shared = None
        reduced = _ep_fused_all_reduce(out, shared)
        if reduced is None:
            assert shared is not None
            return None
        self.done = True
        # The finalize returns it as the MoE output: no copy into its buffer.
        return reduced

    def finish(self) -> bool:
        done, self.done, self.shared = self.done, False, None
        assert done or self.added_to is None
        self.added_to = None
        return done


_EP_COMBINE = _EpFusedCombineState()
if _EP_FUSED_COMBINE:
    from vllm_hcu.model_executor.layers.fused_moe.combine_runtime import (
        set_finalize_combine_hook,
    )

    # Spawn may import this runner while unpickling the quantization config,
    # before Worker callbacks attach finalize. Keep hook state plugin-owned.
    set_finalize_combine_hook(_EP_COMBINE.finalize_hook)
    from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe

    ep_fused_moe.shared_add_provider = _EP_COMBINE.shared_add


def register_layer_for_moe_forward_op(
    vllm_config: VllmConfig,
    layer: "MoERunner",
):
    # For smuggling this layer into the fused moe custom op
    prefix = layer.layer_name
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError("Duplicate layer name: {}".format(prefix))
    compilation_config.static_forward_context[prefix] = layer
    compilation_config.static_all_moe_layers.append(prefix)


def get_layer_from_name(layer_name: str) -> MoERunnerInterface:
    forward_context: ForwardContext = get_forward_context()
    if not _USE_LAYERNAME and layer_name == "from_forward_context":
        all_moe_layers = forward_context.all_moe_layers
        assert all_moe_layers is not None
        moe_layer_index = forward_context.moe_layer_index
        if moe_layer_index >= len(all_moe_layers):
            raise AssertionError(
                "We expected the number of MOE layers in `all_moe_layers` "
                "to be equal to the number of "
                "{vllm.moe_forward, vllm.moe_forward_shared} calls."
            )
        layer_name = all_moe_layers[moe_layer_index]
        forward_context.moe_layer_index += 1
    layer = forward_context.no_compile_layers[layer_name]
    assert isinstance(layer, MoERunnerInterface)
    return layer


# On torch >= 2.11, layer_name is a hoisted LayerName opaque object;
# on older versions it remains a plain str.
if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | LayerName
else:
    _layer_name_type = LayerName if _USE_LAYERNAME else str


@torch.compiler.assume_constant_result
def _resolve_layer_name(layer_name: str | LayerName) -> str:
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(layer_name, LayerName):
        return layer_name.value
    elif isinstance(layer_name, FakeScriptObject):
        return layer_name.real_obj.value
    return layer_name


# Note: _moe_forward and _moe_forward_shared should not contain any
# implementation details, They should merely pass along control to
# the runner's '_forward_impl' method.
# These functions should never be called directly since they do not
# include all the functionality of the MoE layer.
def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor | None,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    quanted_hidden_states: torch.Tensor | None,
    scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return layer._forward_impl(
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
        quanted_hidden_states=quanted_hidden_states,
        scale=scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor | None,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    quanted_hidden_states: torch.Tensor | None,
    scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    # `hidden_dim_unpadded > 0` only on the TRT-LLM MXFP4 path, where the
    # real kernel writes narrower than `hidden_states.shape[-1]`. Plumbed
    # as an op arg (not peeked from the layer registry) to keep the fake
    # a pure shape function of its inputs and preserve subgraph dedup.
    if hidden_dim_unpadded > 0:
        return hidden_states.new_empty((*hidden_states.shape[:-1], hidden_dim_unpadded))
    return torch.empty_like(hidden_states)


def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor | None,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    quanted_hidden_states: torch.Tensor | None,
    scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    shared_output, fused_output = layer._forward_impl(
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
        quanted_hidden_states=quanted_hidden_states,
        scale=scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    if layer._ep_fused_combine:
        # Already a fresh all-reduce output that aliases no input.
        return shared_output, fused_output
    # Custom-op outputs must not alias their inputs. Out-of-place W4A8 and
    # other kernels already own their output; clone only the aliasing paths.
    if torch._C._is_alias_of(fused_output, hidden_states):
        fused_output = fused_output.clone()
    return shared_output, fused_output


def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor | None,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    quanted_hidden_states: torch.Tensor | None,
    scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # `fused_out`: see `_moe_forward_fake` for hidden_dim_unpadded semantics.
    # `shared_out`: matches `shared_experts_input` if provided (latent MoE),
    # else `hidden_states`.
    if hidden_dim_unpadded > 0:
        fused_out = hidden_states.new_empty(
            (*hidden_states.shape[:-1], hidden_dim_unpadded)
        )
    else:
        fused_out = torch.empty_like(hidden_states)
    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


# NOTE: `moe_forward` and `moe_forward_shared` being opaque custom ops is a
# load-bearing assumption for the MoE-LoRA dual-stream path.
direct_register_custom_op(
    op_name="moe_forward",
    op_func=_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=_moe_forward_shared,
    mutates_args=["hidden_states"],
    fake_impl=_moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _unpack(
    result: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if isinstance(result, tuple):
        return result
    else:
        return (None, result)


class MoERunner(MoERunnerInterface):
    """
    Standard MoE runner implementation for executing Mixture of Experts layers.

    This is the primary concrete implementation of MoE execution logic, providing
    comprehensive support for standard MoE operations. It handles:
    - Expert routing and token dispatching using various routing strategies
    - Shared experts computation with optional parallel execution using CUDA streams
    - Tensor model parallel and expert parallel operations
    - Multiple quantization methods and optimized kernel selection
    - Both monolithic and decomposed expert execution paths
    - Integration with various parallel execution modes (TP, EP, DP)

    The runner orchestrates the complete MoE forward pass including routing tokens
    to experts, executing expert computations in parallel, and combining results.
    It supports advanced features like overlapped execution of shared experts,
    optimized kernels for different parallel configurations, and seamless
    integration with vLLM's distributed execution framework.

    Eventually, this class may be split into more specialized implementations
    for different configurations (e.g., with/without shared experts, gates, etc.).
    """

    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: RoutedExperts,
        enable_dbo: bool = False,
        gate: torch.nn.Module | None = None,
        shared_experts: torch.nn.Module | None = None,
        shared_expert_gate: torch.nn.Module | None = None,
        routed_input_transform: torch.nn.Module | None = None,
        routed_output_transform: torch.nn.Module | None = None,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.routed_output_transform = routed_output_transform
        self.routed_scaling_factor = routed_scaling_factor
        self.gate = gate
        self.shared_expert_gate = shared_expert_gate
        self.routed_experts = routed_experts
        self.enable_dbo = enable_dbo

        # When both gates are present and FSE is enabled, fuse their
        # weight matrices into [num_experts + num_shared, hidden] so one
        # F.linear produces combined logits. The topk kernel can then
        # apply routing softmax and shared expert activation (sigmoid)
        # in a single launch.
        self._fse_fuse_gate = gate is not None and shared_expert_gate is not None
        self._combined_gate_weight: torch.Tensor | None = None

        self._shared_experts: SharedExperts | None = None
        if shared_experts is not None:
            can_overlap = lambda: self._quant_method.mk_can_overlap_shared_experts
            # HCU uses ROCm stream semantics while advertising a CUDA-compatible
            # platform to vLLM. As in upstream ROCm, an unquantized routed input
            # aliases the shared-expert input and is not safe for multi-stream
            # overlap. Activation quantization copies it into a distinct buffer.
            routed_input_is_quantized = lambda: (
                self.routed_experts.quant_method.moe_quant_config is not None
                and self.routed_experts.quant_method.moe_quant_config.quant_dtype
                is not None
            )
            self._shared_experts = SharedExperts(
                shared_experts,
                moe_config=moe_config,
                enable_dbo=enable_dbo,
                mk_can_overlap_shared_experts=can_overlap,
                is_multistream_safe=routed_input_is_quantized,
            )

        # Static so the traced forward() branches on a plain bool; runtime
        # conditions (custom AR availability, size) are checked in the op.
        self._ep_fused_combine = (
            _EP_FUSED_COMBINE
            and shared_experts is not None
            and moe_config.dp_size > 1
            and moe_config.pcp_size == 1
            and not moe_config.is_sequence_parallel
            and not moe_config.skip_final_all_reduce
            and routed_scaling_factor == 1.0
            and routed_input_transform is None
            and routed_output_transform is None
        )

        # Needed for string -> MoERunner layer lookup in custom ops.
        self.layer_name = layer_name

        self._forward_entry = self._select_forward()

        # For smuggling this layer into the fused moe custom op
        register_layer_for_moe_forward_op(get_current_vllm_config(), self)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        return self.routed_experts.load_weights(weights)

    def _select_forward(self) -> Callable:
        if current_platform.is_tpu() or current_platform.is_cpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            # Note: CPU doesn't require wrapped _forward_impl.
            return _moe_forward if self._shared_experts is None else _moe_forward_shared

        return (
            torch.ops.vllm.moe_forward
            if self._shared_experts is None
            else torch.ops.vllm.moe_forward_shared
        )

    @property
    def shared_experts(self) -> SharedExperts | None:
        return self._shared_experts

    @property
    def is_internal_router(self) -> bool:
        return self.gate is not None

    # TODO(bnell): Temporary hack. Get rid of this.
    def _replace_quant_method(self, quant_method: FusedMoEMethodBase):
        self.routed_experts._replace_quant_method(quant_method)
        self.__dict__.pop("_supports_quanted_inputs", None)

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config
        self.routed_experts._set_moe_config(new_moe_config)
        if self._shared_experts is not None:
            self._shared_experts._set_moe_config(new_moe_config)

    def _maybe_fuse_gate_weights(self):
        """Fuse router and shared expert gate weights on first call.

        Cannot be done at __init__ because gate weights are loaded after
        module construction (via weight_loader). Called once from
        _forward_impl before the first forward pass.
        """
        if self._combined_gate_weight is None:
            assert self.gate is not None and self.shared_expert_gate is not None
            self._combined_gate_weight = torch.cat(
                [self.gate.weight, self.shared_expert_gate.weight],
                dim=0,
            )

    @property
    def _quant_method(self) -> FusedMoEMethodBase:
        return self.routed_experts.quant_method

    def apply_routed_input_transform(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply transform for routed experts (e.g., latent projection).

        This is called by FusedMoE.forward_native. The original hidden_states
        is saved separately so shared experts get [S, hidden_size] while
        routed experts get the transformed [S, moe_latent_size].

        Returns (possibly transformed) hidden states and the input for shared
        experts (or None if there are no shared experts).
        """
        if self.routed_input_transform is not None:
            result = self.routed_input_transform(hidden_states)
            # ReplicatedLinear returns (output, extra_bias) tuple.
            # We only need the output tensor; extra_bias is not used here.
            if isinstance(result, tuple):
                return result[0], hidden_states
            return result, hidden_states

        return (
            hidden_states,
            hidden_states if self._shared_experts is not None else None,
        )

    def apply_routed_output_transform(
        self,
        fused_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transform to routed expert output (e.g., latent to full dim).

        Used by latent MoE models (e.g., NemotronH) where routed experts
        operate in a compressed latent space and need projection back to
        the full hidden dimension before combining with shared expert output.
        """
        if self.routed_output_transform is not None:
            r = self.routed_output_transform(fused_output)
            fused_output = r[0] if isinstance(r, tuple) else r
        return fused_output

    def _maybe_apply_routed_scale_to_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Apply routed_scaling_factor to the output with FP16 overflow
        protection.

        Scale the fused expert output by routed_scaling_factor. For FP16,
        avoid overflow by dividing shared_output by the scale instead
        (the decoder layer compensates with matching divisions).
        """
        if self.routed_scaling_factor != 1.0:
            if fused_output.dtype != torch.float16 or shared_output is None:
                fused_output *= self.routed_scaling_factor
            elif shared_output is not None:
                shared_output *= 1.0 / self.routed_scaling_factor
        return shared_output, fused_output

    @property
    def _fused_output_is_reduced(self) -> bool:
        return (
            self._quant_method.moe_kernel is not None
            and self._quant_method.moe_kernel.output_is_reduced()
        )

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output_is_reduced: bool | None = None,
    ) -> torch.Tensor | None:
        """All-reduce shared expert output when the combine kernel already
        reduced fused output.

        * If the combine kernel does the reduction for fused_output, reduce
          shared_output separately. O.w, reduce fused_output+shared_output later.
        * If we have SP (TP=N, DP=M, EP), there is a separate AG step handled
          in the model.
        """
        if fused_output_is_reduced is None:
            fused_output_is_reduced = self._fused_output_is_reduced

        if (
            shared_output is not None
            and not self.moe_config.is_sequence_parallel
            and fused_output_is_reduced
        ):
            shared_output = tensor_model_parallel_all_reduce(shared_output)
        return shared_output

    def _maybe_reduce_routed_output_before_transform(
        self,
        fused_output: torch.Tensor,
        fused_output_is_reduced: bool,
    ) -> tuple[torch.Tensor, bool]:
        """Reduce latent routed output before its nonlinear output transform."""
        if (
            self.routed_output_transform is not None
            and not self.moe_config.is_sequence_parallel
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not fused_output_is_reduced
        ):
            fused_output = tensor_model_parallel_all_reduce(fused_output)
            fused_output_is_reduced = True
        return fused_output, fused_output_is_reduced

    def _can_fuse_kimi_k3_latent_all_reduce(
        self,
        shared_output: torch.Tensor | None,
        fused_output: torch.Tensor,
        fused_output_is_reduced: bool,
    ) -> bool:
        """Fuse eligible Kimi-K3 partials by default; the env flag can opt out."""
        fusion_enabled = os.environ.get(
            "VLLM_HCU_KIMI_LATENT_MOE_FUSE_ALLREDUCE",
            "1",
        ).lower() in ("true", "1")
        return (
            fusion_enabled
            and getattr(
                self.routed_output_transform,
                "_hcu_fuse_shared_and_routed_tp_all_reduce",
                False,
            )
            and shared_output is not None
            and self._shared_experts is not None
            and self.moe_config.tp_size > 1
            and not self.moe_config.is_sequence_parallel
            and not fused_output_is_reduced
            and fused_output.dtype == shared_output.dtype
            and fused_output.shape[:-1] == shared_output.shape[:-1]
        )

    def _fuse_kimi_k3_latent_all_reduce(
        self,
        shared_output: torch.Tensor,
        fused_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce Kimi-K3 latent and shared partials in one TP collective."""
        latent_width = fused_output.shape[-1]
        combined_output = torch.cat((fused_output, shared_output), dim=-1)
        combined_output = tensor_model_parallel_all_reduce(combined_output)
        return (
            combined_output[..., :latent_width],
            combined_output[..., latent_width:],
        )

    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int | None,
        output_is_reduced: bool | None = None,
    ) -> torch.Tensor:
        """All-reduce the combined output if needed.

        This is the "late" all-reduce path. When neither fused nor shared
        output was individually reduced, the combined sum is all-reduced
        here. Skipped when sequence-parallel is active (SP handles its
        own reduction) or when the early path already reduced both outputs.
        """
        # skip_final_all_reduce must not coexist with a pre-reduced fused
        # output. This should be enforced by MoE config initialization.
        is_reduced = (
            self._fused_output_is_reduced
            if output_is_reduced is None
            else output_is_reduced
        )
        if self.moe_config.skip_final_all_reduce:
            assert not is_reduced, (
                "skip_final_all_reduce requires an un-reduced fused output"
            )

        # We don't need to reduce the final output if:
        # - We are not running with TP or DP
        # - The MK already reduced the fused output itself.
        if (
            not self.moe_config.is_sequence_parallel
            and not self.moe_config.skip_final_all_reduce
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not is_reduced
        ):
            states = tensor_model_parallel_all_reduce(states)

        return states[..., :trunc_size] if trunc_size is not None else states

    def _encode_layer_name(self) -> str | LayerName:
        if _USE_LAYERNAME:
            return LayerName(self.layer_name)
        # Can be unavailable or None in unittests
        if (
            is_forward_context_available()
            and get_forward_context().all_moe_layers is not None
        ):
            return "from_forward_context"
        return self.layer_name

    def _maybe_pad_hidden_states(
        self,
        shared_experts_input: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, int | None, int | None]:
        """Pad hidden_states to moe_config.hidden_dim and compute the
        original dimension for later truncation.

        For latent MoE, the routed hidden_states may be smaller than
        hidden_dim. Padding ensures uniform tensor sizes through the
        fused MoE kernel. The returned trunc_size is used by
        _maybe_reduce_final_output to strip the padding from the result.
        """
        shared_experts_hidden_dim = (
            shared_experts_input.shape[-1] if shared_experts_input is not None else 0
        )
        transformed_hidden_dim: int | None = hidden_states.shape[-1]
        if (
            not self._quant_method.skip_forward_padding
            and self.moe_config.hidden_dim != transformed_hidden_dim
        ):
            assert transformed_hidden_dim is not None
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        # Truncation sizes for stripping kernel padding from the output.
        # None means no truncation needed (no padding was applied).
        #
        # Two truncation points exist in forward():
        #   pre_xform:  applied to fused_output BEFORE routed_output_transform
        #   post_xform: applied to the final result AFTER all-reduce
        #
        # MoE with routed output transform or shared experts:
        #   - pre_xform applies if the transform needs unpadded routed output
        #     or shared+routed add needs matching hidden dims. For Nemotron-3
        #     Nano, TRTLLM NVFP4 pads routed MoE hidden dim 2688->2816, while
        #     shared output stays 2688.
        #   - post_xform uses shared_experts_hidden_dim when transform and shared
        #     experts make the final output full hidden dim.
        #
        # Standard MoE / MoE without transforms (GPT-OSS, Mixtral):
        #   - pre_xform is None (no early truncation)
        #   - post_xform strips padding after all-reduce (or None if unpadded)
        if transformed_hidden_dim == hidden_states.shape[-1]:
            transformed_hidden_dim = None

        pre_xform_trunc_size = None
        if self.routed_output_transform is not None or shared_experts_hidden_dim > 0:
            pre_xform_trunc_size = transformed_hidden_dim
        post_xform_trunc_size = transformed_hidden_dim
        if self.routed_output_transform is not None and shared_experts_hidden_dim > 0:
            post_xform_trunc_size = shared_experts_hidden_dim

        return hidden_states, pre_xform_trunc_size, post_xform_trunc_size

    def _maybe_apply_shared_experts(
        self,
        shared_experts_input: torch.Tensor | None,
        order: SharedExpertsOrder,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts(
                shared_experts_input,
                order,
                x_and_scale_quanted=x_and_scale_quanted,
            )

    def _quant_method_supports_quanted_inputs(self) -> bool:
        cached = getattr(self, "_supports_quanted_inputs", None)
        if cached is not None:
            return cached
        apply = getattr(self._quant_method, "apply", None)
        if apply is None:
            self._supports_quanted_inputs = False
            return False
        try:
            parameters = inspect.signature(apply).parameters
        except (TypeError, ValueError):
            self._supports_quanted_inputs = False
            return False
        self._supports_quanted_inputs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ) or ("i_q" in parameters and "i_s" in parameters)
        return self._supports_quanted_inputs

    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        quanted_hidden_states: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run expert routing and the fused MoE kernel via the quant method.

        Orchestrates shared expert execution (before/after), expert selection
        via the router, and the actual fused MoE computation. Returns
        (shared_expert_output, fused_expert_output).
        """
        if (topk_weights is None) != (topk_ids is None):
            raise ValueError(
                "HCU preselected MoE routing requires topk_weights and topk_ids "
                "together"
            )
        if quanted_hidden_states is not None and scale is None:
            raise ValueError(
                "HCU pre-quantized MoE input requires quanted_hidden_states and scale"
            )
        if self._quant_method.is_monolithic and (
            quanted_hidden_states is not None or topk_weights is not None
        ):
            raise RuntimeError(
                "HCU pre-quantized/preselected MoE input is not supported by "
                f"monolithic quant method {type(self._quant_method).__name__}"
            )

        shared_experts_x_and_scale_quanted = (
            (quanted_hidden_states, scale)
            if quanted_hidden_states is not None and scale is not None
            else None
        )
        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.NO_OVERLAP,
            x_and_scale_quanted=shared_experts_x_and_scale_quanted,
        )
        if self._ep_fused_combine:
            _EP_COMBINE.arm(self._shared_experts, shared_experts_input)

        if self.routed_experts.quant_method.is_monolithic:
            # Monolithic kernels: pass router_logits to routed_experts
            fused_out = self.routed_experts.forward_monolithic(
                x=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        else:
            # Modular kernels: select experts first, then call routed_experts
            if topk_weights is None or topk_ids is None:
                assert router_logits is not None
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    topk_indices_dtype=self._quant_method.topk_indices_dtype,
                    input_ids=input_ids,
                )

            if quanted_hidden_states is None:
                fused_out = self.routed_experts.forward_modular(
                    x=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    shared_experts=self._shared_experts,
                    shared_experts_input=shared_experts_input,
                )
            else:
                if not self._quant_method_supports_quanted_inputs():
                    raise RuntimeError(
                        "HCU pre-quantized MoE input was enabled, but quant method "
                        f"{type(self._quant_method).__name__}.apply does not accept "
                        "i_q/i_s"
                    )
                fused_out = self._quant_method.apply(
                    layer=self.routed_experts,
                    x=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    shared_experts=self._shared_experts,
                    shared_experts_input=shared_experts_input,
                    i_q=quanted_hidden_states,
                    i_s=scale,
                )

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
            x_and_scale_quanted=shared_experts_x_and_scale_quanted,
        )

        return (
            self._shared_experts.output if self._shared_experts is not None else None,
            fused_out,
        )

    def _sequence_parallel_context(self):
        """Return a context manager for sequence-parallel token
        redistribution.

        When sequence parallelism is active, returns a context that handles
        local size tracking for proper token scatter/gather. Otherwise
        returns a no-op context.
        """
        ctx = get_forward_context()
        return (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

    def _maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor | None,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts.maybe_sync_shared_experts_stream(
                shared_experts_input,
                x_and_scale_quanted=x_and_scale_quanted,
            )

    def _maybe_add_zero_expert_output(
        self,
        result: torch.Tensor,
    ) -> torch.Tensor:
        """Add the zero expert's contribution to the final result.

        When a ZeroExpertRouter is used, it computes a bias-like output
        from the "zero expert" that is added to the combined routed+shared
        expert output.
        """
        if isinstance(self.router, ZeroExpertRouter):
            zero_expert_output = self.router.zero_expert_output
            assert zero_expert_output is not None
            result = result + zero_expert_output
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        quanted_hidden_states: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Invoke the fused moe layer.

        Input:
        - hidden_states
        - router_logits

        Output:
        - The new hidden_states.

        Calling sequence
        - forward
          - self._forward_entry (_moe_forward or _moe_forward_shared custom op)
            - _forward_impl

        Note: The existence of _moe_forward and _moe_forward_shared custom ops are due
        to the following reason:
        1. pytorch cannot handle union types in custom op signatures so
           _moe_forward and _moe_forward_shared must be split.
        """

        if (topk_weights is None) != (topk_ids is None):
            raise ValueError(
                "HCU preselected MoE routing requires topk_weights and topk_ids "
                "together"
            )
        if quanted_hidden_states is not None and scale is None:
            raise ValueError("HCU pre-quantized MoE input requires scale")

        # Apply transform for routed experts (e.g., latent projection
        # for latent MoE)
        hidden_states, shared_experts_input = self.apply_routed_input_transform(
            hidden_states
        )

        # Record before `_maybe_pad_hidden_states` pads activations to match
        # `moe_config.hidden_dim`, e.g. after `align_trtllm_fp4_moe_hidden_dim_for_fi`
        # so routed output can be trimmed before
        # shared+routed add / latent up proj if needed.

        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
                hidden_states,
            )
        )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            quanted_hidden_states,
            scale,
            topk_weights,
            topk_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0,
        )

        #
        # Note: there are two all-reduce points below. They are mutually
        # exclusive, controlled by _fused_output_is_reduced
        #  - When True: the combine kernel already reduced fused_output,
        #    so we reduce shared_output here to match, then skip the
        #    all-reduce in _maybe_reduce_final_output.
        #  - When False: neither output is reduced yet, so we combine
        #    them first and all-reduce the sum in _maybe_reduce_final_output.

        shared_output, fused_output = _unpack(result)

        if self._ep_fused_combine:
            # The op already combined, added shared output and all-reduced.
            assert og_hidden_dim_pre_xform is None
            assert og_hidden_dim_post_xform is None
            return self._maybe_add_zero_expert_output(fused_output)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        fused_output_is_reduced = self._fused_output_is_reduced
        if self._can_fuse_kimi_k3_latent_all_reduce(
            shared_output,
            fused_output,
            fused_output_is_reduced,
        ):
            assert shared_output is not None
            fused_output, shared_output = self._fuse_kimi_k3_latent_all_reduce(
                shared_output,
                fused_output,
            )
            fused_output_is_reduced = True
        elif getattr(
            self.routed_output_transform,
            "_hcu_fuse_shared_and_routed_tp_all_reduce",
            False,
        ):
            fused_output, fused_output_is_reduced = (
                self._maybe_reduce_routed_output_before_transform(
                    fused_output,
                    fused_output_is_reduced,
                )
            )

            # If routed output is reduced before its nonlinear transform,
            # reduce shared output separately so both branches share a TP domain.
            shared_output = self._maybe_reduce_shared_expert_output(
                shared_output,
                fused_output_is_reduced,
            )
        else:
            shared_output = self._maybe_reduce_shared_expert_output(shared_output)

        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
            shared_output, fused_output
        )

        # Apply output transform (e.g. latent -> full dim)
        fused_output = self.apply_routed_output_transform(fused_output)

        if shared_output is not None:
            result = shared_output + fused_output
        else:
            result = fused_output

        result = self._maybe_reduce_final_output(
            result, og_hidden_dim_post_xform, fused_output_is_reduced
        )

        return self._maybe_add_zero_expert_output(result)

    @property
    def do_naive_dispatch_combine(self) -> bool:
        return (
            self.moe_config.dp_size > 1 or self.moe_config.is_sequence_parallel
        ) and not self._quant_method.supports_internal_mk

    def _maybe_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # For naive dispatch/combine Dp/Ep, dispatch the hidden states and
        # router logits to all experts.
        # NOTE: this will be removed once all kernels are migrated into the
        # MoEKernel framework.
        if router_logits is None and (
            self.do_naive_dispatch_combine or self.moe_config.pcp_size > 1
        ):
            raise RuntimeError(
                "HCU preselected routing without router_logits is incompatible "
                "with naive DP/EP or PCP dispatch"
            )
        if self.do_naive_dispatch_combine:
            result = get_ep_group().dispatch_router_logits(
                hidden_states,
                router_logits,
                self.moe_config.is_sequence_parallel,
            )
            assert len(result) == 2
            hidden_states, router_logits = result

        # NOTE: Similar with DP, PCP also needs dispatch and combine. For
        # simplicity, AgRsAll2All was added separately for PCP here. Maybe
        # we should modify All2AllManager abstraction to better support PCP.
        needs_fallback_pcp_collective = (
            self.moe_config.pcp_size > 1
            and not self.moe_config.moe_parallel_config.use_all2all_kernels
        )
        if needs_fallback_pcp_collective:
            hidden_states = get_pcp_group().all_gather(
                hidden_states,
                dim=0,
            )
            router_logits = get_pcp_group().all_gather(
                router_logits,
                dim=0,
            )

        return hidden_states, router_logits

    @contextmanager
    def _pcp_routing_inputs(self, input_ids, num_local_tokens):
        """Match routing auxiliaries to the fallback PCP dispatch ordering."""
        if not (
            self.moe_config.pcp_size > 1
            and not self.moe_config.moe_parallel_config.use_all2all_kernels
        ):
            yield input_ids
            return

        context = get_forward_context() if is_forward_context_available() else None
        local_padding = context.is_padding if context is not None else None
        if input_ids is not None:
            input_ids = get_pcp_group().all_gather(
                input_ids[:num_local_tokens], dim=0
            )
        if local_padding is not None:
            context.is_padding = get_pcp_group().all_gather(
                local_padding[:num_local_tokens], dim=0
            )
        try:
            yield input_ids
        finally:
            if context is not None:
                context.is_padding = local_padding

    def _ep_fused_combine_output(
        self,
        shared_output: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Final MoE output when the finalize hook did not already fuse it."""
        if self.do_naive_dispatch_combine:
            if not self._fused_output_is_reduced:
                reduced = _ep_fused_all_reduce(hidden_states, shared_output)
                if reduced is not None:
                    return reduced
            hidden_states = get_ep_group().combine(hidden_states, False)
        # Unfused tail, identical to forward() for this configuration.
        shared_output = self._maybe_reduce_shared_expert_output(shared_output)
        return self._maybe_reduce_final_output(shared_output + hidden_states, None)

    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor]:
        if self._ep_fused_combine:
            assert shared_output is not None
            if _EP_COMBINE.finish():
                # MK finalize already produced the final output.
                return shared_output, hidden_states
            return shared_output, self._ep_fused_combine_output(
                shared_output, hidden_states
            )

        if self.do_naive_dispatch_combine:
            hidden_states = get_ep_group().combine(
                hidden_states, self.moe_config.is_sequence_parallel
            )

        needs_fallback_pcp_collective = (
            self.moe_config.pcp_size > 1
            and not self.moe_config.moe_parallel_config.use_all2all_kernels
        )
        if needs_fallback_pcp_collective:
            hidden_states = get_pcp_group().reduce_scatter(
                hidden_states,
                dim=0,
            )

        if self.shared_experts is not None:
            assert shared_output is not None
            return shared_output, hidden_states
        else:
            return hidden_states

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        quanted_hidden_states: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Entry point called by the custom op to run the MoE computation.

        Handles pre-dispatch setup (gate application, external shared expert
        triggering, quant config init) then performs the following steps
        within the sequence-parallel context.

        - Performs expert routing
        - fused MoE kernel execution
        - shared expert computation.

        Returns a single tensor of combined fused and shared output (if present).
        """
        if (quanted_hidden_states is not None or topk_weights is not None) and (
            self.do_naive_dispatch_combine or self.moe_config.pcp_size > 1
        ):
            raise RuntimeError(
                "HCU pre-quantized/preselected MoE inputs cannot be combined "
                "with naive DP/EP or PCP dispatch because those auxiliary "
                "tensors are already local to the caller"
            )

        # TODO(bnell): this can be removed after MK migration is complete.
        self.routed_experts._ensure_moe_quant_config_init()

        # Sync aux and main stream for shared expert multi-stream overlap.
        shared_experts_x_and_scale_quanted = (
            (quanted_hidden_states, scale)
            if quanted_hidden_states is not None and scale is not None
            else None
        )
        self._maybe_sync_shared_experts_stream(
            shared_experts_input,
            x_and_scale_quanted=shared_experts_x_and_scale_quanted,
        )

        # If the Runner holds the gate, apply it after the stream sync,
        # so it can run overlapped with the
        # NOTE: in future PR, MoE runner will always hold the gate.
        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = F.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)

        with self._sequence_parallel_context():
            # TODO(bnell): parts of the dispatch/combine steps will go away once
            # #32567 lands and the remaining kernels are made MKs.  The PCP
            # code will probably remain
            num_local_tokens = hidden_states.shape[0]
            hidden_states, router_logits = self._maybe_dispatch(
                hidden_states,
                router_logits,
            )

            with self._pcp_routing_inputs(
                input_ids, num_local_tokens
            ) as routing_input_ids:
                shared_output, hidden_states = self._apply_quant_method(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    shared_experts_input=shared_experts_input,
                    input_ids=routing_input_ids,
                    quanted_hidden_states=quanted_hidden_states,
                    scale=scale,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                )
            return self._maybe_combine(
                shared_output,
                hidden_states,
            )

    #########################################################
    #
    # Old methods from FusedMoE layer. Remove when possible.
    #
    #########################################################

    # Note: maybe_init_modular_kernel should only be called by
    # prepare_communication_buffer_for_model.
    # This is called after all weight loading and post-processing, so it
    # should be safe to swap out the quant_method.
    def maybe_init_modular_kernel(self) -> None:
        # NOTE(rob): WIP refactor. For quant methods that own the MK
        # we create the MK during process_weights_after_loading.
        if (
            self.routed_experts.quant_method.supports_internal_mk
            or self.routed_experts.quant_method.is_monolithic
        ):
            return None

        self.routed_experts._ensure_moe_quant_config_init()
        # routing_tables only needed for round-robin expert placement with
        # DeepEP all2all backend.
        routing_tables = self._expert_routing_tables()

        if isinstance(self.routed_experts.quant_method, FusedMoEModularMethod):
            base_quant_method = self.routed_experts.quant_method.old_quant_method
        else:
            base_quant_method = self.routed_experts.quant_method

        prepare_finalize = base_quant_method.maybe_make_prepare_finalize(
            routing_tables=routing_tables
        )
        if prepare_finalize is not None:
            logger.debug(
                "%s for %s(%s)", prepare_finalize.__class__.__name__, self, id(self)
            )
            self._replace_quant_method(
                FusedMoEModularMethod.make(
                    self.routed_experts,
                    base_quant_method,
                    prepare_finalize,
                )
            )

    #
    # Properties
    #

    @property
    def layer_id(self):
        # Delayed import to avoid circular dependency
        from vllm.model_executor.models.utils import extract_layer_index

        return extract_layer_index(self.layer_name)

    #
    # Attributes still needed by models
    #

    @property
    def is_monolithic(self) -> bool:
        return self.routed_experts.quant_method.is_monolithic

    @property
    def activation(self) -> MoEActivation:
        return self.routed_experts.activation

    #
    # Expert maps
    #

    @property
    def expert_map_manager(self):
        """Forward to routed_experts.expert_map_manager for backward compatibility."""
        return self.routed_experts.expert_map_manager

    @property
    def expert_placement_strategy(self) -> ExpertPlacementStrategy:
        return self.expert_map_manager.placement_strategy

    @property
    def expert_global_to_physical(self) -> torch.Tensor | None:
        tables = self.expert_map_manager.routing_tables
        return tables[0] if tables else None

    @property
    def expert_physical_to_global(self) -> torch.Tensor | None:
        """Routing table: physical expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[1] if tables else None

    @property
    def expert_local_to_global(self) -> torch.Tensor | None:
        """Routing table: local expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[2] if tables else None

    @property
    def expert_map(self) -> torch.Tensor | None:
        return self.routed_experts.expert_map

    def _expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return self.routed_experts._expert_routing_tables()

    def update_expert_map(self):
        self.routed_experts.update_expert_map()

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        """Map global expert ID to local expert ID."""
        return self.routed_experts._map_global_expert_id_to_local_expert_id(expert_id)

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        return self.routed_experts.get_expert_weights()

    #
    # EPLB
    #

    @property
    def eplb_state(self) -> EplbLayerState | None:
        return self.router.eplb_state

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """
        Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        if self.router.eplb_state is not None:
            self.router.eplb_state.set_layer_state(
                moe_layer_idx,
                expert_load_view,
                logical_to_physical_map,
                logical_replica_count,
            )
