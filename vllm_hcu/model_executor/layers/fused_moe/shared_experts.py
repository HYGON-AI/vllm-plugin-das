# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""HCU-owned v0.25.1 shared-expert execution."""

import inspect
import os
from collections.abc import Callable
from enum import IntEnum

import torch

import vllm.envs as envs
import vllm_hcu.platforms.envs as henvs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
)

logger = init_logger("vllm.hcu.shared_experts")

# Batches of at least this many tokens (prefill) run the shared expert on the
# aux stream launched after the routed experts, so it overlaps the DP combine
# (NCCL reduce-scatter); the runner joins it right before the shared add (see
# wait_late).  Launched any earlier it only competes with compute kernels for
# CUs.  Smaller batches (decode) keep the configured order.  0 disables.  Not a
# VLLM_HCU_* name on purpose: it only changes eager stream use inside the MoE
# custom op, not the traced graph, so it must not change the compile hash.
_LATE_STREAM_MIN_TOKENS = (
    int(os.environ.get("HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS", "0"))
    if henvs.custom_ops_enabled() else 0
)
_fused_router_stream_announced = False


def _fused_router_stream_enabled() -> bool:
    # Worker unpickling can import this module before the executor installs
    # its environment. Resolve at use time so an early import cannot freeze
    # the feature off for the lifetime of that worker.
    return henvs.optional_custom_op_enabled(
        os.environ.get("HCU_STEP5_FUSED_ROUTER_STREAM", "0") == "1"
    )


class SharedExpertsOrder(IntEnum):
    # No shared experts.
    NONE = (0,)

    # No overlap - defensively called before MK.
    NO_OVERLAP = (1,)

    # Overlapped with dispatch/combine in DP/EP - called by the MK.
    MK_INTERNAL_OVERLAPPED = (2,)

    # Overlapped with the gate, router, experts in aux stream.
    MULTI_STREAM_OVERLAPPED = (3,)

    # The router has produced an independent shared activation. Only its down
    # projection runs on the aux stream, starting after the producer event.
    FUSED_ROUTER_OVERLAPPED = (4,)


class SharedExperts(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        enable_dbo: bool,
        mk_can_overlap_shared_experts: Callable[[], bool],
        is_multistream_safe: Callable[[], bool],
    ):
        super().__init__()

        # The SharedExperts need to handle DBO since they can be called from
        # an MK's finalize method.  We keep a list of outputs indexed by current
        # DBO ubatch id to handle this case.  If DBO is not enabled, the
        # index is always 0 and the second output list element is ignored.
        from vllm_hcu.runtime_compat.step5 import is_step5_owner

        self._hcu_step5_owner = is_step5_owner(moe_config)
        self.enable_dbo = enable_dbo
        self._output: list[torch.Tensor | None] = [None, None]
        self._output_pending_on_stream: list[bool] = [False, False]
        self._output_pending_late: list[bool] = [False, False]
        self._output_pending_fused_router: list[bool] = [False, False]
        self._layer = layer
        self._moe_config = moe_config

        self._mk_can_overlap_shared_experts = mk_can_overlap_shared_experts
        self._is_multistream_safe = is_multistream_safe

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream")
            self._stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self._stream = aux_stream()
            if self._stream is not None:
                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self._moe_config = new_moe_config

    def _layer_supports_x_and_scale_quanted(self) -> bool:
        cached = getattr(self, "_supports_x_and_scale_quanted", None)
        if cached is not None:
            return cached
        hcu_forward = getattr(self._layer, "_forward_with_hcu_quanted", None)
        if callable(hcu_forward):
            self._supports_x_and_scale_quanted = True
            return True
        forward = getattr(self._layer, "forward", None)
        if forward is None:
            self._supports_x_and_scale_quanted = False
            return False
        try:
            self._supports_x_and_scale_quanted = (
                "x_and_scale_quanted" in inspect.signature(forward).parameters
            )
        except (TypeError, ValueError):
            self._supports_x_and_scale_quanted = False
        return self._supports_x_and_scale_quanted

    def _run_layer(
        self,
        shared_experts_input: torch.Tensor,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if x_and_scale_quanted is not None:
            if not self._layer_supports_x_and_scale_quanted():
                raise RuntimeError(
                    "HCU pre-quantized shared-expert input was enabled, but "
                    f"{type(self._layer).__name__}.forward does not accept "
                    "x_and_scale_quanted"
                )
            hcu_forward = getattr(
                self._layer, "_forward_with_hcu_quanted", None
            )
            if callable(hcu_forward):
                return hcu_forward(shared_experts_input, x_and_scale_quanted)
            return self._layer(
                shared_experts_input,
                x_and_scale_quanted=x_and_scale_quanted,
            )
        return self._layer(shared_experts_input)

    def _can_overlap_deepep_eplb(self) -> bool:
        """Shared weights are independent of routed-expert EPLB moves.

        Limit automatic overlap to quantized DeepEP inputs, an auxiliary
        stream and the non-DBO schedule. The routed computation must not
        overwrite the raw shared input; keep the existing safety callback.
        """
        parallel = self._moe_config.moe_parallel_config
        return bool(
            henvs.optional_custom_op_enabled(
                henvs.VLLM_HCU_SHARED_EXPERTS_DEEPEP_EPLB_OVERLAP
            )
            and parallel.enable_eplb
            and parallel.all2all_backend in ("deepep_auto", "deepep_high_throughput")
            and not parallel.use_fi_nvl_two_sided_kernels
            and not self.enable_dbo
            and self._stream is not None
            and self._is_multistream_safe()
        )

    @property
    def _disable_shared_experts_overlap(self) -> bool:
        if (
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE
        ):
            return False
        if self._can_overlap_deepep_eplb():
            return False
        # Disable shared expert overlap if:
        #   - we are using eplb with non-default backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothing to gain
        parallel_config = self._moe_config.moe_parallel_config
        return (
            parallel_config.enable_eplb
            and parallel_config.all2all_backend != "allgather_reducescatter"
        ) or parallel_config.use_fi_nvl_two_sided_kernels

    def _requires_main_stream_for_fused_activation(
        self,
        hidden_states: torch.Tensor,
    ) -> bool:
        """Return whether a producer left an activation on the main stream."""
        layer = getattr(self, "_layer", None)
        requires_main_stream = getattr(
            layer,
            "requires_main_stream_for_fused_activation",
            None,
        )
        return bool(requires_main_stream(hidden_states)) if callable(
            requires_main_stream
        ) else False

    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> SharedExpertsOrder:
        pending = getattr(self, "_output_pending_fused_router", None)
        if pending is not None and pending[self._output_idx]:
            return SharedExpertsOrder.FUSED_ROUTER_OVERLAPPED
        # A fused router can produce the shared gate_up + activation on the
        # current stream before the runner invokes this layer. Running the
        # shared expert on the auxiliary stream would fail the stream handoff
        # check and recompute both operations, so serialize this short path.
        if self._requires_main_stream_for_fused_activation(hidden_states):
            return SharedExpertsOrder.NO_OVERLAP

        force_stream = bool(
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and (
                henvs.VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH
                or henvs.VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE
            )
        )
        if force_stream and self._stream is None:
            raise RuntimeError(
                "HCU shared-expert stream mode was explicitly enabled, but no "
                "CUDA-alike auxiliary stream is available"
            )
        if self._disable_shared_experts_overlap:
            return SharedExpertsOrder.NO_OVERLAP
        if self._late_stream(hidden_states):
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED

        should_run_shared_in_aux_stream = self._should_run_shared_in_aux_stream(
            hidden_states
        )
        if (
            should_run_shared_in_aux_stream
            and self._is_multistream_safe()
            and (force_stream or self._can_overlap_deepep_eplb())
        ):
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED

        if self._mk_can_overlap_shared_experts():
            return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED

        if should_run_shared_in_aux_stream and self._is_multistream_safe():
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        return SharedExpertsOrder.NO_OVERLAP

    def maybe_launch_fused_router_activation(
        self, hidden_states: torch.Tensor
    ) -> bool:
        """Launch only after the internal router has handed off its activation.

        Take the cache on its producer stream, then record the dependency and
        allocator ownership explicitly. The auxiliary projection never reads
        hidden_states, which routed experts are allowed to overwrite.
        """
        if (
            not getattr(self, "_hcu_step5_owner", False)
            or not _fused_router_stream_enabled()
            or self.enable_dbo
            or self._stream is None
            or self._disable_shared_experts_overlap
            or getattr(self._moe_config, "is_sequence_parallel", False)
            or getattr(self._moe_config, "pcp_size", 1) > 1
            or not self._requires_main_stream_for_fused_activation(hidden_states)
        ):
            return False
        take = getattr(self._layer, "pop_fused_router_activation", None)
        project = getattr(self._layer, "forward_fused_router_activation", None)
        if not callable(take) or not callable(project):
            return False
        index = self._output_idx
        assert self._output[index] is None
        activation = take(hidden_states)
        if activation is None:
            return False
        assert activation.device == hidden_states.device
        assert activation.shape[0] == hidden_states.shape[0]
        self._stream.wait_stream(current_stream())
        activation.record_stream(self._stream)
        with torch.cuda.stream(self._stream):
            self._output[index] = project(activation)
        self._output_pending_on_stream[index] = True
        self._output_pending_fused_router[index] = True
        # HCU replacement modules retain a vllm_hcu logger name, which can be
        # filtered independently of vllm's INFO configuration. Match the other
        # Step5 native activation markers and emit once after a real launch.
        global _fused_router_stream_announced
        if not _fused_router_stream_announced:
            _fused_router_stream_announced = True
            print(
                "Step5 fused-router shared down projection: auxiliary stream active",
                flush=True,
            )
        return True

    def requires_input_preservation(self, hidden_states: torch.Tensor) -> bool:
        """Return whether routed experts can mutate this input concurrently."""
        return self._determine_shared_experts_order(hidden_states) in (
            SharedExpertsOrder.MK_INTERNAL_OVERLAPPED,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        )

    def allows_inplace_routed_output(
        self,
        routed_input: torch.Tensor,
        shared_input: torch.Tensor,
    ) -> bool:
        """Return whether routing may overwrite its input without a data race."""
        return not torch._C._is_alias_of(
            routed_input, shared_input
        ) or not self.requires_input_preservation(shared_input)

    def _late_stream(self, hidden_states: torch.Tensor) -> bool:
        return (
            henvs.custom_ops_enabled()
            and getattr(self, "_hcu_step5_owner", False)
            and _LATE_STREAM_MIN_TOKENS > 0
            and self._stream is not None
            and hidden_states.shape[0] >= _LATE_STREAM_MIN_TOKENS
            # Microbatched: the combine already overlaps the other microbatch.
            and not dbo_enabled()
        )

    def wait_late(self) -> None:
        """Join a late-launched shared expert (see _LATE_STREAM_MIN_TOKENS)."""
        if self._output_pending_late[self._output_idx]:
            assert self._stream is not None
            current_stream().wait_stream(self._stream)
            self._output_pending_late[self._output_idx] = False

    def _should_run_shared_in_aux_stream(
        self,
        hidden_states: torch.Tensor,
    ) -> bool:
        return (
            current_platform.is_cuda_alike()
            and self._stream is not None
            and hidden_states.shape[0]
            <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        )

    def maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if experts_order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            assert self._stream is not None

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            shared_experts_input.record_stream(self._stream)
            if x_and_scale_quanted is not None:
                if not self._layer_supports_x_and_scale_quanted():
                    raise RuntimeError(
                        "HCU pre-quantized shared-expert input requires a layer "
                        "that accepts x_and_scale_quanted"
                    )
                x_and_scale_quanted[0].record_stream(self._stream)
                x_and_scale_quanted[1].record_stream(self._stream)

            # Mark sync start point for the aux stream since we will
            # run in parallel with router/gate.
            self._stream.wait_stream(current_stream())

            if (
                henvs.VLLM_HCU_USE_CUSTOM_OPS
                and (henvs.VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH
                     or self._can_overlap_deepep_eplb())
                and not self._late_stream(shared_experts_input)
            ):
                if self._can_overlap_deepep_eplb():
                    logger.info_once(
                        "Shared experts: early auxiliary-stream overlap enabled "
                        "for quantized DeepEP with EPLB."
                    )
                # Launch before the gate/router, not after routed experts.
                # output() performs the join only when consuming the result.
                self._launch_in_aux_stream(
                    shared_experts_input,
                    x_and_scale_quanted=x_and_scale_quanted,
                )

    def _launch_in_aux_stream(
        self,
        shared_experts_input: torch.Tensor,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        assert self._stream is not None
        with torch.cuda.stream(self._stream):
            self._output[self._output_idx] = self._run_layer(
                shared_experts_input,
                x_and_scale_quanted=x_and_scale_quanted,
            )
        self._output_pending_on_stream[self._output_idx] = True

    def _run_in_aux_stream(
        self,
        shared_experts_input: torch.Tensor,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # TODO: assert that maybe_sync_shared_experts_stream has been called.

        # Run shared experts in parallel on a separate stream.
        with torch.cuda.stream(self._stream):
            output = self._run_layer(
                shared_experts_input,
                x_and_scale_quanted=x_and_scale_quanted,
            )
        current_stream().wait_stream(self._stream)

        return output

    @property
    def _output_idx(self) -> int:
        return dbo_current_ubatch_id() if self.enable_dbo else 0

    @property
    def output(self) -> torch.Tensor:
        assert self._output[self._output_idx] is not None
        if self._output_pending_on_stream[self._output_idx]:
            assert self._stream is not None
            current_stream().wait_stream(self._stream)
            self._output_pending_on_stream[self._output_idx] = False
        output = self._output[self._output_idx]
        self._output[self._output_idx] = None
        pending = getattr(self, "_output_pending_fused_router", None)
        if pending is not None:
            pending[self._output_idx] = False
        return output

    def forward(
        self,
        shared_experts_input: torch.Tensor,
        order: SharedExpertsOrder,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if order != experts_order:
            return None

        if (
            order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
            and henvs.VLLM_HCU_USE_CUSTOM_OPS
            and (henvs.VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH
                 or self._can_overlap_deepep_eplb())
            and self._output[self._output_idx] is not None
        ):
            return None

        assert self._output[self._output_idx] is None

        if order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED and self._late_stream(
            shared_experts_input
        ):
            # Start only once the routed experts are done: the aux stream waits
            # on the main stream as of now, not as of the sync at layer entry.
            assert self._stream is not None
            self._stream.wait_stream(current_stream())
            with torch.cuda.stream(self._stream):
                self._output[self._output_idx] = self._run_layer(
                    shared_experts_input,
                    x_and_scale_quanted=x_and_scale_quanted,
                )
            self._output_pending_late[self._output_idx] = True
        elif order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            self._output[self._output_idx] = self._run_in_aux_stream(
                shared_experts_input,
                x_and_scale_quanted=x_and_scale_quanted,
            )
        else:
            self._output[self._output_idx] = self._run_layer(
                shared_experts_input,
                x_and_scale_quanted=x_and_scale_quanted,
            )

        assert self._output[self._output_idx] is not None
