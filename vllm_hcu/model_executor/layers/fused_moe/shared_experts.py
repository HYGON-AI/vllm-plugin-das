# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""HCU-owned v0.25.1 shared-expert execution."""

import inspect
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
)

logger = init_logger(__name__)


class SharedExpertsOrder(IntEnum):
    # No shared experts.
    NONE = (0,)

    # No overlap - defensively called before MK.
    NO_OVERLAP = (1,)

    # Overlapped with dispatch/combine in DP/EP - called by the MK.
    MK_INTERNAL_OVERLAPPED = (2,)

    # Overlapped with the gate, router, experts in aux stream.
    MULTI_STREAM_OVERLAPPED = (3,)


class SharedExperts(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        enable_dbo: bool,
        mk_can_overlap_shared_experts: Callable[[], bool],
    ):
        super().__init__()

        # The SharedExperts need to handle DBO since they can be called from
        # an MK's finalize method.  We keep a list of outputs indexed by current
        # DBO ubatch id to handle this case.  If DBO is not enabled, the
        # index is always 0 and the second output list element is ignored.
        self.enable_dbo = enable_dbo
        self._output: list[torch.Tensor | None] = [None, None]
        self._output_pending_on_stream: list[bool] = [False, False]
        self._layer = layer
        self._moe_config = moe_config

        self._mk_can_overlap_shared_experts = mk_can_overlap_shared_experts

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

    @property
    def _disable_shared_experts_overlap(self) -> bool:
        if (
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE
        ):
            return False
        # Disable shared expert overlap if:
        #   - we are using eplb with non-default backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothing to gain
        parallel_config = self._moe_config.moe_parallel_config
        return (
            parallel_config.enable_eplb
            and parallel_config.all2all_backend != "allgather_reducescatter"
        ) or parallel_config.use_fi_nvl_two_sided_kernels

    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> SharedExpertsOrder:
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

        should_run_shared_in_aux_stream = self._should_run_shared_in_aux_stream(
            hidden_states
        )
        if force_stream and should_run_shared_in_aux_stream:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED

        if self._mk_can_overlap_shared_experts():
            return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED

        if should_run_shared_in_aux_stream:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        return SharedExpertsOrder.NO_OVERLAP

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
                and henvs.VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH
            ):
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
            and henvs.VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH
            and self._output[self._output_idx] is not None
        ):
            return None

        assert self._output[self._output_idx] is None

        if order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
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
