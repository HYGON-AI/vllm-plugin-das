# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.fused_moe.runner.shared_experts:
allow shared experts to reuse pre-quantized custom SP inputs.
"""

PATCHES = [
(
"""from enum import IntEnum""",
"""import inspect
from enum import IntEnum""",
),
(
"""import vllm.envs as envs""",
"""import vllm.envs as envs
import vllm_hcu.platforms.envs as henvs""",
),
(
"""            if self._stream is not None:
                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")

    @property
    def _disable_shared_experts_overlap(self) -> bool:""",
"""            if self._stream is not None:
                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")

    def _layer_supports_x_and_scale_quanted(self) -> bool:
        cached = getattr(self, "_supports_x_and_scale_quanted", None)
        if cached is not None:
            return cached
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
        if x_and_scale_quanted is not None and self._layer_supports_x_and_scale_quanted():
            return self._layer(
                shared_experts_input,
                x_and_scale_quanted=x_and_scale_quanted,
            )
        return self._layer(shared_experts_input)

    @property
    def _disable_shared_experts_overlap(self) -> bool:""",
),
(
"""        self.enable_dbo = enable_dbo
        self._output: list[torch.Tensor | None] = [None, None]
        self._layer = layer
""",
"""        self.enable_dbo = enable_dbo
        self._output: list[torch.Tensor | None] = [None, None]
        self._output_pending_on_stream: list[bool] = [False, False]
        self._layer = layer
""",
),
(
"""    @property
    def _disable_shared_experts_overlap(self) -> bool:
        # Disable shared expert overlap if:
        #   - we are using eplb with non-default backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothing to gain
        parallel_config = self._moe_config.moe_parallel_config
        return (
            parallel_config.enable_eplb
            and parallel_config.all2all_backend != "allgather_reducescatter"
        ) or parallel_config.use_fi_nvl_two_sided_kernels
""",
"""    @property
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
""",
),
(
"""    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> SharedExpertsOrder:
        if self._disable_shared_experts_overlap:
            return SharedExpertsOrder.NO_OVERLAP

        if self._quant_method.mk_owns_shared_expert:
            return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED

        should_run_shared_in_aux_stream = (
            current_platform.is_cuda()
            and self._stream is not None
            and hidden_states.shape[0]
            <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        )

        if should_run_shared_in_aux_stream:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        else:
            return SharedExpertsOrder.NO_OVERLAP
""",
"""    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> SharedExpertsOrder:
        if self._disable_shared_experts_overlap:
            return SharedExpertsOrder.NO_OVERLAP

        should_run_shared_in_aux_stream = self._should_run_shared_in_aux_stream(
            hidden_states
        )

        if (
            (
                henvs.VLLM_HCU_USE_CUSTOM_OPS
                and (
                    henvs.VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH
                    or henvs.VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE
                )
            )
            and should_run_shared_in_aux_stream
        ):
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED

        if self._quant_method.mk_owns_shared_expert:
            return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED

        if should_run_shared_in_aux_stream:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        else:
            return SharedExpertsOrder.NO_OVERLAP

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
""",
),
(
"""    def maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if experts_order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            assert self._stream is not None
            assert self._moe_config.disable_inplace

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            shared_experts_input.record_stream(self._stream)

            # Mark sync start point for the aux stream since we will
            # run in parallel with router/gate.
            self._stream.wait_stream(current_stream())""",
"""    def maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if experts_order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            assert self._stream is not None
            assert self._moe_config.disable_inplace
            assert self._output[self._output_idx] is None

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            shared_experts_input.record_stream(self._stream)
            if (
                x_and_scale_quanted is not None
                and self._layer_supports_x_and_scale_quanted()
            ):
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
                )""",
),
(
"""    def _run_in_aux_stream(
        self,
        shared_experts_input: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: assert that maybe_sync_shared_experts_stream has been called.

        # Run shared experts in parallel on a separate stream.
        with torch.cuda.stream(self._stream):
            output = self._layer(shared_experts_input)
        current_stream().wait_stream(self._stream)

        return output""",
"""    def _launch_in_aux_stream(
        self,
        shared_experts_input: torch.Tensor,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        assert self._stream is not None

        # Launch shared experts in parallel on a separate stream. The current
        # stream waits only when the shared output is consumed.
        with torch.cuda.stream(self._stream):
            if x_and_scale_quanted is not None:
                self._output[self._output_idx] = self._run_layer(
                    shared_experts_input,
                    x_and_scale_quanted=x_and_scale_quanted,
                )
            else:
                self._output[self._output_idx] = self._run_layer(shared_experts_input)
        self._output_pending_on_stream[self._output_idx] = True

    def _run_in_aux_stream(
        self,
        shared_experts_input: torch.Tensor,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # TODO: assert that maybe_sync_shared_experts_stream has been called.

        # Run shared experts in parallel on a separate stream.
        with torch.cuda.stream(self._stream):
            if x_and_scale_quanted is not None:
                output = self._run_layer(
                    shared_experts_input,
                    x_and_scale_quanted=x_and_scale_quanted,
                )
            else:
                output = self._run_layer(shared_experts_input)
        current_stream().wait_stream(self._stream)

        return output""",
),
(
"""    @property
    def output(self) -> torch.Tensor:
        assert self._output[self._output_idx] is not None
        output = self._output[self._output_idx]
        self._output[self._output_idx] = None
        return output
""",
"""    @property
    def output(self) -> torch.Tensor:
        assert self._output[self._output_idx] is not None
        if self._output_pending_on_stream[self._output_idx]:
            assert self._stream is not None
            current_stream().wait_stream(self._stream)
            self._output_pending_on_stream[self._output_idx] = False
        output = self._output[self._output_idx]
        self._output[self._output_idx] = None
        return output
""",
),
(
"""    def apply(
        self,
        shared_experts_input: torch.Tensor,
        order: SharedExpertsOrder,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if order != experts_order:
            return None

        assert self._output[self._output_idx] is None

        if order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            self._output[self._output_idx] = self._run_in_aux_stream(
                shared_experts_input
            )
        else:
            self._output[self._output_idx] = self._layer(shared_experts_input)

        assert self._output[self._output_idx] is not None""",
"""    def apply(
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
            if x_and_scale_quanted is not None:
                self._output[self._output_idx] = self._run_layer(
                    shared_experts_input,
                    x_and_scale_quanted=x_and_scale_quanted,
                )
            else:
                self._output[self._output_idx] = self._run_layer(shared_experts_input)

        assert self._output[self._output_idx] is not None""",
),
]
