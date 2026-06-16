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
            self._stream.wait_stream(current_stream())""",
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
"""    def _run_in_aux_stream(
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
