# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.fused_moe.runner.moe_runner:
custom SP pre-quantized MoE inputs.
"""

PATCHES = [
(
"""from collections.abc import Callable""",
"""import inspect
from collections.abc import Callable""",
),
(
"""def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return layer.runner._forward_impl(
        layer,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
    )""",
"""def _moe_forward(
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
    return layer.runner._forward_impl(
        layer,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
        quanted_hidden_states=quanted_hidden_states,
        scale=scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )""",
),
(
"""def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:""",
"""def _moe_forward_fake(
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
) -> torch.Tensor:""",
),
(
"""def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return layer.runner._forward_impl(
        layer,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
    )""",
"""def _moe_forward_shared(
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
    return layer.runner._forward_impl(
        layer,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
        quanted_hidden_states=quanted_hidden_states,
        scale=scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )""",
),
(
"""def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:""",
"""def _moe_forward_shared_fake(
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
) -> tuple[torch.Tensor, torch.Tensor]:""",
),
(
"""    def _maybe_apply_shared_experts(
        self,
        shared_experts_input: torch.Tensor | None,
        order: SharedExpertsOrder,
    ):
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts.apply(shared_experts_input, order)""",
"""    def _maybe_apply_shared_experts(
        self,
        shared_experts_input: torch.Tensor | None,
        order: SharedExpertsOrder,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts.apply(
                shared_experts_input,
                order,
                x_and_scale_quanted=x_and_scale_quanted,
            )""",
),
(
"""    def _apply_quant_method(
        self,""",
"""    def _quant_method_supports_quanted_inputs(self) -> bool:
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
        self._supports_quanted_inputs = (
            any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in parameters.values()
            )
            or (
                "i_q" in parameters
                and "i_s" in parameters
            )
        )
        return self._supports_quanted_inputs

    def _apply_quant_method(
        self,""",
),
(
"""    def _apply_quant_method(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        \"\"\"Run expert routing and the fused MoE kernel via the quant method.

        Orchestrates shared expert execution (before/after), expert selection
        via the router, and the actual fused MoE computation. Returns
        (shared_expert_output, fused_expert_output).
        \"\"\"
        self._maybe_apply_shared_experts(
            shared_experts_input, SharedExpertsOrder.NO_OVERLAP
        )

        # Get routing replay buffer from persistent layer attribute
        # (set by bind_routing_capture_to_model during capturer init)
        routing_replay_out = getattr(layer, \"_routing_replay_out\", None)

        if self._quant_method.is_monolithic:
            fused_out = self._quant_method.apply_monolithic(
                layer=layer,
                x=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        else:
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )

            # Write routing data for non-monolithic path (Triton, etc.)
            if routing_replay_out is not None:
                routing_replay_out[: topk_ids.shape[0]].copy_(topk_ids.to(torch.int16))

            # Passing shared_experts_input in case SharedExpertsOrder is
            # MK_INTERNAL_OVERLAPPED.
            fused_out = self._quant_method.apply(
                layer=layer,
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        )

        return (
            self._shared_experts.output if self._shared_experts is not None else None,
            fused_out,
        )""",
"""    def _apply_quant_method(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        quanted_hidden_states: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        \"\"\"Run expert routing and the fused MoE kernel via the quant method.

        Orchestrates shared expert execution (before/after), expert selection
        via the router, and the actual fused MoE computation. Returns
        (shared_expert_output, fused_expert_output).
        \"\"\"
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

        # Get routing replay buffer from persistent layer attribute
        # (set by bind_routing_capture_to_model during capturer init)
        routing_replay_out = getattr(layer, \"_routing_replay_out\", None)

        if self._quant_method.is_monolithic:
            fused_out = self._quant_method.apply_monolithic(
                layer=layer,
                x=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        else:
            if topk_weights is None or topk_ids is None:
                assert router_logits is not None
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    input_ids=input_ids,
                )

            # Write routing data for non-monolithic path (Triton, etc.)
            if routing_replay_out is not None:
                routing_replay_out[: topk_ids.shape[0]].copy_(topk_ids.to(torch.int16))

            # Passing shared_experts_input in case SharedExpertsOrder is
            # MK_INTERNAL_OVERLAPPED.
            quanted_input_kwargs = {}
            if quanted_hidden_states is not None:
                assert self._quant_method_supports_quanted_inputs()
                quanted_input_kwargs = {
                    "i_q": quanted_hidden_states,
                    "i_s": scale,
                }
            fused_out = self._quant_method.apply(
                layer=layer,
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
                **quanted_input_kwargs,
            )

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
            x_and_scale_quanted=shared_experts_x_and_scale_quanted,
        )

        return (
            self._shared_experts.output if self._shared_experts is not None else None,
            fused_out,
        )""",
),
(
"""    def _maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor | None,
    ):
        # If router/gate provided, then apply it here.
        # (Note: This code runs only when \"overlapped mode\" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts.maybe_sync_shared_experts_stream(shared_experts_input)""",
"""    def _maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor | None,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        # If router/gate provided, then apply it here.
        # (Note: This code runs only when \"overlapped mode\" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts.maybe_sync_shared_experts_stream(
                shared_experts_input,
                x_and_scale_quanted=x_and_scale_quanted,
            )""",
),
(
"""    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:""",
"""    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        quanted_hidden_states: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:""",
),
(
"""        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self._trtllm_mxfp4_unpadded_dim(),
        )""",
"""        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            quanted_hidden_states,
            scale,
            topk_weights,
            topk_ids,
            self._encode_layer_name(),
            self._trtllm_mxfp4_unpadded_dim(),
        )""",
),
(
"""    def _forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:""",
"""    def _forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        quanted_hidden_states: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:""",
),
(
"""        self._maybe_sync_shared_experts_stream(shared_experts_input)""",
"""        shared_experts_x_and_scale_quanted = (
            (quanted_hidden_states, scale)
            if quanted_hidden_states is not None and scale is not None
            else None
        )
        self._maybe_sync_shared_experts_stream(
            shared_experts_input,
            x_and_scale_quanted=shared_experts_x_and_scale_quanted,
        )""",
),
(
"""            shared_output, hidden_states = self._apply_quant_method(
                layer=layer,
                hidden_states=hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input,
                input_ids=input_ids,
            )""",
"""            shared_output, hidden_states = self._apply_quant_method(
                layer=layer,
                hidden_states=hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input,
                input_ids=input_ids,
                quanted_hidden_states=quanted_hidden_states,
                scale=scale,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )""",
),
]
