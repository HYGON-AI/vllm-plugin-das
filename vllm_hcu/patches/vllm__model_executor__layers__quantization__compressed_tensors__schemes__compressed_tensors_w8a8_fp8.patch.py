# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8
"""

PATCHES = [
(
"""
logger = init_logger(__name__)
""",
"""
import vllm_hcu.platforms.envs as henvs

logger = init_logger(__name__)
""",
),

(
"""
            weight, weight_scale, input_scale = process_fp8_weight_channel_strategy(
                layer.weight, layer.weight_scale, getattr(layer, "input_scale", None)
            )
            weight = weight.t()
""","""
            weight, weight_scale, input_scale = process_fp8_weight_channel_strategy(
                layer.weight, layer.weight_scale, getattr(layer, "input_scale", None)
            )
            if henvs.VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM:
                weight = weight.contiguous()
            else:
                weight = weight.t()
""",
    ),

(
"""
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.fp8_linear.apply_weights(layer, x, bias)
""",
"""
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if x_and_scale_quanted is not None and hasattr(self.fp8_linear, "supports_quanted_inputs"):
            return self.fp8_linear.apply_weights(layer, x, bias, x_and_scale_quanted)
        return self.fp8_linear.apply_weights(layer, x, bias)

    def supports_quanted_inputs(self):
        return True
""",
    ),
]
