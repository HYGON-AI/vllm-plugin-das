# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8
"""

PATCHES = [
(
"""
logger = init_logger(__name__)
""",
"""
import vllm_hcu.platforms.envs as henvs

logger = init_logger(__name__)

def apply_int8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    params_dtype: torch.dtype,
    input_scale: torch.Tensor | None = None,
    input_zero_point: torch.Tensor | None = None,
    azp_adj: torch.Tensor | None  = None,
    bias: torch.Tensor | None = None,
    x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    from lmslim import quant_ops
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8

    if (henvs.VLLM_HCU_USE_FUSED_SILU_MUL_QUANT or henvs.VLLM_HCU_USE_FUSED_RMS_QUANT) and \
        henvs.VLLM_HCU_USE_CUSTOM_OPS and \
        x_and_scale_quanted is not None:
        assert len(x_and_scale_quanted) == 2
        assert x_and_scale_quanted[0] is not None
        assert x_and_scale_quanted[1] is not None
        x_q, x_scale = x_and_scale_quanted
    else:
        x_q, x_scale=per_token_quant_int8(input)

    m = x_q.shape[0]
    n = weight.shape[0]
    k = x_q.shape[1]
    out_dtype = params_dtype
    if out_dtype not in (torch.bfloat16, torch.float16):
        out_dtype = torch.bfloat16
    _, out = quant_ops.hipblaslt_w8a8_gemm(x_q, weight, x_scale, weight_scale, m, n, k, 'NT', out_dtype)
    if bias is not None:
        out += bias
    return out
""",
),

(
"""
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
""",
"""
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if henvs.VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM:
            layer.weight.data = layer.weight.data.T
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if henvs.VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM:
            return apply_int8_linear(input=x,
                                     weight=layer.weight,
                                     weight_scale=layer.weight_scale,
                                     params_dtype=layer.params_dtype,
                                     input_scale=layer.input_scale,
                                     input_zero_point=layer.input_zero_point,
                                     azp_adj=layer.azp_adj,
                                     bias=bias,
                                     x_and_scale_quanted=x_and_scale_quanted)
        else:
            return self.kernel.apply_weights(layer, x, bias)

    def supports_quanted_inputs(self):
        return True
""",
    ),
]
