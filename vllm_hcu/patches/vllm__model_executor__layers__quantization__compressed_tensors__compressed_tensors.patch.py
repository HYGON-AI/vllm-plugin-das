# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors
"""

PATCHES = [
######################## CompressedTensorsLinearMethod support quantized input ##########################
(
'''
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)
''',
'''
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        x_and_scale_quanted: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")

        if x_and_scale_quanted is not None and \
            hasattr(scheme, "supports_quanted_inputs") and \
                scheme.supports_quanted_inputs():
            return scheme.apply_weights(layer, x, bias=bias, x_and_scale_quanted=x_and_scale_quanted)

        return scheme.apply_weights(layer, x, bias=bias)

    def supports_quanted_inputs(self):
        return True
'''
,
),
######################## CompressedTensorsLinearMethod support quantized input ##########################
]
