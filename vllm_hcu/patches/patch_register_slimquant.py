# SPDX-License-Identifier: Apache-2.0


def patch_register_slimquant() -> None:
    from vllm.model_executor.layers import quantization as vllm_quant

    from vllm_hcu.model_executor.layers.quantization.compressed_tensors.compressed_tensors_marlin import (  # noqa: E501
        SlimQuantCompressedTensorsMarlinConfig,
    )

    for quant_name in [
        "slimquant_marlin",
        "slimquant_compressed_tensors_marlin",
    ]:
        if quant_name not in vllm_quant.QUANTIZATION_METHODS:
            vllm_quant.QUANTIZATION_METHODS.append(quant_name)
        vllm_quant._CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quant_name] = (
            SlimQuantCompressedTensorsMarlinConfig
        )
