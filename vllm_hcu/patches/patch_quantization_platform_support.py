# SPDX-License-Identifier: Apache-2.0

SLIMQUANT_METHODS = [
    "slimquant_marlin",
    "slimquant_compressed_tensors_marlin",
]


def patch_quantization_platform_support() -> None:
    from vllm.platforms import current_platform
    from vllm.platforms.rocm import RocmPlatform

    for method in SLIMQUANT_METHODS:
        if method not in current_platform.supported_quantization:
            current_platform.supported_quantization.append(method)
        if method not in RocmPlatform.supported_quantization:
            RocmPlatform.supported_quantization.append(method)
