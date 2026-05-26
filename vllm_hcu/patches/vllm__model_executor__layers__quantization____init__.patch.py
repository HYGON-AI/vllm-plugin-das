# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.layers.quantization.__init__
"""

PATCHES = [
# 添加两个量化方法到 QuantizationMethods
(
'''
    "cpu_awq",
''',
'''
    "cpu_awq",
    "slimquant_marlin",
    "slimquant_compressed_tensors_marlin",    
    "slimquant_w4a8",
''',
),

(
"""
    from .cpu_wna16 import CPUAWQConfig
""",
"""   
    from .cpu_wna16 import CPUAWQConfig
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors.compressed_tensors_marlin import (SlimQuantCompressedTensorsMarlinConfig)
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8Config
""",
),

(
'''
        "cpu_awq": CPUAWQConfig,
''',
'''
        "cpu_awq": CPUAWQConfig,
        "slimquant_marlin": SlimQuantCompressedTensorsMarlinConfig,
        "slimquant_compressed_tensors_marlin": SlimQuantCompressedTensorsMarlinConfig,
        "slimquant_w4a8": SlimQuantW4A8Int8Config,
''',
),
]
