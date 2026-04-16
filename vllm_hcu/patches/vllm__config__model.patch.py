
"""
Patch for vllm.config.model
"""

PATCHES = [
    (
        '''                "cpu_awq",
            ]''',
        '''                "cpu_awq",
                "slimquant_marlin",
                "slimquant_compressed_tensors_marlin",
            ]'''
    ),
]
