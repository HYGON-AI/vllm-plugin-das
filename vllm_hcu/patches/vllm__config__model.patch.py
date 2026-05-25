
"""
Patch for vllm.config.model
"""

PATCHES = [
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
 '''
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization, hf_config=self.hf_config
                )
 ''',
 '''
                skip_hf_config = name in {"slimquant_marlin", "slimquant_compressed_tensors_marlin", "slimquant_w4a8"}
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization,
                    **({} if skip_hf_config else {"hf_config": self.hf_config})
                )
 ''',
),
]