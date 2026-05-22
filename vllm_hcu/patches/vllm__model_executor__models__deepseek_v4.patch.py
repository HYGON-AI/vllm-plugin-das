"""
Patch for vllm/model_executor/models/deepseek_v4.py
"""

PATCHES = [
(
"""
        self.scale_fmt = config.quantization_config["scale_fmt"]
""",
"""
        self.scale_fmt = (
            config.quantization_config.get("scale_fmt", "ue8m0")
            if hasattr(config, "quantization_config")
            and isinstance(config.quantization_config, dict)
            else "ue8m0"
        )
""",
),

(
"""
        # Pre-hc_head residual stream buffer for the MTP draft. Stable
""",
"""
        self.quant_config = quant_config
        # Pre-hc_head residual stream buffer for the MTP draft. Stable
""",
),

(
"""
        expert_mapping = self.get_expert_mapping()
""",
"""
        expert_mapping = self.get_expert_mapping()
        
        def maybe_remap_compressed_tensors_scale_name(param_name: str) -> str:
            if (
                self.quant_config is None
                or self.quant_config.get_name() != "compressed-tensors"
            ):
                return param_name

            compressed_tensors_names = []
            if param_name.endswith(".weight_scale_inv"):
                compressed_tensors_names.append(
                    param_name.replace(".weight_scale_inv", ".weight_scale")
                )
            if param_name.endswith("_weight_scale_inv"):
                compressed_tensors_names.append(
                    param_name.replace("_weight_scale_inv", "_weight_scale")
                )

            for compressed_tensors_name in compressed_tensors_names:
                if compressed_tensors_name in params_dict:
                    return compressed_tensors_name
            return param_name
            
""",
),

(
"""
                name = name.replace(weight_name, param_name)
""",
"""
                name = name.replace(weight_name, param_name)
                
                if name not in params_dict:
                    name = maybe_remap_compressed_tensors_scale_name(name)
""",
),

(
"""
                        name_mapped = name.replace(weight_name, param_name)
""",
"""
                        name_mapped = name.replace(weight_name, param_name)
                        if name_mapped not in params_dict:
                            name_mapped = maybe_remap_compressed_tensors_scale_name(name_mapped)
""",
),

(
"""
                else:
                    if is_pp_missing_parameter(name, self):
                        continue

""",
"""
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        name = maybe_remap_compressed_tensors_scale_name(name)
""",
),
]