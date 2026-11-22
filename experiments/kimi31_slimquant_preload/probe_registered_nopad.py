"""Register an experimental load-only quant class in this process only.

Runs the prior config probe; this does not load weights or install an extension.
"""
from pathlib import Path
import runpy
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
from hcu_vllm_das.quantization.slimquant_load_only import Kimi31SlimQuantLoadOnlyConfig
if "slimquant_w4a8" in QUANTIZATION_METHODS:
    raise RuntimeError("refusing to replace an existing quantization owner")
register_quantization_config("slimquant_w4a8")(Kimi31SlimQuantLoadOnlyConfig)
runpy.run_path(str(Path(__file__).parent / "probe_nopad_config.py"), run_name="__main__")
