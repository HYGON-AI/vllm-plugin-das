# SPDX-License-Identifier: Apache-2.0

from .patch_utils import import_hook, patch_module_class_function
from .patches.patch_deepseek_r1_distill_llama_70b_tokenizer import (
    patch_deepseek_r1_distill_llama_70b_tokenizer,
)

def hcu_platform_plugin():
    """Register the HCU platform."""
    import_hook()
    patch_deepseek_r1_distill_llama_70b_tokenizer()
    return "vllm_hcu.platforms.hcu.HCUPlatform"

def hcu_platform_register_model():
    """Register models for training and inference"""
    from .models import register_model as _reg
    _reg()
    
def hcu_platform_register_ops():
    patch_module_class_function()
    import vllm_hcu.ops
