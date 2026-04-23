# SPDX-License-Identifier: Apache-2.0

from .patch_utils import import_hook, patch_module_class_function

def hcu_platform_plugin():
    """Register the HCU platform."""
    import_hook()
    return "vllm_hcu.platforms.hcu.HCUPlatform"

# def hcu_platform_register_model():
#     """Register models for training and inference"""
#     from .model_executor.models import register_model as _reg
#     _reg()
    
def hcu_platform_register_ops():
    patch_module_class_function()
    import vllm_hcu.ops
