import sys
import importlib
import builtins
from typing import Optional, Callable
from vllm.logger import init_logger

logger = init_logger(__name__)

#磁盘加载替换
_patches_applied = False
def _apply_vllm_patches() -> None:
    
    global _patches_applied
    if _patches_applied:
        return

    try:
        from .patches import apply_patches

        apply_patches()
    except Exception as e:
        logger.error(f"Failed to apply vLLM patches: {e}")

    _patches_applied = True

def _register_patches() -> None:
    """Apply vLLM source patches for hcu."""
    _apply_vllm_patches()

#Import Hook (软拦截)：在内存中重定向模块。适合“李代桃僵”，即用你自己的模块替换官方模块。
OLD_IMPORT_HOOK = builtins.__import__
def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    try:
        module_mappings = {  
            "vllm.model_executor.parameter": "vllm_hcu.model_executor.parameter",
            "vllm.model_executor.layers.linear": "vllm_hcu.model_executor.layers.linear",
        }
        
        if module_name in module_mappings:
            target_module = module_mappings[module_name]
            
            if module_name in sys.modules:
                module = importlib.import_module(target_module)
                sys.modules[module_name] = module
                sys.modules[target_module] = module
                return module
            
            module = importlib.import_module(target_module)
            sys.modules[module_name] = module
            sys.modules[target_module] = module
            return module
            
    except Exception:
        pass

    return OLD_IMPORT_HOOK(
        module_name,
        globals=globals,
        locals=locals,
        fromlist=fromlist,
        level=level
    )

def import_hook():
    """Apply import hook for VLLM hcu"""
    builtins.__import__ = _custom_import


#补丁替换类中函数
def patch_module_class_function():
    # Lazy import so plugin loading does not eagerly import optional deps.
    from .patches.patch_unified_kv_cache_update import (
        patch_unified_kv_cache_update,
    )
    from .patches.patch_w8a8_channelwise_blaslt_apply_scaled_mm import (
        patch_fp8_scaled_mm,
    )
    from .patches.patch_weight_utils_skip_debug import (
        patch_safetensors_weights_iterator,
    )
    patch_unified_kv_cache_update()
    patch_fp8_scaled_mm()
    patch_safetensors_weights_iterator()

#自定义函数或者类的补丁
def patch_fuction_class(custom_function):
    # 配置需要 patch 的模块和对应的 patch 函数
    PATCH_CONFIG = {
        # 'vllm.model_executor.model_loader.weight_utils': {
        #     'another_function': custom_function,
        #     # 可以添加更多要替换的属性
        #     # 'another_function': custom_function,
        # },
    }

    # 使用标志避免重复 patch
    _patched_modules = set()
    original_import = __import__

    def patched_import(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)

        # 检查是否在配置中
        if name in PATCH_CONFIG:
            if id(module) not in _patched_modules:
                _patched_modules.add(id(module))
                # 应用所有 patch
                for attr_name, patch_func in PATCH_CONFIG[name].items():
                    setattr(module, attr_name, patch_func)

        return module

    # 替换 __import__
    __builtins__['__import__'] = patched_import

    # 如果已经导入过了，直接替换
    for module_name, patches in PATCH_CONFIG.items():
        if module_name in sys.modules:
            module = sys.modules[module_name]
            if id(module) not in _patched_modules:
                _patched_modules.add(id(module))
                for attr_name, patch_func in patches.items():
                    setattr(module, attr_name, patch_func)
