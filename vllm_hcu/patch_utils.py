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
            #"vllm.attention.ops.merge_attn_states": "vllm_hcu.ops.attention.merge_attn_states",
        }
        
        if module_name in module_mappings:
            if module_name in sys.modules:
                return sys.modules[module_name]
            target_module = module_mappings[module_name]
            module = importlib.import_module(target_module)
            sys.modules[module_name] = module
            sys.modules[target_module] = module
            
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