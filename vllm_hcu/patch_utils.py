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


# ---------------------------------------------------------------------------
# 模块交换：将 vllm_hcu 实现注册为官方 vllm 模块名（写入 sys.modules）。
# 由下方 import_hook 在首次命中相关 import 时惰性执行；映射须全部成功，否则回滚。
# ---------------------------------------------------------------------------
_module_exchanges_done = False
_module_exchange_in_progress = False

# 祖先包 import 触发交换所需的最小名深度（2 = 至少 vllm.xxx，避免仅 import vllm 就加载）。
MODULE_EXCHANGE_LAZY_MIN_DEPTH: int = 2

# 官方模块名 -> vllm_hcu 实现模块名
MODULE_EXCHANGE_MAP: dict[str, str] = {
    "vllm.model_executor.layers.fused_moe.modular_kernel": "vllm_hcu.model_executor.layers.fused_moe.modular_kernel",
    "vllm.model_executor.layers.fused_moe.deep_gemm_utils": "vllm_hcu.model_executor.layers.fused_moe.deep_gemm_utils",
    "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe": "vllm_hcu.model_executor.layers.fused_moe.experts.deep_gemm_moe",
    "vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe": "vllm_hcu.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe",
}


def _import_name_depth(module_name: str) -> int:
    if not module_name:
        return 0
    return module_name.count(".") + 1


def _import_targets_exchange_candidate(module_name: str) -> bool:
    """本次 import 是否会牵涉 MODULE_EXCHANGE_MAP 中的官方名（含其子模块或祖先包）。"""
    depth = _import_name_depth(module_name)
    min_depth = MODULE_EXCHANGE_LAZY_MIN_DEPTH
    for canonical in MODULE_EXCHANGE_MAP:
        if module_name == canonical or module_name.startswith(canonical + "."):
            return True
        if canonical.startswith(module_name + ".") and depth >= min_depth:
            return True
    return False


def _maybe_lazy_module_exchanges(module_name: str, level: int) -> None:
    """首次命中映射相关 import 时再加载并注册，避免插件初始化阶段过早拉起重模块。"""
    if not MODULE_EXCHANGE_MAP or _module_exchanges_done:
        return
    if level != 0:
        return
    if not _import_targets_exchange_candidate(module_name):
        return
    module_exchanges_all_ok()


def module_exchanges_all_ok() -> bool:
    """加载 MODULE_EXCHANGE_MAP 全部条目；任一失败则回滚已注册项并返回 False。"""
    global _module_exchanges_done, _module_exchange_in_progress
    if _module_exchanges_done:
        return True
    if _module_exchange_in_progress:
        # 嵌套 import 时由外层完成整轮加载，内层直接返回。
        return False

    all_ok = True
    loaded: list[tuple[str, str]] = []
    _module_exchange_in_progress = True
    try:
        for canonical, impl in MODULE_EXCHANGE_MAP.items():
            try:
                mod = importlib.import_module(impl)
            except Exception as e:
                all_ok = False
                sys.modules.pop(impl, None)
                logger.warning(
                    "module_exchanges: failed to load %r for %r: %s",
                    impl,
                    canonical,
                    e,
                )
                continue
            sys.modules[canonical] = mod
            sys.modules[impl] = mod
            loaded.append((canonical, impl))
    finally:
        _module_exchange_in_progress = False

    if not all_ok:
        for canonical, impl in loaded:
            sys.modules.pop(canonical, None)
            sys.modules.pop(impl, None)
        return False

    _module_exchanges_done = True
    return True


#Import Hook (软拦截)：在内存中重定向模块。适合“李代桃僵”，即用你自己的模块替换官方模块。
OLD_IMPORT_HOOK = builtins.__import__
def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    _maybe_lazy_module_exchanges(module_name, level)
    try:
        module_mappings = {  
        "vllm.model_executor.parameter": "vllm_hcu.model_executor.parameter",
        "vllm.model_executor.layers.linear": "vllm_hcu.model_executor.layers.linear",
        "vllm.model_executor.layers.sparse_attn_indexer": "vllm_hcu.model_executor.layers.sparse_attn_indexer",
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
    from .patches.patch_aiter_moe_bf16_fp16 import (
        patch_aiter_moe_asm,
    )
    from .patches.patch_w8a8_channelwise_blaslt_apply_scaled_mm import (
        patch_fp8_scaled_mm,
    )
    from .patches.patch_weight_utils_skip_debug import (
        patch_safetensors_weights_iterator,
    )
    patch_aiter_moe_asm()
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
