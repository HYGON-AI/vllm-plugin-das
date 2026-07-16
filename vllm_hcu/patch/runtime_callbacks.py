# SPDX-License-Identifier: Apache-2.0
"""Ordered exact callbacks for HCU-owned runtime compatibility helpers."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType

from ._stage3_common import (
    Stage3CompatibilityError,
    require_callable,
    require_exact_module,
    require_type,
)
from .import_coordinator import (
    IMPORT_COORDINATOR,
    ExactImportCoordinator,
    ImportRegistration,
)


def _load_runtime_callable(
    module_name: str,
    function_name: str,
) -> Callable[..., None]:
    """Import an HCU-owned implementation only after its target is loaded."""

    module = importlib.import_module(module_name)
    function = require_callable(module, function_name, f"{module_name}.{function_name}")
    return function


def apply_base_linear_parameter(module: ModuleType) -> None:
    """Patch the completed official parameter module without replacing it."""

    target = "vllm.model_executor.parameter"
    module = require_exact_module(module, target)
    for class_name in (
        "BasevLLMParameter",
        "_ColumnvLLMParameter",
        "RowvLLMParameter",
        "ModelWeightParameter",
        "PackedColumnParameter",
        "PackedvLLMParameter",
    ):
        require_type(module, class_name, f"{target}.{class_name}")
    function = _load_runtime_callable(
        "vllm_hcu.runtime_compat.base_linear_parameter",
        "install_base_linear_parameter_compat",
    )
    function(module)
    if not getattr(module, "_hcu_base_linear_parameter_patch_applied", False):
        raise Stage3CompatibilityError(
            "base-linear parameter runtime patch did not apply"
        )
    removed = sys.modules.get("vllm_hcu.model_executor.parameter")
    if removed is not None and removed is not module:
        raise Stage3CompatibilityError(
            "obsolete HCU parameter module loaded alongside official classes"
        )


def apply_fp8_scaled_mm(module: ModuleType) -> None:
    target = "vllm.model_executor.kernels.linear.scaled_mm.pytorch"
    module = require_exact_module(module, target)
    channel_class = require_type(
        module,
        "ChannelWiseTorchFP8ScaledMMLinearKernel",
        f"{target}.ChannelWiseTorchFP8ScaledMMLinearKernel",
    )
    require_type(
        module,
        "TorchFP8ScaledMMLinearKernel",
        f"{target}.TorchFP8ScaledMMLinearKernel",
    )
    function = _load_runtime_callable(
        "vllm_hcu.runtime_compat.scaled_mm",
        "install_fp8_scaled_mm_compat",
    )
    function()
    if not getattr(channel_class, "_hcu_fp8_patch_applied", False):
        raise Stage3CompatibilityError("FP8 scaled-mm runtime patch did not apply")


def apply_hcu_lora_column_parallel(module: ModuleType) -> None:
    target = "vllm.lora.layers.column_parallel_linear"
    module = require_exact_module(module, target)
    require_type(
        module,
        "ColumnParallelLinearWithLoRA",
        f"{target}.ColumnParallelLinearWithLoRA",
    )
    function = _load_runtime_callable(
        "vllm_hcu.runtime_compat.lora_column_parallel",
        "install_hcu_lora_column_parallel_compat",
    )
    # The exact child module has completed ``exec_module`` at this point, but
    # its ``vllm.lora.layers`` parent may still be initialising.  Passing the
    # completed module keeps this callback path from importing through that
    # partial parent package.
    function(module)
    if not getattr(module, "_hcu_lora_column_parallel_linear_patch_applied", False):
        raise Stage3CompatibilityError(
            "HCU LoRA column-linear runtime patch did not apply"
        )


def apply_qwen35_lora_cudagraph(module: ModuleType) -> None:
    target = "vllm.v1.cudagraph_dispatcher"
    module = require_exact_module(module, target)
    dispatcher_class = require_type(
        module,
        "CudagraphDispatcher",
        f"{target}.CudagraphDispatcher",
    )
    function = _load_runtime_callable(
        "vllm_hcu.runtime_compat.qwen35_lora_cudagraph",
        "install_qwen35_lora_cudagraph_compat",
    )
    function()
    marker = "_hcu_qwen35_lora_piecewise_cudagraph_patch_applied"
    if not getattr(dispatcher_class, marker, False):
        raise Stage3CompatibilityError(
            "Qwen3.5 LoRA cudagraph runtime patch did not apply"
        )


def apply_weight_debug_skip(module: ModuleType) -> None:
    # ``weight_utils`` is imported from reload.layerwise while base_loader is
    # still being initialised.  Importing default_loader from a weight_utils
    # post-import callback therefore creates this real v0.21 cycle:
    #
    #   base_loader -> reload.layerwise -> weight_utils -> default_loader
    #                                                -> base_loader (partial)
    #
    # default_loader is the last consumer that copies the iterator binding,
    # so its exact post-import point is both safe and sufficient.  Resolve the
    # already-loaded producer from sys.modules and pass both modules to the
    # installer; the callback itself never imports the partially initialised
    # model_loader package.
    target = "vllm.model_executor.model_loader.default_loader"
    module = require_exact_module(module, target)
    require_callable(
        module,
        "safetensors_weights_iterator",
        f"{target}.safetensors_weights_iterator",
    )
    weight_utils_name = "vllm.model_executor.model_loader.weight_utils"
    weight_utils = sys.modules.get(weight_utils_name)
    if not isinstance(weight_utils, ModuleType):
        raise Stage3CompatibilityError(
            f"weight-debug patch requires loaded target {weight_utils_name}"
        )
    weight_utils = require_exact_module(weight_utils, weight_utils_name)
    require_callable(
        weight_utils,
        "safetensors_weights_iterator",
        f"{weight_utils_name}.safetensors_weights_iterator",
    )
    function = _load_runtime_callable(
        "vllm_hcu.runtime_compat.weight_loading",
        "install_weight_debug_skip_compat",
    )
    function(weight_utils, module)
    if not getattr(weight_utils, "_hcu_skip_weight_patch_applied", False):
        raise Stage3CompatibilityError("weight-debug runtime patch did not apply")

    patched_iterator = getattr(weight_utils, "safetensors_weights_iterator", None)
    if getattr(module, "safetensors_weights_iterator", None) is not patched_iterator:
        raise Stage3CompatibilityError(
            "weight-debug patch did not update default_loader iterator binding"
        )


def apply_kimi_k25_vision_prompt(module: ModuleType) -> None:
    target = "vllm.model_executor.models.kimi_k25"
    module = require_exact_module(module, target)
    processor_class = require_type(
        module,
        "KimiK25MultiModalProcessor",
        f"{target}.KimiK25MultiModalProcessor",
    )
    function = _load_runtime_callable(
        "vllm_hcu.runtime_compat.kimi_k25_vision_prompt",
        "install_kimi_k25_vision_prompt_compat",
    )
    function()
    if not getattr(processor_class, "_hcu_kimi_k25_prompt_patch_applied", False):
        raise Stage3CompatibilityError(
            "Kimi K2.5 vision-prompt runtime patch did not apply"
        )


# Explicit order preserves ``patch_module_class_function`` behavior while each
# callback remains dormant until its own exact target is imported.
_ORDERED_CALLBACKS: tuple[
    tuple[str, str, Callable[[ModuleType], None], tuple[str, ...]], ...
] = (
    (
        "runtime_method.base_linear_parameter",
        "vllm.model_executor.parameter",
        apply_base_linear_parameter,
        (
            "vllm.model_executor.parameter.BasevLLMParameter."
            "load_column_parallel_weight",
            "vllm.model_executor.parameter.BasevLLMParameter."
            "load_row_parallel_weight",
            "vllm.model_executor.parameter._ColumnvLLMParameter."
            "load_column_parallel_weight",
            "vllm.model_executor.parameter._ColumnvLLMParameter."
            "load_merged_column_weight",
            "vllm.model_executor.parameter._ColumnvLLMParameter."
            "load_qkv_weight",
            "vllm.model_executor.parameter.RowvLLMParameter."
            "load_row_parallel_weight",
        ),
    ),
    (
        "runtime_method.fp8_scaled_mm",
        "vllm.model_executor.kernels.linear.scaled_mm.pytorch",
        apply_fp8_scaled_mm,
        (
            "vllm.model_executor.kernels.linear.scaled_mm.pytorch."
            "TorchFP8ScaledMMLinearKernel.get_output_padding",
            "vllm.model_executor.kernels.linear.scaled_mm.pytorch."
            "ChannelWiseTorchFP8ScaledMMLinearKernel.apply_scaled_mm",
        ),
    ),
    (
        "runtime_method.hcu_lora_column_parallel",
        "vllm.lora.layers.column_parallel_linear",
        apply_hcu_lora_column_parallel,
        ("vllm.lora.layers.column_parallel_linear",),
    ),
    (
        "runtime_method.qwen35_lora_cudagraph",
        "vllm.v1.cudagraph_dispatcher",
        apply_qwen35_lora_cudagraph,
        (
            "vllm.v1.cudagraph_dispatcher.CudagraphDispatcher.dispatch",
            "vllm.v1.cudagraph_dispatcher.CudagraphDispatcher.get_capture_descs",
        ),
    ),
    (
        "runtime_method.weight_debug_skip",
        "vllm.model_executor.model_loader.default_loader",
        apply_weight_debug_skip,
        (
            "vllm.model_executor.model_loader.weight_utils."
            "safetensors_weights_iterator",
            "vllm.model_executor.model_loader.default_loader."
            "safetensors_weights_iterator",
        ),
    ),
    (
        "runtime_method.kimi_k25_vision_prompt",
        "vllm.model_executor.models.kimi_k25",
        apply_kimi_k25_vision_prompt,
        (
            "vllm.model_executor.models.kimi_k25."
            "KimiK25MultiModalProcessor._get_prompt_updates",
            "vllm.model_executor.models.kimi_k25."
            "KimiK25DummyInputsBuilder.get_dummy_text",
        ),
    ),
)


def register_runtime_method_callbacks(
    coordinator: ExactImportCoordinator = IMPORT_COORDINATOR,
) -> tuple[ImportRegistration, ...]:
    """Arm all six runtime patches in explicit deterministic order."""

    return tuple(
        coordinator.register_callback(
            patch_id,
            module_name,
            callback,
            targets=targets,
        )
        for patch_id, module_name, callback, targets in _ORDERED_CALLBACKS
    )


def runtime_callback_names() -> tuple[tuple[str, str], ...]:
    return tuple(
        (patch_id, module_name)
        for patch_id, module_name, _, _ in _ORDERED_CALLBACKS
    )


__all__ = [
    "apply_base_linear_parameter",
    "apply_fp8_scaled_mm",
    "apply_hcu_lora_column_parallel",
    "apply_kimi_k25_vision_prompt",
    "apply_qwen35_lora_cudagraph",
    "apply_weight_debug_skip",
    "register_runtime_method_callbacks",
    "runtime_callback_names",
]
