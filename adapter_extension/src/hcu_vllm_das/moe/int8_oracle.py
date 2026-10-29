"""Register the HCU AITER channel/token INT8 backend in vLLM 0.32."""

from __future__ import annotations

from enum import Enum
import functools
from importlib import import_module as _import_module
import inspect
import sys
from types import ModuleType
from typing import Any

from .fused_moe_aiter import apply_aiter_int8_moe


TARGET_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int8"
EXPERTS_MODULE = "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe"
QUANT_UTILS_MODULE = (
    "vllm.model_executor.layers.quantization.utils.quant_utils"
)
INT8_CONSUMER_MODULES = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_int8",
    "vllm.model_executor.layers.quantization.online.int8",
    "vllm.model_executor.layers.quantization.quark.quark_moe",
)
INSTALL_MARKER = "_hcu_vllm_das_aiter_int8_installed"


class Int8OracleInstallError(RuntimeError):
    """Raised before a changed or late vLLM INT8 oracle can be modified."""


HcuAiterInt8Experts: type | None = None


_REQUIRED_SIGNATURES = {
    "backend_to_kernel_cls": ("backend",),
    "map_int8_backend": ("runner_backend",),
    "select_int8_moe_backend": ("config", "weight_key", "activation_key"),
    "convert_to_int8_moe_kernel_format": (
        "int8_backend",
        "w13",
        "w2",
        "layer",
        "w13_scale",
    ),
    "make_int8_moe_quant_config": (
        "int8_backend",
        "w1_scale",
        "w2_scale",
        "a1_scale",
        "a2_scale",
        "w1_bias",
        "w2_bias",
        "per_act_token_quant",
        "layer",
    ),
    "make_int8_moe_kernel": (
        "int8_backend",
        "moe_quant_config",
        "moe_config",
        "experts_cls",
        "routing_tables",
    ),
}


def _require_callable(module: ModuleType, name: str) -> Any:
    value = getattr(module, name, None)
    if not callable(value):
        raise Int8OracleInstallError(
            f"required vLLM 0.32 INT8 oracle callable is unavailable: "
            f"{TARGET_MODULE}.{name}"
        )
    actual = tuple(inspect.signature(value).parameters)
    expected = _REQUIRED_SIGNATURES[name]
    if actual != expected:
        raise Int8OracleInstallError(
            f"vLLM 0.32 INT8 oracle signature changed for {name}: "
            f"expected {expected}, got {actual}"
        )
    return value


def _make_experts_class(
    parent: type,
    int8_weight_key: object,
    int8_activation_key: object,
) -> type:
    class _HcuAiterInt8Experts(parent):
        @staticmethod
        def _supports_current_device() -> bool:
            return True

        @staticmethod
        def _supports_quant_scheme(
            weight_key: object,
            activation_key: object,
        ) -> bool:
            if (
                weight_key == int8_weight_key
                and activation_key == int8_activation_key
            ):
                return True
            return parent._supports_quant_scheme(weight_key, activation_key)

        def apply(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        ):
            if not bool(getattr(self.quant_config, "use_int8_w8a8", False)):
                return super().apply(
                    output,
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    activation,
                    global_num_experts,
                    expert_map,
                    a1q_scale,
                    a2_scale,
                    workspace13,
                    workspace2,
                    expert_tokens_meta,
                    apply_router_weight_on_input,
                )
            if expert_tokens_meta is not None:
                raise Int8OracleInstallError(
                    "HCU AITER INT8 MoE does not support expert token metadata"
                )
            if apply_router_weight_on_input:
                raise Int8OracleInstallError(
                    "HCU AITER INT8 MoE does not support router weight on input"
                )
            result = apply_aiter_int8_moe(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                moe_config=self.moe_config,
                quant_config=self.quant_config,
                activation=activation,
                expert_map=expert_map,
                output_dtype=output.dtype,
            )
            output.copy_(result)
            return None

    _HcuAiterInt8Experts.__name__ = "HcuAiterInt8Experts"
    _HcuAiterInt8Experts.__qualname__ = "HcuAiterInt8Experts"
    _HcuAiterInt8Experts.__module__ = __name__
    return _HcuAiterInt8Experts


def install() -> bool:
    """Install the explicit AITER INT8 route before consumers freeze aliases."""

    global HcuAiterInt8Experts

    target = _import_module(TARGET_MODULE)
    if getattr(target, INSTALL_MARKER, False):
        return False

    loaded_consumers = [name for name in INT8_CONSUMER_MODULES if name in sys.modules]
    if loaded_consumers:
        raise Int8OracleInstallError(
            "vLLM INT8 consumer already imported before AITER oracle install: "
            + ", ".join(loaded_consumers)
        )

    old_enum = getattr(target, "Int8MoeBackend", None)
    if not isinstance(old_enum, type) or not issubclass(old_enum, Enum):
        raise Int8OracleInstallError(
            f"required enum is unavailable: {TARGET_MODULE}.Int8MoeBackend"
        )
    old_values = {member.name: member.value for member in old_enum}
    if "AITER" in old_values:
        raise Int8OracleInstallError(
            "vLLM INT8 oracle already exposes AITER outside this extension"
        )

    backend_to_kernel_cls = _require_callable(target, "backend_to_kernel_cls")
    map_int8_backend = _require_callable(target, "map_int8_backend")
    _require_callable(target, "select_int8_moe_backend")
    convert_to_int8_moe_kernel_format = _require_callable(
        target, "convert_to_int8_moe_kernel_format"
    )
    _require_callable(target, "make_int8_moe_quant_config")
    _require_callable(target, "make_int8_moe_kernel")

    experts_module = _import_module(EXPERTS_MODULE)
    parent_experts = getattr(experts_module, "AiterExperts", None)
    if not isinstance(parent_experts, type):
        raise Int8OracleInstallError(
            f"required class is unavailable: {EXPERTS_MODULE}.AiterExperts"
        )
    quant_utils = _import_module(QUANT_UTILS_MODULE)
    try:
        int8_weight_key = quant_utils.kInt8StaticChannelSym
        int8_activation_key = quant_utils.kInt8DynamicTokenSym
    except AttributeError as error:
        raise Int8OracleInstallError(
            "vLLM 0.32 INT8 channel/token quant keys are unavailable"
        ) from error

    extended_values = dict(old_values)
    extended_values["AITER"] = "AITER"
    extended_enum = Enum("Int8MoeBackend", extended_values, module=TARGET_MODULE)
    experts_class = _make_experts_class(
        parent_experts,
        int8_weight_key,
        int8_activation_key,
    )

    @functools.wraps(backend_to_kernel_cls)
    def hcu_backend_to_kernel_cls(backend):
        if backend == extended_enum.AITER:
            return [experts_class]
        try:
            official_backend = old_enum(backend.value)
        except (AttributeError, ValueError) as error:
            raise ValueError(f"Unknown Int8 MoE backend: {backend!r}") from error
        return backend_to_kernel_cls(official_backend)

    @functools.wraps(map_int8_backend)
    def hcu_map_int8_backend(runner_backend):
        if runner_backend == "aiter":
            return extended_enum.AITER
        official_backend = map_int8_backend(runner_backend)
        return extended_enum(official_backend.value)

    @functools.wraps(convert_to_int8_moe_kernel_format)
    def hcu_convert_to_int8_moe_kernel_format(
        int8_backend,
        w13,
        w2,
        layer=None,
        w13_scale=None,
    ):
        if int8_backend == extended_enum.AITER:
            return w13, w2
        try:
            official_backend = old_enum(int8_backend.value)
        except (AttributeError, ValueError) as error:
            raise ValueError(f"Unknown Int8 MoE backend: {int8_backend!r}") from error
        return convert_to_int8_moe_kernel_format(
            official_backend,
            w13,
            w2,
            layer,
            w13_scale,
        )

    target._hcu_vllm_das_original_int8_moe_backend = old_enum
    target._hcu_vllm_das_original_backend_to_kernel_cls = backend_to_kernel_cls
    target._hcu_vllm_das_original_map_int8_backend = map_int8_backend
    target._hcu_vllm_das_original_convert_to_int8_moe_kernel_format = (
        convert_to_int8_moe_kernel_format
    )
    target.Int8MoeBackend = extended_enum
    target.backend_to_kernel_cls = hcu_backend_to_kernel_cls
    target.map_int8_backend = hcu_map_int8_backend
    target.convert_to_int8_moe_kernel_format = hcu_convert_to_int8_moe_kernel_format
    HcuAiterInt8Experts = experts_class
    setattr(target, INSTALL_MARKER, True)
    return True


__all__ = [
    "HcuAiterInt8Experts",
    "INSTALL_MARKER",
    "INT8_CONSUMER_MODULES",
    "Int8OracleInstallError",
    "install",
]
