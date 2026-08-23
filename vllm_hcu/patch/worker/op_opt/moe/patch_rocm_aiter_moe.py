# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Add GELU-tanh support to the ROCm AITER expert implementation."""

from __future__ import annotations

import ast
import functools
import inspect
import textwrap
import types
from contextvars import ContextVar
from types import ModuleType, SimpleNamespace

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe"
PATCH_ID = "worker.op_opt.moe.experts.rocm_aiter"
TARGETS = (
    f"{TARGET_MODULE}.ActivationMethod",
    f"{TARGET_MODULE}.rocm_aiter_fused_experts",
    f"{TARGET_MODULE}.AiterExperts._supports_activation",
    f"{TARGET_MODULE}.AiterExperts._supports_current_device",
    f"{TARGET_MODULE}.AiterExperts.is_supported_config",
    f"{TARGET_MODULE}.AiterExperts._supports_quant_scheme",
)
_MARKER = "_vllm_hcu_aiter_gelu_tanh_applied"
_EXPLICIT_CAPABILITY_CHECK: ContextVar[bool] = ContextVar(
    "vllm_hcu_aiter_explicit_capability_check", default=False
)


def _build_workspace_aiter_fused_experts(
    function,
    globals_override: dict[str, object] | None = None,
):
    """Clone vLLM's expert wrapper without its official FlyDSL dependency."""

    function_globals = dict(function.__globals__)
    if globals_override:
        function_globals.update(globals_override)

    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        # Synthetic unit-test functions have no recoverable source. Their
        # audited bytecode contains no FlyDSL import, so only enum globals need
        # replacing.
        return types.FunctionType(
            function.__code__,
            function_globals,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )

    tree = ast.parse(source)

    class RemoveOfficialGateMode(ast.NodeTransformer):
        import_count = 0
        value_count = 0

        def visit_ImportFrom(self, node: ast.ImportFrom):
            if node.module != "aiter.ops.flydsl.moe_common":
                return self.generic_visit(node)
            if [alias.name for alias in node.names] != ["GateMode"]:
                raise PatchCompatibilityError(
                    "unexpected imports from aiter.ops.flydsl.moe_common"
                )
            self.import_count += 1
            return None

        def visit_Attribute(self, node: ast.Attribute):
            node = self.generic_visit(node)
            if (
                node.attr == "value"
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "GateMode"
            ):
                values = {
                    "INTERLEAVE": "interleave",
                    "SEPARATED": "separated",
                }
                value = values.get(node.value.attr)
                if value is None:
                    raise PatchCompatibilityError(
                        f"unexpected GateMode member {node.value.attr!r}"
                    )
                self.value_count += 1
                return ast.copy_location(ast.Constant(value=value), node)
            return node

    transformer = RemoveOfficialGateMode()
    tree = transformer.visit(tree)
    if transformer.import_count not in (0, 1):
        raise PatchCompatibilityError(
            "unexpected number of FlyDSL GateMode imports in "
            f"{function.__qualname__}: {transformer.import_count}"
        )
    if transformer.import_count == 0 and transformer.value_count:
        raise PatchCompatibilityError(
            f"{function.__qualname__} uses GateMode without its audited import"
        )
    ast.fix_missing_locations(tree)
    namespace: dict[str, object] = {}
    filename = inspect.getsourcefile(function) or function.__code__.co_filename
    exec(compile(tree, filename, "exec"), function_globals, namespace)
    rebuilt = namespace.get(function.__name__)
    if not callable(rebuilt):
        raise PatchCompatibilityError(
            f"could not rebuild required HCU target {function.__qualname__}"
        )
    rebuilt.__kwdefaults__ = function.__kwdefaults__
    return rebuilt


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    activation_method = require_class(target, "ActivationMethod", TARGETS[0])
    fused_experts = require_callable(target, "rocm_aiter_fused_experts", TARGETS[1])
    experts_class = require_class(target, "AiterExperts", TARGETS[2].rsplit(".", 1)[0])
    supports = require_callable(experts_class, "_supports_activation", TARGETS[2])
    supports_device = require_callable(
        experts_class, "_supports_current_device", TARGETS[3]
    )
    is_supported_config = require_callable(
        experts_class, "is_supported_config", TARGETS[4]
    )
    supports_quant_scheme = require_callable(
        experts_class, "_supports_quant_scheme", TARGETS[5]
    )
    require_parameter_names(
        fused_experts,
        TARGETS[1],
        (
            "hidden_states", "w1", "w2", "topk_weights", "topk_ids",
            "moe_config", "activation", "apply_router_weight_on_input",
            "expert_map", "quant_config", "a1q_scale", "num_local_tokens",
            "output_dtype", "moe_sorting_dispatch_policy",
        ),
    )
    require_parameter_names(supports, TARGETS[2], ("activation",))
    require_parameter_names(supports_device, TARGETS[3], ())
    require_parameter_names(
        is_supported_config,
        TARGETS[4],
        ("cls", "moe_config", "weight_key", "activation_key", "activation_format"),
    )
    require_parameter_names(
        supports_quant_scheme,
        TARGETS[5],
        ("weight_key", "activation_key"),
    )
    values = {member.name: member.value for member in activation_method}
    if values != {"SILU": 0, "GELU": 1}:
        raise PatchCompatibilityError(
            f"required HCU target {TARGETS[0]} has unexpected values {values}"
        )
    hcu_activation_method = target.IntEnum(
        "ActivationMethod",
        {"SILU": 0, "GELU": 1, "GELU_TANH": 3},
        module=target.__name__,
    )
    moe_activation = target.MoEActivation
    gelu_tanh = getattr(moe_activation, "GELU_TANH", None)
    if gelu_tanh is None:
        raise PatchCompatibilityError("MoEActivation.GELU_TANH is missing")

    # Rebuild the audited upstream body without its unconditional GateMode
    # import. HCU AITER has no FlyDSL package or gate_mode ABI.
    normal_impl = _build_workspace_aiter_fused_experts(fused_experts)
    special_globals = dict(fused_experts.__globals__)
    special_globals["MoEActivation"] = SimpleNamespace(
        SILU=moe_activation.SILU,
        GELU=gelu_tanh,
        SWIGLUOAI=moe_activation.SWIGLUOAI,
        SWIGLUOAI_UNINTERLEAVE=moe_activation.SWIGLUOAI_UNINTERLEAVE,
    )
    special_globals["ActivationMethod"] = SimpleNamespace(
        SILU=hcu_activation_method.SILU,
        GELU=hcu_activation_method.GELU_TANH,
    )
    special_impl = _build_workspace_aiter_fused_experts(
        fused_experts,
        special_globals,
    )
    fused_experts_signature = inspect.signature(fused_experts)

    int8_weight_key = getattr(target, "kInt8StaticChannelSym", None)
    int8_activation_key = getattr(target, "kInt8DynamicTokenSym", None)
    if int8_weight_key is None or int8_activation_key is None:
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kInt8DynamicTokenSym,
            kInt8StaticChannelSym,
        )

        int8_weight_key = kInt8StaticChannelSym
        int8_activation_key = kInt8DynamicTokenSym

    @functools.wraps(fused_experts)
    def hcu_fused_experts(*args, **kwargs):
        bound = fused_experts_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        activation = arguments.get("activation")
        moe_config = arguments.get("moe_config")
        quant_config = arguments.get("quant_config")
        from vllm_hcu.model_executor.layers.fused_moe.aiter_runtime import (
            aiter_moe_request_context,
        )

        with aiter_moe_request_context(moe_config):
            if bool(getattr(quant_config, "use_fp8_w8a8", False)) or bool(
                getattr(quant_config, "use_int8_w8a8", False)
            ):
                from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import (
                    apply_aiter_quantized_moe,
                )

                return apply_aiter_quantized_moe(
                    hidden_states=arguments["hidden_states"],
                    w1=arguments["w1"],
                    w2=arguments["w2"],
                    topk_weights=arguments["topk_weights"],
                    topk_ids=arguments["topk_ids"],
                    vllm_moe_config=moe_config,
                    activation=activation,
                    apply_router_weight_on_input=arguments[
                        "apply_router_weight_on_input"
                    ],
                    expert_map=arguments.get("expert_map"),
                    quant_config=quant_config,
                    a1q_scale=arguments.get("a1q_scale"),
                    num_local_tokens=arguments["num_local_tokens"],
                    output_dtype=arguments.get("output_dtype"),
                    moe_sorting_dispatch_policy=arguments[
                        "moe_sorting_dispatch_policy"
                    ],
                )
            if activation == gelu_tanh:
                return special_impl(*args, **kwargs)
            return normal_impl(*args, **kwargs)

    @functools.wraps(supports)
    def hcu_supports_activation(activation):
        if activation == moe_activation.SWIGLUOAI_UNINTERLEAVE:
            return False
        return activation == gelu_tanh or supports(activation)

    @functools.wraps(supports_quant_scheme)
    def hcu_supports_quant_scheme(weight_key, activation_key):
        if (
            weight_key == int8_weight_key
            and activation_key == int8_activation_key
        ):
            return True
        return supports_quant_scheme(weight_key, activation_key)

    @functools.wraps(supports_device)
    def hcu_supports_current_device():
        if not _EXPLICIT_CAPABILITY_CHECK.get():
            return supports_device()
        from vllm._aiter_ops import is_aiter_found_and_supported

        return is_aiter_found_and_supported()

    @functools.wraps(is_supported_config)
    def hcu_is_supported_config(
        cls, moe_config, weight_key, activation_key, activation_format
    ):
        token = _EXPLICIT_CAPABILITY_CHECK.set(
            getattr(moe_config, "moe_backend", None) == "aiter"
        )
        try:
            supported, reason = is_supported_config(
                cls, moe_config, weight_key, activation_key, activation_format
            )
            mxfp4_key = getattr(target, "kMxfp4Static", None)
            if supported and mxfp4_key is not None and weight_key == mxfp4_key:
                return False, (
                    "HCU AITER has no FlyDSL gate-mode/bias ABI "
                    "required by vLLM's MXFP4 AITER expert"
                )
            return supported, reason
        finally:
            _EXPLICIT_CAPABILITY_CHECK.reset(token)

    target._vllm_hcu_original_activation_method = activation_method
    target.ActivationMethod = hcu_activation_method
    target._vllm_hcu_original_rocm_aiter_fused_experts = fused_experts
    target.rocm_aiter_fused_experts = hcu_fused_experts
    experts_class._vllm_hcu_original_supports_activation = supports
    experts_class._supports_activation = staticmethod(hcu_supports_activation)
    experts_class._vllm_hcu_original_supports_current_device = supports_device
    experts_class._supports_current_device = staticmethod(hcu_supports_current_device)
    experts_class._vllm_hcu_original_is_supported_config = is_supported_config
    experts_class.is_supported_config = staticmethod(hcu_is_supported_config)
    experts_class._vllm_hcu_original_supports_quant_scheme = supports_quant_scheme
    experts_class._supports_quant_scheme = staticmethod(hcu_supports_quant_scheme)
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
