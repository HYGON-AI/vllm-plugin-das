"""Channel/token INT8 MoE dispatch through the public HCU AITER API."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import logging
from threading import Lock
from typing import Any, Callable


LOG = logging.getLogger(__name__)
VLLM_LOGGER_NAME = f"vllm.{__name__}"


@dataclass(frozen=True)
class _AiterApi:
    quant_type_w8a8: object
    get_config: Callable[..., tuple[bool, object]]
    aiter_moe: Callable[..., object]


_evidence_lock = Lock()
_reported_evidence: set[tuple[object, ...]] = set()


def _runtime_logger() -> logging.Logger:
    """Use vLLM's configured logger once the product module is active."""

    try:
        init_logger = import_module("vllm.logger").init_logger
    except (ImportError, AttributeError):
        return LOG
    # vLLM configures a non-propagating handler on the ``vllm`` namespace.
    # Keep the Plugin module identity as a child so INFO evidence reaches it.
    return init_logger(VLLM_LOGGER_NAME)


def _load_aiter_api() -> _AiterApi:
    module = import_module("aiter.moe")
    return _AiterApi(
        quant_type_w8a8=module.MoeQuantType.W8A8,
        get_config=module.get_aiter_moe_config,
        aiter_moe=module.aiter_moe,
    )


def _load_triton_fallback() -> Callable[..., object]:
    module = import_module(
        "vllm.model_executor.layers.fused_moe.fused_moe"
    )
    return module.fused_experts_impl


def _activation_name(activation: object) -> str:
    value = getattr(activation, "value", activation)
    if not isinstance(value, str) or not value:
        raise ValueError("AITER INT8 MoE requires a named activation")
    return value.lower()


def _shape(tensor: object, rank: int, label: str) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if not isinstance(shape, tuple):
        try:
            shape = tuple(shape)
        except TypeError as error:
            raise ValueError(f"{label} has no tensor shape") from error
    if len(shape) != rank or not all(
        isinstance(value, int) and value > 0 for value in shape
    ):
        raise ValueError(f"{label} must have a positive rank-{rank} shape")
    return shape


def _record_once(
    level: int,
    event: str,
    problem: tuple[int, ...],
    solution: object,
    config: object,
) -> None:
    config_values = getattr(config, "config", None)
    stable_config = (
        tuple(sorted(config_values.items()))
        if isinstance(config_values, dict)
        else repr(config_values)
    )
    key = (event, *problem, str(solution), stable_config)
    with _evidence_lock:
        if key in _reported_evidence:
            return
        _reported_evidence.add(key)
    _runtime_logger().log(
        level,
        "HCU AITER INT8 MoE %s: M=%d E=%d N1=%d N2=%d K=%d top_k=%d "
        "solution=%s config=%s",
        event,
        *problem,
        solution,
        config_values,
    )


def _triton_fallback(
    *,
    hidden_states: object,
    w1: object,
    w2: object,
    topk_weights: object,
    topk_ids: object,
    activation: str,
    global_num_experts: int,
    expert_map: object,
    quant_config: object,
) -> object:
    return _load_triton_fallback()(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
        apply_router_weight_on_input=False,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=getattr(quant_config, "w1_scale", None),
        w2_scale=getattr(quant_config, "w2_scale", None),
        w1_zp=getattr(quant_config, "w1_zp", None),
        w2_zp=getattr(quant_config, "w2_zp", None),
        a1_scale=getattr(quant_config, "a1_scale", None),
        a2_scale=getattr(quant_config, "a2_scale", None),
        block_shape=None,
    )


def apply_aiter_int8_moe(
    *,
    hidden_states: object,
    w1: object,
    w2: object,
    topk_weights: object,
    topk_ids: object,
    moe_config: object,
    quant_config: object,
    activation: object,
    expert_map: object,
    output_dtype: object,
) -> object:
    """Execute a native-layout AITER W8A8 config or official Triton fallback."""

    if not bool(getattr(quant_config, "use_int8_w8a8", False)):
        raise ValueError("AITER INT8 MoE requires an INT8 W8A8 quant config")

    m, k = _shape(hidden_states, 2, "hidden_states")
    e, n1, weight_k = _shape(w1, 3, "w1")
    e2, n2, intermediate = _shape(w2, 3, "w2")
    topk_m, top_k = _shape(topk_ids, 2, "topk_ids")
    if weight_k != k or e2 != e or intermediate * 2 != n1 or topk_m != m:
        raise ValueError("AITER INT8 MoE received inconsistent tensor shapes")
    if _shape(topk_weights, 2, "topk_weights") != (m, top_k):
        raise ValueError("AITER INT8 MoE top-k weights and IDs must match")

    activation_name = _activation_name(activation)
    global_num_experts = int(getattr(moe_config, "num_experts", e))
    problem = (m, e, n1, n2, k, top_k)
    api = _load_aiter_api()
    found, selected = api.get_config(
        m,
        e,
        n1,
        n2,
        k,
        top_k,
        0,
        getattr(hidden_states, "dtype", None),
        api.quant_type_w8a8,
        activation=activation_name,
        use_shuffle=0,
    )
    solution = getattr(selected, "solution_type", None)
    unsupported_layout = bool(
        getattr(selected, "need_shuffle", False)
        or getattr(selected, "need_shuffle_scale", False)
    )
    if not found or unsupported_layout:
        reason = "config-miss" if not found else "unsupported-layout"
        _record_once(logging.WARNING, f"fallback:{reason}", problem, solution, selected)
        return _triton_fallback(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation_name,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            quant_config=quant_config,
        )

    _record_once(logging.INFO, "selected", problem, solution, selected)
    return api.aiter_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        moe_config=selected,
        inplace=False,
        activation=activation_name,
        w1_scale=getattr(quant_config, "w1_scale", None),
        w2_scale=getattr(quant_config, "w2_scale", None),
        w1_zp=getattr(quant_config, "w1_zp", None),
        w2_zp=getattr(quant_config, "w2_zp", None),
        a1_scale=getattr(quant_config, "a1_scale", None),
        a2_scale=getattr(quant_config, "a2_scale", None),
        block_shape=None,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        routed_scaling_factor=1.0,
        use_weight_shuffle=False,
        output_dtype=output_dtype,
    )


__all__ = ["apply_aiter_int8_moe"]
