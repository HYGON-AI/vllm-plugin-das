"""Channel/token INT8 MoE dispatch through the public HCU AITER API."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import logging
from threading import Lock
from typing import Callable
from types import SimpleNamespace

import torch


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

    def execute(**arguments):
        output = arguments.pop("output", None)
        solution = arguments["moe_config"].solution_type
        if solution != "asm":
            if solution == "ck":
                arguments["expert_map"] = _aiter_expert_mask(arguments["expert_map"])
            return module.aiter_moe(**arguments)
        from .int8_asm import execute as execute_asm, unsupported_reason

        native = import_module("aiter")
        # ASM sorts actual local IDs; retain the canonical map until that
        # device-side conversion. Only CK consumes a binary ownership mask.
        native_arguments = dict(arguments)
        reason = unsupported_reason(native_arguments, native)
        if reason is None:
            return execute_asm(
                native=native, **native_arguments,
                **({"output": output} if output is not None else {}),
            )
        hidden, w1, w2, topk = (
            arguments[name] for name in ("hidden_states", "w1", "w2", "topk_ids")
        )
        problem = (
            hidden.shape[0],
            w1.shape[0],
            w1.shape[1],
            w2.shape[1],
            hidden.shape[1],
            topk.shape[1],
        )
        _record_once(
            logging.WARNING,
            f"fallback:{reason}",
            problem,
            "asm",
            arguments["moe_config"],
        )
        # Fall back before any native launch, preserving the canonical map.
        canonical_map = arguments["expert_map"]
        return _triton_fallback(
            hidden_states=hidden,
            w1=w1,
            w2=w2,
            topk_weights=arguments["topk_weights"],
            topk_ids=topk,
            activation=arguments["activation"],
            global_num_experts=arguments["global_num_experts"],
            expert_map=canonical_map,
            quant_config=SimpleNamespace(
                **{
                    name: arguments[name]
                    for name in (
                        "w1_scale",
                        "w2_scale",
                        "w1_zp",
                        "w2_zp",
                        "a1_scale",
                        "a2_scale",
                    )
                }
            ),
        )

    return _AiterApi(
        quant_type_w8a8=module.MoeQuantType.W8A8,
        get_config=module.get_aiter_moe_config,
        aiter_moe=execute,
    )


def _load_triton_fallback() -> Callable[..., object]:
    module = import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    return module.fused_experts_impl


def _activation_name(activation: object) -> str:
    value = getattr(activation, "value", activation)
    if not isinstance(value, str) or not value:
        raise ValueError("AITER INT8 MoE requires a named activation")
    return value.lower()


def _aiter_expert_mask(expert_map: object) -> object:
    if expert_map is None:
        return None
    mask = (expert_map >= 0).to(dtype=expert_map.dtype)
    return torch.cat((mask, expert_map.new_zeros((1,))))


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


def _qualified_tp1_device(device: object) -> bool:
    if not isinstance(device, torch.device) or device.type != "cuda" or device.index is None:
        return False
    from hcu_adapter.bridges.vllm import detect_hcu

    # Torch is already active; avoid a PCI sysfs scan on every expert call.
    detection = detect_hcu(torch)
    if not detection.detected or detection.architecture != "gfx936":
        return False
    # Adapter reads visible ordinal zero. TP1 alone does not guarantee that
    # the actual tensor uses that ordinal, especially on mixed HCU hosts.
    if device.index == 0:
        return True
    try:
        properties = torch.cuda.get_device_properties(device)
    except Exception:
        return False  # Device identity unavailable; keep the official path.
    return str(getattr(properties, "gcnArchName", "")).strip().split(":", 1)[0] == "gfx936"


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
    asm_only: bool = False,
    allow_ep_global_profile: bool = False,
    output: object | None = None,
) -> object:
    """Execute a native-layout AITER W8A8 config or official Triton fallback."""

    if not bool(getattr(quant_config, "use_int8_w8a8", False)):
        raise ValueError("AITER INT8 MoE requires an INT8 W8A8 quant config")
    if any(getattr(quant_config, name, None) is not None for name in ("w1_bias", "w2_bias")):
        raise ValueError("HCU AITER INT8 MoE does not support expert bias")
    activation_name = _activation_name(activation)
    if activation_name not in ("silu", "gelu"):
        raise ValueError(f"HCU AITER INT8 MoE does not support {activation_name} activation")

    m, k = _shape(hidden_states, 2, "hidden_states")
    e, n1, weight_k = _shape(w1, 3, "w1")
    e2, n2, intermediate = _shape(w2, 3, "w2")
    topk_m, top_k = _shape(topk_ids, 2, "topk_ids")
    if weight_k != k or e2 != e or intermediate * 2 != n1 or topk_m != m:
        raise ValueError("AITER INT8 MoE received inconsistent tensor shapes")
    if _shape(topk_weights, 2, "topk_weights") != (m, top_k):
        raise ValueError("AITER INT8 MoE top-k weights and IDs must match")

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
        **({"spec_sol_type": "asm"} if asm_only else {}),
    )
    # Actual local lookup remains authoritative. Reuse a global routing
    # profile only for the exact installed-stage qualification, never label it
    # a locally tuned config. Other token counts keep whole-call Core fallback.
    # SOL_ID1, SOL_ID2 and sorting block are one qualified plan. Larger M
    # changes both stages and the block; a matching first stage is insufficient.
    qualified_profiles = {
        1: (10010, 20000, 16),
        2: (10013, 20000, 16),
        3: (10013, 20000, 16),
        4: (10013, 20000, 16),
        8: (10013, 20000, 16),
        16: (10013, 20000, 16),
        17: (10009, 20000, 16),
        32: (10011, 20000, 16),
        64: (10008, 20000, 16),
        128: (10011, 20000, 16),
        256: (10011, 20000, 16),
        512: (11005, 21000, 32),
        1024: (12001, 22000, 64),
    }
    if (not found and allow_ep_global_profile and asm_only
            and expert_map is not None and global_num_experts == 256
            and problem[1:] == (128, 1024, 2048, 2048, 8)
            and getattr(hidden_states, "dtype", None) == torch.bfloat16
            and output_dtype == torch.bfloat16 and activation_name == "silu"
            and m in qualified_profiles):
        profile_found, profile = api.get_config(
            m, global_num_experts, n1, n2, k, top_k, 0,
            getattr(hidden_states, "dtype", None), api.quant_type_w8a8,
            activation=activation_name, use_shuffle=0, spec_sol_type="asm",
        )
        values = getattr(profile, "config", None)
        if (profile_found and getattr(profile, "solution_type", None) == "asm"
                and not getattr(profile, "need_shuffle", False)
                and not getattr(profile, "need_shuffle_scale", False)
                and isinstance(values, dict) and values.get("PADDED_K") is None
                and not values.get("USE_SHUFFLE", 0)
                and tuple(values.get(name) for name in (
                    "SOL_ID1", "SOL_ID2", "BLOCK_SIZE_M",
                )) == qualified_profiles[m]):
            found, selected = True, profile
            _record_once(logging.INFO, "selected-ep-global-profile:E256-after-local-miss",
                         problem, "asm", selected)
    # E128/N768 has no local vendor profile in the qualified build. These
    # exact donor stage/block plans passed independent TP1 BF16/FP16 numeric
    # and changed-input Graph checks on this raw geometry. Keep other shapes,
    # token counts and parallel modes on the original whole-call fallback.
    parallel = getattr(moe_config, "moe_parallel_config", None)
    if (not found and asm_only and expert_map is None
            and global_num_experts == 128
            and problem[1:] == (128, 1536, 2048, 2048, 8)
            and getattr(hidden_states, "dtype", None) in (torch.bfloat16, torch.float16)
            and output_dtype == hidden_states.dtype
            and getattr(quant_config, "per_act_token_quant", False)
            and all(getattr(quant_config, name, None) is None
                    for name in ("w1_zp", "w2_zp", "a1_scale", "a2_scale"))
            and activation_name == "silu" and m in qualified_profiles
            and all(getattr(parallel, name, None) == 1
                    for name in ("tp_size", "ep_size", "dp_size", "pcp_size"))
            and getattr(parallel, "ep_rank", None) == 0
            and all(getattr(parallel, name, None) is False for name in (
                "use_ep", "is_sequence_parallel", "enable_eplb", "use_passthrough_all2all",
            )) and _qualified_tp1_device(getattr(hidden_states, "device", None))):
        profile_found, profile = api.get_config(
            m, 256, 1024, 2048, 2048, 8, 0, hidden_states.dtype,
            api.quant_type_w8a8, activation="silu", use_shuffle=0, spec_sol_type="asm",
        )
        values = getattr(profile, "config", None)
        if (profile_found and getattr(profile, "solution_type", None) == "asm"
                and not getattr(profile, "need_shuffle", False)
                and not getattr(profile, "need_shuffle_scale", False)
                and isinstance(values, dict) and values.get("PADDED_K") is None
                and not values.get("USE_SHUFFLE", 0)
                and tuple(values.get(name) for name in (
                    "SOL_ID1", "SOL_ID2", "BLOCK_SIZE_M",
                )) == qualified_profiles[m]):
            found, selected = True, profile
            _record_once(logging.INFO, "selected-tp1-qualified:E256-N512-after-local-miss",
                         problem, "asm", selected)
    solution = getattr(selected, "solution_type", None)
    unsupported_layout = bool(
        getattr(selected, "need_shuffle", False)
        or getattr(selected, "need_shuffle_scale", False)
    )
    unsupported_expert_map = expert_map is not None and solution not in (
        "asm",
        "ck",
        "triton",
        "moe_c",
    )
    unsupported_provider = asm_only and solution != "asm"
    if not found or unsupported_layout or unsupported_expert_map or unsupported_provider:
        reason = (
            "config-miss"
            if not found
            else "unsupported-layout"
            if unsupported_layout
            else "unsupported-provider"
            if unsupported_provider
            else "unsupported-expert-map"
        )
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
        # Keep canonical routing until the final solution-specific boundary.
        expert_map=expert_map,
        routed_scaling_factor=1.0,
        use_weight_shuffle=False,
        output_dtype=output_dtype,
        **({"output": output} if output is not None else {}),
    )


__all__ = ["apply_aiter_int8_moe"]
