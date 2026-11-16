# SPDX-License-Identifier: Apache-2.0
"""Leaf adapters for the official FLA pipeline and Qwen GDN decode.

No algorithm, model, backend selector, convolution or state allocator is
replaced. Remove these consumer aliases when Core offers a leaf registry.
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps
from importlib import import_module

from .gdn_contracts import AITER, BOLTOPS, OFFICIAL

LOGGER = logging.getLogger("vllm.hcu_vllm_das.gdn")
PREFILL = tuple(OFFICIAL)[:3]
DECODE = tuple(OFFICIAL)[3:]
MODULES = {"hcu-boltops": "boltops.fla.gdn", "hcu-aiter": "aiter.ops.fla"}


def require_signature(function, contract, label):
    """Fail before publication on parameter, default or execution-kind drift."""
    names, defaults = contract
    implementations = (
        function,
        getattr(function, "__call__", None),
        inspect.unwrap(function),
    )
    if not callable(function) or any(
        check(implementation)
        for implementation in implementations
        for check in (
            inspect.iscoroutinefunction,
            inspect.isasyncgenfunction,
            inspect.isgeneratorfunction,
        )
    ):
        raise RuntimeError(label + " must be a synchronous callable")
    parameters = tuple(inspect.signature(function).parameters.values())
    if tuple(p.name for p in parameters) != names or any(
        p.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        or p.default != defaults.get(p.name, inspect.Parameter.empty)
        for p in parameters
    ):
        raise RuntimeError(label + " has an incompatible GDN signature")


def resolve(provider, name):
    module_name = MODULES[provider]
    contract = (BOLTOPS if provider == "hcu-boltops" else AITER).get(name)
    if contract is None:
        return None
    attribute = name if provider == "hcu-boltops" else contract[0]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name not in {
            module_name,
            module_name.rsplit(".", 1)[0],
            module_name.split(".")[0],
        }:
            raise  # A broken installed dependency is not an absent provider.
        return None
    function = getattr(module, attribute, None)
    if function is None:
        return None
    require_signature(
        function,
        contract if provider == "hcu-boltops" else contract[1:],
        module_name + "." + attribute,
    )
    return module, attribute, function


def prepare_routes(policy):
    routes = {}
    for name in OFFICIAL:
        requested = policy[0 if name in PREFILL else 1]
        candidates = []
        chain = {
            "triton": (),
            "hcu-boltops": ("hcu-boltops",),
            "hcu-aiter": ("hcu-aiter", "hcu-boltops"),
        }[requested]
        for provider in chain:
            entry = resolve(provider, name)
            if entry is not None:
                candidates.append((provider, *entry))
            else:
                LOGGER.warning(
                    "GDN %s: %s unavailable; retaining the next provider",
                    name,
                    provider,
                )
        routes[name] = tuple(candidates)
        LOGGER.info(
            "GDN %s: requested=%s prepared=%s",
            name,
            requested,
            [entry[0] for entry in candidates] + ["triton"],
        )
    return routes


def _aiter_supports(name, a):
    import torch

    if name == "chunk_gated_delta_rule_fwd_h":
        k, u = a["k"], a["u"]
        return (
            k.shape[-1] == u.shape[-1] == 128
            and a["chunk_size"] == 64
            and k.dtype in (torch.float16, torch.bfloat16)
            and all(
                a[key] is None or a[key].dtype == torch.float32
                for key in ("g", "gk", "initial_state")
            )
        )
    if name == "chunk_fwd_o":
        # Gated HIP chunk-O has unresolved numerical divergence. Do not
        # enable it through this migration, even on an explicit AITER policy.
        return (
            a["g"] is None
            and a["q"].shape[-1] == a["v"].shape[-1] == 128
            and a["chunk_size"] == 64
            and a["q"].dtype in (torch.float16, torch.bfloat16)
            and a["h"].dtype == a["q"].dtype
        )
    if name == "fused_sigmoid_gating_delta_rule_update":
        q, state, indices = a["q"], a["initial_state"], a["ssm_state_indices"]
        return (
            state is not None
            and state.dtype == torch.float32
            and state.stride(-1) == 1
            and (indices is None or indices.dtype == torch.int32)
            and (a["cu_seqlens"] is None or a["cu_seqlens"].dtype == torch.int32)
            and (
                a["num_accepted_tokens"] is None
                or (indices is not None and indices.ndim == 2)
            )
            and q.shape[-1] in (32, 64, 128, 256, 512)
            and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and all(a[key].dtype == q.dtype for key in ("k", "v", "a", "b", "dt_bias"))
            and a["A_log"].dtype == torch.float32
        )
    if name == "fused_recurrent_gated_delta_rule_packed_decode":
        state, indices = a["initial_state"], a["ssm_state_indices"]
        return (
            indices.ndim == 1
            and indices.dtype in (torch.int32, torch.int64)
            and state.stride(-1) == 1
            and state.stride(-2) == state.shape[-1]
            and a["out"].is_contiguous()
        )
    return False


def _prefer_official_h(a):
    import torch

    k, u = a["k"], a["u"]
    return (
        k.ndim == u.ndim == 4
        and k.dtype == u.dtype == torch.bfloat16
        and k.shape[-2:] == (4, 128)
        and u.shape[-2:] == (12, 128)
    )


def _call(name, provider, kernel, a):
    if name == "recompute_w_u_fwd":
        return kernel(**a)
    if name == "chunk_gated_delta_rule_fwd_h":
        args = dict(
            a, initial_state_indices=None, transpose_state_layout=True, kernel_cfg=None
        )
        if provider == "hcu-boltops":
            args.pop("chunk_offsets")  # Derived from cu_seqlens/chunk_size.
            args.update(inplace_final_state=False, null_state_index=0)
        elif args["chunk_offsets"] is not None:
            import torch

            args["chunk_offsets"] = args["chunk_offsets"].to(dtype=torch.int64)
        return kernel(**args)
    if name == "chunk_fwd_o":
        args = dict(a)
        backing = args.pop("core_attn_out")
        if backing is not None and (
            backing.ndim != 1
            or not backing.is_contiguous()
            or backing.numel() < a["v"].numel()
            or backing.dtype != a["v"].dtype
            or backing.device != a["v"].device
        ):
            raise ValueError(
                "core_attn_out must be a compatible flat backing with sufficient capacity"
            )
        output = kernel(
            **args,
            g_gamma=None,
            use_exp2=False,
            transpose_state_layout=True,
            kernel_cfg=None,
        )
        if backing is None:
            return output
        if (
            output.shape != a["v"].shape
            or output.dtype != backing.dtype
            or output.device != backing.device
        ):
            raise RuntimeError("GDN chunk-O provider returned an incompatible tensor")
        out = backing[: a["v"].numel()].view(a["v"].shape)
        out.copy_(output)
        return out
    if provider == "hcu-boltops":
        return kernel(**a, null_state_index=0)
    import torch

    args = dict(a)
    indices = args["ssm_state_indices"]
    if indices is not None:
        args["ssm_state_indices"] = torch.where(
            indices > 0, indices, torch.full_like(indices, -1)
        )
    return kernel(**args)


def make_wrapper(name, original, candidates, require_healthy):
    signature = inspect.signature(original)

    @wraps(original)
    def wrapper(*args, **kwargs):
        require_healthy()
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        for provider, _, _, kernel in candidates:
            if provider == "hcu-aiter" and not _aiter_supports(name, a):
                continue
            if (
                provider == "hcu-boltops"
                and name == "chunk_gated_delta_rule_fwd_h"
                and _prefer_official_h(a)
            ):
                break
            # No exception fallback after native execution has been selected.
            return _call(name, provider, kernel, a)
        return original(*args, **kwargs)

    return wrapper
