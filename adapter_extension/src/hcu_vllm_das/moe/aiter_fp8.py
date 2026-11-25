# SPDX-License-Identifier: Apache-2.0
"""Shared AITER channel-FP8 config/weight/execution interface.

Query the vendor first. A caller may supply explicitly qualified profiles when
its exact geometry has no tuning row. Prepare one immutable weight layout before
capture; subsequent token counts must use a compatible config. Native and model
qualification remain the consumer's responsibility. No platform/oracle is patched.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Problem:
    tokens: int
    experts: int
    n1: int
    n2: int
    k: int
    topk: int
    dtype: Any
    activation: str = "silu"
    gated: bool = True
    gemm1_alpha: float | None = None
    gemm1_limit: float | None = None
    architecture: str | None = None

    @property
    def weight_key(self):
        return (
            self.experts,
            self.n1,
            self.n2,
            self.k,
            self.topk,
            self.dtype,
            self.activation,
            self.gated,
            self.gemm1_alpha,
            self.gemm1_limit,
            self.architecture,
        )


@dataclass(frozen=True)
class QualifiedProfile:
    """Consumer-owned native evidence, never an implicit vendor tuning hit."""

    weight_key: tuple
    use_shuffle: bool
    config_items: tuple
    evidence: str
    max_tokens: int
    solution_type: str = "asm"


def _vendor():
    from importlib import import_module

    return import_module("aiter.moe")


def _layout_key(problem, config):
    shuffled = bool(config.need_shuffle)
    # Raw channel weights have a canonical layout. For packed weights, retain
    # exact stage IDs conservatively: different M tuning must never silently
    # reinterpret a previously packed weight buffer.
    packing = ()
    if shuffled:
        if config.solution_type == "asm":
            packing = (config.config.get("SOL_ID1"), config.config.get("SOL_ID2"))
        else:
            packing = tuple(
                sorted((key, repr(value)) for key, value in config.config.items())
            )
    return (
        problem.weight_key,
        shuffled,
        bool(config.need_shuffle_scale),
        config.solution_type if shuffled else "canonical",
        packing,
    )


def _definition_key(config):
    return (
        config.quant_type,
        config.solution_type,
        getattr(config, "dtype", None),
        bool(config.need_shuffle),
        bool(config.need_shuffle_scale),
        tuple(sorted((key, repr(value)) for key, value in config.config.items())),
    )


@dataclass(frozen=True)
class ResolvedConfig:
    problem: Problem
    vendor_config: Any
    provenance: str
    layout_key: tuple
    definition_key: tuple
    evidence: str | None = None


def get_config(problem, *, use_shuffle=True, profiles=(), solution_type="asm"):
    """Use actual AITER lookup, otherwise an explicit matching native profile."""
    if type(use_shuffle) is not bool:
        raise TypeError("use_shuffle must be a boolean")
    if (
        type(problem.tokens) is not int
        or problem.tokens < 0
        or any(
            type(n) is not int or n <= 0
            for n in (problem.experts, problem.n1, problem.n2, problem.k, problem.topk)
        )
        or problem.topk > problem.experts
        or (problem.gated and problem.n1 % 2)
    ):
        raise ValueError("Invalid AITER MoE problem geometry")
    api = _vendor()
    found, config = api.get_aiter_moe_config(
        problem.tokens,
        problem.experts,
        problem.n1,
        problem.n2,
        problem.k,
        problem.topk,
        0,
        problem.dtype,
        api.MoeQuantType.FP8_W8A8,
        activation=problem.activation,
        gated=problem.gated,
        spec_sol_type=solution_type,
        use_shuffle=int(use_shuffle),
    )
    evidence = None
    if found:
        config = deepcopy(config)
        if (
            config.quant_type != api.MoeQuantType.FP8_W8A8
            or not isinstance(config.config, dict)
            or not config.config
            or (solution_type is not None and config.solution_type != solution_type)
            or getattr(config, "dtype", None) not in (None, problem.dtype)
        ):
            raise ValueError(
                "AITER config query returned incompatible quantization/layout/dtype"
            )
        provenance = "vendor_query"
    else:
        profile = next(
            (
                p
                for p in profiles
                if (
                    p.weight_key == problem.weight_key
                    and p.use_shuffle == use_shuffle
                    and problem.tokens <= p.max_tokens
                    and (solution_type is None or p.solution_type == solution_type)
                )
            ),
            None,
        )
        if profile is None:
            raise NotImplementedError(
                "No AITER tuning match or explicitly qualified FP8 profile"
            )
        config = api.AiterMoeConfig(
            quant_type=api.MoeQuantType.FP8_W8A8,
            solution_type=profile.solution_type,
            config=dict(profile.config_items),
            need_shuffle=use_shuffle,
            need_shuffle_scale=False,
            dtype=problem.dtype,
        )
        provenance, evidence = "qualified_profile", profile.evidence
    return ResolvedConfig(
        problem,
        config,
        provenance,
        _layout_key(problem, config),
        _definition_key(config),
        evidence,
    )


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError as error:  # Inference tensors have no version counter.
        if "version counter" not in str(error):
            raise
        return None


def _storage(tensor):
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.data_ptr(),
    )


@dataclass(frozen=True)
class PreparedWeights:
    """Own a fixed layout and snapshot each prepared storage binding.

    Storage replacement is checked even for inference tensors. Value changes
    that bypass Torch version counters cannot be detected here, including
    in-place writes to inference tensors; prepared values must stay immutable.
    """

    w1: Any
    w2: Any
    s1: Any
    s2: Any
    layout_key: tuple
    versions: tuple
    storages: tuple

    @property
    def tensors(self):
        return self.w1, self.w2, self.s1, self.s2

    def is_current(self, tensors):
        return (
            len(tensors) == 4
            and all(a is b for a, b in zip(self.tensors, tensors))
            and self.storages == tuple(_storage(t) for t in tensors)
            and self.versions == tuple(_version(t) for t in tensors)
        )

    def validate_bindings(self, tensors):
        if (
            len(tensors) != 4
            or self.storages != tuple(_storage(t) for t in self.tensors)
            or self.storages != tuple(_storage(t) for t in tensors)
            or self.versions != tuple(_version(t) for t in self.tensors)
        ):
            raise ValueError(
                "Prepared AITER weights changed; prepare again before capture"
            )

    def rebind(self, tensors):
        self.validate_bindings(self.tensors)
        if len(tensors) != 4 or self.storages != tuple(_storage(t) for t in tensors):
            raise ValueError(
                "Replacement parameters do not own the prepared AITER storage"
            )
        return replace(
            self,
            w1=tensors[0],
            w2=tensors[1],
            s1=tensors[2],
            s2=tensors[3],
            versions=tuple(_version(t) for t in tensors),
        )

    def copy_into(self, tensors):
        """Install packed values into existing owners before Graph capture.

        Core keeps parameter storage when packed shape/stride/dtype/device match.
        Copy explicitly, then bind to those owners; pointer-only rebind remains
        strict and cannot establish that a separate allocation has these values.
        """
        self.validate_bindings(self.tensors)
        if len(tensors) != 4 or any(
            _storage(a)[:4] != _storage(b)[:4] for a, b in zip(self.tensors, tensors)
        ):
            raise ValueError("AITER destination metadata does not match packed weights")
        for source, owner in zip(self.tensors, tensors):
            if _storage(source) != _storage(owner):
                owner.copy_(source)
            for name, value in source.__dict__.items():
                if name not in ("weight_loader", "_weight_loader"):
                    setattr(owner, name, value)
        return replace(
            self,
            w1=tensors[0],
            w2=tensors[1],
            s1=tensors[2],
            s2=tensors[3],
            versions=tuple(_version(t) for t in tensors),
            storages=tuple(_storage(t) for t in tensors),
        )


def prepare_weights(w1, w2, s1, s2, config):
    """Pack once via public AITER APIs; retain no duplicate raw GPU weights."""
    import torch

    problem = config.problem
    if _definition_key(config.vendor_config) != config.definition_key:
        raise ValueError("Resolved AITER config was mutated")
    intermediate = problem.n1 // 2 if problem.gated else problem.n1
    if (
        tuple(w1.shape) != (problem.experts, problem.n1, problem.k)
        or tuple(w2.shape) != (problem.experts, problem.n2, intermediate)
        or tuple(s1.shape) != (problem.experts, problem.n1, 1)
        or tuple(s2.shape) != (problem.experts, problem.n2, 1)
        or w1.dtype != torch.float8_e4m3fn
        or w2.dtype != torch.float8_e4m3fn
        or s1.dtype != torch.float32
        or s2.dtype != torch.float32
        or any(t.device != w1.device or not t.is_contiguous() for t in (w2, s1, s2))
        or not w1.is_contiguous()
    ):
        raise ValueError(
            "AITER FP8 requires canonical contiguous FN weights/channel scales"
        )
    if config.vendor_config.need_shuffle:
        w1, w2 = _vendor().aiter_moe_shfl_weight(w1, w2, config.vendor_config)
    if config.vendor_config.need_shuffle_scale:
        s1, s2 = _vendor().aiter_moe_shfl_scale(s1, s2, config.vendor_config)
    tensors = (w1, w2, s1, s2)
    return PreparedWeights(
        *tensors,
        config.layout_key,
        tuple(_version(t) for t in tensors),
        tuple(_storage(t) for t in tensors),
    )


def run(
    hidden_states,
    topk_weights,
    topk_ids,
    prepared,
    config,
    *,
    output=None,
    routed_scaling_factor=1.0,
):
    """Execute one selected backend; no silent layout swap, packing or fallback."""
    import torch

    problem = config.problem
    if _definition_key(config.vendor_config) != config.definition_key:
        raise ValueError("Resolved AITER config was mutated")
    if config.layout_key != prepared.layout_key:
        raise ValueError(
            "AITER execution config is incompatible with the prepared weight layout"
        )
    prepared.validate_bindings(prepared.tensors)
    if (
        tuple(hidden_states.shape) != (problem.tokens, problem.k)
        or hidden_states.dtype != problem.dtype
        or not hidden_states.is_contiguous()
        or tuple(topk_ids.shape) != (problem.tokens, problem.topk)
        or tuple(topk_weights.shape) != (problem.tokens, problem.topk)
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_weights.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or any(
            t.device != hidden_states.device
            for t in (*prepared.tensors, topk_ids, topk_weights)
        )
    ):
        raise ValueError(
            "AITER input/routing tensors do not match the resolved problem"
        )
    if output is not None and (
        tuple(output.shape) != (problem.tokens, problem.n2)
        or output.dtype != problem.dtype
        or output.device != hidden_states.device
    ):
        raise ValueError("AITER output buffer does not match the resolved problem")
    if problem.tokens == 0:
        return (
            output if output is not None else hidden_states.new_empty((0, problem.n2))
        )
    # Match the official AITER boundary: Core router implementations can emit
    # int64 IDs, while the native sorting ABI consumes int32/F32 routing data.
    native_ids = topk_ids.to(torch.int32)
    native_weights = topk_weights.to(torch.float32)
    result = _vendor().aiter_moe(
        hidden_states,
        prepared.w1,
        prepared.w2,
        native_weights,
        native_ids,
        config.vendor_config,
        inplace=False,
        activation=problem.activation,
        w1_scale=prepared.s1,
        w2_scale=prepared.s2,
        block_shape=None,
        global_num_experts=problem.experts,
        expert_map=None,
        routed_scaling_factor=routed_scaling_factor,
        use_weight_shuffle=bool(config.vendor_config.need_shuffle),
        output_dtype=problem.dtype,
        gemm1_alpha=problem.gemm1_alpha,
        gemm1_limit=problem.gemm1_limit,
    )
    if (
        tuple(result.shape) != (problem.tokens, problem.n2)
        or result.dtype != problem.dtype
        or result.device != hidden_states.device
    ):
        raise ValueError("AITER returned an incompatible output")
    if output is not None:
        output.copy_(result)
        return output
    return result
