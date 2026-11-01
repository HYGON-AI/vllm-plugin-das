"""SlimQuant byte ABI and pre-mutation AITER capability selection."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module

import torch


class W4A8Error(RuntimeError):
    """An explicitly selected W4 route cannot preserve its contract."""


def unpack(packed):
    """Checkpoint order is signed high nibble, then signed low nibble."""
    if packed.dtype != torch.int8 or packed.ndim != 3:
        raise W4A8Error("SlimQuant expects signed-byte [E,N,K/2]")
    output = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        device=packed.device,
        dtype=torch.int8,
    )
    # Bound post-load staging: E512 model tensors must not create several
    # full-expert INT16 intermediates beside resident checkpoint weights.
    for start in range(0, packed.shape[0], 4):
        value = packed[start : start + 4].view(torch.uint8)
        nibbles = torch.stack((value >> 4, value & 15), dim=-1).to(torch.int16)
        output[start : start + 4].copy_(
            torch.where(nibbles >= 8, nibbles - 16, nibbles).to(torch.int8).flatten(-2)
        )
    return output


def int8_weights_and_scales(weights, scales):
    # W4A8 kernels compensate packed signed INT4 by x16 internally. The
    # ordinary INT8 kernel must receive that factor in each GEMM's scale.
    return tuple(unpack(w) for w in weights), tuple(s * 16 for s in scales)


@dataclass(frozen=True)
class Shape:
    experts: int
    hidden: int
    intermediate: int
    top_k: int
    max_tokens: int
    dtype: object


@dataclass
class NativePlan:
    shape: Shape
    configs: tuple
    provider: object

    @property
    def layout(self):
        config = self.configs[0]
        return (config.solution_type, config.need_shuffle, config.need_shuffle_scale)

    def pack(self, weights):
        # The installed public W4 shuffle mutates its input. Finish both
        # private copies before the caller publishes either parameter.
        copies = tuple(w.clone() for w in weights)
        if self.configs[0].need_shuffle:
            copies = self.provider.aiter_moe_shfl_weight(
                *copies,
                self.configs[0],
                block_shape=None,
            )
        return copies

    def run(self, x, w1, w2, scales, topk_weights, topk_ids, *, global_experts):
        if (
            x.ndim != 2
            or x.shape[1] != self.shape.hidden
            or x.dtype != self.shape.dtype
            or tuple(topk_ids.shape) != (x.shape[0], self.shape.top_k)
            or topk_weights.shape != topk_ids.shape
            or not x.is_contiguous()
            or topk_ids.dtype != torch.int32
            or topk_weights.dtype != torch.float32
            or not topk_ids.is_contiguous()
            or not topk_weights.is_contiguous()
            or any(
                t.device != x.device for t in (w1, w2, *scales, topk_ids, topk_weights)
            )
        ):
            raise W4A8Error(
                "Native W4A8 input differs from its pre-pack geometry/dtype"
            )
        tokens = x.shape[0]
        if not 0 <= tokens <= self.shape.max_tokens:
            raise W4A8Error("M exceeds the all-M pre-pack contract")
        if tokens == 0:
            return torch.empty_like(x)
        return self.provider.aiter_moe(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            self.configs[tokens - 1],
            inplace=False,
            activation="silu",
            w1_scale=scales[0],
            w2_scale=scales[1],
            block_shape=None,
            global_num_experts=global_experts,
            expert_map=None,
            use_weight_shuffle=self.configs[0].need_shuffle,
            output_dtype=x.dtype,
        )


def all_m_plan(shape, provider):
    """Check every reachable M and a single shared weight ABI before packing."""
    if (
        any(
            type(n) is not int or n <= 0
            for n in (
                shape.experts,
                shape.hidden,
                shape.intermediate,
                shape.top_k,
                shape.max_tokens,
            )
        )
        or shape.top_k > shape.experts
        or shape.hidden % 64
        or shape.intermediate % 64
        # Installed W4 selector does not forward dtype to the CSV lookup;
        # it defaults indtype to BF16. FP16 retains the official INT8 route.
        or shape.dtype != torch.bfloat16
    ):
        return None
    configs = []
    layout = None
    for tokens in range(1, shape.max_tokens + 1):
        ok, config = provider.get_aiter_moe_config(
            M=tokens,
            E=shape.experts,
            N1=2 * shape.intermediate,
            N2=shape.hidden,
            K=shape.hidden,
            top_k=shape.top_k,
            block_size=0,
            dtype=shape.dtype,
            quant_type=provider.MoeQuantType.W4A8,
            activation="silu",
            gated=True,
            use_shuffle=0,
        )
        if not ok or config.quant_type != "w4a8" or config.need_shuffle_scale:
            return None
        current = (config.solution_type, config.need_shuffle, config.need_shuffle_scale)
        # Only the physically verified MOE_C packed ABI is enabled. Vendor
        # Triton W4 remains a separate gate; fallback here is official INT8.
        if current != ("moe_c", True, False):
            return None
        # Nonempty tuned config is required; a selector success alone is not
        # execution evidence. Different per-M tuning is safe, layout is not.
        if not isinstance(config.config, dict) or not config.config:
            return None
        if layout is not None and layout != current:
            return None
        layout = current
        configs.append(config)
    return NativePlan(shape, tuple(configs), provider)


def prepare_native(moe):
    parallel = moe.moe_parallel_config
    if not (
        moe.moe_backend in ("auto", "aiter")
        and moe.activation.value == "silu"
        and moe.is_act_and_mul
        and not moe.has_bias
        and not moe.is_lora_enabled
        and not parallel.use_ep
        and parallel.dp_size == parallel.pcp_size == 1
        and not parallel.is_sequence_parallel
        and not parallel.enable_eplb
        and all(
            getattr(moe, name, None) is None
            for name in (
                "swiglu_limit",
                "swiglu_alpha",
                "swiglu_beta",
                "activation_situ_beta",
                "activation_situ_linear_beta",
            )
        )
    ):
        return None
    if not torch.version.hip or torch.device(moe.device).type != "cuda":
        return None
    props = torch.cuda.get_device_properties(moe.device)
    arch = str(getattr(props, "gcnArchName", "")).split(":", 1)[0]
    if arch not in ("gfx928", "gfx936", "gfx938"):
        return None
    try:
        provider = import_module("aiter.moe")
    except ImportError:
        return None
    for name, args in (
        ("get_aiter_moe_config", {"M", "E", "N1", "N2", "K", "top_k", "quant_type"}),
        ("aiter_moe", {"moe_config", "w1_scale", "w2_scale", "output_dtype"}),
        ("aiter_moe_shfl_weight", {"w1", "w2", "moe_config"}),
    ):
        function = getattr(provider, name, None)
        if (
            not callable(function)
            or not args <= inspect.signature(function).parameters.keys()
        ):
            raise W4A8Error(f"AITER W4A8 ABI missing: {name}")
    intermediate = ((moe.intermediate_size_per_partition + 63) // 64) * 64
    if intermediate > moe.intermediate_size_per_partition * 1.25:
        return None
    # Use the official physical-padding protocol, with real zero weight/input
    # tails. CSV selectors require exact K; never borrow a different K's config
    # for an unpadded tensor. Bound optional H expansion to at most 25%.
    hidden_sizes = [moe.hidden_dim]
    rounded = ((moe.hidden_dim + 1023) // 1024) * 1024
    if moe.hidden_dim < rounded <= moe.hidden_dim * 1.25:
        hidden_sizes.append(rounded)
    for hidden in hidden_sizes:
        plan = all_m_plan(
            Shape(
                moe.num_local_experts,
                hidden,
                intermediate,
                moe.experts_per_token,
                moe.max_num_tokens,
                moe.in_dtype,
            ),
            provider,
        )
        if plan is not None:
            return plan
    return None
