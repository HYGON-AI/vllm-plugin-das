# SPDX-License-Identifier: Apache-2.0
"""Unregistered, preload-only K3.1 SlimQuant model construction candidate.

Private global dictionaries let derived constructors select split KDA layers
without changing upstream globals or their other model registrations. This is
not an installed extension, nor proof of full-model loading or strict IPC.
"""
from __future__ import annotations
from functools import wraps
from types import FunctionType, CodeType

from vllm.models.kimi_k3.nvidia import model as upstream
from vllm.models.kimi_k3.nvidia import kda as upstream_kda
from .mixed_kda import MixedKDAProjection, load_mixed_kda_weights
from .slimquant_load_only import Kimi31SlimQuantLoadOnlyConfig

INNER_ARCH = "HcuKimi31SlimQuantLoadOnlyInner"
OUTER_ARCH = "HcuKimi31SlimQuantLoadOnly"


def _contains_name(code, name):
    return name in code.co_names or any(
        isinstance(item, CodeType) and _contains_name(item, name)
        for item in code.co_consts
    )


def _clone(original, replacements):
    if not isinstance(original, FunctionType):
        raise TypeError("upstream constructor must remain a Python function")
    for name, expected, _ in replacements:
        if original.__globals__.get(name) is not expected or not _contains_name(original.__code__, name):
            raise RuntimeError(f"upstream constructor binding changed: {name}")
    glob = original.__globals__.copy()
    glob.update({name: replacement for name, _, replacement in replacements})
    result = FunctionType(original.__code__, glob, original.__name__, original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    result.__annotations__ = dict(original.__annotations__)
    return wraps(original)(result)


def _mixed_projection(*args, **kwargs):
    if not isinstance(kwargs.get("quant_config"), Kimi31SlimQuantLoadOnlyConfig):
        raise ValueError("KDA candidate requires K3.1 SlimQuant load-only config")
    return MixedKDAProjection(*args, column_linear=upstream_kda.ColumnParallelLinear,
                              merged_linear=upstream_kda._KimiGDNMergedColumnParallelLinear,
                              **kwargs)


class _Delta(upstream_kda.KimiK3DeltaAttention):
    __init__ = _clone(upstream_kda.KimiK3DeltaAttention.__init__, [
        ("_KimiGDNMergedColumnParallelLinear", upstream_kda._KimiGDNMergedColumnParallelLinear, _mixed_projection)])


class _Decoder(upstream.KimiDecoderLayer):
    __init__ = _clone(upstream.KimiDecoderLayer.__init__, [
        ("KimiK3DeltaAttention", upstream_kda.KimiK3DeltaAttention, _Delta)])


class _Model(upstream.KimiLinearModel):
    __init__ = _clone(upstream.KimiLinearModel.__init__, [
        ("KimiDecoderLayer", upstream.KimiDecoderLayer, _Decoder)])

    def load_weights(self, weights):
        return load_mixed_kda_weights(self, weights, super().load_weights)


class Kimi31SlimQuantLoadOnlyInner(upstream.KimiLinearForCausalLM):
    __init__ = _clone(upstream.KimiLinearForCausalLM.__init__, [
        ("KimiLinearModel", upstream.KimiLinearModel, _Model)])

    def forward(self, *args, **kwargs):
        raise RuntimeError("SlimQuant preload candidate refuses forward")


_original_inner_factory = upstream.init_vllm_registered_model

def _registered_inner(*, vllm_config, prefix="", hf_config=None, architectures=None):
    outer = vllm_config.model_config.hf_config
    if getattr(outer, "model_type", None) != "kimi_k31" or hf_config is not outer.text_config:
        raise ValueError("unexpected outer/text Kimi model configuration")
    if architectures != ["KimiLinearForCausalLM"]:
        raise ValueError("unexpected upstream inner architecture")
    if not isinstance(vllm_config.quant_config, Kimi31SlimQuantLoadOnlyConfig):
        raise ValueError("expected explicit SlimQuant load-only config")
    if vllm_config.parallel_config.tensor_parallel_size != 16:
        raise ValueError("this preload candidate requires TP16")
    if getattr(hf_config, "min_moe_intermediate_per_partition", None) != 192:
        raise ValueError("this candidate requires explicit unpadded expert width 192")
    return _original_inner_factory(vllm_config=vllm_config, prefix=prefix,
                                   hf_config=hf_config, architectures=[INNER_ARCH])


class Kimi31SlimQuantLoadOnly(upstream.KimiK3ForConditionalGeneration):
    __init__ = _clone(upstream.KimiK3ForConditionalGeneration.__init__, [
        ("init_vllm_registered_model", _original_inner_factory, _registered_inner)])

    def forward(self, *args, **kwargs):
        raise RuntimeError("SlimQuant preload candidate refuses forward")
