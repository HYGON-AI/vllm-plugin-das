# SPDX-License-Identifier: Apache-2.0
"""Admit the verified draft Post by default; retain an explicit opt-out."""
from __future__ import annotations

import ast
from functools import lru_cache
import inspect
import logging
import os
from textwrap import dedent
from types import FunctionType

import torch

from hcu_adapter.operator_contracts import supports_operator_contract

KEY = "hcu_dsv41_dspark_final_post"


class DraftPostUnavailable(RuntimeError):
    """The optional draft constructor cannot use the native Post seam."""


def requested(config):
    extra = getattr(config, "additional_config", None)
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise ValueError("DSpark additional_config must be a dict")
    explicit = KEY in extra
    value = extra.get(KEY, True)
    if value is False:
        return False
    if value is not True:
        raise ValueError("W4A8 draft final Post must be an explicit boolean")
    if not explicit:
        # Default admission must not resolve private native bodies for other
        # models/topologies. An explicit request retains the strict error path.
        if getattr(config, "quant_config", None) is None:
            return False
        try:
            _validate(config)
        except (AttributeError, ValueError):
            return False
        return _implementation_if_supported() is not None
    _validate(config)
    implementation()
    return True


@lru_cache(maxsize=1)
def _implementation_if_supported():
    try:
        return implementation()
    except (ImportError, DraftPostUnavailable) as error:
        logging.getLogger("vllm." + __name__).info(
            "HCU optional draft final Post unavailable, using native Post: %s", error)
        return None


def _validate(config):
    from .dspark_agrs import _owned_quant_kind, is_owned_draft, draft_config
    from .dspark_graph import native_decode_dspark
    owned = is_owned_draft(config)
    if _owned_quant_kind(config) != "ct_w4a8" or not (owned or native_decode_dspark(config)):
        raise ValueError("Draft final Post requires the native W4A8 TP1/DP8/EP8 target")
    if owned:
        draft_config(config)  # Recheck the graph loader's fixed-capacity lease.
    mc = config.model_config
    hf = getattr(mc, "hf_text_config", None)
    if hf is None:
        hf = getattr(mc.hf_config, "text_config", mc.hf_config)
    if (getattr(hf, "hidden_size", None), getattr(hf, "hc_mult", None)) != (5120, 4):
        raise ValueError("Draft final Post requires its kernel's H5120/HC4 layout")
    return True


def _clone(owner, name, substitutions):
    original = getattr(owner, name, None)
    function = inspect.unwrap(original)
    if not isinstance(function, FunctionType) or any(
        key not in function.__globals__ for key in substitutions
    ):
        raise DraftPostUnavailable("DSpark native namespace anchor missing: " + name)
    namespace = dict(function.__globals__)
    namespace.update(substitutions)
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__
    clone.__qualname__ = function.__qualname__
    clone.__module__ = function.__module__
    clone.__doc__ = function.__doc__
    return clone, (owner, name, original)


def _has_final_post_seam(native, original_post):
    """Verify the cloned globals refer to the final Post and owned inner model."""
    inner = native.DSparkDeepseekV4Model
    outer = native.DSparkDeepseekV4ForCausalLM
    forward = inspect.unwrap(inner.forward)
    constructor = inspect.unwrap(outer.__init__)
    if (not all(isinstance(fn, FunctionType) for fn in (forward, constructor))
            or forward.__globals__.get('mhc_post_torch') is not original_post
            or constructor.__globals__.get('DSparkDeepseekV4Model') is not inner):
        return False

    def calls(function, name):
        body = ast.parse(dedent(inspect.getsource(function))).body[0].body
        all_calls = [node for statement in body for node in ast.walk(statement)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == name]
        if len(all_calls) != 1:
            return None
        return next((stmt for stmt in body if isinstance(stmt, ast.Assign)
                     and stmt.value is all_calls[0] and len(stmt.targets) == 1), None)

    try:
        last_post = calls(forward, 'mhc_post_torch')
        inner_model = calls(constructor, 'DSparkDeepseekV4Model')
    except (AttributeError, OSError, SyntaxError, TypeError):
        return False
    return (last_post is not None and isinstance(last_post.targets[0], ast.Name)
            and last_post.targets[0].id == 'hidden_states'
            and inner_model is not None and isinstance(inner_model.targets[0], ast.Attribute)
            and inner_model.targets[0].attr == 'model'
            and isinstance(inner_model.targets[0].value, ast.Name)
            and inner_model.targets[0].value.id == 'self')


@lru_cache(maxsize=8)
@torch.inference_mode()
def _qualify_post(pid, device, original):
    from hcu_adapter.kernels import mhc_post_5120_triton as post

    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return False
        try:
            generator = torch.Generator(device=device).manual_seed(81217)
            for rows in (1, 8, 32, 48, 64, 192):
                def normal(shape, dtype=torch.bfloat16):
                    return torch.randn(shape, device=device, dtype=dtype, generator=generator)

                x = normal((rows, 5120))
                residual = normal((rows, 4, 5120))
                mix = normal((rows, 4, 1), torch.float32)
                comb = normal((rows, 4, 4), torch.float32)
                if not post.supported(x, residual, mix, comb):
                    return False
                expected = original(x, residual, mix, comb)
                actual = post.post_5120(x, residual, mix, comb)
                if (not isinstance(expected, torch.Tensor) or expected.shape != actual.shape
                        or expected.dtype != actual.dtype
                        or not torch.equal(expected.contiguous().view(torch.uint8), actual.view(torch.uint8))):
                    return False
        except (AttributeError, OSError, TypeError, ValueError, RuntimeError):
            return False
    logging.getLogger('vllm.' + __name__).info('HCU draft final Post admitted after native preflight')
    return True


@lru_cache(maxsize=1)
def implementation():
    from hcu_adapter.kernels import mhc_post_5120_triton as post
    from .adapter_api import adapter_module

    op_disabled = adapter_module('kill_switch').op_disabled
    from vllm.models.deepseek_v41.amd import dspark as native

    original_post = getattr(native, 'mhc_post_torch', None)
    try:
        inspect.signature(original_post).bind(None, None, None, None)
    except (ValueError, TypeError) as error:
        raise DraftPostUnavailable("native DSpark Post call signature changed") from error
    if not all(isinstance(getattr(native, name, None), type) for name in (
        'DSparkDeepseekV4Model', 'DSparkDeepseekV4ForCausalLM'
    )):
        raise DraftPostUnavailable("native DSpark model class is missing")
    if not supports_operator_contract(native, 'dsv41_draft_final_mhc_post'):
        if (not _has_final_post_seam(native, original_post) or not torch.cuda.is_available()
                or not _qualify_post(os.getpid(), torch.device('cuda', torch.cuda.current_device()),
                                     original_post)):
            raise DraftPostUnavailable("native DSpark final Post could not be qualified")

    def final_post(x, residual, post_mix, comb):
        if not op_disabled('vllm_mhc') and post.supported(x, residual, post_mix, comb):
            return post.post_5120(x, residual, post_mix, comb)
        return original_post(x, residual, post_mix, comb)

    forward, forward_anchor = _clone(native.DSparkDeepseekV4Model, "forward",
        {"mhc_post_torch": final_post})

    class DraftModel(native.DSparkDeepseekV4Model):
        pass

    DraftModel.forward = forward
    outer, outer_anchor = _clone(native.DSparkDeepseekV4ForCausalLM, "__init__",
        {"DSparkDeepseekV4Model": DraftModel})
    return DraftModel, outer, (forward_anchor, outer_anchor,
                               (native, "mhc_post_torch", original_post))


def initialize_owned(model, *, vllm_config, prefix):
    _, constructor, anchors = implementation()
    if any(getattr(owner, name) is not original for owner, name, original in anchors):
        raise RuntimeError("Owned DSpark native implementation identity changed")
    constructor(model, vllm_config=vllm_config, prefix=prefix)
    logging.getLogger("vllm." + __name__).info(
        "HCU W4A8 private draft final Post constructor admitted: H=5120 HC=4")
