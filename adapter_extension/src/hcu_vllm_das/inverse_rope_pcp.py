# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Replace the inverse RoPE call in Core 5afd482's PCP forward_mqa."""
from __future__ import annotations

from .adapter_api import adapter_module as _adapter_module

import ast
import functools
import hashlib
import inspect
import os
import textwrap
from types import FunctionType

registry = _adapter_module('registry')
from .inverse_rope_policy import selected_mode

_FORWARD_MQA_HASH = "6394555a5e86a54c484fc1dc08ac4dd525fa42f0881df164f078010b7f9868e6"
_state = None
_failure = None
_worker_routes = {}


def _require_source(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    if ast.get_docstring(node) is not None:
        node.body = node.body[1:]
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if digest != _FORWARD_MQA_HASH or function.__closure__:
        raise RuntimeError("V41 PCP forward_mqa changed; review the inverse RoPE call site")


def check_health():
    global _failure
    if _failure is not None:
        raise RuntimeError(_failure)
    if _state is None:
        return
    try:
        pid, base, original, _ = _state
        if (pid != os.getpid() or selected_mode() != "hcu"
                or base.forward_mqa is not original):
            raise RuntimeError("V41 PCP inverse RoPE policy/source changed; restart process")
    except BaseException as error:
        _failure = "V41 PCP inverse RoPE policy/source changed; restart process"
        raise RuntimeError(_failure) from error


def prepare_worker_model(model):
    """Qualify only models that instantiated the provider's PCP attention."""
    if _state is None:
        return
    from hcu_adapter.kernels.pcp_fork import build_classes

    attention_class = build_classes()["attention"]
    if any(isinstance(module, attention_class) for module in model.modules()):
        import torch

        prepare_worker_device(torch.device("cuda", torch.cuda.current_device()))


def prepare_worker_device(device):
    """Qualify the native operator on the device already bound to this worker."""
    check_health()
    if _state is None:
        raise RuntimeError("V41 PCP inverse RoPE did not bind the attention class")
    if device in _worker_routes:
        return
    from .inverse_rope_hcu import prepare_operator, route
    from vllm.logger import init_logger

    original_inverse = _state[2].__globals__["rocm_inverse_rope_rows_"]
    operator = prepare_operator(original_inverse, device)
    _worker_routes[device] = route(original_inverse, operator)
    init_logger(__name__).info("HCU V41 PCP inverse RoPE qualified on %s", device)


def _worker_inverse(output, positions, cos_sin_cache, rope_head_dim):
    inverse = _worker_routes.get(output.device)
    if inverse is None:
        raise RuntimeError(f"V41 PCP inverse RoPE was not qualified on {output.device}")
    return inverse(output, positions, cos_sin_cache, rope_head_dim)


def prepare_forward_mqa(base):
    """Bind the reviewed Python call site without running GPU work."""
    global _state
    check_health()
    if _state is not None:
        if _state[1] is not base:
            raise RuntimeError("V41 PCP attention base class changed")
        return _state[3]

    original = base.forward_mqa
    if selected_mode() == "stock":
        return original
    _require_source(original)
    arch_name = _adapter_module('hcu_env').arch_name
    from vllm.platforms import current_platform

    if not current_platform.is_rocm() or arch_name() != "gfx938":
        raise RuntimeError("V41 PCP inverse RoPE requires real gfx938 HIP")

    namespace = dict(original.__globals__)
    if not callable(namespace.get("rocm_inverse_rope_rows_")):
        raise RuntimeError("V41 PCP forward_mqa inverse RoPE call target is missing")
    namespace["rocm_inverse_rope_rows_"] = _worker_inverse
    run = FunctionType(original.__code__, namespace, original.__name__,
                       original.__defaults__, original.__closure__)
    run.__kwdefaults__ = original.__kwdefaults__

    @functools.wraps(original)
    def forward_mqa(self, q, kv, positions, output):
        check_health()
        return run(self, q, kv, positions, output)

    if selected_mode() != "hcu":
        raise RuntimeError("V41 PCP inverse RoPE policy changed during preparation")
    registry.register(op="vllm_inverse_rope", impl="hcu",
                      target=f"{base.__module__}.{base.__name__}.forward_mqa",
                      mechanism=registry.MECH_SEAM, source=__name__,
                      detail="HCU inverse RoPE in the owned PCP attention")
    _state = os.getpid(), base, original, forward_mqa
    return forward_mqa
