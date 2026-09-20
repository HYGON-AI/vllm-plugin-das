# SPDX-License-Identifier: Apache-2.0
"""Keep Kimi LL stateful MoE eager between graph segments, including FULL."""
import functools

import torch
from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    is_breakable_cudagraph_enabled,
)
from vllm.utils.torch_utils import weak_ref_tensor


def uses_kimi_ll_graph_boundary(layer):
    method = getattr(layer, "_quant_method", None)
    method = getattr(method, "old_quant_method", method)
    return bool(getattr(method, "_hcu_kimi_ll_graph_boundary", False)
                and is_breakable_cudagraph_enabled())


def kimi_ll_graph_boundary(function):
    """Adapt the HCU runner's extended argument list to stable output buffers."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if not is_breakable_cudagraph_enabled():
            return function(*args, **kwargs)
        # Registered HCU MoE custom ops pass their schema arguments positionally.
        from vllm_hcu.model_executor.layers.fused_moe.moe_runner import (
            get_layer_from_name, _resolve_layer_name,
        )
        layer_name = args[8] if len(args) > 8 else kwargs["layer_name"]
        layer = get_layer_from_name(_resolve_layer_name(layer_name))
        if not uses_kimi_ll_graph_boundary(layer):
            return function(*args, **kwargs)

        shared_buffer = None

        def invoke(call_args, call_kwargs):
            nonlocal shared_buffer
            result = function(*call_args, **call_kwargs)
            hidden = call_args[0] if call_args else call_kwargs["hidden_states"]
            width = call_args[9] if len(call_args) > 9 else call_kwargs["hidden_dim_unpadded"]
            target = hidden[..., :width] if width > 0 else hidden
            if isinstance(result, tuple):
                # shared_experts_input is read-only in the custom-op schema.
                # Keep a separate output alive across eager graph replays.
                if shared_buffer is None:
                    shared_buffer = torch.empty_like(result[0])
                shared_buffer.copy_(result[0])
                target.copy_(result[1])
                return shared_buffer, hidden
            target.copy_(result)
            return hidden

        capture = BreakableCUDAGraphCapture.current()
        if capture is None or not capture._capturing:
            return invoke(args, kwargs)
        weak_args = tuple(weak_ref_tensor(x) if isinstance(x, torch.Tensor) else x for x in args)
        weak_kwargs = {k: weak_ref_tensor(v) if isinstance(v, torch.Tensor) else v
                       for k, v in kwargs.items()}
        # Unlike the paired vLLM decorator, deliberately break FULL graphs too.
        return capture.add_eager(lambda: invoke(weak_args, weak_kwargs))

    return wrapped
