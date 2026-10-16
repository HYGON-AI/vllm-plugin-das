# SPDX-License-Identifier: Apache-2.0
# Ported from vllm-hcu-main 3d26cca (+ 02ec8ce): fuse the shared-expert gate
# path on HCU.  Decode-sized batches (rows <= 8) use the dedicated
# sigmoid-mul HIP kernel (fp16/bf16); larger prefill batches and other
# dtypes retain PyTorch's native vectorized implementation.

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_class,
    require_unpatched,
)

TARGET_MODULE = "vllm.model_executor.models.qwen2_moe"
PATCH_ID = "worker.op_opt.qwen2_moe.expert_gate"
TARGETS = (f"{TARGET_MODULE}.Qwen2MoeMLP.forward",)
_MARKER = "_vllm_hcu_qwen2_moe_expert_gate_applied"
_WRAPPER = "_vllm_hcu_qwen2_moe_expert_gate_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    import vllm_hcu.ops.sigmoid_mul  # noqa: F401  (op registration)
    qwen = load_exact_module(TARGET_MODULE, module)
    mlp_cls = require_class(qwen, "Qwen2MoeMLP", TARGETS[0])
    if getattr(qwen, _MARKER, False):
        if not getattr(mlp_cls.forward, _WRAPPER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale; "
                "restart the process"
            )
        return False
    original = require_unpatched(mlp_cls, "forward", TARGETS[0], _WRAPPER)
    # Qwen3NextMLP is a plain alias of Qwen2MoeMLP in v0.25.1, so this patch
    # covers the Qwen3.5 / Qwen3-Next shared-expert MLP as well.
    import torch.nn.functional as F  # noqa: F401

    from vllm_hcu.ops.sigmoid_mul import sigmoid_mul_hcu

    @functools.wraps(original)
    def hcu_forward(self, x):
        import vllm_hcu.platforms.envs as henvs

        if not henvs.VLLM_HCU_USE_CUSTOM_OPS:
            return original(self, x)
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)

        if self.expert_gate is not None:
            expert_gate = self.expert_gate(x)[0]
            # The dedicated kernel is optimized for decode-sized batches and
            # supports fp16/bf16; other dtypes keep the PyTorch path.
            if (
                expert_gate.shape[0] <= 8
                and expert_gate.dtype in (torch.float16, torch.bfloat16)
                and out.dtype == expert_gate.dtype
            ):
                out = sigmoid_mul_hcu(expert_gate, out)
            else:
                out = F.sigmoid(expert_gate) * out

        return out

    setattr(hcu_forward, _WRAPPER, True)
    qwen._vllm_hcu_original_qwen2_moe_mlp_forward = original
    mlp_cls.forward = hcu_forward
    setattr(qwen, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
