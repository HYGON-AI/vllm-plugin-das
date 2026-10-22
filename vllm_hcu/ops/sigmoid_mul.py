# SPDX-License-Identifier: Apache-2.0

import torch

import vllm_hcu.hcu_ops as hcu_ops  # noqa: F401  (registers torch.ops.hcu_ops)

# The fused kernel supports fp16 and bf16; any other dtype takes the
# equivalent PyTorch expression instead of failing in the kernel's
# TORCH_CHECK.
_FUSED_DTYPES = (torch.float16, torch.bfloat16)

# Meta/Fake kernel so FakeTensor and torch.compile can propagate the output
# metadata of the C++ op (the extension only registers the CUDA kernel).
_LIB = torch.library.Library("hcu_ops", "FRAGMENT")


def _sigmoid_mul_fake(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    assert gate.dtype in _FUSED_DTYPES, "sigmoid_mul requires fp16 or bf16"
    assert gate.dtype == value.dtype, "sigmoid_mul dtypes must match"
    return torch.empty_like(value)


_LIB.impl("sigmoid_mul", _sigmoid_mul_fake, "Meta")


def sigmoid_mul_hcu(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Multiply ``value`` by a per-row sigmoid gate on HCU.

    The fused kernel supports fp16 and bf16; any other dtype (or a dtype
    mismatch) takes the equivalent PyTorch expression.
    """
    if (
        gate.dtype not in _FUSED_DTYPES
        or gate.dtype != value.dtype
    ):
        return torch.sigmoid(gate) * value
    return torch.ops.hcu_ops.sigmoid_mul(gate, value)
