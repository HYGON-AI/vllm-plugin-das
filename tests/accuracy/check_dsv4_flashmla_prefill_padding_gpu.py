# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Real HCU parity for the installed prefill adapter versus original AITER.

Run from the repository in the HCU vLLM/FlashMLA environment:
    python tests/accuracy/check_dsv4_flashmla_prefill_padding_gpu.py

SWA/C4A/C128A use the gathered single-head KV layout consumed by prefill.
This checks kernel parity, not the upstream gather/combiner loop.
"""

from __future__ import annotations

import os
from types import ModuleType

import torch
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_rocm_flashmla_sparse as patch


def main():
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("This parity check requires an HCU/ROCm GPU")
    os.environ["VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL"] = "1"
    os.environ["VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL_HEAD_PADDING"] = "1"
    # Save the unpatched AITER callable and install the production wrapper.
    module = ModuleType(patch.TARGET_MODULE)
    module.rocm_sparse_attn_prefill = rocm_sparse_attn_prefill
    assert patch._apply_prefill_to_module(module)
    adapter = module.rocm_sparse_attn_prefill
    assert adapter is not rocm_sparse_attn_prefill
    for heads in (16, 32, 96, 64, 128):
        for tokens in (17, 257):
            for name, compressed_len in (("swa", 0), ("c4a", 64), ("c128a", 32)):
                torch.manual_seed(20260923 + heads + tokens + compressed_len)
                width = compressed_len + 128
                q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
                kv = torch.randn(width, 1, 512, device="cuda", dtype=torch.bfloat16)
                sink = torch.randn(heads, device="cuda", dtype=torch.float32)
                indices = torch.arange(width, device="cuda", dtype=torch.int32).repeat(tokens, 1)
                lengths = 17 + torch.arange(tokens, device="cuda", dtype=torch.int32) % (width - 16)
                lengths[-1] = width
                indices[:, 5] = -1  # Interior holes must be masked by both kernels.
                indices.masked_fill_(
                    torch.arange(width, device="cuda")[None, :] >= lengths[:, None], -1
                )
                expected = torch.empty_like(q)
                actual = torch.empty_like(q)
                args = (q, kv, indices, lengths, 512 ** -0.5, 512, 448, 64, sink)
                rocm_sparse_attn_prefill(*args, expected)
                adapter(*args, actual)
                torch.cuda.synchronize()
                assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
                error = (actual.float() - expected.float()).abs()
                max_error, mean_error = error.max().item(), error.mean().item()
                print(name, heads, tokens, max_error, mean_error, flush=True)
                assert max_error <= 0.015625, (name, heads, tokens, max_error, mean_error)
    print("PREFILL PADDING CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
