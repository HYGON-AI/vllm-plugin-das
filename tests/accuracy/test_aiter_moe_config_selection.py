# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Print AITER's automatic backend choice for QGPT GPTQ shapes.

Run directly with ``python test_aiter_moe_config_selection.py --tokens 1 32``
or through ``pytest -s``. This only queries the selector; it does not load
model weights or execute the selected MoE kernels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import pytest
import torch

TOKEN_COUNTS = (1, 4, 32, 64, 256, 2048, 8192)


def select_and_print(num_tokens: int, use_shuffle: int = 0, tp: int = 1):
    from aiter.moe import MoeQuantType, get_aiter_moe_config

    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if use_shuffle not in (0, 1):
        raise ValueError("use_shuffle must be 0 or 1")
    if tp not in (1, 2):
        raise ValueError("tp must be 1 or 2")
    problem = dict(
        M=num_tokens,
        E=256,
        N1=1024 // tp,
        N2=2048,
        K=2048,
        top_k=8,
        block_size=128,
        dtype=torch.bfloat16,
        quant_type=MoeQuantType.W4A16,
        activation="silu",
        use_shuffle=use_shuffle,
    )
    # Intentionally omit spec_sol_type: AITER chooses the backend itself.
    status, config = get_aiter_moe_config(**problem)
    print(json.dumps(
        {
            "selection": "auto",
            "tensor_parallel_size": tp,
            "problem": problem,
            "status": status,
            "selected_backend": getattr(config, "solution_type", None),
            "aiter_config": asdict(config) if config is not None else None,
        },
        default=str,
        ensure_ascii=False,
        indent=2,
    ), flush=True)
    return status, config


@pytest.mark.hcu
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("use_shuffle", [0, 1])
@pytest.mark.parametrize("tp", [1, 2])
def test_aiter_automatic_moe_config(num_tokens, use_shuffle, tp):
    if not torch.cuda.is_available():
        pytest.skip("HCU is unavailable")
    status, config = select_and_print(num_tokens, use_shuffle, tp)
    assert status is True, f"AITER has no solution for M={num_tokens}"
    assert config is not None
    # Do not pin Triton: future tuning tables may select ASM or MOE_C.
    assert config.solution_type in ("moe_c", "asm", "triton", "ck")
    assert isinstance(config.config, dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKEN_COUNTS)
    parser.add_argument("--use-shuffle", type=int, choices=(0, 1), default=0)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("An HCU/ROCm device is required for automatic selection")
    device = torch.cuda.get_device_properties(0)
    print(f"device={device.name}, arch={device.gcnArchName}", flush=True)
    for count in args.tokens:
        supported, selected = select_and_print(count, args.use_shuffle, args.tp)
        if not supported or selected is None or selected.solution_type is None:
            raise SystemExit(f"AITER has no solution for M={count}")
