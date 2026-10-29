"""Reproduce the warmed vLLM 0.32 operator takeover measurements.

Run this script in separate fresh processes with the Plugin switches OFF and
ON. It is a measurement harness, not a pytest test or correctness gate.
"""

from __future__ import annotations

import json
import os
import time

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.plugins import load_general_plugins


def _measure(function, arguments: tuple[object, ...], iterations: int):
    for _ in range(10):
        function(*arguments)
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(iterations):
            function(*arguments)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    return samples


def main() -> None:
    load_general_plugins()

    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm

    torch.manual_seed(20261029)
    config = VllmConfig()
    with set_current_vllm_config(config):
        silu = SiluAndMul()
        rms = RMSNorm(4096).to(device="cuda", dtype=torch.bfloat16)
        rms.weight.data.normal_(mean=1.0, std=0.1)

        silu_graph = torch.compile(lambda value: silu(value), dynamic=True)
        rms_graph = torch.compile(lambda value: rms(value), dynamic=True)
        fused_graph = torch.compile(
            lambda value, residual: rms(value, residual),
            dynamic=True,
        )

        results = {}
        for tokens, iterations in ((1, 200), (16, 100), (128, 50)):
            silu_input = torch.randn(
                (tokens, 24576),
                device="cuda",
                dtype=torch.bfloat16,
            )
            norm_input = torch.randn(
                (tokens, 4096),
                device="cuda",
                dtype=torch.bfloat16,
            )
            residual = torch.randn_like(norm_input)
            results[f"silu_t{tokens}_us"] = _measure(
                silu_graph,
                (silu_input,),
                iterations,
            )
            results[f"rms_t{tokens}_us"] = _measure(
                rms_graph,
                (norm_input,),
                iterations,
            )
            results[f"fused_rms_t{tokens}_us"] = _measure(
                fused_graph,
                (norm_input.clone(), residual.clone()),
                iterations,
            )

    report = {
        "mode": os.environ["HCU_BENCH_MODE"],
        "silu_forward": silu._forward_method.__name__,
        "rms_forward": rms._forward_method.__name__,
        "shapes": {
            "silu": ["tokens", 24576],
            "rms": ["tokens", 4096],
            "tokens": [1, 16, 128],
        },
        "iterations": {"1": 200, "16": 100, "128": 50},
        "warmup_iterations": 10,
        "measurements_per_case": 3,
        "results": results,
    }
    print("HCU_BENCH_RESULT=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
