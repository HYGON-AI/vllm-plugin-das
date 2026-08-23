#!/usr/bin/env python3
"""Compare official vLLM Triton and AITER ASM FP8 MoE stage boundaries.

This is a diagnostic, not a regression test. It loads one expert layer from a
real checkpoint, reuses identical activations/routing for both backends, and
captures the two quantizers, GEMMs, gated activation, and expert reduction.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class Metrics:
    max_abs: float
    mae: float
    nmae_percent: float
    cosine: float


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> Metrics:
    actual_f = actual.float().reshape(-1)
    reference_f = reference.float().reshape(-1)
    if actual_f.shape != reference_f.shape:
        raise ValueError(
            f"shape mismatch: actual={tuple(actual.shape)}, "
            f"reference={tuple(reference.shape)}"
        )
    difference = (actual_f - reference_f).abs()
    denominator = reference_f.abs().mean().clamp_min(1e-12)
    norm_product = actual_f.norm() * reference_f.norm()
    cosine = torch.dot(actual_f, reference_f) / norm_product.clamp_min(1e-12)
    return Metrics(
        max_abs=float(difference.max().item()),
        mae=float(difference.mean().item()),
        nmae_percent=float((difference.mean() / denominator).item() * 100.0),
        cosine=float(cosine.item()),
    )


def _clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def _load_tensor(path: Path, key: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _load_layer(
    model: Path,
    layer: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    keys = {
        "w1": f"{prefix}.gate_up_proj",
        "w1_scale": f"{prefix}.gate_up_proj_scale",
        "w2": f"{prefix}.down_proj",
        "w2_scale": f"{prefix}.down_proj_scale",
    }
    with (model / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle)["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    for name, key in keys.items():
        shard = model / weight_map[key]
        tensors[name] = _load_tensor(shard, key).to(device).contiguous()
    return tensors


@contextmanager
def _capture_official_stages():
    from vllm.model_executor.layers.fused_moe import fused_moe as module

    captures: dict[str, torch.Tensor] = {}
    originals = {
        "quant": module.moe_kernel_quantize_input,
        "gemm": module.dispatch_fused_moe_kernel,
        "activation": module.apply_moe_activation,
        "sum": module.ops.moe_sum,
    }
    counts = {"quant": 0, "gemm": 0}

    def quant(*args: Any, **kwargs: Any):
        result = originals["quant"](*args, **kwargs)
        counts["quant"] += 1
        captures[f"quant{counts['quant']}_value"] = _clone(result[0])
        if result[1] is not None:
            captures[f"quant{counts['quant']}_scale"] = _clone(result[1])
        return result

    def gemm(*args: Any, **kwargs: Any):
        result = originals["gemm"](*args, **kwargs)
        counts["gemm"] += 1
        captures[f"gemm{counts['gemm']}"] = _clone(args[2])
        return result

    def activation(*args: Any, **kwargs: Any):
        result = originals["activation"](*args, **kwargs)
        captures["activation"] = _clone(args[1])
        return result

    def moe_sum(*args: Any, **kwargs: Any):
        result = originals["sum"](*args, **kwargs)
        captures["combine"] = _clone(args[1])
        return result

    module.moe_kernel_quantize_input = quant
    module.dispatch_fused_moe_kernel = gemm
    module.apply_moe_activation = activation
    module.ops.moe_sum = moe_sum
    try:
        yield captures
    finally:
        module.moe_kernel_quantize_input = originals["quant"]
        module.dispatch_fused_moe_kernel = originals["gemm"]
        module.apply_moe_activation = originals["activation"]
        module.ops.moe_sum = originals["sum"]


@contextmanager
def _capture_aiter_stages(activation_mode: str, quant2_mode: str):
    import aiter
    from aiter import fused_moe_asm_wna16 as module

    captures: dict[str, torch.Tensor] = {}
    originals = {
        "quant": module.per_token_quant_hip,
        "gemm": aiter.asm_fmoe_a8,
        "activation": module._apply_activation,
        "sum": module.triton_moe_sum,
    }
    counts = {"quant": 0, "gemm": 0}

    def quant(*args: Any, **kwargs: Any):
        counts["quant"] += 1
        if counts["quant"] == 2 and quant2_mode == "vllm":
            from vllm import _custom_ops as ops

            result = ops.scaled_fp8_quant(
                args[0], None, use_per_token_if_dynamic=True
            )
        else:
            result = originals["quant"](*args, **kwargs)
        captures[f"quant{counts['quant']}_value"] = _clone(result[0])
        captures[f"quant{counts['quant']}_scale"] = _clone(result[1])
        return result

    def gemm(*args: Any, **kwargs: Any):
        result = originals["gemm"](*args, **kwargs)
        counts["gemm"] += 1
        captures[f"gemm{counts['gemm']}"] = _clone(args[0])
        return result

    def activation(*args: Any, **kwargs: Any):
        output = kwargs.get("activated_out", args[2] if len(args) > 2 else None)
        source = kwargs.get("ffn1_out_2d", args[3] if len(args) > 3 else None)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("could not capture AITER activation output")
        if activation_mode == "vllm":
            if not isinstance(source, torch.Tensor):
                raise RuntimeError("could not find AITER activation input")
            torch.ops._C.silu_and_mul(output, source)
            result = None
        else:
            result = originals["activation"](*args, **kwargs)
        captures["activation"] = _clone(output)
        return result

    def moe_sum(*args: Any, **kwargs: Any):
        result = originals["sum"](*args, **kwargs)
        captures["combine"] = _clone(args[1])
        return result

    module.per_token_quant_hip = quant
    aiter.asm_fmoe_a8 = gemm
    module._apply_activation = activation
    module.triton_moe_sum = moe_sum
    try:
        yield captures
    finally:
        module.per_token_quant_hip = originals["quant"]
        aiter.asm_fmoe_a8 = originals["gemm"]
        module._apply_activation = originals["activation"]
        module.triton_moe_sum = originals["sum"]


def _run_official(
    hidden: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl

    with _capture_official_stages() as captures:
        output = fused_experts_impl(
            hidden_states=hidden,
            w1=tensors["w1"],
            w2=tensors["w2"],
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="silu",
            use_fp8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=tensors["w1"].shape[0],
            w1_scale=tensors["w1_scale"],
            w2_scale=tensors["w2_scale"],
        )
    torch.cuda.synchronize()
    return output, captures


def _run_aiter(
    hidden: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    activation_mode: str,
    quant2_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], str]:
    from aiter.moe import (
        MoeQuantType,
        aiter_moe,
        aiter_moe_shfl_weight,
        get_aiter_moe_config,
    )
    status, config = get_aiter_moe_config(
        M=hidden.shape[0],
        E=tensors["w1"].shape[0],
        N1=tensors["w1"].shape[1],
        N2=tensors["w2"].shape[1],
        K=tensors["w1"].shape[2],
        top_k=topk_ids.shape[1],
        block_size=0,
        dtype=hidden.dtype,
        quant_type=MoeQuantType.FP8_W8A8,
        activation="silu",
    )
    if not status or config is None:
        raise RuntimeError("AITER returned no FP8 W8A8 configuration")
    solution = str(config.solution_type)
    if not solution.upper().endswith("ASM"):
        raise RuntimeError(f"expected ASM but selected {solution}")
    if bool(getattr(config, "need_shuffle", False)):
        w1, w2 = aiter_moe_shfl_weight(tensors["w1"], tensors["w2"], config)
    else:
        w1, w2 = tensors["w1"], tensors["w2"]
    with _capture_aiter_stages(activation_mode, quant2_mode) as captures:
        output = aiter_moe(
            hidden_states=hidden,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights.float(),
            topk_ids=topk_ids.int(),
            moe_config=config,
            inplace=False,
            activation="silu",
            w1_scale=tensors["w1_scale"],
            w2_scale=tensors["w2_scale"],
            global_num_experts=tensors["w1"].shape[0],
            use_weight_shuffle=bool(getattr(config, "need_shuffle", False)),
            output_dtype=hidden.dtype,
        )
    torch.cuda.synchronize()
    return output, captures, solution


def _compare_captures(
    official: dict[str, torch.Tensor],
    aiter: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    stages = (
        "quant1_value",
        "quant1_scale",
        "gemm1",
        "activation",
        "quant2_value",
        "quant2_scale",
        "gemm2",
        "combine",
    )
    comparison: dict[str, dict[str, Any]] = {}
    for stage in stages:
        official_value = official[stage]
        aiter_value = aiter[stage]
        comparison[stage] = {
            "official_shape": list(official_value.shape),
            "aiter_shape": list(aiter_value.shape),
            **asdict(_metrics(aiter_value, official_value)),
        }
    return comparison


def run_case(
    tokens: int,
    tensors: dict[str, torch.Tensor],
    seed: int,
    device: torch.device,
    activation_mode: str,
    quant2_mode: str,
) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + tokens)
    hidden_size = tensors["w1"].shape[2]
    num_experts = tensors["w1"].shape[0]
    top_k = 8
    hidden = torch.randn(
        (tokens, hidden_size),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    routing_logits = torch.randn(
        (tokens, num_experts),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    topk_logits, topk_ids = routing_logits.topk(top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)

    official_output, official_captures = _run_official(
        hidden, topk_weights, topk_ids, tensors
    )
    aiter_output, aiter_captures, solution = _run_aiter(
        hidden, topk_weights, topk_ids, tensors, activation_mode, quant2_mode
    )
    torch.testing.assert_close(official_captures["combine"], official_output)
    torch.testing.assert_close(aiter_captures["combine"], aiter_output)
    return {
        "tokens": tokens,
        "solution": solution,
        "aiter_activation": activation_mode,
        "aiter_quant2": quant2_mode,
        "output": asdict(_metrics(aiter_output, official_output)),
        "stages": _compare_captures(official_captures, aiter_captures),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/models/Qwen3.5-35B-A3B-CHANNEL-FP8"),
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 1024])
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--aiter-activation", choices=("native", "vllm"), default="native"
    )
    parser.add_argument(
        "--aiter-quant2", choices=("native", "vllm"), default="native"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA/ROCm GPU is required")
    device = torch.device("cuda:0")
    import vllm_hcu.ops  # noqa: F401 -- registers HCU vLLM custom ops

    tensors = _load_layer(args.model, args.layer, device)
    print(
        "loaded",
        {
            name: (tuple(value.shape), str(value.dtype))
            for name, value in tensors.items()
        },
        flush=True,
    )
    report = {
        "model": str(args.model),
        "layer": args.layer,
        "seed": args.seed,
        "cases": [],
    }
    for tokens in args.tokens:
        case = run_case(
            tokens,
            tensors,
            args.seed,
            device,
            args.aiter_activation,
            args.aiter_quant2,
        )
        report["cases"].append(case)
        print(json.dumps(case, indent=2), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
