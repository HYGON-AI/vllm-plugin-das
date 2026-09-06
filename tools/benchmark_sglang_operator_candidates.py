#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Benchmark candidate LightOp/AITER routes against current vLLM paths."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Robust summary of repeated device timings in microseconds."""

    samples_us: tuple[float, ...]
    median_us: float
    minimum_us: float
    maximum_us: float


def _positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def summarize_timings(samples_us: Sequence[float]) -> TimingSummary:
    """Validate and summarize repeated timing samples."""

    samples = tuple(float(sample) for sample in samples_us)
    if not samples or not all(_positive_finite(sample) for sample in samples):
        raise ValueError("timing samples must be positive finite values")
    return TimingSummary(
        samples_us=samples,
        median_us=float(statistics.median(samples)),
        minimum_us=min(samples),
        maximum_us=max(samples),
    )


def speedup_percent(baseline_us: float, candidate_us: float) -> float:
    """Return candidate latency improvement relative to the baseline."""

    baseline = float(baseline_us)
    candidate = float(candidate_us)
    if not _positive_finite(baseline) or not _positive_finite(candidate):
        raise ValueError("baseline and candidate timings must be positive finite values")
    return (baseline - candidate) / baseline * 100.0


def build_comparison_record(
    *,
    shape: dict[str, int | bool | float],
    baseline_name: str,
    baseline: TimingSummary,
    candidate_name: str,
    candidate: TimingSummary,
) -> dict[str, object]:
    """Build one JSON-safe performance comparison record."""

    speedup = speedup_percent(baseline.median_us, candidate.median_us)
    return {
        "shape": dict(shape),
        "baseline": {"name": baseline_name, **asdict(baseline)},
        "candidate": {"name": candidate_name, **asdict(candidate)},
        "speedup_percent": speedup,
        "meets_five_percent_gate": speedup >= 5.0,
    }


def _measure_cuda(
    function,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> TimingSummary:
    import torch

    samples = []
    for _ in range(repeats):
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        torch.cuda.synchronize()
        elapsed_us = (time.perf_counter_ns() - started) / 1_000.0
        samples.append(elapsed_us / iterations)
    return summarize_timings(samples)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--json-output", type=Path)


def build_parser() -> argparse.ArgumentParser:
    """Build the operator benchmark command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operator", required=True)

    sqrtsoftplus = subparsers.add_parser(
        "sqrtsoftplus-gate",
        help="Compare LightOp sqrt-softplus routing with the vLLM path.",
    )
    _add_common_arguments(sqrtsoftplus)
    sqrtsoftplus.add_argument("--experts", type=int, nargs="+", default=[256, 384])
    sqrtsoftplus.add_argument("--top-k", type=int, nargs="+", default=[6, 8])
    sqrtsoftplus.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 1024],
    )

    silu = subparsers.add_parser(
        "silu-and-mul",
        help="Compare AITER, LightOp, and vLLM gated-SiLU paths.",
    )
    _add_common_arguments(silu)
    silu.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 2048])
    silu.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 1024],
    )

    w16a16 = subparsers.add_parser(
        "w16a16-moe",
        help="Compare LightOp W16A16 Marlin, AITER, and Triton MoE paths.",
    )
    _add_common_arguments(w16a16)
    w16a16.add_argument("--experts", type=int, default=256)
    w16a16.add_argument("--hidden-size", type=int, default=2048)
    w16a16.add_argument("--intermediate-size", type=int, default=512)
    w16a16.add_argument("--top-k", type=int, default=8)
    w16a16.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    mla_cat = subparsers.add_parser(
        "mla-decode-cat",
        help="Compare categorized LightOp MLA decode concat with torch.cat.",
    )
    _add_common_arguments(mla_cat)
    mla_cat.add_argument("--heads", type=int, nargs="+", default=[8, 16, 32])
    mla_cat.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768],
    )
    tgemm = subparsers.add_parser(
        "aiter-tgemm",
        help="Compare AITER tuned BF16 GEMM with the current torch linear path.",
    )
    _add_common_arguments(tgemm)
    tgemm.add_argument(
        "--dimensions",
        nargs="+",
        default=["4096x4096", "4096x1024", "2048x4096", "2048x512"],
        help="KxN GEMM dimensions.",
    )
    tgemm.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    return parser


def _validate_counts(args: argparse.Namespace) -> None:
    for name in ("warmup", "iterations", "repeats"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def _run_sqrtsoftplus(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import vllm
    from lightop import __version__ as lightop_version
    from vllm.model_executor.layers.fused_moe.router import (
        fused_topk_bias_router as official_module,
    )

    from vllm_hcu.model_executor.layers.fused_moe.sqrtsoftplus_routing import (
        run_lightop_sqrtsoftplus,
    )

    official = getattr(
        official_module,
        "_vllm_hcu_original_vllm_topk_softplus_sqrt",
        official_module.vllm_topk_softplus_sqrt,
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    records = []
    for experts in args.experts:
        for topk in args.top_k:
            if topk > experts:
                raise ValueError("--top-k cannot exceed --experts")
            for tokens in args.token_counts:
                logits = torch.randn(
                    (tokens, experts),
                    dtype=torch.float32,
                    device="cuda",
                    generator=generator,
                ).contiguous()
                bias = torch.randn(
                    (experts,),
                    dtype=torch.float32,
                    device="cuda",
                    generator=generator,
                ).contiguous()

                def baseline_call():
                    weights = torch.empty(
                        (tokens, topk), dtype=torch.float32, device="cuda"
                    )
                    ids = torch.empty(
                        (tokens, topk), dtype=torch.int32, device="cuda"
                    )
                    token_expert = torch.empty_like(ids)
                    return official(
                        weights,
                        ids,
                        token_expert,
                        logits,
                        True,
                        bias,
                        None,
                        None,
                        1.5,
                    )

                def candidate_call():
                    # vLLM allocates these buffers before entering the patched
                    # function. Include that cost even though LightOp returns
                    # its own tensors.
                    torch.empty((tokens, topk), dtype=torch.float32, device="cuda")
                    torch.empty((tokens, topk), dtype=torch.int32, device="cuda")
                    torch.empty((tokens, topk), dtype=torch.int32, device="cuda")
                    return run_lightop_sqrtsoftplus(
                        logits,
                        bias,
                        topk=topk,
                        renormalize=True,
                        routed_scaling_factor=1.5,
                        indices_dtype=torch.int32,
                    )

                baseline = _measure_cuda(
                    baseline_call,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
                candidate = _measure_cuda(
                    candidate_call,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
                records.append(
                    build_comparison_record(
                        shape={
                            "tokens": tokens,
                            "experts": experts,
                            "top_k": topk,
                            "renormalize": True,
                            "routed_scaling_factor": 1.5,
                        },
                        baseline_name="vllm-official",
                        baseline=baseline,
                        candidate_name="lightop",
                        candidate=candidate,
                    )
                )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "operator": "sqrtsoftplus-gate",
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "environment": {
            "device_name": properties.name,
            "gcn_arch": getattr(properties, "gcnArchName", None),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "lightop": lightop_version,
        },
        "records": records,
    }


def _run_w16a16(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import vllm
    from lightop import __version__ as lightop_version
    from vllm.model_executor.layers.fused_moe import fused_moe as fused_moe_module

    from vllm_hcu.model_executor.layers.fused_moe.lightop_w16a16_runtime import (
        pack_lightop_w16a16_weights,
        run_lightop_w16a16,
    )
    from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
        AiterMoeProblem,
        execute_aiter_moe,
        prepare_aiter_moe_weights,
        select_aiter_moe_config,
    )

    if args.top_k <= 0 or args.top_k > args.experts:
        raise ValueError("--top-k must be positive and cannot exceed --experts")
    if args.hidden_size % 32 or args.intermediate_size % 16:
        raise ValueError(
            "LightOp W16A16 requires --hidden-size divisible by 32 and "
            "--intermediate-size divisible by 16"
        )
    if any(tokens <= 0 for tokens in args.token_counts):
        raise ValueError("--token-counts must contain positive values")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    w13 = (
        torch.randn(
            (args.experts, 2 * args.intermediate_size, args.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    ).contiguous()
    w2 = (
        torch.randn(
            (args.experts, args.hidden_size, args.intermediate_size),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    ).contiguous()
    packed13, packed2, _ = pack_lightop_w16a16_weights(w13, w2)
    official_triton = getattr(
        fused_moe_module,
        "_vllm_hcu_original_fused_experts_impl",
        fused_moe_module.fused_experts_impl,
    )
    records = []
    for tokens in args.token_counts:
        hidden_states = torch.randn(
            (tokens, args.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        ).contiguous()
        topk_ids = torch.stack(
            [
                torch.randperm(
                    args.experts, device="cuda", generator=generator
                )[: args.top_k]
                for _ in range(tokens)
            ]
        ).to(torch.int32)
        weights = torch.rand(
            (tokens, args.top_k),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        topk_weights = weights / weights.sum(dim=-1, keepdim=True)
        workspace13 = torch.empty(
            (
                tokens * args.top_k,
                max(2 * args.intermediate_size, args.hidden_size),
            ),
            dtype=torch.bfloat16,
            device="cuda",
        )
        workspace2 = torch.empty(
            (tokens * args.top_k, args.intermediate_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        output = torch.empty_like(hidden_states)
        aiter_problem = AiterMoeProblem(
            M=tokens,
            E=args.experts,
            N1=2 * args.intermediate_size,
            N2=args.hidden_size,
            K=args.hidden_size,
            top_k=args.top_k,
            block_size=0,
            dtype=torch.bfloat16,
            device=hidden_states.device,
            quant_type="w16a16",
            activation="silu",
            use_shuffle=True,
        )
        aiter_config = select_aiter_moe_config(
            aiter_problem, cache_owner=w13
        )
        aiter_weights = (
            prepare_aiter_moe_weights(
                w13, w2, aiter_config, cache_owner=w13
            )
            if aiter_config is not None
            else None
        )

        def baseline_call():
            return official_triton(
                hidden_states,
                w13,
                w2,
                topk_weights,
                topk_ids,
                activation="silu",
            )

        def candidate_call():
            run_lightop_w16a16(
                output=output,
                hidden_states=hidden_states,
                w13=packed13,
                w2=packed2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                workspace13=workspace13,
                workspace2=workspace2,
                global_num_experts=args.experts,
            )
            return output

        def aiter_call():
            assert aiter_config is not None and aiter_weights is not None
            return execute_aiter_moe(
                aiter_config,
                hidden_states=hidden_states,
                w1=aiter_weights[0],
                w2=aiter_weights[1],
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                global_num_experts=args.experts,
                use_weight_shuffle=True,
                output_dtype=torch.bfloat16,
            )

        expected = baseline_call()
        actual = candidate_call()
        torch.cuda.synchronize()
        maximum_absolute_error = float(
            (actual.float() - expected.float()).abs().max().item()
        )
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
        baseline = _measure_cuda(
            baseline_call,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        candidate = _measure_cuda(
            candidate_call,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        record = build_comparison_record(
            shape={
                "tokens": tokens,
                "experts": args.experts,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "top_k": args.top_k,
            },
            baseline_name="vllm-triton",
            baseline=baseline,
            candidate_name="lightop-w16a16-marlin",
            candidate=candidate,
        )
        record["maximum_absolute_error_vs_triton"] = maximum_absolute_error
        records.append(record)
        if aiter_config is not None:
            aiter_output = aiter_call()
            torch.testing.assert_close(actual, aiter_output, rtol=0.03, atol=0.03)
            aiter = _measure_cuda(
                aiter_call,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            aiter_record = build_comparison_record(
                shape={
                    "tokens": tokens,
                    "experts": args.experts,
                    "hidden_size": args.hidden_size,
                    "intermediate_size": args.intermediate_size,
                    "top_k": args.top_k,
                },
                baseline_name="hcu-aiter",
                baseline=aiter,
                candidate_name="lightop-w16a16-marlin",
                candidate=candidate,
            )
            aiter_record["maximum_absolute_error_vs_aiter"] = float(
                (actual.float() - aiter_output.float()).abs().max().item()
            )
            records.append(aiter_record)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "operator": "w16a16-moe",
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "environment": {
            "device_name": properties.name,
            "gcn_arch": getattr(properties, "gcnArchName", None),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "lightop": lightop_version,
        },
        "records": records,
    }


def _run_silu_and_mul(args: argparse.Namespace) -> dict[str, object]:
    import aiter
    import torch
    import vllm
    from lightop import __version__ as lightop_version
    from lightop.activation import silu_and_mul_opt

    try:
        import vllm._C_stable_libtorch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("vLLM silu_and_mul extension is unavailable") from exc
    vllm_op = getattr(getattr(torch.ops, "_C", None), "silu_and_mul", None)
    if not callable(vllm_op):
        raise RuntimeError("vLLM _C.silu_and_mul is unavailable")
    if not callable(getattr(aiter, "silu_and_mul", None)):
        raise RuntimeError("AITER silu_and_mul is unavailable")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    records = []
    for hidden in args.hidden_sizes:
        if hidden <= 0:
            raise ValueError("--hidden-sizes must contain positive values")
        for tokens in args.token_counts:
            if tokens <= 0:
                raise ValueError("--token-counts must contain positive values")
            source = torch.randn(
                (tokens, 2 * hidden),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            ).contiguous()
            lightop_output = torch.empty(
                (tokens, hidden), dtype=source.dtype, device=source.device
            )
            aiter_output = torch.empty_like(lightop_output)
            vllm_output = torch.empty_like(lightop_output)

            def lightop_call():
                silu_and_mul_opt(lightop_output, source)
                return lightop_output

            def aiter_call():
                # The installed HCU AITER schema is (out, input).  Newer
                # SGLang builds pass an optional clamp-limit as a third
                # argument, which this pinned runtime does not expose.
                aiter.silu_and_mul(aiter_output, source)
                return aiter_output

            def vllm_call():
                vllm_op(vllm_output, source)
                return vllm_output

            lightop_call()
            aiter_call()
            vllm_call()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                aiter_output, lightop_output, rtol=0.02, atol=0.05
            )
            torch.testing.assert_close(
                aiter_output, vllm_output, rtol=0.02, atol=0.05
            )
            max_error_lightop = float(
                (aiter_output.float() - lightop_output.float()).abs().max().item()
            )
            max_error_vllm = float(
                (aiter_output.float() - vllm_output.float()).abs().max().item()
            )
            lightop = _measure_cuda(
                lightop_call,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            aiter_summary = _measure_cuda(
                aiter_call,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            vllm_summary = _measure_cuda(
                vllm_call,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            shape = {"tokens": tokens, "hidden_size": hidden, "dtype": "bf16"}
            lightop_record = build_comparison_record(
                shape=shape,
                baseline_name="current-plugin-lightop",
                baseline=lightop,
                candidate_name="aiter",
                candidate=aiter_summary,
            )
            lightop_record["maximum_absolute_error"] = max_error_lightop
            records.append(lightop_record)
            vllm_record = build_comparison_record(
                shape=shape,
                baseline_name="vllm-native",
                baseline=vllm_summary,
                candidate_name="aiter",
                candidate=aiter_summary,
            )
            vllm_record["maximum_absolute_error"] = max_error_vllm
            records.append(vllm_record)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "operator": "silu-and-mul",
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "environment": {
            "device_name": properties.name,
            "gcn_arch": getattr(properties, "gcnArchName", None),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "lightop": lightop_version,
            "aiter": getattr(aiter, "__version__", "unknown"),
        },
        "records": records,
    }


def _run_mla_decode_cat(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import vllm
    from lightop import __version__ as lightop_version

    from vllm_hcu.model_executor.layers.attention.lightop_concat_runtime import (
        _call_registered_lightop,
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    records = []
    for heads in args.heads:
        if heads <= 0 or heads % 8:
            raise ValueError("--heads must contain positive multiples of 8")
        for tokens in args.token_counts:
            if tokens <= 0 or tokens >= 1024:
                raise ValueError("--token-counts must be in the range [1, 1023]")

            # These are the non-contiguous views produced by FlashMLA decode,
            # rather than easier contiguous stand-ins.
            left_storage = torch.randn(
                tokens * heads * 512,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            left = torch.as_strided(
                left_storage,
                size=(tokens, heads, 512),
                stride=(512, 512 * tokens, 1),
            )
            right_storage = torch.randn(
                1536 * (heads // 8) * tokens,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            right = torch.as_strided(
                right_storage,
                size=(tokens, heads, 64),
                stride=(1536 * (heads // 8), 192, 1),
            )

            def baseline_call():
                return torch.cat((left, right), dim=-1)

            def candidate_call():
                return _call_registered_lightop(left, right)

            expected = baseline_call()
            actual = candidate_call()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            baseline = _measure_cuda(
                baseline_call,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            candidate = _measure_cuda(
                candidate_call,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            record = build_comparison_record(
                shape={"tokens": tokens, "heads": heads},
                baseline_name="torch-cat",
                baseline=baseline,
                candidate_name="lightop-ds-cat-mode-0",
                candidate=candidate,
            )
            record["maximum_absolute_error"] = float(
                (actual.float() - expected.float()).abs().max().item()
            )
            records.append(record)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "operator": "mla-decode-cat",
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "environment": {
            "device_name": properties.name,
            "gcn_arch": getattr(properties, "gcnArchName", None),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "lightop": lightop_version,
        },
        "records": records,
    }


def _run_aiter_tgemm(args: argparse.Namespace) -> dict[str, object]:
    import aiter
    import torch
    import vllm
    from aiter.tuned_gemm import tgemm

    dimensions = []
    for raw in args.dimensions:
        try:
            k, n = (int(value) for value in raw.lower().split("x", maxsplit=1))
        except (TypeError, ValueError) as exc:
            raise ValueError("--dimensions values must use positive KxN syntax") from exc
        if k <= 0 or n <= 0:
            raise ValueError("--dimensions values must use positive KxN syntax")
        dimensions.append((k, n))

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    records = []
    for k, n in dimensions:
        weight = (
            torch.randn(
                (n, k),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            * 0.02
        ).contiguous()
        for tokens in args.token_counts:
            if tokens <= 0:
                raise ValueError("--token-counts must contain positive values")
            source = torch.randn(
                (tokens, k),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            ).contiguous()

            def baseline_call():
                return torch.nn.functional.linear(source, weight)

            def candidate_call():
                return tgemm.mm(source, weight, otype=source.dtype)

            expected = baseline_call()
            actual = candidate_call()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.1)
            baseline = _measure_cuda(
                baseline_call,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            candidate = _measure_cuda(
                candidate_call,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            record = build_comparison_record(
                shape={"tokens": tokens, "input_size": k, "output_size": n},
                baseline_name="torch-linear",
                baseline=baseline,
                candidate_name="aiter-tgemm",
                candidate=candidate,
            )
            record["maximum_absolute_error"] = float(
                (actual.float() - expected.float()).abs().max().item()
            )
            records.append(record)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "operator": "aiter-tgemm",
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "environment": {
            "device_name": properties.name,
            "gcn_arch": getattr(properties, "gcnArchName", None),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "aiter": getattr(aiter, "__version__", "unknown"),
        },
        "records": records,
    }


def main() -> int:
    args = build_parser().parse_args()
    _validate_counts(args)
    if args.operator == "sqrtsoftplus-gate":
        report = _run_sqrtsoftplus(args)
    elif args.operator == "silu-and-mul":
        report = _run_silu_and_mul(args)
    elif args.operator == "w16a16-moe":
        report = _run_w16a16(args)
    elif args.operator == "mla-decode-cat":
        report = _run_mla_decode_cat(args)
    elif args.operator == "aiter-tgemm":
        report = _run_aiter_tgemm(args)
    else:
        raise RuntimeError(
            f"benchmark runner for {args.operator!r} has not passed its "
            "candidate-specific accuracy gate"
        )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
