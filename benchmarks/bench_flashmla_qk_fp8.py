# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Single-device sparse prefill accuracy/timing; no model or HTTP service.

HIP_VISIBLE_DEVICES=0 python benchmarks/bench_flashmla_qk_fp8.py --output result.json
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import torch


def load_adapter():
    # Load the complete production adapter without bootstrapping a model/runtime.
    path = Path(__file__).resolve().parents[1] / (
        "vllm_hcu/v1/attention/ops/flashmla_qk_fp8.py"
    )
    sys.path.insert(0, str(path.parents[4]))
    spec = importlib.util.spec_from_file_location("hcu_qk_fp8_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(q_len, kv_len, heads, seed=20261114):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        q_len, heads, 576, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    kv = torch.randn(
        kv_len, 1, 576, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    indices = torch.randint(
        kv_len, (q_len, 1, 2048), device="cuda", dtype=torch.int32, generator=generator
    )
    counts = torch.full((q_len,), 2048, device="cuda", dtype=torch.int32)
    return q, kv, indices, counts


def reference(q, kv, indices, counts=None, attn_sink=None):
    """Independent FP32 softmax attention, including valid-count/index masks."""
    selected = indices[:, 0].long()
    valid = (selected >= 0) & (selected < kv.shape[0])
    if counts is not None:
        valid &= (
            torch.arange(selected.shape[1], device=q.device)[None] < counts[:, None]
        )
    keys = kv[:, 0].float()[selected.clamp(0, kv.shape[0] - 1)]
    scores = torch.einsum("qhd,qkd->qhk", q.float(), keys) * 576**-0.5
    scores.masked_fill_(~valid[:, None], float("-inf"))
    lse = torch.logsumexp(scores, -1)
    if attn_sink is None:
        probs = scores.softmax(-1)
    else:
        # Independent oracle: append the virtual zero-value token to logits,
        # rather than reusing the implementation's sigmoid correction.
        logits = torch.cat(
            (scores, attn_sink[None, :, None].expand(q.shape[0], -1, 1)), dim=-1
        )
        probs = logits.softmax(-1)[..., : selected.shape[1]]
        probs = torch.where(torch.isposinf(attn_sink)[None, :, None], 0.0, probs)
    output = torch.einsum("qhk,qkd->qhd", probs, keys[..., :512])
    return output, lse


def errors(output, lse, expected, expected_lse):
    delta = output.float() - expected
    return {
        "max_abs": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (delta.norm() / expected.norm().clamp_min(1e-12)).item(),
        "lse_max_abs": (lse.float() - expected_lse).abs().max().item(),
        "finite": bool(torch.isfinite(output).all() & torch.isfinite(lse).all()),
    }


def timing(fn, warmup, repeats, trials=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gpu, wall = [], []
    for _ in range(trials):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        t0 = perf_counter()
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        gpu.append(start.elapsed_time(end) / repeats)
        wall.append((perf_counter() - t0) * 1000 / repeats)
    return {"event_ms": median(gpu), "wall_ms": median(wall), "trials_event_ms": gpu}


def run_case(adapter, shape, warmup, repeats, profile=False, with_sink=False):
    from flash_mla.flash_mla_interface import flash_mla_cuda, flash_mla_sparse_fwd

    q, kv, indices, counts = inputs(*shape)
    scale = 576**-0.5
    sink = (
        torch.linspace(4, 11, q.shape[1], device=q.device, dtype=torch.float32)
        if with_sink
        else None
    )
    # Explicit non-None counts force the same wheel's masked BF16 path.
    bf16 = lambda: flash_mla_sparse_fwd(q, kv, indices, scale, 512, sink, counts)
    direct = lambda: flash_mla_cuda.sparse_prefill_fwd(
        q, kv, indices, scale, 512, None, None
    )

    def checked():
        out, lse = adapter.sparse_prefill_qk_fp8(
            q, kv, indices, scale, topk_length=counts
        )
        if sink is not None:
            out = adapter._apply_sink(out, lse, sink)
        return out, lse

    adaptive = lambda: adapter.flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sink, topk_length=counts
    )
    expected, expected_lse = reference(
        q[:4], kv, indices[:4], counts[:4], attn_sink=sink
    )
    base_out, _, base_lse = bf16()
    output, lse = checked()
    adaptive_out, adaptive_max, adaptive_lse = adaptive()
    assert bool(torch.isfinite(adaptive_out).all() & torch.isfinite(adaptive_lse).all())
    adaptive_error = errors(adaptive_out[:4], adaptive_lse[:4], expected, expected_lse)
    assert adaptive_error["finite"] and adaptive_error["relative_l2"] < 0.05
    assert adaptive_error["max_abs"] < 0.05 and adaptive_error["lse_max_abs"] < 0.05
    assert bool(torch.isfinite(output).all() & torch.isfinite(lse).all())
    error = errors(output[:4], lse[:4], expected, expected_lse)
    assert (
        error["finite"]
        and error["relative_l2"] < 0.05
        and error["max_abs"] < 0.05
        and error["lse_max_abs"] < 0.05
    ), error
    result = {
        "q_shape": list(q.shape),
        "kv_shape": list(kv.shape),
        "indices_shape": list(indices.shape),
        "topk_length_shape": list(counts.shape),
        "q_dtype": str(q.dtype),
        "has_sink": with_sink,
        "raw_timing_excludes_sink_correction": with_sink,
        "kv_dtype": str(kv.dtype),
        "fp8_vs_fp32": error,
        "bf16_vs_fp32": errors(base_out[:4], base_lse[:4], expected, expected_lse),
        "sampled_reference_rows": min(4, q.shape[0]),
        "bf16": timing(bf16, warmup, repeats),
        "qk_fp8_direct": timing(direct, warmup, repeats),
        "qk_fp8_adapter": timing(checked, warmup, repeats),
        "adaptive": timing(adaptive, warmup, repeats),
        "adaptive_dispatch": "qk_fp8" if adaptive_max is None else "bf16",
        "adaptive_vs_fp32": adaptive_error,
    }
    result["adapter_speedup"] = (
        result["bf16"]["wall_ms"] / result["qk_fp8_adapter"]["wall_ms"]
    )
    result["direct_speedup"] = (
        result["bf16"]["event_ms"] / result["qk_fp8_direct"]["event_ms"]
    )
    if profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            adaptive()
            torch.cuda.synchronize()
        result["device_kernels"] = sorted(
            {e.name for e in prof.events() if str(e.device_type).endswith("CUDA")}
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--sink", action="store_true", help="include the HYV4 per-head attention sink"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="sweep query/head work versus full-KV preprocessing cost",
    )
    args = parser.parse_args()
    os.environ["FLASH_MLA_DSA_FP8_QK"] = "1"
    os.environ.pop("FLASH_MLA_DISABLE_DSA_MLS_PREFILL", None)
    props = torch.cuda.get_device_properties(0)
    assert "gfx938" in props.gcnArchName
    adapter = load_adapter()
    shapes = [(16, 2048, 64), (378, 85000, 64), (6168, 177280, 64), (378, 85000, 128)]
    if args.sweep:
        shapes = [
            (q, kv, h)
            for h in (64, 128)
            for kv in (24576, 85000, 177280)
            for q in (16, 64, 128, 256, 512, 1024)
        ]
    if args.quick:
        shapes = shapes[:2]
    result = {
        "flash_mla": importlib.metadata.version("flash-mla"),
        "torch": torch.__version__,
        "arch": props.gcnArchName,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": [],
    }
    for shape in shapes:
        row = run_case(
            adapter,
            shape,
            args.warmup,
            args.repeats,
            profile=(shape[0] == 6168 if args.sink else not result["cases"]),
            with_sink=args.sink,
        )
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
