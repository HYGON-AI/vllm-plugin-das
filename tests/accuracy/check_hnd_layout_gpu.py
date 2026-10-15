# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Single-device layout regression; no checkpoint downloads or server changes.

Run in the matching gfx938 image with this checkout on PYTHONPATH. Results
cover single-head MLA cache views, decode kernels and graph replay;
they do not certify full-model accuracy or distributed KV transfers.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import traceback
from types import SimpleNamespace

import torch

from vllm.config import CacheConfig, get_current_vllm_config_or_none
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    resolve_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import (
    KVCacheTensor,
    MLAAttentionSpec,
    create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm_hcu.v1.attention import kv_cache_layout as layout_state


LAYOUTS = {"NHD": KVCacheLayout.LBNHC, "HND": KVCacheLayout.LBHNC}


def cache_view(spec, blocks, layout):
    size = blocks * spec.page_size_bytes
    raw = torch.zeros(size, dtype=torch.int8, device="cuda")
    placement = KVCacheTensor(
        size=size,
        layers=["attention"],
        layer_stride=size,
        block_stride=spec.page_size_bytes,
    )
    return create_kv_cache_views(raw, spec, blocks, LAYOUTS[layout], placement)[0]


def compare_outputs(result, reference, *, quantized=False):
    # FP8 kernels are compared with the quantized-input reference. These
    # bounds cover kernel accumulation and output rounding, not model accuracy.
    tolerance = 0.04 if quantized else 0.02
    torch.testing.assert_close(
        result.float(), reference.float(), rtol=tolerance, atol=tolerance
    )
    return float((result.float() - reference.float()).abs().max())


def benchmark_pair(functions, repeats, rounds):
    measurements = {name: [] for name in functions}
    for name, function in functions.items():
        layout_state._worker_kv_cache_layout = name
        for _ in range(5):
            function()
    torch.cuda.synchronize()
    # Alternate order to reduce systematic warmup / clock bias.
    for turn in range(rounds):
        order = list(functions) if turn % 2 == 0 else list(reversed(functions))
        for name in order:
            layout_state._worker_kv_cache_layout = name
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(repeats):
                functions[name]()
            end.record()
            end.synchronize()
            measurements[name].append(begin.elapsed_time(end) / repeats)
    medians = {name: statistics.median(v) for name, v in measurements.items()}
    return {
        "samples_ms": measurements,
        "median_ms": medians,
        "hnd_over_nhd": medians["HND"] / medians["NHD"],
    }


def check_pair(name, functions, reference, views, args, *, quantized=False):
    row = {
        "case": name,
        "strides": {k: list(v.stride()) for k, v in views.items()},
        "errors": {},
    }

    def check_comparison(layout, phase, result, expected):
        # Only numerical assertions are recoverable. Native calls, graph
        # capture and other runtime failures must still stop the GPU run.
        try:
            return compare_outputs(result, expected, quantized=quantized)
        except AssertionError:
            row["errors"].setdefault(layout, {})[phase] = traceback.format_exc()
            return None

    outputs = {}
    for layout, function in functions.items():
        print(
            json.dumps({"case": name, "layout": layout, "status": "running"}),
            flush=True,
        )
        layout_state._worker_kv_cache_layout = layout
        outputs[layout] = function().clone()
        row[f"{layout}_max_error"] = check_comparison(
            layout, "eager_reference", outputs[layout], reference
        )
        for _ in range(3):
            function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = function()
        graph.replay()
        torch.cuda.synchronize()
        row[f"{layout}_graph_max_error"] = check_comparison(
            layout, "graph_reference", captured, reference
        )
    row["layout_max_difference"] = check_comparison(
        "HND_vs_NHD", "eager_layouts", outputs["HND"], outputs["NHD"]
    )
    if row["errors"]:
        row["status"] = "failed"
        row["performance_status"] = "skipped_accuracy_failure"
        row["performance_review_required"] = None
        print(json.dumps(row), flush=True)
        return row
    row.update(benchmark_pair(functions, args.repeats, args.rounds))
    row["performance_review_required"] = row["hnd_over_nhd"] > args.max_ratio
    row["status"] = (
        "performance_review_required"
        if row["performance_review_required"]
        else "passed"
    )
    row["performance_status"] = row["status"]
    print(json.dumps(row), flush=True)
    return row


def mla_case(sparse, cache_dtype, args):
    name = (
        f"{'FLASHMLA_SPARSE' if sparse else 'FLASHMLA'}/"
        f"{cache_dtype if cache_dtype != 'auto' else 'bf16'}"
    )
    print(json.dumps({"case": name, "status": "running"}), flush=True)
    from vllm_hcu.v1.attention.ops import flashmla

    batch, length, heads, width, value_width, block_size = 4, 1023, 64, 576, 512, 64
    blocks_per_seq = (length + block_size - 1) // block_size
    blocks = batch * blocks_per_seq
    quantized = cache_dtype != "auto"
    dtype = {
        "auto": torch.bfloat16,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }[cache_dtype]
    spec = MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=width, dtype=dtype
    )
    source = torch.randn(
        blocks, 1, block_size, width, device="cuda", dtype=torch.bfloat16
    ).to(dtype)
    query = torch.randn(batch, 1, heads, width, device="cuda", dtype=torch.bfloat16)
    if cache_dtype == "fp8_e4m3":
        query = query.to(dtype)
    table = torch.arange(blocks, device="cuda", dtype=torch.int32).view(batch, -1)
    lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    topk = 128
    indices = torch.arange(topk, device="cuda", dtype=torch.int32)[
        None, None, :
    ].repeat(batch, 1, 1)
    indices += (
        torch.arange(batch, device="cuda", dtype=torch.int32)[:, None, None]
        * blocks_per_seq
        * block_size
    )
    topk_length = torch.full((batch,), topk, device="cuda", dtype=torch.int32)
    functions, views = {}, {}
    for layout in LAYOUTS:
        view = cache_view(spec, blocks, layout)
        view.copy_(source)
        views[layout] = view
        if sparse:
            cache = view.transpose(1, 2).reshape(-1, 1, width)

            def run(cache=cache):
                result = flashmla.flash_mla_sparse_fwd(
                    query.squeeze(1),
                    cache,
                    indices,
                    width**-0.5,
                    d_v=value_width,
                    topk_length=topk_length,
                )
                return result[0].unsqueeze(1)
        elif quantized:
            cache = view.transpose(1, 2)
            scheduler, splits = flashmla.get_mla_metadata_dense_fp8(lengths, heads, 1)
            scale = torch.ones(1, device="cuda", dtype=torch.float32)

            def run(cache=cache, scheduler=scheduler, splits=splits):
                return flashmla.flash_mla_with_kvcache_fp8(
                    query,
                    cache,
                    table,
                    lengths,
                    value_width,
                    scheduler,
                    splits,
                    softmax_scale=width**-0.5,
                    causal=True,
                    descale_q=None if cache_dtype == "fp8_e5m2" else scale,
                    descale_k=scale,
                    kv_cache_dtype=cache_dtype,
                )[0]
        else:
            cache = view.transpose(1, 2)

            def run(cache=cache):
                # A fresh scheduler mirrors the decode builder and ensures
                # graph capture records planning instead of reusing a plan.
                scheduler, _ = flashmla.get_mla_metadata(
                    lengths, heads, 1, is_fp8_kvcache=False
                )
                return flashmla.flash_mla_with_kvcache(
                    query,
                    cache,
                    table,
                    lengths,
                    value_width,
                    scheduler,
                    softmax_scale=width**-0.5,
                    causal=True,
                    is_fp8_kvcache=False,
                )[0]

        functions[layout] = run
    # NHD/HND must remain byte-equivalent for these single-head MLA views.
    assert torch.equal(views["NHD"].view(torch.uint8), views["HND"].view(torch.uint8))
    keys = source.float().reshape(batch, blocks_per_seq * block_size, width)
    keys = keys[:, : topk if sparse else length]
    scores = torch.einsum("bqhd,bkd->bhqk", query.float(), keys) * width**-0.5
    reference = torch.einsum(
        "bhqk,bkd->bqhd", scores.softmax(-1), keys[..., :value_width]
    )
    return check_pair(
        name,
        functions,
        reference,
        views,
        args,
        quantized=quantized,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-ratio", type=float, default=1.15)
    parser.add_argument("--expect-arch", choices=("gfx938",))
    parser.add_argument(
        "--backend",
        choices=("all", "FLASHMLA", "FLASHMLA_SPARSE"),
        default="all",
        help="Run one backend family in a fresh process, or all families (default)",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.rounds < 3:
        parser.error("repeats must be positive and rounds must be at least 3")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Select exactly one idle gfx938 GPU using HIP_VISIBLE_DEVICES"
        )
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx938":
        raise RuntimeError(f"This MLA regression supports gfx938 only, got {arch}")
    if args.expect_arch is not None and arch != args.expect_arch:
        raise RuntimeError(f"Expected {args.expect_arch}, got {arch}")
    if os.environ.get("VLLM_KV_CACHE_LAYOUT") is not None:
        raise RuntimeError("Unset VLLM_KV_CACHE_LAYOUT for this test process")
    if get_current_vllm_config_or_none() is not None:
        raise RuntimeError(
            "Run this regression in a fresh process without an engine config"
        )
    torch.manual_seed(20261015)
    report = {
        "arch": arch,
        "backend": args.backend,
        "device": torch.cuda.get_device_name(0),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "vllm", "vllm_hcu")
        },
        "plugin_source": str(Path(layout_state.__file__).resolve()),
        "scope": "gfx938 single-head MLA decode kernel/graph regression; not full-model accuracy",
        "not_covered": [
            "gfx936",
            "ordinary attention",
            "distributed KV transfer",
            "full models",
            "packed sparse FP8 kernel",
        ],
        "cases": [],
        "status": "running",
    }
    original_layout = layout_state._worker_kv_cache_layout
    try:
        report["hardware_default_resolution"] = []
        for family, module_name, class_name in (
            (
                "FLASHMLA",
                "vllm_hcu.v1.attention.backends.mla.flashmla",
                "HcuFlashMLABackend",
            ),
            (
                "FLASHMLA_SPARSE",
                "vllm_hcu.v1.attention.backends.mla.flashmla_sparse",
                "HcuFlashMLASparseBackend",
            ),
        ):
            if args.backend not in ("all", family):
                continue
            backend = getattr(importlib.import_module(module_name), class_name)
            supported = get_supported_kv_cache_layouts([backend])
            config = SimpleNamespace(
                cache_config=CacheConfig(block_size=64), kv_transfer_config=None
            )
            resolved = resolve_kv_cache_layout(config, [[v.name for v in supported]])
            if resolved is not KVCacheLayout.LBHNC:
                raise AssertionError(
                    f"{class_name} did not default to HND on real {arch}"
                )
            report["hardware_default_resolution"].append(
                {"backend": class_name, "resolved": resolved.name}
            )
        fp8_cache_dtype = "fp8_e4m3"
        for sparse, cache_dtype in (
            (False, "auto"),
            (False, fp8_cache_dtype),
            (True, "auto"),
        ):
            family = "FLASHMLA_SPARSE" if sparse else "FLASHMLA"
            if args.backend not in ("all", family):
                continue
            report["cases"].append(mla_case(sparse, cache_dtype, args))
        report["failed_cases"] = [
            row["case"] for row in report["cases"] if row["status"] == "failed"
        ]
        if report["failed_cases"]:
            raise AssertionError(
                "Numerical validation failed in "
                f"{len(report['failed_cases'])} case(s): "
                + ", ".join(report["failed_cases"])
                + ". See the per-layout errors in the report."
            )
        review = any(row["performance_review_required"] for row in report["cases"])
        report["status"] = "performance_review_required" if review else "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        layout_state._worker_kv_cache_layout = original_layout
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report: {args.output} ({report['status']})", flush=True)


if __name__ == "__main__":
    main()
