#!/usr/bin/env python3
"""Non-CI benchmark for official, mixed, and fused-CK GDN prefill."""

import argparse
import importlib.metadata
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import aiter
import aiter.ops.fla as aiter_fla
import boltops.fla.gdn as boltops_gdn
import torch
import torch.nn.functional as F
from vllm.third_party.flash_linear_attention.ops import chunk as fla_chunk
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril

D = 128
CHUNK = 64
FLAGS = dict(
    use_g=True, use_gk=False, use_exp2=False, transpose_state=True,
    qk_l2norm=False, gate_in_kernel=False, has_dt_bias=False,
    beta_sigmoid_in_kernel=False, output_final_state=True,
    cp_context=False, cp_world_size=2,
)


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def metadata(length):
    cu = torch.tensor([0, length], device="cuda", dtype=torch.int64)
    indices = prepare_chunk_indices(cu, CHUNK)
    offsets = torch.zeros(1, device="cuda", dtype=torch.int64)
    return cu, indices, offsets


def inputs(length, seed):
    torch.manual_seed(seed)
    q = F.normalize(
        torch.randn(1, length, 2, D, device="cuda").float(), dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(torch.randn_like(q).float(), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, length, 8, D, device="cuda", dtype=torch.bfloat16)
    g = -0.01 - 0.02 * torch.rand(1, length, 8, device="cuda")
    beta = torch.sigmoid(torch.randn(1, length, 8, device="cuda"))
    state = torch.randn(1, 8, D, D, device="cuda") * 0.01
    return q, k, v, g, beta, state, *metadata(length)


def official(data):
    q, k, v, g, beta, state, cu, indices, offsets = data
    return fla_chunk.chunk_gated_delta_rule(
        q, k, v, g, beta, scale=D**-0.5, initial_state=state,
        output_final_state=True, cu_seqlens=cu, chunk_indices=indices,
        chunk_offsets=offsets, use_qk_l2norm_in_kernel=False,
    )


def mixed(data):
    q, k, v, g, beta, state, cu, indices, offsets = data
    gc = chunk_local_cumsum(g, chunk_size=CHUNK, cu_seqlens=cu,
                            chunk_indices=indices)
    a = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=gc, cu_seqlens=cu, chunk_indices=indices,
        output_dtype=torch.float32,
    )
    a = solve_tril(a, cu_seqlens=cu, chunk_indices=indices,
                   output_dtype=k.dtype)
    w, u = boltops_gdn.recompute_w_u_fwd(
        k, v, beta, gc, a, cu_seqlens=cu, chunk_indices=indices
    )
    h, v_new, final = aiter_fla.chunk_gated_delta_rule_fwd_vllm_hip_blockdim64(
        k, w, u, gc, None, state, None, True, CHUNK, True, cu, indices,
        offsets, False, True, None,
    )
    out = aiter_fla.chunk_fwd_o_vllm_hip_blockdim64(
        q, k, v_new, h, gc, None, D**-0.5, cu, CHUNK, indices,
        False, True, None,
    )
    return out, final


def ck(data, workspace):
    q, k, v, g, beta, state, *_ = data
    return aiter.ck_gdn_prefill(
        q[0], k[0], v[0], g[0], beta[0], workspace,
        initial_state=state, cu_seqlens=None, scale=D**-0.5,
        **FLAGS, cp_rank=1,
    )


def errors(got, ref):
    rows = {}
    for name, actual, expected in zip(("output", "state"), got, ref):
        if actual.ndim + 1 == expected.ndim and expected.shape[0] == 1:
            expected = expected[0]
        delta = (actual.float() - expected.float()).abs()
        ref_rms = expected.float().square().mean().sqrt().clamp_min(1e-8)
        rows[name] = {
            "finite": bool(torch.isfinite(actual).all().item()),
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
            "nrmse": (delta.square().mean().sqrt() / ref_rms).item(),
        }
    return rows


def timed(fn, repetitions):
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - start) * 1000
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples)
    return {
        "cold_ms": cold_ms,
        "p50_ms": statistics.median(samples),
        "p90_ms": ordered[min(len(ordered) - 1,
                              math.ceil(0.9 * len(ordered)) - 1)],
        "mean_ms": statistics.mean(samples),
        "samples_ms": samples,
    }


def run(length, seed, repetitions):
    data = inputs(length, seed)
    q, _, v, *rest = data
    del rest
    workspace_bytes = int(aiter.ck_gdn_prefill_workspace_bytes(
        q[0], v[0], None, **FLAGS
    ))
    workspace = torch.empty(workspace_bytes, device="cuda", dtype=torch.uint8)
    ref = official(data)
    torch.cuda.synchronize()
    outputs = {"official": ref, "mixed": mixed(data),
               "ck": ck(data, workspace)}
    torch.cuda.synchronize()
    providers = {}
    functions = {
        "official": lambda: official(data),
        "mixed": lambda: mixed(data),
        "ck": lambda: ck(data, workspace),
    }
    for name, fn in functions.items():
        stats = timed(fn, repetitions)
        stats["tokens_per_s"] = length * 1000 / stats["mean_ms"]
        stats["vs_official"] = errors(outputs[name], ref)
        providers[name] = stats
    return {"length": length, "seed": seed,
            "workspace_bytes": workspace_bytes, "providers": providers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="1,64,65,128,512,1024,2048,4096")
    parser.add_argument("--seeds", default="7,17,29")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "versions": {
            "torch": torch.__version__, "vllm": version("vllm"),
            "vllm_hcu": version("vllm-hcu"), "aiter": version("aiter"),
            "boltops": version("boltops"),
            "plugin_git": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, text=True,
                capture_output=True,
            ).stdout.strip(),
        },
        "device": torch.cuda.get_device_name(0),
        "cases": [],
    }
    for seed in map(int, args.seeds.split(",")):
        for length in map(int, args.lengths.split(",")):
            print(f"benchmark length={length} seed={seed}", flush=True)
            result["cases"].append(run(length, seed, args.repetitions))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
