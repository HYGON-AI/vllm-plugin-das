#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only scratch/output lifetime regression using the exact source helpers.

The AST loader avoids importing the vLLM worker and its device bootstrap.
This covers byte-range ownership and GEMM2 cache reuse, not GPU MoE numerics
or full-model memory profiling. It never calls torch.cuda or creates a device
tensor; a Meta tensor only tests the cross-device early return.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


def _extract(source: str, name: str, torch):
    tree = ast.parse(source)
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise ValueError(f"Expected exactly one source function {name}")
    namespace = {"torch": torch}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, f"<exact-source:{name}>", "exec"), namespace)
    segment = ast.get_source_segment(source, functions[0])
    return namespace[name], hashlib.sha256(segment.encode()).hexdigest()


def run_checks(modular_source: str, ep_source: str) -> dict:
    import torch

    torch.set_num_threads(1)
    private_output, helper_sha = _extract(
        modular_source, "_step5_ep_private_output", torch)
    carve, carve_sha = _extract(ep_source, "_carve", torch)
    cases = []

    @contextmanager
    def track_allocations():
        original_empty, original_like = torch.empty, torch.empty_like
        counts = {"empty": 0, "empty_like": 0}

        def empty(*args, **kwargs):
            counts["empty"] += 1
            return original_empty(*args, **kwargs)

        def empty_like(*args, **kwargs):
            counts["empty_like"] += 1
            return original_like(*args, **kwargs)

        torch.empty, torch.empty_like = empty, empty_like
        try:
            yield counts
        finally:
            torch.empty, torch.empty_like = original_empty, original_like

    def record(name, function):
        try:
            details = function()
            cases.append(dict(case=name, passed=True, **(details or {})))
        except Exception as error:
            cases.append(dict(case=name, passed=False, error=repr(error)))

    def selection(workspace, output, should_allocate):
        with track_allocations() as allocations:
            result = private_output(workspace, output)
        assert result.shape == output.shape and result.dtype == output.dtype
        assert result.device == output.device
        assert allocations == {"empty": 0, "empty_like": int(should_allocate)}, allocations
        if should_allocate:
            assert result is not output and result.data_ptr() != output.data_ptr()
            # Write the expert result after allocation, then reuse the scratch
            # as the next operation would. The result must survive unchanged.
            expected = torch.arange(result.numel(), dtype=torch.float32).reshape(result.shape).to(result.dtype)
            result.copy_(expected)
            workspace.zero_()
            assert torch.equal(result, expected)
        else:
            assert result is output, "Disjoint/external outputs must be retained"
        return dict(private=should_allocate, allocation_calls=allocations)

    arena = torch.empty(256, dtype=torch.uint8)
    workspace = arena[64:128].view(torch.float32)
    selections = [
        ("same_pointer", workspace, arena[64:96].view(torch.float32), True),
        ("contained_nonzero_offset", workspace, arena[80:112].view(torch.float32), True),
        ("overlap_left_one_byte", workspace, arena[63:65], True),
        ("overlap_right_one_byte", workspace, arena[127:129], True),
        ("mixed_dtype_two_byte_overlap", workspace, arena[126:130].view(torch.bfloat16), True),
        ("same_storage_disjoint_after", workspace, arena[160:192].view(torch.float32), False),
        ("same_storage_disjoint_before", workspace, arena[0:32].view(torch.float32), False),
        ("touching_right_boundary", workspace, arena[128:160].view(torch.float32), False),
        ("touching_left_boundary", workspace, arena[32:64].view(torch.float32), False),
        ("external_output", workspace, torch.empty(16, dtype=torch.float32), False),
        ("empty_scratch", arena[64:64], arena[64:80], False),
        ("empty_output", workspace, arena[80:80], False),
        ("both_empty", arena[:0], arena[:0], False),
        ("noncontiguous_scratch", arena[64:128:2], arena[80:96], False),
        ("noncontiguous_output", workspace, arena[64:96:2], False),
        # Meta has no real device allocation; cross-device comparisons must
        # return before reading or comparing either data pointer.
        ("cpu_output_meta_scratch", torch.empty(16, device="meta"), torch.empty(16), False),
        ("meta_output_cpu_scratch", torch.empty(16), torch.empty(16, device="meta"), False),
    ]
    for name, scratch, output, allocate in selections:
        record(name, lambda s=scratch, o=output, a=allocate: selection(s, o, a))

    class NoPointerScratch:
        """Metadata-only stand-in; reading its address is always an error."""

        def __init__(self, device, elements):
            self.device, self.elements = device, elements

        def numel(self):
            return self.elements

        def is_contiguous(self):
            return True

        def data_ptr(self):
            raise AssertionError("An empty or different-device address was read")

    def no_pointer_read(device, elements):
        output = torch.empty(16)
        with track_allocations() as allocations:
            result = private_output(NoPointerScratch(device, elements), output)
        assert result is output and allocations == {"empty": 0, "empty_like": 0}
        return dict(no_address_read=True)

    record("empty_before_pointer_read", lambda: no_pointer_read(torch.device("cpu"), 0))
    # Constructing a Device value does not initialize CUDA or allocate a tensor.
    record("cross_device_before_pointer_read", lambda: no_pointer_read(torch.device("cuda", 1), 16))

    def poison_lifetime(dtype):
        scratch = torch.empty((64, 16), dtype=dtype)
        old_output = scratch.view(-1)[:8 * 16].view(8, 16)
        with track_allocations() as allocations:
            result = private_output(scratch, old_output)
        expected = torch.arange(result.numel(), dtype=torch.float32).reshape(result.shape).to(dtype)
        result.copy_(expected)
        scratch.fill_(float("nan"))
        assert bool(torch.isnan(old_output).all()), "Control alias was not actually poisoned"
        assert torch.equal(result, expected), "Scratch reuse clobbered the private result"
        for value in (0., -777., float("nan")):
            scratch.fill_(value)
            assert torch.equal(result, expected)
        assert allocations == {"empty": 0, "empty_like": 1}
        return dict(dtype=str(dtype), original_alias_poisoned=True,
                    private_result_survived_scratch_reuse=True)

    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        record(f"poison_lifetime_{dtype}", lambda d=dtype: poison_lifetime(d))

    def gemm2_cache_reuse(dtype):
        # A small volume keeps this portable. The real EP layout relationship
        # is retained: scratch[M,topk,K] versus output[M,K], topk=8.
        m, topk, k = 29, 8, 16
        scratch = torch.empty((m * topk + 256, k), dtype=dtype)
        old_output = scratch.view(-1)[:m * k].view(m, k)
        with track_allocations() as control_allocations:
            duplicate = carve(scratch, (m, topk, k), dtype,
                              torch.device("cpu"), avoid=old_output)
        assert control_allocations == {"empty": 1, "empty_like": 0}
        assert duplicate.data_ptr() != scratch.data_ptr()
        del duplicate
        with track_allocations() as fixed_allocations:
            result = private_output(scratch, old_output)
            routes = carve(scratch, (m, topk, k), dtype,
                           torch.device("cpu"), avoid=result)
        assert fixed_allocations == {"empty": 0, "empty_like": 1}, fixed_allocations
        assert routes.data_ptr() == scratch.data_ptr(), "GEMM2 failed to reuse scratch"
        payload = (torch.arange(m * topk * k, dtype=torch.float32) % 17).reshape(m, topk, k).to(dtype)
        routes.copy_(payload)
        result.copy_(routes.float().sum(dim=1).to(dtype))
        expected = payload.float().sum(dim=1).to(dtype)
        assert torch.equal(result, expected)
        scratch.fill_(float("nan"))
        assert torch.equal(result, expected), "Later scratch overwrite clobbered finalized result"
        output_bytes = result.numel() * result.element_size()
        cache_bytes = routes.numel() * routes.element_size()
        assert cache_bytes == 8 * output_bytes
        return dict(dtype=str(dtype), old_alias_forced_duplicate=True,
                    cache3_reuses_workspace=True, route_sum_exact=True,
                    result_survives_later_scratch_poison=True,
                    old_duplicate_scratch_bytes=cache_bytes,
                    private_output_bytes=output_bytes,
                    fixed_allocation_calls=fixed_allocations)

    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        record(f"gemm2_reuse_{dtype}", lambda d=dtype: gemm2_cache_reuse(d))

    errors = sum(not row["passed"] for row in cases)
    return dict(complete=errors == 0, device="cpu", cuda_operations=0,
                errors=errors, checks=len(cases), cases=cases,
                helper_source_sha256=helper_sha, carve_source_sha256=carve_sha,
                scope="Exact helper and _carve byte ownership, allocation and poison lifetime only; not GPU MoE or full-model validation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    layer = args.source_root / "vllm_hcu/model_executor/layers/fused_moe"
    result = run_checks((layer / "modular_kernel.py").read_text(encoding="utf-8"),
                        (layer / "ep_fused_moe.py").read_text(encoding="utf-8"))
    if args.result:
        args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
