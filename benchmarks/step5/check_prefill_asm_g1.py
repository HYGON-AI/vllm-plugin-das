"""Proof of the normal preloaded ASM G1 path, using exact-M input storage.

Uses the actual normal cache-preparation API and normal ep flag dispatch.
Observers delegate unchanged implementations; no private ASM/activation math
is imported. Protocol-shaped fake owners exercise the public post-load API,
not the complete model_loader or a model throughput/quality benchmark.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import statistics
import traceback
import weakref
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("HCU_STEP5_TUNING_CONFIG"):
        parser.error("Set HCU_STEP5_TUNING_CONFIG to an external qualified tuning profile")
    assert args.run_gpu and not args.output.exists()
    from vllm_hcu.ops.step5.tuning import asm_recipe

    solution, persist = asm_recipe(22)
    previous_env = os.environ.get("HCU_STEP5_PREFILL_ASM_G1")
    os.environ["HCU_STEP5_PREFILL_ASM_G1"] = "1"
    report = dict(complete=False, scope="normal ASM G1 local complete-MoE proof; not distributed DBO/model TPS",
                  actual_normal_cache_API=True, complete_model_loader_exercised=False,
                  private_math_imported=False, weight_ring=8, local_sort_bm=256,
                  solution=solution, persist=persist, shuffle=1,
                  amplitudes=[1.0, 0.25, 0.0, 4.0], relative_l2_limit=0.005,
                  exact_M_AQ_AS_storage_required=True, logical_storage_only_not_memory_sanitizer=True,
                  checks=[], graph_checks=[], cache_checks=[])
    ep = asm_g1 = g2 = aiter = None
    originals = {}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def phase(value):
        report["phase"] = value
        save()
        print(json.dumps(dict(phase=value)), flush=True)

    try:
        phase("imports")
        import numpy as np
        import torch
        import triton
        import aiter
        from vllm.plugins import load_general_plugins
        from vllm_hcu.patch.worker import apply_worker_patches

        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        assert torch.cuda.mem_get_info()[0] > 60 * 1024**3
        load_general_plugins()
        apply_worker_patches()
        from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe as ep

        asm_g1 = importlib.import_module("vllm_hcu.ops.step5.prefill_asm_g1")
        g2 = importlib.import_module("vllm_hcu.ops.step5.prefill_g2")
        native_asm = importlib.import_module("aiter.jit.module_moe_asm")
        sources = {name: dict(path=module.__file__, sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
                   for name, module in (("ep", ep), ("normal_asm_g1", asm_g1), ("normal_g2", g2), ("aiter_native", native_asm))}
        package_root = Path(asm_g1.__file__).resolve().parents[2]
        native_libraries = list(package_root.glob("hcu_ops*.so"))
        assert len(native_libraries) == 1, ("Expected normal source-integrated hcu_ops", native_libraries)
        native_path = native_libraries[0]
        sources["normal_hcu_ops"] = dict(path=str(native_path), sha256=hashlib.sha256(native_path.read_bytes()).hexdigest())
        hook = package_root / "patch/worker/op_opt/patch_compressed_tensors_moe_w8a8_fp8.py"
        sources["normal_postload_hook"] = dict(path=str(hook), sha256=hashlib.sha256(hook.read_bytes()).hexdigest())
        report["sources"] = sources
        originals = {name: getattr(ep, name) for name in (
            "STEP5_PREFILL_ASM_G1", "STEP5_PREFILL_G2", "CHUNK", "DIRECT_MIN",
            "shared_add_provider", "direct_scratch", "ep_gather_sorted", "_gemm", "ep_swiglustep_quant")}
        originals["normal_try"] = asm_g1.try_step5_prefill_asm_g1
        originals["g2_try"] = g2.try_step5_prefill_g2
        originals["aiter_asm"] = aiter.asm_fmoe_a8
        meta = json.loads(args.fixture_meta.read_text())
        assert hashlib.sha256(args.fixture.read_bytes()).hexdigest() == meta["fixture_sha256"]
        with np.load(args.fixture) as archive:
            ids_cpu = archive["topk_ids"].copy()
        m, top_k = ids_cpu.shape
        e, n, k, global_e = 22, 3072, 4096, 352
        assert (m, top_k) == (65536, 8)
        offset = int(meta["rank_expert_start"])
        assert 0 <= offset <= global_e - e
        report["fixture"] = meta
        torch.manual_seed(451523)
        ep.CHUNK = ep.DIRECT_MIN = m
        ep.shared_add_provider = None
        ep.STEP5_PREFILL_G2 = True
        ep.STEP5_PREFILL_ASM_G1 = False
        ids = torch.as_tensor(ids_cpu.astype(np.int32), device="cuda")
        saved_ids = ids.clone()
        mapping = torch.full((global_e,), -1, device="cuda", dtype=torch.int32)
        mapping[offset:offset + e] = torch.arange(e, device="cuda", dtype=torch.int32)
        full_back = torch.empty(2 * m + 256, k, device="cuda", dtype=torch.bfloat16)
        full = full_back[128:-128]
        out = full[:m]
        ep.direct_scratch = full[m:]
        reference = torch.empty_like(out)
        rows = triton.cdiv(m * top_k + e * 255, 256) * 256
        ws13_size = max(rows * n * 2, m * top_k * k * 2)
        byte_guard = 128 * k * 2
        ws13_back = torch.empty(ws13_size + 2 * byte_guard, device="cuda", dtype=torch.uint8)
        ws13 = ws13_back[byte_guard:-byte_guard]
        ws2 = torch.empty(rows * (k + 4), device="cuda", dtype=torch.uint8)

        def owner(index, weight):
            layer = torch.nn.Module()
            layer.layer_name = f"model.layers.{index + 3}.moe.experts.routed_experts"
            layer.activation = "swiglustep"
            layer.register_parameter("w13_weight", torch.nn.Parameter(weight, requires_grad=False))
            moe = SimpleNamespace(tp_size=1, num_experts=352, num_local_experts=22,
                                  experts_per_token=8, is_lora_enabled=False, swiglu_limit=7.0)
            quant = SimpleNamespace(use_fp8_w8a8=True, per_act_token_quant=True,
                                    per_out_ch_quant=True, block_shape=None, gemm1_clamp_limit=7.0)
            method = SimpleNamespace(moe=moe, moe_quant_config=quant)
            context = dict(is_step5=True, is_draft=False, tp_size=1, dp_size=16,
                           enable_ep=True, enable_eplb=False, layer_name=layer.layer_name)
            return layer, method, context

        ring = []
        phase("actual-normal-cache-prepare-eight-layers")
        cache_allocated = 0
        for index in range(8):
            raw = (torch.randn(e, n, k, device="cuda", dtype=torch.bfloat16) * 32).to(ep.FP8)
            layer, method, context = owner(index, raw)
            del raw
            raw = layer.w13_weight
            raw_before = raw.view(torch.uint8).clone()
            version_before = int(raw._version)
            allocated_before = int(torch.cuda.memory_allocated())
            assert asm_g1.prepare_prefill_asm_g1_cache(method, layer, context) is True
            record = getattr(layer, "_hcu_step5_prefill_asm_g1_cache")
            packed = getattr(layer, "_hcu_step5_prefill_asm_g1_weight")
            assert packed.data_ptr() != raw.data_ptr() and packed.shape == raw.shape
            assert packed.dtype == raw.dtype and packed.is_contiguous()
            assert record.source_ref() is raw and record.packed is packed
            assert torch.equal(raw.view(torch.uint8), raw_before) and int(raw._version) == version_before
            assert "_hcu_step5_prefill_asm_g1_weight" in dict(layer.named_buffers())
            assert "_hcu_step5_prefill_asm_g1_weight" not in layer.state_dict()
            assert asm_g1._lookup(raw.detach()) is record, "Same-pointer alias must resolve while source lives"
            packed_bytes = packed.numel() * packed.element_size()
            cache_allocated += packed_bytes
            report["cache_checks"].append(dict(index=index, layer=layer.layer_name,
                bytes=packed_bytes, raw_ptr=raw.data_ptr(), packed_ptr=packed.data_ptr(),
                allocation_delta=int(torch.cuda.memory_allocated()) - allocated_before,
                raw_source_all_bytes_unchanged=True, registered_nonpersistent_buffer=True,
                live_source_alias_lookup=True, source_version=version_before))
            del raw_before, record, packed
            w2 = (torch.randn(e, k, n // 2, device="cuda", dtype=torch.bfloat16) * 32).to(ep.FP8)
            s1 = torch.rand(e, n, 1, device="cuda") * 0.0002 + 0.0005
            s2 = torch.rand(e, k, 1, device="cuda") * 0.0002 + 0.0005
            # Direct allocations, no slicing from M+256 backing storage.
            aq = torch.empty(m, k, device="cuda", dtype=ep.FP8)
            aq.copy_((torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 32).clamp(-448, 448).to(ep.FP8))
            asc = torch.empty(m, 1, device="cuda", dtype=torch.float32)
            asc.uniform_(0.015, 0.020)
            assert aq.storage_offset() == asc.storage_offset() == 0
            assert aq.untyped_storage().nbytes() == m * k
            assert asc.untyped_storage().nbytes() == m * 4
            weights = torch.rand(m, top_k, device="cuda")
            weights.div_(weights.sum(1, keepdim=True))
            ring.append((layer, method, context, aq, asc, w2, s1, s2, weights))
        assert cache_allocated == 8 * e * n * k
        report.update(cache_layers=8, cache_bytes=cache_allocated, cache_GiB=cache_allocated / 1024**3,
                      actual_AQ_storage_bytes=m*k, actual_AS_storage_bytes=m*4)
        del raw

        phase("cache-guard-contracts")
        # Guard tests call the real registry/prepare API. No fake packed record
        # or alternate implementation is injected into the normal helper.
        contracts = {}
        layer0, method0, owner0 = ring[0][:3]
        missing = torch.nn.Parameter(layer0.w13_weight.detach().clone(), requires_grad=False)
        try:
            asm_g1._lookup(missing)
        except RuntimeError as error:
            assert "cache" in str(error).lower()
            contracts["unprepared_source_rejected"] = True
        else:
            raise AssertionError("Unprepared weight was accepted")
        del missing
        with torch.no_grad():
            layer0.w13_weight.copy_(layer0.w13_weight)
        try:
            asm_g1._lookup(layer0.w13_weight)
        except RuntimeError as error:
            assert "stale" in str(error).lower()
            contracts["version_change_rejected"] = True
        else:
            raise AssertionError("Stale version was accepted")
        assert asm_g1.prepare_prefill_asm_g1_cache(method0, layer0, owner0)
        contracts["normal_prepare_refreshes_reload"] = True
        extra, extra_method, extra_owner = owner(80, layer0.w13_weight.detach().clone())
        assert asm_g1.prepare_prefill_asm_g1_cache(extra_method, extra, extra_owner)
        old_parameter = extra.w13_weight
        dead_ref = weakref.ref(old_parameter)
        alias = old_parameter.detach()
        old_pointer = old_parameter.data_ptr()
        extra.w13_weight = torch.nn.Parameter(alias, requires_grad=False)
        del old_parameter
        gc.collect()
        assert dead_ref() is None and extra.w13_weight.data_ptr() == old_pointer
        try:
            asm_g1._lookup(extra.w13_weight)
        except RuntimeError as error:
            assert "stale" in str(error).lower()
            contracts["dead_source_same_pointer_rejected"] = True
        else:
            raise AssertionError("Cache survived loss of its source Parameter")
        assert not asm_g1.prepare_prefill_asm_g1_cache(extra_method, extra, dict(extra_owner, is_draft=True))
        contracts["draft_owner_not_cached"] = True
        del extra, extra_method, extra_owner, alias
        gc.collect()
        dummy_ids = torch.empty((256,), device="cuda", dtype=torch.int32)
        dummy_weights = torch.empty((256,), device="cuda", dtype=torch.float32)
        dummy_eids = torch.empty((1,), device="cuda", dtype=torch.int32)
        dummy_nv = torch.empty((1,), device="cuda", dtype=torch.int32)
        cache_view = ws13[:rows*n*2].view(torch.bfloat16).view(rows, n)
        assert originals["normal_try"](
            ring[0][3], layer0.w13_weight, ring[0][5], cache_view, ring[0][4], ring[0][6],
            dummy_ids, dummy_weights, dummy_eids, dummy_nv, num_tokens=m-1, top_k=top_k,
            sort_block_m=256, limit=7.0, workspace2=ws2) is None
        contracts["unsupported_token_count_returns_none"] = True
        del dummy_ids, dummy_weights, dummy_eids, dummy_nv, cache_view
        report["cache_guard_contracts"] = contracts
        resident = sum(getattr(entry[0], "_hcu_step5_prefill_asm_g1_weight").numel() for entry in ring)
        assert resident == cache_allocated
        report["active_cache_records"] = [dict(index=index, source_ptr=entry[0].w13_weight.data_ptr(),
            packed_ptr=getattr(entry[0], "_hcu_step5_prefill_asm_g1_weight").data_ptr(),
            source_version=int(entry[0].w13_weight._version)) for index, entry in enumerate(ring)]

        configs = copy.deepcopy(ep.pick_configs(layer0.w13_weight, ring[0][5], top_k, m))
        assert configs[0]["BLOCK_SIZE_M"] == configs[1]["BLOCK_SIZE_M"] == 256
        assert configs[0].get("a_presort", False)
        candidate = False
        current_index = 0
        counts = {path: dict(helper_try=0, helper_hit=0, asm_call=0, gather=0,
                            native_g1=0, native_activation=0, g2_try=0, g2_accepted=0)
                  for path in ("native", "normal_asm")}
        last_gates = {}
        packed_owner = {getattr(entry[0], "_hcu_step5_prefill_asm_g1_weight").data_ptr(): index
                        for index, entry in enumerate(ring)}

        def path():
            return "normal_asm" if candidate else "native"

        def observe_try(*positional, **keywords):
            assert candidate
            counts[path()]["helper_try"] += 1
            result = originals["normal_try"](*positional, **keywords)
            assert result is not None, "Supported normal helper silently missed"
            counts[path()]["helper_hit"] += 1
            return result

        def observe_asm(*positional, **keywords):
            assert candidate and not keywords and len(positional) == 17
            assert positional[12:] == (4, solution, 1, persist, 1)
            layer, method, context, aq, asc, w2, s1, s2, weights = ring[current_index]
            assert packed_owner[positional[2].data_ptr()] == current_index
            assert positional[1].data_ptr() == aq.data_ptr() and positional[9].data_ptr() == asc.data_ptr()
            assert positional[1].untyped_storage().nbytes() == m*k
            assert positional[9].untyped_storage().nbytes() == m*4
            assert positional[3].data_ptr() == w2.data_ptr() and positional[10].data_ptr() == s1.data_ptr()
            result = originals["aiter_asm"](*positional, **keywords)
            counts[path()]["asm_call"] += 1
            return result

        def observe_gather(*positional, **keywords):
            counts[path()]["gather"] += 1
            return originals["ep_gather_sorted"](*positional, **keywords)

        def observe_gemm(a, w, *positional, **keywords):
            if tuple(w.shape) == (e, n, k):
                counts[path()]["native_g1"] += 1
            elif tuple(w.shape) == (e, k, n//2):
                raise AssertionError("Unchanged fused G2 must be accepted")
            return originals["_gemm"](a, w, *positional, **keywords)

        def observe_activation(*positional, **keywords):
            counts[path()]["native_activation"] += 1
            return originals["ep_swiglustep_quant"](*positional, **keywords)

        def observe_g2(*positional, **keywords):
            counts[path()]["g2_try"] += 1
            last_gates[path()] = positional[9]
            assert originals["g2_try"](*positional, **keywords) is True
            counts[path()]["g2_accepted"] += 1
            return True

        asm_g1.try_step5_prefill_asm_g1 = observe_try
        aiter.asm_fmoe_a8 = observe_asm
        ep.ep_gather_sorted = observe_gather
        ep._gemm = observe_gemm
        ep.ep_swiglustep_quant = observe_activation
        g2.try_step5_prefill_g2 = observe_g2

        def call(index, optimized, poison=False):
            nonlocal candidate, current_index
            candidate, current_index = optimized, index
            ep.STEP5_PREFILL_ASM_G1 = optimized
            assert ep.STEP5_PREFILL_G2 is True
            if poison:
                full_back.fill_(float("nan"))
                ws13_back.view(torch.bfloat16).fill_(float("nan"))
                ws2.fill_(255)
            layer, method, context, aq, asc, w2, s1, s2, weights = ring[index]
            ep.ep_fused_experts(aq, layer.w13_weight, w2, weights, ids, mapping, s1, s2,
                               a1_scale=asc, global_num_experts=global_e,
                               output_dtype=torch.bfloat16, routed_scaling_factor=1.0,
                               configs=configs, out=out, workspace13=ws13, workspace2=ws2)

        def compare(tag, reference_gate):
            assert bool(torch.isfinite(reference).all() and torch.isfinite(out).all())
            error = float((out.float()-reference.float()).norm() / reference.float().norm().clamp_min(1.0e-12))
            gate = [int(value) for value in last_gates["normal_asm"].tolist()]
            assert error < 0.005 and gate == reference_gate, (tag, error, gate, reference_gate)
            assert bool(torch.isnan(full_back[:128]).all() and torch.isnan(full_back[-128:]).all())
            assert bool(torch.isnan(ws13_back[:byte_guard].view(torch.bfloat16)).all())
            assert bool(torch.isnan(ws13_back[-byte_guard:].view(torch.bfloat16)).all())
            return dict(tag=tag, relative_l2=error, bit_exact=torch.equal(out, reference),
                        direct_rows=gate[0], fallback_rows=gate[1], overflow=gate[2], guards=True)

        def check_pair(index, tag):
            before = copy.deepcopy(counts)
            call(index, False, True)
            torch.cuda.synchronize()
            reference.copy_(out)
            native_gate = [int(value) for value in last_gates["native"].tolist()]
            call(index, True, True)
            torch.cuda.synchronize()
            result = compare(tag, native_gate)
            expected = {
                "native": dict(gather=1, native_g1=1, native_activation=1, g2_try=1, g2_accepted=1),
                "normal_asm": dict(helper_try=1, helper_hit=1, asm_call=1, g2_try=1, g2_accepted=1),
            }
            for name, required in expected.items():
                delta = {key: counts[name][key] - before[name][key] for key in counts[name]}
                assert delta == {key: required.get(key, 0) for key in counts[name]}, (tag, name, delta)
            return result

        phase("normal-dispatch-four-amplitudes-eight-pairs")
        for generation, amplitude in enumerate(report["amplitudes"]):
            for index, entry in enumerate(ring):
                layer, method, context, aq, asc, w2, s1, s2, weights = entry
                aq.copy_((torch.randn_like(aq, dtype=torch.bfloat16)*32*amplitude).clamp(-448, 448).to(ep.FP8))
                asc.mul_(1.015625)
                weights.uniform_()
                weights.div_(weights.sum(1, keepdim=True))
                check = check_pair(index, f"g{generation}-pair{index}")
                assert check["direct_rows"] > 0 and check["fallback_rows"] == 0
                if amplitude == 0.0:
                    assert check["bit_exact"] and bool((out == 0).all()) and bool((reference == 0).all())
                report["checks"].append(check)
                save()
        for name in ("empty", "overflow"):
            if name == "overflow":
                ids.copy_((torch.arange(m, device="cuda")[:, None]+torch.arange(top_k, device="cuda")[None, :]) % e+offset)
            else:
                remote = (torch.arange(top_k, device="cuda", dtype=torch.int32)+offset+e) % global_e
                ids.copy_(remote[None, :].expand(m, top_k))
            check = check_pair(0, name)
            assert check["direct_rows"] == 0
            assert (check["fallback_rows"] > 0) == (name == "overflow")
            if name == "empty":
                assert check["bit_exact"] and bool((out == 0).all())
            report["checks"].append(check)
        ids.copy_(saved_ids)
        for layer, method, context, aq, asc, w2, s1, s2, weights in ring:
            aq.copy_((torch.randn_like(aq, dtype=torch.bfloat16)*32).clamp(-448, 448).to(ep.FP8))
        for optimized in (False, True):
            for index in range(8):
                call(index, optimized)
        torch.cuda.synchronize()

        phase("eight-pair-normal-graph-numerical")
        for index in range(8):
            pair_graphs = {}
            for name in ("native", "normal_asm"):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call(index, name == "normal_asm")
                pair_graphs[name] = graph
            full_back.fill_(float("nan")); ws13_back.view(torch.bfloat16).fill_(float("nan"))
            pair_graphs["native"].replay(); torch.cuda.synchronize()
            reference.copy_(out)
            native_gate = [int(value) for value in last_gates["native"].tolist()]
            full_back.fill_(float("nan")); ws13_back.view(torch.bfloat16).fill_(float("nan"))
            pair_graphs["normal_asm"].replay(); torch.cuda.synchronize()
            report["graph_checks"].append(compare(f"graph-pair{index}", native_gate))
            del pair_graphs, graph
        save()

        def eight(optimized):
            for index in range(8):
                call(index, optimized)

        def measure(function):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); function(); end.record(); end.synchronize()
            return start.elapsed_time(end)/8

        phase("paired-normal-integrated-timing")
        samples = dict(native=[], normal_asm=[])
        for repetition in range(11):
            for name in list(samples) if repetition % 2 else reversed(list(samples)):
                samples[name].append(measure(lambda: eight(name == "normal_asm")))
        report.update(samples_ms=samples, median_ms={key: statistics.median(values) for key, values in samples.items()})
        report["ratio"] = report["median_ms"]["normal_asm"] / report["median_ms"]["native"]
        graphs = {}
        for name in ("native", "normal_asm"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                eight(name == "normal_asm")
            graphs[name] = graph
        graphs["native"].replay(); torch.cuda.synchronize()
        reference.copy_(out)
        native_gate = [int(value) for value in last_gates["native"].tolist()]
        graphs["normal_asm"].replay(); torch.cuda.synchronize()
        report["ring_graph_check"] = compare("eight-weight-ring-last-pair", native_gate)
        samples = dict(native=[], normal_asm=[])
        for repetition in range(11):
            for name in list(samples) if repetition % 2 else reversed(list(samples)):
                samples[name].append(measure(graphs[name].replay))
        report.update(graph_samples_ms=samples, graph_median_ms={key: statistics.median(values) for key, values in samples.items()},
                      dispatch_counts=counts, actual_cache_used_layers=sum(bool(getattr(entry[0], "_hcu_step5_prefill_asm_g1_cache").used) for entry in ring),
                      current_cache_bytes=sum(getattr(entry[0], "_hcu_step5_prefill_asm_g1_weight").numel() for entry in ring),
                      complete=True, phase="complete")
        assert report["actual_cache_used_layers"] == 8 and report["current_cache_bytes"] == cache_allocated
        report["graph_ratio"] = report["graph_median_ms"]["normal_asm"] / report["graph_median_ms"]["native"]
        save()
        print(json.dumps({key: report[key] for key in ("complete", "median_ms", "ratio", "graph_median_ms", "graph_ratio", "cache_bytes")}), flush=True)
        return 0
    except BaseException:
        report.update(error=traceback.format_exc(), failed_phase=report.get("phase"), context_health_unknown=True)
        save()
        print(report["error"], flush=True)
        return 2
    finally:
        if ep is not None:
            for name, value in originals.items():
                if name not in ("normal_try", "g2_try", "aiter_asm"):
                    setattr(ep, name, value)
        if asm_g1 is not None and "normal_try" in originals:
            asm_g1.try_step5_prefill_asm_g1 = originals["normal_try"]
        if g2 is not None and "g2_try" in originals:
            g2.try_step5_prefill_g2 = originals["g2_try"]
        if aiter is not None and "aiter_asm" in originals:
            aiter.asm_fmoe_a8 = originals["aiter_asm"]
        if previous_env is None:
            os.environ.pop("HCU_STEP5_PREFILL_ASM_G1", None)
        else:
            os.environ["HCU_STEP5_PREFILL_ASM_G1"] = previous_env


if __name__ == "__main__":
    raise SystemExit(main())
