"""Normal source G2 fusion: complete MoE and overflow/empty/metadata fallbacks.

No transport, model TPS, checkpoint, or installed-source mutation.
"""
import argparse, copy, hashlib, json, statistics, time, traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-gpu", action="store_true")
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--fixture-meta", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert a.run_gpu and not a.output.exists()
    report = dict(
        complete=False,
        scope="complete isolated MoE; only G2 launch/destination specialized; no model TPS",
        model_integrated=False,
        weight_ring=8,
        checks=[],
    )

    def save():
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    def phase(v):
        report["phase"] = v
        save()
        print(json.dumps(dict(phase=v)), flush=True)

    ep = None
    original = None
    try:
        phase("imports")
        import numpy as np
        import torch, triton, triton.language as tl

        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        assert torch.cuda.mem_get_info()[0] > 60 * 1024 ** 3
        from vllm.plugins import load_general_plugins
        from vllm_hcu.patch.worker import apply_worker_patches

        load_general_plugins()
        apply_worker_patches()
        from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe as ep
        from vllm_hcu.ops.step5 import prefill_g2 as g2

        assert hasattr(ep, "STEP5_PREFILL_G2")
        meta = json.loads(a.fixture_meta.read_text())
        assert (
            hashlib.sha256(a.fixture.read_bytes()).hexdigest() == meta["fixture_sha256"]
        )
        ids_cpu = np.load(a.fixture)["topk_ids"]
        M, T = ids_cpu.shape
        E, N, K, G = 22, 3072, 4096, 352
        assert (M, T) == (65536, 8)
        torch.manual_seed(451044)
        ep.CHUNK = M
        ep.DIRECT_MIN = M
        ep.shared_add_provider = None
        ids = torch.from_numpy(ids_cpu.copy()).to("cuda")
        saved_ids = ids.clone()
        mapping = torch.full((G,), -1, device="cuda", dtype=torch.int32)
        offset = meta["rank_expert_start"]
        mapping[offset : offset + E] = torch.arange(E, device="cuda", dtype=torch.int32)
        full = torch.empty(2 * M, K, device="cuda", dtype=torch.bfloat16)
        out = full[:M]
        ep.direct_scratch = full[M:]
        reference = torch.empty_like(out)
        rows = triton.cdiv(M * T + E * 255, 256) * 256
        ws13 = torch.empty(
            max(rows * N * 2, M * T * K * 2), device="cuda", dtype=torch.uint8
        )
        ws2 = torch.empty(rows * (K + 4), device="cuda", dtype=torch.uint8)
        ring = []
        phase("eight-weight-pairs")
        for _ in range(8):
            w1 = (torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 32).to(
                ep.FP8
            )
            w2 = (
                torch.randn(E, K, N // 2, device="cuda", dtype=torch.bfloat16) * 32
            ).to(ep.FP8)
            s1 = torch.rand(E, N, 1, device="cuda") * 0.0002 + 0.0005
            s2 = torch.rand(E, K, 1, device="cuda") * 0.0002 + 0.0005
            aq = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 32).to(
                ep.FP8
            )
            asc = torch.rand(M, 1, device="cuda") * 0.005 + 0.015
            weights = torch.rand(M, T, device="cuda")
            weights /= weights.sum(1, keepdim=True)
            ring.append((aq, asc, w1, w2, s1, s2, weights))
        configs = copy.deepcopy(ep.pick_configs(ring[0][2], ring[0][3], T, M))
        assert configs[0]["BLOCK_SIZE_M"] == configs[1]["BLOCK_SIZE_M"] == 256
        original = g2.try_step5_prefill_g2
        original_gemm = ep._gemm
        enabled = False
        counts = dict(fast=0, native_g2=0)
        last_gates = {}
        last_args = {}

        def record_gemm(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            topk_ids,
            sids,
            sorted_w,
            eids,
            nvalid,
            top_k,
            compute_type,
            config,
            **kwargs,
        ):
            if tuple(B.shape) == (22, 4096, 1536):
                counts["native_g2"] += 1
                last_gates["direct" if topk_ids.shape[0] > M else "fallback"] = nvalid
            return original_gemm(
                A,
                B,
                C,
                A_scale,
                B_scale,
                topk_weights,
                topk_ids,
                sids,
                sorted_w,
                eids,
                nvalid,
                top_k,
                compute_type,
                config,
                **kwargs,
            )

        def intercept(*args, **kwargs):
            used = original(*args, **kwargs)
            assert used, "The source-integrated specialized G2 must actually execute"
            counts["fast"] += 1
            last_gates["direct"] = args[9][0:1]
            last_gates["fallback"] = args[9][1:2]
            last_args.update(args=args, kwargs=kwargs)
            return used

        ep._gemm = record_gemm
        g2.try_step5_prefill_g2 = intercept

        def call(i, optimized, poison=False):
            nonlocal enabled
            enabled = optimized
            ep.STEP5_PREFILL_G2 = optimized
            if poison:
                full.fill_(float("nan"))
                ws13.view(torch.bfloat16).fill_(float("nan"))
            aq, asc, w1, w2, s1, s2, weights = ring[i]
            ep.ep_fused_experts(
                aq,
                w1,
                w2,
                weights,
                ids,
                mapping,
                s1,
                s2,
                a1_scale=asc,
                global_num_experts=G,
                output_dtype=torch.bfloat16,
                routed_scaling_factor=1.0,
                configs=configs,
                out=out,
                workspace13=ws13,
                workspace2=ws2,
            )

        def compare(tag):
            assert bool(torch.isfinite(out).all()), tag
            err = float(
                (out.float() - reference.float()).norm()
                / reference.float().norm().clamp_min(1.0e-12)
            )
            assert err < 0.005, (tag, err)
            return dict(
                tag=tag,
                relative_l2=err,
                bit_exact=torch.equal(out, reference),
                direct_rows=int(last_gates["direct"].item()),
                fallback_rows=int(last_gates["fallback"].item()),
            )

        phase("complete-pipeline-numerical")
        for gen, amp in enumerate([1.0, 0.25, 0.0, 4.0]):
            for i, (aq, asc, w1, w2, s1, s2, weights) in enumerate(ring):
                aq.copy_(
                    (torch.randn_like(aq, dtype=torch.bfloat16) * 32 * amp)
                    .clamp(-448, 448)
                    .to(ep.FP8)
                )
                asc.mul_(1.015625)
                weights.uniform_()
                weights.div_(weights.sum(1, keepdim=True))
                call(i, False, True)
                torch.cuda.synchronize()
                reference.copy_(out)
                call(i, True, True)
                torch.cuda.synchronize()
                check = compare(f"g{gen}-pair{i}")
                assert check["direct_rows"] > 0 and check["fallback_rows"] == 0
                if amp == 0.0:
                    assert check["bit_exact"] and bool((out == 0).all())
                report["checks"].append(check)
                save()
        phase("empty-and-overflow-routing")
        for label, all_local in [("empty", False), ("overflow", True)]:
            if all_local:
                ids.copy_(
                    (
                        torch.arange(M, device="cuda")[:, None]
                        + torch.arange(T, device="cuda")[None, :]
                    )
                    % E
                    + offset
                )
            else:
                ids.copy_(
                    (
                        torch.arange(T, device="cuda", dtype=torch.int32)[None, :]
                        + ((offset + E) % G)
                    ).expand(M, T)
                )
            call(0, False, True)
            torch.cuda.synchronize()
            reference.copy_(out)
            call(0, True, True)
            torch.cuda.synchronize()
            check = compare(label)
            assert check["direct_rows"] == 0
            if all_local:
                assert check["fallback_rows"] > 0
            else:
                assert (
                    check["fallback_rows"] == 0
                    and check["bit_exact"]
                    and bool((out == 0).all())
                )
            report["checks"].append(check)
            save()
        phase("metadata-eligibility-fallbacks")
        aa = last_args["args"]
        kk = last_args["kwargs"]
        rejects = []
        for name, changes in [
            ("short-M", dict(num_tokens=32768)),
            ("other-topk", dict(top_k=4)),
            ("sort128", dict(sort_block_m=128)),
            ("other-tile", dict(config=dict(kk["config"], BLOCK_SIZE_N=128))),
            ("extra-config", dict(config=dict(kk["config"], SPLIT_K=2))),
        ]:
            assert not original(*aa, **dict(kk, **changes)), name
            rejects.append(name)
        for name, index, tensor in [
            ("noncontiguous-A", 0, aa[0].as_strided(aa[0].shape, (0, 1))),
            ("uint8-A", 0, aa[0].view(torch.uint8)),
            ("wrong-out", 2, aa[2][:32768]),
            ("wrong-gate", 9, aa[9][:2]),
        ]:
            bb = list(aa)
            bb[index] = tensor
            assert not original(*bb, **kk), name
            rejects.append(name)
        report["eligibility_rejections"] = rejects
        save()
        ids.copy_(saved_ids)
        for aq, asc, w1, w2, s1, s2, weights in ring:
            aq.copy_((torch.randn_like(aq, dtype=torch.bfloat16) * 32).to(ep.FP8))
        for optimized in [False, True]:
            for _ in range(2):
                for i in range(8):
                    call(i, optimized)
        torch.cuda.synchronize()
        assert counts["fast"] > 0 and counts["native_g2"] > 0
        phase("paired-complete-MoE-timing")
        samples = dict(native=[], specialized=[])

        def measure(fn):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            fn()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / 8

        def eight(optimized):
            for i in range(8):
                call(i, optimized)

        for rep in range(15):
            for label in list(samples) if rep % 2 == 0 else list(reversed(samples)):
                samples[label].append(measure(lambda: eight(label == "specialized")))
        report.update(
            samples_ms=samples,
            median_ms={k: statistics.median(v) for k, v in samples.items()},
        )
        report["ratio"] = (
            report["median_ms"]["specialized"] / report["median_ms"]["native"]
        )
        save()
        phase("paired-complete-MoE-graphs")
        graphs = {}
        for label in ["native", "specialized"]:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                eight(label == "specialized")
            graph.replay()
            torch.cuda.synchronize()
            graphs[label] = graph
        samples = dict(native=[], specialized=[])
        for rep in range(15):
            for label in list(samples) if rep % 2 == 0 else list(reversed(samples)):
                samples[label].append(measure(graphs[label].replay))
        report.update(
            graph_samples_ms=samples,
            graph_median_ms={k: statistics.median(v) for k, v in samples.items()},
            g2_dispatch_counts=counts,
            source=ep.__file__,
            fixture=meta,
            complete=True,
            phase="complete",
        )
        report["graph_ratio"] = (
            report["graph_median_ms"]["specialized"]
            / report["graph_median_ms"]["native"]
        )
        save()
        print(
            json.dumps(
                dict(
                    complete=True,
                    checks=len(report["checks"]),
                    median_ms=report["median_ms"],
                    ratio=report["ratio"],
                    graph_median_ms=report["graph_median_ms"],
                    graph_ratio=report["graph_ratio"],
                )
            ),
            flush=True,
        )
        return 0
    except BaseException:
        report["error"] = traceback.format_exc()
        save()
        print(report["error"], flush=True)
        return 2
    finally:
        if ep is not None and original is not None:
            g2.try_step5_prefill_g2 = original
            ep._gemm = original_gemm


if __name__ == "__main__":
    raise SystemExit(main())
