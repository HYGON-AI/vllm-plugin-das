# SPDX-License-Identifier: Apache-2.0
"""Check installed Step5 Dense tuning on an idle GPU in a fresh process.

Uses the normal vllm_hcu.hcu_ops extension and installed Python selector. This
is an operator correctness check, not a model-throughput or language-quality
benchmark. Run with a matching normal vllm_hcu build installed; no standalone
library, source-directory override, or runtime compilation is used.

Example:
    python check_fp8_dense_tuned.py --device 0 --output dense-check.json
"""

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0,
                        help="GPU index within the process's visible devices")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON result path; existing results are not overwritten")
    args = parser.parse_args()
    if not os.environ.get("HCU_STEP5_TUNING_CONFIG"):
        parser.error("Set HCU_STEP5_TUNING_CONFIG to an external qualified tuning profile")
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if args.output.exists():
        parser.error("--output already exists; choose a new result path")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Parse --help before importing Torch. Set the opt-in before any operator
    # module reads its environment; never alter sys.path or load a source file.
    os.environ["HCU_STEP5_DENSE_TUNING"] = "1"
    started = time.time()
    rows = []
    metadata = dict(device_index=args.device, model_tps_not_measured=True)
    try:
        import torch
        import vllm_hcu.hcu_ops as extension
        from lightop.gemm_ops import hipblaslt_w8a8_channelwise_gemm
        from vllm_hcu.ops.step5 import fp8_dense_tuned as selector

        torch.cuda.set_device(args.device)
        device = torch.device("cuda", args.device)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(127321)

        extension_path = Path(extension.__file__).resolve()
        backend_path = Path(
            torch.ops.hcu_ops.step5_fp8_dense_backend_library()
        ).resolve()
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            extension_path=str(extension_path),
            extension_sha256=hashlib.sha256(extension_path.read_bytes()).hexdigest(),
            backend_library_path=str(backend_path),
            backend_library_sha256=hashlib.sha256(backend_path.read_bytes()).hexdigest(),
            selector_path=str(Path(selector.__file__).resolve()),
            architecture=properties.gcnArchName,
            compute_units=properties.multi_processor_count,
            torch_version=torch.__version__,
            torch_hip_version=torch.version.hip,
        )
        shapes = [dict(M=m, N=n, K=k, algorithm_id=algorithm)
                  for (m, n, k), algorithm in selector._CHOICES.items()]
        assert len(shapes) == 6, "Update this check when the six-shape recipe changes"
        assert not selector._warmed, "Run this checker in a fresh process"

        def data(m, n, k, seed):
            generator = torch.Generator(device=device).manual_seed(seed)
            x = (torch.randn(m, k, device=device, generator=generator) * 20)
            w = (torch.randn(n, k, device=device, generator=generator) * 20)
            x = x.clamp(-448, 448).to(torch.float8_e4m3fn)
            w = w.clamp(-448, 448).to(x.dtype)
            xs = torch.rand(m, 1, device=device, generator=generator) * .01 + .005
            ws = torch.rand(n, 1, device=device, generator=generator) * .001 + .0005
            return x, xs, w, ws

        def native(x, xs, w, ws):
            m, k = x.shape
            n = w.shape[0]
            ok, out = hipblaslt_w8a8_channelwise_gemm(
                x, w, xs, ws, m, n, k, "NT", torch.bfloat16, None)
            assert ok
            # LightOp returns a leading batch dimension for an unbatched GEMM.
            return out.view(m, n)

        def check(out, reference):
            assert out is not None and tuple(out.shape) == tuple(reference.shape)
            assert out.dtype == torch.bfloat16 and bool(torch.isfinite(out).all())
            relative = float((out.float() - reference.float()).norm()
                             / reference.float().norm().clamp_min(1e-12))
            assert relative < .01, relative
            return dict(relative_l2_native=relative,
                        exact_native=torch.equal(out, reference))

        def check_fp32(out, x, xs, w, ws):
            reference = (x.float() * xs) @ (w.float() * ws).T
            relative = float((out.float() - reference).norm()
                             / reference.norm().clamp_min(1e-12))
            assert relative < .005, relative
            return relative

        # A first, untuned M64 call must preserve fallback without a cold plan.
        x, xs, w, ws = data(64, 14080, 4096, 1000)
        assert selector.tuned_fp8_gemm(x, xs, w, ws) is None
        assert not selector._warmed
        del x, xs, w, ws
        cold_small_fallback = True

        for index, (n, k) in enumerate(sorted({(s["N"], s["K"]) for s in shapes})):
            x, xs, w, ws = data(selector._UNSPLIT_PROFILE_ROWS, n, k, 2000 + index)
            assert selector.tuned_fp8_gemm(x, xs, w, ws) is None
            torch.cuda.synchronize(device)
            del x, xs, w, ws
        assert len(selector._warmed) == 4

        for index, shape in enumerate(shapes):
            m, n, k = [shape[key] for key in ("M", "N", "K")]
            x, xs, w, ws = data(m, n, k, 3000 + index)
            out = selector.tuned_fp8_gemm(x, xs, w, ws)
            proof = [dict(eager=True, **check(out, native(x, xs, w, ws)),
                          relative_l2_fp32=check_fp32(out, x, xs, w, ws))]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = selector.tuned_fp8_gemm(x, xs, w, ws)
            for trial, factor in enumerate((.25, 1., 4.)):
                # Keep captured addresses, change all four tensor contents,
                # and poison the output to detect missing or stale writes.
                changed = data(m, n, k, 4000 + 10 * index + trial)
                x.copy_(changed[0])
                xs.copy_(changed[1] * factor)
                w.copy_(changed[2])
                ws.copy_(changed[3] * factor)
                del changed
                captured.fill_(float("nan"))
                graph.replay()
                proof.append(dict(
                    factor=factor, changed_x_w_xs_ws=True, poisoned_output=True,
                    **check(captured, native(x, xs, w, ws)),
                    relative_l2_fp32=check_fp32(captured, x, xs, w, ws),
                ))

            # Reuse the C++ plan with new allocations. The old graph must keep
            # its original pointers after descriptor scale pointers change.
            old_reference = native(x, xs, w, ws).clone()
            x2, xs2, w2, ws2 = data(m, n, k, 5000 + index)
            other = selector.tuned_fp8_gemm(x2, xs2, w2, ws2)
            proof.append(dict(
                changed_tensor_pointers=True,
                **check(other, native(x2, xs2, w2, ws2)),
                relative_l2_fp32=check_fp32(other, x2, xs2, w2, ws2),
            ))
            captured.fill_(float("nan"))
            graph.replay()
            proof.append(dict(captured_old_pointers_preserved=True,
                              **check(captured, old_reference)))
            del old_reference, x2, xs2, w2, ws2, other

            x.zero_()
            captured.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            assert bool((captured == 0).all())
            proof.append(dict(zero_exact=True, poisoned_output=True))
            assert selector.tuned_fp8_gemm(x[:64], xs[:64], w, ws) is None
            rows.append(dict(shape=shape, proof=proof))
            del x, xs, w, ws, out, captured, graph
            torch.cuda.synchronize(device)

        # Concurrent host enqueues reuse one plan with different scale pointers.
        barrier = threading.Barrier(2)

        def thread_run(seed):
            torch.cuda.set_device(args.device)
            x, xs, w, ws = data(4096, 14080, 4096, 6000 + seed)
            xs.mul_(seed + 1)
            barrier.wait()
            outputs = [selector.tuned_fp8_gemm(x, xs, w, ws) for _ in range(4)]
            return outputs, native(x, xs, w, ws), (x, xs, w, ws)

        with ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(thread_run, [1, 2]))
        torch.cuda.synchronize(device)
        thread_proof = [check(out, reference)
                        for outputs, reference, _inputs in values for out in outputs]

        maps_path = Path("/proc/self/maps")
        experimental_mapping = None
        if maps_path.exists():
            experimental_mapping = "libstep5_dense_tuned" in maps_path.read_text()
            assert not experimental_mapping, "Experimental standalone Dense library is loaded"
        result = dict(
            **metadata, complete=True, normal_hcu_ops=True, shape_count=len(rows),
            rows=rows, thread_proof=thread_proof,
            cold_untuned_M64_fallback=cold_small_fallback,
            plans_prepared_before_DBO=True, workspace_bytes=0,
            known_experimental_library_loaded=experimental_mapping,
            elapsed_s=time.time() - started,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(complete=True, shape_count=len(rows),
                              thread_checks=len(thread_proof), output=str(args.output),
                              elapsed_s=result["elapsed_s"])), flush=True)
    except BaseException as error:
        result = dict(**metadata, complete=False, completed_shapes=len(rows),
                      error=repr(error), elapsed_s=time.time() - started)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
