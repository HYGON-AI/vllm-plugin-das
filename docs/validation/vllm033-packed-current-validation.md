# Native packed integration validation, 2026-11-14

The native packed candidate was merged normally with current v0.33.0
`2cdc04242cee17fb24f7e758e62cb5c5ea0a3222`, producing runtime source
`01dee6db8abfe689d0aa796de2edfd2361bdef07`. The integration preserves the
0.33 package version, current Adapter minimum, AITER entry points, root-build
guard and concurrent upstream additions. It does not reintroduce the legacy
platform plugin. Unrelated DeepSeek V4.1 testing was deferred by the user.

The normally resolved Plugin wheel has SHA256
`2ae2e1a21b984502921743e235106b8dad4290bdbaeb49d6ace70c71cec935ee`.
Core `ae5ebff332de0bd467f285e1df0127b5e7d6b3ac` HIP modules were rebuilt;
the installed wheel, Python sources and parent/child origins were checked.
Portable target-source contracts passed 622 tests with 6 HCU cases deselected;
the final installed packed CPU contracts passed 64 tests. The source-overlay
and fully installed checks are distinct evidence scopes.

On gfx936, the four existing TP1 native packed tests passed for BF16/FP16 and
padding/no-padding, with two official raw reload generations and two changed
input/routing states each: 16 replay states. The tests use four experts,
hidden size 256, independent signed INT4 CPU references, stable raw/scale/output
addresses and wrong-nibble/signedness negative controls. These are component
results, not TP2/TP4 or full-checkpoint/model quality/performance acceptance.

The retained `resume-20261114/packed-device-summary.json` indexes the native
receipts. Interpreter teardown still reports the installed BoltOps
duplicate-qualified schema registration/finalizer error. An isolated vendor
source reproduction and candidate correction passed on CPU; the installed
vendor package was left unchanged. Therefore clean native lifecycle is not
accepted, despite the numerical assertions passing and VRAM returning to its
baseline. This vendor issue must not be hidden by ignoring teardown errors.

Current project image/device CI remains a separate gate: the configured
`HARBOR_SITE` and approved immutable `HCU_CI_IMAGE_V033` must resolve before
device tests can run. The local component results do not waive that gate.
Full packed checkpoint HumanEval 0-31, required TP2/TP4 coverage and clean
shutdown remain open for this combined artifact.

## Current-main component follow-up, 2026-11-21

The separate validation tree normally merges PR367 head `a4af6441389c` with
Plugin v0.33.0 `32a1a6c05d6d`, without conflicts, at `4222b3d94640`.
It uses Core v0.33.0 `c440c089fe96` and Adapter main `9c20b33d8ebf`.
Adapter and Plugin Python wheels were built into two new physical prefixes;
the compatible native dependencies were reused, not rebuilt. This is a
development pairing, not a newly published release image. FlashAttention
was neither modified nor rebuilt, and no DeepSeek V4.1 checkpoint was tested.

Both prefixes pass full owned RECORD/wheel/source checks; shared Torch,
Triton and LightOp are verified at their physical owners. The installed
BoltOps distribution passes its complete 912-file RECORD check. The current
baseline passes 34 selected SlimQuant CPU cases. The candidate passes the
same 34 plus 64 packed contracts, with zero failures/errors/skips; ten
hardware cases are deliberately deselected in each CPU run. Current CI
selection tests also pass: 15 tests and 14 subtests.

On an actual gfx938/BW1100 device, all four unchanged packed native cases
pass with the current installed candidate: BF16/FP16, padding/no-padding,
two official reload generations and 16 changed-input Graph replay states.
The independent signed INT4 reference and stable-pointer assertions remain
unchanged. This component process exits without weakref, nanobind or traceback
diagnostics. The first toolchain preflight selected the Ninja console script
instead of its ELF and was refused before any device process started; the
separate corrected run uses the shipped `ninja/data/bin/ninja`, verified against
its RECORD. No installed Ninja file was changed.

These component results do not qualify a new full-model score, serving
performance, remote project device CI or full-model lifecycle. The completed
historical TP2/TP4 expanded-versus-packed HE32 pairs retain their original
dependencies and raw/normalized scores; they are not rerun to improve a score
or transferred to this new combination. Full-model lifecycle failures and the
earlier TP8 timeout remain recorded.

### Paired serving performance and shutdown

The same sealed `Qwen3.8-Flash-Next-Channel-INT4-w4a8` checkpoint was run on
the same gfx938/BW1100 devices 0/1, TP2 and BF16. Both arms use the official
default FULL_AND_PIECEWISE Graph policy, prefix caching ON, context 4096,
max sequences 16, max batched tokens 1024, fixed KV allocation 1 GiB/rank,
and device PLE (`engram_config.cpu_offload=false`). Vendor configuration
and installed native packages are unchanged. This shared node was not
exclusively reserved.

The official `vllm bench serve` random workload uses 32 requests, concurrency
4, 256 input tokens and 128 forced output tokens, temperature 0 and
ignore-EOS. After warmup, seeds 0/1/2 provide three measured repetitions.
All 192 measured requests succeed, each arm producing 4096 output tokens
per repetition. Every final server-tokenizer-aligned model-input field
matches between arms. The official client's random per-run request IDs are
retained separately; complete request receipts are not byte-identical.

| Measurement | Expanded INT8 | Packed BoltOps |
| --- | ---: | ---: |
| Output token/s, repetition 1 | 151.731 | 88.307 |
| Output token/s, repetition 2 | 151.517 | 87.958 |
| Output token/s, repetition 3 | 151.681 | 86.655 |
| Median output token/s | 151.681 | 87.958 |
| Median mean TTFT, ms | 338.649 | 382.525 |
| Median mean TPOT, ms | 23.901 | 42.800 |
| Median mean end-to-end latency, ms | 3373.949 | 5818.064 |

**Packed throughput regresses 42.01%; mean TPOT increases 79.07%. This is
a failed performance gate, and PR367 remains Draft.** This synthetic
workload establishes neither accuracy nor a general SLA. Prefix hits are
zero in both arms; input length 256 is below the observed cache block 1568,
so this is not a prefix-hit performance qualification.

Independent post-measurement profiler sessions preserve both TP-rank traces.
For rank 0 the expanded/packed traces contain 14485/19573 kernel events.
Packed adds route-preparation sort/scan/copy work and uses distinct packed
GEMM kernels. The vendor warns that a dedicated gfx938 INT4 configuration
for this geometry is absent and selects its default. Neither observation
alone establishes the dominant cause of the serving regression; profiler
times and Graph annotations are diagnostic, not additional timed benchmark
repetitions.

Both server frontends exit 0 after one controller SIGINT, within the original
15-second cleanup deadline. Both executors nevertheless escalate to worker
SIGTERM after their internal grace period, and both servers and all benchmark
clients report nanobind leaks. The candidate additionally reports
`AsyncLLM output_handler failed` / `EngineDeadError` during shutdown, after
all requests completed. Profiler teardown is present in these server runs;
this does not establish a packed-only or unprofiled-production root cause.
Neither arm passes clean lifecycle. No external forced kill was needed;
owned process trees are gone, ports are bindable and VRAM returns to baseline.

Double-read checkpoint seals match before/after (144 assets,
127135690592 bytes). Owned RECORD/source/wheel and shared dependency checks
remain equal before/after; both arms have identical Core/Adapter/AITER/
sgl-kernel payloads, and BoltOps' complete 912-file RECORD remains unchanged.
All-rank inspection confirms 48 routed layers per rank, stable storage and
the expected expanded versus packed dimensions. No HumanEval requests were
generated. Local evidence: `evidence/pr367-current-performance-20261121/`
(`performance-audit.json`, original results, requests, traces and exit logs).
