# Standard DeepGEMM MegaMoE decode integration

Select the backend explicitly with `--moe-backend deep_gemm_mega_moe`. Channel-FP8
experts then use `deepgemm.mega` through `deepgemm_megamoe.py`. The adapter does
not inspect the model name and does not pass private kernel/layout arguments.
Other CLI backends retain their existing method selection. The existing
standalone INT4 adapter remains separate because it has a different public API.

This is an optional **single-node D-instance** path in PR #504. Existing P/D,
EP16 Decode and WASP128 Prefill launch profiles retain their current backends
and dependency pins. Do not replace the Prefill nightly wheel or enable this
EP8 decode configuration on the existing dual-node profiles.

The legacy `step5_pair32_megamoe.py` entry and the
`HCU_PAIR32_MEGAMOE_DECODE` / `HCU_MEGAMOE_ALL_DECODE` gates are removed.

Activation limits are resolved from the explicit layer/config limit and the
activation enum. `SWIGLUSTEP` intrinsically requires a 7.0 clamp even when the
optional `swiglu_limit` field is `None`; ordinary SiLU remains unclamped unless
an explicit limit is supplied. Both quantization metadata and the public
`fp8_mega_moe` call receive the resolved limit.

For the `megamoe-ll-opt` branch, set these variables before loading D weights:

```bash
export DG_MEGA_DECODE_PHASED=1
export DG_MEGA_DECODE_MAX_CAP=64
export DG_MEGA_DISABLE_RING=1
export DG_MEGA_CFG_ALLGATHER=0
export HCU_STEP5_DENSE_TUNING=0
unset HCU_STEP5_TUNING_CONFIG STEP5_DENSE_TUNING
```

The measured Decode path does not load or require `dense-qualified.json`.
Optional dense-tuning JSON support for historical EP8 Prefill reproduction is
separate from this Decode configuration. AITER/DTK's ordinary built-in dispatch
tables and runtime libraries remain normal package dependencies.

## PR #504 integration checks (2026-11-25)

The port is based on PR head `0a598a24`. It changes only the new adapter,
its selection/shutdown hooks, tests and this document. Existing native C++,
P/D and WASP128 implementation and launch profiles are byte-for-byte unchanged.
The adapter, hooks and replay test match standalone source `4af38e34`.

Targeted CPU checks, with `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` and
`CUDA_VISIBLE_DEVICES` all set to `-1`, completed with **484 passed, 6 skipped**.
Coverage includes the generic adapter, existing MegaMoE behavior, IPC gather,
shutdown lifecycle, P/D ownership/transfer/graph contracts and WASP128/P-prefill
contracts. Syntax, whitespace and minimal-port identity checks also passed.
These CPU checks do not replace combined-revision GPU validation.

DeepGEMM chooses weight packing, symmetric-buffer layout and the kernel from the
same environment and compiled geometry. `DG_MEGA_DECODE_MAX_CAP` is an advisory
warning threshold, **not a fallback selector**. At commit `6dc94c9`, the compiled
phased instance is EP8, top-k8, H4096, I1536, 44 local experts. Other geometries
remain subject to the original unified LL/HT kernel's supported instances.
The framework adapter itself does not make additional sizes supported.

Weights must be symmetric per-channel E4M3FN with dynamic per-token FP8
activations. Inputs must be BF16. EP is required; physical expert IDs must be
contiguous on each rank. Bias, LoRA, PCP, sequence parallelism, EPLB, routing
weights on the input and deferred routed-output reduction are rejected.
Shared experts remain with the ordinary runner and its separate TP reduction.

For TP replicas, the adapter partitions the current input rows into consecutive
slices of `ceil(rows / TP)` rows. Empty slices still enter the EP kernel. A
graph-capturable IPC all-gather restores the TP-replicated output, with padding
removed after gathering. Source capacity is `ceil(max_num_batched_tokens / TP)`.
For TP4 and a 256-token scheduler budget this is 64; C32/DP2/MTP3 normally gives
64 rows per DP and 16 source tokens per EP rank, before routing.

Packing replaces the canonical weight storage instead of retaining another
complete expert copy. Weight reload into already-packed parameters is not
supported by this adapter; restart for new weights or a different layout.
Keep the DeepGEMM environment fixed for the process lifetime. Do not switch
between decode packing and W6 packing after model loading.

The upstream branch documents the phased path for a separate D instance.
The single-node validation campaign prepares context in unmeasured small
chunks under the same D configuration and reports only the common decode
window. This does not establish support for large-token mixed P/D workloads.
The branch's default activation uses `swiglu_nv` clamp ordering; its mathematical
and model accuracy results must be checked independently of older Step5 scores.

CPU contract tests:

```bash
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  VLLM_PLUGINS=__disabled__ python -m pytest -q \
  tests/runtime_patch/test_deepgemm_megamoe.py \
  tests/runtime_patch/test_deepseek_v4_megamoe.py \
  tests/runtime_patch/test_shutdown_lifecycle.py
```

GPU/operator, trace and GSM8K results belong to the pinned campaign artifacts;
CPU contract tests alone do not establish GPU correctness or performance.

The GPU replay script is the same mathematical and TP4 test used for the
validation below. It requires eight idle gfx938 GPUs, PyTorch, DeepGEMM and
the normal plugin extension; it is not included in the CPU test command.
The report timing includes preprocessing and MegaMoE, not just GEMM time.

```bash
DG_MEGA_DECODE_PHASED=1 DG_MEGA_DECODE_MAX_CAP=64 DG_MEGA_DISABLE_RING=1 \
  DG_MEGA_TEST_OUTPUT_DIR=/tmp/deepgemm-mega-replay \
  python tests/distributed/single_node/tp_ep/deepgemm_megamoe_replay.py
```

## Standalone validation snapshot (2026-11-24)

DeepGEMM is the unmodified `megamoe-ll-opt` commit
`6dc94c996d2610b7adc264163844a1666f4b26ad`. The tested framework runtime is
`ed93ffa13aa97610cb74acd5d23f270796d95fac`, published in standalone integration
`4af38e348cd43b24999cc9162fce97d13b2a9cd6`. The generic adapter and replay test
are ported unchanged into PR #504; existing PR-specific P/D and Dense code is
preserved rather than replaced by that standalone branch.

**The GPU, accuracy and warmup results below belong to the standalone revision,
not the combined PR #504 revision.** The PR integration is checked on CPU only
at the user's request. No new GPU process or reservation is started. Combined
P/D end-to-end accuracy and performance remain pending validation.

| Check | Result |
| --- | --- |
| Focused CPU regression before source submission | 139 passed, 3 skipped (optional INT4 native dependency) |
| Submission CPU checks including upstream INT8/Kimi tests | 159 passed, 3 skipped; all GPU visibility masks disabled |
| Eight-rank operator checks | 96 rank/case combinations passed, including eager/Graph, TP4 gather, empty sources and skewed routing |
| Intrinsic activation regression | GPU TP4 cases use `SWIGLUSTEP` with the optional limit unset, including large activations that require clipping |
| GSM8K frozen first 100 | **99/100**, zero request errors, 100 normal stops, max output 5437 of 32768 tokens |
| Sampling | C32, temperature 1.0, top-p 0.95, top-k -1, seed 42; MTP disabled |
| Source/Graph inspection | Eight ranks, 88 MoE layers; resolved and captured decode-kernel activation clamp is 7.0 |
| Trace | Eight-rank, 10 main decode steps; validation passed |

The first C32 full performance **warmup** used 131072 input tokens and 4096
output tokens per request, TP4/DP2/EP8, synthetic MTP3 with acceptance rates
`[0.838992, 0.598322, 0.362686]`, prompt seed 44 and no request seed.

| Warmup metric | Value |
| --- | ---: |
| Common-window decode throughput | 1557.67 tok/s |
| Mean request ITL | 58.62 ms/chunk |
| Mean request TPOT | 20.43 ms/token |
| Batch duration | 1628.69 s |
| Common decode-window token coverage | 95.84% |
| KV preemptions | 0 |

This excluded warmup includes profiler activity and two first-use JIT events.
The user stopped the campaign during the second warmup. **No formal round was
completed**, so this value is not a three-round performance result or a claim
of stable throughput. The model, startup guard, holders and monitor were stopped;
the final KFD snapshot was empty. Source-submission checks are CPU-only.

The earlier 93/100 run used an incorrect framework activation mapping that
passed infinity for the unset optional limit. It is retained as failed evidence,
not attributed to the unmodified DeepGEMM kernel. Correcting the intrinsic
`SWIGLUSTEP` limit produced the 99/100 result above with the same prompts,
extractor and sampling settings.

Runtime extensions are ordinary source-built wheel contents:
`deepgemm.op`, `deepgemm.mega._C`, and `vllm_hcu.hcu_ops`. No `_C_step5` extension,
external `.so` override or `LD_PRELOAD` is required. The normal DTK HIP/hipBLASLt
libraries remain runtime dependencies. Benchmark scripts and traces are campaign
artifacts, not production import patches.
