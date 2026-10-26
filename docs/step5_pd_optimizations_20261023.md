# Step5 P/D optimization snapshot — 2026-10-23

Historical measurement record. Current cleanup removes the retired Triton G1
experiment and keeps tuning data external; see [current source contract](step5_pd_verification_20261024.md#source-cleanup-after-the-recorded-measurements).

This branch consolidates the tested Step5 source and reproduction records for
vLLM 0.28.1 / DTK2610 / gfx938. It is not based on the fork's v0.25.1 default
branch. Existing commit history is retained; the source snapshot is
`1ea46e1c094d870db6e118d42de1720d80214a59` plus the verified Decode Fabric2
source overlay and the publication integration described below.

## Source inventory

| Change | Source | Selection and limits |
| --- | --- | --- |
| DP4 packed dispatch/combine over AITER Fabric | `distributed/device_communicators/step5_dp_fabric.py`, communicator adapter | Default off; only two measured payload shapes, DP4, packed dispatch; other calls retain RCCL. Custom-op master-off prevents allocation. |
| Context-bounded Decode graph with long-context fallback | `v1/step5_hot_context.py`, `step5_execution.py`, `hcu_model_runner_v2.py`, `layers/step4_dsa.py` | Default off. CPU upper bound votes for eager through existing DP mode synchronization; exact host mirror rejects unsafe FULL replay. Model/KV limit remains 1M. |
| Equal-token DP microbatch collectives | `layers/fused_moe/modular_kernel.py` | `HCU_DP_MOE_PIPE_UNIFORM=1`; equal positive full-DP microbatches only; other layouts keep the original grouped path. |
| GPU-only DSA request-index construction | `layers/step4_dsa.py` | Replaces CPU count upload and dynamic repeat_interleave with GPU prefix-boundary lookup; token ownership is preserved. |
| Shape-selected Dense FP8 GEMMs | `ops/step5/fp8_dense*.py`, `csrc/step5_fp8_dense.cpp` | Opt-in `HCU_STEP5_DENSE_TUNING=1`; six gfx938 shapes, prewarmed zero-workspace hipBLASLt plans, existing fallback for other shapes. |
| EP chunk-output and DeepEP DBO lifetime fixes | `layers/fused_moe/{modular_kernel,deepep_runtime}.py` | Preserve caller-owned results and publish handles after the communication fence. These are correctness fixes, not a DeepEP performance recommendation. |
| Packed DeepGEMM Step5 activation contract | `layers/fused_moe/{ep_fused_moe.py,experts/deep_gemm_moe.py}` | Preserves post-SiLU clamp7, BF16 rounding and token scales for SWIGLUSTEP. |
| Shape-specialized MoE G1 | `ops/step5/prefill_g1.py` | Default off, `HCU_STEP5_PREFILL_G1=1`; E22/M65536/topk8/BM256 only. Candidate, not promoted. |
| Single-launch MoE G2/overflow selection | `ops/step5/prefill_g2.py` | Default off, `HCU_STEP5_PREFILL_G2=1`; same fixed route layout; retains Directsum accumulation and overflow fallback. |

Paths above are relative to `vllm_hcu/`. The current large-Prefill expert GEMMs
use BoltOps/Triton and the optional specialized Triton kernels. They must not
be described as the AITER ASM implementation. Decode AITER Fabric is a
communication path, separate from the expert GEMM provider.

## Recorded performance and quality

Each phase independently uses four nodes x four GPUs. Prefill and Decode are
separate measurements, not a simultaneous P/D deployment. Performance uses
synthetic MTP3 rates `[0.838992, 0.598322, 0.362686]`; quality runs use standard
MTP3. All throughput values below are three clean formal-round means.

| Stage / variant | Topology | Throughput | ITL | Separate GSM8K100 |
| --- | --- | ---: | ---: | ---: |
| Prefill, Dense baseline | DP16/TP1/EP16 | 41,580.14 tps | — | 98/100 |
| Prefill, G1 candidate | DP16/TP1/EP16 | 42,045.84 tps | — | 96/100 on both runs; fresh off-control 98/100 on both runs |
| Prefill, G2 candidate, G1 off | DP16/TP1/EP16 | 42,224.93 tps | — | 97/100 |
| Decode, hot-context graph + Fabric | DP4/TP4/EP16 | 3,149.20 tps | 57.038 ms | 99/100 |

Prefill input/output is 65536/1, concurrency64, batched-tokens8192,
max-model-len70000. Decode input/output is 65536/4096, concurrency64,
batched-tokens32768 and max-model-len1048576. ITL is the original streaming
**chunk** interval; MTP may emit multiple tokens in a chunk, so this is not TPOT.
The measured hot graph covers up to 69888 context tokens; longer batches use
the native synchronized eager path. Boundary behavior was tested, but an
actual 1M-token end-to-end workload was not run.

The G1 short-GSM difference is unresolved and the candidate remains off by
default. Short GSM prompts do not hit the M65536 specialization. Long24K and
long64K smoke scores are 15/16 for both G1 and G2 candidates; G2 evaluation-window
fast-path counts were not collected. Neither small-sample quality nor isolated
bitexact tests establish full-dataset equivalence. G1/G2 isolated complete-MoE
checks and metadata/overflow guards are documented in their individual notes.
The 45000 tps Prefill target is not reached.

See [machine-readable results](../benchmarks/step5/results-pd-image1021-20261023.json),
[G1 details](step5_prefill_g1.md), [G2 details](step5_prefill_g2.md), and the
[historical migration record](step5_multinode_tuning.md). Historical files remain
historical; this snapshot does not overwrite their reported scores.

## Reproduction inputs

Build the normal plugin and `hcu_ops` extension against the matching CPython3.12,
torch and DTK image. No external binary is vendored. Dense tuning checks the
loaded hipBLASLt build hash; algorithm IDs must be retuned if that library changes.
Obtain model weights, benchmark tokenizer and measured placement PT files
separately; their hashes are in the JSON report. The original benchmark client
and datasets are external assets. No credentials or raw model outputs are added.

On every node, set the existing required values with your own paths/IPs:

```bash
export STEP5_HEAD_IP=HEAD_NODE_IP
export VLLM_HOST_IP=THIS_NODE_IP
export STEP5_SOURCE_DIR=/path/to/this/checkout
export STEP5_MODEL_DIR=/path/to/Step5-Channel-FP8-w8a8
export STEP5_TOKENIZER=/path/to/benchmark-tokenizer
export STEP5_STATE_DIR=/path/to/data-backed/run-directory
```

For Prefill, choose a recorded profile explicitly. `baseline` retains the
historical launcher defaults; `dense_1021` reproduces the selected Dense recipe;
`g1_1021` and `g2_1021` are opt-in candidates and never enable each other.

```bash
export STEP5_PREFILL_PLACEMENT=/external/prefill-placement.pt
export STEP5_PREFILL_PROFILE=dense_1021
# For a controlled candidate: g1_1021 or g2_1021.
STEP5_DRY_RUN=1 bash benchmarks/step5/launch_multinode.sh prefill NODE_RANK
bash benchmarks/step5/launch_multinode.sh prefill NODE_RANK
```

For the measured Decode profile:

```bash
export STEP5_DECODE_PROFILE=itl6479
export STEP5_DECODE_PLACEMENT=/external/decode-placement.pt
export STEP5_ROUTER_SHARED_MAX_M=64
export STEP5_SHARED_STREAM_MIN_TOKENS=64
export STEP5_DECODE_EXPERT_GRID=1
export HCU_STEP5_DP_FABRIC=1
export HCU_STEP5_CONTEXT_GRAPH_REGIONS=8736
STEP5_DRY_RUN=1 bash benchmarks/step5/launch_multinode.sh decode NODE_RANK
bash benchmarks/step5/launch_multinode.sh decode NODE_RANK
```

Use node ranks0..3, stop the preceding phase before starting another, and use
fresh result directories with full warmup and per-round compilation audits.
The publication launcher wiring was checked with CPU dry-run tests; the
consolidated checkout has not been rebenchmarked on the live cluster.

## Excluded work

Ongoing ASM solution/shuffle/persistent tuning, experimental AITER Fabric16
builds, model weights, placement PTs, native binaries, trace archives and
private SSH orchestration scripts are not part of this branch. Rejected backend
experiments are retained as clearly labelled result records where already
committed; they do not become defaults. The original working checkout and
running tuning jobs are left untouched.


## Publication checks

The portable publication suite passed61 tests and4 subtests on Python3.12,
covering Fabric routing/master-off/fallback, context bounds, DeepEP dispatch
lifetime, E22 tables, reproduction arguments and source packaging. All707
Python files parsed; the patch inventory found171 adapters with no missing
contracts or test references. The changed shell launcher passed `bash -n`.
An attempted environment-contract run had16 import failures because this
Windows environment has no vLLM distribution; these are not reported as passes.
The matching Linux/DTK extension build and full model were not rerun for this
consolidated publication tree. Existing hardware evidence is labelled separately.
