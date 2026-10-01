# DeepSeek-V4.1 DSpark query capacity validation

## Root cause

The DSpark drafter inherited the target's `max_num_batched_tokens` for its
`InputBuffers` and model-side allocations. A target batch with `R` requests,
however, expands to `R * num_query_per_req` draft query tokens. The sequential
Markov head reads the anchor at `input_ids[request_id * num_query_per_req]`.

With a target token budget of 256 and DSpark block size 5:

| max_num_seqs | Initial profiling requests | Largest anchor index | Old input capacity |
| --- | --- | --- | --- |
| 32 | 32 | 155 | 256 |
| 64 | 64 | 315 | 256 |
| 1024 (default) | 256 | 1275 | 256 |

The last two cases access beyond the logical `input_ids` allocation. During
the attention-skipping profiling path, the requested draft forward length is
also larger than the input slice actually supplied to the model.

The core dump from worker DP6/EP6 identifies queue 4, GPU node 14, Debug trap.
Its 1 MiB AQL ring has 16384 slots; hipprof's text output contains two copies.
`rptr=24594` maps to slot 8210. The first unretired dispatch is PyTorch's
`index_elementwise_kernel` / `index_kernel_impl<OpaqueType<4>>`, immediately
before the five Markov sampling iterations. This matches the anchor read in
`DSparkSpeculator._sample_sequential`. `OpaqueType<4>` denotes 4-byte indexed
data, not the index dtype. AQL grid sizes are work-item counts, not block counts.
The trap wave PC was not symbolized, so the packet evidence alone was not used
as proof; CPU reproduction of the actual sequential method and the corrected
GPU runs establish the indexing failure and its fix.

## Plugin fix

`patch_dspark_query_capacity.py` gives the drafter a private, shallow-copied
scheduler configuration with token capacity
`max(target_token_budget, max_num_seqs * num_query_per_req)`.

This covers its input IDs, positions, context/hidden buffers, model allocations,
and draft attention metadata. Before capture, the shared block-table slot
mapping allocation is enlarged to the same capacity, preserving existing
contents and filling new slots with `PAD_SLOT_ID`. It retains a stable address
for both target and draft captures.

The target scheduling configuration is preserved. The existing draft EP8
AG-RS patch remains active; target uses DeepEP LL + DeepGEMM and draft uses
AG-RS + DeepGEMM. No source changes were made in `vllm/`.

Both anchor-first (`N`) and fill-in (`1+N`) layouts are covered by regression
tests. The checked DAS source in `pkg/vllm` has the same target-sized input
allocation and anchor read, with no existing capacity fix found there.

## GPU validation (2026-10-01, nmz18)

Common configuration: TP1, DP8, EP enabled, target `deepep_low_latency`,
target/draft MoE `deep_gemm`, DSpark block size 5, target batched-token budget
256, Engram CPU offload with `embedding_across_dp=true` and
`dp_shared_memory=false`. Evaluation is 100 HumanEval samples, batch size 16,
temperature 0, thinking disabled.

| max_num_seqs | Run directory relative to project | HumanEval pass@1 | TTFT | TPOT |
| --- | --- | --- | --- | --- |
| 64 | `acc/20261001_143927` | 99/100 | 1645.8 ms | 113.8 ms |
| default 1024 | `acc/20261001_145122` | 99/100 | 1724.9 ms | 124.8 ms |

Both runs finish graph capture, start the service, complete evaluation, and
stop the service normally. These validate startup and the sampled serving
workload; the evaluation does not exercise 1024 concurrent live requests.
Timing differences are recorded observations, not a controlled performance
comparison.

Reports:

- `acc/20261001_143927/20261001_144357/reports/dsv41-flash/humaneval.json`
- `acc/20261001_145122/20261001_145615/reports/dsv41-flash/humaneval.json`

## Checks

- `tests/runtime_patch/test_deepseek_v4_dspark_patches.py`: 42 passed,
  including 9 query-capacity regressions.
- Worker dispatcher inventory checks: 2 passed (run from the plugin root with
  `VLLM_V0251_SOURCE_ROOT=../vllm`).
- New patch: Ruff check and format check passed.
- Broader plugin lifecycle suite: 29 passed, 5 setup errors from its mock
  `gpu_worker` missing `_num_workspace_lanes`; those tests did not reach the
  capacity patch.
- Upstream `vllm/` working tree is clean.
