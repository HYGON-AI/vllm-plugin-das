# Qwen4Exp HCU prefill indexer spike design

## Goal

Determine whether the request-major prefill indexer design from upstream vLLM
PR 54513 can outperform the existing HCU FlashAttention/CUTLASS
`mqa_paged_score_func` on BW1100. The target serving workload is DP4/EP4,
batch 16, input 4096, output 1024. A kernel-level win must be demonstrated
before the new path is connected to serving.

## Current evidence

- PR 248 replaces context-free prefill QSA attention with dense attention. Its
  aligned A/B result improved mean TTFT from 2726.194 ms to 2704.471 ms, only
  0.797%.
- Trace aggregation showed that the dense FlashAttention work added about
  400.236 ms while avoided indexer and sparse-reader work totaled about
  397.111 ms. The dense replacement therefore removed the potential gain.
- The installed HCU stack already replaces upstream's generic Triton QSA score
  kernel with `flash_attn.mqa_paged_score_func`. The upstream PR's reported
  speedup against its generic NVIDIA Triton baseline cannot be assumed here.
- vLLM 0.28.1 AMD QSA metadata already exposes `visible_blocks`,
  `query_start_loc`, and decode/prefill split counts. No metadata ABI change is
  required for the experiment.
- The installed FlashAttention extension exposes only a generic paged MQA
  score entry point. It has no request-major prefill-specialized entry point.

## Chosen approach

Add a standalone HCU Triton prefill score kernel modeled on the upstream
request-major kernel. It consumes packed prefill Q rows, paged compressed K,
request `query_start_loc`, and precomputed `visible_blocks`; it tiles multiple
query rows from the same request so each key tile is reused across those rows.
The prototype retains the existing HCU top-k, index expansion, FP8 sparse
attention reader, cache writer, and cache layout.

The first deliverable is a kernel benchmark, not a default runtime route. It
compares the new kernel directly with the installed CUTLASS score function on
identical synthetic paged inputs. It checks the finite score values, masks,
visible counts, and top-k sets before timing synchronized HIP events. Required
cases are context-free batch 4 and 16 at 4096 input tokens, plus batch 4 with
8192-token history and a 4096-token chunk. Decode and mixed batches are outside
the specialized kernel and remain on their current paths.

## Advancement gate

Advance from the benchmark to serving integration only when all correctness
checks pass and the new score kernel's median time is at least 10% lower than
CUTLASS in the batch-16 4096-token target case over at least three independent
rounds. A top-k difference is acceptable only when it is explained by a
measured cutoff near-tie within the observed numeric error.

If the gate passes, add an opt-in runtime route for prefill-only batches.
Decode, speculative decode, mixed batches, and any unsupported shape fall back
before launching the new kernel. Then run three warmups and five measured
serving A/B rounds at batch 16, input 4096, output 1024 with dense prefill
disabled on both sides. Report TTFT, TPOT, throughput, failures, and any HCU
VMFault. Do not enable the route by default unless the end-to-end result is
reproducible and accuracy remains unchanged.

## Failure and safety rules

- Do not modify the installed vLLM or FlashAttention files in place.
- Do not replace sparse attention with dense attention.
- Do not catch and retry after a GPU kernel fault; reject unsupported inputs
  before launch.
- Keep the prototype opt-in until kernel and serving gates pass.
- On a VMFault, stop that shape, collect the kernel arguments and system log,
  and return to root-cause analysis before another launch.
