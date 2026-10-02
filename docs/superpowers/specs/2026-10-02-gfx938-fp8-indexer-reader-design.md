# gfx938 FP8 sparse-indexer reader design

## Intent and scope

Make GLM-5.3-Flash W8A8 sparse-indexer decode read the actual HIPC KPool
layout correctly on BW1100 without copying a full batch of long-context keys
on every decode step. Keep the change in one focused `vllm-plugin-das` MR,
based on `origin/v0.28.1-dev` (`9788dc6`). It may change the FP8 paged-QK
reader and its dispatch, plus focused tests and a kernel benchmark. It must
not change the KPool writer/gather ABI, block-table construction, sparse-MLA
attention proper, MTP scheduling, or the separate KPool page-table extension.

The user wants vLLM accuracy aligned with the working SGLang deployment,
not a score improvement obtained by silently changing parallelism, model,
evaluation settings, or numerical format. W8A8 remains enabled.

## Evidence and current behavior

- On the same `9788dc6` base and the same 30-question HMMT25 EvalScope
  configuration, unpatched vLLM scored 7/30 and 8/30; the FP8-reader-only
  worktree scored 22/30 twice. SGLang scored 21/30 once. The runs used
  batch 8, temperature 0, top_p 1, max_tokens 30000, thinking_budget 30000,
  and seed 42. The earlier 24/30 result came from a different, unmerged
  PR #170 base and is not a same-base comparison.
- After matching question index and target, the SGLang run and the two
  patched-vLLM runs agreed on pass/fail for 27/30 and 29/30 questions;
  index 21 was the only stable disagreement (vLLM correct). The aggregate
  score therefore does not establish a remaining vLLM-only accuracy defect.
- The HIPC writer stores 16/32/64-token physical pages with K bytes
  preshuffled inside each page and per-token FP8 scales after the K bytes.
  vLLM can expose a collapsed page view; `_indexer_cache_as_hipc_view`
  restores the physical page axis without a copy. The current gfx938 patch
  then calls `_linearize_preshuffled_paged_cache` on every decode and passes
  a temporary token-major cache to LightOp.
- A true physical page size of one must not enter that linearizer. With
  AITER enabled, its existing stage-1 path matched the GPU reference at
  lengths 2, 513, and 32768. With AITER disabled, the current dispatch
  reaches LightOp, which rejects `block_kv=1`; the existing Torch paged
  reference matched the GPU result at lengths 513 and 32768.
- On BW1100, page size 32 and context 32768, the current full wrapper versus
  LightOp on an already-linearized input took 0.200/0.024 ms (batch 1),
  0.732/0.055 ms (batch 8), and 5.030/0.268 ms (batch 64). The wrapper's
  extra peak allocation at batch 64 was about 784.82 MiB. These HIP-event
  numbers are diagnostic, not an end-to-end serving benchmark.

## Considered approaches

1. **Chosen: direct layout-aware paged reader on gfx938.** Read the HIPC
   preshuffled bytes and scales through the existing block table, compute
   weighted FP8 QK logits, and write the existing logits ABI. This preserves
   the writer, prefill gather, cache allocation, prefix-cache and MTP layout.
   The implementation cost is a new GPU reader whose numerical behavior and
   performance must be proven.
2. Change the gfx938 writer and prefill gather to token-major layout, then
   call LightOp directly as SGLang does. This avoids a decode copy but changes
   the persistent cache ABI and every consumer of it. It is rejected for this
   MR because a mistake can corrupt prefix caching, prefill, or mixed MTP
   batches even when a decode microtest passes.
3. Keep full-cache linearization. This already improves model scores, but its
   O(batch × context × head_dim) work and allocation on each decode are too
   costly for a production fix. It remains a temporary numerical oracle, not
   the default supported path in the proposed MR.

## Reader contract and dispatch

`rocm_fp8_paged_mqa_logits` retains its existing inputs and output:
FP8 query `[B, next_n, H, D]`, packed KPool, weights, context lengths,
logical-to-physical block table, schedule metadata, and maximum model
length in; float32 logits `[B * next_n, max_model_len]` out. Invalid or
causally future token positions remain negative infinity. No result may
depend on stale or unused block-table entries.

On gfx938 with a restored physical page size of 16, 32, or 64, a new
layout-aware reader is selected. For logical token `t`, it resolves
`page_id = block_table[row, t // page_size]` and `slot = t % page_size`.
The K-byte address follows the HIPC 16-by-16 tile arrangement, while the
scale address remains token-major at the tail of that physical page. The
reader dequantizes K with its token scale and reproduces the existing
per-head QK, ReLU, weight, head-reduction, and causal-mask semantics.
It produces final logits directly, without a batch-wide key tensor or a
rewritten block table. Any scratch is bounded independently of
`batch × context × head_dim`; the unavoidable logits output is excluded
from the additional-scratch metric.

On gfx938 with true physical page size one, keep the already verified
AITER stage-1 route when available. If AITER is unavailable and the caller
has not required it, use the existing Torch paged reference as a correctness
fallback; never pass page size one to LightOp. `force_aiter_triton=True`
must either select an actually supported AITER kernel or raise a clear
unsupported-backend error; it must not silently execute LightOp or the new
reader. Other architectures retain their current dispatch. Unsupported
physical page sizes or packed layouts fail visibly before a GPU read rather
than being reinterpreted as a supported format.

The reader does not remap, extend, or infer missing KPool block-table entries.
The separate page-table issue remains a separate investigation and must not
be claimed fixed by this MR.

## Failure handling and rollout

Kernel selection is decided before execution and is graph-capture-safe;
there is no catch-and-retry after a failing GPU kernel. A missing optional
backend uses only a path whose input layout is proven compatible. A layout
or capability mismatch is an explicit error, not a silent precision change.
The current linearizing implementation may be retained only as an opt-in
diagnostic comparison while the new reader is validated, then removed from
the production selection before proposing the MR.

If the direct reader fails the correctness or performance gates below,
do not submit the MR as an optimization or switch the persistent KPool
layout inside this scope. Return to design review with the measured failure.

## Validation and MR gates

1. Extend the nearby runtime-patch tests rather than adding a large mock
   suite. Cover physical page sizes 1/16/32/64, collapsed and explicit page
   views, non-identity and partially invalid block tables, causal lengths,
   batch and `next_n` padding, backend availability, and forced-AITER
   dispatch. Keep one behavior per test. A page-one/no-AITER regression must
   fail before its fix and pass after it.
2. On BW1100, compare direct-reader logits with an independent token-order
   reference for batch 1/8/64 and lengths 2/513/32768. Check finite values
   with `rtol=1e-3, atol=1e-2`, exact negative-infinity masks, and selected
   top-k indices; investigate any near-tie differences rather than accepting
   a changed answer on an aggregate score alone. Verify prefill and decode,
   prefix-cache reuse, MTP, and eager versus graph execution.
3. Put performance measurements under `benchmarks/kernels/`, not `tests/`.
   At page size 32, length 32768, batch 64, measure synchronized median
   over at least three independent runs. The provisional acceptance gate is
   at most 1.7 ms for the full reader and below 128 MiB additional peak
   allocation beyond its logits output, compared with the current 5.030 ms
   and 784.82 MiB. Record batch 1 and 8 too, including device, versions,
   warmup, graph/eager mode, and timing method. Failure requires design
   review, not benchmark cherry-picking.
4. Re-run the same 30-question, batch-8 HMMT25 EvalScope command twice on
   the exact branch to be submitted. Compare total numeric accuracy and
   per-question pass/fail, answers, finish reasons, and output lengths with
   both existing same-base 22/30 runs. Investigate new repeatable failures
   before proposing the MR. State separately that the single SGLang run was
   21/30 and the old PR #170 run is not same-base evidence.
5. Run relevant targeted pytest through `uv` or the prepared venv, lint and
   `git diff --check`. The human submitter must review every changed line,
   personally run the relevant tests, check duplicate issues/open PRs, and
   disclose AI assistance, commands, results, and model eval in the MR.
   If an open PR already implements the same fix, do not create another.

The existing 88 passing targeted tests and two 22/30 model runs validate
the pre-direct-reader snapshot only; they do not certify the final code.
