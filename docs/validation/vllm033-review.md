# v0.33 Plugin review and follow-up validation

Subsequent [INT8 math follow-up](vllm033-int8-math.md) fixes the default ASM
quantization/activation compatibility in `529855a5`; its installed strict EP
checks now pass. The ASM failures below remain the original review snapshot.
Unchanged vendor-Triton and model quality gates still prevent merging PR330.

Review base: `153e4d9eb88a82f3c59a6fd5e87d790c59675104` on `v0.33.0`.
Reviewed production revisions: hybrid admission fix `597bbaa8`, then EP map fix
`ecf67052`. Both belong to existing [PR330](https://github.com/HYGON-AI/vllm-plugin-das/pull/330)
from the authorized fork. The review covers the production diff, public attention
contracts, publication/health checks, RMS compile factors, INT8 dispatch and the
installed evidence. AI assistance was used. This is not a compatibility lock;
`last_good=false`. The [candidate JSON](../../validation/v0.33.0/review-continuation-gfx936-candidate.json)
retains the distinct build identities and failures.

**Merge decision: hold PR330.** The user authorized merging after a clean review.
The following directly relevant accuracy/numerical gates are still unresolved.
They are concrete PR gates, rather than a requirement to finish unrelated Core
features before reviewing Plugin work. No merge has been performed.

## Findings and fixes

1. **Hybrid manager admission, fixed in 597bbaa8.** The Plugin rejected every
   framework block except64, including the official aligned576 manager of
   Qwen3.5-0.8B. Remove that override and inherit the public admission rule.
   The exact kernel-page declaration stays `[64]`; official physical-page
   selection splits the manager into kernel pages. No Core or Adapter patch.
   Target-source admission/selection tests cover64/128/576/1088/2176. The
   rebuilt installed Plugin passes52 KV/attention/changed-input Graph tests.
2. **Solution-specific EP ABI, fixed in ecf67052; acceptance remains partial.**
   Applying the ASM/CK ownership-mask conversion to vendor Triton/MoE-C loses
   remote `-1` entries and local indices. Preserve canonical global-to-local
   maps for those index-based solutions; unknown EP solution ABIs fall back to
   official Triton before vendor execution. Eight new CPU regressions fail
   before the fix. All38 focused INT8 contracts pass after it. Full portable
   suite:245 passed,58 skipped,2 fork deprecation warnings. The old installed
   provider's forced vendor-Triton EP probe exits139 with invalid-address/
   VMFault; its uninstrumented native phase is not uniquely attributed from
   that log alone. The fixed provider completes all16 finite cases below.
3. **Strict INT8 EP numerical gate remains OPEN.** For global512/local256
   experts, linear/round-robin mappings on both logical ranks, tuned ASM and
   a diagnostic forced public vendor-Triton config, M16 passes all8 eager and
   24 changed-input Graph checks. M128 passes eager and the first two replays,
   but the third replay's maximum absolute error is0.00134–0.00171 against the
   original0.001 limit. All8 M128 cases therefore fail the strict gate. A
   separate same-input check finds Graph and eager bit-identical in all24
   M128 replays. Stale Graph inputs are not needed to explain this difference;
   the vendor/reference numerical discrepancy still needs diagnosis. No limit
   or input range was relaxed. No distributed dispatch or full MoE model claim.
4. **Model accuracy remains OPEN.** Existing Qwen2 Flash E5M2 normalized13/32
   versus BF16 Flash16/32, and Qwen3-1.7B E5M2 27/32 versus BF16 28/32, remain
   unaccepted. The new Qwen3.5-0.8B BF16 pair is15/32 official Triton versus
   14/32 Plugin Flash, newly failing /0. All32 stop normally. The failing code
   uses `<= threshold` where the task requires `< threshold`; this is a real
   generated-code failure, rather than token-budget exhaustion. Mechanism and
   prefix checks do not waive it. Repeat/aligned-token diagnosis remains open.

## Full hybrid model observations

Both arms use the same local Qwen3.5-0.8B checkpoint with its real MTP weights,
MTP3, prefix caching and official FULL_AND_PIECEWISE Graph. The observed four
cache groups have manager576; hash/hit alignment is64. Official Triton exposes
attention tensors with576-token blocks, while Plugin Flash exposes64-token
physical blocks (13,806 blocks from1,534 manager blocks). Eighteen Mamba layers
retain their official state backing. The official source already seeds resumed
state from `mamba_block_size`; no Plugin reseeding wrapper is introduced.

At common-prefix lengths640 and1216, owner/junction/third-sibling streams report
warm cached-token counts0/640/640 and0/1216/1216. A repeated third sibling retains
that hit. Every cold/warm/replay output matches its corresponding prompt. These
positive64-aligned hits are inside576-token manager pages. Both arms also pass
four smoke requests and two waves of four concurrent requests.

This completed profile explicitly sets Inductor `combo_kernels=False`. Graph,
prefix and MTP remain enabled. The original official-control default-combo run
and the repaired Plugin default-combo run both expire their1800-second readiness
budgets during LLVM draft-head compilation with zero inference. Its stacks and active compiler evidence are preserved; the
separate compiler-policy retry does not certify the default policy. The initial
Plugin exact64 manager rejection is also preserved. Native counters show
4848 drafted/4108 accepted tokens for control and4812/4072 for Flash; snapshots
may lag final requests. One trained MTP layer is reused for three positions.
Both servers exit0 with resource-tracker semaphore warnings retained. Raw
HumanEval scores are1/32 for both, kept separate from normalized15/14.

## Official MTP1/2 observations outside the Plugin merge gate

GLM-4.7-Flash TP2+EP2/TRITON_MLA, official Triton BF16 MoE/RCCL, without Plugin:

| MTP tokens | Raw | Normalized | Drafted / accepted | Accepted by position |
| --- | --- | --- | --- | --- |
| OFF | 0/32 | 31/32 | none | none |
| 1 | 0/32 | 29/32 | 2459 / 2403 | 2403 |
| 2 | 0/32 | 31/32 | 4268 / 2626 | 2103 / 523 |
| 3 | 0/32 | 31/32 | 6432 / 2709 | 2097 / 526 / 86 |

Checkpoint hashes and all32 request bodies match. MTP1 newly fails /11,/19:
its generated XOR drops leading zeros, and its sorting function returns digits
rather than the required number words. All32 requests stop normally. MTP1 uses
the same physical pair as OFF/3; MTP2 uses another explicit physical pair, so no
timing/performance inference is made. Both new arms complete Graph/prefix,
smoke and two four-request concurrent waves and exit0; shared-memory warnings
remain. This new official-path quality difference needs its own repeat and
sampling/kernel diagnosis, rather than a speculative Plugin/Core patch.

Seven additional installed official CPU draft-buffer waves pass stable-address,
nonaliasing and inactive/dummy `-1` checks. They do not prove constructor,
device-stream ordering, dedicated Graph or DP acceptance. Native AITER AR,
W4A8 lifecycle, sparse MLA, other architectures and broader model/topology/
performance requirements remain separate unfinished tasks.


## Hybrid-quality follow-up

See [hybrid-quality investigation](vllm033-hybrid-quality.md) and
`validation/v0.33.0/hybrid-quality-gfx936-candidate.json`. Original MTP3 new
failure/0 remains recorded. The diagnostic pair has15/14 with new failure/20;
MTPoff13/13 still loses/2 and gains/20. Same-input operator query-shape
dependence is confirmed, but the continuous-KV Graph diagnosis excludes
paged gather and model quality. No production fix or merge is claimed.
The earlier review's failure IDs describe its frozen original profile; they
are not silently replaced by these diagnostics. BF16 MoE admission/neutral
ABI work belongs to Adapter around the official expert and is not an added
Plugin merge prerequisite. All owned jobs exited and latest ports18161–18164
are closed. Goal remains active;1passed/46partial/101not-run,last_good=false.
