# v0.33 hybrid BF16 quality investigation

Review remains **HOLD** for PR330. Qwen3.5-0.8B still has new failing
HumanEval tasks relative to the official Adapter-only baseline. No production
Core, Adapter, Plugin or installed vendor code changed in this investigation.
The original E5M2 and vendor-Triton INT8 numerical gates also remain open.

The frozen checkpoint is `/model/Qwen3.5-0.8B/Qwen3.5-0.8B`, on gfx936,
BF16 weights/KV, TP1, Graph sizes1/2/4, hybrid prefix enabled, manager576 /
physical attention page64, combo kernels explicitly disabled. This compiler
profile does not certify the previously failed default compiler policy.

## Full-model evidence and exact task sets

| Profile | Official normalized | Plugin normalized | Plugin new failures | Plugin improvements |
| --- | ---: | ---: | --- | --- |
| Original MTP3 | 15/32 | 14/32 | /0 | none |
| Diagnostic MTP3 | 15/32 | 14/32 | /20 | none |
| MTP off diagnostic | 13/32 | 13/32 | /2 | /20 |

All six arms have raw1/32, all32 normal stops and server exit0. Each paired
comparison has identical checkpoint hashes and all32 request bodies. The
diagnostic MTP3 pair swaps physical GPUs and adds logprobs/top-logprobs and
token-ID response fields relative to the original pair. It is not an exact
repeat of the original request protocol. Official messages reproduce32/32;
Plugin messages reproduce29/32, changing tasks0/20/31. The original failure
and these changed failure sets are separately retained. Neither the equal
MTP-off totals nor the improved task0 waives a newly failing task.

In diagnostic task0, the first differing generated token follows207 common
tokens, giving a363-token full input. Official selects token361 (` <`),
Plugin2564 (` <=`); their raw logprob margins are +0.125 and -0.125.
Two fresh full-prefix one-token replays in both services select361; Plugin
has a raw tie, official retains +0.125. Diagnostic Plugin task0 eventually
passes with a different implementation despite this token split. Thus this
observation does **not** explain the original failing completion or identify
the sole cause of the current task20 regression.

## Independent checkpoint and same-input operator diagnosis

Public Transformers eager attention on the frozen full prefix has raw BF16
logits27.25/27.125 for361/2564. A separately registered FP64 attention
reference yields final BF16 logits27.125/27.125, a tie. Other model operations
remain BF16. SDPA selects361 too, but logs a Torch cutlassfa Adapter: it is
not independent vendor-kernel evidence. These are fresh eager prefills,
without vLLM incremental hybrid state, speculation or serving Graph replay.

The public HF attention callback snapshots real Q/K/V at all six full-attention
layers3/7/11/15/19/23. The saved snapshot is hashed. All72 operator cases use
the same Q/K/V, randomized physical page tables and exact cache-writer checks.
Queries of length1/2/4/363 compare the same final logical token:

| Provider | Final-token words differing from query-length1, across six layers | Maximum difference |
| --- | ---: | ---: |
| Official Triton | 0 for each length | 0 |
| Current Plugin/vendor | 3860/12288 for each of2/4/363 | 0.03125 |
| Separate valid max-query upper-bound regular-prefill diagnostic | 0 for each length | 0 |

This establishes a query-shape numerical dependence on these captured inputs.
It does not establish a unique serving failure cause or a new accuracy tolerance.
FP64 errors and individual rows remain in the original result JSON.

## Why no blanket reroute is shipped

The installed vendor wrapper selects direct paged attention for M1. The
Plugin's existing upper-bound adjustment routes rectangular M2..15 through
regular prefill to avoid an incompatible direct-paged tile. That regular
vendor path reserves at least262144 contiguous KV tokens and reads a device
length sum outside Graph capture. With actual two KV heads, dimension256 and
BF16, K+V workspace alone is512MiB per call. Forcing every decode into that
route needs memory and latency evidence in addition to numerical tests.

A separate public **already-contiguous** varlen diagnostic completes24 cases
and72 changed-Q/K/V Graph replays, all word-exact against same-input eager.
All six layers also preserve the same final-token output across query lengths.
Largest measured incremental Graph peak is1500160 bytes in this bounded
experiment. It excludes paged gather and cannot establish serving memory,
performance or model quality. The installed public non-paged wrapper ignores
`out`; this experiment explicitly copies into the caller buffer. An implementation
must preserve that contract. The existing E5M2 helper applies scales; it does
not implement bounded paged gathering. No new gather kernel, vendor private-call
copy or global rebinding is shipped.

Next implementation work must compare a bounded paged route with the current
native path, including layouts, ragged/empty/padded batches, changed metadata,
Graph, latency/memory, and the original MTP3 full-model protocol. An operator
diagnostic alone does not close the original quality gate.

## Separate BF16 MoE ownership audit

Official v0.33 already owns `AiterUnquantizedExperts`. Adapter-only startup
reports actual HcuPlatform but the official CDNA admission rejects AITER;
the dedicated fused-MoE flag does not enable this path. A separate direct CPU
metadata ABI diagnostic bypassing admission finds that official neutral
`swiglu_limit=0.0` is rejected by the installed vendor wrapper, which requires
`None` for that route. No GPU kernel or model was accepted. This is an
Adapter narrow admission/neutral-ABI task, not a duplicate Plugin BF16 MoE
algorithm. Global CDNA identity spoofing would enable unrelated paths and is
not an acceptable fix. QNT-005 remains partial.

## Records and cleanup

`validation/v0.33.0/hybrid-quality-gfx936-candidate.json` records task sets,
checkpoint hashes, exact request comparisons, wheel/14-file source and both
installed environment checks, immutable artifact hashes and scope limits.
Raw artifacts remain under the recorded workspace `evidence/` directories;
only summaries/hashes are committed to this PR. The scripts are retained as
experiment material, not production hooks.

All owned services/probes exited, ports18161–18164 closed and all eight GPUs
returned to2MiB. Protected GPUs5/6 and unrelated services were untouched;
no device reset. The148-row migration matrix is1 passed /46 partial /
101 not-run. Goal remains v0.33 active; `last_good=false`. AI assistance was used.
