# Explicit E4M3 KV cache with HCU FlashAttention on v0.33

The HCU backend now accepts `--attention-backend FLASH_ATTN
--kv-cache-dtype fp8_e4m3 --block-size 64`. Query computation remains BF16 or
FP16; cached K/V use OCP E4M3 bytes. `fp8_e4m3fnuz` is a different encoding,
and the ambiguous `fp8` alias is not added by this change.

Previously the backend declared only unquantized cache names and explicit
E5M2, so the official selector rejected E4M3 before loading model weights.
Adding the dtype name alone is insufficient: the tested vendor paged-prefill
wrapper casts Q to the cache dtype and its E4M3 prefill kernel rejects head
size 64. A separate direct probe also found that its mixed-dtype native gather
did not preserve the expected E4M3-to-BF16 values on the tested build.

Core continues to own metadata, allocation, block tables and scheduling. The
existing native `reshape_and_cache_flash` writes E4M3 with the supplied scales.
The Plugin explicitly views one-byte storage as `torch.float8_e4m3fn` at the
reader boundary. For prefill, Core's `cp_gather_cache` copies exact bytes from
the declared HND/NHD views into bounded, device-length-indexed contiguous
storage; Torch converts those bytes to the query dtype before vendor varlen
attention. The existing descale adapter applies key scales to Q and value
scales to output. Single-token decode without requested LSE retains vendor
paged attention. Explicit decode-LSE requests use the same bounded byte-gather
and unquantized varlen composition: the vendor paged fast path otherwise drops
the requested normalizers, and its finite-window fallback quantizes Q to E4M3.

Prefill groups sequences against a 64-MiB target for the combined FP8 gather
and BF16/FP16 conversion buffers. A single maximum-length context is the
minimum group: peak KV scratch is bounded by the larger of this target and
that one context with its rounded KV tile, plus query-sized compaction/scatter
buffers. It does not
multiply one long context's storage by every sequence in a mixed batch.
For 256 sequences, max context32768, eight KV heads and head128, the KV
scratch bound is192MiB rather than48GiB. Query compaction, rebased cumulative
lengths, output and LSE scatter use device offsets and retain changed-length
Graph replay and output padding. Initialize backing storage through the final
256-token vendor KV tile as well: masked V lanes beyond the logical allocation
otherwise propagate NaNs under changed-length Graph replay. This padding is
included in the byte budget. Extra group calls/copies remain a performance
tradeoff; this is not a native-FP8-compute performance claim.

This keeps the vendor attention kernels and original BF16/FP16 query precision;
it does not change Core attention selection, add an attention algorithm, replace
the native cache writer or alter E5M2 behavior. Remove this composition when a
vendor mixed-dtype E4M3 prefill path passes the same byte, numerical, Graph and
model gates with BF16/FP16 queries.

Validation must distinguish these independent gates:

- Portable admission, output-buffer identity, dtype and layout contracts.
- Installed-wheel native writer bytes against an independent E4M3 cast,
  non-unit scales, paged prefill/decode and changed-input Graph replay in both
  HND/NHD layouts with BF16/FP16 queries. The test matrix includes head sizes
  64, 128, 192 and 256; an unexecuted case is not qualified.
- A same-checkpoint BF16-KV/E4M3-KV model pair, explicit `FLASH_ATTN`, default
  Graph and prefix caching, original HumanEval 0–31, preserved raw and
  normalized results, and the original shutdown deadline.

## Recorded validation (2026-11-18)

The original publication snapshot below precedes the scratch correction.
Review identified its batch-times-maximum-context amplification even though
the original small native/Graph matrix and exact-head CI passed. Keep those
results bound to their original source; the correction adds the reported
48-GiB metadata-shape regression, forced grouping, empty query rows, query
and KV length changes during Graph replay, padding and independent LSE checks.
The subsequent decode-LSE review found the original native matrix requested
LSE only for multi-token queries. A new native negative control confirms the
old single-token path returns only a Tensor; decode with and without LSE now
has separate eager/Graph coverage in both layouts and all advertised dtypes
and head sizes. Keep that failure and requalify this functional follow-up.

Local correction validation uses installed Core `c440c08` on gfx938 and Adapter
`1c9448`: 131 affected portable cases and 80 native writer/reader/Graph cases
pass. Native cases poison uninitialized allocations and change query/KV lengths
without changing addresses. A separate real 256-sequence/32768-max-K case
passes with 192.07 MiB peak extra allocated device memory (including vendor
and query buffers), against a 768-MiB guard. Original failures and independent
vendor canary probes remain preserved. Full exact-head CI is required after
publishing the correction; current CI status is recorded on the MR. These
component results do not claim a new model/HumanEval or lifecycle pass.

Core and Plugin both use the 0.33 release line. The paired installed Core is
`98a09850d56b4ab99a8c9bc4b28d411b72ed5147`, and the Plugin base is
`13bc3236e0dbe1110e5de5ef63d9fec46917d0d2`. The installed candidate Plugin
wheel SHA256 is `7fb6fa9fa5fba104ee888668f15a93b7ba7f6839e9057dc56b4461e623029c37`; its executable Python payload
matches the reviewed source. Adapter remains independently versioned at
`0.5.22+dtk2610.torch2130.2611172048.g1c9448`.

- 126 portable admission/Flash/activation cases passed.
- 8 source/packaging cases passed.
- 123 CI selection cases and 99 subtests passed. The release device tier
  automatically discovers `tests/vllm033/test_e4m3_attention_hcu.py`; this
  change does not add entries to the legacy HCU job registry.
- 32 installed native cases passed without skips: both cache layouts,
  BF16/FP16 Q, head sizes 64/128/192/256, prefill and single-token decode,
  independent exact E4M3 writer bytes, non-unit scales, and changed-input
  Graph replay. Prefill checks the output and an independent FP32 LSE.

Qwen3-8B uses TP1, BF16 weights/Q, explicit `FLASH_ATTN`, block64,
model length4096, sequences16, batched tokens1024, memory utilization0.7,
seed0, default `FULL_AND_PIECEWISE` Graph and prefix caching. HumanEval
keeps the original IDs0–31, temperature0, seed0, max_tokens2048 and
thinking disabled with the matching Qwen3 reasoning parser.

| KV cache | Raw HumanEval | Existing format normalization | Cache tokens |
| --- | --- | --- | --- |
| BF16 (`auto`) | 20/32 | 30/32 | 203,072 |
| OCP E4M3 (`fp8_e4m3`) | 20/32 | 31/32 | 405,888 |

Both arms completed all requests with `stop` endings, passed arithmetic,
prefix-reuse and two four-request concurrency waves, and retained nonzero
prefix hits. Full checkpoint and installed package records were verified
before/after each arm; both ports and all observed device owners were
released. Raw scores remain distinct from normalized scores.

The original 15-second EngineCore shutdown deadline remains unchanged.
Both arms report nanobind reference leaks and forced process cleanup;
all lifecycle diagnostics are preserved in the local pair review. Process
exit is not a clean lifecycle pass. At original publication the increment was
Draft pending lifecycle resolution and target-device qualification; consult the
MR for its current review state. These shared limits remain separately tracked.

The observed device is **gfx936**, not gfx836. These results do not qualify
gfx836, other checkpoints, softcap, context parallelism, or full release
acceptance. No production performance claim is made for the extra prefill
byte gather/conversion. The original paired requests, raw responses, scores,
runtime source hashes, profiles and shutdown observations are retained in
`flash-e4m3-v033-20261118-r1` evidence.


## Strict PA provider follow-up (2026-11-28)

The user-specified `2.8.4+dtk2610.torch2130.2611272153.gece162` provider
reproduced `PA tensors must be contiguous` during default-Graph warmup of
Qwen3.6-27B BF16 weights, TP4, E4M3 KV, 64-token kernel pages and prefix on.
The Plugin preserves Core allocation and native writer views. At the PA call,
dense read-only views include intervening half-pages and device block indices
are multiplied by the integral page-stride ratio. Each real page keeps its
original address. Query, output and metadata are normalized only as needed;
caller output aliases and input tables are preserved. A subpage padding gap
uses the existing bounded gather/ordinary-varlen composition. Its page table
also needs contiguous columns for the native Core gather.

A separate D256/KV4 E4M3 non-unit-scale test exceeded the existing 0.03
absolute/relative FP32-reference tolerance in both layouts (one of 12288
values, absolute error 0.033935546875). Native writer bytes matched the
independent cast, and contiguous native PA was bitwise identical to remapped
PA. Keep this original failure: zero-copy page remapping does not repair that
native numerical limitation. D256/KV4 E4M3 therefore uses the bounded
unquantized varlen composition; D256/KV1, including the requested TP4 model,
retains native PA. This geometry guard can be removed after native PA passes
the same non-unit-scale, changed-input Graph and model gates. Gather/conversion
scratch and extra calls are a performance tradeoff, not native FP8 compute.

An artificial FP8 cache offset of 13 bytes faulted native PA after a successful
native-writer synchronization. The regression fixture uses a nonzero 64-element
aligned offset, matching the allocator's alignment requirement; those earlier
failed runs are retained separately from the original model failure. No
FlashAttention source or binary is modified by this follow-up.


The corrected installed artifact passes 107 portable cases, 88 native
writer/reader/Graph cases (48 new and 40 existing), and 25 CI selection/environment
cases with 298 subtests on gfx938, Core c440c08 and Adapter/kernel 51bd28. Native
cases retain the existing 0.03 tolerance; no skips or xfails qualify as passes.
The PA regression is selected by both v0.33 device tiers.

The final Qwen3.6-27B BF16-weight TP4 replay retains FLASH_ATTN, E4M3 KV,
MRV2, default FULL_AND_PIECEWISE Graph, prefix, context 4096 and 1 GiB KV per rank.
Startup and output 13/13 / prefix 3/3 pass, with 3200 prefix hits. HumanEval 0–31 is
raw 2/32 and format-normalized 32/32, with 4961 generated tokens and no length
truncation. Raw and normalized pass sets match the saved old-provider BF16
control (4974 tokens); raw 2/32 is not relabeled as32/32. The service exceeds
the original 15-second exit gate and needs owned-only SIGTERM; EngineDeadError
and native binding leak diagnostics remain. These are retained lifecycle gaps,
not clean-exit, performance or whole-upgrade acceptance. Test-owned processes
are gone. The final checkpoint/provider seals and installed RECORDs are recorded
separately from those original failures.

### Optional-output review correction

The D256/KV4 decode fallback initially required `out` even though the previous
native helper path allowed the vendor to allocate it. Allocate a contiguous
query-shaped output at the shared gather entry when it is omitted or `None`,
before selecting chunked or unchunked work. Explicit output aliases are retained.
Sixteen portable regressions cover both layouts, both grouping paths and both
`run_e4m3_attention` / `_run_flash_attention` entry points. All sixteen fail on
reviewed head `2999642` and pass after the fix; the affected portable suite passes
123 cases. These CPU tests substitute gather/vendor leaves and do not qualify
native numerics, Graph replay or models for the new head. Prior installed-tree
results above retain their original provenance. Full remote CI remains required;
run `46130974985` failed collection with missing Adapter interfaces, with no
native PA acceptance. The runtime image, not a skipped test or import fallback,
must supply those dependencies before a meaningful full device rerun.
