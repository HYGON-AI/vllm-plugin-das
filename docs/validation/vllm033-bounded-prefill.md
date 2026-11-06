# Optional bounded Flash prefill: gfx936 validation

For paged BF16/FP16 prefill with more than one query token, this default-OFF
Plugin composition reuses official vLLM C++ `cp_gather_cache` per head and the
public vendor nonpaged varlen API. It bounds capacity at 262144 tokens, keeps
device cumulative lengths and actual caller output, and includes the frozen
policy in official normal/AOT compile factors. Single-token decode, FP8 KV
and larger capacities retain their existing routes. No Core/Adapter source
change or extra platform/general entry point is introduced.

The [machine-readable receipt](../../validation/v0.33.0/bounded-prefill-gfx936-candidate.json)
binds source, wheel payloads, checkpoint assets, protocol and retained raw
artifacts. The runtime commit is `65493627ac2c6b34f3e747c5490bd3a039d3c1d9`;
the follow-up documentation commit does not change its 15 Python files.

## Functionality and precision

The preceding CPU suite passed 133 tests; an affected two-file suite passed 85
(overlapping counts). After the source-only collection review, the full
non-device suite passed 281 tests with zero skips, and all 11 separate
public-selector lifecycle cases passed. A regression subprocess rejects all real vLLM imports while
collecting the gather contracts; AOT checks explicitly load the target source.
All 15 production Python files remain byte-identical to the measured runtime.
The installed device tier passed 70, zero skips,
including four new BF16/FP16 x HND/NHD cases. These cover ragged/empty query
rows, 63/64/65/129-token contexts, page padding, independent FP32 reference,
sentinel output boundaries and three changed-input/page/length Graph replays
of the complete cache-writer -> gather -> attention dataflow.

The full `/model/Qwen3.5-4B` model uses BF16 weights/KV, TP1, MTP3, default
combo compilation and automatic Graph planning, max length8192, maxseq16,
batch4096, block/prefix64 and official shared Mamba checkpoints. All three
arms complete two four-request concurrency waves, 14 semantic prefix requests
(hits640/1216 with the real576-token manager) and the original isolated
HumanEval32 protocol (temperature0, seed0, thinking off, max2048).

flash-on: raw 12/32, normalized 32/32; flash-off: raw 12/32, normalized 32/32; triton: raw 12/32, normalized 32/32. The frozen normalization removes leading think,
extracts a single code fence and prepends a newline to an already parseable
full function; it does not repair syntax or logic. Raw results are retained.
No small-model waiver is needed to equate the
normalized 32/32 results, and no larger-model acceptance follows from them.

## Performance

Same physical GPU7, sequential ON/OFF/official Triton. Each arm has 384
retained requests: 16 frozen512-token inputs, forced64/128 output tokens,
C1/4/16, one warmup and three measured rounds per bucket. Prefix hits are
zero for these timing requests. Values below are median output tokens/s
across the three measured rounds. Ratios compare throughput directly.

| Output tokens | Concurrency | Flash ON | Flash OFF | Official Triton | ON/OFF | ON/Triton |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 1 | 122.77 | 98.35 | 125.57 | 1.2482 | 0.9777 |
| 64 | 4 | 387.20 | 317.18 | 390.77 | 1.2207 | 0.9909 |
| 64 | 16 | 1028.57 | 915.92 | 1028.18 | 1.1230 | 1.0004 |
| 128 | 1 | 151.16 | 115.13 | 154.13 | 1.3130 | 0.9808 |
| 128 | 4 | 516.35 | 404.21 | 520.24 | 1.2774 | 0.9925 |
| 128 | 16 | 1553.64 | 1272.19 | 1590.19 | 1.2212 | 0.9770 |

The 384 ON/OFF request bodies and full output token sequences match exactly.
The receipt retains the actual ON/Triton token comparisons as well. Official
Triton remains the default; these results do not change backend selection.
After timing, public profiling observes official C++ gather kernels in ON
and none in OFF. Profiler latency is excluded from the table. Actual worker
target/draft frozen policies and both compile hash factors are inspected;
FULL and PIECEWISE log windows are sampled evidence, not exhaustive coverage.

## Full35B TP2 quality follow-up

The separate [35B quality receipt](../../validation/v0.33.0/bounded-prefill-35b-quality-gfx936-candidate.json)
binds full `/model/Qwen3.5-35B-A3B` (total35B, no under7B waiver) to the same
15-file runtime. Three disjoint TP2 arms use BF16 weights, MTP3, default combo
and automatic Graph planning, maxseq4, batch1024, maxlen8192 and the `qwen3`
reasoning parser. Each completes two four-request waves,14 semantic prefix
requests and the original isolated HumanEval32; all32 finish with stop.

| Arm | KV dtype | Raw HumanEval | Existing normalized HumanEval | Manager page | Prefix hits |
| --- | --- | --- | --- | --- | --- |
| Flash flag ON | BF16 | 30/32 | 32/32 | 1088 | 1152,2240 |
| Official Triton | BF16 | 29/32 | 32/32 | 1088 | 1152,2240 |
| Flash flag ON, gather bypass | E5M2 | 30/32 | 32/32 | 2176 | 2240,4416 |

Both actual workers' source identities and frozen normal/AOT policies are
checked; all three MTP positions have accepted tokens. Independent review
rechecks full checkpoint bytes, Core3656/Adapter213/Plugin15 payload files and
original request/scoring artifacts. A supplemental pre-request live-worker
snapshot binds actual mapped Flash/Lightop DSO bytes;195 provider files match
after execution. This is limited to those provider files, not every system DSO.

All APIs exit0 and release owned groups/ports/GPUs, but each logs one leaked
shared-memory object. Clean lifecycle remains open. These parallel runs on
different cards prove scoped functional/quality behavior; same-card35B
ON/OFF/Triton performance is recorded separately below. Those later traces
prove sampled gather dispatch; they do not retroactively profile these parallel
quality runs.
E5M2 retains the existing path and does not execute this BF16/FP16 gather.

## Full35B TP2 performance follow-up: acceptance remains open

The [same-card35B receipt](../../validation/v0.33.0/bounded-prefill-35b-performance-gfx936-candidate.json)
records sequential Flash gatherON/OFF/official Triton on physicalGPU0/4, using
Core `0.33.0+das.da8b48faa.dtk2610` and the merged Plugin's same15 runtime files.
BF16 weights/KV, MTP3, defaultcombo/autoGraph, maxseq4, batch1024, maxlen8192
and qwen3 parsing are retained. C16 queues behind maxseq4; this is not saturation.
Each arm retains384 strict-valid requests:96 warmup and288 measured, using16
exact512-token inputs,64/128 output tokens, C1/4/16 and three measured rounds.
Fresh TCP per POST includes connection setup; every round records16 created
and0 reused connections, with no retries. Timing and sources differ from the
failed pooled attempt and the4B run; compare only these matched arms.

| Output tokens | Concurrency | Flash ON | Flash OFF | Official Triton | ON/OFF | ON/Triton |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 1 | 87.24 | 80.37 | 87.88 | 1.0855 | 0.9928 |
| 64 | 4 | 180.53 | 189.48 | 179.76 | 0.9528 | 1.0043 |
| 64 | 16 | 125.87 | 145.55 | 216.32 | 0.8648 | 0.5819 |
| 128 | 1 | 125.00 | 112.38 | 128.13 | 1.1123 | 0.9756 |
| 128 | 4 | 267.22 | 219.50 | 229.68 | 1.2174 | 1.1634 |
| 128 | 16 | 207.65 | 299.08 | 246.25 | 0.6943 | 0.8433 |

These are median output tokens/s across all three retained measured rounds.
Performance remains **HOLD**: at C16, ON/OFF is0.8648/0.6943 and ON/Triton
is0.5819/0.8433 for64/128 outputs. C4 rounds also vary; the receipt keeps all
round values and min/max ratios. The OFF64/C1 warmup and approximately0.537s
of measurement round0 overlap a source-CI wall window. That window does not
prove instantaneous CPU causality; the samples were neither dropped nor
retried, and fully isolated speedup for this bucket needs a matched repeat.

Request bodies match384/384. Full returned token sequences match251/384
ON/OFF and248/384 ON/Triton; differing MoE/MTP token paths and round variation
prevent treating this table as a token-identical gather-kernel causal estimate.
The original isolated HumanEval32 remains a separate quality gate: raw29/28/28,
existing normalized32/32/32, all stop, with no under7B waiver. All three arms
also complete14 semantic prefix and8 concurrent requests, sampledFULL and
PIECEWISE dispatch, and positive accepted tokens at all three MTP positions.

Post-timing profiling verifies both actual TP ranks through official
`distributedInfo`, Core version, Worker_TP startup PID, process name and physical
GPU. ON contains88 official gather kernels across the two worker traces;
OFF/Triton contain0. The first reader incorrectly required filenames beginning
with `rank`; the official topology-prefixed traces existed. Its failure is
preserved, and the corrected reader passes six real-trace and nine negative
identity controls before reviewing the original retained data without model
reruns. CPU AsyncLLM traces do not substitute for a worker rank.

Core3656/Adapter213/Plugin15 payloads and the full checkpoint are independently
rechecked;195 previously observed provider files remain byte-identical.
Every API exits0 and releases owned groups, ports and GPUs, while each retains
one shared-memory leak warning. Clean lifecycle, performance acceptance and
independent restarted-server repeats remain open. Official Triton stays the
default; the optional gather policy stays defaultOFF. Prior4B gains are not
full35B performance acceptance.

## Operational limits and reproduction

All three APIs exit0, owned process groups are empty and ports closed. Any
semaphore/resource-tracker warnings are retained in the receipt; resource
release does not establish clean lifecycle. These are one-server warm-run
measurements, not independent startup repetitions or a saturation claim.
OFF reuses only verified terminal-ON Triton compiler kernels; ordinary and
AOT caches are fresh. No startup latency comparison is made.

Use the isolated installed Plugin package, explicit target v0.33 site-packages
and `VLLM_PLUGINS=hcu_adapter`; do not accidentally import a system v0.28
platform plugin. The CPU tests are `tests/vllm033/test_prefill_gather.py`,
`test_flash_attention.py`, `test_flash_activation.py`, `test_rms_compile_cache.py`
and `test_package_boundary.py`; set `VLLM_TARGET_ROOT` to the actual Core
checkout. The installed device suite comprises `test_flash_attention_hcu.py`,
`test_e5m2_scales_hcu.py`, `test_operator_graph_hcu.py`, `test_rms_norm_hcu.py`
and `test_aiter_int8_hcu.py` under `tests/vllm033`, with their explicit HCU opt-ins.
The four new cases are `test_bounded_prefill_gather_ragged_writer_reader_graph`.

Enable with `VLLM_HCU_FLASH_PREFILL_GATHER=1`; unset or set0 and restart for
rollback. It is independent of RMSNorm and requires the existing Flash
extension/provider. No FP8 gather, DCP or additional-architecture support is
implied. The scoped35B results above do not close lifecycle, performance or the remaining
migration feature gates.
