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
implied. Full35B TP2 BF16/E5M2 quality and the remaining migration feature
matrix stay separate and open.
