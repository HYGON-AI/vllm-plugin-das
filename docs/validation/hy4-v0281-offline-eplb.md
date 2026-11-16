# HY4 v0.28.1 offline EPLB validation

## Review closure, 2026-10-15 UTC

This record covers PR #187 after review head
`95a3be0f76090eb2e7366d3c1bdaafa95578ba42`. The corrective commits are:

- `4b168b710c8e77c348834e121f37135441381509` publishes record-only policy
  results independently of the ROCm online-migration threshold, permits a
  locally dummy EP0 rank to publish globally nonzero load, rejects all-zero
  windows, keeps target/MTP ownership separate, and validates an existing
  artifact before resetting a calibration session.
- `3925c1f29b9758472bc30efe51dd1dc1b37c24da` makes static
  `nearest`/`locality_fair` routing consume the selected primary replica while
  preserving the real replica-count metadata and ordinary online hashing.
- `7123fa10c6d02ff0786326bb98738c882fb9ea56` gives standalone
  `disable_rearrange=true` its lifecycle activation and removes the fixed
  DeepEP low-latency wrapper that bypassed upstream DP synchronization and DBO
  microbatch selection.

The branch remains based on
`38b179181080e00b184a8bd0e3d2508fbce350bf` (`v0.28.1-dev`).

## Runtime boundary

- Model: `/models/Hy4-preview-Channel-FP8-w8a8-v2`.
- Model `config.json` SHA-256:
  `d4a648cb09bb89f4b8778e60629e43618f1abb581ef3aa38bd67b2b2cd998441`.
- vLLM: `0.28.1+dtk2604.torch2110.2609171627.g77acaf`.
- PyTorch: `2.11.0+dtk2604.2609201441.gf74cd7`.
- Python plugin code was imported from this worktree. The installed
  `vllm_hcu` package supplied the existing compiled extension; none of these
  review fixes changes compiled code.
- Eight HCUs report 147,440 MiB each. All owned services were stopped after
  validation and all eight cards returned to 2 MiB used.

The calibration and static profiles used DP8/TP1/EP8, eight redundant experts,
DeepEP low latency, DeepGEMM, MTP3, FP8 E4M3 KV cache, prefix caching, Model
Runner V2, and the default `FULL_AND_PIECEWISE` graph policy. A fixed 512 MiB
KV cache and 256 MiB DeepEP buffer kept the eight-device run within the node's
startup-memory envelope.

Raw evidence is retained under
`/tmp/pr187-review-closure-20261015/` for this validation session.

## Software gates

The fixes followed focused failing-to-passing regressions:

| Scope | Result |
| --- | ---: |
| Offline record/lifecycle suite | 33 passed |
| Static/online physical-ID routing integration | 188 passed |
| Dispatcher, framework, and plugin lifecycle | 112 passed |
| Combined affected gate, excluding the three characterized baseline cases | 569 passed, 3 deselected |

The unfiltered combined gate reported **569 passed, 3 failed**. The failures
are the unchanged HYV4 Transformers attention-alias assertion and its two MTP
parameterizations. They fail identically at `95a3be0`: this installed vLLM
retains `deepseek_sparse_attention`, while the tests expect
`indexed_attention`. The corrective commits do not modify HYV4 attention,
configuration, or MTP model code. `git diff --check` passed. Ruff is not
installed on the host.

The routing regressions execute both the HCU Torch fallback and delegated
upstream-style mapper. For a legal map where logical expert 0 has physical
replicas 0/GPU0 and 132/GPU4, every static rank-0 token resolves to physical 0;
removing the static marker restores hashed distribution across both replicas.
The real replica count remains two.

The DBO regression executes the installed upstream helper with fixed
`deepep_low_latency`. It returns `num_ubatches=2` and a non-null synchronized
DP state. No HCU callback now replaces that helper.

## Record-only calibration

The calibration service reached readiness on all eight API servers and
constructed eight `HcuGPUModelRunnerV2` workers. A driver sent 128 varied,
concurrent prompts containing 227 to 1,283 prompt tokens. All 128 requests
returned HTTP 200. Sixteen request-time policy calculations ran after the
profile calculation; record mode did not transfer or commit live weights.

The resulting artifact is
`/tmp/pr187-review-closure-20261015/hy4-eplb-map.json`, SHA-256
`c648b6d38e7890aaf06820d73eb77bb69e507eef589b221b7f952b08699b6feb`:

| Entry | Layers | Logical + redundant | Row width | State |
| --- | ---: | ---: | ---: | --- |
| `HYV4ForCausalLM` | 77 | 256 + 8 | 264 | `proposal` |
| `HYV4MTP` | 1 | 256 + 8 | 264 | `proposal` |

Both entries retain the exact runtime model path and class identity. The
calibration log contains no `ERROR` or `Traceback`.

## Static load and inference

A fresh service loaded the exact artifact with
`static_dispatch_policy=locality_fair`. Target and MTP each logged eight
matching static commits with the artifact SHA above, shapes `(77, 264)` and
`(1, 264)`, and `transfer_events=0 rearrangement_events=0`. Rank 0 reported
33/264 local/global physical experts. All eight Model Runner V2 workers and all
eight API servers reached readiness under DeepEP-LL, DeepGEMM, MTP3, E4M3 KV,
and default target/draft graph capture. The log contains no `ERROR` or
`Traceback`.

Deterministic HumanEval/0-31 used temperature 0, seed 0, and
`reasoning_effort=no_think`. All 32 requests ended normally and the EvalScope
1.9.1 functional checker reported **32/32, pass@1=1.0**. The predictions file
has exactly 32 ordered unique task IDs and the result file has 32 passing
reviews.

Runtime metrics across all eight engines recorded 3,420 MTP draft tokens and
3,200 accepted tokens. Repeating one long prefix across the DP engines raised
the prefix-cache counters to 21,504 hits out of 36,587 queried tokens; every
engine recorded hits.

## DeepEP low-latency fake-prefill gate without EPLB

The separate fake-prefill service intentionally omitted `--enable-eplb` and
all record/load paths. It used DP8/TP1/EP8, fixed `deepep_low_latency`,
DeepGEMM, Model Runner V2, chunked prefill with 64 scheduled tokens, and DBO
thresholds of 32 prefill / 8 decode tokens. vLLM 0.28.1 rejects DBO combined
with MTP speculative decoding and CUDA Graphs at configuration time, so this
DBO-focused gate used target-only eager execution, the supported combination.
The startup log explicitly states that DBO is enabled and microbatched steps
run without graphs.

Three request stages completed:

| Stage | Real requests | Prompt tokens | Decode limits | Result |
| --- | ---: | --- | --- | --- |
| one real rank / seven idle-fake ranks | 1 | 2,445 | 24 | 1/1 HTTP 200 |
| uneven prefills | 3 | 246 / 1,046 / 2,046 | 12–40 | 3/3 HTTP 200 |
| staggered uneven batch | 8 | 146–3,246 | 8–64 | 8/8 HTTP 200 |

Short requests completed while longer ranks continued, with elapsed times from
2.55 to 27.65 seconds. The final health check returned HTTP 200; metrics
recorded 17,211 prompt tokens and 287 generation tokens. Eight workers and
eight API servers reached readiness, and the log contains no `ERROR` or
`Traceback`.

## Acceptance statement

The confirmed review defects are closed by tests, and the supported offline
EPLB topology completed calibration, immutable static loading, graph inference,
MTP acceptance, prefix-cache use, and HumanEval-32 on hardware. The fixed
DeepEP-LL DBO/fake-prefill path also completed without offline EPLB. This
supports keeping PR #187 in Ready-for-review state. It does not certify other
topologies, DBO with MTP/graphs (rejected by upstream configuration), full
HumanEval, or performance improvement.
