# Final Flash E4M3 model validation boundary

PR #455's final reviewed payload passes its component/Graph and CI gates, but
its new Qwen3-8B model comparison has an unresolved accuracy loss. Preserve
that distinction when selecting a KV dtype or assessing later changes.

## Frozen environment and protocol

The 2026-11-18 run uses `/models/Qwen3-8B`, TP1 on gfx938/BW1100, Core
`c440c089fe9621c5b03e1618cf86a13e95de8bf1`, Adapter
`1c9448e1b09514a8d1424ad0d50d70cb88a6d078`, and reviewed Plugin
`c4b8754c936c81824548aa8bec529b18e0ad4235`. The delivered Plugin merge is
`c13610b944a92bd75970f6fa5f3f6ddf5a9bf1fc`; its execution payload matches
the reviewed source and installed wheel. Wheel SHA256:
`e047a216159a454cfe65252bac437d358b414b85a5d0acaea76c73e643a5e62c`.

Both arms keep explicit FLASH_ATTN, BF16 activations, block64, context4096,
max sequences16, batch tokens1024, prefix caching and official default
FULL_AND_PIECEWISE Graph. Each reserves 1GiB for KV, so the formats have
different logical cache capacities. Independent cache directories are used.
Concurrent shared-node workloads change available memory between launches;
these runs do not establish comparative performance.

HumanEval uses original tasks0–31, temperature0, seed0, output limit2048,
thinking-off, the qwen3 reasoning parser and concurrency4. The exact same
request bytes are used in all three services; SHA256:
`a7bcbc5c3a1ae1394c0c28ca967ca094df5aac55ff1d880694e4c8f98c3d4f09`.
This is the explicitly recorded replacement protocol, not a claim to recover
missing older handoff request bytes. Dataset revision is
`6d43fb980f9fee3c892a914eda09951f772ad10d`.

## Results

| Arm | Raw /32 | Format-normalized /32 | Normalized failures | Completion tokens |
| --- | --- | --- | --- | --- |
| BF16 KV | 20 | 30 | 18,26 | 4199 |
| E4M3 KV | 19 | 30 | 17,26 | 4348 |
| Unchanged BF16 fresh-service repeat | 20 | 30 | 18,26 | 4204 |

There are no length-limited completions. E4M3 loses previously passing
HumanEval/17 in both raw and normalized scoring, while gaining normalized
HumanEval/18. Equal normalized totals do not establish precision parity.
Raw scoring retains Markdown/format failures. Generated programs are scored
only in mount/network/PID namespaces and a read-only chroot as uid65534.

The initial pair has six different completion texts: tasks1,9,17,18,25,31.
For task17, BF16 generates a dictionary lookup over whitespace-separated
notes. E4M3 generates a scanner that recognizes a whole note `o` only when
followed by a space, omitting a final bare `o`. This is a semantic failure.
The unchanged BF16 full32 repeat preserves both pass sets; only task25's
completion text changes. It reproduces none of the differing E4M3 sequences.
This narrows the investigation without identifying a faulty kernel or
waiving the original accuracy loss.

## Execution and lifecycle evidence

All three services pass arithmetic13/13, prefix3/3 with1280 hit tokens, and
concurrency1/4/8 probes. Each actual worker trace records40 `hipGraphLaunch`
calls; GPU kernel counts are528,1188,516 respectively. Frontend-only traces
are excluded from GPU execution evidence. The pair observes Core native KV
writers, Flash BF16 compute, and36 actual HcuFlashAttentionImpl layers. E4M3
storage is uint8 interpreted as OCP E4M3; K/V scales are1.0 and Q remains
unquantized. This is not an all-FP8 compute performance claim.

Four complete package RECORDs and17 checkpoint assets totaling
16,397,449,445 bytes pass fresh checks before/after execution. Each checkpoint
seal reads every asset twice. All owned process identities exit, ports are
released and no controller-forced cleanup occurs. Frontend exits are0, with
observed shutdown durations9.78s,10.18s,9.63s. Nanobind native-owner leak
warnings remain in all services: normal lifecycle is still unqualified.

## Evidence and remaining work

Local evidence root on the validation host:
`/models/zb/vllm033-resume-20261114/evidence/goal-resume-20261118-r116/`.
The original requests/responses, isolated results, logs, worker metadata,
traces and pre/post seals remain under `qwen3-final-merged-pair/` and
`qwen3-final-bf16-repeat/`. Independent summaries are
`qwen3-pair-independent-review.json` and
`baseline-repeat-independent-review.json`; the validated source/wheel binding
is `qwen3-final-merged-pair/source-wheel-installed-binding.json`.

Precision attribution, native-owner finalization and applicable performance
remain open. Other checkpoints, architectures, topologies and the full
upgrade are not qualified by this comparison; `last_good=false`. DeepSeek
V4.1 model tests remain deferred. Further HCU adaptation should prefer Adapter
lifecycle/platform hooks and Plugin public extension points, retaining Core
changes only where the required public interface is demonstrably missing.
