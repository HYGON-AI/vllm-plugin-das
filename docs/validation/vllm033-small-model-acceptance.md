# v0.33 PR330: user-authorized small-model precision acceptance

The user clarified **“允许7B以下有精度误差”**: precision differences are allowed
for models below 7B. This is a model-size policy, not a 7% or seven-task limit.
It applies to this upgrade's acceptance; it is not a new global default for
unrelated users/releases. Original raw/normalized scores, requests, failed-task
sets and historical HOLD reviews remain preserved.

The earlier Flash/E5M2 model-score HOLD is superseded for the small-model
profiles below. Larger-model accuracy, crashes, garbled output, cache/registry
errors, strict operator numerical/Graph gates and remaining upgrade features
are not waived. No tolerance, production kernel or model behavior is changed.

| Profile | Original normalized HumanEval32 | Current disposition |
| --- | --- | --- |
| Qwen2-0.5B, Flash BF16 / full E5M2 | 16/32 / 13/32 | Accepted small-model precision difference |
| Qwen3-1.7B, Flash BF16 / full E5M2 | 28/32 / 27/32 | Accepted small-model precision difference |
| Qwen3.5-0.8B MTP3, official Triton / Plugin Flash BF16 | 15/32 / 14/32 | Accepted small-model precision difference, including changed task sets |
| Qwen2-0.5B latest BF16 / mixed E5M2 with layers0/2 auto | 16/32 / 14/32 | Diagnostic recipe; not substituted for full E5M2 |

The original raw scores and token/reasoning protocols remain in
[Qwen2 evidence](../../validation/v0.33.0/e5m2-investigation-gfx936-candidate.json),
[TP2 feature evidence](../../validation/v0.33.0/feature-batch-gfx936-candidate.json)
and [hybrid evidence](../../validation/v0.33.0/hybrid-quality-gfx936-candidate.json).
The [policy record](../../validation/v0.33.0/small-model-user-acceptance.json)
also counts stored tensor elements from actual unquantized safetensors headers
to verify that these checkpoints are below 7B; it does not use a MoE's active
parameter count. Tied/buffer/vision/MTP tensors may be included in that count.

## Review scope and current implementation

The reviewed runtime is unchanged from 8bfc11f59 and the installed
official-ownership wheel SHA256
`6a71808eddedf310136bb50f60b08b674bf897297d1df0a37b1dd14e5ca6bbec`.
All14 packaged Python files match source, wheel and both isolated environments.
The current extension advertises RMSNorm and a separate explicit Flash provider;
official INT8 selection/conversion/construction stay official. AITER INT8 model
integration remains unavailable and is not restored by accepting score variance.

The official Core retains models, runner/MTP/Graph, scheduling, KV specs,
allocation and per-layer KV skip. Adapter retains true HCU identity, admission
and process health. Plugin owns the optional public OOT/Flash increments and
its E5M2 physical-format/scales adapter. No native-AR or official-attention
experiment is included in this MR, and no Core runtime patch is added.

The current remote v0.33.0 branch has no reported checks/status contexts; the
existing PR workflows target the legacy v0.28.1-dev branch. Absence of checks
is not a CI pass. Manual verification and exact commands/results are recorded
in the policy candidate and the commit-bound PR review. Overall148-feature
upgrade acceptance and `last_good=false` remain separate from this MR decision.

Fresh verification: **263 portable passed/66 skipped**, **55 installed GPU
checks passed/zero skipped** (Flash cache/scales/numerics, RMSNorm and changed
input Graph ON/OFF), and installed verifier schema4 passed. Fourteen Torch
deprecation warnings and the verifier's HIP-runtime discovery warning are
retained. No operator tolerance was relaxed. XML/log digests and wheel/source
hashes are retained in the policy JSON.

Review decision: no remaining blocking issue found for the currently advertised
capabilities under the user's explicit small-model policy; eligible for merge.
The exact new commit, AI-assisted review and remote merge result will be retained
on [PR330](https://github.com/HYGON-AI/vllm-plugin-das/pull/330). This local review
is not a formal independent human approval and does not certify the full upgrade.
