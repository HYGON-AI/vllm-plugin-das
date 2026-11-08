# v0.33 SlimQuant format candidate

This draft adds an explicitly requested checkpoint format for the local
Qwen3.8-Flash-Next-Channel-INT4-w4a8 checkpoint. The official compressed-tensors
method does not decode its mixed packed-INT4 expert / INT8 dense / per-row INT8
PLE representation. The Plugin owns that format translation; Core owns model
construction, official CT arithmetic, backend selection and kernels. Adapter
retains the platform/general entry points and the generic extension host.

## Dependency and activation

This is **not ready for deployment**. It requires the unmerged, generic Core
`Qwen4ExpPLEPerRowEmbeddingMethod` API. The validated Core candidate is based on
OpenDAS `5afd48292984d91cf35057e231e4986bd792a284`, with patch SHA256
`aa5ac1d05932c27d8c7157d04755772bc107113b837de001abbb3974913f9763`.
Its wheel SHA256 is
`a6118ab0342b1100b6d04fd251ff31cb8802fce0509d0cc452a5a6947f6773ee`.
This document identifies a local dependency, not a published Core release.

Request `HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=hcu-slimquant-w4a8` explicitly.
Activation refuses before registry publication if the public Core API is absent.
The only new registry name is `slimquant_w4a8`; existing `compressed-tensors`,
`fp8`, `gptq`, and INT8 oracle owners stay unchanged. Remove the capability and
use the appropriate official format to disable this candidate.

Packed expert bytes decode high nibble first to signed values in [-8, 7].
Expert scales convert from the checkpoint's high-nibble units by multiplying
by 16 before the official INT8 loader. Dense and PLE scales retain their units.
The expanded expert weights consume more memory than packed W4A8 kernels.

Ignore patterns use Core's existing glob/regex matcher. Expert gate/up and down
projections share one method: all ignored selects BF16; partial ignores refuse.
PLE shards also share one storage method. Their actual count comes from the
current public model context's `hf_text_config.split_ngram_parts`, without a
hardcoded count or cached model context. All ignored selects BF16; partial
ignores or partial quantized declarations refuse. Missing positive shard-count
metadata refuses unless the runtime parent is already ignored.

## Scoped validation on gfx938

The publication runtime wheel SHA256 is
`a21cdc87c24a33f31f4a1bee217cc77faf332693a1a865100f867073cc6dba58`.
All 23 recorded payload files match the isolated installation; all 17 runtime
Python files match this source tree. Core's 3661 recorded payload files also
match its dependency wheel. The original Adapter wheel is unchanged.

- 305 portable tests passed; six HCU tests were deselected, not counted as passes.
- Eight additional CPU activation tests passed, including concurrent publication,
  missing Core API, foreign ownership, post-write failure and actual CPU fork.
- Eleven focused tests passed with the HCU opt-in: five portable contracts and
  six actual-device PLE/expert loader, independent arithmetic and changed-input
  Graph tests. HIP PCI identity was `0000:49:00.0`, architecture `gfx938`,
  Torch 2.13.0, DTK 26.10. Only the standalone PLE fixture's public shard-count
  metadata is doubled; backend, loader, tensors and kernels remain real.
- The original selector produced 15 failures / four passes on the 19 new ignore
  contracts. The corrected selector passes all 19 and independent review.

Local receipts: `plugin-slimquant-publication-cpu-r2.{xml,log}`,
`slimquant-publication-activation-r1.{xml,log}`,
`slimquant-publication-components-r5.{xml,log}`, and corresponding origin JSONs.
A first component launcher omitted the existing `HCU_SLIMQUANT_LIVE=1` opt-in:
its five passes / six skips remain archived as `components-r4` and do not count
as device validation. The corrected `components-r5` passed with zero skips.

Historical runtime r3 scored raw 0/32 and normalized 32/32 on original
HumanEval tasks 0–31 (all responses used code fences), with prefix 14/14 and
concurrent 8/8. That run used TP2, default Graph, device PLE, official Triton
attention and disabled custom all-reduce. Cleanup diagnostics remained; engine
and worker exit codes were not fully observed. This is a functional result,
not clean-lifecycle or performance acceptance. The prior wrong-scale runtime's
0/32 result is retained.

The corrected r4 selector completed a separate full model run: raw 0/32 (all
code-fenced responses), normalized 32/32, all 32 `stop`; prefix 14/14 with 20800
cache hits and concurrent 8/8. Both actual workers reported 128 PLE shards,
48 SlimQuant expert methods, 144 official CT dense methods, 423 BF16 linear
methods and one public per-row PLE method each. All 32 request bodies and wire
bodies match historical r3; all 32 archived r3 normalizations are identical
under the old v2 and current resource-safe v3 policies. The new run uses the
original budget and safe namespace/chroot scorer, on GPU0–1 rather than
historical GPU4–5. No same-device causal performance comparison is claimed.

API exit was 0, the owned group emptied, port 18714 was released and VRAM
returned from 4 MiB to 4 MiB on both devices. The log still contains four
vendor cleanup unpack exceptions and a two-worker SIGTERM escalation. It has
no two-worker SIGKILL or EngineDeadError entry. Full worker exit codes remain
unobserved; clean teardown is not accepted. Local terminal receipt:
`qwen38-flash-slimquant-tp2-public-r4/quality-review-r1.json`.

## Open gates

Core API publication/review, prepared-image CI and clean model teardown remain
open. Native packed W4A8,
Marlin/LightOp/AITER model integration, heterogeneous shard methods, EP/DP,
multi-node, other checkpoints and performance remain unvalidated. This bridge
does not close those migration features.
