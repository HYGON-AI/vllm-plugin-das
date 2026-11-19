# Runtime-SP model boundary candidates on vLLM 0.33

These explicit GLM4/HY3 CausalLM subclasses compose the previously delivered
projection and early MoE factory policies around the unchanged official inner
model loops. They are **not published through ModelRegistry or automatically
enabled**. Full custom-SP model acceptance remains incomplete; keep #479 Draft.

**Subsequent correction:** [dense TP follow-up](vllm033-runtime-sp-dense-tp.md)
restores ordinary dense MLP gather/TP/reduce-scatter while shared experts remain
replicated. The original complete-model log also contains three `mp.spawn`
SIGTERM attempts after rank numerical exceptions. Its controller recorded no
signals, but that is not signal-free diagnostic shutdown; original evidence
and failed numerical gates remain preserved.

## Implementation

Vocabulary-TP embeddings receive the same complete input IDs on every rank
before hidden rows are token-sharded. Supplied embeddings follow the same row
layout. Global positions survive attention's token/head exchange; residuals and
MLP inputs remain local. Pipeline stages shard and restore both hidden states
and residuals, then gather/trim outputs to the stock runner's token shape.

Device Boolean masks preserve changing runner padding on Graph replay and add
SP divisibility padding. Unwritten attention rows and local projection inputs
are cleared with `masked_fill`, including NaNs, before exchange and GEMM. The
injected public `runner_cls` supplies the mask corresponding to Core's local or
naively dispatched MoE layout. Official routing, experts, shared experts,
combine, loading and logits algorithms remain inherited. Copied forward
contexts restore the official MoE layer-name cursor even on exceptions.

Core, Adapter and native FlashAttention are unchanged by this follow-up.
Adapter communicator-selector dependency `004cce0a83055b2f67acdc1060258773c3275787`
remains unpublished; its presence in an isolated installation is not delivery.

## Completed component validation

The candidate uses Core `c440c089fe9621c5b03e1618cf86a13e95de8bf1`, the above
Adapter and a private installed Plugin wheel. Source and wheel/installed Python
payloads match; all 18 owner RECORDs are verified before/after native runs.

- Source CPU: **202 passed**. Independent installed CPU: **202 passed**.
  These run the actual official outer/inner loops with explicit CPU layer and
  collective doubles, covering both families, all four ranks, IDs/embeddings,
  PP first/middle/last boundaries, padding, exception/cursor restoration and
  Dynamo fullgraph with the eager backend. They do not qualify real PP or
  native default-runner compilation.
- Clean CI stdlib contracts: **98 passed**. Source/package: **8 passed**.
- Native gfx938 TP4/SP4/EP4 attention padding: GLM4/HY3 × BF16/FP16 × KV2/KV8
  × four ranks = **32 distinct rank/cases**. Eager and initialized/changed-input
  raw Graph replay pass unchanged `atol=rtol=0.016`; maximum absolute error is
  `0.00390625`. Actual token counts change 5→7 within fixed buffers; NaN padding
  is cleared. Controller exit0, tracker0, no signals/survivors, RECORDs unchanged.

Original failed CPU fixtures and the first padding probe remain in evidence.
The first native padding probe read capture storage before its initial replay;
its rank failure stranded peer cleanup. Exact owned PID/starttick identities
were confirmed before external SIGTERM, recorded separately from controller
signals. The repaired probe changes no production code, backend or threshold.

The native-tested wheel SHA256 is
`78f757f4ddd52c85db51a854a623abd3ed59d3e7562b26220d88205c7d571ebe`.
The delivery wheel `aa784a52acb999434bd84df6ad96335598ac4aead99e4a925d536f068dfdb027`
updates only the README description in METADATA and its RECORD hash. All 113
Python payloads, other wheel members and metadata headers are byte-identical;
the delivery wheel was not reinstalled or presented as a new native run.

## Complete synthetic models: numerical gate remains failed

The real official two-layer GLM4 model (dense plus routed/shared MoE), real
HF/ModelConfig, official weight loaders and native kernels were constructed at
BF16 TP4/SP4/EP4. Deterministic synthetic weights do not represent a trained
checkpoint. The original eager comparison with ordinary official TP fails:
maximum output difference `0.0390625` exceeds the unchanged `0.016` gate.

The first attempt also preserved a low-level harness initialization error:
`ParallelConfig.disable_custom_all_reduce=True` had not been applied through
stock `set_custom_all_reduce`. The second run applies that existing startup
policy and checks that no custom-AR owner exists, then reaches the numerical
failure above. No production backend is changed to repair the harness.

Frozen per-layer tensors identify the first difference at layer0 attention
output projection, not QKV/cache layout: embeddings, normalized inputs and QKV
are identical. The full projection weight equals concatenated official TP
shards and head inputs are identical. Independent FP32 CPU arithmetic verifies
that the candidate full GEMM equals FP32 rounded to BF16. Its error against
FP32 is `0.0038805008`, versus ordinary TP's `0.0079648495`; TP versus SP differs
by `0.0078125` there. Residual/norm propagation later exceeds the original
model tolerance. A closer local FP32 result **does not pass the model gate** or
establish a trained-model accuracy conclusion.

The final diagnostic completes both real two-layer model families on all four
ranks. It retains all original comparisons and exits1 because numerical gates
fail. There are eight distinct family/rank cases, not eight checkpoints.

| Family | Eager vs TP maximum error | Graph vs TP, active5 / active7 | Graph vs SP eager |
| --- | --- | --- | --- |
| GLM4 | 0.0390625 | 0.0335693359 / 0.03125 | 0 / 0 |
| HY3 | 0.03125 | 0.048828125 / 0.03125 | 0 / 0 |

Each rank fails the eager and active5 official-TP comparisons for both families;
the active7 comparisons pass the unchanged absolute-plus-relative assertion.
All 16 family/rank/replay comparisons against SP eager are exactly equal.
Padding stays finite/zero and addresses remain fixed with active tokens5→7.
Controller exit1 is expected from the retained numerical failures; tracker0,
no controller cleanup signals, no owned survivors and all18 RECORDs unchanged.
`mp.spawn` did log three SIGTERM attempts; see the correction above. This is not model
qualification or a normal-serving shutdown result.

The preceding diagnostic completed GLM4 before HY3 construction failed because
its synthetic HF config omitted Core's required `route_norm` field. Supplying
that field fixes only the probe configuration; original logs are retained.
No production, backend, precision, synthetic weights or threshold changed.

## Reproduction and acceptance boundaries

`tools/probe_runtime_sp_padding.py` reproduces native attention components with
the candidate installed on four visible devices; it is not a model test.
`tools/probe_runtime_sp_models.py` reproduces the full synthetic-model
collector. Use four visible HCU devices, the paired installed Adapter/Plugin,
`HCU_ADAPTER_HOST=vllm`, `VLLM_PLUGINS=hcu_adapter` and required capabilities
`runtime-sp-communication,hcu-flash-attention`. Pass a new existing output
directory as its only argument. A nonzero exit preserves failed numerical
gates even when all model/Graph cases complete. Run artifacts and terminal
receipts are retained in the delivery workspace under
`evidence/runtime-sp-model-boundaries-r189-20261119/`.

Default-runner compilation, actual pretrained checkpoint loading/HumanEval32,
quantized weights/scales and reloads, target/draft/MTP, real PP/DP/PCP,
DeepEP/stateless combinations, serving performance and the original 15-second
normal model shutdown gate remain open. Raw Graph component cleanup does not
qualify the model service lifecycle. No DeepSeek V4.1 checkpoint was run.
AI-assisted implementation and self-review are not an approving review.
