# Hy4 Prefill Context Parallel Design

**Date:** 2026-08-31

**Status:** Proposed

## Context

The HCU vLLM 0.25.1 plugin already provides a Model Runner V2 prefill
context-parallel (PCP) implementation for GLM-5.2 sparse MLA and for selected
FlashAttention GQA models. It owns the PCP process group, partitions scheduler
batches, gathers MLA and sparse-indexer cache inputs, restores global token
order before sampling, and dispatches MoE tokens across PCP ranks.

Hy4 uses the same sparse-MLA and lightning-indexer building blocks, but it is
currently rejected by the PCP configuration gate because its architecture is
`HYV4ForCausalLM`, not `GlmMoeDsaForCausalLM`. Its sink-capable
`HYV4FlashMLASparseImpl` also inherits the upstream default
`supports_pcp = False`, so removing only the architecture check would fail
during model construction.

This design extends the existing PCP implementation to Hy4 without creating a
second model runner or weakening the fail-closed checks for other models.

## Goals

- Run `/models/Hy4-preview-Channel-FP8-w8a8-v2` on eight HCUs with
  `TP=4`, `PCP=2`, expert parallelism, Model Runner V2, and eager execution.
- Support the existing Hy4 sparse MLA, lightning indexer, learnable attention
  sink, IHC residual path, shared experts, and Channel-FP8 MoE weights.
- Validate both Triton and AITER MoE backends, with Triton used first to isolate
  PCP correctness from backend-specific behavior.
- Support target-only decoding first, followed by built-in MTP with one and two
  speculative tokens.
- Preserve PCP-disabled behavior and all existing GLM-5.2 and GQA PCP
  contracts.
- Fail early for unsupported PCP combinations with model-specific messages.

## Non-goals

- Full CUDA/HIP graph support for Hy4 PCP.
- MTP with three or more speculative tokens.
- Pipeline, data, or decode context parallelism combined with PCP.
- LoRA, KV offload, P/D disaggregation, lightly-CP, or HCU multi-layer MTP with
  PCP.
- Multimodal PCP.
- New PCP collective algorithms, ring attention, or a Hy4-specific model
  runner.
- Performance tuning of AITER, FlashMLA, or MoE kernels.
- Changing the semantics of PCP-disabled Hy4 serving.

## Supported configuration

The first supported Hy4 topology is:

| Setting | Required value |
| --- | --- |
| Model architecture | `HYV4ForCausalLM` |
| Model runner | V2 |
| Tensor parallel size | 4 on the eight-HCU validation host |
| Prefill context parallel size | 2 |
| Expert parallelism | enabled |
| Pipeline/data/decode context parallel size | 1 |
| Execution mode | eager |
| Attention | sparse MLA through the HCU FlashMLA sparse backend |
| MoE backend | Triton or AITER |
| Speculation | off, or built-in MTP with 1 or 2 tokens |

PCP increases the distributed world size. On the target eight-HCU host,
`TP=4, PCP=2` consumes eight ranks; `TP=8, PCP=2` would require sixteen ranks
and is not a valid launch.

## Design

### 1. Model-aware PCP configuration contract

The configuration adapter will classify MLA PCP architectures explicitly:

- GLM-5.2: `GlmMoeDsaForCausalLM`
- Hy4: `HYV4ForCausalLM`

Both models will use the existing sparse-MLA PCP contract: Model Runner V2,
MLA enabled, expert parallelism enabled, eager execution, and PP/DP/DCP sizes
equal to one. Existing restrictions for LoRA, KV offload, P/D disaggregation,
lightly-CP, multimodal models, and multi-layer MTP remain unchanged.

The adapter will continue to reject every other MLA architecture. Error text
will identify the selected architecture and list the supported HCU MLA PCP
architectures instead of referring only to GLM-5.2.

For Hy4 and GLM-5.2, speculative decoding must use built-in `mtp`, and
`num_speculative_tokens` must be one or two. The generic FlashAttention GQA
PCP contract remains unchanged and continues to reject speculation.

### 2. Hy4 sparse attention capability

`HYV4FlashMLASparseImpl` will declare `supports_pcp = True`. This is a narrow
capability declaration, not a new attention implementation:

- `HcuPCPManager` already partitions query tokens and constructs PCP metadata.
- The HCU MLA wrapper already gathers latent KV and RoPE K tensors with their
  rank-ordered cache slots before writing the replicated cache.
- The HCU sparse indexer already gathers indexer K inputs and validates the PCP
  metadata world size.
- Each rank continues to compute attention only for its local queries against
  the complete gathered cache.
- Hy4's sink-specific kernel wrappers continue to pass `attn_sink`; PCP does
  not change the TP-local head layout or sink tensor.

No inheritance change is planned. Hy4 will remain a subclass of the upstream
sink-compatible sparse implementation and opt in only to the PCP capability it
actually exercises. DCP remains rejected, so the HCU DCP-specific
`forward_mqa` override is not required.

### 3. Sparse indexer and shared top-k ownership

The existing indexer PCP path remains the source of truth. Tests will verify
that a Hy4 full indexer layer:

- receives PCP metadata and rank-local query tensors;
- gathers only cache-write K tensors and their matching slot mappings;
- leaves the local top-k output in the correct local token order; and
- keeps shared-indexer consumers attached to the producer's local top-k
  buffer.

If integration exposes a Hy4-only ownership mismatch, the fix will be made at
the smallest existing boundary (`pcp.py`, the indexer adapter, or Hy4's
attention wrapper). The PCP manager will not gain a parallel Hy4 code path
unless a failing test proves the generic contract cannot represent the model.

### 4. MoE and IHC data flow

Hy4 will continue to construct the standard vLLM `FusedMoE` object. Its
existing PCP behavior gathers rank-local hidden states and router logits before
expert execution and reduce-scatters the routed result back to local token
order. Shared expert output stays rank-local and is added only after the routed
result returns to that same order.

Expert parallelism shards the 256 experts across the combined TP/PCP ranks.
The current Channel-FP8 quantization and AITER selection adapters must preserve
the resulting expert map and must not preselect or prequantize token-local
routing inputs before the PCP dispatch. Triton is validated first; AITER is
then required to produce a normal completion and comparable deterministic
outputs.

IHC pre/post/head operations are token-local. They consume the PCP-local token
rows during model execution and require no collective or layout change.

### 5. Sampling and MTP

For target-only decoding, `HcuPCPManager` restores global hidden-state order
before sampling and then publishes the sampled tokens back through the normal
runner path.

For MTP, the existing replicated-MTP scope restores the global batch before
the draft model executes. Within that scope the effective PCP width is one, so
the draft model must not gather MLA or indexer cache inputs a second time.
Hy4's native `HYV4MTPModel` remains the only draft architecture; no separate
draft checkpoint is introduced.

Validation proceeds in this order:

1. target-only decoding;
2. built-in MTP with one speculative token;
3. built-in MTP with two speculative tokens.

MTP3 remains rejected in the configuration layer for the initial support
scope.

### 6. Graph behavior

Hy4 PCP is eager-only in this design. `--enforce-eager` is mandatory and the
configuration adapter will reject a graph-enabled Hy4 PCP launch. Existing
platform logic that downgrades generic PCP full graphs to piecewise graphs is
not treated as Hy4 graph support.

Graph support can be evaluated separately after eager PCP correctness and
accuracy are stable.

## Error handling

Configuration errors must occur before model weights are loaded. The adapter
will reject:

- Hy4 PCP without Model Runner V2, MLA, EP, or eager execution;
- Hy4 PCP combined with PP, DP, DCP, LoRA, KV offload, P/D disaggregation,
  lightly-CP, multimodal input, or multi-layer MTP;
- speculative methods other than built-in MTP;
- MTP token counts outside one or two; and
- MLA architectures outside the explicit GLM-5.2 and Hy4 list.

Runtime assertions remain for internal ownership invariants such as one KV
cache group, matching PCP metadata/group sizes, and rank-ordered slot counts.
They are not substitutes for user-facing configuration validation.

## Testing strategy

All production changes follow red-green-refactor.

### CPU-safe contract tests

- A valid Hy4 `TP=4, PCP=2, EP, eager` configuration is accepted.
- Each unsupported combination above is rejected with a targeted message.
- Existing GLM-5.2 and GQA PCP acceptance/rejection tests stay green.
- `HYV4FlashMLASparseImpl` advertises PCP while preserving sink capability and
  the existing kernel argument forwarding.
- PCP cache-write gathers preserve Hy4 latent KV, indexer K, slot mapping, and
  local top-k ownership.
- PCP MoE dispatch preserves local token order for routed and shared expert
  outputs.
- The replicated MTP scope does not re-enter PCP cache gathers.

### Regression suites

Run the focused Hy4, PCP, MLA/indexer, MoE, MTP, quantization, dispatcher, and
platform configuration suites against the pinned vLLM 0.25.1 source tree.
Run compile checks and `git diff --check` before every commit.

### Eight-HCU integration

Use the Channel-FP8 model and the following staged matrix:

| Stage | TP | PCP | EP | Graph | MoE | MTP |
| --- | ---: | ---: | --- | --- | --- | ---: |
| Baseline | 8 | 1 | off | existing validated mode | AITER | off |
| PCP target | 4 | 2 | on | eager | Triton | off |
| PCP AITER | 4 | 2 | on | eager | AITER | off |
| PCP MTP1 | 4 | 2 | on | eager | AITER | 1 |
| PCP MTP2 | 4 | 2 | on | eager | AITER | 2 |

Each PCP stage must load all weights, initialize all eight ranks, return HTTP
200 from `/v1/chat/completions`, and shut down without orphan worker
processes. A long-prefill smoke request will be added after the 4K functional
request succeeds so PCP partitioning is exercised with a meaningful context.

HumanEval uses the first 16 tasks with batch 16, greedy decoding, seed 42, and
`reasoning_effort=no_think`. The target-only PCP result must not regress below
the existing 15/16 TP8 sample. MTP1 and MTP2 scores and pass/fail flips are
reported separately rather than hidden behind an aggregate score.

## Documentation and delivery

The validation document will record:

- exact TP4+PCP2 server commands for Triton, AITER, MTP1, and MTP2;
- curl and long-prefill smoke commands;
- HumanEval-16 commands and scores;
- graph, topology, and unsupported-feature constraints; and
- observed correctness and performance differences from the TP8 PCP-disabled
  baseline.

After tests and integration pass, an independent code review must report no
unresolved Critical or Important findings. The implementation, commands, and
results will then be pushed to the existing Hy4 MR #34, with the test matrix
posted as an MR comment.

## Success criteria

The adaptation is complete when all of the following hold:

1. Hy4 PCP is accepted only for the documented fail-closed configuration.
2. TP4+PCP2 target-only Triton and AITER services start and return valid chat
   completions on all eight HCUs.
3. Hy4 learnable sinks and sparse-indexer cache/top-k ownership remain active
   under PCP.
4. Built-in MTP1 and MTP2 services start and return valid completions without
   double-applying PCP cache gathers.
5. HumanEval-16 target-only accuracy is at least 15/16, and MTP score changes
   are reported explicitly.
6. Focused and regression test suites pass, code review has no unresolved
   Critical/Important findings, and no vLLM worker processes remain after
   validation.
7. Commands, results, and constraints are present in MR #34.
