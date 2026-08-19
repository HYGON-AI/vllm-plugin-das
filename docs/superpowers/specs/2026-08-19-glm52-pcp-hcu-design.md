# GLM-5.2 HCU Prefill Context Parallel Design

## Goal

Add plugin-owned Prefill Context Parallelism (PCP) support for
`/models/GLM-5___1-Channel-FP8-w8a8` while keeping the vLLM dependency fixed at
v0.25.1. The first accepted deployment is eight HCU devices using Model Runner
V2, `TP=4`, `PCP=2`, expert parallelism, and eager execution.

The implementation is a focused HCU backport of vLLM pull request #46570
(`b6ff8a2f50`, "Add MRV2 virtual-batch PCP for MLA"). It must live entirely in
`vllm-plugin-das`; `/models/zb/vllm_025/vllm` must remain unchanged.

## Scope

The first release supports only this configuration family:

- Model Runner V2 selected with `VLLM_USE_V2_MODEL_RUNNER=1`.
- An MLA or sparse-MLA causal language model. The acceptance model uses
  `GlmMoeDsaForCausalLM` with `index_topk=2048`.
- `pipeline_parallel_size=1`.
- `prefill_context_parallel_size>1` and
  `decode_context_parallel_size=1`.
- No speculative configuration, including MTP.
- Eager execution with CUDA/HIP graph mode disabled.
- No LoRA, multimodal inputs, P/D disaggregation, KV offload, or data
  parallelism. Expert parallelism is supported and required by the acceptance
  topology.
- HCU lightly-CP is disabled; it is a separate context-parallel algorithm and
  cannot be combined with PCP in this release.

The public interface remains vLLM's existing
`--prefill-context-parallel-size`; the plugin does not add a second PCP flag.
`PCP=1` must retain the current runner, attention, KV-cache, and MoE behavior.

## Chosen Approach

Use a selective plugin-side backport rather than copying the upstream runner or
upgrading the vLLM checkout. The plugin will add an HCU PCP manager and extend
the existing MRV2 subclass and exact-name runtime patch framework at the narrow
points required by PCP.

This approach preserves the v0.25.1 deployment contract and the existing
PP+MTP fixes. It also keeps upstream-derived PCP behavior recognizable: copied
algorithms retain their Apache license headers and name pull request #46570 in
their module documentation.

The alternatives were rejected for these reasons:

- Copying all 39 files changed by #46570 would fork large parts of vLLM inside
  the plugin and make future compatibility checks ambiguous.
- Updating the vLLM baseline would pull hundreds of unrelated upstream commits
  into an HCU compatibility change and violate the plugin-only delivery
  boundary.

## Runtime Architecture

### Configuration and fail-closed validation

The platform-side `VllmConfig` adapter will permit MRV2 PCP only for MLA models
and validate the restricted scope before worker startup. It will produce one
specific error for each unsupported combination: V1 runner, PP, DCP, DP,
speculative decoding/MTP, non-eager graph mode, LoRA, multimodal input, KV
offload, or non-MLA attention.

Validation must run before the plugin removes v0.25.1's generic
"Model Runner V2 does not yet support prefill context parallelism" gate. The
gate is removed only when all HCU PCP invariants pass; otherwise the original
configuration remains rejected.

### Virtual-batch partitioning

`HcuGPUModelRunnerV2` will own an `HcuPCPManager` adapted from upstream
#46570. The scheduler continues to create a global batch. Immediately after
the upstream runner prepares its `InputBatch`, the manager converts every
prefill request into rank-local virtual rows using DualChunkSwap partitioning.

For PCP size `P`, a request is split into `2P` contiguous chunks. PCP rank `r`
receives chunks `r` and `2P-1-r`. Pairing an early and late causal chunk keeps
the attention work more balanced than assigning one contiguous range per
rank. Decode rows are replicated across PCP ranks so request state stays
synchronized.

The manager keeps the global `InputBatch` private until sampling. Rank-local
buffers are separately allocated for request mappings, positions, sequence
lengths, block tables, slot mappings, logits indices, and padding state. This
avoids overwriting the global batch while allowing up to two local rows for
each global request.

### KV-cache ownership

PCP partitions prefill computation; it does not shard decode KV ownership.
With the first-release requirement `DCP=1`, every PCP rank owns a complete KV
cache for each request.

The plugin will patch the v0.25.1 scheduler/cache calculations that currently
multiply effective block size by `PCP`. PCP must be excluded from:

- scheduler and hash block-size resolution;
- per-request block counts and maximum KV memory calculations;
- full-attention cache-manager alignment;
- unitary KV-cache coordinator block sizing; and
- slot-mapping context-parallel rank calculations.

These patches are conditional on HCU PCP and retain the original behavior for
`PCP=1`. DCP behavior is not redesigned in this release.

### MLA and GLM DSA attention

During prefill, each PCP rank computes only its local Q rows but requires the
complete latent KV sequence. Before cache insertion, the attention adapter
all-gathers rank-local `kv_c_normed` and RoPE K tensors across the PCP group,
aligns them with the gathered slot mappings, and writes the complete cache on
every rank.

The acceptance model uses sparse DSA/MLA. Its indexer follows the same rule:
the indexer K values and their slot mappings are gathered before cache
insertion, while Q/index selection remains rank-local. Prefill metadata must
use explicit request-phase information so short prefill continuations are not
misclassified as decode rows.

Because this release is eager-only, the HCU MLA adapter will use the direct
Python attention path for PCP. This avoids changing v0.25.1's already
registered opaque custom-op graph and makes the cache-gather side effect
explicit. Existing opaque/custom-op behavior remains unchanged for `PCP=1`.

The HCU backend selector will advertise PCP only for the MLA, sparse MLA, and
DeepSeek-v3.2-style indexer backends exercised by this design. Other attention
backends continue to fail capability validation.

### MoE and expert parallelism

After attention, hidden states and router logits contain only the local PCP
token shard. The plugin-owned MoE runner must perform PCP-aware dispatch:

- When the selected EP all-to-all implementation consumes PCP-local token
  shards directly, it must not perform an extra PCP all-gather.
- For the non-all-to-all fallback, hidden states and router logits are
  all-gathered before expert execution and reduce-scattered after combine.

The acceptance topology uses one EP group spanning all eight TP/PCP ranks.
Shared-expert output must follow the same token ordering as the routed expert
output before the reduce-scatter result is returned to attention order.

### Global restoration and sampling

Before logits, prompt logprobs, sampling, or postprocessing, the manager
all-gathers local hidden states and restores the original global request and
token order. `HcuGPUModelRunnerV2.sample_tokens` will replace the saved execute
state with the restored hidden states and global `InputBatch` before delegating
to upstream sampling.

This placement ensures logits indices, request IDs, prompt logprobs, and
request-state updates all observe the scheduler's original batch rather than
PCP virtual rows.

## Plugin Structure

PCP code will be split by ownership rather than collected in one broad patch:

- `vllm_hcu/v1/pcp_manager.py`: virtual-batch partition and restore logic.
- `vllm_hcu/v1/hcu_model_runner_v2.py`: MRV2 lifecycle integration.
- `vllm_hcu/model_executor/layers/attention/pcp.py`: latent KV and indexer K
  gather helpers.
- Platform config adapters: capability gate and restricted-scope validation.
- Worker callbacks: block-table, cache-sizing, attention metadata, MLA, and
  indexer compatibility patches.
- Plugin-owned sparse indexer and MoE runner: HCU kernel-facing PCP behavior.

Every runtime patch must validate the exact v0.25.1 target signature before
installation and register a stable patch ID in the existing lifecycle
registry. A missing or changed target is a startup error, not a silent skip.

## Error Handling

- Unsupported PCP combinations fail during configuration with the incompatible
  feature named in the message.
- Missing PCP process groups, mismatched tensor counts, invalid virtual-batch
  slices, and restore-index inconsistencies use assertions with rank and shape
  context because they indicate internal contract violations.
- Backend capability failures name the selected backend and the required PCP
  capability.
- Collective or HCU kernel failures retain their original traceback; the
  plugin does not fall back to PCP=1 or Model Runner V1.
- `PCP=1` never constructs `HcuPCPManager` and never executes a PCP collective.

## Testing Strategy

Implementation follows test-driven development. Each production behavior is
preceded by a focused failing test.

### CPU and contract tests

1. Configuration permits MRV2+MLA+PCP+eager and rejects each unsupported
   combination independently.
2. DualChunkSwap partitions multiple requests, preserves every token exactly
   once per prefill, handles uneven lengths, and restores global order.
3. Decode-only rows remain replicated and mixed prefill/decode batches restore
   correct logits indices and request mappings.
4. KV-cache sizing depends on DCP but not PCP when HCU PCP is active.
5. Block-table and slot-mapping helpers accept PCP-local output buffers without
   mutating the global batch buffers.
6. MLA latent KV and indexer K gather helpers preserve decode writes and order
   gathered prefill writes by PCP rank.
7. Backend selection accepts only the intended HCU MLA/indexer backends.
8. MoE dispatch executes exactly one PCP gather/combine path for both all-to-all
   and fallback configurations.
9. MRV2 integration partitions after input preparation and restores before
   sampling and prompt-logprob computation.
10. Patch lifecycle tests verify exact targets, idempotence, and fail-closed
    behavior.

The pre-change baseline is the focused configuration, lifecycle, MLA, and
sparse-indexer suite: 80 tests passing.

### Eight-HCU acceptance

Run both deployments with the same model, tokenizer, KV-cache dtype, maximum
length, deterministic sampling parameters, and HCU MoE backend:

- Baseline: `TP=8`, `PCP=1`, EP enabled, MRV2, eager.
- Candidate: `TP=4`, `PCP=2`, EP enabled, MRV2, eager.

Acceptance requires:

1. The candidate server reaches healthy state on all eight ranks.
2. Fixed smoke prompts with `temperature=0` complete without rank divergence;
   generated token IDs match the baseline for the deterministic comparison
   set.
3. One 32K and one 64K input complete prefill and at least one decode token.
4. EvalScope HumanEval runs 32 samples with thinking disabled and
   `temperature=0`; its passed count is not below the PCP=1 baseline.
5. Logs contain no Model Runner V1 fallback, PCP capability assertion,
   collective shape mismatch, KV-cache write error, or worker crash.
6. Peak memory, TTFT, inter-token latency, and throughput are recorded for both
   deployments. The first release has no mandatory performance-improvement
   threshold.

## Rollback and Compatibility

`--prefill-context-parallel-size 1` is the rollback path and must select the
current implementation without PCP patches affecting runtime behavior. The
existing `VLLM_USE_V2_MODEL_RUNNER=0` V1 fallback remains available for
non-PCP deployments.

The change must not alter the existing GLM PP+MTP+graph command because PCP is
not part of that configuration. PP, MTP, and graph support for PCP require
separate designs and are explicitly outside this work.

## Out of Scope

- PCP combined with PP, MTP, DCP, DP, CUDA/HIP graphs, LoRA, multimodal input,
  P/D disaggregation, KV offload, or HCU lightly-CP.
- PCP support for GQA, MHA, hybrid attention, sliding-window attention, Mamba,
  or non-causal models.
- Multi-node PCP.
- A new ring-attention algorithm or a performance rewrite of HCU collectives.
- Changing, committing, or opening a merge request against the vLLM checkout.
