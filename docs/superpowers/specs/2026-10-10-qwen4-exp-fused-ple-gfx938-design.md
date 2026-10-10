# Qwen4Exp fused PLE backport for gfx938

## Goal

Backport the AMD fused Qwen4Exp PLE implementation merged in upstream vLLM
PR #60021 onto the plugin's vLLM 0.28.1 integration seam. The change must
preserve the plugin's existing Qwen3.8 INT8/SlimQuant PLE storage, ETP-across-DP
sharding, UVA CPU offload, side-stream prefetch, MTP, and FULL_AND_PIECEWISE
CUDA graph behavior.

The optimization is accepted only when focused unit tests demonstrate semantic
parity and gfx938 measurements show a useful reduction in PLE work without an
accuracy regression. It will be delivered as one PR stacked on PR #199 rather
than added to PR #199 itself.

## Upstream basis

Use upstream vLLM main commit `90ba2f34b5a29d85b8301672e5735676f78fa00c`
as the comparison point and PR #60021 merge commit
`e2414ec1dbb177b54ac1bbce60e94769ebacbb28` as the behavioral source.

The upstream change contributes four related optimizations:

1. Fused Triton n-gram ID generation without padded
   `[num_requests, max_length]` workspaces.
2. Fused normalization, gate, and value preparation.
3. One fused dilated PLE convolution for decode, speculative decode, prefill,
   and mixed batches, including residual addition.
4. Packing `key_proj` and `value_proj` into one `kv_proj` linear layer.

## Scope

### Included

- Add plugin-owned common PLE Triton operators derived from upstream #60021.
- Route the HCU Qwen4Exp replacement PLE module through the fused operators.
- Preserve the existing HCU INT8/SlimQuant embedding methods and ETP group.
- Adapt the narrow AMD model integration points required by `kv_proj`, packed
  checkpoint loading, and fused-convolution residual ownership.
- Keep the current non-fused path as a fallback controlled by the HCU custom-op
  master switch.
- Add unit coverage for numerical parity, graph-facing buffer lifetime,
  checkpoint mapping, ETP-across-DP behavior, and fallback selection.
- Validate on gfx938 with Qwen3.8, MTP3, and FULL_AND_PIECEWISE.
- Record commands, results, and limitations in the model command document and
  the upgrading-vllm-hcu skill after validation.

### Excluded

- Upgrading the installed vLLM package beyond 0.28.1.
- Backporting DeepSeek V4.1 `dp_shared_memory` or transparent huge-page support.
- Adding a new n-gram quantization format.
- Changing MoE routing, AITER, AGRS, or EP collective implementations.
- Claiming CPU-offload support without a successful pinned-memory runtime test.
- Merging upstream PLE metadata-builder PR #58114 in the same PR. That change
  has workload-dependent performance and will be evaluated separately.

## Runtime selection and fallback

The fused PLE path is an HCU optimization and therefore obeys
`VLLM_HCU_USE_CUSTOM_OPS`.

- `VLLM_HCU_USE_CUSTOM_OPS=1`: the fused implementation is eligible when the
  Qwen4Exp HCU replacement module is active.
- `VLLM_HCU_USE_CUSTOM_OPS=0`: retain the existing supported fallback and do
  not select the fused PLE operators.
- `VLLM_HCU_PLE_PREFETCH_STREAM=1`: independently enables the existing UVA
  side-stream lookup only when CPU offload and a supported embedding method are
  active.

The implementation must not require users to add a new serving flag for the
fused kernels. Logs must identify fused versus fallback execution clearly
enough to audit a runtime test.

## Components and data flow

### Fused operators

Create a plugin-owned common PLE operator module instead of patching the
installed vLLM package. Port the plain Triton kernels and their custom-op
wrappers from upstream, retaining platform feature checks such as PDL gating.

The optimized forward data flow is:

1. Packed request tokens and n-gram context enter fused `ple_ngram_ids`.
2. Existing HCU embedding storage gathers IDs across the configured ETP group,
   performs device or UVA lookup, reduces owned rows, and selects local tokens.
3. Packed `kv_proj` produces key and value together.
4. Fused `ple_gate` normalizes and gates the embedding value.
5. Fused `ple_conv` updates PLE state, adds the PLE residual, and adds the
   decoder residual exactly once.

### Existing HCU storage and prefetch

The current storage layer remains authoritative for:

- compressed-tensors INT8 PLE weights;
- SlimQuant INT8 or unquantized PLE weights;
- pinned host allocation and UVA device views;
- persistent prefetch ID/output workspaces;
- ETP-across-DP ID gather and row reduction.

The fused ID kernel writes into persistent storage when the side-stream path
requires the IDs to outlive a graph break. No tensor consumed asynchronously
may be backed only by a CUDA graph-pool temporary.

### Model integration

Use a narrow, signature-checked patch rather than replacing the whole AMD model
module. It will:

- map checkpoint `ple.key_proj` and `ple.value_proj` weights to `ple.kv_proj`;
- add `kv_proj` to packed-module metadata;
- switch decoder residual ownership only for the fused PLE class;
- fail closed if the vLLM 0.28.1 target signatures or class structure differ
  from the audited source.

The MTP draft model does not own a PLE layer, so only compatible mapper metadata
may be shared with it; its forward path must remain unchanged.

## Offload and parallel topology

The following settings solve different problems and must remain independent:

| Setting | Effect on PLE | Memory consequence |
| --- | --- | --- |
| `cpu_offload=false` | PLE shards stay on accelerator memory | No host pinning; accelerator memory holds the shards |
| `cpu_offload=true` | PLE shards live in pinned host memory and are read through UVA | Every process pins its local shard; enough host RAM and locked-memory capacity are required |
| `embedding_across_dp=false` | ETP size equals TP | Every DP replica owns a TP-sharded full table copy |
| `embedding_across_dp=true` | ETP size equals TP multiplied by DP | One logical table is split across TP/DP ranks; per-rank table memory falls, but aggregate table bytes do not |
| `--enable-expert-parallel` | Changes MoE expert placement/communication | Does not itself shard or offload PLE |
| `VLLM_HCU_PLE_PREFETCH_STREAM=1` | Overlaps supported UVA lookup with earlier layer compute | Does not reduce table size or pinned bytes |

For the validated TP1/DP4/EP4 topology, `embedding_across_dp=true` makes ETP4.
Each rank owns roughly one quarter of the PLE table. With CPU offload, all four
quarters are still pinned, so the node must be able to pin approximately one
complete table in aggregate. Increasing DP reduces per-process pinning but does
not reduce that aggregate requirement.

Upstream `dp_shared_memory` is not part of this backport. It targets replicated
CPU-offloaded DP tables, still relies on host registration, and is not a remedy
for the aggregate size of an already ETP-sharded Qwen4Exp table.

## Failure handling

- Unsupported Triton compilation or operator registration fails during focused
  startup validation, before publishing the PR as supported.
- Weight-mapping or residual-ownership mismatches fail closed with an explicit
  compatibility error.
- CPU offload without a live UVA alias fails fast.
- CPU offload that exceeds pinned-memory capacity is reported as an environment
  limitation; it must not be described as validated.
- Fused-path test or performance regressions leave the fallback intact and the
  optimization out of the integration commit.

## Test strategy

### Unit and compatibility tests

Use test-first development. The first tests must fail against the current
implementation because the fused operators and packed `kv_proj` integration do
not yet exist.

- Fused n-gram IDs match the Python reference across EOS boundaries, empty
  requests, variable request lengths, and dynamic request counts.
- Fused gate matches the existing composed normalization/gate behavior.
- Fused convolution matches decode, MTP speculative decode, prefill, mixed
  batches, both supported state layouts, and residual semantics.
- Packed `kv_proj` loads separate checkpoint key/value tensors into the correct
  shards.
- Persistent prefetch IDs survive graph breaks and continue to cover the full
  ETP-across-DP token workspace.
- `VLLM_HCU_USE_CUSTOM_OPS=0` selects the stable fallback.
- Existing Qwen4Exp PLE INT8, prefetch, PP, QSA, and dispatcher tests remain
  green, followed by the complete plugin test suite.

### gfx938 runtime validation

Use the existing Qwen3.8 SlimQuant n-gram INT8 checkpoint. Start with the
smallest viable TP and use DP4/EP4/ETP4 when four cards are free:

- MTP3;
- FULL_AND_PIECEWISE;
- `allgather_reducescatter` and AITER MoE;
- `embedding_across_dp=true`;
- accelerator-resident PLE first;
- CPU offload plus prefetch only when pinned-memory capacity is available.

Run HumanEval 16 at temperature zero and require no regression from the current
16/16 result. Compare fused on/off with identical random-serving workloads and
capture PLE kernel count, output throughput, TPOT, TTFT, MTP acceptance, startup
success, and memory use. Integrate only if the fused path is numerically correct
and shows a repeatable benefit or a material workspace reduction on gfx938.

## Delivery

The stacked PR will contain the backport, tests, validation evidence, command
documentation, and skill updates. PR #199 remains unchanged and is its base.
