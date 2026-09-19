# HY V4 DCP with DeepEP Low-Latency Design

## Goal

Make HY V4 serve and decode correctly with tensor parallel size 8, decode
context parallel size 2, expert parallelism, DeepEP low-latency all-to-all,
DeepGEMM MoE, and the existing FP8 KV cache dequantization path. The change is
stacked on the current LightOp sparse-mask TopK branch and does not add or use
the rejected direct Aiter paged-MQA optimization.

## Current Failures

Three independent correctness and startup failures block this configuration.

First, `HYV4FlashMLASparseImpl` inherits the upstream sparse FlashMLA
implementation. It does not advertise decode LSE support and always returns a
null LSE. DCP requires every attention rank to return its local output and the
matching softmax LSE so vLLM can combine partial attention results with a
numerically stable cross-rank reduction. The existing HY V4 FP8-to-BF16 path
also discards the LSE returned by `flash_mla_sparse_fwd`. FlashMLA applies an
attention sink to its output denominator but explicitly leaves the sink out of
its returned LSE, so the DCP path must reconstruct the effective denominator
LSE rather than forwarding that raw value unchanged.

Second, `worker.framework_opt.communicator.deep_ep_runtime` is an import-time
callback for vLLM's all-to-all module. The callback may legitimately remain
armed until distributed initialization or warmup loads that module, but the
worker currently requires every terminal callback immediately after model
loading. This rejects a valid lazy callback before the lifecycle reaches the
point that consumes it. The existing HCU communicator gate also enables
DeepEP for PCP+EP with DP1, but not for DCP+EP with DP1. In the requested
TP8/DCP2/EP8 topology that leaves `use_all2all` false, so the CUDA communicator
never imports the all-to-all module or constructs the requested DeepEP manager.

Third, the HCU sparse-indexer entry point does not receive the DCP rank,
world size, or cache interleave. Each rank therefore selects only its local
TopK candidates and leaves rank-local token IDs in the shared buffer. The
attention backend expects global logical token IDs and localizes them for its
rank, so treating these local IDs as global IDs reads the wrong KV rows and
produces invalid decode output.

## DCP Sparse-Indexer Design

Each DCP rank will keep its existing local logits and local TopK calculation.
It will pack each candidate's score and global logical token ID, all-gather
those compact candidate lists across the DCP group, and select the final
global TopK. This is exact because a candidate in the global TopK must also be
in its owning rank's local TopK. The exchange is limited to
`dcp_world_size * topk_tokens` candidates per query rather than the complete
logit row.

The existing CUDA path will continue using its CuTeDSL stable selector. HCU
will use device-side PyTorch gather, all-gather, and TopK operations because
CuTeDSL is unavailable on the platform. Prefill score lookup accounts for
each row's packed sequence offset. Token IDs are converted with the configured
cache interleave so the attention backend can apply its existing DCP
localization.

The fused LightOp mask-TopK decode route returns indices without their scores,
so DCP decode will use the existing logits-producing HCU route before the
global merge. DCP size one retains the LightOp route and existing custom-op
schema. This change does not introduce a direct Aiter paged-MQA operator.

## DCP Attention Design

`HYV4FlashMLASparseImpl` will advertise
`can_return_lse_for_decode = True` and provide a DCP-specific `forward_mqa`
path. The existing inherited path remains unchanged when DCP size is one.

For DCP decode, the implementation will:

1. Concatenate split MLA query tensors with the existing query buffer when
   needed.
2. Convert the shared logical sparse TopK indices into rank-local physical
   cache slots with `triton_filter_and_convert_dcp_index`, requesting a valid
   count for every query row.
3. Have the HY V4 MLA layer invoke its selected backend's post-weight-load
   hook after preparing MLA projections, then use that backend hook to
   all-gather each rank's local attention-sink shard once across the DCP group.
   The gathered sink order
   matches the rank-ordered query-head all-gather performed by vLLM, avoiding
   a collective in every decode layer invocation. Before the DCP kernel call,
   subtract `log(dcp_world_size)` from each gathered sink logit. Every DCP
   rank includes the virtual sink in its local softmax denominator, so this
   adjustment makes the later cross-rank LSE reduction count the sink exactly
   once instead of once per rank.
4. For `fp8_ds_mla`, gather and dequantize only those rank-local selected
   slots into the compact BF16 cache already used by HY V4. For BF16 cache,
   use the local cache directly.
5. Invoke the sink-aware HY V4 BF16 sparse FlashMLA wrapper with the local
   TopK lengths and preserve its output and raw LSE. When a sink is active,
   compute `logaddexp(raw_lse, normalized_sink)` so the LSE describes the same
   denominator already applied to the kernel output.
6. Set output to zero for ranks whose local TopK row is empty. Without an
   attention sink, set LSE to negative infinity to provide the common DCP
   reduction identity. With an attention sink, preserve the kernel LSE: the
   normalized virtual sink remains a real denominator contribution even when
   that rank owns no selected KV slot.

The BF16 wrapper will expose the kernel's output and LSE internally. Existing
non-DCP callers will continue returning the same public output shape and will
ignore LSE when it is not requested. Attention sinks remain active in both
the local attention output and LSE computation.

Fake LSE values and bypassing vLLM's DCP compatibility assertion are excluded
because either option would produce numerically invalid cross-rank attention.

## DeepEP Patch Lifecycle Design

Terminal patch validation will distinguish model-load completion from runtime
warmup completion. Model-load validation will continue to require all patches
whose target modules must be loaded while constructing the model, while
allowing only the DeepEP runtime all-to-all callback to remain armed.

The context-parallel communicator gate will treat either PCP greater than one
or DCP greater than one as a reason to enable the explicit DeepEP backend for
an EP communicator when upstream left it disabled. This makes CUDA
communicator construction import and patch the real all-to-all manager before
it is instantiated.

After `compile_or_warm_up_model` completes, the worker will drain callbacks
whose imports have finished and run full terminal validation. At this point a
configured DeepEP backend must have loaded and patched the all-to-all runtime;
an armed, skipped, or failed required callback remains a startup error. This
retains fail-closed behavior without forcing a distributed runtime module to
load earlier than vLLM's normal lifecycle.

Feature-off callbacks may remain armed as before. No validation exception is
added for an enabled callback after warmup.

## Error Handling

The HY V4 DCP path will fail with a clear runtime error if FlashMLA does not
return LSE. Shape and dtype checks already used by the attention sink and FP8
dequantization paths remain authoritative. DeepEP incompatibilities continue
to latch in the patch registry and fail worker startup during full terminal
validation.

## Test Strategy

Unit tests will cover:

- HY V4 advertises decode LSE support.
- DCP gathers the loaded local sink shards once in rank order and uses the
  gathered values for the corresponding gathered query heads, adjusted by
  `-log(dcp_world_size)` during DCP attention.
- DCP localizes sparse indices, forwards valid lengths, preserves attention
  sinks, and returns an effective LSE that includes the kernel's documented
  sink denominator adjustment.
- Empty local TopK rows produce zero output; their LSE is negative infinity
  without a sink and the normalized sink LSE when a sink is present.
- FP8 DCP uses the compact gather/dequantization path rather than the native
  FP8 FlashMLA geometry.
- HCU DCP exchanges local indexer scores and token IDs and chooses the correct
  global TopK for decode; the rank/interleave mapping is covered separately.
- Model-load validation defers only the enabled DeepEP runtime callback.
- DCP+EP with DP1 enables the configured DeepEP all-to-all manager just as the
  existing PCP+EP path does.
- Post-warmup validation drains ready callbacks and rejects a DeepEP callback
  that is still not applied.

Hardware validation will use the model
`/models/Hy4-preview-Channel-FP8-w8a8` with TP8, DCP2, EP enabled,
`deepep_low_latency`, DeepGEMM, FP8 KV cache, block size 64, and the existing
HY V4 FP8 KV dequantization setting. Validation requires successful server
startup, successful completion requests through decode, no failed or pending
enabled terminal patches after warmup, and clean worker shutdown.

The final accuracy check will run the same eight HumanEval samples used by the
current TP8 baseline. The acceptance target is 8/8 correct and Pass@1 100%.
Focused runtime and model tests, Python compilation, and diff whitespace
checks must also pass before committing and pushing the branch.

## Validation Result

The exact TP8/DCP2/EP8 configuration started with
`DeepEPLLAll2AllManager`, completed a deterministic multi-token decode, and
shut down cleanly. EvalScope executed `HumanEval/0` through `HumanEval/7` with
8/8 successful requests, Accuracy 100%, and Pass@1 100%. The observed mean
latency was 38.118 seconds and average output throughput was 3.5 tokens/s for
this eight-sample correctness run; these figures are observational and are not
an A/B performance comparison.

## Delivery

The implementation and tests will be committed on
`feat/hy4-lightop-mask-topk-adapt` and pushed to its existing remote MR. Only
the DCP/LSE and DeepEP lifecycle fixes are added; no direct Aiter paged-MQA
operator or cache-layout conversion is included.
