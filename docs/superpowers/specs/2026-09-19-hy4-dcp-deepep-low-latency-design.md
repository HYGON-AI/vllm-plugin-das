# HY V4 DCP with DeepEP Low-Latency Design

## Goal

Make HY V4 serve and decode correctly with tensor parallel size 8, decode
context parallel size 2, expert parallelism, DeepEP low-latency all-to-all,
DeepGEMM MoE, and the existing FP8 KV cache dequantization path. The change is
stacked on the current LightOp sparse-mask TopK branch and does not add or use
the rejected direct Aiter paged-MQA optimization.

## Current Failures

Two independent startup failures block this configuration.

First, `HYV4FlashMLASparseImpl` inherits the upstream sparse FlashMLA
implementation. It does not advertise decode LSE support and always returns a
null LSE. DCP requires every attention rank to return its local output and the
matching softmax LSE so vLLM can combine partial attention results with a
numerically stable cross-rank reduction. The existing HY V4 FP8-to-BF16 path
also discards the LSE returned by `flash_mla_sparse_fwd`.

Second, `worker.framework_opt.communicator.deep_ep_runtime` is an import-time
callback for vLLM's all-to-all module. The callback may legitimately remain
armed until distributed initialization or warmup loads that module, but the
worker currently requires every terminal callback immediately after model
loading. This rejects a valid lazy callback before the lifecycle reaches the
point that consumes it.

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
3. For `fp8_ds_mla`, gather and dequantize only those rank-local selected
   slots into the compact BF16 cache already used by HY V4. For BF16 cache,
   use the local cache directly.
4. Invoke the sink-aware HY V4 BF16 sparse FlashMLA wrapper with the local
   TopK lengths and preserve both its output and LSE.
5. Set output to zero and LSE to negative infinity for ranks whose local TopK
   row is empty. This gives the common vLLM DCP reduction the identity value
   it expects.

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
- DCP localizes sparse indices, forwards valid lengths, preserves attention
  sinks, and returns the real kernel LSE.
- Empty local TopK rows produce zero output and negative-infinity LSE.
- FP8 DCP uses the compact gather/dequantization path rather than the native
  FP8 FlashMLA geometry.
- Model-load validation defers only the enabled DeepEP runtime callback.
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

## Delivery

The implementation and tests will be committed on
`feat/hy4-lightop-mask-topk-adapt` and pushed to its existing remote MR. Only
the DCP/LSE and DeepEP lifecycle fixes are added; no direct Aiter paged-MQA
operator or cache-layout conversion is included.
