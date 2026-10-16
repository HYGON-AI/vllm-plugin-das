# Gemma4 FA fast path on HCU

This opt-in implementation applies only to target-model Gemma4 full-attention
layers with head dimension 512. It preserves Triton's metadata builder, KV-cache
layout, cache writes, and backend name. It does not globally override
`TRITON_ATTN`, change sliding-attention layers, or change the MTP draft model.

```bash
# Default: normal vLLM backend selection (Triton in the current HCU Gemma4 setup).
export VLLM_HCU_GEMMA4_USE_FA=0

# Gemma4 full D512: FA for supported Q>16 or Q=5 calls, Triton otherwise.
export VLLM_HCU_GEMMA4_USE_FA=1
```

Read the switch at plugin model registration; restart the service to change it.
`HCU_FLASH_ATTN_Q5` is obsolete and does not affect routing. A conflicting
explicit backend setting fails at initialization instead of being overridden.

The native path currently requires BF16 and an unquantized NHD cache with block
size 32 or 64. Gemma4's hybrid page-size unification may grow the full-attention
block size from a requested 32 to 64. Query and KV head counts come from the model's local TP configuration;
there is no batch=1 or fixed TP2 head-count check. Unsupported attention masks,
output quantization, and other semantics fall back to the original forward.
The vLLM 0.25.1 integration still requires numerical and CUDA Graph validation
on HCU; CPU dispatch tests do not establish model accuracy or a speedup.

For route diagnostics, set `HCU_DEBUG_ATTN_Q_LEN=1` before startup and look for
`[GEMMA4_FA_ROUTE]`. Capture logs report the captured path; full graph replay
does not re-execute the Python logger. Disable diagnostics for performance runs.

The existing `register_model()` entry point registers a post-import callback
only when enabled. That callback binds a model-local Attention subclass after importing
`vllm.model_executor.models.gemma4`. The subclass supplies `attn_backend` through
the existing Attention constructor. It does not patch the shared Attention
class or write upstream source files.

All Gemma4-specific implementation and registration live in `models/gemma4.py`.
Its private helper gathers paged KV into temporary
contiguous K/V and calls the HCU native `pagedkv_to_contiguouskv` and
`varlen_fwd` entry points. It does not modify or globally replace
`flash_attn_interface.py`. It requires the HCU Flash Attention extension ABI
used by the plugin's existing DSpark wrapper, rather than an arbitrary upstream
Flash Attention installation.

Eager capacity is `max(262144, sum(seq_lens))` tokens. Capture capacity is
`max(262144, batch_size * max_seqlen_k)` tokens: replay cannot grow a captured
buffer, and capture's dummy KV lengths are not a safe bound. This intentionally
keeps the current policy without adding a budget switch. It can OOM for large
capture batches; moving allocation into the plugin does not reduce its size.
For BF16 D512 and two KV heads per rank, both buffers together consume
`capacity * 4096` bytes (about 1.90 GiB at 25 requests and max length 19968).
Limit capture sizes to the intended workload and available memory. Capture
sizes count query tokens, not requests: Q5 with 24 requests needs size 120.
D512 Q1 remains on Triton.

When migrating a locally modified legacy server, remove its hard-coded FA
dispatch and source-patch adapter and restore the FA package before enabling
this plugin implementation. Flag=0 preserves normal backend selection; it is
not the historical Q>16-FA baseline. Prior HCU vLLM 0.21.0 GPU results do not
validate this vLLM 0.25.1 integration.
