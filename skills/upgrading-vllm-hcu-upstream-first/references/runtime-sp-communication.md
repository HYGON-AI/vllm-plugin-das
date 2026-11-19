# Runtime-SP communicator construction

Trace `make_parallel_config`, `ParallelConfig.use_all2all` and the actual EP
communicator together. In the inspected 0.33 target, TP2/DP1/PCP1 with explicit
MoE SP and EP requires all-to-all kernels, but the global parallel config does
not admit a manager. Changing a runner flag after factory construction cannot
repair this mismatch.

Use the official platform communicator-class interface. Adapter owns a single
selector registry and retains the official default; an explicit Plugin
communication provider owns the custom-SP sidecar policy and subclasses the
official communicator. Pass EP admission into its constructor, preserving all
arguments including `tcp_store_group` and stateless rank metadata. Reuse Core's
manager/backend selection and destruction; do not construct a replacement in
dispatch or during capture. Required capability health must bind the exact
current-process report before the selector returns a provider class.

Transport token/head split buffers through public Torch collectives over the
borrowed TP device group. Refuse a missing multi-rank group instead of falling
back to the world group. Check independent layouts/inverse transforms, changed
inputs on Graph replay, stable addresses and natural cleanup. Byte transport
does not qualify FP8 scaling or quantization.

Core's preallocated reduce-scatter may return a new view of the supplied output.
Check storage address, shape, dtype and contents rather than Python object
identity; retain the full numerical and changed-input Graph assertions.

The `runtime-sp-communication` capability is a communication subfeature under
development, not complete GLM4/HY3 custom-SP. It uses the existing
`additional_config['hcu']['enable_custom_sp']` boolean and an explicit Adapter
capability request; no new environment policy is needed. Keep model embeddings,
residual/position layouts, replicated GQA, weight/scale reload, shared experts,
factory routing/configuration, target/draft batching, MTP/PP/DP/PCP, model
HumanEval32, performance and the original shutdown gate independently open.

## Official attention projections

For GLM4/HY3, retain the official attention module, RMS/RoPE, cache writer and
reader. Concrete `QKVParallelLinear` and `RowParallelLinear` public OOT
registrations can adapt token/head layouts without copying model algorithms.
Use a scoped construction policy and exact attention projection prefixes;
ordinary construction must retain the official TP behavior. Resolve foreign
owners before publishing either registration, and latch partial publication
through Adapter's process-local lifecycle.

Select replicated weights with `disable_tp=True` before official allocation
and loader creation. Use official split/fused weight and scale loaders, then
the normal post-load/`update_param_tp_status` path for refitting. Replacing
initialized sharded weights or changing only a late runner flag is insufficient.

Replicated GQA uses contiguous rank groups: TP4/KV2 maps ranks0/1 to KV0 and
ranks2/3 to KV1. Every source token shard must send the correct head to every
destination replica. Selecting `rank % kv_heads` and then gathering mixes heads
from different sources. Verify KV1, replicated KV2 and partitioned KV heads
against an independent unsharded reference, plus the inverse output exchange,
distinct V dimensions, bias contracts and reloads.

CPU allocation/loader tests with a kernel-init double do not qualify FP8 GEMM.
Native attention component tests do not qualify full model SP, shared experts,
early MoE factory policy, batching or HumanEval32. Preserve precision, backend,
Graph/prefix policy and numerical assertions while diagnosing capture failures.
