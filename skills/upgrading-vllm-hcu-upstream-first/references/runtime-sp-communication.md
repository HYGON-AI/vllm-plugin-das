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


## Dense/shared MLP and early MoE construction

Independent token shards cannot use ordinary TP-sharded dense/shared MLP weights:
an all-reduce would sum different tokens. Under the same explicit model
construction scope, select replicated merged gate/up and down projections before
Core allocates weights and loaders. Keep official model MLP forward, activation,
checkpoint names, scale loaders, post-load processing and TP-status updates.
Use distinct attention-only/model scopes and precise dense/shared prefixes;
unrelated projections retain their ordinary TP behavior. Check differing token
shards, dimensions not divisible by TP, weight reloads and ownership drift.

At the inspected 0.33 target, GLM4/HY3 call the plain FusedMoEFactory without its
existing is_sequence_parallel argument. runner_cls is too late: parallel config,
router and routed weights already exist. A temporary scoped bridge can bind only
these model-local factory aliases and pass that argument before the unchanged
Core factory executes. Leave the shared Core factory intact, preflight all alias
owners and ABI, reject later owner changes and latch a partial publication failure.
Retire this bridge when the official models expose an equivalent early policy.
Validate actual official model constructors reaching make_parallel_config before
router/weight creation; a fake returned runner does not establish that ordering.

These projection/factory components still require full model composition across
embeddings/inputs_embeds, residuals, positions, padding and PP boundaries. Keep
checkpoint loading, native quantization, target/draft batching, complete topology,
HumanEval32, performance and original model shutdown qualification separate.


## Full model token and padding boundaries

Keep the official inner model loop and outer loading/logits interfaces. An
outer CausalLM subclass can compute full vocabulary-TP embeddings before
sharding token rows; sharding IDs first makes embedding all-reduces mix
unrelated tokens. Preserve inputs_embeds, both hidden/residual PP fields, and
stock runner/send-buffer token shapes by gathering and trimming at stage exits.
Attention exchanges restore global token rows, so positions stay global even
while MLP/residual rows are local. Test nondivisible token counts and runner
padding separately from the extra SP alignment rows.

Use the runner's device Boolean padding tensor, including in-place changes on
Graph replay. Do not capture a Python num_actual_tokens as a constant mask.
FlashAttention can leave padded output rows unwritten: masked_fill them before
the inverse exchange and projection/activation quantization. Multiplication by
zero does not clear NaNs. Also mask local projection inputs so padding cannot
pollute a tensor-wide scale.

Core's MoE routing may happen before internal dispatch or after naive
all-gather. Use the local mask in the former layout and the exact gathered EP
mask in the latter; with DP, use the same group and per-rank sizes as dispatch.
The public runner_cls factory argument can supply this narrow context adapter,
while is_sequence_parallel still has to precede config/router/weight creation.
Do not overwrite a foreign runner policy. Restore context on exceptions and
preserve the official custom-op layer-name cursor across context copies.

A Graph capture allocates outputs but does not establish their values. Replay
the unchanged initial inputs and compare them with the reference before using
the captured output for change detection. An all-padding rank can correctly
remain zero; compare against the expected change rather than uninitialized
capture storage. Preserve the original failed probe and unchanged numerical
gates. CPU Dynamo with layer doubles, raw native Graphs, complete synthetic
models, and trained-checkpoint/default-runner/HumanEval acceptance are distinct
validation scopes; do not transfer one scope to another.

## Independent native MoE reference boundaries

Derive reference casts from the actually selected activation and expert kernels.
For the inspected Core activation template, SiLU returns the input dtype before
multiplication by the up half. The selected Triton expert GEMM2 multiplies its
FP32 accumulator by router weights before the final output cast. A reference
that skips the first cast and rounds GEMM2 before weighting can fail despite
identical loaded parameters and routing. Do not respond by loosening tolerances
or changing the backend.

Retain the first failed run, then save frozen inputs, loaded expert maps/weights,
router logits/IDs/weights and actual/reference tensors before the assertion.
Check shared and routed branches separately. Compare independently implemented
CPU arithmetic with documented cast boundaries, keeping the original numerical
gates, and verify fresh eager and changed-input Graph execution. A corrected
reference does not retroactively pass the original run or qualify a checkpoint,
quantized provider, different kernel selection or normal model shutdown.


## Full GEMM versus ordinary TP rounding

A replicated SP projection and an ordinary TP projection can load identical
weights and receive identical head inputs yet round differently: the full BF16
GEMM casts once, while sharded GEMMs cast partial outputs before the collective
reduction. Trace the first differing module, reconstruct the full weight/input,
and compare both outputs with independent FP32 arithmetic before attributing a
later model mismatch to attention or routing. Record intervening residual/norm
amplification and top-k changes separately. A closer FP32 result does not waive
an existing model tolerance or establish checkpoint accuracy. Keep that gate
failed; do not change weights, precision or thresholds just to obtain a pass.

When an early numerical assertion prevents later model/Graph cases from
running, a diagnostic collector may preserve each original assertion and finish
the remaining cases, then exit nonzero if any gate failed. Report candidate
Graph-versus-eager consistency separately from candidate-versus-official TP
agreement; neither replaces the other or trained-checkpoint acceptance.
