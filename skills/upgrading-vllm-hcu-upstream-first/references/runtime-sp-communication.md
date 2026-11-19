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

Independent token shards must agree on token rows before a TP reduction. Trace
dense and shared MLP policy separately rather than replicating every projection.
In the inspected legacy GLM4/HY3 path, ordinary dense MLPs gather tokens before
the official TP-sharded gate/up GEMM and reduce-scatter the down-projection
partials back to local token rows. Set down-projection reduce_results=False
before construction; an intervening all-reduce would duplicate the reduction.
Shared experts retain local tokens and select replicated weights before official
allocation. Keep official model MLP forward, activation, checkpoint names,
scale loaders, post-load processing and TP-status updates. Preserve TP
divisibility for dense weights; nondivisible shared dimensions are a separate
case. Scoped prefixes must distinguish ordinary MLPs from shared experts.

Clear local NaN padding before all-gather and global padding before the down
GEMM; mask the local result after reduce-scatter. Exercise device-mask changes,
different rank token shards, weight reloads and changed-input Graph replay.
Compare ordinary TP and SP collective cast/reduction behavior without loosening
existing numerical gates. Native component correctness with bounded allocations
alongside another service does not establish uncontended serving performance.

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

For explicit candidate model admission, inspect the target's loader cache and
worker-extension initialization order before loading a checkpoint. In the
inspected Core, the architecture cache key omits registry ownership, so public
registration after a stock class was cached can still load that stock class.
Refuse stale admission rather than clearing Core caches; verify the resolved
class before loading and the actual class on every live worker afterward.
Registry-only CPU checks do not qualify the real initialization hook or model.
Trace the wrapper and underlying worker together: an `init_worker_extension`
hook called after `init_device` can follow runner construction and a rank-local
architecture lookup. One rank may already cache the stock class while others
begin loading the candidate. Check whether the target provides an earlier
general-plugin startup seam before first resolution; retain the warm-cache
rejection and verify actual per-process ordering before qualifying that route.
Preserve other loader errors observed during peer shutdown separately; this
timing failure alone does not attribute or resolve them.

Early admission and checkpoint loading do not qualify the default compiler.
Run the actual target fullgraph compiler over real projection classes: native
Graph components or a synthetic outer model can miss Python guards traced by
Core's compiled inner loop. In the inspected target, an inlined `os.getpid()`
projection check is rejected by Dynamo before serving becomes healthy.
Keep process/owner enforcement at a verified execution boundary, including
cached FULL replay; do not freeze a PID with constant-result annotations or
skip health checks merely because compilation is active. Use the target's
supported tensor/collective custom-op boundaries where needed, preserving
ownership, masks and numerical assertions. A CPU compiler reproducer isolates
this blocker; a fix still needs the unchanged trained-model/default-Graph,
changed-input replay, HumanEval and normal serving-shutdown gates.

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

Freeze the actual attention input and preserve each GEMM's M/K/N, padding,
strides and selected quantization method. Removing inactive rows can select a
different kernel and invalidate the arithmetic control. Distinguish captured
loaded weights from weights reconstructed from a retained synthetic formula.
Compare full and partial GEMMs with an independent higher-precision result.
A matching sequential BF16 sum proves a compatible rounding explanation, not
the collective kernel's actual summation order; do not hardcode that order into
production to imitate a less accurate result.

A full-model control that replaces reference TP partial GEMMs/reduction with
a gathered full GEMM changes the reference. Keep its failures and subsequent
first-divergence analysis separate from the original TP gate. Fewer failures
under that control do not pass the original gate or prove remaining MLP
reduction arithmetic, trained-model accuracy or lifecycle correctness.

When an early numerical assertion prevents later model/Graph cases from
running, a diagnostic collector may preserve each original assertion and finish
the remaining cases, then exit nonzero if any gate failed. Report candidate
Graph-versus-eager consistency separately from candidate-versus-official TP
agreement; neither replaces the other or trained-checkpoint acceptance.

## Intermediate numerical diagnosis

When a complete synthetic model differs from ordinary TP, preserve the original
weights, backend, precision, Graph settings and tolerances. Do not infer the
root cause from its final maximum error. The diagnostic model tool accepts
`--trace-intermediates` to snapshot eager module outputs at embedding, decoder
norms, QKV, Q/K norm, RoPE, attention, output projection, MLP and final norm.
Capture is default-off and is forbidden inside a GPU Graph. Inactive hooks do
not query capture state or copy tensors. Trace snapshots synchronize the device
and must never be included in serving/performance measurements.

With a fresh output directory and the same admitted dependency/device setup as
the original model probe:

```bash
python tools/probe_runtime_sp_models.py /path/to/new-output --trace-intermediates
python tools/runtime_sp_trace.py /path/to/new-output glm4 --output /path/to/glm4-comparison.json
python tools/runtime_sp_trace.py /path/to/new-output hy3 --output /path/to/hy3-comparison.json
```

The additional initial traced forwards must exactly match each arm's original
warm result. Later samples are eager intermediates before changed-input Graph
replay, not traces of Graph execution. Original full-model assertions remain
unchanged and failures still produce exit1. Snapshot failures also fail the
diagnostic, after the ranks retain their result/cleanup path.

The offline comparator joins candidate token shards in rank order, retains
head shards per rank, excludes only declared padding and uses the original
`atol=rtol=.016`. Missing/duplicate ranks, incomplete samples, mismatched event
orders/layouts/dtypes and empty outputs are refused. Report the first unequal
intermediate separately from the first outside the original tolerance. Neither
is root-cause proof: later controlled experiments must distinguish layout,
weight loading, provider arithmetic and rounding/reduction order. Ordinary TP's
partial GEMMs plus reduction and an SP replicated full GEMM can differ in BF16;
that observation alone neither waives a model gate nor justifies a policy change.

Use `sample` and `event_index` to identify each invocation, retaining `event`
as the canonical module name. Core can cache one RoPE module across layers;
`named_modules()` then supplies one name for several calls. Do not merge those
rows by name or assign a later call to the layer encoded in that shared name.

`tests/vllm033/test_runtime_sp_trace.py` covers CPU capture and offline controls.
Its success does not qualify a device trace, SP collective, trained checkpoint,
HumanEval, normal shutdown or gfx836 support. Record the exact runtime/source
and archive identities when the original failed cohort can actually be traced.

## Projection compiler and cached replay boundaries

Compile the real QKV, Row and MergedColumn OOT classes with the target Core;
projection doubles can hide unsupported process/group checks. Keep the original
GEMM, loaders and arithmetic visible to Core. Narrow data-producing mask/layout
and collective custom ops can contain Python ownership checks while their fake
rules describe exact shapes and non-aliasing outputs. Keep the guarded result
on the GEMM dependency path. Pass device masks as explicit tensor arguments:
reading them only inside an opaque op hides their addresses from the compiler
and can retain capture-time storage after the outer batch context changes.
Verify the captured FX graph with no Python SP batch and with replacement mask
storage, in addition to in-place mask-value changes. An unused output-free guard
can be eliminated. Preserve rejection before GEMM even when the supplied token
shape is invalid.
Without a batch mask, a functional preparation op may need a copy; record that
cost rather than returning an alias that violates its declared schema.

Python custom-op checks run during capture, not cached CUDA Graph replay. Bind
model/group and concrete projection ownership into required capability health
and verify Adapter's actual Worker boundary before PP wait, offload and FULL
replay. A public platform hook alone is bypassed by that path. Refuse missing
Adapter capabilities rather than freezing PID, skipping health, disabling
compilation or repairing owners during forward. Keep model health registrations
process-local, weakly owned and serialized; an expired owner must refuse replay.

Separate eager-backend Dynamo, AOT functionalization, default CPU Inductor,
installed-wheel behavior and actual device/default-serving compilation. Exercise
changed mask values at stable addresses, NaN padding, owner drift and real fork.
A CPU Graph-manager test with a replay double qualifies the host boundary only;
native Graph, trained-model HumanEval, numerics, performance and lifecycle gates
remain separate. See the [fullgraph validation record](../../../docs/validation/vllm033-runtime-sp-fullgraph.md).
