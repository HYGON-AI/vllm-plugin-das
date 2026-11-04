# GLM5Next KDA CP-head (experimental, parameterized PCP width)

This opt-in path retains the existing DualChunkSwap token
ownership. The previously tested baseline is TP1/PCP8/PP1/DP1/DCP1 with EP
and eager mode. Only KDA projections, short-convolution channels, per-head parameters
and conv/SSM state are head-sharded. MLA, K-pool, MoE and the low-level KDA
kernels are unchanged. Ordinary TP and the replicated PCP baseline remain the
default.

## Enable / disable

Set the same environment variable in the engine and **all** workers, before
starting the server (for containers, pass it into the serving process):

```bash
export VLLM_HCU_GLM53_KDA_CP_HEAD_SHARD=1
```

Unset it or set it to `0` to run the original baseline. Restart the serving
process when changing the setting; live changes to a loaded model are not
supported. No existing profiling script is modified by this change.

### Compilation cache isolation

Before compilation hashing, platform validation resolves the launch switch
into `additional_config["hcu"]["glm53_kda_cp_head_shard"]`. The initial `None`
value means unresolved, not enabled; validation freezes it to a boolean.
Only the GLM5Next target with PCP>1 can enable this policy. PCP1 and unrelated
models resolve to `False` and retain their existing paths.

The serialized mode participates in `VllmConfig.compute_hash()`, so feature-on
and feature-off restarts cannot select the same configuration cache key.
Engine state sizing and worker construction read this same frozen value;
worker-local environment changes do not override it. To switch modes, start
with a fresh engine config and restart the service, rather than mutating or
reusing an already resolved config. Legacy CPU adapters without platform
validation still resolve the environment as before.

This partitions compilation caches; it does not establish graph-mode execution
support or change the eager/greedy restrictions of the MTP decision-sync fix.

With PCP1, native TP construction and state sizing are retained even if this
switch is set to `1`; this does not assert that every native TP configuration
has been qualified on the current runtime.

With PCP>1, the head group size comes from `prefill_context_parallel_size`.
It must divide the positive KDA head count. Configuration admission no longer
requires TP1/PP1/DP1/DCP1 or EP when DP=1. The runtime still requires
`--enforce-eager` for GLM5Next PCP, with either Head switch value and with or
without MTP. Current breakable graph capture records local dummy rows, while
real PCP prefill supplies global KDA metadata; disabling torch compilation
alone does not make PIECEWISE graph replay safe.

DP>1 together with PCP>1 is rejected unless EP is enabled. The non-EP MoE
dispatch path does not yet reconcile post-PCP-gather rows with native DP split
sizes. These are execution-safety guards, not a hardcoded PCP8 restriction.
Positive integer sizes and existing GLM5Next language-only/MLA/KDA checks
remain. Joint TP*PCP ownership/reductions and additional PP/DP/DCP execution
paths are not qualified merely by admission. Do not treat this relaxation as
production support for new topologies; graph support remains deferred.

The width parameter is exercised by CPU regressions at PCP2/4/8. Full-model
multi-HCU correctness, MTP acceptance/state rollback, performance and model
memory capacity still need validation separately at each newly used width.
The earlier PCP8 startup and small-workload A/B results must not be presented
as new PCP2/4 or native-TP8 service results.

## Execution contract

### Head policy and full-model capability are separate

`resolve_kda_head_parallel` is the single CPU-safe selector. It returns an
immutable `KdaHeadParallelPlan(group, size)`: native TP when the switch is off
or PCP=1, otherwise PCP with its configured width. It never multiplies TP by
PCP or rewrites either global process group. With the switch off, "native TP"
also means preserving the existing replicated-KDA PCP baseline; it does not
turn off PCP for the rest of the model.

Head ownership arithmetic itself does not require EP, eager, PP1, DP1 or DCP1.
The runtime scope separately requires eager execution and EP for DP>1+PCP>1.
Passing either level is not full-model execution qualification.
`validate_glm53_pcp_topology` now checks configuration sizes, and the platform
scope validator retains other model-specific checks. Worker construction and
engine state sizing use this same relaxed rule. The legacy `kda_cp_head_size`
API follows the relaxed validation; head count, actual group size and rank
checks remain. Neither a head plan nor admitted config proves mixed-parallel
correctness.

Both head selection and engine sizing read `hf_text_config` when available,
falling back to `hf_config`. Worker construction validates the runtime PCP
group width/rank against the plan and uses its local head-count rule. Native
TP construction and state sizing are still delegated to the original methods.

Reference: the inspected `gf_test_sglang_028_1015` build selects CP heads when
DSA prefill CP is enabled and attention-TP heads otherwise. Its DSA CP model
override sets `attn_cp_size = tp_size // dp_size` explicitly to retain effective
attention TP=1, because its communicator does not reduce attention-TP partial
outputs before replicated dense FFNs. Global SGLang TP and vLLM TP cannot be
equated by their CLI numbers. This refactor copies the group-selection
principle, not a promise of arbitrary TP x PCP, PP, DP, DCP or graph support.
It adds no eight-card constant and qualifies no additional hardware topology.

For `H` heads and PCP width `C`, rank `r` owns heads
`[(H/C)*r, (H/C)*(r+1))`. With H=64, PCP2/4/8 yields 32/16/8 local heads.
Q/K/V/beta, f_b/g_b,
conv channels, A_log, dt_bias and o_proj input channels follow that ownership;
f_a/g_a remain replicated. TP-disabled local linears retain normal vLLM/HCU
loaders (including NN layout) after CP checkpoint slicing, without replacing
the global TP group or changing TP sizes in other layers.

Per KDA layer, gather the current step's full ordered token rows, compute only
the local heads, run local o_proj, SUM all-reduce across PCP, and then use the
existing localize map. The returned tensor from all_reduce is used even when
the communicator is out-of-place. Static head sharding also requires the sum
on decode/verify/dummy forwards; it is not a prefill-only optimization.

Reduce-scatter is intentionally deferred: DualChunkSwap output rows are not
rank-contiguous, unlike SGLang's plain CP contract. A future reduce-scatter
must pack rows by the existing PCP ownership map first.

Both model-level and layer-level state sizing use the head group size. They
retain num_spec, state dtype/orientation, global request/slot metadata and the
existing hybrid KV ownership workaround. KDA head-sharded state does **not**
mean the MLA/K-pool cache becomes context-sharded.

At the scheduler block-size resolution entry point, the GLM5Next architecture,
PCP topology and presence of both Mamba and K-pool-tail state groups explicitly
stamp the replicated token/block ownership contract on the cache config. The
hybrid coordinator consumes this stamp rather than inferring policy from
`pcp_world_size == 8` or from hybrid spec types alone. Logical cache PCP1 is
separate from the KDA head-group width C. Unmarked/unrelated hybrid models are
not rewritten; a stamped width mismatch is rejected. The original vLLM block
size algorithm is still called unchanged.

## Validation boundary

### Explicit prefix-cache hit and MTP replay smoke

HumanEval's short prompts can all be smaller than the hybrid cache block
(the inspected PCP8 run used 640 tokens), so enabling prefix caching and
passing HumanEval does not prove that a cache hit occurred.

Use the opt-in client against a **dedicated, caller-owned** TP1/PCP8/EP8 eager
server with Head sharding, MTP3 and prefix caching enabled. Add
`--enable-prompt-tokens-details` so each response reports `cached_tokens`.
The client never launches/stops a service or resets the shared cache:

```bash
python -m tests.integration.server.glm53_pcp_prefix_cache \
  --base-url http://127.0.0.1:8202 \
  --model default_model \
  --output-dir /tmp/glm53-prefix-hit-smoke-unique-run
```

The output directory must not exist. Four requests execute sequentially:
unique-prefix A cold, A warm, a different suffix B sharing the prefix, and B
warm. They share exactly 4352 input token IDs and generate 256 tokens each.
For the inspected 640-token blocks, this also crosses a block boundary during
generation. A fresh unique nonce avoids needing a global cache reset.

Checks require zero hits on A cold, at least 1280 cached tokens on each later
request, complete multi-step generation, exact greedy replay token IDs for
each repeated prompt, and positive cache-hit/drafted/accepted metric deltas.
Requests, responses and metric snapshots are saved. Replay differences fail
the smoke and require numerical/cache-state investigation; they are not by
themselves proof of a cache bug. Passing is not strict multi-rank numerical
alignment, a full accuracy evaluation, or a performance measurement.

CPU/mock contracts are separate and never load a model or reserve HCUs:

```bash
python -m pytest -q tests/runtime_patch/test_glm53_prefix_cache_probe.py
```

The live client is intentionally not a default PR hardware job: it needs an
explicitly prepared eight-card server and the exact checkpoint. CPU checks
must not be described as a completed live cache-hit validation.

### Observed replay limitations

The inspected PCP8/MTP3 smoke is **not a passing cache-correctness result**.
With Head enabled, the resolved block size was 640 tokens: A cold had zero
cached tokens and the next three requests each reused 3200 tokens. The A
replay matched, but B differed from its third generated token. With the exact
same request payloads and Head disabled, A still matched and B differed from
its thirteenth generated token. Both runs completed all four 256-token
outputs, so output variability is not unique to the head-sharded path; its
numerical or state-related cause has not been established.

Head-off selected 4480-token blocks, larger than these 4363/4366-token
prompts, and therefore had no cache hits. It is not a matched cache-hit
control or evidence that cached KDA state is correct. A follow-up should use
identical longer prefixes, confirm nonzero reuse in both modes, and preserve
the strict replay failures rather than treating them as successful tests.
Fixed-seed greedy token replay is a diagnostic contract, not a substitute
for task accuracy or strict multi-rank numerical alignment.

Run CPU regressions without starting a model service:

```bash
pytest -q tests/runtime_patch/test_glm53_kda_cp_heads.py \
  tests/runtime_patch/test_glm53_pcp.py
```

The new suite checks the opt-in contract, separate/fused/tuple checkpoint
loading through both loader APIs, replicated small projections, per-head
parameters, local projection partial sums, both state sizing entry points,
and reduce-before-localize/out-of-place reduction behavior at PCP2/4/8. It
also covers native-TP fallback with PCP1, uneven causal token reconstruction,
explicit cache ownership stamping, unrelated-cache isolation and MTP config
validation across the width matrix.

CPU arithmetic and simulated rank ownership are not fused-kernel accuracy or
multi-HCU end-to-end validation. Before production use:

1. Compare fixed prompts with the baseline, first without MTP; include short,
   uneven and chunked prefill, continued prompts and multi-token decode.
2. Verify every participating rank's head ranges, conv/SSM shapes and request
   slot ownership. Check that global rows/collective call order match on every
   participating rank, including padding and empty local token partitions.
3. Validate MTP3 acceptance/rejection, state commit/rollback and slot reuse.
4. Benchmark the same workload and actual runtime environment. Measure KDA plus
   new reduction cost and end-to-end TTFT/throughput; do not promise an 8x gain.

The implementation reuses the audited native KDA forward and kernels, but its
CP-specific constructor must be re-audited if upstream adds constructor fields,
changes parameter layouts, or changes checkpoint shard naming.
