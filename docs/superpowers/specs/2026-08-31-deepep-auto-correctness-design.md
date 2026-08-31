# DeepEP Auto Correctness Design

## Goal

Correct the DeepSeek-V4 Channel-INT8 and dynamic DeepEP HT/LL execution paths
so that production configuration preserves model numerics, selects the intended
dispatch quantization, classifies cached prefills conservatively, and safely
reuses the shared DeepEP buffer.

## Confirmed failures

1. The HCU per-activation-token INT8 oracle calls
   `int8_w8a8_moe_quant_config` without preserving
   `layer.swiglu_limit`. The resulting `gemm1_clamp_limit=None` selects the
   unclamped DeepGEMM SwiGLU operation.
2. The `deepep_auto` factory passes `use_fp8_dispatch` but omits
   `use_int8_dispatch`. Channel-INT8 therefore reaches LL dispatch with
   `quant_type=0` instead of `quant_type=1`.
3. The speculative-decode selector infers phase from maximum lengths alone.
   Cached short prefill can satisfy the same length bounds as speculative
   decode even though `CommonAttentionMetadata.is_prefilling` identifies it as
   prefill.
4. HT and LL delegates share one DeepEP buffer, but the LL path does not clean
   its low-latency regions before reuse. The installed DeepEP API explicitly
   requires `clean_low_latency_buffer` after normal dispatch/combine, and the
   upstream vLLM test cleans before every reused LL dispatch.
5. The current HT/LL snapshot is taken inside `prepare`/`prepare_async`, after
   the modular kernel has already queried expert activation and quantization
   contracts. The first call after a phase transition can therefore query the
   previous delegate's contract.
6. vLLM ubatching creates concurrent per-microbatch contexts while the auto
   kernel stores one mutable mode snapshot. The HCU context fields are also not
   propagated into those derived contexts.
7. The HCU DSpark context-KV helper always invokes the uint8 LightOp layout,
   although upstream supports BF16 and other FP8 cache layouts.

## Design

### Preserve the SwiGLU clamp

The HCU INT8 oracle will pass
`getattr(layer, "swiglu_limit", None)` as `gemm1_clamp_limit` when it builds the
per-token W8A8 configuration. The HCU config adapter for
`int8_w8a8_moe_quant_config` will expose an optional clamp argument and use
`FusedMoEQuantConfig.make` when a clamp is provided, because the pinned
upstream helper does not accept that field. Existing calls without a clamp
continue through the upstream helper unchanged.

The test will construct a layer with `swiglu_limit=10.0`, invoke the production
oracle factory, and assert the returned configuration carries
`gemm1_clamp_limit == 10.0`. Both contiguous and masked DeepGEMM expert tests
will then assert that this factory-produced configuration selects the clamped
operation.

### Propagate INT8 LL dispatch

The auto all-to-all factory will derive both flags from
`quant_config.quant_dtype`, reject simultaneous FP8 and INT8 selection, and
pass both flags into `DeepEPLLPrepareAndFinalize`. A factory-level test will
build the Channel-INT8 auto delegate and verify `use_int8_dispatch=True`,
`use_fp8_dispatch=False`; the existing LL dispatch contract test will continue
to verify that this state becomes `quant_type=1`.

### Classify decode from explicit phase metadata

The speculative-decode metadata walker will require every attention metadata
item that contributes sequence lengths to also provide `is_prefilling`, and it
will require every row to be explicitly false. A scalar boolean, a Python
sequence, NumPy array, or tensor is accepted through conservative truth-value
normalization. Missing, empty, malformed, or mixed phase evidence returns HT.

The existing uniform batch descriptor remains authoritative when it explicitly
identifies a uniform decode batch. Metadata-based fallback covers speculative
decode when that descriptor is non-uniform. Tests cover cached short prefill,
mixed prefill/decode, pure decode, pure speculative decode, and absent phase
metadata.

### Clean the shared buffer before LL dispatch

Immediately before each HCU low-latency dispatch, all ranks will call
`buffer.clean_low_latency_buffer` with the same maximum-token count, hidden
size, global expert count, and dispatch quantization group size. Cleaning each
LL invocation satisfies both HT-to-LL transitions and consecutive LL reuse;
tracking only HT-to-LL transitions would leave the documented consecutive-LL
reuse requirement uncovered.

If the selected HCU DeepEP buffer lacks the cleanup API, the path will fail
with a clear compatibility error instead of silently invoking a dirty LL
buffer. Unit tests will verify cleanup ordering and exact arguments for INT8
and FP8 group sizes. The hardware acceptance sequence remains
HT -> LL -> LL -> HT -> LL across all ranks.

### Snapshot mode before modular contract queries

`DeepEPAutoPrepareAndFinalize` will expose `begin_moe_call()`. The modular
kernel calls this hook at the start of `_prepare`, before reading
`activation_format` or `expects_unquantized_inputs`. The hook chooses HT or LL
once and synchronizes the auto experts. `prepare`, experts execution, and
`finalize` use that snapshot and do not resample the forward context midway
through the invocation.

The fixed Mooncake producer/consumer layout uses the same hook but always
returns its fixed mode. Tests will exercise HT-to-LL and LL-to-HT first-call
transitions and verify that contract queries, preparation, expert execution,
and finalization agree.

### Reject concurrent ubatching for deepep_auto

This change does not claim concurrent invocation-local state support. Startup
validation will reject `deepep_auto` whenever
`parallel_config.use_ubatching` is true, including DBO and explicit ubatch
sizes. This prevents both loss of the HCU mode field in derived contexts and
races on the mutable per-kernel snapshot.

Future DBO support requires a separate design that propagates phase state into
each microbatch and associates prepare, experts, and finalize with a
thread-local or invocation-token delegate selection.

### Preserve non-uint8 DSpark cache behavior

The HCU DSpark helper will use the LightOp only when the actual cache tensor is
`torch.uint8`. For every other cache dtype it will delegate to upstream
`_insert_context_kv`, retaining upstream BF16 and per-tensor FP8 behavior.
Tests will assert both the uint8 LightOp call and the BF16 fallback.

## Validation

Each behavior change follows a red-green test cycle. The final software-only
gate is:

- targeted factory, selector, lifecycle, cleanup, config, and DSpark tests;
- the complete affected runtime-patch test files;
- `python -m compileall -q vllm_hcu tests`;
- `git diff --check`.

Multi-rank HCU validation must run the alternating HT/LL sequence and compare
against a fixed-HT numerical baseline. If this host cannot execute HCU
collectives, the handoff will explicitly distinguish software verification
from outstanding hardware verification.

## Non-goals

- Adding deepep_auto DBO/ubatching support.
- Changing the supported DeepSeek-V4 Channel-INT8/FP8 deployment recipes.
- Refactoring unrelated MoE or attention implementations.
