# DSV4.1 Channel-FP8 target loading and empty local rows

## Scope and source decisions

This increment follows configuration routing commit `7c59fb1`. Core is read-only
at `368701db06c78f3e8654bb51f44753e5064e262c`, Adapter is read-only at
`3257a934e683d31edcf570c7bc455564aa6db00b`. The historical empty-row reference
is `fae177c6`; loading references are `17c91bab` and `cc1b9a7d`.

The actual v0.33 V41 AMD target loader already resolves direct and inverse scale
suffixes at `model.py:1026-1035`, used for stacked, expert and direct parameters.
The draft loader does not have that resolver. Target compatibility therefore
reuses the native loader, mapper, CT parameter classes and weight finalization.
Draft alias publication/accounting remains a separate increment; this change
must not advertise DSpark support.

Native FP8 scaled-MM already accepts `QuantizedActivation`. Do not restore the
old `x_and_scale_quanted` keyword or patch the official kernel class. Empty local
rows must return through a Plugin-owned CT LinearMethod, leaving the enclosing
Linear forward and its distributed collectives intact. Nonempty execution and
CT weight processing remain native. Register the new LinearMethod through the
public weight-loader-v2 decorator so merged projection loading keeps its ABI.

## Failure modes recorded before production changes

The replay below will be written and run before the production adapter changes.
It is CPU source/contract integration using small tensors, not GPU/model E2E.

| ID | Failure | Required observation |
| --- | --- | --- |
| L01 | Target scales map to `_inv` while CT stores the direct name. | Real native target `load_weights` loads direct/scaled names and returns canonical parameter names. |
| L02 | Fused projection scale shards are mixed, reordered or only partly loaded. | Real merged projection loader loads distinct shard values in order, with exact tensors. |
| L03 | New LinearMethod silently falls back to weight-loader-v1. | The public v2 registry includes the Plugin method and constructed merged layers use their v2 loader. |
| L04 | Channel weights/scales change during CT finalization. | Native postprocessing produces the expected transposed weight and channel scales; mapper-owned input names remain untouched. |
| L05 | Empty BF16/FP16 input launches quantization or GEMM. | Sentinel local kernels refuse invocation; zero-row result still has exact shape/dtype/device. |
| L06 | Higher-rank empty leading dimensions lose their shape. | Preserve each leading dimension for shapes `(0,K)`, `(2,0,K)`, `(0,3,K)`. |
| L07 | Empty prequantized input is quantized again or loses original output dtype. | Matching native QuantizedActivation returns empty output with original shape/dtype. |
| L08 | Bad empty input K, device, processed layout or bias is silently accepted. | Reject malformed local contracts explicitly. |
| L09 | Nonempty input is intercepted or native errors are swallowed. | Exact call arguments and native failure propagate through the inherited CT method. |
| L10 | BF16 wo_a, ignored layers, experts or unrelated models are captured. | Existing routing contracts stay green; the new method is selected only for native channel FP8 Linear schemes. |
| L11 | Official classes are rebound. | Official CT LinearMethod and FP8 kernel method identities remain unchanged across activation and construction. |
| L12 | The replay imports the wrong core or initializes a GPU. | Record actual source paths, SHAs, installed metadata and assert CUDA remains uninitialized. |
| L13 | CPU layer doubles are reported as model/kernel acceptance. | Record doubled boundaries and omit accuracy/performance/device claims. |

## Reproduction and results

Run in the prepared environment, with the pinned Core and Adapter as siblings:

```bash
VLLM_TEST_PYTHON=/path/to/python3.12 \
  bash tests/vllm033/run_dsv41_channel_fp8_loading_cpu.sh /path/to/output
```

The wrapper disables GPU visibility and network model lookups, imports production
Plugin modules from source, and saves `replay.log` and `report.json`. Run as the
checkout owner. It does not install or update packages. The report records the
three Git SHAs, actual native target source path, installed vLLM metadata, every
case and CUDA initialization state.

The fixed pre-implementation replay recorded **8/25 passed, 17/25 failed**. Native
ordinary/fused scale loading, CT finalization and nonempty delegation already
passed; missing Plugin method registration and empty-row paths failed. The first
preliminary replay used broad exception checks; before implementation those
checks were narrowed to require the relevant contract error, and prequantized
shape/dtype/scale and inferred output dtype boundaries were added. No new cases
were added after production implementation.

The same fixed replay, reimporting final production modules in a new process,
recorded **25/25 passed**, with `cuda_initialized=false`. Evidence is retained at
`tests/vllm033/.cache/dsv41-channel-fp8-loading/green-final/`. The earlier final
red run remains in `red-final/`. `source-manifest.json` records the exact source
hashes corresponding to the candidate.

The existing portable suite finished with **312 passed, 66 skipped, 14 warnings
in 162.58 seconds**. Logs and JUnit are retained as `regression-final.log` and
`regression-final.xml` in the same evidence root. The first regression invocation
omitted its required `VLLM_TARGET_ROOT`: 13 source-root checks failed, 299 passed,
66 skipped. That run remains as `regression.log`/`regression.xml`; the rerun only
added the required environment variable and did not change test expectations.
AST parsing, shell syntax and `git diff --check` passed.

This replay constructs real CT channel-FP8 parameters and real merged/ordinary
Linear layers at TP1, loads tiny tensors through the native V41 target mapper
and loader, finalizes through CT, and invokes the native outer Linear forward.
Hardware capability admission, TP group queries, quantization and GEMM are CPU
doubles. The tiny target fixture has no experts; expert alias resolution is a
static source finding, not an expert-loading runtime pass. Real GEMM, native
FP8 quantization, distributed collective order, Graph and checkpoint/model
inference are not exercised.

The existing container is `vllm_032_1030_laibao` on bw47, Python 3.12.12. Its
installed vLLM metadata remains `0.32.0+rocm723`; Python source was explicitly
selected from the pinned v0.33 checkout. The missing `_C_AVX2` import warning is
retained, not counted as CPU native-ABI or kernel acceptance. No accuracy or
performance result is claimed. The Adapter's dense Triton fallback is validated
for gfx936 only; this change makes no gfx938 execution claim.

## Integration summary

- Production scope: the distinct Plugin config selects
  `HcuDSV41ChannelFP8LinearMethod` only for a native CT W8A8 FP8 Linear scheme.
- Registration: the official `register_weight_loader_v2_supported_method`
  decorator preserves merged projection loading. No legacy PATCH_ID/dispatcher
  callback is added, and no official class method is rebound.
- Missing/malformed processed layout, input K/device, prequantized dtype/scale
  and bias contracts fail explicitly. Native nonempty errors propagate.
- Difference from `fae177c6`: the guard is owned by the distinct Plugin method;
  there is no global scaled-MM wrapper/custom-op patch. Native
  `QuantizedActivation` replaces the old extra keyword argument.
- No Core or Adapter source changes. DSpark draft naming/accounting remains for
  the next increment and is not fixed by target's existing resolver.

## Follow-up contract review

A late static review of local commit `09d188e` identified three local-contract
risks before the follow-up implementation:

- Empty input could bypass malformed weight scale shape/device/floating dtype.
- Processed-layout metadata alone did not validate the weight's physical stride.
- Requiring bias dtype to equal output dtype rejected a floating bias that the
  native target Triton contract permits (`triton_scaled_mm.py:200-206`).

The correction must validate positive weight dimensions, native weak-contiguous
layout and scalar/channel floating scales, while keeping floating bias admission
consistent with the native path. Use the existing native layout predicate;
do not introduce a column-major-only restriction where native Triton also accepts
row-major/leading-dimension-padded layouts. No Core class or registry changes are
needed. The original fixed 25-case replay remains unchanged; no unit tests or
assertions are appended after implementation. These additional malformed-weight
and mixed-bias-dtype boundaries are reviewed against source, not presented as
new runtime test coverage. Native GPU execution remains unverified.

After the correction the unchanged replay passed **25/25**, with CUDA still
uninitialized. Output is retained in `review-correction/report.json` and
`review-correction/replay.log` under the evidence root above. AST parsing and
diff checks passed. The 312/66 full-suite result above belongs to the preceding
candidate; this small local-contract correction reran only the existing replay.
