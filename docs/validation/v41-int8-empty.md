# DSV4.1 independent CT INT8 empty rows

This change is limited to the owned CT W4A8 configuration's ordinary linear routing and a separate CT W8A8 INT8 method. It does not change Engram, expert kernels, native factories, selectors or CT classes. Constructor, parameter creation and weight loading remain native. QuantizedActivation is unsupported by native INT8 and is explicitly refused.

## Fixed failure contract (written before implementation)

1. Native Triton scaled_mm rejects M=0. Empty rows must bypass local quantization/GEMM only, leaving outer Column all-gather and Row all-reduce in place.
2. Determine empty M from the product of leading dimensions. A zero K is invalid, even when input.numel()==0. Require rank >=2 and positive K/N.
3. Native checkpoint weights are [N,K]; audited ROCm AITER/Triton finalization makes them [K,N]. A square shape cannot prove finalization. Record a lifecycle/layout witness only after native postprocess succeeds, and reject repeated finalization without applying another transpose. A failed/repeated postprocess invalidates empty execution.
4. Validate native scheme/kernel ownership and finalized weight identity, shape, stride, int8 dtype, positive [K,N], local output partition width, floating input dtype/device, FP32 scalar/per-channel weight scales, and bias width/device/dtype before returning empty output. Preserve native static/asymmetric scale, zero-point and AZP state; reject malformed metadata rather than silently bypassing it.
5. Native INT8 output dtype follows Tensor input dtype. Do not borrow FP8 kernel.config.out_dtype or infer N from global output_size.
6. Native INT8 has no prequantized activation key or zero-point metadata contract. Reject every QuantizedActivation, including empty or quant_key=None, before considering an empty return.
7. Nonempty inputs delegate to native CT unchanged. Native exceptions propagate. Unknown/packed kernels do not obtain an empty fast path; native constructor/postprocess/apply stay responsible for their behavior.
8. Preserve public v2 loader registration: Column/Merged select it by method class name. Do not replace init_int8_linear_kernel, global registries, selectors, native method classes, or quantconfig factories.
9. Route only exact native CT W8A8 INT8 linear methods/schemes. Preserve FP8, ignored/unquantized layers and expert routing. Per-group format is material: pack-quantized INT may select WNA8Int before W8A8Int8; mixed fixtures must explicitly use int-quantized for the INT8 linear group.
10. Native CT does not declare preprocessed-cache support. Do not claim supports_pre_processed_weights, bypass loader finalization, or add a second finalized-weight loader format.

## Fixed CPU replay, before production code

The replay uses real owned config parsing, native CT scheme selection, real Column/Merged/Row constructors, public v2 parameter loaders, and real native INT8 weight finalization on CPU tensors. Hardware admission, device selectors, activation quantization and GEMM are controlled doubles. CUDA/HIP/ROCR visibility is empty; plugins, usage stats, downloads and bytecode writes are disabled. No GPU numerical/performance claim is made.

The fixed cases cover mixed FP8/W8A8/ignored/pack format routing; [N,K] -> [K,N] values/strides; direct/merged loads, TP rank and disable_tp; channel/tensor/fused/static/asymmetric metadata; M=0 at rank 2/3; dtype/device/output width and bias; outer collective spies; nonempty delegation/error identity; malformed input, scale, zp/AZP, QA and lifecycle; official callable identities. Both rectangular and square weights are used. Replay evidence is restricted to tests/vllm033/.cache/v41-int8-empty, ignored by git.

Validation is recorded below only after executing the fixed replay. Existing FP8 replay is not evidence for this INT8 path.

## Execution evidence

- Core HEAD: `368701db06c78f3e8654bb51f44753e5064e262c`; plugin HEAD at replay: `16e15b12b40b6c72a381f3fb97f722be905b210b`.
- Container: bw47 / `vllm_032_1030_laibao`, uid/gid 1151, existing `/usr/local/bin/python3.12`; no installation.
- Preimplementation `red-fixed/report.json`: **20/49 passed**, exit 1. Empty rows reached the deliberately rejecting native quantizer double; QA and repeat/failure-finalization cases also failed before implementation. This baseline preceded production-file creation.
- `green/report.json`: 48/49 passed. The existing mixed-routing case exposed missing FP8 model-config context after its earlier owned-route assertion began passing. Corrected that fixture context only; no replay cases or pytest tests were added after implementation.
- Final `final-storage/report.json`: **49/49 passed**, exit 0, `cuda_initialized=false`; GPU visibility was empty. Real CT parameter creation/loading/finalization and native nonempty dispatch were used; hardware admission, quantization and GEMM were doubles. This repeats the same 49 frozen cases after review added storage identity to the witness; no cases were added.
- Python 3.12 AST parsing for the two production Python files plus the replay, and `bash -n` for the runner, passed without writing bytecode. Scoped `git diff --check` passed.
- Import logs contain the existing missing optional CPU `_C_AVX2` extension notice. The replay completed using its explicit quantizer/GEMM doubles; this is not evidence of native CPU or GPU GEMM execution.

All report/log evidence is under `tests/vllm033/.cache/v41-int8-empty/`. This directory is already ignored; no ignore rules were changed. The implementation task did not stage or commit files. The Lead subsequently ran the existing combined CPU suite: **312 passed, 66 skipped, 14 warnings in 190.00s**, exit 0; evidence is `tests/vllm033/.cache/v41-pcp-engram/regression.log` / `regression.xml`. No new pytest unit tests were added.

## Final review boundaries

The owned method inherits the exact native constructor and create_weights. Its only new postprocess work is lifecycle state and a reference/shape/stride/data-pointer/device/dtype witness after native finalization. Repeating finalization refuses before a second transpose, clears the witness and leaves empty execution invalid; a native finalization exception leaves that state sticky and propagates the original exception. Recovery requires reconstruction. Unknown kernels retain native postprocess/nonempty execution and cannot use this empty path.

Every nonempty Tensor delegates immediately to native CT, including malformed nonempty shapes. QuantizedActivation is explicitly refused for both empty and nonempty inputs because native INT8 has no such capability. Lifecycle/metadata checks specialize only the empty Tensor path; they do not add new nonempty validation or a fallback implementation. The empty path only admits exact native AITER/Triton kernels and the audited [K,N] ABI; AITER additionally retains its alignment, dtype and per-token/channel restrictions.

The fixed replay covers direct/merged/row public loading rather than the full DSV4.1 model checkpoint mapper or Engram model construction. The separate Engram/PCP replay and the Lead's overall regression are necessary complementary evidence. Reload/refit, post-finalization weight storage/device migration and preprocessed weight-cache support are not added. Replacing the same Parameter's .data with new storage also invalidates empty execution; the storage witness prevents same-shape/same-stride replacement from being mistaken for the finalized operand.

Independent read-only final review found no concrete blocker in the frozen implementation. One inherited checkpoint-routing boundary remains: a group explicitly setting `format: null` while the global format is `pack-quantized` has not been replayed. The existing owned auto-selector uses `group.get('format', global_format)`, whereas native CT scheme parsing can use the global format when the group format is None. The fixed mixed-schema fixture sets explicit formats; this change does not broaden automatic checkpoint recognition.

The replay calls each Parameter's attached `weight_loader`; for the registered owned method, native Linear construction selects the bound layer `weight_loader_v2` and CT passes it into those Parameters (`linear.py:530-544`, INT8 scheme `:59-81`). Thus the parameter loader calls traverse the actual native v2 dispatch, including direct, merged and nonzero TP-rank fixtures; they do not prove whole-checkpoint mapper or model loading.
