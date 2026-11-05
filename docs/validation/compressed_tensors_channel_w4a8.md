# compressed-tensors channel W4A8: minimal HCU adapter

This proposal replaces the model-specific quantization entry in #308 with a
worker-side adapter under the existing `compressed-tensors` configuration.
It does not register `glm_moe_dsa_w4a8` or change `-q` names.

Native `get_scheme_dict` parses targets, class-name regexes, format inheritance
and projection ignore rules. The adapter checks all logical experts and three
projections, rather than probing a fixed number of layers or expert 0 only.
Unsupported mixed expert formats are rejected before constructing a backend.
W8A8 linears and W4A16 owners retain their existing dispatch.

The only checkpoint conversion is uint4b8 int32 packing to signed, high-nibble
first SlimQuant bytes, plus FP32 channel scales divided by 16. Existing
SlimQuant Triton execution supplies the inverse scale compensation. Main and
MTP loaders rename packed tensors only for owners using this adapter; native
packed owners keep their names. GLM MTP keeps its numeric runtime layer
prefix. A local configuration copy resolves `model.decoder` quantization rules
through the native matcher when a physical path has no explicit rule. Physical
path rules and ignore entries retain priority; the main config is not replaced.

## Deliberately limited first proposal

- `--moe-backend triton`, EP=1, symmetric channel INT4 weights and dynamic,
  symmetric per-token INT8 activations; no expert bias or fused shared experts.
- TP slicing is covered by CPU native-loader tests at TP 1/2/4, all ranks.
  This is not a multi-HCU numerical validation.
- AITER gemm2, DeepEP, EPLB, online weight replacement/reload and Mooncake PD
  equivalence to #308 are not claimed. No common runtime/reload code is changed.
- The new PR stays draft pending v0.28.1-dev service and accuracy validation.
  This host has DAS v0.25.1 installed; its CPU parser/loader tests do not establish
  wheel compatibility with the target version.

Example for the target DAS v0.28.1-dev image, after building the plugin there:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
vllm serve /module/GLM-5.3-Channel-INT4-w4a8 \
  --tensor-parallel-size 8 --moe-backend triton \
  --kv-cache-dtype fp8_ds_mla --block-size 64
```

No `-q` override is required. This command is a validation starting point,
not a successfully executed service run on this host.

```bash
VLLM_PLUGINS=__disabled__ PYTHONPATH="$PWD" python -m pytest -q \
  tests/runtime_patch/test_compressed_tensors_w4a8.py \
  tests/runtime_patch/test_quant_gemm_aiter.py -k 'compressed or prequantized'
```

Local validation: 36 targeted CPU tests pass, covering all packed byte patterns,
native scheme selection, owner-local naming (including MTP), and native expert
weight/scale TP loading. The actual checkpoint config selects this adapter for
layers 3, 60, 77 and the physical MTP prefix `model.layers.78` using native
matching plus the decoder alias; this config probe stubs the execution-method
constructor and does not run inference. Constructor regression coverage executes
the production MTP and decoder `__init__` bodies with attention/MoE leaves
stubbed; importing the complete target model needs the target wheel.

Self-review also ran the complete two-file suite: 304 passed, 15 failed. An
unchanged upstream checkout produced the exact same 15 failure IDs (277 passed),
including missing v0.28 APIs and the installed BoltOps contract mismatch.

On one BW1101 HCU, converted tensors were compared directly through BoltOps
W4A8 and W8A8 MoE kernels at E=4, H=I=128, top-k=2, BF16/FP16 and
batch 1/7/32. Maximum relative L2 difference was 5.51e-5; maximum absolute
difference was 6.11e-5. This validates the tested packing/scale convention, not
model accuracy. The full adapter probe stops at the existing BoltOps signature
guard: this host lacks `use_mxfp4_w4a8`, `use_ue8m0`, `use_mxfp8`, and
`gemm1_beta` required by the target branch's audited API. No production guard
was relaxed; the direct kernel comparison is separate from integration testing.
