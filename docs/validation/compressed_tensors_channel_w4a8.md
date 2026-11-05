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
packed owners keep their names. GLM MTP constructs its decoder using the
checkpoint's `model.decoder` quantization prefix, so its own rules apply.

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

Local validation: 31 targeted CPU tests pass, covering all packed byte patterns,
native scheme selection, owner-local naming (including MTP), and native expert
weight/scale TP loading. The actual checkpoint config selects this adapter for
layers 3, 60, 77 and `model.decoder` using the native parser; this config probe
stubs the execution-method constructor and does not run inference.
