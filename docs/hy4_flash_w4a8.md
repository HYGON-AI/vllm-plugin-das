# HYV4 Flash channel INT4 / dynamic-token INT8 (experimental)

This opt-in lane depends on the HYV4 Flash/DFly foundation in the Flash support
PR. It is separate from the FP8 greedy-head and router performance experiments.

## Contract

* Routed experts: compressed-tensors `pack-quantized`, symmetric channel INT4,
  low-nibble-first offset-binary packing in INT32 words.
* Attention/shared/dense/DFly linears: the checkpoint's channel INT8 groups,
  dynamic token INT8 activations, selected by the existing CT linear machinery.
* Keep BF16/ignored weights and activation metadata intact. Do not relabel this
  checkpoint as SlimQuant, drop input quantization, or silently run W4A16.
* Expert loads retain Core's TP sharding. `.weight_packed`, `.weight_scale` and
  `.weight_shape` participate in the Flash streaming-load completeness audit.
* The functional reference reuses
  `hcu_adapter.kernels.w4a8_int8_native_triton.w4a8_int8_native_masked_experts`.
  Its two integer dot products preserve the adapter's input-dtype token-scale,
  division and projection stores, including the BF16 SiLU-times-up store and
  configured SwiGLU clipping. Packed byte views alias the original weights;
  no full-model BF16 unpack is made.
* Dispatch keeps original token order within each expert; combine keeps the
  original router-slot order. Core owns shared experts and TP reduction.
* CT target matching also follows Core: expand `Linear` to `RoutedExperts`
  with the inherited helper before resolving individual projections. Explicit
  `RoutedExperts` schemes retain precedence, and ignore rules still apply.
  Expansion happens on the model-local config, preserving the source config.

## Admission / limitations

The default is `flash_w4a8_backend="disabled"`. Explicitly set the **text** config
field to `"adapter_reference"` to test this lane. It requires the reviewed HCU
platform, BF16 compute, eager V2 execution and TP-only deployment. EP, DP,
sequence parallelism, EPLB and model CUDA/HIP Graph execution are not qualified
and must not be enabled for this reference lane.

This implementation prioritizes a bounded functional baseline, not throughput:
it dispatches at most 32 tokens per call into expert-major scratch buffers.
Allocations and per-expert work are deliberately not presented as a fast path.
Do not use these measurements to claim optimized W4A8 model performance.

For the DFly validation recipe, retain:

```bash
--max-model-len 8192 \
--speculative-config '{"num_speculative_tokens":5,"method":"dfly"}' \
--attention-backend FLASHMLA_SPARSE \
--enforce-eager --tensor-parallel-size 8
```

8192 is the total input + thinking + final-output budget. The experimental
DFly bootstrap from the foundation PR is still required; this change does not
establish standard-dispatcher, wheel or container-image qualification.

## Validation boundaries

CPU contract tests:

```bash
python -m pytest -q tests/models/hy_v4/test_flash_w4a8.py \
  tests/models/hy_v4/test_flash_runtime_contracts.py
```

With the target Core installed, also run
`tests/models/hy_v4/test_flash_w4a8_selector.py`. It imports the complete W4A8
module and uses real Core CT target matching for class and regex targets,
explicit fused-layer precedence, ignore rules and repeated selection. Only
the legacy package namespaces and routed-layer construction are isolated;
this CPU check does not qualify standard package startup or model execution.

GPU qualification must use an otherwise idle, exclusively allocated device set.
Record the checkpoint/config/index hashes, source snapshot, actual Core method,
loader comparisons and all rank results. The operator qualification loads four
real routed experts through the actual Core TP8 loader and compares both integer
projections, clipping, dispatch and combine with an independent raw-checkpoint
reference. This is **not** full-model accuracy, DFly acceptance, HumanEval,
long-context semantic accuracy, performance or image qualification.

The next release gates are full-model loading; target and DFly smoke inference;
then matched EvalScope HumanEval with a total 8192-token budget, followed by
end-to-end performance and package/image testing. FP8 accuracy numbers cannot
be inherited by this INT4 checkpoint. Keep the MR Draft until those gates pass.

Run the explicit TP8 qualification program in the same activated experimental
plugin/adapter environment (do not launch it alongside a serving/evaluation job):

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  tests/models/hy_v4/qualify_flash_w4a8_tp8.py \
  --model "$MODEL" --output "$NEW_EVIDENCE_DIRECTORY"
```

Initial operator evidence (2026-11-21): Torch 2.13.0, HIP 7.2.26365, vLLM 0.33.0,
TP8 HCU. M=1/2/7/33/128/512, contiguous and feature-strided inputs: 12 cases
per rank, all eight ranks passed with exact equality before and after TP sum.
The actual Core loader's packed gate/up/down weights and channel scales matched
independently sliced raw checkpoint tensors bitwise. Storage aliasing and clean
process-group teardown passed; torchrun exited 0. These are twelve scenarios
repeated on eight ranks, not 96 independently sampled model cases. Testing used
an explicit source overlay, not a standard-dispatcher or installed-image lane.

To enable the text-config flag from the CLI for a VL checkpoint, create a
**separate metadata-only config snapshot** with the complete original config and
`text_config.flash_w4a8_backend="adapter_reference"`, then supply it through
`--hf-config-path`. Keep `--model` pointing at the original checkpoint. Do not
pass a partial `text_config` dict through `--hf-overrides`: the installed
Transformers `update` replaces that attribute rather than deep-merging it.
Never edit the original checkpoint or installed packages for this experiment.


### Whole-parameter completeness

The outer Flash loading session must require all six expert
`w13/w2_weight_{packed,scale,shape}` parameters, even if no shard for a parameter
was observed. The per-expert shard audit only sees names present in the stream.
Keep the official Core tracker and shard checks after this explicit audit.
Core builds with stricter quantized tracking already provide an additional
check; compatible Core builds must not be assumed to have that stricter policy.
The completeness regressions isolate the final Core tracker to verify the
Plugin audit itself, including nested sessions and exception cleanup. They do
not test checkpoint sharding, weight postprocessing or full-model inference.
