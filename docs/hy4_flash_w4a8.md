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

## Explicit AITER candidate

For the original `/quant_model/Hy4-flash-Channel-INT4-w4a8` checkpoint, select
`--moe-backend aiter` in the qualified ordinary-TP command. This is sufficient
for model-local W4A8 admission even when the checkpoint's
`flash_w4a8_backend` is absent or `"disabled"`; no `hcu_adapter.json`,
`--hf-config-path`, or checkpoint edit is required. Retain the existing BF16,
TP4, DFly5, sparse-attention and FULL_AND_PIECEWISE execution arguments below.
The same explicit backend can use ordinary-TP Eager for a matched comparison.
`auto` does not implicitly select AITER W4A8; `auto` with the explicit
`adapter_reference` flag keeps the exact CT reference.

The candidate reuses CT parameter creation/sharding/completeness checks and
losslessly converts the packed INT4 bytes once after loading. Public AITER
configuration and weight preparation choose the native layout; callers do not
force shuffle. Packed Parameter owners and INT32 storage views remain stable
for Graph replay, and the original full CT packed storage is released.
SwiGLU clipping is passed as `gemm1_limit`; Core retains routing, shared-expert
execution and TP reduction. Only the admitted Flash serialized channel-INT4 /
dynamic-token-INT8 recipe is eligible; FP8 selection is unchanged.

CPU weight offload is temporarily rejected at model configuration admission
for this AITER W4A8 candidate, before expert loading or the first forward.
This includes active `--cpu-offload-gb` (with or without UVA, including
selective parameter lists) and prefetch weight offload. Core's non-UVA
`functional_call` swaps layer parameters during forward, while this method's
prepared native views/scales assume resident storage. UVA and prefetch
lifecycles are not yet qualified either. Leave CPU offload at its default zero
until migration-aware native aliases and Graph replay are validated. This
protection does not change the exact CT reference or FP8 selector admission;
it does not establish offload support for those separate methods.

AITER token INT8 arithmetic uses FP32 `absmax/127`, while the exact CT oracle
uses `absmax/127.5` with BF16 scale/quotient rounding. These are different
numerical contracts. The local four-expert real-weight screening and
changed-input Graph replay do not establish full-model accuracy or speedup.
Acceptance remains pending matched EvalScope HumanEval 164/C64 within three
percentage points of the frozen W4A8 Eager score, followed by profiler-disabled
throughput/TPOT and trace validation on the actual candidate. Unsupported
layouts/devices/configurations and missing requested APIs fail closed; real
native launch errors propagate.

## Admission / limitations

The exact CT reference defaults to `flash_w4a8_backend="disabled"`. Explicitly
set the **text** config field to `"adapter_reference"` to test that reference.
The explicit AITER candidate described above uses the original checkpoint
metadata with `--moe-backend aiter`; no extra metadata file is required. It requires the reviewed HCU
platform, BF16 compute and ordinary TP deployment. Eager V2 execution remains the
numerical reference. The active Core033 extension additionally admits the
restricted FULL_AND_PIECEWISE configuration documented below; EP, DP, sequence
parallelism and EPLB remain rejected.

The exact CT reference prioritizes a bounded functional baseline, not throughput:
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


## Qualified FULL_AND_PIECEWISE lane

The active Core033 extension also admits the existing Flash Graph-bank
configuration: V2, ordinary TP4, DFly5, fp8_ds_mla KV, disabled prefix caching
and custom all-reduce, no LoRA/KV transfer/prompt embeddings or multimodal
requests. Other graph modes and topologies remain rejected. Keep the explicit
text-config adapter_reference opt-in during qualification. Eager remains the
numerical reference. This changes admission only; native integer arithmetic,
clipping, packed weights and Core reduction ownership are unchanged.

Component changed-input capture/replay was verified separately with random CT
packed tensors at routed latentH1536/TP4intermediate384/E512 and M1/33/64;
this is not wholemodel accuracy or performance acceptance. The exact-reference whole-model Eager and FULL_AND_PIECEWISE runs each
completed EvalScope HumanEval 164/C64 at 161/164 (98.17%), with no request or
infrastructure failures. Failed task sets differ: Eager 116/145/146 and Graph
47/145/146. Equal aggregate scores do not imply token-wise equality. This
evidence belongs to the original CT reference, not the AITER candidate.
The AITER candidate still requires actual installed-method Graph replay,
whole-model loading and HumanEval, followed by matched performance validation.

## Installed-package candidate command (qualification pending)

Install this Plugin branch with the compatible Adapter/AITER packages in the
activated vLLM 0.33 environment. The command below is the AITER candidate recipe,
not a claim that its full-model validation has completed. No Core source patch,
extra HCU JSON, capability environment variable, or checkpoint edit is needed.
Choose four idle devices for `HIP_VISIBLE_DEVICES`; the example indices do not
reserve them or describe the two reference validation groups.

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
vllm serve /quant_model/Hy4-flash-Channel-INT4-w4a8 \
  --host 127.0.0.1 --port 8000 \
  --served-model-name hy4-flash-int4-w4a8 \
  --tensor-parallel-size 4 --dtype bfloat16 --max-model-len -1 \
  --max-num-seqs 64 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 --kv-cache-memory-bytes 17179869184 \
  --kv-cache-dtype fp8_ds_mla --generation-config vllm \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --speculative-config '{"method":"dfly","num_speculative_tokens":5,"enforce_eager":false,"kv_cache_dtype":"fp8_ds_mla"}' \
  --attention-backend FLASHMLA_SPARSE --no-enable-prefix-caching \
  --reasoning-parser hy_v4 --moe-backend aiter --disable-custom-all-reduce \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","inductor_compile_config":{"benchmark_combo_kernel":false}}'
```

For a matched AITER Eager comparison, remove `--compilation-config` and its JSON,
add `--enforce-eager`, and set speculative `enforce_eager` to `true`. Retain all
other arguments and the same physical devices/workload. For the original exact
CT reference, use `--moe-backend auto` plus the complete metadata-only
`--hf-config-path` snapshot described above; those extra metadata steps are
specific to the experimental reference and are not required by the AITER CLI.

The fixed 16 GiB KV budget is a qualification workaround for the installed
Core's automatic startup memory-profile allocator policy. It is not an
automatic memory-profile fix or a universally optimal production cache size.
The proposed AITER method keeps one packed weight owner and obeys native
per-token unique route IDs, including zero-weight Graph padding. Native INT8
rounding remains different from CT and must pass whole-model validation.
