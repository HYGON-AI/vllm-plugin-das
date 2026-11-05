# Channel W4A8 for GLM Mooncake PD

The target is the scenario reported in #308: two-node 1P1D Mooncake, P TP8 and
D TP8, with GLM-5.3 channel INT4 experts and INT8 linears. D-side DP8+EP was
not validated in that report. This adapter keeps `quant_method: compressed-tensors`
and uses native parsed schemes; Mooncake does not require a model-specific
`glm_moe_dsa_w4a8` registry name. The connector transfers KV caches, not model
weights, and does not call the weight-reload APIs.

## Implementation

- Native matching covers all logical experts and gate/up/down projections.
  Mixed expert schemes and unsupported output-activation quantization fail.
  Whole-owner ignore rules bypass quantized selection, including MTP aliases.
- Checkpoint uint4b8 int32 weights become signed SlimQuant bytes; channel scales
  become FP32 divided by 16. Packed-name normalization resolves the native
  `MoERunner.routed_experts` hierarchy and remains owner-local.
- `auto` and `aiter` install the eight-nibble regroup plus AITER gemm2 layout
  once at load time. P/D token counts can choose different MOE_C configs. The
  adapter copies each selected config and disables generic weight/scale shuffle
  locally, preserving the selection cache and other quantization methods.
  Existing runtime checks still reject an incompatible physical layout.
- Explicit `triton` retains canonical packing and existing scale compensation.
- GLM MTP keeps numeric runtime/attention layer names. A local quantization
  config resolves canonical `model.decoder` rules without replacing the main
  config; explicit physical rules and ignore entries have priority. This also
  preserves the layer identities used by KV registration.

Initial scope: EP=1, symmetric channel INT4 weights and dynamic symmetric token
INT8 activations, no expert bias or fused shared experts. Set the shared-expert
fusion flag below on both P and D. DeepEP, EPLB, online weight replacement and
AITER checkpoint reload are outside this MR; an AITER checkpoint reload is
rejected before loading into converted storage. No runner/worker reload hook or
common EP fallback is added. Mooncake KV transfer does not invoke those APIs.

## P/D reproduction commands

These are target-image commands, **not a successfully executed two-node PD run**.
Use the same DAS v0.28.1-dev image, plugin commit, checkpoint, tokenizer, dtype,
block size and KV dtype on both nodes. The image must provide the matching
vLLM/Mooncake APIs and AITER `w4a8_moe_layout_shuffle_gemm2` plus W4A8 MOE_C configs.
No `-q` override is needed. These commands do not enable MTP.

Prefill node (set P_HOST to this node's reachable address):

```bash
export P_HOST='<prefill-address>'
export VLLM_HOST_IP="$P_HOST"
export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
vllm serve /module/GLM-5.3-Channel-INT4-w4a8 \
  --served-model-name glm53-w4a8 --host 0.0.0.0 --port 8010 \
  --tensor-parallel-size 8 --moe-backend aiter \
  --kv-cache-dtype fp8_ds_mla --block-size 64 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

Decode node (set D_HOST to this node's reachable address):

```bash
export D_HOST='<decode-address>'
export VLLM_HOST_IP="$D_HOST"
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
vllm serve /module/GLM-5.3-Channel-INT4-w4a8 \
  --served-model-name glm53-w4a8 --host 0.0.0.0 --port 8020 \
  --tensor-parallel-size 8 --moe-backend aiter \
  --kv-cache-dtype fp8_ds_mla --block-size 64 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

Proxy: set P_HOST/D_HOST to the addresses above and VLLM_SOURCE_ROOT to a full
checkout matching the target image. vLLM versions place this example in either
of the paths below; require an existing script instead of assuming a wheel
contains examples. The `--prefill URL BOOTSTRAP_PORT` arguments were checked
against the locally available proxy source.

```bash
export P_HOST='<prefill-address>' D_HOST='<decode-address>'
export VLLM_SOURCE_ROOT='<matching-vllm-source-checkout>'
PD_PROXY="$VLLM_SOURCE_ROOT/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py"
if [ ! -f "$PD_PROXY" ]; then
  PD_PROXY="$VLLM_SOURCE_ROOT/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py"
fi
test -f "$PD_PROXY"
python "$PD_PROXY" --host 0.0.0.0 --port 8000 \
  --prefill "http://$P_HOST:8010" 8998 --decode "http://$D_HOST:8020"
```

Client on the proxy host:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-w4a8","messages":[{"role":"user","content":"请解释为什么天空是蓝色的。"}],"temperature":0,"max_tokens":128,"stream":false}'
```

Acceptance still required: verify actual KV transmission/receipt, compare fixed
prompts and accuracy evaluation against a single-service TP8 AITER baseline,
and exercise short/long prefill, decode and concurrency. A successful HTTP
response alone is insufficient. MTP+PD and D DP8+EP need separate validation.
For transfer tracing, enable `VLLM_HCU_MOONCAKE_TTFT_TRACE=1` and
`VLLM_LOGGING_LEVEL=DEBUG` on both servers, then inspect `p_send_kv_done`,
`d_kv_ready` and `d_first_token` events.

## Local evidence and remaining blockers

```bash
VLLM_PLUGINS=__disabled__ PYTHONPATH="$PWD" python -m pytest -q \
  tests/runtime_patch/test_compressed_tensors_w4a8.py \
  tests/runtime_patch/test_quant_gemm_aiter.py -k 'compressed or prequantized'
```

43 targeted CPU tests pass, including native TP=1/2/4/8 loading on every rank,
production main/MTP loader bodies with the real MoERunner hierarchy, owner
ignore rules, MTP constructors/aliases, gemm2 regroup and alternating P/D token counts without
repacking or mutating cached AITER configs. An earlier wider two-file run passed 307 tests
and failed the same 15 IDs as unchanged upstream (before adding the TP8 case).
The actual checkpoint config selects layers 3/60/77 and MTP 78 through native
matching; this config-only probe stubs method construction.

On one BW1101 HCU, the actual TP8 rank shape E=256, hidden=6144, intermediate=256,
top-k=8 passed adapter post-load and BF16 AITER forwards at M=1/32/512. The M=512
config differs from M=1/32. Relative L2 differences against the same quantized
weights through BoltOps W8A8 were 0.00627/0.00670/0.00643. These are kernel
comparisons, not model-accuracy results. The gemm2 conversion is also compared
byte-for-byte with #308 at representative w13/w2 TP8 shapes.

The host has a DAS v0.25.1 wheel. Full model import needs the target
`V32SparseAttnIndexer`; Mooncake contract tests stop at missing `KpoolTailSpec`.
Explicit Triton integration also encounters a pre-existing BoltOps API mismatch.
No successful full-model/MTP/two-node PD result is claimed. Keep the MR draft
until the target-image PD and accuracy checks above complete.
