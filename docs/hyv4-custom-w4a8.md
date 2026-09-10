# HYV4 custom W4A8 on HCU

This adapter loads `hy4-w4a8-custom-v1` checkpoints through the HCU runtime plugin and existing aiter signed INT4-weight / dynamic INT8-activation kernels. Enable it explicitly through the `slimquant_w4a8` registry entry and `checkpoint_format` in the HF quantization override. The normal SlimQuant path retains its existing behavior.

## Launch

The local validation environment has vLLM `0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a`, aiter `0.1.6+dtk2604.torch2110.2609082106.g31fdae`, and eight 64 GiB Hygon accelerators. Install/build the plugin against the matching environment. For this checkout, the existing matching `vllm_hcu/hcu_ops` binary was copied from the installed plugin; no installed vLLM or aiter source was edited.

```bash
bash tools/hy_v4/serve_custom_w4a8.sh
```

Use `--enforce-eager` only when explicitly testing the eager baseline. The script uses `/data/models/hy4_w4a8_unverified`, TP8, aiter, CUDA Graphs, one sequence and a 1024-token context. It binds the OpenAI-compatible API to `127.0.0.1:8000`, with model name `hy4-w4a8`. It adds this checkout to `PYTHONPATH`; keep all HCU plugin entry points enabled (`VLLM_PLUGINS` unset, or `hcu,hcu_model,hcu_ops`). The model files are read-only inputs; HF overrides supply the quantization declaration missing from their `config.json`.

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy4-w4a8","messages":[{"role":"user","content":"你好，请用一句话介绍自己。"}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"reasoning_effort":"no_think"}}'
```

## Format and execution

- `conversion.json` identifies the quantized modules. Only those modules receive INT4 parameters. BF16 attention, dense layers, embeddings and the unquantized MTP layer retain their declared precision.
- Custom `.int4_packed` names map to the model's existing weight loaders; `.scale` maps to channel scales. Expert scales gain a trailing singleton dimension so the HYV4 fused gate/up loader splits the correct axis.
- Checkpoint bytes have the even input element in the low nibble. aiter's contiguous channelwise W4A8 kernel expects the even input element in the high nibble. The adapter swaps nibbles after tensor-parallel loading and processes one local expert at a time. Signed two's-complement values and scales are preserved.
- The RTN checkpoint's `input_scale` tensors are identity smoothing vectors. Every loaded vector is checked; nonidentity vectors are rejected because this adapter does not implement their smoothing semantics. Runtime activation scales come from aiter's per-token INT8 quantizer.
- Routed experts call `aiter.ops.triton.fused_moe.fused_experts_impl` with W4A8 channel quantization and the model's `swiglu_limit`. Quantized shared Linear layers call aiter's W4A8 GEMM through the single-expert interface, with no additional activation or routing multiplier.
- The existing vLLM MoE runner owns expert selection, shared-expert combination, the routed scaling factor, and TP reductions. The adapter does not apply these a second time.

## Validation

```bash
python -m pytest tests/models/hy_v4/test_custom_w4a8.py tests/accuracy/test_hyv4_w4a8_kernels.py -q
python -m pytest tests/models/hy_v4 tests/hy_v4 -q
PYTHONPATH="$PWD" python -m vllm_hcu.doctor
```

GPU tests compare aiter against explicit signed INT4 integer matrices and per-token INT8 reference calculations, including HYV4 shared-Linear dimensions and routed SwiGLU clipping. They test runtime arithmetic, not the quality loss from converting the original checkpoint. The supplied model's conversion manifest says `NOT_EVALUATED` and records uncalibrated RTN conversion.

The checkpoint chat template uses `reasoning_effort: no_think` to request a direct answer; it does not use `enable_thinking`. Local API clients must bypass this environment's HTTP proxy (`curl --noproxy '*'` or a proxy-free HTTP client).

HYV4's FP8 indexer cache also needs two gfx936 runtime adaptations in this branch: select cache readers/writers by the cache storage dtype, and use aiter's FP8 prefill logits implementation on gfx936. GPU checks cover FP8 cache roundtrip plus prefill/decode logits for 8 and 129 keys. These are HCU-owned runtime changes.

## MTP on this gfx936 environment

The BF16 MTP expert geometry has no matching aiter ASM W16A16 solution in the installed build. Select Triton for the unquantized MTP layer:

```bash
bash tools/hy_v4/serve_custom_w4a8.sh \
  --moe-backend triton \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

`HYV4W4A8MoEMethod` explicitly calls aiter's W4A8 kernel, so this backend argument changes the unquantized MTP implementation; the target's quantized experts and shared Linear layers continue to use aiter W4A8.

## Local run results (2026-09-10)

208 affected tests passed: 140 HYV4, 6 W4A8 numerical, 58 indexer regression, and 4 indexer GPU tests. Plugin doctor passed. Baseline TP8 loaded 52.53 GiB of model weights per device and served both completion and chat requests.

The MTP configuration above also served both requests. The direct Chinese chat returned `你好！我是混元，是由腾讯开发的大模型。` (13 tokens), matching baseline text. The 32-token completion was coherent and had finite log probabilities, but differed from baseline tokens; exact MTP equivalence is **not established**. Smoke timings include differing warmup conditions and are not a performance benchmark. The checkpoint still requires model-quality evaluation.

Machine-readable run results and logs are in `/tmp/hy4-validation/summary.json` and `/tmp/hy4-validation/`.

## CUDA Graph validation

The launch script defaults to `enforce_eager=False`. On this V2 runner, the existing breakable-CUDA-Graph configuration sets `compilation_config.mode=NONE` and `cudagraph_mode=FULL_AND_PIECEWISE`; this run tests Graph capture/replay, not Inductor compilation. With TP8 and MTP 3 tokens, prefill and decode Graph capture completed in 20 seconds and used 0.40 GiB per device.

The Graph-enabled completion matched all 32 baseline tokens, and the Chinese chat matched baseline text. Three further completion requests also matched baseline tokens (about 2.71–2.72 seconds each). These short requests are smoke tests, not throughput or quality benchmarks. The earlier eager MTP difference remains recorded above and should not be interpreted as broad equivalence across execution modes.

A 207-token prompt also exercised chunked prefill with the 128-token batch limit and returned a correct answer (32 generated tokens, finite log probabilities).

The plugin contract suite (`python tools/run_patch_tests.py --suite contract -- -q`) also passed: 1317 tests, exit code 0.

W4A8 kernel tests live in `tests/accuracy/test_hyv4_w4a8_kernels.py`. The CPU nibble reference is unmarked and runs in the contract suite; the five GPU numerical cases carry `hcu` and run in `accuracy-hcu` and nightly. Hardware availability is checked in a fixture at execution time.
