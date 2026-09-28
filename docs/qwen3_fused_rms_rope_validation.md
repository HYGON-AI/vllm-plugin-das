# Qwen3 fused RMSNorm and rotary embedding validation

## Runtime policy

The Qwen3 text-attention patch uses the public
`lightop.attention.rms_rotary_embedding_fuse` API. It is enabled by default
when `VLLM_HCU_USE_CUSTOM_OPS` is enabled. Set
`VLLM_HCU_USE_FUSED_RMS_ROPE=0` to restore the original vLLM Qwen3 attention
method. Unsupported inputs, including M-RoPE positions and dual-chunk
attention, delegate to the original method before QKV projection.

## Framework-aligned HCU operator check

The reference follows the current HCU framework path: categorized LightOp
RMSNorm for Q and K, followed by `vllm._custom_ops.rotary_embedding`. The
following command was run on 2026-09-29:

```bash
pytest -q -s \
  tests/accuracy/test_lightop_qwen3_fused_rms_rope_accuracy.py \
  --disable-warnings
```

| Tokens | Current path (ms) | Fused path (ms) | Speedup | Max Q/K error |
|---:|---:|---:|---:|---:|
| 1 | 0.052862 | 0.039187 | 1.349x | 0.00000 |
| 8 | 0.052994 | 0.038876 | 1.363x | 0.03125 |
| 128 | 0.053786 | 0.039923 | 1.347x | 0.03125 |
| 512 | 0.091780 | 0.040657 | 2.257x | 0.06250 |

Mean absolute error remained below `0.000653`; result: `4 passed`.

The custom op is registered as an in-place mutation with no tensor return.
This matches the vendor function, whose returned Q and K are the input
tensors themselves, and avoids PyTorch's deprecated mutable-input alias
return contract.

## Qwen3-8B service check

The exact candidate wheel was
`vllm_hcu-0.28.1rc1.dev491+das.3b7eb3b.dtk26041-cp310-cp310-linux_x86_64.whl`.
The unchanged installed DTK native extension was packaged with the candidate
Python sources because this change contains no native extension edits.

Recommended service command, with fused RMSNorm plus RoPE enabled by default:

```bash
export HIP_VISIBLE_DEVICES=0

vllm serve /model/Qwen3-8B \
  -tp 1 \
  --dtype bfloat16 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e5m2 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9
```

The baseline used the identical command and wheel with this one additional
opt-out:

```bash
export VLLM_HCU_USE_FUSED_RMS_ROPE=0
```

The validation run appended `--host 127.0.0.1 --port 8002`. Its local probes
explicitly bypassed configured proxies:

```bash
curl --noproxy '*' http://127.0.0.1:8002/health
curl --noproxy '*' http://127.0.0.1:8002/v1/models
```

Both baseline and candidate completed FULL and PIECEWISE graph capture,
returned HTTP 200 from `/health`, listed the model through `/v1/models`, and
passed the official tests for HumanEval tasks 0 through 7 (`8/8`, pass@1
`1.0`). The server log confirms `fp8_e5m2` KV-cache storage.

The exact request, scoring, and token-throughput calculation is committed as
`tools/qwen3_humaneval8.py`. The validation used OpenAI HumanEval commit
`6d43fb980f9fee3c892a914eda09951f772ad10d` and these commands for each
server arm:

```bash
git clone https://github.com/openai/human-eval.git /tmp/human-eval
git -C /tmp/human-eval checkout 6d43fb980f9fee3c892a914eda09951f772ad10d

NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
http_proxy= https_proxy= all_proxy= \
python3 tools/qwen3_humaneval8.py \
  --human-eval-root /tmp/human-eval \
  --base-url http://127.0.0.1:8002 \
  --model /model/Qwen3-8B \
  --output-dir /tmp/qwen3-fused-rms-rope-candidate
```

For the baseline, start the server with
`VLLM_HCU_USE_FUSED_RMS_ROPE=0` and use a fresh
`--output-dir /tmp/qwen3-fused-rms-rope-baseline`. The tool selects ordered
tasks 0 through 7, sends one request at a time with temperature 0,
`max_tokens=2048`, and `enable_thinking=false`, executes the official tests,
and reports generated tokens divided by summed request wall time.

The sequential eight-request sample measured 63.637 generated tokens/s for
the opt-out baseline and 62.710 generated tokens/s for the candidate. The
generated lengths differed (1082 versus 1085 tokens), so this small service
delta does not demonstrate an end-to-end throughput improvement. The
repeatable performance evidence for this change is the framework-aligned
operator table above.

After the alias-contract correction, the candidate completed compilation,
FULL/PIECEWISE graph capture, and HumanEval without the PyTorch mutable-input
alias warning.
