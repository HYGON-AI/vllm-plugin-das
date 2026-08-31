# HYV4 reasoning and tool parsers on vLLM 0.25.1

The HCU plugin owns and lazily registers both parsers as `hy_v4`; no files
under the installed `vllm` package need to be replaced. The tool parser also
provides the HYV4 structural-tag grammar that is absent from vLLM 0.25.1 so
the checkpoint emits its native suffixed tags before they are converted to
OpenAI-compatible `tool_calls`.

Start the TP8 Model Runner V2 server with CUDA graphs and AITER MoE:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/models/zb/hy4
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_CACHE_ROOT=/tmp/vllm-cache-hyv4
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300
export VLLM_ENGINE_ITERATION_TIMEOUT_S=60

python -m vllm.entrypoints.openai.api_server \
  --model /models/Hy4-preview-Testing-Channel-FP8-w8a8-v2 \
  --served-model-name hy4 \
  --tensor-parallel-size 8 \
  --no-enable-expert-parallel \
  --moe-backend aiter \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v4 \
  --port 10138
```

Use top-level `reasoning_effort` (`high` or `low`) on an individual OpenAI chat
request. To select the checkpoint-native `no_think` mode explicitly, pass
`"chat_template_kwargs":{"reasoning_effort":"no_think"}` instead. Tool
requests support `auto`, `required`, and a named function choice, in streaming
and non-streaming mode. The generic OpenAI values `medium` and `none` are not
native values accepted by this checkpoint's chat template; use `high` and
`no_think`, respectively.
Keep `VLLM_ENFORCE_STRICT_TOOL_CALLING` enabled (its vLLM 0.25.1 default),
because the checkpoint's native HYV4 tool syntax depends on constrained
decoding for reliable structural tags.
