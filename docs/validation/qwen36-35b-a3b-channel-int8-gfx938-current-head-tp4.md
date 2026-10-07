# Qwen3.6 35B-A3B Channel-INT8 gfx938 current-head TP4 validation

## Result

`/llm-models-2/hygon/Qwen3.6-35B-A3B-Channel-INT8-w8a8` passed the
four-card gfx938 gate at plugin commit
`5717e40594f126d4332d99186bafa28cb772bde7` with the pinned
`vllm==0.28.1+dtk2604.torch2110.2609171627.g77acaf` wheel.

- Raw HumanEval Accuracy and Pass@1: 16/16.
- Mean output throughput: 13.07 tokens/s; mean latency: 11.712 seconds;
  mean TTFT: 9,790.3 ms; mean TPOT: 12.8 ms.
- All four ranks constructed `HcuGPUModelRunnerV2`. The route used TP4,
  HND-facing/LBHNC physical FLASH_ATTN, public `fp8_e4m3` KV, MTP3, the
  complete Qwen fine-grained prefix trio, and the default
  `FULL_AND_PIECEWISE` graph policy for target and speculator paths.
- AITER loaded the gfx938 ordinary and bottom-layer
  `E=256,N=128,dtype=int8_w8a8` configurations plus
  `tuned_fmoe_asm_w8a8_channel_shuffle.csv`. The compressed-tensors startup
  line named the Triton INT8 frontend object; both HCU custom-op switches were
  unset, retaining the plugin-default dense W8A8 redirect.
- The 64-token attention page resolved to a 1,088-token hybrid manager page.
  Fresh owner, junction, and sibling prompts returned `17`; their hit deltas
  were 0, 0, and 2,112 tokens. The sibling hit is 33 x 64 and is not divisible
  by 1,088, directly proving fine-grained junction reuse.
- Complete-session MTP counters were 1,830 accepted of 1,926 drafted tokens
  (95.02%). All 19 inference requests ended with `stop`; none ended with
  `length`, `abort`, `error`, or `repetition`.
- The server log contained no traceback, runtime error, dead-engine marker,
  VM fault, or segmentation fault. Exact PTY teardown closed port 10235,
  left no matching Python process, and returned cards 0--3 to 4 MiB.

No runtime-code change is required for this TP4 route.

## Service command

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve \
  /llm-models-2/hygon/Qwen3.6-35B-A3B-Channel-INT8-w8a8 \
  --served-model-name Qwen3.6-35B-A3B-Channel-INT8-w8a8 \
  --port 10235 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --gpu-memory-utilization 0.50 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

## HumanEval client command

HumanEval executes generated code. Keep the evaluator in the declared
isolated boundary and pass the API key through the protected environment.
The repository-local `evalscope_secure_cli` module is only a credential
transport guard: after rejecting API keys from the OS command line, it calls
the installed official `evalscope.cli.cli.run_cmd` entry point unchanged. The
generated task config and log record EvalScope 1.11.0 with the Native backend;
the wrapper does not implement a scorer.

```bash
env -i \
  HOME=/models/.eval-home-qwen36-35b-int8-tp4-20261008 \
  PATH="$PATH" \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3.6-35B-A3B-Channel-INT8-w8a8 \
  --api-url http://127.0.0.1:10235/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream \
  --eval-batch-size 8 \
  --timeout 7200 \
  --limit 16 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{}}' \
  --dataset-dir /models/eval-datasets/humaneval-local \
  --work-dir \
    /models/eval-results/qwen36-35b-int8-currenthead-tp4-mtp3-kvfp8-run1-20261008 \
  --no-timestamp
```

## Evidence

- `/models/validation-logs/qwen36-35b-int8-currenthead-tp4-mtp3-kvfp8.log`
- `/models/eval-results/qwen36-35b-int8-currenthead-tp4-mtp3-kvfp8-run1-20261008/evalscope.log`
- `/models/eval-results/qwen36-35b-int8-currenthead-tp4-mtp3-kvfp8-run1-20261008/reports/Qwen3.6-35B-A3B-Channel-INT8-w8a8/humaneval.json`
- `/models/eval-results/qwen36-35b-int8-currenthead-tp4-mtp3-kvfp8-run1-20261008/prefix-probe-transcript.json`
- `/models/eval-results/qwen36-35b-int8-currenthead-tp4-mtp3-kvfp8-run1-20261008/final-metrics.txt`
- `/models/eval-results/qwen36-35b-int8-currenthead-tp4-mtp3-kvfp8-run1-20261008/isolation-transcript.txt`
- `/models/eval-results/qwen36-35b-int8-currenthead-tp4-mtp3-kvfp8-run1-20261008/teardown-transcript.txt`
