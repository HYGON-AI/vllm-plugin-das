# Qwen3.8 27B Channel-FP8 gfx938 current-head TP4 validation

## Result

`/llm-models-2/hygon/Qwen3.8-27B-Channel-FP8` passed the four-card gfx938
gate at plugin commit `1f1c5172314498b009743a7afeb5086112c43db7` with the
pinned `vllm==0.28.1+dtk2604.torch2110.2609171627.g77acaf` wheel.

- Raw HumanEval Accuracy and Pass@1: 16/16.
- Mean output throughput: 80.25 tokens/s; mean latency: 1.797 seconds; mean
  TTFT: 415.1 ms; mean TPOT: 9.9 ms.
- All four ranks constructed `HcuGPUModelRunnerV2`. The route used TP4,
  channel-wise FP8 dense weights, HND-facing/LBHNC physical FLASH_ATTN,
  public `fp8_e4m3` KV, MTP3, the complete Qwen fine-grained prefix trio,
  and the default `FULL_AND_PIECEWISE` graph policy for target and speculator
  paths. Startup selected `ChannelWiseTorchFP8ScaledMMLinearKernel`.
- The 64-token attention page resolved to a 1,600-token hybrid manager page.
  Fresh owner, junction, and sibling prompts returned `17`; their hit deltas
  were 0, 0, and 1,536 tokens. The sibling hit is 24 x 64 and is not divisible
  by 1,600, directly proving fine-grained junction reuse.
- Complete-session MTP counters were 1,732 accepted of 1,794 drafted tokens
  (96.54%). All 19 inference requests ended with `stop`; none ended with
  `length`, `abort`, `error`, or `repetition`.
- The server log contained no traceback, runtime error, dead-engine marker,
  VM fault, or segmentation fault. Exact PTY teardown closed port 10236,
  left no matching Python process, and returned cards 0--3 to 2 MiB.

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
  vllm serve /llm-models-2/hygon/Qwen3.8-27B-Channel-FP8 \
  --served-model-name Qwen3.8-27B-Channel-FP8 \
  --port 10236 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --gpu-memory-utilization 0.50 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

## Official EvalScope HumanEval command

HumanEval executes generated code. Keep the evaluator in the declared
isolated boundary and pass the API key through the protected environment.
The repository-local `evalscope_secure_cli` module is only a credential
transport guard: it directly calls the installed official
`evalscope.cli.cli.run_cmd` entry point and does not implement a scorer. The
generated task config and log record EvalScope 1.11.0 with the Native backend.

```bash
env -i \
  HOME=/models/.eval-home-qwen38-27b-fp8-tp4-20261008 \
  PATH="$PATH" \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3.8-27B-Channel-FP8 \
  --api-url http://127.0.0.1:10236/v1 \
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
    /models/eval-results/qwen38-27b-channel-fp8-currenthead-tp4-mtp3-kvfp8-run1-20261008 \
  --no-timestamp
```

## Evidence

- `/models/validation-logs/qwen38-27b-channel-fp8-currenthead-tp4-mtp3-kvfp8.log`
- `/models/eval-results/qwen38-27b-channel-fp8-currenthead-tp4-mtp3-kvfp8-run1-20261008/evalscope.log`
- `/models/eval-results/qwen38-27b-channel-fp8-currenthead-tp4-mtp3-kvfp8-run1-20261008/reports/Qwen3.8-27B-Channel-FP8/humaneval.json`
- `/models/eval-results/qwen38-27b-channel-fp8-currenthead-tp4-mtp3-kvfp8-run1-20261008/prefix-probe-transcript.json`
- `/models/eval-results/qwen38-27b-channel-fp8-currenthead-tp4-mtp3-kvfp8-run1-20261008/final-metrics.txt`
- `/models/eval-results/qwen38-27b-channel-fp8-currenthead-tp4-mtp3-kvfp8-run1-20261008/isolation-transcript.txt`
- `/models/eval-results/qwen38-27b-channel-fp8-currenthead-tp4-mtp3-kvfp8-run1-20261008/teardown-transcript.txt`
