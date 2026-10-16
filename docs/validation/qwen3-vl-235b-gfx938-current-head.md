# Qwen3-VL-235B gfx938 current-head validation

This record covers `/llm-models-2/hygon/Qwen3-VL-235B-A22B-Instruct-Channel-FP8`
on four gfx938 cards at PR #188 plugin commit
`62f3052ee76783f6924f2d180c83424bd9eaec0e`. The installed runtime reported
vLLM `0.28.1+das.77acaf6.dtk2604`.

## Server command

```bash
env -u VLLM_PLUGINS \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve \
  /llm-models-2/hygon/Qwen3-VL-235B-A22B-Instruct-Channel-FP8 \
  --served-model-name Qwen3-VL-235B-A22B-Instruct-Channel-FP8 \
  --port 10234 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm
```

All four ranks constructed `HcuGPUModelRunnerV2`. FLASH_ATTN retained a
64-token kernel page and the cache manager used physical LBHNC. Public E4M3
KV allocated 1,904,000 tokens. AITER loaded its channel-shuffle table and the
gfx938 FP8 stage1/stage2 modules without a MoE provider fallback. The default
target FULL plus PIECEWISE Graph policy captured successfully.

## HumanEval client command

```bash
env -i \
  HOME=/models/.eval-home-qwen3-vl-235b-secure-run2 \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3-VL-235B-A22B-Instruct-Channel-FP8 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config '{"temperature":0,"do_sample":false,"max_tokens":2048}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --dataset-dir /models/eval-datasets/humaneval-local \
  --work-dir /models/eval-results/qwen3-vl-235b-currenthead-tp4-kvfp8-secure-run2-20261007 \
  --no-timestamp
```

The client used a dedicated local HumanEval cache, so it needed neither
online dataset access nor ModelScope credential discovery. The fresh HOME
remained empty. Raw Accuracy and Pass@1 passed 16/16. The report measured
24.22 output tok/s, 344.2 ms mean TTFT, 36.2 ms mean TPOT, and 2.157 s mean
latency.

Two identical 2,441-token probes returned `17`; the second added 2,432
prefix-hit tokens. All 18 chat requests returned HTTP 200. The server log had
no ERROR, Traceback, RuntimeError, VM fault, or dead-engine marker. Exact PTY
teardown closed port 10234, left no matching vLLM process, and returned owned
cards 0--3 to 4 MiB while unrelated work continued on cards 4--7.

## Evidence

- `/models/validation-logs/qwen3-vl-235b-currenthead-tp4-kvfp8-secure-rerun.log`
- `/models/eval-results/qwen3-vl-235b-currenthead-tp4-kvfp8-secure-run2-20261007/evalscope.log`
- `/models/eval-results/qwen3-vl-235b-currenthead-tp4-kvfp8-secure-run2-20261007/reports/Qwen3-VL-235B-A22B-Instruct-Channel-FP8/humaneval.json`
- `/models/eval-results/qwen3-vl-235b-currenthead-tp4-kvfp8-secure-run2-20261007/isolation-transcript.txt`
- `/models/eval-results/qwen3-vl-235b-currenthead-tp4-kvfp8-secure-run2-20261007/prefix-probe-transcript.json`
- `/models/eval-results/qwen3-vl-235b-currenthead-tp4-kvfp8-secure-run2-20261007/teardown-transcript.txt`
