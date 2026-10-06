# `/llm-models-2/hygon` vLLM 0.28.1 validation

This note records the additional gfx938 validation performed against the
checkpoints under `/llm-models-2/hygon` on top of PR #183. The runtime was
`vllm==0.28.1+dtk2604.torch2110.2609171627.g77acaf` with the plugin imported
from this branch. Every accepted feature route used Model Runner V2, prefix
caching, the public `fp8_e4m3` KV dtype, and the default
`FULL_AND_PIECEWISE` graph policy; explicitly labeled auto-KV controls were
retained only as accuracy diagnostics. DeepSeek V4.1 remained excluded by
request.

## Results

| Checkpoint | Topology | HumanEval | Additional evidence |
| --- | --- | ---: | --- |
| `Qwen3-0.6B-Channel-FP8` | TP2 | auto KV 12/16 twice; E4M3 8/16 and 9/16 | Service passed; checkpoint is accuracy-sensitive at this size |
| `Qwen3-4B-Channel-FP8` | TP2 | 16/16 | E4M3 KV |
| `Qwen3-4B-Channel-INT8-w8a8` | TP2 | 15/16 | E4M3 KV |
| `Qwen3-8B-Channel-FP8` | TP2 | 16/16 | E4M3 KV |
| `Qwen3-8B-Channel-INT8-w8a8` | TP2 | 16/16 | E4M3 KV |
| `Qwen3-14B-Channel-INT8-w8a8` | TP2 | 15/16 | E4M3 KV |
| `Qwen3-4B-Thinking-2507-Channel-FP8` | TP2 | diagnostic only | The checkpoint template always emits a thinking segment; disabling thinking is not a valid accuracy contract |
| `Qwen3.5-27B-Channel-FP8` | TP2 | 16/16 | MTP3; fine-grained third-request prefix hit 2,240 tokens |
| `Qwen3.5-35B-A3B-Channel-FP8-w8a8` | TP2 | 15/16, then 16/16 | MTP3; the single HumanEval/10 miss did not reproduce; fine-grained hit 2,240 tokens |
| `Qwen3.5-35B-A3B-Channel-INT8-w8a8` | TP4 resource-control run | 16/16 | MTP acceptance 1,274/1,341 (95.0%); third-request fine-grained hit 2,240 tokens |
| `Qwen3.6-35B-A3B-Channel-FP8-w8a8` | TP4 resource-control run | 16/16 | MTP acceptance 1,861/1,959 (95.0%); third-request fine-grained hit 2,112 tokens |
| `Qwen3.6-35B-A3B-Channel-INT8-w8a8` | TP2 | 16/16 | 69.34 output tok/s; MTP acceptance 1,817/1,902 (95.5%); third-request fine-grained hit 2,112 tokens |
| `Qwen3.8-27B-Channel-FP8` | TP2 | 16/16 | 61.10 output tok/s; MTP acceptance 1,717/1,785 (96.2%); 6,016-token probe reused 4,736 tokens on the consumer request with correct output |
| `Qwen3.8-Flash-Next-Channel-FP8` | TP4 | 16/16 | BLNHC hybrid cache; 25.69 output tok/s; MTP acceptance 1,771/1,974 (89.7%); align-only probe reused 3,200 tokens |

The TP4 resource-control runs were used only while unrelated stale KFD
contexts constrained available memory. TP remained enabled, and later runs
returned to TP2 after complete process-group cleanup released the stale
contexts.

The Flash-Next run selected QSA successfully, but the AITER MoE backend did
not have tuned entries for every `E=512, N=160, K=2560` shape and logged
shape-local fallback to the official Triton MoE implementation. The result is
therefore end-to-end route evidence, not a claim that every MoE layer used an
AITER kernel.

## Qwen3.5 and Qwen3.6 hybrid service command

Use the following command for the 35B-A3B hybrid checkpoints. Replace
`MODEL`, `SERVED`, `GPU_LIST`, `TP`, and `GPU_MEMORY_UTILIZATION` with the
values in the result table. The accepted TP2 runs used `GPU_LIST=0,1`,
`TP=2`, and `GPU_MEMORY_UTILIZATION=0.50`; the temporary TP4 resource-control
runs used `GPU_LIST=0,1,2,3`, `TP=4`, and
`GPU_MEMORY_UTILIZATION=0.10`.

```bash
env -u VLLM_PLUGINS \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES="$GPU_LIST" \
  vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --port 10234 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size "$TP" \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-cache-mode align \
  --prefix-match-unit 64 \
  --enable-mamba-fine-grained-prefix-cache \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

For dense `Qwen3.5-27B-Channel-FP8` and
`Qwen3.8-27B-Channel-FP8`, use the same command without
`--moe-backend aiter`. The Qwen3.8 run used `--max-model-len 8192` and
`--max-num-batched-tokens 2048` so that the 6,016-token state-reuse probe fit.

The three hybrid prefix options are intentional. `align` keeps attention and
GDN cache ownership aligned. `prefix-match-unit=64` and
`enable-mamba-fine-grained-prefix-cache` allow reusable junctions below the
large physical hybrid manager page. A valid probe uses three requests with
the same long prefix and different suffixes: the first owns the page, the
second materializes the junction, and the third consumes it.

## Qwen3.8 Flash-Next service command

Qwen4Exp QSA does not support the HND/LBHNC family. Do not export
`VLLM_KV_CACHE_LAYOUT=HND`; allow vLLM to resolve the common supported layout,
which was `BLNHC` in this run. This profile used align-only caching because
its correctness contract is the 1,600-token hybrid manager page.

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve /llm-models-2/hygon/Qwen3.8-Flash-Next-Channel-FP8 \
  --served-model-name Qwen3.8-Flash-Next-Channel-FP8 \
  --port 10234 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN \
  --moe-backend aiter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-cache-mode align \
  --gpu-memory-utilization 0.60 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The initial diagnostic attempt deliberately retained the ordinary
FlashAttention `HND` setting and failed before serving with:

```text
VLLM_KV_CACHE_LAYOUT=HND does not satisfy every supported set;
valid layouts: ['BLNHC', 'BLHNC']
```

Unsetting the forced layout started the service successfully and selected
`BLNHC`. No plugin runtime change was needed.

## HumanEval client command

Each score above used a fresh work directory and proxy-free loopback access.
Set `SERVED` and `WORK_DIR` for the active service.

```bash
env -i \
  HOME=/tmp/vllm-hcu-eval-home \
  PATH="$PATH" \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model "$SERVED" \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream \
  --eval-batch-size 8 \
  --timeout 7200 \
  --limit 16 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{}}' \
  --work-dir "$WORK_DIR" \
  --no-timestamp
```

All services were started in isolated process groups. Teardown first sent
TERM to the complete PGID, verified every process in that PGID, and escalated
only the Flash-Next PGID after its workers exceeded the grace period. The
final device state was 2 MiB used on every card.
