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
| `Qwen3-4B-Channel-INT8-w8a8` | TP2 | 15/16 initially; current-head repeat 16/16 | E4M3 KV; repeated prompt reused 2,368 tokens |
| `Qwen3-8B-Channel-FP8` | TP2 | 16/16 | E4M3 KV |
| `Qwen3-8B-Channel-INT8-w8a8` | TP2 | 16/16 | E4M3 KV |
| `Qwen3-14B-Channel-INT8-w8a8` | TP2 | 15/16 twice | Current-head repeat reproduced the same checkpoint-level HumanEval/10 logic error; E4M3 KV and prefix reuse passed |
| `Qwen3-4B-Thinking-2507-Channel-FP8` | TP2 | diagnostic only | The checkpoint template always emits a thinking segment; disabling thinking is not a valid accuracy contract |
| `Qwen3-VL-2B-Instruct-Channel-FP8` | TP2 | 14/16 twice | 188.04 and 198.51 output tok/s; repeated batch reused 1,600 prefix tokens |
| `Qwen3-VL-4B-Instruct-Channel-FP8` | TP2 | 15/16 twice | 127.57 and 137.41 output tok/s; repeated batch reused 1,600 prefix tokens |
| `Qwen3-VL-8B-Instruct-Channel-FP8` | TP2 | 15/16 twice | 117.39 and 123.67 output tok/s; repeated batch reused 1,600 prefix tokens |
| `Qwen3-VL-2B-Thinking-Channel-FP8` | TP2 | 2/16 at 7,800 output tokens | `qwen3` reasoning parser; both normally stopped answers passed and 14 answers exhausted the 8K context; duplicate long prompt reused 832 tokens |
| `Qwen3-VL-4B-Thinking-Channel-FP8` | TP2 | 6/16 at 7,800 output tokens | `qwen3` reasoning parser; all six normally stopped answers passed and ten answers exhausted the 8K context; duplicate long prompt reused 832 tokens |
| `Qwen3-VL-8B-Thinking-Channel-FP8` | TP2 | 9/16 at 2,048; 11/16 at 3,800; 15/16 at 7,800 output tokens | `qwen3` reasoning parser; every normally stopped answer passed; the final miss was a checkpoint reasoning loop on HumanEval/1; duplicate long prompt reused 960 tokens |
| `Qwen3-VL-235B-A22B-Instruct-Channel-FP8` | TP4 | 16/16 twice | AITER channel-FP8 MoE with no logged provider fallback; 18.60 and 25.05 output tok/s; repeated batch reused 1,600 prefix tokens |
| `Qwen3-235B-A22B-Channel-INT8-w8a8` | TP4 and TP8 diagnostics | best 15/16; not accepted | The same corrupted identifiers survived KV, Graph, topology, dense-GEMM, and MoE-provider controls; the checkpoint has no MTP layer |
| `DeepSeek-V4-Flash-0731-Channel-INT8-w8a8` | TP8 | raw 14/16; normalized 16/16, twice | DSpark7; public E4M3-to-`fp8_ds_mla`; tuned AITER W8A8 MoE; target FULL plus PIECEWISE and DSpark FULL Graphs; this is V4-Flash-0731, not excluded V4.1 |
| `DeepSeek-V4-Flash-0731-Channel-FP8-w8a8` | TP8 | raw 13/16; normalized 16/16, twice | DSpark7; BLHNC E4M3 KV; channel-wise FP8 dense and tuned AITER FP8 MoE; target FULL plus PIECEWISE and DSpark FULL Graphs |
| `DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel` | TP8 | 16/16 raw and normalized, twice | SlimQuant W4A8 MoE; W8A8 attention; BLHNC E4M3 KV; DSpark7; target FULL plus PIECEWISE and DSpark FULL Graphs |
| `DeepSeek-V4-Flash-Channel-FP8-w8a8` | TP8 | raw 15/16; normalized 16/16, twice | no DSpark metadata; MTP3; BLHNC E4M3 KV; AITER FP8 MoE; target and MTP prefill FULL plus PIECEWISE, MTP decode FULL Graphs |
| `DeepSeek-V4-Pro-0813-Channel-INT4-w4a8` | TP8 | 16/16 | SlimQuant W4A8 MoE; W8A8 attention; BLHNC E4M3 KV; DSpark7; target FULL plus PIECEWISE and DSpark FULL Graphs; 1,726/2,303 draft tokens accepted |
| `DeepSeek-V4-Pro-0813-INT4-Channel` | TP8 | 16/16 | Independently cold-started at current MR head; SlimQuant W4A8 MoE, W8A8 attention, BLHNC E4M3 KV, DSpark7, target FULL plus PIECEWISE and DSpark FULL Graphs; 1,726/2,303 HumanEval draft tokens accepted |
| `DeepSeek-V4-Pro-0813-INT8-Channel` | TP8 capacity gate | not launched | 1,545.42 GiB total and 193.18 GiB/rank before runtime overhead exceed the available 143.98 GiB/card; requires TP16 or larger-memory devices |
| `DeepSeek-R1-W4A8-V2_6` | TP8 | 16/16 raw and normalized, twice | FLASHMLA, LBNHC E4M3 KV, AITER W4A8 MoE, MTP3, target/speculator FULL plus PIECEWISE Graphs; 30,794/52,617 session draft tokens accepted |
| `DeepSeek-V3.2-Channel-INT8-w8a8` | TP8 | 16/16 raw and normalized, twice | FLASHMLA_SPARSE, LBNHC public-E4M3-to-`fp8_ds_mla`, AITER W8A8 MoE, MTP3, target/speculator FULL plus PIECEWISE Graphs; 3,458/4,893 session draft tokens accepted |
| `GLM-5.3-Channel-FP8-w8a8` | TP8 | 16/16 | FLASHMLA_SPARSE, LBNHC public-E4M3-to-`fp8_ds_mla`, channel-wise FP8 dense and AITER FP8 MoE, MTP3, target/speculator FULL plus PIECEWISE Graphs; 875/963 draft tokens accepted |
| `GLM-5.3-Flash-Channel-INT8-w8a8` | TP4 | auto/BF16 KV 16/16 twice | NoPE `Dqk=512`; MTP3 local argmax; target/speculator FULL plus PIECEWISE Graphs; FP8 KV is blocked because the installed sparse FlashMLA FP8 kernel rejects `Dqk=512` |
| `GLM-5.3-Flash-Channel-FP8-w8a8` | TP4 | auto/BF16 KV 16/16 twice | FP8 weights; NoPE `Dqk=512`; MTP3 local argmax; target/speculator FULL plus PIECEWISE Graphs; 2,247/2,340 session draft tokens accepted |
| `MiniMax-M2.5-Channel-INT8-w8a8` | TP4 | 16/16 twice at 3,800 output tokens | E4M3 KV; AITER W8A8 MoE; target FULL plus PIECEWISE Graphs; built-in MTP is not registered for MiniMax M2 in vLLM 0.28.1 |
| `Hy3-Channel-FP8-w8a8` | TP4 | 16/16 twice | Channel-wise FP8 dense; E4M3 KV; MTP2; target/speculator FULL plus PIECEWISE Graphs; AITER config miss fell back to official Triton for the observed `M=1` MoE shape |
| `Hy3-Channel-INT8-w8a8` | TP4 | 16/16 twice | E4M3 KV; MTP2; target/speculator FULL plus PIECEWISE Graphs; AITER config miss fell back to official Triton for the observed `M=1` MoE shape |
| `Hy4-preview-Channel-FP8-w8a8` | TP8 | 16/16 | FLASHMLA_SPARSE, LBNHC E4M3 KV, tuned AITER FP8 MoE, MTP3, target/speculator FULL plus PIECEWISE Graphs; 1,549/1,689 draft tokens accepted |
| `Qwen3.5-27B-Channel-FP8` | TP2 | 16/16 | Current-head rerun: 53.32 output tok/s; MTP acceptance 1,337/1,449 (92.27%); fine-grained third-request prefix hit 3,136 tokens |
| `Qwen3.5-35B-A3B-Channel-FP8-w8a8` | TP2 | 15/16, then 16/16; current-head rerun 16/16 | Tuned AITER channel-FP8 MoE; 35.71 output tok/s; current-head MTP acceptance 1,096/1,170 (93.68%); fine-grained sibling hit 2,112 tokens |
| `Qwen3.5-35B-A3B-Channel-INT8-w8a8` | TP2 | 15/16, then 16/16 | Default LightOp dense W8A8 plus tuned AITER INT8 MoE; 62.66 output tok/s on the passing rerun; MTP acceptance 2,336/2,505 (93.25%); fine-grained sibling hit 2,112 tokens |
| `Qwen3.5-122B-A10B-Channel-FP8-w8a8` | TP4 | 16/16 | AITER channel-FP8 MoE; MTP acceptance 1,862/1,968 (94.6%); 2,176-token manager page and third-request fine-grained hit 2,112 tokens |
| `Qwen3.5-397B-A17B-Channel-FP8-w8a8` | TP8 | 13/16 raw; 16/16 normalized, twice | Correct bare function bodies were undercounted by raw EvalScope; AITER channel-FP8 MoE; final MTP acceptance 2,643/2,760 (95.8%); 1,088-token manager page and third-request fine-grained hit 2,624 tokens |
| `Qwen3.6-35B-A3B-Channel-FP8-w8a8` | TP4 resource-control run | 16/16 | MTP acceptance 1,861/1,959 (95.0%); third-request fine-grained hit 2,112 tokens |
| `Qwen3.6-35B-A3B-Channel-INT8-w8a8` | TP2 | 16/16 | 69.34 output tok/s; MTP acceptance 1,817/1,902 (95.5%); third-request fine-grained hit 2,112 tokens |
| `Qwen3.8-27B-Channel-FP8` | TP2 | 16/16 | 61.10 output tok/s; MTP acceptance 1,717/1,785 (96.2%); 6,016-token probe reused 4,736 tokens on the consumer request with correct output |
| `Qwen3.8-Flash-Next-Channel-FP8` | TP4 | 16/16 | BLNHC hybrid cache; 25.69 output tok/s; MTP acceptance 1,771/1,974 (89.7%); align-only probe reused 3,200 tokens |

The TP4 resource-control runs were used only while unrelated stale KFD
contexts constrained available memory. TP remained enabled, and later runs
returned to TP2 after complete process-group cleanup released the stale
contexts.

The Qwen3-VL Instruct runs were completed at the pre-integration branch head
`dd65c88`. After the integration branch was rebased, the 8B Thinking profile
and its 8K budget gate were run again from the new remote head `040f351` on a
clean follow-up branch. Both heads used the same pinned vLLM wheel.

The Flash-Next run selected QSA successfully, but the AITER MoE backend did
not have tuned entries for every `E=512, N=160, K=2560` shape and logged
shape-local fallback to the official Triton MoE implementation. The result is
therefore end-to-end route evidence, not a claim that every MoE layer used an
AITER kernel.

## Qwen3-14B Channel-INT8 current-head repeat

The non-reasoning 14B checkpoint was independently cold-started again at
plugin commit `20dc8cc`. The exact server command was:

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve /llm-models-2/hygon/Qwen3-14B-Channel-INT8-w8a8 \
  --served-model-name Qwen3-14B-Channel-INT8-w8a8 \
  --port 10244 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The exact isolated HumanEval client command was:

```bash
env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-qwen3-14b \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3-14B-Channel-INT8-w8a8 \
  --api-url http://127.0.0.1:10244/v1 \
  --eval-type openai_api \
  --generation-config '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir /tmp/vllm-hcu-evalscope/qwen3-14b-channel-int8-current-tp2-kvfp8-run2-20261007 \
  --no-timestamp
```

Both TP ranks constructed `HcuGPUModelRunnerV2`; HND resolved to physical
LBHNC, and the default `FULL_AND_PIECEWISE` policy captured both FULL and
PIECEWISE Graphs. The public E4M3 cache used 64-token pages. The startup
message naming `TritonInt8ScaledMMLinearKernel` is the upstream frontend
object, not proof that the executed dense GEMM bypassed the plugin: with both
custom-op switches unset, the default HCU wrapper still sends dense W8A8 to
LightOp per-token quantization and hipBLASLt GEMM.

Raw Accuracy and Pass@1 were 15/16 (93.8%), matching the earlier run. Both
runs failed only HumanEval/10 and emitted the same incorrect loop: it checks
the empty suffix first and therefore returns the unmodified string. This is a
stable checkpoint generation error, not an E4M3, Graph, TP, or dense-kernel
failure. The report observed 86.68 output tok/s, 124.2 ms mean TTFT, 10.6 ms
mean TPOT, and 1.48 s mean latency. Two identical 2,425-token prefix probes
both returned `17`; the second added 2,368 cache-hit tokens (37 x 64). All 18
chat requests returned HTTP 200 and the log contained no ERROR, Traceback, or
RuntimeError.

Evidence:

- `/tmp/vllm-hcu-validation/qwen3-14b-channel-int8-current-tp2-kvfp8-run2-20261007.log`
- `/tmp/vllm-hcu-evalscope/qwen3-14b-channel-int8-current-tp2-kvfp8-run2-20261007`

## Qwen3-4B Channel-INT8 current-head repeat

The 4B INT8 checkpoint was then cold-started at plugin commit `518dbcb` with
the same accepted dense-Qwen route. The exact server command was:

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve /llm-models-2/hygon/Qwen3-4B-Channel-INT8-w8a8 \
  --served-model-name Qwen3-4B-Channel-INT8-w8a8 \
  --port 10245 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The exact isolated HumanEval client command was:

```bash
env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-qwen3-4b-int8 \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3-4B-Channel-INT8-w8a8 \
  --api-url http://127.0.0.1:10245/v1 \
  --eval-type openai_api \
  --generation-config '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir /tmp/vllm-hcu-evalscope/qwen3-4b-channel-int8-current-tp2-kvfp8-run2-20261007 \
  --no-timestamp
```

Both TP ranks constructed `HcuGPUModelRunnerV2`, HND resolved to physical
LBHNC, public E4M3 used 64-token pages, and the default
`FULL_AND_PIECEWISE` policy captured both Graph forms. Raw Accuracy and
Pass@1 passed 16/16, replacing the earlier 15/16 observation. The report
measured 141.82 output tok/s, 101.5 ms mean TTFT, 6.3 ms mean TPOT, and
0.894 s mean latency. Two identical 2,425-token requests both returned
`8 + 9 = 17`; the second added 2,368 prefix-hit tokens. All 18 chat requests
returned HTTP 200 and the log contained no ERROR, Traceback, or RuntimeError.
No runtime-code change is needed.

Evidence:

- `/tmp/vllm-hcu-validation/qwen3-4b-channel-int8-current-tp2-kvfp8-run2-20261007.log`
- `/tmp/vllm-hcu-evalscope/qwen3-4b-channel-int8-current-tp2-kvfp8-run2-20261007`

## Qwen3.5 and Qwen3.6 hybrid service command

Use the following command for the 35B-A3B, 122B-A10B, and 397B-A17B hybrid
checkpoints. Replace
`MODEL`, `SERVED`, `GPU_LIST`, `TP`, and `GPU_MEMORY_UTILIZATION` with the
values in the result table. The accepted TP2 runs used `GPU_LIST=0,1`,
`TP=2`, and `GPU_MEMORY_UTILIZATION=0.50`; the temporary TP4 resource-control
runs used `GPU_LIST=0,1,2,3`, `TP=4`, and
`GPU_MEMORY_UTILIZATION=0.10`.

The accepted 122B-A10B run used `GPU_LIST=0,1,2,3`, `TP=4`, and
`GPU_MEMORY_UTILIZATION=0.50`. Its checkpoint contains one MTP layer; the
three-token configuration intentionally exercises the official repeated-layer
MTP behavior.

The accepted 397B-A17B run used all eight devices, `TP=8`, and
`GPU_MEMORY_UTILIZATION=0.50`. It loaded 378.93 GiB from 94 shards and used
50.38 GiB of model memory per rank. Its checkpoint also contains one MTP
layer, so the same repeated-layer MTP3 behavior applies.

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

### Qwen3.5-27B current-head rerun

The dense 27B checkpoint was rerun at plugin commit `bd9157f` with the pinned
vLLM wheel. The exact server command was:

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve /llm-models-2/hygon/Qwen3.5-27B-Channel-FP8 \
  --served-model-name Qwen3.5-27B-Channel-FP8 \
  --port 10241 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN \
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

The exact isolated HumanEval client command was:

```bash
env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-qwen35-27b \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3.5-27B-Channel-FP8 \
  --api-url http://127.0.0.1:10241/v1 \
  --eval-type openai_api \
  --generation-config '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir /tmp/vllm-hcu-evalscope/qwen35-27b-channel-fp8-current-tp2-mtp3-kvfp8-fine-run1-20261007 \
  --no-timestamp
```

Both TP ranks constructed `HcuGPUModelRunnerV2`, dense FP8 selected
`ChannelWiseTorchFP8ScaledMMLinearKernel`, the public E4M3 cache resolved to
physical LBHNC, and the 64-token attention page was managed in 1,600-token
hybrid pages. Target and MTP prefill captured PIECEWISE and FULL Graphs, and
MTP decode captured FULL Graphs under the default `FULL_AND_PIECEWISE` policy.
The service allocated 474,680 KV tokens.

Raw Accuracy and Pass@1 were 16/16. Mean output throughput was 53.32 tok/s,
TTFT 838.3 ms, TPOT 11.4 ms, and latency 2.112 s. A fresh three-suffix probe
reported per-request prefix-hit deltas of 0, 1,600, and 3,136 tokens. The last
value is 49 x 64 but is not divisible by the 1,600-token manager page, directly
proving fine-grained junction reuse. Complete-session MTP counters were
1,337/1,449 accepted drafts (92.27%). There was no `ERROR`, `Traceback`,
`RuntimeError`, or provider failure. Exact process-group teardown closed the
port and returned all eight cards to their idle reading.

Evidence:

- `/tmp/vllm-hcu-validation/qwen35-27b-channel-fp8-current-tp2-mtp3-kvfp8-fine.log`
- `/tmp/vllm-hcu-evalscope/qwen35-27b-channel-fp8-current-tp2-mtp3-kvfp8-fine-run1-20261007`

### Qwen3.5-35B-A3B Channel-FP8 current-head rerun

The MoE checkpoint was independently cold-started at plugin commit `8350f24`.
The exact server command was:

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve /llm-models-2/hygon/Qwen3.5-35B-A3B-Channel-FP8-w8a8 \
  --served-model-name Qwen3.5-35B-A3B-Channel-FP8-w8a8 \
  --port 10242 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 2 \
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

The exact isolated HumanEval client command was:

```bash
env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-qwen35-35b \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3.5-35B-A3B-Channel-FP8-w8a8 \
  --api-url http://127.0.0.1:10242/v1 \
  --eval-type openai_api \
  --generation-config '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir /tmp/vllm-hcu-evalscope/qwen35-35b-a3b-channel-fp8-current-tp2-mtp3-kvfp8-fine-run1-20261007 \
  --no-timestamp
```

Both TP ranks constructed `HcuGPUModelRunnerV2`. Dense FP8 selected
`ChannelWiseTorchFP8ScaledMMLinearKernel`; MoE selected AITER and loaded both
the ordinary and bottom-layer gfx938 `E=256,N=256,dtype=fp8_w8a8` configs plus
the channel-shuffle tuned CSV. No provider fallback was logged. The public
E4M3 cache resolved to LBHNC, with a 64-token FLASH_ATTN page and a
2,176-token hybrid manager page. Target and MTP prefill captured PIECEWISE and
FULL Graphs, MTP decode captured FULL Graphs, and the service allocated
1,104,474 KV tokens.

Raw Accuracy and Pass@1 were 16/16, so the older one-off HumanEval/10 miss did
not reproduce at the current head. Mean output throughput was 35.71 tok/s,
TTFT 1,241 ms, TPOT 16.5 ms, and latency 2.569 s. The accepted fresh
owner/junction/sibling requests were each 3,655 prompt tokens, returned `17`,
and produced hit deltas of 0/0/2,112. The sibling hit is 33 x 64 but is not
divisible by 2,176, directly proving fine-grained junction reuse.
Complete-session MTP counters were 1,096/1,170 accepted drafts (93.68%). One
initial over-context diagnostic was rejected with HTTP 400 before model
execution and was excluded; the 20 accepted inference requests returned HTTP
200. No fatal/error marker occurred. Exact PGID teardown closed port 10242 and
released the two cards used by this service; unrelated external contexts were
then visible only on cards 4-7 and were left untouched.

Evidence:

- `/tmp/vllm-hcu-validation/qwen35-35b-a3b-channel-fp8-current-tp2-mtp3-kvfp8-fine.log`
- `/tmp/vllm-hcu-evalscope/qwen35-35b-a3b-channel-fp8-current-tp2-mtp3-kvfp8-fine-run1-20261007`

### Qwen3.5-35B-A3B Channel-INT8 current-head rerun

The INT8 sibling was independently cold-started at plugin commit `6f8253e`.
The exact server command was:

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve /llm-models-2/hygon/Qwen3.5-35B-A3B-Channel-INT8-w8a8 \
  --served-model-name Qwen3.5-35B-A3B-Channel-INT8-w8a8 \
  --port 10243 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 2 \
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

The exact isolated HumanEval client command was run twice with distinct
`--work-dir` values:

```bash
env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-qwen35-35b-int8 \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Qwen3.5-35B-A3B-Channel-INT8-w8a8 \
  --api-url http://127.0.0.1:10243/v1 \
  --eval-type openai_api \
  --generation-config '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir /tmp/vllm-hcu-evalscope/qwen35-35b-a3b-channel-int8-current-tp2-mtp3-kvfp8-fine-run1-20261007 \
  --no-timestamp
```

The rerun changed only the directory suffix from `run1-20261007` to
`run2-20261007`.

The startup message `Selected TritonInt8ScaledMMLinearKernel for
CompressedTensorsW8A8Int8` describes the upstream frontend kernel object; it
does not identify the dense GEMM that executes after plugin patching. With
both `VLLM_HCU_USE_CUSTOM_OPS` and
`VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM` unset, their defaults are true. The
plugin wrapper therefore replaces the scheme's `apply_weights` branch with
`apply_int8_linear`, which invokes `lightop.quant.per_token_quant_int8` and
`lightop.gemm_ops.hipblaslt_w8a8_gemm`. The route fails explicitly if those
LightOp APIs are unavailable; it does not silently execute Triton. Setting
either switch to `0` is the control that delegates dense W8A8 back to the
official Triton path. The independently selected AITER messages and config
hits apply only to routed MoE experts.

Both TP ranks constructed `HcuGPUModelRunnerV2`. AITER loaded the ordinary and
bottom-layer gfx938 `E=256,N=256,dtype=int8_w8a8` configs plus the
channel-shuffle tuned CSV. Public E4M3 KV resolved to physical LBHNC with a
64-token kernel page and a 2,176-token hybrid manager page. Target and MTP
prefill captured PIECEWISE and FULL Graphs, MTP decode captured FULL Graphs,
and the service allocated 1,104,474 KV tokens.

The first raw HumanEval run scored 15/16: HumanEval/10 generated a longest
palindromic-prefix implementation instead of the requested suffix. The fresh
rerun passed Accuracy and Pass@1 at 16/16, so the miss was not a stable kernel
failure. The passing report measured 62.66 output tok/s, TTFT 354.6 ms, TPOT
12.8 ms, and latency 1.529 s. Three accepted 3,845-token prefix probes all
returned `17` and produced hit deltas of 0/0/2,112. The sibling hit is 33 x 64
but is not divisible by 2,176, proving fine-grained reuse. Complete-session
MTP counters were 2,336/2,505 accepted drafts (93.25%). All 36 inference
requests returned HTTP 200, no fatal/error marker occurred, and exact teardown
closed the service and returned every card to idle.

Evidence:

- `/tmp/vllm-hcu-validation/qwen35-35b-a3b-channel-int8-current-tp2-mtp3-kvfp8-fine.log`
- `/tmp/vllm-hcu-evalscope/qwen35-35b-a3b-channel-int8-current-tp2-mtp3-kvfp8-fine-run1-20261007`
- `/tmp/vllm-hcu-evalscope/qwen35-35b-a3b-channel-int8-current-tp2-mtp3-kvfp8-fine-run2-20261007`

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

## Qwen3-VL text-only service commands

The Instruct checkpoints use the ordinary text-only FLASH_ATTN profile. Set
`SIZE` to `2B`, `4B`, or `8B`, and adjust memory utilization if required.

```bash
env -u VLLM_PLUGINS \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve "/llm-models-2/hygon/Qwen3-VL-${SIZE}-Instruct-Channel-FP8" \
  --served-model-name "Qwen3-VL-${SIZE}-Instruct-Channel-FP8" \
  --port 10234 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm
```

The Thinking checkpoint must expose reasoning separately and needs a realistic
reasoning budget. The final 8B gate used:

```bash
env -u VLLM_PLUGINS \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES=0,1 \
  vllm serve /llm-models-2/hygon/Qwen3-VL-8B-Thinking-Channel-FP8 \
  --served-model-name Qwen3-VL-8B-Thinking-Channel-FP8 \
  --port 10234 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 8192 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --reasoning-parser qwen3
```

Use `max_tokens=7800` in the HumanEval client for that 8K Thinking profile.
At 2,048 and 3,800 tokens, every failed sample was an output-budget
exhaustion. At 7,800, 15 normally stopped samples passed and HumanEval/1
remained in a repetitive reasoning loop until the limit; this is checkpoint
generation behavior rather than a plugin execution failure.

The 4B Thinking checkpoint used the same command with `4B` substituted for
`8B`. Even at the 7,800-token limit, ten samples exhausted the context inside
reasoning; all six normally stopped samples passed. Treat its 6/16 as a
checkpoint generation-budget limitation, not an accepted precision score.
The 2B Thinking checkpoint was more extreme: only two answers stopped within
7,800 tokens and both passed, while the other 14 exhausted the context. Its
raw 2/16 score has the same budget-bound classification.

The 235B-A22B Instruct MoE route used four cards and AITER:

```bash
env -u VLLM_PLUGINS \
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

This route loaded 221.40 GiB across 24 checkpoint shards, consumed 56.96 GiB
of model memory per TP rank, selected AITER FP8 MoE and its channel-shuffle
tuned CSV on all ranks, and captured both FULL and PIECEWISE Graphs. No MoE
provider fallback, ERROR, or Traceback was logged.

## Qwen3-235B INT8 diagnostic command and accuracy boundary

The best complete HumanEval16 result used TP4, public E4M3 KV, default
FULL_AND_PIECEWISE Graphs, the plugin's default dense INT8 route, and Triton
MoE. It scored raw and normalized 15/16. HumanEval/2 stopped after the invalid
text `def truncate游戏副本`, so this route is diagnostic and is not an accepted
precision result.

```bash
env -u VLLM_PLUGINS \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve \
    /llm-models-2/hygon/Qwen3-235B-A22B-Channel-INT8-w8a8 \
  --served-model-name Qwen3-235B-A22B-Channel-INT8-w8a8 \
  --port 10234 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN \
  --moe-backend triton \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

The complete control matrix was:

| Dense route | MoE | KV | Execution | TP | Result |
| --- | --- | --- | --- | ---: | ---: |
| plugin default LightOp redirect | AITER | E4M3 | default Graph | 4 | 12/16 |
| plugin default LightOp redirect | Triton | E4M3 | default Graph | 4 | 15/16 |
| plugin default LightOp redirect | Triton | auto/BF16 | default Graph | 4 | 14/16 |
| plugin default LightOp redirect | Triton | E4M3 | default Graph | 8 | 14/16 |
| plugin default LightOp redirect | Triton | E4M3 | eager | 4 | targeted failures reproduced |
| official Triton (`VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0`) | Triton | E4M3 | eager | 4 | HumanEval/2 and /11 still failed in the targeted replay |
| official Triton (`VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0`) | AITER | E4M3 | eager | 4 | HumanEval/0, /2, and /12 still failed in the targeted replay |

Changing to BF16 KV or TP8 did not improve the complete score, and eager
targeted replays did not eliminate the failures. Disabling the plugin dense
INT8 redirect improved some individual prompts but did not remove the fixed
HumanEval/2 corruption. AITER MoE used
the gfx938 `int8_w8a8/E=128,N=384` ordinary and bottom-layer configurations;
it also did not remove the failure. The checkpoint config declares
`Qwen3MoeForCausalLM` with 94 hidden layers and no MTP/next-N layer, so MTP3
is not applicable.

Do not infer the dense execution provider from
`Selected TritonInt8ScaledMMLinearKernel` alone. The plugin wraps the same
compressed-tensors scheme and, while
`VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM` retains its default true value,
redirects `apply_weights` to `apply_int8_linear` and the LightOp-backed HCU
path. Set that environment variable to `0` and inspect the effective patch
route before claiming official Triton execution. Since the corruption crossed
all controls above, no model-specific runtime workaround was added.

## DeepSeek-R1 W4A8 TP8 service and client commands

The accepted W4A8 route used regular FLASHMLA, not the V3.2/V4 sparse
backend. Leave the KV layout unset so FLASHMLA resolves LBNHC.

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve /llm-models-2/hygon/DeepSeek-R1-W4A8-V2_6 \
  --served-model-name DeepSeek-R1-W4A8-V2_6 \
  --port 10234 \
  --trust-remote-code \
  --quantization slimquant_w4a8 \
  --tensor-parallel-size 8 \
  --attention-backend FLASHMLA \
  --moe-backend aiter \
  --reasoning-parser deepseek_r1 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --generation-config vllm
```

Use the isolated client with an 8,192-token reasoning budget and no invented
`enable_thinking` override:

```bash
work_dir=/tmp/vllm-hcu-evalscope/deepseek-r1-w4a8-fresh-run
env -i \
  HOME=/tmp/vllm-hcu-eval-home \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model DeepSeek-R1-W4A8-V2_6 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":8192}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" --no-timestamp
```

The checkpoint loaded 333 GiB from 70 shards and consumed 44.27 GiB of model
memory per rank. The runtime selected HcuGPUModelRunnerV2, FLASHMLA with a
64-token block, LBNHC E4M3 KV, and the gfx938 AITER
`int8_w4a8/E=256,N=256` configuration on every rank. The target and MTP3
speculator both captured FULL and PIECEWISE Graphs. The two independent runs
passed raw and normalized HumanEval16 at 16/16 and observed 29.50 and 29.52
output tok/s. Across the service session, MTP accepted 30,794/52,617 draft
tokens (58.5%); prefix metrics recorded 1,664 hits over 6,508 queried tokens.
No ERROR or Traceback was logged, and teardown returned all devices to the
measured 2 MiB idle baseline.

## DeepSeek-V3.2 Channel-INT8 TP8 commands

This checkpoint exposes the V3.2 sparse indexer through `index_topk` while
retaining the `DeepseekV3ForCausalLM` and `DeepSeekMTPModel` display names.
Use the sparse backend and let it resolve the physical cache layout.

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve \
    /llm-models-2/hygon/DeepSeek-V3.2-Channel-INT8-w8a8 \
  --served-model-name DeepSeek-V3.2-Channel-INT8-w8a8 \
  --port 10234 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"thinking":false}'
```

```bash
work_dir=/tmp/vllm-hcu-evalscope/deepseek-v32-int8-fresh-run
env -i \
  HOME=/tmp/vllm-hcu-eval-home \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model DeepSeek-V3.2-Channel-INT8-w8a8 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":4096,"extra_body":{"chat_template_kwargs":{"thinking":false}}}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" --no-timestamp
```

The 642.65 GiB checkpoint loaded from 163 shards and used 85.66 GiB of model
memory per rank. Public `fp8_e4m3` resolved to sparse-MLA `fp8_ds_mla` with
LBNHC, and the service allocated 921,024 KV tokens. AITER selected its INT8
MoE backend, loaded `tuned_fmoe_asm_w8a8_channel_shuffle.csv` on every rank,
and executed the gfx938 W8A8 stage1/stage2 modules. Both HumanEval runs passed
raw and normalized 16/16, observing 26.73 and 25.19 output tok/s. Session MTP
counters were 3,458/4,893 accepted draft tokens (70.7%); prefix counters were
1,600/6,476 hit/query tokens, including a 1,152-token second-request hit in
the explicit prefix probe. The log contained no ERROR or Traceback, and all
cards returned to 2 MiB after exact process-group teardown.

## DeepSeek-V4-Flash-0731 Channel-INT8 TP8 commands

This is the supported V4-Flash-0731 checkpoint, not DeepSeek V4.1. Its
architecture-specific speculative route is DSpark7 rather than a copied MTP3
profile. Keep the 256-token sparse cache block and DeepSeek-V4 tokenizer
mode, and leave the physical KV-cache layout to the backend.

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve \
    /llm-models-2/hygon/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
  --served-model-name DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
  --port 10234 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tensor-parallel-size 8 \
  --moe-backend aiter \
  --speculative-config \
    '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --generation-config vllm
```

```bash
work_dir=/tmp/vllm-hcu-evalscope/deepseek-v4-flash-0731-int8-fresh-run
env -i \
  HOME=/tmp/vllm-hcu-eval-home \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model DeepSeek-V4-Flash-0731-Channel-INT8-w8a8 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"thinking":false}}}' \
  --stream --eval-batch-size 1 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" --no-timestamp
```

The 286.61 GiB, 48-shard checkpoint used 40.07 GiB of model memory per rank.
Public E4M3 resolved to DeepSeek's internal `fp8_ds_mla` layout with BLHNC
physical storage, a 256-token cache block, and 851,896 KV tokens. Dense INT8
selected official Triton scaled-MM. AITER INT8 MoE loaded the gfx938
`E=256,N=256` ordinary and bottom-layer configurations plus the
channel-shuffle table on every rank. The target captured FULL and PIECEWISE
Graphs, while DSpark7 captured its FULL Graphs.

Both fresh HumanEval16 runs scored raw 14/16 and independently normalized
16/16, at 39.73 and 46.16 output tok/s. In both raw reports HumanEval/1 and
/4 contained valid complete Python under an opening but unclosed Markdown
fence; the source-controlled syntax-aware normalizer executed all 16 answers
successfully with no failed task IDs. Report raw and normalized results
separately. The duplicate-prefix probe returned `17` twice and added
1,280/3,156 hit/query tokens. Session DSpark counters were 3,382/4,144
accepted/drafted tokens (81.6%). The log had no ERROR or Traceback, and exact
teardown returned all cards to 2 MiB.

## DeepSeek-V4-Flash-0731 Channel-FP8 TP8 commands

The FP8 checkpoint uses the same V4-Flash-0731 DSpark7 topology as the INT8
checkpoint. It is not the excluded DeepSeek V4.1 model.

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve \
    /llm-models-2/hygon/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
  --served-model-name DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
  --port 10234 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tensor-parallel-size 8 \
  --moe-backend aiter \
  --speculative-config \
    '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --generation-config vllm
```

```bash
work_dir=/tmp/vllm-hcu-evalscope/deepseek-v4-flash-0731-fp8-fresh-run
env -i \
  HOME=/tmp/vllm-hcu-eval-home \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model DeepSeek-V4-Flash-0731-Channel-FP8-w8a8 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"thinking":false}}}' \
  --stream --eval-batch-size 1 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" --no-timestamp
```

The 286.61 GiB, 48-shard checkpoint used about 39.7--39.8 GiB per rank and
allocated 851,950 KV tokens. Public E4M3 resolved to `fp8_ds_mla` with BLHNC
physical storage. Dense layers selected
`ChannelWiseTorchFP8ScaledMMLinearKernel`; AITER FP8 MoE loaded the gfx938
`fp8_w8a8/E=256,N=256` ordinary and bottom-layer configurations and the
channel-shuffle table on every rank. The target captured FULL and PIECEWISE
Graphs, and DSpark7 captured FULL Graphs.

Both HumanEval16 runs scored raw 13/16 and independently normalized 16/16,
at 35.75 and 44.40 output tok/s. In both raw reports HumanEval/1, /4, and /5
contained valid complete Python under an opening but unclosed Markdown fence;
the syntax-aware normalizer passed all 16 with no failed task IDs. The
duplicate-prefix probe returned `17` twice and added 2,816/6,148 hit/query
tokens. Session DSpark counters were 3,218/4,046 accepted/drafted tokens
(79.5%). The log contained no ERROR or Traceback, and teardown returned every
card to 2 MiB.

Evidence:

- `/tmp/vllm-hcu-validation/deepseek-v4-flash-0731-fp8-tp8-dspark7-kvfp8.log`
- `/tmp/vllm-hcu-evalscope/deepseek-v4-flash-0731-fp8-tp8-dspark7-kvfp8-run1`
- `/tmp/vllm-hcu-evalscope/deepseek-v4-flash-0731-fp8-tp8-dspark7-kvfp8-run2`

## DeepSeek-V4-Flash-0731 W4A8 TP8 commands

This 148.61 GiB SlimQuant checkpoint stores the routed-expert path as W4A8
and the attention-side dense path as W8A8. It retains the V4-Flash-0731
DSpark7 architecture and is not the excluded V4.1 model.

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve \
    /llm-models-2/hygon/DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel \
  --served-model-name \
    DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel \
  --port 10234 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tensor-parallel-size 8 \
  --quantization slimquant_w4a8 \
  --moe-backend aiter \
  --speculative-config \
    '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --generation-config vllm
```

```bash
work_dir=/tmp/vllm-hcu-evalscope/deepseek-v4-flash-0731-w4a8-fresh-run
env -i \
  HOME=/tmp/vllm-hcu-eval-home \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model DeepSeek-V4-Flash-0731-W4A8-INT4-Channel-Attn-W8A8-INT8-Channel \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"thinking":false}}}' \
  --stream --eval-batch-size 1 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" --no-timestamp
```

Model loading used 21.39 GiB/rank; the complete runtime footprint was about
22.5--22.7 GiB/rank and provided 1,016,273 KV tokens. Public E4M3 resolved to
`fp8_ds_mla` with BLHNC physical storage. Attention-side dense layers
constructed the `TritonInt8ScaledMMLinearKernel` frontend object while the
default plugin wrapper redirected the executed dense operation to LightOp;
the routed experts selected AITER and loaded
the gfx938 `int8_w4a8/E=256,N=256` ordinary and bottom-layer configurations on
all ranks. The target captured FULL and PIECEWISE Graphs, and DSpark7 captured
FULL Graphs.

Both HumanEval16 runs passed raw and independently normalized 16/16 at 41.86
and 44.98 output tok/s. The duplicate-prefix probe returned `17` twice and
added 2,816/6,148 hit/query tokens. Session DSpark counters were 3,414/4,424
accepted/drafted tokens (77.2%). The log contained no ERROR or Traceback.
The API and worker processes exited cleanly; ROCm reported no live KFD PID,
although the driver retained 0.6--1.9 GiB of delayed per-card accounting
immediately after teardown. A subsequent read returned every card to 2 MiB.

Evidence:

- `/tmp/vllm-hcu-validation/deepseek-v4-flash-0731-w4a8-tp8-dspark7-kvfp8.log`
- `/tmp/vllm-hcu-evalscope/deepseek-v4-flash-0731-w4a8-tp8-dspark7-kvfp8-run1`
- `/tmp/vllm-hcu-evalscope/deepseek-v4-flash-0731-w4a8-tp8-dspark7-kvfp8-run2`

## DeepSeek-V4-Flash Channel-FP8 TP8 commands

Unlike the 0731 checkpoints, this 274.13 GiB, 46-shard checkpoint has one
next-N layer but no `dspark_*` metadata. Use generic MTP3 and retain the
official repeated-single-layer warning; do not copy the 0731 DSpark7 command.

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve \
    /llm-models-2/hygon/DeepSeek-V4-Flash-Channel-FP8-w8a8 \
  --served-model-name DeepSeek-V4-Flash-Channel-FP8-w8a8 \
  --port 10234 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tensor-parallel-size 8 \
  --moe-backend aiter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --generation-config vllm
```

```bash
work_dir=/tmp/vllm-hcu-evalscope/deepseek-v4-flash-fp8-fresh-run
env -i \
  HOME=/tmp/vllm-hcu-eval-home \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model DeepSeek-V4-Flash-Channel-FP8-w8a8 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"thinking":false}}}' \
  --stream --eval-batch-size 1 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" --no-timestamp
```

Model loading used 38.31 GiB/rank and the service allocated 868,343 KV
tokens. Public E4M3 resolved to `fp8_ds_mla` with BLHNC physical storage.
Dense layers selected `ChannelWiseTorchFP8ScaledMMLinearKernel`; AITER FP8
MoE loaded the gfx938 `fp8_w8a8/E=256,N=256` ordinary and bottom-layer
configurations plus the channel-shuffle table on all ranks. The target and
MTP prefill captured FULL and PIECEWISE Graphs; MTP decode captured FULL
Graphs.

Both HumanEval16 runs scored raw 15/16 and independently normalized 16/16 at
27.70 and 28.01 output tok/s. HumanEval/9 was complete, valid Python under an
opening but unclosed Markdown fence in both raw reports. The duplicate-prefix
probe returned `17` twice and added 2,816/6,148 hit/query tokens. Session MTP
counters were 2,770/3,894 accepted/drafted tokens (71.1%). The log contained
no ERROR or Traceback, and teardown returned every card to 2 MiB.

Evidence:

- `/tmp/vllm-hcu-validation/deepseek-v4-flash-fp8-tp8-mtp3-kvfp8.log`
- `/tmp/vllm-hcu-evalscope/deepseek-v4-flash-fp8-tp8-mtp3-kvfp8-run1`
- `/tmp/vllm-hcu-evalscope/deepseek-v4-flash-fp8-tp8-mtp3-kvfp8-run2`

## DeepSeek-V4-Pro-0813 SlimQuant W4A8 single-node TP8 validation

The 66-shard checkpoint is 789.42 GiB on disk, or 98.68 GiB/rank before
runtime overhead at TP8. The following profile completed a cold NFS load,
target and DSpark graph capture, API startup, and a 64-token completion on one
eight-card gfx938 node:

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve \
    /llm-models-2/hygon/DeepSeek-V4-Pro-0813-Channel-INT4-w4a8 \
  --served-model-name DeepSeek-V4-Pro-0813-Channel-INT4-w4a8 \
  --port 10234 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tensor-parallel-size 8 \
  --quantization slimquant_w4a8 \
  --moe-backend aiter \
  --speculative-config \
    '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --generation-config vllm
```

Use a direct local client. This host's proxy environment returns an empty
proxy-generated HTTP 502 unless loopback is explicitly bypassed:

```bash
curl --noproxy '*' -sS \
  http://127.0.0.1:10234/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"DeepSeek-V4-Pro-0813-Channel-INT4-w4a8",
    "prompt":"Write a Python function add(a, b) that returns the sum.\n",
    "max_tokens":64,
    "temperature":0
  }'
```

Run HumanEval16 from a new result directory and bypass all proxy variables:

```bash
work_dir=/tmp/vllm-hcu-evalscope/deepseek-v4-pro-0813-w4a8-fresh-run
test ! -e "$work_dir"
env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model DeepSeek-V4-Pro-0813-Channel-INT4-w4a8 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"thinking":false}}}' \
  --stream --eval-batch-size 1 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" --no-timestamp
```

All ranks constructed `HcuGPUModelRunnerV2`. Public E4M3 resolved to the
DeepSeek sparse `fp8_ds_mla` format with BLHNC storage. The target resolved
`FULL_AND_PIECEWISE`; target and DSpark PIECEWISE/FULL captures completed
without eager fallback. The cold NFS load took 912.7--913.2 seconds and used
104.97 GiB/rank. Final consumed weights plus non-Torch memory was about
106.7--106.9 GiB/rank, graph memory was 0.76--0.79 GiB/rank, and the service
allocated about 21.4--21.6 GiB/rank to KV cache at 0.90 utilization. The
smoke request returned HTTP 200 with 64 generated tokens; DSpark accepted
41/168 drafted tokens during this short request.

The fresh HumanEval16 run passed raw Accuracy and Pass@1 at 16/16. All 16
requests finished with `stop`; none exhausted the 2,048-token budget or
errored, so no syntax-aware normalization was needed. It observed 16.88
output tok/s, 620.9 ms mean TTFT, and 54.5 ms mean TPOT. The run generated
2,057 tokens and accepted 1,726/2,303 DSpark draft tokens (75.0%).

The separately stored `DeepSeek-V4-Pro-0813-INT4-Channel` checkpoint was then
independently cold-started at current MR head `6733c6e`. Its exact server
command was:

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve \
    /llm-models-2/hygon/DeepSeek-V4-Pro-0813-INT4-Channel \
  --served-model-name DeepSeek-V4-Pro-0813-INT4-Channel \
  --port 10239 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tensor-parallel-size 8 \
  --quantization slimquant_w4a8 \
  --moe-backend aiter \
  --speculative-config \
    '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 256 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --generation-config vllm
```

Its isolated HumanEval16 command was:

```bash
env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  -u GH_TOKEN -u GITHUB_TOKEN -u OPENAI_API_KEY \
  HOME=/tmp/vllm-hcu-eval-home-v4pro-int4 \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model DeepSeek-V4-Pro-0813-INT4-Channel \
  --api-url http://127.0.0.1:10239/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"thinking":false}}}' \
  --stream --eval-batch-size 1 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir \
    /tmp/vllm-hcu-evalscope/deepseek-v4-pro-0813-int4-channel-current-tp8-dspark7-kvfp8-run1-20261007 \
  --no-timestamp
```

All ranks again constructed `HcuGPUModelRunnerV2`. The target cold load took
863.98 seconds; total model initialization used 104.97 GiB/rank and about
901.1 seconds. Public E4M3 resolved to `fp8_ds_mla` with BLHNC storage. The
gfx938 AITER ordinary and bottom-layer W4A8 configurations both loaded. The
target captured PIECEWISE and FULL Graphs and DSpark7 captured FULL Graphs;
the service allocated 143,064 KV tokens.

Raw HumanEval Accuracy and Pass@1 passed 16/16. The report observed 19.03
output tok/s, 447.8 ms mean TTFT, 49.3 ms mean TPOT, and 6.755 s mean latency.
The HumanEval window accepted 1,726/2,303 DSpark draft tokens (75.0%). Two
2,506-token prefix probes added 2,048 hits over 5,012 queried tokens. Because
the requested DSpark profile uses probabilistic draft sampling, the two short
probe continuations differed; this is recorded only as prefix-reuse evidence,
not a deterministic-output claim. No ERROR, Traceback, VM fault, or OOM
occurred, and exact teardown returned all cards to 2 MiB.

`DeepSeek-V4-Pro-0813-INT8-Channel` remains capacity-blocked: it occupies
1,545.42 GiB physically, or 193.18 GiB/rank at TP8 before runtime overhead,
and cannot fit this node's 143.98 GiB/card. It requires at least TP16 or a
higher-memory topology.

Evidence:

- `/tmp/vllm-hcu-validation/deepseek-v4-pro-0813-w4a8-tp8-dspark7-kvfp8.log`
- `/tmp/vllm-hcu-validation/deepseek-v4-pro-0813-w4a8-smoke-direct.json`
- `/tmp/vllm-hcu-validation/deepseek-v4-pro-0813-w4a8-metrics.txt`
- `/tmp/vllm-hcu-validation/deepseek-v4-pro-0813-w4a8-tp8-dspark7-kvfp8-humaneval16.log`
- `/tmp/vllm-hcu-evalscope/deepseek-v4-pro-0813-w4a8-tp8-dspark7-kvfp8-run1-20261007`
- `/tmp/vllm-hcu-validation/deepseek-v4-pro-0813-int4-channel-current-tp8-dspark7-kvfp8.log`
- `/tmp/vllm-hcu-evalscope/deepseek-v4-pro-0813-int4-channel-current-tp8-dspark7-kvfp8-run1-20261007`

## GLM-5.3 Channel-FP8 TP8 commands

The `/llm-models-2/hygon` copy is 707.90 GiB over 141 shards. Its config,
weight-index, and filename-plus-size-list hashes match the previously
validated `/models/GLM-5.3-Channel-FP8-w8a8` artifact. It is a separate copy,
so the following run independently cold-started and evaluated this path:

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve /llm-models-2/hygon/GLM-5.3-Channel-FP8-w8a8 \
  --served-model-name GLM-5.3-Channel-FP8-w8a8 \
  --port 10234 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --attention-backend FLASHMLA_SPARSE \
  --reasoning-parser glm45 \
  --moe-backend aiter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}'
```

Run the client from a fresh directory and bypass the host proxy for loopback:

```bash
work_dir=/tmp/vllm-hcu-evalscope/glm53-channel-fp8-tp8-mtp3-fresh-run
test ! -e "$work_dir"
env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model GLM-5.3-Channel-FP8-w8a8 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"low"}}}' \
  --stream \
  --eval-batch-size 8 \
  --timeout 7200 \
  --limit 16 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" \
  --no-timestamp
```

All eight ranks constructed `HcuGPUModelRunnerV2`. Public `fp8_e4m3`
resolved to `fp8_ds_mla` with LBNHC storage. Dense layers selected
`ChannelWiseTorchFP8ScaledMMLinearKernel`; AITER selected FP8 MoE and loaded
the gfx938 `fp8_w8a8/E=256,N=256` ordinary and bottom-layer configurations
plus the channel-shuffle table. The target and MTP prefill captured FULL and
PIECEWISE Graphs, while MTP decode captured FULL Graphs. Model loading used
94.38 GiB/rank, and the service allocated 728,768 KV tokens.

The fresh HumanEval16 run passed raw Accuracy and Pass@1 at 16/16. All 16
requests finished with `stop`; none reached the length limit or errored. It
observed 14.86 output tok/s, 2,273.8 ms mean TTFT, and 37.0 ms mean TPOT.
MTP accepted 875/963 drafted tokens (90.9%). The server log contained no
ERROR or Traceback, and teardown returned all eight cards to 2 MiB.

Evidence:

- `/tmp/vllm-hcu-validation/llm-models-2-glm53-channel-fp8-tp8-mtp3-kvfp8.log`
- `/tmp/vllm-hcu-evalscope/llm-models-2-glm53-channel-fp8-tp8-mtp3-kvfp8-run1-20261007`

## GLM-5.3 Flash Channel-INT8 TP4 commands

This 315.23 GiB, 50-shard GLM5Next checkpoint has native NoPE sparse MLA
(`qk_rope_head_dim=0`, `Dqk=512`). The accepted accuracy control therefore
uses auto/BF16 KV; do not add public E4M3 until the installed HCU FlashMLA
sparse FP8 kernel supports the D512 contract.

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve \
    /llm-models-2/hygon/GLM-5.3-Flash-Channel-INT8-w8a8 \
  --served-model-name GLM-5.3-Flash-Channel-INT8-w8a8 \
  --port 10234 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --attention-backend FLASHMLA_SPARSE \
  --reasoning-parser glm45 \
  --moe-backend aiter \
  --speculative-config \
    '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}'
```

```bash
work_dir=/tmp/vllm-hcu-evalscope/glm53-flash-int8-bf16-fresh-run
env -i \
  HOME=/tmp/vllm-hcu-eval-home \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model GLM-5.3-Flash-Channel-INT8-w8a8 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"low"}}}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" --no-timestamp
```

The accepted run used 79.27 GiB of model memory per rank. GLM5Next selected
`mamba-cache-mode=align` automatically, a 64-token FlashMLA kernel page, a
1,152-token hybrid manager page, LBHNC, and 499,712 KV tokens. Target and
MTP3 speculator captured the default FULL and PIECEWISE Graphs, and the local
argmax marker was present. The requested AITER INT8 MoE route had no supported
`M=1,E=288,N1=1024,N2=4096,K=4096,top_k=8` solution, so that decode shape
fell back to official vLLM Triton rather than claiming an AITER config hit.

Both fresh HumanEval16 runs passed raw 16/16, observing 11.03 and 15.72 output
tok/s. The duplicate-prefix probe returned `17` twice and reused 1,152 tokens;
session totals were 1,152/10,018 prefix hit/query tokens and 1,862/3,024 MTP
draft tokens accepted (61.6%). The accepted log had no ERROR or Traceback,
and exact process-group teardown returned all cards to 2 MiB.

Adding `--kv-cache-dtype fp8_e4m3` resolves to the physical `fp8_ds_mla`
format and fails before serving because the current writer requires a 64-wide
RoPE component. A zero-tail writer experiment was not retained: direct calls
to the installed FlashMLA sparse FP8 kernel rejected `Dqk=512` with either a
656-byte DS-MLA page or a 528-byte NoPE page. This is a provider capability
gap; constructing a page the consumer cannot execute would not be a fix.

## GLM-5.3 Flash Channel-FP8 TP4 commands

This FP8-weight sibling uses the same native NoPE sparse-MLA contract. Its
accepted route also uses auto/BF16 KV; the provider boundary documented above
applies equally to public E4M3 KV.

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve \
    /llm-models-2/hygon/GLM-5.3-Flash-Channel-FP8-w8a8 \
  --served-model-name GLM-5.3-Flash-Channel-FP8-w8a8 \
  --port 10234 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --attention-backend FLASHMLA_SPARSE \
  --reasoning-parser glm45 \
  --moe-backend aiter \
  --speculative-config \
    '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}'
```

```bash
work_dir=/tmp/vllm-hcu-evalscope/glm53-flash-fp8-bf16-fresh-run
env -i \
  HOME=/tmp/vllm-hcu-eval-home \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model GLM-5.3-Flash-Channel-FP8-w8a8 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"low"}}}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" --no-timestamp
```

The TP4 run loaded 306.33 GiB of checkpoint data and used 78.99 GiB of model
memory per rank. It selected MRV2, LBHNC, a 64-token FlashMLA kernel page, a
1,152-token hybrid-manager page, and 503,398 KV tokens. GLM5Next selected
Mamba align automatically. Target and MTP3 speculator captured the default
FULL and PIECEWISE Graphs, and local argmax was active. The requested AITER
FP8 MoE route had no supported
`M=1,E=288,N1=1024,N2=4096,K=4096,top_k=8` solution, so that decode shape
fell back to official vLLM Triton.

Two fresh HumanEval16 runs passed raw 16/16 at 16.12 and 28.06 output tok/s.
The duplicate-prefix probe returned `17` twice and accumulated 2,304/12,110
prefix hit/query tokens. Session MTP acceptance was 2,247/2,340 (96.0%). The
log had no ERROR or Traceback, and exact teardown returned all cards to 2 MiB.

## MiniMax M2.5 Channel-INT8 TP4 commands

MiniMax M2.5 exposes `num_mtp_modules=3`, but vLLM 0.28.1 does not register a
MiniMax M2 built-in MTP draft architecture. Its main-model loader deliberately
drops the appended prediction layers, so `method=mtp` is rejected during
configuration. The accepted TP route therefore omits speculative decoding.

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve \
    /llm-models-2/hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --served-model-name MiniMax-M2.5-Channel-INT8-w8a8 \
  --port 10234 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN \
  --reasoning-parser minimax_m2_append_think \
  --moe-backend aiter \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm
```

```bash
work_dir=/tmp/vllm-hcu-evalscope/minimax-m25-int8-tp4-kvfp8-fresh-run
env -i \
  HOME=/tmp/vllm-hcu-eval-home \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model MiniMax-M2.5-Channel-INT8-w8a8 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":3800}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" --no-timestamp
```

The 214.37 GiB checkpoint used 56.15 GiB of model memory per rank. Dense INT8
selected official Triton scaled-MM, while MoE loaded the gfx938 AITER
`E=256,N=384` ordinary and bottom-layer configs plus the channel-shuffle
table. MRV2 captured default FULL and PIECEWISE Graphs, and public E4M3 used
LBNHC with 64-token blocks and 1,487,936 KV tokens.

At a 2,048-token client limit the raw score was 14/16: HumanEval/1 and /10
both stopped at `max_tokens` inside reasoning. Raising only the client limit
to 3,800 produced raw 16/16 twice at 49.66 and 49.47 output tok/s. Duplicate
prefix requests both returned `17`; session prefix counters reached
6,656/11,954 hit/query tokens. The successful log had no ERROR or Traceback,
and exact teardown returned all cards to 2 MiB.

## Hy3 Channel-INT8 TP4 commands

HYV3 has one next-token prediction layer, so the accepted profile uses MTP2.
Do not enable local argmax reduction: `HYV3MTP` does not implement
`get_top_tokens()`, and vLLM correctly rejects that optional setting.

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve \
    /llm-models-2/hygon/Hy3-Channel-INT8-w8a8 \
  --served-model-name Hy3-Channel-INT8-w8a8 \
  --port 10234 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN \
  --reasoning-parser hy_v3 \
  --moe-backend aiter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}'
```

```bash
work_dir=/tmp/vllm-hcu-evalscope/hy3-channel-int8-tp4-mtp2-kvfp8-fresh-run
env -i \
  HOME=/tmp/vllm-hcu-eval-home \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Hy3-Channel-INT8-w8a8 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" --no-timestamp
```

The 279.71 GiB checkpoint used 71.6 GiB of model memory per rank. MRV2 used
LBNHC with 64-token blocks, public E4M3 KV, and 1,073,792 KV tokens. Target
and MTP2 speculator captured default FULL and PIECEWISE Graphs. Dense INT8
selected official Triton scaled-MM. AITER had no supported solution for the
observed `M=1,E=192,N1=768,N2=4096,K=4096,top_k=8` MoE shape, so execution
fell back to official vLLM Triton for that shape.

Two fresh HumanEval16 runs passed raw 16/16 at 15.53 and 16.21 output tok/s.
The duplicate-prefix probe returned `17` twice; session counters were
2,752/8,616 prefix hit/query tokens and 1,222/1,622 accepted/drafted MTP
tokens (75.3%). The corrected log had no ERROR or Traceback, and exact
teardown returned all cards to 2 MiB.

## Hy3 Channel-FP8 TP4 commands

This checkpoint has the same HYV3 topology and one next-token prediction
layer as the INT8 checkpoint, so MTP2 remains the accepted speculative route.
It uses channel-wise FP8 compressed tensors for dense linear layers. Do not
copy the local-argmax option from GLM profiles.

```bash
env -u VLLM_PLUGINS \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_KV_CACHE_LAYOUT=HND \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve /llm-models-2/hygon/Hy3-Channel-FP8-w8a8 \
  --served-model-name Hy3-Channel-FP8-w8a8 \
  --port 10234 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN \
  --reasoning-parser hy_v3 \
  --moe-backend aiter \
  --linear-backend auto \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}'
```

```bash
work_dir=/tmp/vllm-hcu-evalscope/hy3-channel-fp8-tp4-mtp2-kvfp8-fresh-run
env -i \
  HOME=/tmp/vllm-hcu-eval-home \
  PATH="$PATH" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Hy3-Channel-FP8-w8a8 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \
  --stream --eval-batch-size 8 --timeout 7200 --limit 16 \
  --datasets humaneval --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" --no-timestamp
```

The 279.71 GiB checkpoint used 71.6 GiB of model memory per rank. MRV2
selected `ChannelWiseTorchFP8ScaledMMLinearKernel`, used LBHNC with 64-token
blocks and public E4M3 KV, and allocated 1,073,536 KV tokens. Target and MTP2
speculator captured default FULL and PIECEWISE Graphs. Although AITER FP8 MoE
was requested, it had no supported solution for the observed
`M=1,E=192,N1=768,N2=4096,K=4096,top_k=8` shape and correctly fell back to
official vLLM Triton for that shape. This checkpoint-specific fallback must
not be replaced by provider evidence from the older sero checkpoint.

Two fresh HumanEval16 runs passed raw 16/16 at 8.78 and 19.71 output tok/s.
The duplicate-prefix probe returned `17` twice and added 2,304/4,738 prefix
hit/query tokens. Session MTP counters were 1,301/1,682 accepted/drafted
tokens (77.3%). The successful log had no ERROR or Traceback, and exact
teardown returned all cards to 2 MiB.

## Hy4 preview Channel-FP8 TP8 commands

The `/llm-models-2/hygon` copy is 736.24 GiB over 131 shards. Its config,
weight-index, and filename-plus-size-list hashes match the previously
validated `/models/Hy4-preview-Channel-FP8-w8a8` artifact. The files are not
hard links, so this path was independently cold-started and evaluated:

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve /llm-models-2/hygon/Hy4-preview-Channel-FP8-w8a8 \
  --served-model-name Hy4-preview-Channel-FP8-w8a8 \
  --port 10234 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend aiter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 64 \
  --enable-prefix-caching \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}'
```

```bash
work_dir=/tmp/vllm-hcu-evalscope/hy4-preview-channel-fp8-tp8-mtp3-fresh-run
test ! -e "$work_dir"
env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  VLLM_HCU_EVALSCOPE_API_KEY=EMPTY \
  VLLM_HCU_HUMANEVAL_ISOLATED=1 \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  python -m tests.integration.server.evalscope_secure_cli eval \
  --model Hy4-preview-Channel-FP8-w8a8 \
  --api-url http://127.0.0.1:10234/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \
  --stream \
  --eval-batch-size 8 \
  --timeout 7200 \
  --limit 16 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{}}' \
  --work-dir "$work_dir" \
  --no-timestamp
```

All eight ranks constructed `HcuGPUModelRunnerV2`. Public `fp8_e4m3`
resolved to `fp8_ds_mla` with LBNHC storage. Dense layers selected
`ChannelWiseTorchFP8ScaledMMLinearKernel`; AITER loaded the gfx938
`fp8_w8a8/E=256,N=256` ordinary and bottom-layer configurations plus the
channel-shuffle table. The plugin auto-enabled breakable CUDA Graph for HY4.
The target and MTP prefill captured FULL and PIECEWISE Graphs; MTP decode
captured FULL Graphs. The cold NFS load took 876.09 seconds, total model
loading used 98.24 GiB/rank, and the service allocated 662,976 KV tokens.

The fresh HumanEval16 run passed raw Accuracy and Pass@1 at 16/16. All 16
requests finished with `stop`; none reached the length limit or errored. It
observed 26.23 output tok/s, 1,001.1 ms mean TTFT, and 30.8 ms mean TPOT.
MTP accepted 1,549/1,689 drafted tokens (91.7%). The server log contained no
ERROR or Traceback, and teardown returned all eight cards to 2 MiB.

Evidence:

- `/tmp/vllm-hcu-validation/llm-models-2-hy4-preview-channel-fp8-tp8-mtp3-kvfp8.log`
- `/tmp/vllm-hcu-evalscope/llm-models-2-hy4-preview-channel-fp8-tp8-mtp3-kvfp8-run1-20261007`

## Hy4 preview Channel-FP8 DP8/EP8 command

The same `/llm-models-2` checkpoint also passed a dynamic DP8/TP1/EP8 run
with DeepEP low latency, DeepGEMM, MTP3, public E4M3 sparse KV, Model Runner
V2, and the default `FULL_AND_PIECEWISE` Graph policy:

```bash
env -u VLLM_PLUGINS -u VLLM_KV_CACHE_LAYOUT \
  -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve /llm-models-2/hygon/Hy4-preview-Channel-FP8-w8a8 \
  --served-model-name Hy4-preview-Channel-FP8-w8a8 \
  --api-server-count 8 \
  --port 10236 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --attention-backend FLASHMLA_SPARSE \
  --moe-backend deep_gemm \
  --all2all-backend deepep_low_latency \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype fp8_e4m3 \
  --block-size 64 \
  --kv-cache-memory-bytes 536870912 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --max-model-len 4096 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 8 \
  --generation-config vllm \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}'
```

Use the following client command for the matching HumanEval16 result:

```bash
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
http_proxy= https_proxy= all_proxy= \
EVALSCOPE_API_KEY=EMPTY \
VLLM_HCU_HUMANEVAL_ISOLATED=1 \
PYTHONPATH=/models/.worktrees/vllm-plugin-das-v0281-gfx938-validation \
python -m tests.integration.server.evalscope_secure_cli eval \
  --model Hy4-preview-Channel-FP8-w8a8 \
  --api-url http://127.0.0.1:10236/v1 \
  --eval-type openai_api \
  --generation-config \
    '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \
  --stream \
  --eval-batch-size 8 \
  --timeout 7200 \
  --limit 16 \
  --datasets humaneval \
  --dataset-args '{"humaneval":{}}' \
  --work-dir \
    /tmp/vllm-hcu-evalscope/llm-models-2-hy4-dp8-ep8-mtp3-kvfp8-bt256-run1-20261007 \
  --no-timestamp
```

The run passed raw Accuracy and Pass@1 at 16/16, with all 16 requests ending
in `stop`. HumanEval traffic accepted 1,489/1,602 MTP draft tokens (92.9%).
Nine identical long-prefix requests then produced 4,992 aggregate prefix-hit
tokens, proving reuse after the load balancer repeated DP ranks. All eight
ranks used `HcuGPUModelRunnerV2`, `DeepEPLLAll2AllManager`,
`DeepEPLLPrepareAndFinalize`, and the low-latency
`DeepEPDeepGemmMaskedExperts` route. Public `fp8_e4m3` resolved to
`fp8_ds_mla` with LBNHC storage and 9,792 KV tokens per engine. The log had
no ERROR, Traceback, VM fault, or dead engine.

Keep `--max-num-batched-tokens 256` for this dynamic DP8/EP8 profile. A
single-variable run at 64 returned HTTP 200 for every request but scored only
1/16, generated long multilingual/repetitive completions, and accepted only
10,763/35,658 MTP draft tokens (30.2%). Restoring 256 recovered 16/16 and
92.9% MTP acceptance. This is an observed command boundary, not evidence of
an internal root cause.

The current branch does not contain the historical offline static-EPLB source
modules. Therefore, do not treat an older static-map run or its 64-token
scheduler setting as evidence for this dynamic profile or as proof that this
MR supports static EPLB.

Evidence:

- `/tmp/vllm-hcu-validation/llm-models-2-hy4-dp8-ep8-mtp3-kvfp8-bt256.log`
- `/tmp/vllm-hcu-evalscope/llm-models-2-hy4-dp8-ep8-mtp3-kvfp8-bt256-run1-20261007`
- `/tmp/vllm-hcu-validation/llm-models-2-hy4-dp8-ep8-mtp3-kvfp8.log`
- `/tmp/vllm-hcu-evalscope/llm-models-2-hy4-dp8-ep8-mtp3-kvfp8-run1-20261007`

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

EvalScope may score a logically correct HumanEval completion as invalid when
the checkpoint returns an unindented function body instead of a complete
function. Preserve the raw report, then use the integration harness's
syntax-aware normalizer inside the isolated execution boundary. It restores
either all body indentation or only the stripped first line, accepting a
transformation only when a synthetic function wrapper parses. It must not
rewrite a completion that already defines the expected entry point. For the
397B-A17B run, raw EvalScope scored 13/16 twice; the misses were correct bare
bodies and the normalized official checker scored 16/16 twice. The warm
second run observed 42.16 output tok/s.

All services were started in isolated process groups. Teardown first sent
TERM to the complete PGID, verified every process in that PGID, and escalated
only the Flash-Next PGID after its workers exceeded the grace period. The
final device state was 2 MiB used on every card.
