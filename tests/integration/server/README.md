# Server integration

This layer contains real OpenAI-compatible server tests.

Current coverage:

- `test_qwen3_protocol_features.py` starts one `Qwen3-4B` server and verifies
  completions, chat/Jinja rendering, reasoning and logprobs, JSON mode,
  JSON Schema, EBNF grammar, named function calls, Anthropic Messages,
  streaming, seeded top-k/top-p sampling, Prometheus metrics, request-length
  rejection, truncation, and `ignore_eos`.
- `test_qwen25_server_smoke.py` verifies that `Qwen2.5-1.5B-Instruct`
  starts as an OpenAI-compatible chat service.
- `test_qwen3_pooling_server.py` starts Qwen3 embedding and reranking models
  with the pooling runner, and verifies `/v1/embeddings`, `/score`, and
  `/rerank` responses.
- `test_evalscope_qwen3_8b_gsm8k.py` starts a single-HCU, eager-mode
  `Qwen3-8B` server and runs a 10-sample EvalScope GSM8K smoke test.
- `test_evalscope_qwen35_9b_gsm8k.py` starts a single-HCU, eager-mode
  `Qwen3.5-9B` server and checks EvalScope GSM8K Pass@1.
- `test_evalscope_qwen3_vl_8b_mmmu.py` starts a single-HCU, eager-mode
  `Qwen3-VL-8B-Instruct` server and checks EvalScope MMMU multimodal accuracy.
- `test_evalscope_deepseek_r1_gsm8k.py` starts `vllm serve` for
  DeepSeek-R1 Channel-FP8 W8A8 with TP=8, waits for `/health`, then runs
  EvalScope on GSM8K through the OpenAI API.
- `test_evalscope_glm52_pcp_humaneval.py` checks the collection-safe
  GLM-5.2 model-runner-v2 TP=4, PCP=2, EP, eager, Triton-MoE launch contract,
  then runs 32 deterministic HumanEval samples through the OpenAI API and
  requires `mean_acc >= 0.90`.
- `test_evalscope_deepseek_v4_dspark_humaneval.py` defines TP8+DSpark and
  DP8+EP8+DSpark profiles for both Channel-FP8 and Channel-INT8 checkpoints on
  the first 32 ModelScope HumanEval samples. The DP profile exposes only
  `--all2all-backend deepep_auto`; PCP, P/D
  disaggregation, a public MoE backend, and separate HT/LL switches are
  excluded. Acceptance requires 32 predictions, 32 reviews, and exact `1.0`
  scores for both HumanEval metrics.
- `test_evalscope_deepseek_v4_dspark_mooncake_pd.py` adds the isolated
  4-prefill-card + 4-decode-card Mooncake topology for the same FP8 and INT8
  checkpoints. It starts and stops only its own P, D, and proxy process trees,
  runs HumanEval-32 through the proxy, and requires positive Mooncake transfer,
  DeepEP/DeepGEMM HT+LL, and DSpark metrics evidence.

The Qwen3-8B smoke test needs one local HCU device, the checkpoint at
`/models/llm-models/qwen3/Qwen3-8B`, `vllm`, and `evalscope`. Select it with
`-k qwen3_8b_gsm8k_evalscope_server`.

The Qwen3.5-9B and Qwen3-VL-8B accuracy tests are selected with
`-k qwen35_9b_gsm8k_evalscope_server` and
`-k qwen3_vl_8b_mmmu_evalscope_server`.

Run the protocol coverage with:

```bash
python tools/run_patch_tests.py --suite model -- -k qwen3_protocol
```

Set `VLLM_HCU_PROTOCOL_MODEL` to override the default `qwen3/Qwen3-4B`
checkpoint. Server output is stored under
`/tmp/vllm-hcu-integration/logs` unless
`VLLM_HCU_INTEGRATION_LOG_DIR` is set.

The DeepSeek-R1 test needs eight local HCU devices, the local model path,
`vllm`, and `evalscope`. It is marked `hcu`, `model`, `multi_hcu`,
`hcu_count(8)`, `slow`, `nightly`, and `external_service("evalscope")`, so it
is excluded from `integration-smoke`.

The GLM-5.2 acceptance server defaults to
`/models/GLM-5___1-Channel-FP8-w8a8`; override it with
`VLLM_HCU_GLM52_MODEL`. The EvalScope configuration itself can be replaced
with `VLLM_HCU_GLM52_HUMANEVAL_CONFIG`. It requires eight HCU devices and
EvalScope, and intentionally uses model-runner v2 with `TP=4`, `PCP=2`, EP,
eager execution, and the Triton MoE backend.

The DeepSeek-V4 acceptance server defaults to
`/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8`; override it with
`VLLM_HCU_DEEPSEEK_V4_FLASH_0731_MODEL`. Override the profiled YAML with
`VLLM_HCU_DEEPSEEK_V4_DSPARK_HUMANEVAL_CONFIG`.

The Channel-INT8 acceptance profile defaults to
`/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8`; override it with
`VLLM_HCU_DEEPSEEK_V4_FLASH_0731_INT8_MODEL`. Override its profiled YAML with
`VLLM_HCU_DEEPSEEK_V4_INT8_DSPARK_HUMANEVAL_CONFIG`.

The P/D profiles use those same model override variables. Override their
shared YAML with `VLLM_HCU_DEEPSEEK_V4_DSPARK_MOONCAKE_PD_CONFIG` and point
`VLLM_V0251_SOURCE_ROOT` at the vLLM source tree that contains the official
Mooncake proxy. See
`docs/deepseek_v4_flash_0731_dspark_mooncake_pd_validation.md` for the exact P,
D, proxy, curl, and pytest commands.
