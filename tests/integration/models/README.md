# Model integration

Real checkpoint loading and normalized generation comparisons.

The first smoke test uses `qwen3.5/Qwen3.5-9B` because its attention head
dimension is compatible with `VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1` while still
fitting a single-HCU sanity pass. Resolve it with one of:

- `--model-root /models/llm-models`
- `VLLM_HCU_TEST_MODEL_ROOT=/models/llm-models`
- `VLLM_HCU_QWEN35_9B_MODEL=/absolute/path/to/Qwen3.5-9B`

On the shared HCU hosts, the helper also accepts `/models/llm-models` as a
last-resort local default so ad-hoc runs work without extra flags.

Additional model axes use:

- `qwen2.5/Qwen2.5-1.5B-Instruct` for Qwen2.5 OpenAI serving.
- `vllm-gptq-models/qwen2.5/Qwen2.5-VL-3B-Instruct` for image input and
  M-RoPE.
- `qwen3/Qwen3-Embedding-0.6B` for pooling embeddings.
- `qwen3/Qwen3-Reranker-0.6B` for relevance ordering.
- `vllm-optest-models/tiiuae/falcon-mamba-tiny-dev` for real Mamba prefill
  and decode.

Override these paths with `VLLM_HCU_QWEN25_15B_MODEL`,
`VLLM_HCU_QWEN25_VL_3B_MODEL`, `VLLM_HCU_QWEN3_EMBEDDING_06B_MODEL`, and
`VLLM_HCU_QWEN3_RERANKER_06B_MODEL`.

The real Mamba path can be overridden through `VLLM_HCU_FALCON_MAMBA_MODEL`.

`test_glm52_pcp_mrv2.py` provides the eight-HCU GLM-5.2 PCP acceptance
workload. It makes two fixed-ID deterministic smoke requests and verifies
32K- and 64K-token prefills each produce decode token IDs while the server's
worker group remains live. Per-case JSON artifacts include generated token
IDs, latency, TTFT, aggregate token throughput, and peak observed device
memory. Set `VLLM_HCU_TEST_ARTIFACT_DIR` to select the output directory; the
default is `/tmp/vllm-hcu-integration/glm52-pcp`.
