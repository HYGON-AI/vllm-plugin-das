# Step5 W8A8 single-node DeepEP and DeepGEMM

The Prefill run uses TP1/DP8/EP8, `deepep_high_throughput`, and the EP8
WASP128 DeepGEMM kernels. Set
`VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE=1` and
`DEEP_EP_NORMAL_MNVL=0` with the existing DBO buffer-pair and fused scatter
options. The intranode path derives row-to-expert indices from the aligned
receive counts.

The Decode trace uses TP4/DP2/EP8 on one node, `deepep_low_latency`, and a
DeepGEMM wheel with gfx938 channel-FP8 masked pair32 support. It was captured
with the DTK 2610 `1htllv5` wheel. Enable these options on the Decode worker:

| Setting | Value in the trace |
| --- | --- |
| `VLLM_HCU_USE_CUSTOM_OPS` | `1` |
| `HCU_STEP5_DEEPEP_SWIGLUSTEP` | `1` |
| `HCU_STEP5_DEEPEP_TP_TOKENS` | `1` |
| `HCU_STEP5_DEEPEP_TP_DIRECT_OUT` | `1` |
| `W8A8_INT8_MASKED_BLOCKS_PER_CU` | `2` |
| `HCU_DBO_MIN_TOKENS` | `0` |

Use replicated TP inputs for the token partition, with DBO and sequence
parallelism disabled. To enable the LightOp RMSNorm in Inductor, set
`VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=1` and add `+GemmaRMSNorm` to the
compilation config's `custom_ops`.

On S7, the run used `--max-num-batched-tokens 512` and
`--long-prefill-token-threshold 32`. The 1,024-token limit failed to allocate
the ROCShmem heap. The trace was captured with a diagnostic cache release
in an isolated DeepEP Python package. The 1,024-token limit failed even
with the cache release. Startup at 512 tokens with the stock Python
package has not been validated. The run used the stock DeepEP native
library and ROCShmem.

The trace uses 131,072 input tokens, 32 concurrent requests, 4,096 output
tokens per request, and synthetic MTP3 with per-position probabilities
`[0.838992, 0.598322, 0.362686]`. Capture the middle 10 decode steps with
vLLM's Torch profiler. The measured trace does not establish an unprofiled
throughput gain for the pair32 launch setting or direct TP gather.
