# Step5 W8A8 single-node DeepEP and DeepGEMM

The Prefill run uses TP1/DP8/EP8, `deepep_high_throughput`, and the EP8
WASP128 DeepGEMM kernels. Set
`VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE=1` and
`DEEP_EP_NORMAL_MNVL=0` with the existing DBO buffer-pair and fused scatter
options. The intranode path derives row-to-expert indices from the aligned
receive counts.

## Prefill reproduction on eight gfx938 devices

Use the DTK 26.10 vLLM image tagged
`0.28.1rc1-ubuntu22.04-dtk2610-py3.12-20261113-0532` with DeepEP installed.
The measured DeepGEMM wheel version was `2.1.0+das.dtk2610.655ebb3`, built
from the DeepGEMM `step5-gfx938-ep8-wasp128` branch at
`655ebb3c4c9cf8df9d7692f636b0dfa5db019f80`.

Run [`serve_ep8_wasp128_prefill.sh`](serve_ep8_wasp128_prefill.sh) on the
eight-device host. Set the model, tokenizer, qualified hipBLASLt directory,
dense tuning JSON, and static EP8 expert placement using the environment
variables listed by `--help`. Put `STEP5_RUN_DIR` on local storage with
room for compilation caches. The script writes the 16-SMS communication
configuration, sets the DBO threshold to 4,096, and starts the service
with `--max-num-batched-tokens 8192`. Use `--dry-run` to inspect the
arguments and communication JSON.

For the recorded performance, the model was Step5 Preview Channel-FP8
W8A8, with the benchmark's selected EP8 placement and gfx938 dense tuning.
Those input assets and the wheel binaries are supplied by the deployment.
The opt-in `HCU_STEP5_DSA_TOPK_EXACT_EMIT=1` enables a single prefix scan
when the resident prefill selector finds an exact Top-K count in a
16,384-region tile. Other rows retain the usual tie handling.

Warm the service with 32 requests of 131,072 input tokens and one output
token. Use the same shape for the formal round. Prefill throughput is
total input tokens divided by the largest TTFT in the batch. The 2026-11-22
S9 formal round with seed 44 completed all 32 requests in 176.0746 s at
23,821.18 input tok/s. It recorded zero preemptions and no inference JIT.
Worker logs confirmed `EP8 WASP128 native G1/G2 active through DeepGEMM`
and `DeepEP dual-buffer overlap active` on all eight ranks.

## Decode

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
