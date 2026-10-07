# DSV4.1 Flash INT8 validation on bw7/bw6

## Scope and baseline

The checkpoint is named `DeepSeek-V4.1-Flash-Channel-INT4-w4a8`, but its
configuration and expert storage select W8A8 INT8. These results validate that
existing checkpoint, without changing its format.

The baseline is `feat/dsv41-flash-w4a8` at `836763f8`, inherited from
`refactor/v0.28.1-hcu-dev-dsv41`. The vLLM checkout is
`v0.29.1-dev-dsv41` at `4c349cf634`. Validation used the existing
`vllm_028_0915_laibao` containers on both nodes.

## Patch groups

| Group | Failure evidence and fix | Scope |
| --- | --- | --- |
| Sparse MLA/indexer | Packed NORMAL cache values must retain their FP8 dtype and scales. The installed gfx936 LightOp prefill FP8 MQA probe returned zeros; use the BF16 path after dequantizing K with its scale. | gfx936 prefill and packed indexer cache handling |
| Registered expert offload | The pinned allocator rounded a 1.0546875 GiB expert tensor to a 2 GiB allocation. Allocate exact-size registered anonymous memory. Preserve the UVA marker across DeepGEMM post-load replacement only when device, dtype, pointer and element count remain identical. | Opt-in `VLLM_HCU_REGISTERED_CPU_OFFLOAD=1`; post-load preservation additionally requires HCU DeepGEMM |
| INT8 MoE clamp | The original INT8 expert activation omitted the model's `swiglu_limit=10.0`. Select the clamp activation when the configured limit is positive. | Standard/HT and batched/LL INT8 DeepGEMM expert paths |

The offload allocation and marker-preservation changes are one patch group:
fixing the initial allocation alone left post-load replacement able to trigger
another offload allocation.

The accepted HT clamp implementation uses `quant_config.gemm1_clamp_limit` and
the existing dpsk LightOp wrapper. The accepted LL implementation uses
`activation_config.clamp_limit`, imports the LightOp clamp kernel directly, and
passes `mask_m=expert_num_tokens`, `expect_m=max_num_tokens`.

A prior LL variant aligned to the remote W4A8 pattern used
`quant_config.gemm1_clamp_limit`, the dpsk wrapper and `expect_m=expected_m`.
That P/D attempt timed out during D initialization before graph capture or
evaluation. Restoring the old LL call passed standalone D startup and the full
P/D evaluation below. Several call details changed together, so the timeout is
not established to be caused solely by `expect_m`; no collective deadlock was
confirmed from the available logs.

## Accepted service configuration

| Setting | P: bw7 / 10.16.1.47 | D: bw6 / 10.16.1.46 |
| --- | --- | --- |
| Parallelism | TP1 PCP8 DP1 EP8 | TP1 PCP1 DP8 EP8 |
| MoE | DeepGEMM | DeepGEMM |
| All-to-all | DeepEP high throughput | DeepEP low latency |
| Execution | Eager | CUDA Graph, capture sizes 1/2/4/8/16/32 |
| Speculation | DSpark5, draft DeepGEMM | DSpark5, draft DeepGEMM |
| Expert offload budget | 32 GiB, w13/w2 | 32 GiB, w13/w2 |
| KV cache | 4 GiB/rank | 4 GiB/rank |
| Max batched tokens | 1024 | 128 |

Both services use max model length 32768 and max sequences 16. Transfer uses
Mooncake RDMA, `mlx5_0`, `MC_MTU=1024`. The D batching limit is a capacity
adaptation for the approximately 64 GiB gfx936 cards.

The DeepGEMM package version is
`2.1.0+dtk2604.torch2110.2609230001.g821965`.

## End-to-end evidence

Evaluation: HumanEval164, batch16, seed42, temperature0, top_p0.95,
thinking=false, max_tokens16384, review_timeout30.

| Check | Result |
| --- | --- |
| P to D | 160/164, 97.56%, ACCURACY_PASS |
| Direct evaluation of the same D instance | 161/164, 98.17%, ACCURACY_PASS |
| Generation and review coverage | 164 predictions and 164 reviews in each evaluation; acceptance errors empty |
| Transfer pairing | 164 matching transfer IDs; each has 8 P `p_send_kv_done` events and 1 D `d_kv_ready` event; transfer errors empty |
| Overall verdict | PD_ACCURACY_PASS, exit 0 |
| Final health and cleanup | Both final health requests succeeded; all run-owned process groups cleaned |

P/D failed tasks are 32/83/132/145. Direct-D failed tasks are 83/132/145.
The preceding no-clamp run scored 126/164 and 127/164 respectively. A TP8
DeepGEMM control also reproduced the accuracy loss without PCP or DeepEP,
and adding clamp restored approximately 96.3%. The no-DSpark D control
improved from 132/164 to 160/164 after the clamp fix.

Standalone D startup with the accepted LL call also passed DSpark5 and Graph:
8/8 graph captures, 8 API startups, and a direct smoke response of `READY`.

Artifacts live relative to the parent project directory:

- `acc/20261007_int8_pd_bw7_bw6_oldll_clampfix/`: accepted P/D run, commands,
  configuration, source hashes, source diff, predictions/reviews, accuracy
  summaries, paired-transfer evidence, metrics and cleanup.
- `acc/20261007_int8_bw6_dp8_dspark5_oldclamp_startup/`: standalone D startup
  evidence and aligned-to-old LL diff.
- `acc/20261007_int8_pd_bw7_bw6_clampfix/`: prior initialization timeout.
- `acc/20261008_w4a8_commit_checks/`: pre-commit checks and patch snapshot.

## Pre-commit checks

The tracked diff and recorded source hashes matched the accepted P/D run before
staging. Existing focused tests were rerun in the node-specific environments:

```bash
# bw7
python -m pytest -q \
  tests/runtime_patch/test_lightop_attention_api.py \
  tests/runtime_patch/test_attention_mla_fla_mamba.py \
  -k '(lightop or sparse_indexer or sparse_mla_cache) and not dcp'

# bw6, including real HCU registered-memory and packed GEMM checks
python -m pytest -q \
  tests/runtime_patch/test_int8_moe_postload.py \
  tests/runtime_patch/test_hcu_uva_buffer.py
```

Results: bw7 31 passed, 46 deselected; bw6 17 passed. Each used
`HIP_VISIBLE_DEVICES=7`, `CUDA_VISIBLE_DEVICES=7`, and `PYTHONPATH` pointing to
the sibling vLLM checkout and this repository. DCP tests were outside the
focused attention selection. `git diff --check` also passed.

This records a validated configuration, not proof that every changed line is
indispensable. Packed-INT4 W4A8, Channel-FP8 regression, long context and
performance were not part of this acceptance run.
