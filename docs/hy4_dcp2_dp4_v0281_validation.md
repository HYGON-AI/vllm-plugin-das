# Hy4 DCP2 + DP4 validation on v0.28.1-dev

Date: 2026-09-26

## Scope

This change adds the exact Hy4 topology TP2/DCP2/PCP1/PP1/DP4 with EP8,
Model Runner V2, FP8 E4M3 KV cache, MTP3, and non-eager CUDA Graph. It does
not broaden support to arbitrary DCP/DP combinations, BF16 DCP, query
replication, PCP+DCP, or other query-head envelopes.

The branch starts at `v0.28.1-dev` commit
`293b3ee74d752ee4dbd64aad14ba084ea0e5c5ec`. The implementation commits are:

- `ca0cb84`: admit the exact topology and its 32-local/64-gathered query-head
  envelope.
- `54404c9`: retain the mixed-batch FP8 sparse-MLA path at the 32-head
  boundary.

The v0.25.1 implementation was checked as a historical design reference. It
provides generic DCP/DP process groups but no Hy4-specific configuration gate
or direct validation for this topology, so the current support is implemented
and tested against the v0.28.1-dev ownership boundaries.

## Root cause and fix

Hy4 previously accepted only TP8/DCP2/DP1, whose sparse-MLA query shape is 8
local heads and 16 heads after the DCP gather. TP2/DCP2/DP4 is valid because
each DCP group remains inside one DP replica, but it has 32 local and 64
gathered heads.

The first candidate passed configuration validation and then failed during
CUDA Graph memory profiling. Upstream
`FlashMLASparseMetadataBuilder.__init__` selects its FP8 mixed-batch path only
when `num_heads < 32`; exactly 32 heads entered the separate prefill/decode
path, which explicitly rejects DCP because that path returns LSE only for
decode tokens.

The plugin now admits only the exact Hy4 TP2/DCP2/DP4/EP configuration and
keeps it on the mixed-batch path when the local and gathered query counts map
to the same supported FP8 kernel envelope. For this model, 32 and 64 query
heads both pad to 64. All near-miss topologies and unrelated upstream errors
continue to use the upstream behavior.

## Isolated artifact

The hardware run imported both `vllm_hcu` and `hcu_ops` from the isolated
install of:

`vllm_hcu-0.28.1rc1.dev491+das.54404c9.dtk2604-cp310-cp310-linux_x86_64.whl`

SHA-256:
`9c34be221cc02a07f578cb3f1fcf4ecaf03155ff5dcb2a54bfd3890cd8628fd7`

The paired framework was installed vLLM
`0.28.1+das.77acaf6.dtk2604`.

## Server command

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_USE_NN=1 \
VLLM_DCP_Q_REPLICATE=0 \
VLLM_HCU_USE_CUSTOM_OPS=1 \
VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM=1 \
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
PYTHONNOUSERSITE=1 \
env -u VLLM_PLUGINS \
  -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  /usr/bin/python -m vllm.entrypoints.cli.main serve \
  /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --moe-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --max-model-len 4096 \
  --block-size 64 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --seed 0 \
  --data-parallel-size 4 \
  --decode-context-parallel-size 2 \
  --dcp-comm-backend ag_rs \
  --cp-kv-cache-interleave-size 1 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

For an isolated wheel install, prepend its directory to `PYTHONPATH` as was
done during this run.

## Results

- Four DP engines created eight TP/DCP workers, and the log identified all
  workers as Model Runner V2.
- DCP, DP, and EP process groups initialized; EP world size was eight.
- The target captured 19 PIECEWISE CUDA Graph sizes. The MTP speculator then
  captured its model and prefill PIECEWISE graphs on all eight workers.
- The API reached ready state. `/health` returned HTTP 200 before and after
  the requests.
- Ordered HumanEval/0-7 requests all returned HTTP 200 with
  `finish_reason=stop`; EvalScope scored 8/8.
- AITER had no tuned solution for some profiled MoE shapes and reported its
  existing Triton fallback. This run therefore proves the requested AITER
  selection policy, not that every MoE shape executed an AITER kernel.
- After shutdown, no owned process remained and all eight cards reported 0%
  memory and compute utilization.

Final focused Hy4, sparse-MLA adapter, and launcher coverage passed 275 tests.
The full-repository first-failure run passed 209 tests before reproducing the
known W4A8 shared-storage numerical baseline failure in
`test_auto_w4a8_shared_storage_feeds_ht_and_ll_with_empty_expert`; its mismatch
counts and maximum error match the prior merged-branch record. The result and prediction files
are retained under
`.superpowers/sdd/2026-09-26-hy4-dcp-dp4-v0281/hardware/` in this worktree.
