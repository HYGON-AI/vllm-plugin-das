# Custom-Op Master Follow-up Implementation Plan

**Goal:** Close the remaining safe `VLLM_HCU_USE_CUSTOM_OPS=0` routing gaps
without changing ordinary FlashAttention selection or FP8 paged-MQA decode.

**Base:** `origin/v0.28.1-dev` at `de83182b0ed267eb8f81975c28857d0bc1afe1e8`.

## Boundaries

- Sparse-indexer FP8 prefill uses BoltOPs Triton MQA when the master is off.
- FP8 paged-MQA decode is unchanged.
- Ordinary `FLASH_ATTN` is never rewritten by the master; users select
  `TRITON_ATTN` explicitly.
- Existing dense/sparse MLA behavior from MR #163 is unchanged.
- Explicit MoE backends and FP8 QSA reader/writer remain independent.
- Only AITER linear and custom-all-reduce capabilities are master-gated;
  broad AITER/MHA/MLA/MoE capability probes remain unchanged.

## Task 1: Sparse-indexer prefill routing

**Files:**

- Modify: `vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py`
- Modify: `tests/runtime_patch/test_lightop_attention_api.py`
- Modify: `tests/runtime_patch/test_sparse_indexer_loading.py`

1. Add tests proving master-off FP8 prefill calls BoltOPs and reaches neither
   AITER module lookup nor LightOp, including `force_aiter_triton=True`.
2. Add boundary coverage proving the paged-MQA path remains unchanged.
3. Run the tests and observe the prefill test fail for the current route.
4. Add a lazy, ABI-checked BoltOPs resolver. Route master-off prefill through
   `boltops.mqa_logits.triton_mqa_logits`; use the existing Torch reference only
   when BoltOPs is absent, and surface ABI drift.
5. Run the complete sparse-indexer/LightOp attention suites.

## Task 2: Safe AITER child capability gates

**Files:**

- Modify: `vllm_hcu/model_executor/layers/fused_moe/aiter_ops.py`
- Modify: `tests/patch/test_module_exchange.py`
- Modify: `tests/runtime_patch/test_quant_gemm_aiter.py`

1. Add a four-way master/child matrix for AITER linear and custom all-reduce.
2. Add exception tests proving broad AITER, MHA, MLA and fused-MoE probes are
   unchanged while the master is off.
3. Run the tests and observe the uncovered capability tests fail.
4. Gate `is_linear_enabled`, `is_linear_fp8_enabled`,
   `is_custom_all_reduce_enabled`, and `get_aiter_allreduce` with the dynamic
   master policy. Preserve signatures and cold-module replacement validation.
5. Run the AITER exchange, quantization and communicator suites.

## Task 3: Attention boundary regression

**Files:**

- Modify: `tests/runtime_patch/test_platform_hcu_config.py`

1. Add a regression test proving `CUSTOM_OPS=0` does not rewrite an explicit
   ordinary `FLASH_ATTN` backend and that explicit `TRITON_ATTN` remains valid.
2. Run it before production changes; it should already pass and therefore acts
   as a characterization/boundary test, not the RED test for Tasks 1 or 2.
3. Run the complete platform configuration suite.

## Task 4: Review and verification

1. Re-scan production LightOp/AITER/BoltOPs entry points and classify the
   intentional exceptions.
2. Run focused suites, then the complete repository `pytest` command.
3. Review the full diff against the merge base and fix important findings with
   a RED-to-GREEN regression test.

## Task 5: Hardware validation and MR

1. Validate GLM-5.3 Channel-FP8 master-off prefill route, unchanged paged
   decode, DP+EP+MTP3 graph mode, and HumanEval/0-7.
2. Regress Hy4 DP+EP+MTP3 master-off behavior.
3. Validate `/models/Qwen3.6-35B-A3B` twice on TP8: explicit `FLASH_ATTN` and
   explicit `TRITON_ATTN`, both with `CUSTOM_OPS=0`; start without MTP/graphs,
   then add graph/MTP regression if the isolated runs pass.
4. Update `/models/upgrading-vllm-hcu` with the finalized boundary matrix.
5. Rebase on the latest remote `v0.28.1-dev`, push the separate branch, open a
   separate MR, and post exact server commands and evidence in an MR comment.
