# LightOp categorized API validation

Validation date: 2026-09-01 UTC

## Environment

- Repository: `/models/.worktrees/vllm-plugin-das-lightop-no-lmslim`
- Repository-test vLLM source: `/models/zb/vllm_025/vllm`
- Live-HCU vLLM root: `/usr/local/lib/python3.10/dist-packages` (the matching
  installed binary package and compiled extensions)
- Python: 3.10.12
- pytest: 9.1.1
- torch: `2.11.0+das.opt1.dtk2604.202604021232.g1175f0`
- vLLM: `0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a`
- LightOp: `0.6.0+das.dtk2604.torch2110.2608171227.g8c835c`
- AITER: `0.1.5+das185.dtk2604.torch2110.2608180853.g40a705`
- DeepGEMM: `2.1.0+das185.dtk2604.torch2110.2608171132.g493d80`
- LMSlim: `0.3.1+das.opt4.dtk2604.torch2110.2608171800.gf9a687`
  remains installed but is not imported or called by plugin production code.
  The Docker `das-install lmslim` line is outside this change by design.
- `vllm-plugin-das` is exercised from the worktree and is not installed as a
  distribution in this environment.

At 2026-09-01T02:25:38Z all eight BW1100 devices were owned by an unrelated
DP8 vLLM job at approximately 143.6--144.0 GiB of 147.4 GiB VRAM each. No
process was signalled or terminated. At 2026-09-01T02:45:34Z physical devices
0, 1, 5, 6, and 7 had returned to their approximately 6.1 GiB baseline while
devices 2--4 remained occupied. The live suites selected physical device 7
with `HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7`.

## Boundary TDD and residual audit

The final production scan passed immediately after the scanner was written,
so a copied temporary production fixture was mutated with `import lmslim`.
The targeted RED command was:

```text
python -m pytest -q tests/patch/test_lightop_api_boundary.py::test_scanner_rejects_external_lmslim_with_location
```

It exited 1 with one failed test and the exact detected diagnostic
`vllm_hcu/mutation.py:1: external LMSlim import 'lmslim'`. The permanent
mutation test now also proves file-and-line diagnostics for `lightop.op`,
`lightop.gemmopt`, a moved package-root import, and an aliased package-root
attribute call. The final command exits 0 with 3 passed:

```text
python -m pytest -q tests/patch/test_lightop_api_boundary.py
```

The final scanner audit reports:

```text
violations: []
categorized symbols: 30
allowed top-level calls: 2
```

The two exceptions are present exactly once, and only in
`vllm_hcu/model_executor/layers/fused_moe/experts/dpsk_v4_deep_gemm_moe.py`:

- `fuse_silu_mul_clamp_quant`
- `fuse_silu_mul_clamp_quant_ep`

The structural guard and call-count checks reject a missing, duplicated,
stale, or broadened exception. The 30 categorized production symbols are:

```text
lightop.activation.fuse_silu_mul_fp8_quant
lightop.activation.fuse_silu_mul_fp8_quant_ep
lightop.activation.fuse_silu_mul_per_token_quant
lightop.activation.fuse_silu_mul_quant
lightop.activation.fuse_silu_mul_quant_ep
lightop.activation.silu_and_mul_opt
lightop.attention.fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32
lightop.attention.mqa_logits
lightop.attention.paged_mqa_logits
lightop.attention.split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant
lightop.attention.top_k_per_row_decode
lightop.attention.top_k_per_row_prefill
lightop.gemm_ops.hipblaslt_w8a8_gemm
lightop.gemm_ops.m_grouped_w8a8_gemm_nt_contig_asm
lightop.gemm_ops.m_grouped_w8a8_gemm_nt_masked
lightop.moe.ep_gather
lightop.moe.ep_scatter
lightop.moe.fused_experts_impl_fp8_marlin
lightop.moe.fused_experts_impl_int8_marlin
lightop.moe.moe_align_block_size_out
lightop.moe.moe_fused_gate
lightop.norm.fused_add_rms_norm
lightop.norm.gemma_fused_add_rmsnorm
lightop.norm.gemma_rmsnorm
lightop.norm.rms_norm_dynamic_per_token_quant
lightop.norm.rmsnorm_forward_autograd
lightop.quant.per_token_quant_fp8
lightop.quant.per_token_quant_int8
lightop.sampling.top_k_top_p_sampling_from_probs
lightop.tensor.ds_cat
```

Every symbol is in the installed category module's public `__all__`.
`lightop.gemm_ops.hipblaslt_w8a8_gemm` is also pinned explicitly by the
installed export contract. The audit scans only this repository's
`vllm_hcu` package; it intentionally does not inspect or reject LightOp's own
internal implementation namespaces.

## Final verification

All commands below exited 0.

| Command | Result |
| --- | --- |
| `git diff --check origin/v0.25.1...HEAD` | clean |
| `python -m compileall -q vllm_hcu tests` | clean |
| `python tools/check_production_boundary.py` | 248 Python files, clean |
| `VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm python tools/check_patch_test_coverage.py` | 96 files: 95 adapters, 1 helper, 0 untested, 0 invalid contracts |
| `python -m pytest -q tests/patch/test_lightop_api_boundary.py` | 3 passed |
| `python -m pytest -q tests/runtime_patch/test_lightop_categorized_api.py` | 8 passed |
| `VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm python tools/run_patch_tests.py --suite contract` | 1,199 passed, 82 deselected, 15 warnings in 332.99s |
| `VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm python tools/run_patch_tests.py --suite integration-smoke -- -rs` | 73 passed, 3 skipped, 56 deselected in 3.62s |
| `HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 env -u VLLM_V0251_SOURCE_ROOT python tools/run_patch_tests.py --suite accuracy-hcu -- -k 'lightop or int8 or deepseek_v4 or dspark' -rs` | 41 passed, 98 deselected, 14 warnings in 24.24s |
| `HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 env -u VLLM_V0251_SOURCE_ROOT python tools/run_patch_tests.py --suite contract-hcu -- -k 'lightop or moe_align or deepseek_v4' -rs` | 7 passed, 1,135 deselected, 14 warnings in 22.20s |
| `VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm python -m pytest -q tests/patch/test_hcu_ci_selector.py` | 44 passed |
| `python .github/scripts/hcu_ci/verify_hcu_registration.py` | 40 registrations across 22 jobs; valid |

The contract total remains above the documented 1,162-pass baseline. Its 15
warnings are one source-checkout `vllm._version` runtime warning and 14
third-party `torch.jit.script_method` deprecation warnings. Each live-HCU
suite reports only the same 14 Torch deprecation warnings.

The three integration-smoke skips are not counted as passes:

- `tests/integration/graph/test_qwen35_9b_graph_parity.py::test_qwen35_9b_eager_graph_token_parity` — Qwen3.5-9B graph model path unavailable at `qwen3.5/Qwen3.5-9B`.
- `tests/integration/lora/test_qwen3_4b_lora_switching.py::test_qwen3_4b_lora_adapter_switching` — Qwen3-4B LoRA base model path unavailable at `qwen3/Qwen3-4B`.
- `tests/integration/models/test_qwen35_9b_smoke.py::test_qwen35_9b_greedy_generation_smoke` — Qwen3.5-9B model path unavailable at `qwen3.5/Qwen3.5-9B`.

There were no skips in either selected live-HCU suite.

## Diagnosed non-final runs

The first full contract run exited 1 with 17 failed, 1,211 passed, 53
deselected, and 15 warnings. Four failures exposed cached real LightOp modules
from the new export probe; three DeepGEMM fixtures still modeled the removed
root API, four EPLB cases exposed an accidentally removed `SimpleNamespace`
test import, and nine live unified-AITER cases lacked the `hcu` marker. A
focused regression run after the corrections passed 12 with 12 correctly
deselected. A later selector run intentionally failed because the newly
HCU-marked export probe was not yet in the literal CI registry; the existing
registration invariant covered that defect before its registry/map fix.

The first selected accuracy-HCU run used the repository source checkout and
reported 38 passed and 3 failed. The failures were confined to the upstream
vLLM reference, whose source checkout has no compiled `_C.silu_and_mul` or
`_moe_C.moe_align_block_size` extensions. Repeating the unchanged test
selection against the installed vLLM binary root produced the final 41-pass
result above.
