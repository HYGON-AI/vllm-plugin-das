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

The clamp exception is accepted only when `_lightop_clamp` is the exact cached
resolver: one `name` argument, one exact membership guard whose sole body is
`raise AttributeError(name)`, one exact `import lightop`, and one exact
`return getattr(lightop, name)`. Extra statements, branches, assignments,
decorators beyond the required one, lookups, missing calls, duplicated calls,
and broadened/stale names are rejected; the one exact `lru_cache` decorator is
required.
`_lightop_activation` has an equally exact categorized resolver
shape; every call to it must pass one literal symbol. Literal category
`getattr` calls are recorded, while unsupported dynamic category lookups are
rejected. The 30 categorized production symbols are:

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
The direct installed export contract validates each required module's
`__all__` as a string-only, duplicate-free list/tuple, requires every pinned
name to be a member, and separately requires every pinned name to remain a
bound module attribute. This permanently pins exact
`lightop.gemm_ops.hipblaslt_w8a8_gemm`. The audit scans only this repository's
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
| `python -m pytest -q tests/patch/test_lightop_api_boundary.py` | 10 passed after fix round 1 |
| `python -m pytest -q tests/runtime_patch/test_lightop_categorized_api.py` | 9 passed after fix round 1 |
| `VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm python tools/run_patch_tests.py --suite contract` | 1,199 passed, 82 deselected, 15 warnings in 332.99s |
| `python tools/run_patch_tests.py --suite contract` | 1,207 passed, 82 deselected, 14 warnings in 310.54s against the installed vLLM root after the model-evidence commit |
| `VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm python tools/run_patch_tests.py --suite integration-smoke -- -rs` | 73 passed, 3 skipped, 56 deselected in 3.62s |
| `HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 env -u VLLM_V0251_SOURCE_ROOT python tools/run_patch_tests.py --suite accuracy-hcu -- -k 'lightop or int8 or deepseek_v4 or dspark' -rs` | 41 passed, 98 deselected, 14 warnings in 24.24s |
| `HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 env -u VLLM_V0251_SOURCE_ROOT python tools/run_patch_tests.py --suite contract-hcu -- -k 'lightop or moe_align or deepseek_v4' -rs` | 7 passed, 1,135 deselected, 14 warnings in 22.20s |
| `VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm python -m pytest -q tests/patch/test_hcu_ci_selector.py` | 44 passed |
| `python .github/scripts/hcu_ci/verify_hcu_registration.py` | 40 registrations across 22 jobs; valid |

Both contract totals remain above the documented 1,162-pass baseline. The
source-checkout run's 15 warnings are one `vllm._version` runtime warning and
14 third-party `torch.jit.script_method` deprecation warnings; the final
installed-root run and each live-HCU suite report only the same 14 Torch
deprecation warnings.

The three integration-smoke skips are not counted as passes:

- `tests/integration/graph/test_qwen35_9b_graph_parity.py::test_qwen35_9b_eager_graph_token_parity` — Qwen3.5-9B graph model path unavailable at `qwen3.5/Qwen3.5-9B`.
- `tests/integration/lora/test_qwen3_4b_lora_switching.py::test_qwen3_4b_lora_adapter_switching` — Qwen3-4B LoRA base model path unavailable at `qwen3/Qwen3-4B`.
- `tests/integration/models/test_qwen35_9b_smoke.py::test_qwen35_9b_greedy_generation_smoke` — Qwen3.5-9B model path unavailable at `qwen3.5/Qwen3.5-9B`.

There were no skips in either selected live-HCU suite.

## Qwen3.5 W8A8 model validation

At 2026-09-01T03:30:14Z, a fresh read-only device check reported all eight
BW1100 devices at 2 MiB used with no KFD PIDs. Port 18012 accepted a local
bind probe, so the validation selected physical device 7 and retained the
planned port. The server was started from this worktree with captured PID
782837:

```text
export SELECTED_HCU=7
export PLUGIN_ROOT=/models/.worktrees/vllm-plugin-das-lightop-no-lmslim
export HIP_VISIBLE_DEVICES="$SELECTED_HCU"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_GLOBAL_MOE_CACHE=1
export VLLM_CACHE_ROOT=/tmp/vllm-cache-qwen35-lightop
export PYTHONPATH="${PLUGIN_ROOT}"

vllm serve /models/Qwen3.5-35B-A3B-W8A8 \
  --served-model-name qwen35-int8-lightop \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --quantization slimquant_marlin \
  --port 18012
```

The engine loaded 14 checkpoint shards and 36.7 GiB of weights, completed
startup, and returned HTTP 200 for `/health`. The prescribed temperature-zero
completion request also returned HTTP 200 with `finish_reason="stop"`, 9
prompt tokens, 39 completion tokens, and this non-empty text:

```text
Interface stability is crucial because it ensures that changes to a system's
internal implementation do not disrupt dependent components, thereby
maintaining system reliability and reducing the cost of software evolution.
```

The runtime route is explicit in the server log:

- line 11 records the requested `quantization='slimquant_marlin'`;
- line 29 resolves it to `quantization=slimquant_compressed_tensors_marlin`;
- lines 136--137 show LightOp successfully loading the Marlin W8A8 MoE UP and
  DOWN code objects from `lightop/hsa/gfx938/moe_w8a8_channel`;
- lines 926--929 record completed application startup, two successful health
  requests, and the successful completion request.

The sole case-insensitive `lmslim` log match is line 135 under LightOp's own
internal `lightop._lmslim_native.vllm_compat` implementation namespace. It is
not an import or external LMSlim fallback by plugin production code and is an
explicit non-goal of this migration. The error/traceback scan found 47 repeated
Torch Dynamo metrics-only serialization reports, all ending with
`TypeError: Object of type function is not JSON serializable`; model startup,
health, and generation continued successfully afterward. There was no LightOp
ABI, kernel, dtype, shape, device, or external LMSlim fallback failure.

Of the server processes, only captured API server PID 782837 was directly sent
SIGTERM. Its engine child shut down, the API completed application shutdown,
and the post-run process check found neither PID. Physical device 7 returned
to 2 MiB used. Runtime evidence is at
`/tmp/vllm-hcu-lightop-qwen35/server.log` and
`/tmp/vllm-hcu-lightop-qwen35/response.json`.

## Diagnosed non-final runs

The first model health client inherited `ALL_PROXY=http://127.0.0.1:2097` and
held its pre-start request open even after the server became healthy. A second
bounded probe established HTTP 200; only the task-owned stuck curl PID 782840
was terminated, after which the original loop retried successfully. The server
remained PID 782837 throughout, and port 18012 did not require replacement.

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

## Review fix round 1

Permanent mutations now cover clamp-name reassignment, an extra post-guard
branch, a broadened guard, a stale missing clamp caller, a nonliteral
activation-resolver call, an unsupported dynamic category lookup, and a
literal category lookup that must appear in the categorized-symbol audit. The
export test also covers both failure directions: bound-but-not-public and
public-but-not-bound.

The pre-fix focused run exited 1 with 5 failed and 13 passed. The failures were
the two resolver-shape evasions, nonliteral activation call, dynamic category
lookup, and missing public-export assertion helper. After the fixes:

```text
python -m pytest -q tests/patch/test_lightop_api_boundary.py
10 passed in 3.10s

python -m pytest -q tests/runtime_patch/test_lightop_categorized_api.py
9 passed in 2.80s

VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm python tools/run_patch_tests.py --suite contract -- -k 'lightop or hcu_ci or deep_gemm_utils'
120 passed, 1169 deselected, 15 warnings in 56.87s
```

The affected HCU selector remains at 44 passed, and the registry remains valid
with 40 registrations across 22 jobs.
