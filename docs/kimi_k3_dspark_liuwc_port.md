# Kimi-K3 `kimi-k3-dspark-liuwc` port to vLLM-HCU plugin

Audit date: 2026-09-28
Source branch tip: `c4fc87056e132001ed877d1155d10b9d7ec361cc`
Plugin development base: `d184d30` (`kimi-k3_speed_up_by_follow_sglang`)
Port branch: `kimi-k3-dspark-liuwc-port`

The source branch contains a Kimi-K3 implementation plus later performance
and correctness updates. The plugin already owns HCU-specific Kimi-K3 model,
KDA, routing, W4A8, and DeepEP paths. CUDA/ROCm kernels were not copied into
the plugin: their device ABI and weight layouts are backend-specific.

| Source update | Disposition | HCU plugin handling |
| --- | --- | --- |
| Kimi-K3 model/config/vision/MTP and chat parsing support (`f68f4fd`, `bca88c0`) | `already-covered` | HCU-owned model, config, renderer, and reasoning adapters are in `vllm_hcu/models/kimi_k3`, `vllm_hcu/patch`, and `vllm_hcu/runtime_compat`; contracts already live under `tests/models/kimi_k3`. |
| NMZ Triton fallback; selectable KDA Conv1D and Flash KDA prefill (`d155cdb`, `789b218`, `f5d8f36`) | `already-covered` | `kimi_gdn_linear_attn.py` has lazy external KDA backends and HCU Triton fallback; backend and numerical contracts are in the Kimi test tree. |
| FlashMLA and single-group LightOP router (`3cf8756`, `a0f3bdf`) | `already-covered` | Plugin-owned HCU FlashMLA and grouped-top-k routing paths already provide the corresponding dispatch seams and fallback checks. |
| AMD KDA/attention-residual delegation to bolt_ops (`2667ae4`, `74671e8`) | `already-covered` | Those ROCm kernels are not ABI-compatible HCU imports. The plugin has HCU-owned KDA and attention-residual implementations. |
| Rename the `bolt_ops` import package to `boltops` (`20bbcb8`) | `ported` | Updated Kimi HCU imports to the current `boltops` categorized modules. The container exposes both namespaces today; the source branch's rename avoids relying on the legacy alias. |
| Shared/routed expert TP AllReduce fusion (`0d84be0`) | `already-covered` | Kimi's `KimiRoutedOutputTransform` advertises the fused reduction capability; the plugin MoE runner consumes it and has Kimi DP/local-MLP contracts. |
| AMD AITER W4A16 (`25ef7d4`) | `no-vllm-seam` | The plugin's supported Kimi INT4 paths are W4A8 through HCU AITER/DeepGEMM/LightOP or Triton. The source branch's ROCm W4A16 ABI does not apply to those packed weights. |
| SlimQuant W4A8, AITER weight repack, TP8/TP16 fixes, DeepEP LL/HT and LightOP EP SiTU (`a3f9be6` through `39e731d`) | `already-covered` | Plugin-owned Kimi W4A8/HT/LL runtimes already cover these operations, including layout checks, SiTU selection, and categorized LightOP calls. Existing contracts are in `tests/models/kimi_k3` and `tests/accuracy`. |
| DSpark draft should ignore partial-layer weight filters (`c4fc870`) | `ported` | Added `patch_dspark_smoke_layer`, an optional exact-module callback. It wraps DSpark's captured `get_model` binding with the plugin's context-local debug-filter disable and, when present, vLLM's `disable_smoke_layer_limit()` context. |
| DSpark full-CUDA-Graph capability for non-causal multi-token MLA decode (`73e9e86`) | `already-covered` | HCU FlashMLA and sparse-SWA metadata builders report `UNIFORM_BATCH`; no second graph-capability patch was added. |

## Validation

- Ported and compatibility changes were exercised on the two-node TP16 HCU
  setup with the c4fc870 vLLM runtime overlay and this plugin branch. The
  exact environment, launch metadata, per-rank logs, and benchmark outputs are
  recorded under `task_dir/support_dspark/runs/` in the workspace.
- Both nodes passed preflight with eight HCUs each. Target checkpoint manifests
  contained 100 files per node and the draft manifests contained 2 files per
  node; all file sizes and SHA256 hashes matched.
- The 12-layer TP16 smoke became ready and served ten non-empty responses. Full
  DSpark target+draft and matched non-DSpark target runs both served ten fixed
  prompts and scored 63/64 on GSM8K.
- Three measured runs per configuration completed 32/32 requests each. Median
  throughput was 164.03 tok/s for DSpark and 93.89 tok/s for baseline (+74.71%);
  median p50 TTFT was 0.343 s and 0.571 s respectively. The measurements used
  concurrency 8 and 128 output tokens per request.
- Startup logs contain host RCCL/runtime warnings, and teardown logs contain
  process-group/resource-tracker warnings after completed measurements. No
  measured request failed, and there was no OOM or worker loss during the runs.
- Draft weights remain staged under `task_dir/support_dspark/weights/` and were
  not copied to the shared model directory. The tool-call-format evaluation
  remains unavailable because the installed runtime wheel does not register
  the `kimi_k3` tool-call parser.
