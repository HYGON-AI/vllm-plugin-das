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

## Follow-up port: RCCL hang fix (commit `877e6996`)

Port date: 2026-10-08
Source commit: `877e6996b94f2e935c3605d01c85ce4ae9bc2c47`
Plugin branch: `fix/kimi-k3-dspark-v0251-hang-lwc` (local, based on `75ed6db`)

Root cause: one NCCL communicator hosting two
collective roles — captured (CUDA-graph replay) and eager — deadlocks the RCCL
LL protocol during DSpark decode. The source branch fixes this with two
complementary mechanisms: an auxiliary eager-phase TP all-reduce communicator
(`VLLM_ROLE_SEP`) and moving the DSpark draft-logits all_gather out of the
captured draft graph.

| Source concern | Disposition | HCU plugin handling |
| --- | --- | --- |
| Role separation module (`role_separation.py`, aux eager-AR comm) | `ported` | HCU-owned `vllm_hcu/models/kimi_k3/role_separation.py`, same env contract (`VLLM_ROLE_SEP`, `VLLM_ROLE_SEP_STRICT`), instance-level swap of
`GroupCoordinator._all_reduce_out_place` (name verified on the paired
v0.25.1 runtime). |
| Scope around the target LM forward (`kimi_k3/amd/linear.py`) | `ported` | `eager_ar_scope()` wraps `KimiLinearForCausalLM.forward` in `vllm_hcu/models/kimi_k3/amd/linear.py` (plugin-owned model), capture/compile no-op preserved. |
| Scope around the draft forward (`kimi_k3/nvidia/dspark_mla.py`) | `adapted` | The paired runtime has no `dspark_mla.py` for K3; the plugin owns the draft model, so `eager_ar_scope()` wraps `K3DSparkForCausalLM.forward` in `vllm_hcu/models/kimi_k3/dspark.py`. |
| `compute_draft_logits_shard` API | `adapted` | Added to `K3DSparkForCausalLM`. The source runtime's `LogitsProcessor._apply_head` (GEMM-only) does not exist in the paired wheel — 0.25.1 predates the `_apply_head`/`_get_logits` split, and its `_get_logits` appends the gather — so the shard projection dispatches on the runtime seam via `apply_lm_head_shard` (`_apply_head` when present, else the raw `quant_method.apply` GEMM that `_get_logits` gathers around); contract-tested both seams. |
| Draft-logits AG out of the captured graph (speculator changes) | `adapted` | Interposed at worker phase by `patch_dspark_role_separation` (framework_opt): `load_draft_model` installs shard state or falls back with the strict-guard report; `_sample_sequential` branches on `torch.cuda.is_current_stream_capturing()` (the capture driver enqueues with `CUDAGraphMode.NONE`); the eager tail hooks `query_cudagraph_manager.run_fullgraph` (inherited DFlash entry) with the capture-driver padding rule. The in-graph Markov loop is reproduced verbatim from the paired runtime's `_sample_sequential` (snapshot recorded in the patch). |
| Tail all_gather docking point | `deviated` | The source branch leaves the eager tail on `tensor_model_parallel_all_gather` (c10d): in ITS runtime every c10d gather is eager, so the comm stays single-role. The paired wheel CAPTURES the target decode graph's own logits all_gather on c10d, making c10d a captured-role comm there — docking the eager tail on it re-creates the captured-x-eager same-communicator pair this commit exists to kill (incurred the 2026-10-09 08:21 serving hang on nmz20/nmz22, hipgdb: TP odd ranks spinning inside `runTreeUpDown<bf16, ProtoSimple>` AR while even ranks sat idle). The tail therefore publishes through `role_separation.eager_all_gather`, which rides the auxiliary eager communicator (single role: eager AR + eager AG), falls back to the legacy c10d gather only when `VLLM_ROLE_SEP=0` or while capturing (the caller's capture guard owns that case). Contract-tested (`test_tail_all_gather_docks_on_aux_comm_not_c10d`, `test_eager_all_gather_*`). |
| Capture guard placement | `realigned` | Fork guards capture inside `_generate_draft` (sees `cudagraph_runtime_mode`); the paired runtime exposes only `_sample_sequential`, so the guard fired only for the shard-enabled branch. The unavailable-shard branch now reports the too (fork's `_generate_draft` else-branch parity). |
| Plugin log visibility | `added` | vLLM attaches handlers to the `vllm` logger only, so plugin INFO (aux comm construction, shard enablement — the operator's validation signals) never reached serve logs. `vllm_hcu/__init__.py` bridges the `vllm_hcu` logger onto vLLM's handlers (idempotent). |

Runtime pairs verified against the installed vLLM 0.25.1 wheel
(`speculator.py` md5 `bf382b4ea89487bbf7b27e5fa82da9bf`):
`compute_draft_logits` has exactly one call site (`_sample_sequential`); the
draft graph capture replays through `dflash/speculator.py:402`
(`run_fullgraph`); `load_dspark_model` aliases the target lm_head
(`has_own_lm_head = False`), so the shard width is
`num_embeddings_per_partition` with a non-replicated head.

## Follow-up validation

- Contract tests: `tests/patch/test_kimi_dspark_role_separation.py`
  (interposition, idempotence, feature gating, shard install/fallback,
  capture bypass, eager replay tail, strict-mode guard) run CPU-only.
- Dispatcher inventory, patch-coverage, production-boundary, and HCU
  registry static gates must be re-run on this branch.
- Runtime/parity stages (two-node TP16 DSpark run, GSM8K parity, kernel and
  trace comparison, NCCL_DEBUG A1 comm check) are pending a maintenance
  window; this port must not be described as runtime-validated until then.
