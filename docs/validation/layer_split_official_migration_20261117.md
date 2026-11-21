# Layer Split migration to official 0.33

## Subsequent FP4 composition

The follow-up [FP4 + Layer Split integration](fp4_layer_split_20261117.md)
implements FP4 cache-owner writes, shared main/SWA replay semantics and
owner-aware PD admission. It supersedes the historical FP4-combination refusal
below. CPU checks and pending device qualification are documented separately.

## Official refresh on 2026-11-17

This section supersedes the first migration's baseline and admission notes below.
The first migration's results remain historical evidence for their recorded SHAs.

- Adapter official main: `4b9ea6d761a67ddc677414d4c50a74b83ba46e01`.
- Plugin official v0.33.0: `89ee352c671d49fd1702029d6388caeb2919a434`.
- Both feature branches remain `codex/layer-split-official-20261117`.
- The previous heads are preserved locally and on the server under
  `codex/layer-split-before-refresh-20261117` (Adapter `40c9a26`, Plugin `75ba86c`).
- Worker conflict: retain official `_make_model_runner` and Plugin-delegating
  `initialize_from_config`; neither lifecycle entry replaces the other.
- Official PRs #125/#426 are now included. Explicit PCP `deep_gemm` now admits
  SWA bounded replay, while `deep_gemm_mega_moe` still rejects it. The constructor
  regression now checks both policies. The earlier recommendation to use `auto`
  to avoid the DeepGEMM replay rejection no longer applies to this baseline.
- Layer Split's FP8 scope is unchanged. FP4 PCP without Layer Split is preserved;
  the combination FP4 + Layer Split remains explicitly refused pending its own
  implementation and device validation. No official guard is weakened.

Refresh validation (installed Core unchanged; GPUs hidden):

- `validation/refresh1_cpu`: **247 passed**, 13/13 PCP source replays passed,
  CUDA not initialized. Tested Adapter `3d2ae32`, Plugin `93883ee`.
- `validation/refresh1_upstream.log` / `.xml`: **163 passed, 1 failed, 1 skipped**.
  Covers official FP4/PCP workspace, SWA profile worker, MoE admission, runner
  factory/fork and Worker lifecycle. Failure is
  `test_dsv41_megamoe_cpu.py::test_native_provider_signature`: installed private
  DeepGEMM lacks `deepgemm.mega`. Its production file and test match the official
  target exactly; no guard/test was removed and no vendor package was replaced.
  The skip is the optional exact-checkout provenance test (no Core source root
  supplied); installed Core provenance is recorded separately in environment.json.
- `validation/refresh1_l0.log`: Adapter L0 GREEN (32 warnings).
- This refresh does not qualify MegaMoE on the current container or any GPU
  execution of the rebased Layer Split feature. No GPU service was started.

## First migration (historical)

## Source mapping

| Repository | Official target / base | Legacy feature |
|---|---|---|
| HYGON-AI/adapter-das | main / e1aea4667954a9f6980f8f1692c21c151dc75448 | 21b36002c3606a5cf958d5288ca7349c881d5f29 |
| HYGON-AI/vllm-plugin-das | v0.33.0 / 8b47c0284fe917b7a368cd5b4faa2705f60efa5e | 8b21a1a245b42a037e5ce4ecf37f6ed2eefdb4f1 |

Branches: `codex/layer-split-official-20261117`. Original working trees and
uncommitted changes were preserved. This migration does not include the separate
FP4 PCP PRs #125/#426 and does not modify official Core.

## Ownership and conflict decisions

| Component | Owner / decision | Call sites and regression |
|---|---|---|
| Native PCP admission, preparation, local scoring and empty-query handling | Adapter `kernels/pcp_fork.py`; keep official implementation | Native decoder selector and real attention base; existing PCP tests/replay |
| Main/SWA KV owner writes and broadcasts | Plugin `v41_pcp_layer_split.py`; replace old full PCP class with subclass | Model selector, indexer, `forward_kv`, `forward_mqa`, cache specs; owner/replay and empty-rank tests |
| Independent pools, scheduler, physical views | Plugin; retain feature | Adapter platform builder/worker hooks; pool/allocator/scheduler CPU tests |
| Generic KV builder/localization registry | Adapter; replay legacy commit | Platform and Worker initialization; registry contracts; official profile Worker inherits this hook |
| Transfer admission | Adapter optional owner-bound validator; default still refuses | Both base class constructors call the same validator; no config-field masking or bypass of eager/PCP guards |
| Mooncake owners protocol v3 | Plugin; retain feature | Only exact connector module/name and producer role admit PCP transfer; PD transport CPU tests |
| Model publication and profile worker | Plugin; merge both official profile policy and Layer Split policy | Activation, immutable health checks and compilation factor; existing publication tests |
| Stock/kill switch and FP4 | Preserve rollback; reject incompatible Layer Split requests | Actual model selector and activation; explicit negative tests |

Layer Split is opt-in with `vllm serve --enable-layer-split`,
`VLLM_HCU_DSV41_PCP_ATTENTION=1`, `HCU_VLLM_PCP_IMPL=native`.
The indexer remains FP8. TP/DP/PP/DCP=1, PCP>1, eager, no dSpark remain
the migrated scope. SWA bounded decode replay and owner-aware transfer retain
their original independent configuration; migration does not enable them by default.
Source-write interception is installed only when selecting Layer Split attention,
not as a side effect of importing the base PCP module.

## Validation boundaries

Run `tests/vllm033/run_layer_split_migration_cpu.sh` against the installed Core
wheel with GPUs hidden. It records exact commits and import paths, pytest/JUnit,
13 PCP source replays and **theoretical** pool capacity separately.
Device kernels and transport in these tests include CPU doubles; this does not
qualify GPU performance, GPU replay accuracy or live Mooncake RDMA.

The historical validation uses the installed Python 3.12 and Core wheel in the
Shenzhen validation container.
GPU service launch is outside this migration's current validation run.

Official PCP8/EP8 Channel-FP8 + DeepEP HT now routes explicit `deep_gemm` to a
new provider whose admission rejects SWA bounded replay (also true for
`deep_gemm_mega_moe`). This migration retains that guard. For a later replay
trial, explicit `--moe-backend auto` selects the retained Channel-FP8 HT method;
the CPU constructor test checks that selection and preserves rejection of the
explicit new provider. This is a candidate command change, not GPU qualification.
`auto` also falls outside the new profile worker's scoped route; remeasure the
actual GPU memory budget instead of inheriting the newer provider's capacity.

The older validation documents retained by rebase describe **historical** runs:
legacy Layer Split+Replay PD scored 157/164, versus historical OFF/OFF 159/164.
Equivalence and single-switch attribution remain unresolved. The separate Shenzhen
FP4+SWA run scored 160/164 and did not exercise this Layer Split feature.
None of those results is evidence that this migrated revision passed a GPU run.

## Recorded CPU results

- `validation/cpu_01`: 140 passed / 2 failed; both old tests addressed the base
  PCP module instead of the new Layer Split subclass. The failing logs remain.
- `validation/cpu_02`: 244 pytest passed, 13/13 PCP source replays passed;
  CUDA not initialized. Adapter `fafd790`, Plugin `04cd3a3`.
- `validation/registry_final.log`: 5 passed, including one additional clean
  subprocess that blocks framework imports during registry import and dispatch.
  Adapter `40c9a26`; only tests/docs changed after the main run.
- `validation/migration_final.log`: 10 passed, including one additional real
  MoE constructor admission test (auto HT / explicit DeepGEMM replay refusal).
  Plugin `9be33da`; only tests/docs changed after the main run.
- Total unique pytest cases: 246; rerun cases are not counted twice.
- Adapter L0: GREEN, 32 warnings; full output in `validation/adapter_l0.log`.
- Installed Core: `0.33.0+das.dtk2610.gfx936gfx938.g5afd482`; Torch
  `2.11.0+das.opt1.dtk2610.2609201632.gf74cd7`, Python 3.12. Import paths and
  Triton version are recorded in `validation/cpu_02/environment.json`.

Final production source matches the main CPU run; later commits only add tests
and documentation. No GPU serving, model accuracy rerun, push or PR update was
performed. The old FP4 validation container was used only for CPU commands with
all three GPU visibility variables empty; its installed packages were untouched.
