# Hy4 combined PR #152 integration validation (2026-09-26)

## DCP #154 integration candidate

The user requested folding DCP #154 into #152 after PCP #156. The integration
candidate is `17f1f2baf1b23f2b975098efd97130e729fb9a2e`, merging #152 head
`085a589` into #154 without rewriting either branch's history. The four
conflicts preserve current PCP documentation, both test families, the PP2
layer-41 producer check, and a union of separately supported PCP/DCP
topologies. Combined PCP+DCP remains rejected.

The complete 21 changed-test-file set passed **551 tests, 1 skipped** before
commit and again after independent review (142.43 seconds). Independent review
found no Critical/Important code blocker. The full suite remains non-green:
its unconfigured legacy source-root check fails during collection; with the
installed source root supplied, 209 tests pass before the same historical
`test_auto_w4a8_shared_storage_feeds_ht_and_ll_with_empty_expert` failure
(1,572,632 mismatches, max abs 2,211,653.5, max relative 15.082396507263184).

The clean candidate plugin wheel is
`vllm_hcu-0.28.1rc1.dev491+das.17f1f2b.dtk2604-cp310-cp310-linux_x86_64.whl`,
SHA-256 `5f88edbe0120d3d82c1027752195daf72c06b09c4bc9697f7876ba11f60bc515`.
It is paired with the same isolated vLLM wheel described below, SHA-256
`2d6b392dcff0d5064c754e838d9ac5ea3ffd5d6c7cac86ed2b5b1b2811cb9b3b`.
Both Python packages and compiled `hcu_ops` resolve under their respective
isolated installs. No global installation is modified.

New evidence directory `D`:
`/models/vllm-plugin-das/.worktrees/feat-hy4-dcp-pcp-v0281/.superpowers/sdd/2026-09-26-hy4-154-into-152`.
It retains the launcher, original completions, scoring reports, server logs,
metrics, provenance JSON and complete committed-tree test output.

Fresh DCP2/TP8/EP8 FP8 E4M3 target-only and MTP3 each passed HumanEval/0–31
**32/32**, with all finish reasons `stop`, under default non-eager
`PIECEWISE` Graph. Both constructed eight MRV2 workers and captured Graphs.
Three concurrent MTP3 requests additionally passed **3/3**; accepted/drafted
counts for /0, /3 and /7 were 130/135, 38/42 and 81/81. Their respective
fourth request-local steps accepted `[3, 1, 3]`; no global-step identity is
claimed. The log recorded three running requests, final health was HTTP 200,
and each owned service fully released all eight cards before the next launch.
Raw text matched the old standalone DCP wheel in 22/32 target and 23/32 MTP3
tasks; these are functional passes, not byte-identical-generation claims.

The same paired-wheel candidate also passed these fresh eight-request
concurrent regressions. Every response finished with `stop`, every final
health check returned HTTP 200, and each log records eight MRV2 workers.

| Mode | Resolved Graph | HumanEval | Accepted tokens by draft position | Evidence prefix under D |
| --- | --- | ---: | --- | --- |
| DP8/TP1/EP8, MTP3, E4M3, DeepEP LL/DeepGEMM | FULL_AND_PIECEWISE | 8/8 | 274/264/249 | `dp8_fp8` |
| DP2/TP4/EP8, MTP3, BF16 KV, DeepEP LL/DeepGEMM | FULL_AND_PIECEWISE | 8/8 | 275/264/250 | `dp2_tp4_bf16` |
| TP8, MTP3, E4M3, AITER selector | FULL_AND_PIECEWISE | 8/8 | 279/266/252 | `tp8_fp8` |
| TP4/PCP2/EP8, MTP3, BF16 KV, AITER selector | NONE (eager) | 8/8 | 277/270/256 | `tp4_mtp3` |
| PP2/TP1/PCP4/EP4, MTP2, E4M3, DeepEP HT/DeepGEMM | NONE (eager) | 8/8 | 357/342 | `pp2_mtp2` |

DCP prefixes are `dcp_graph_target` and `dcp_graph_mtp3`; the latter's metrics
sum to 1201/1139/1065 accepted tokens by draft position across its sequential
and concurrent requests. All seven modes retain metrics snapshots. The PP2
command uses `VLLM_PP_LAYER_PARTITION=41,37`. Source `serve_pair.py` and the
first JSON line of each server log retain the exact argv, isolated import
paths and separate cache roots; the launch commands are also posted to #152.
All launches explicitly enable `VLLM_USE_NN=1`; none overrides the default
Graph policy, except the two PCP modes' required `--enforce-eager`.

The AITER selector still uses per-shape Triton fallback where tuning is
absent. These passes do not certify every AITER kernel shape, PCP+Graph,
PCP+DCP, BF16 DCP, other DCP topologies, full-dataset accuracy, or a per-call
native writer trace. Prior no-DCP and PCP target-only/32-sample gates below
remain historical evidence, not reruns on this new DCP integration wheel.

The independent review and planned live gates are complete. This supports
folding #154 into #152's feature branch, not merging #152 into `v0.28.1-dev`.
After the final service exited, no owned process remained and all eight
HCUs returned to 2 MiB used. No unrelated process was signalled.

## Earlier PCP #156 integration and review

At the user's request, #156 was squash-merged into #152's feature branch
as `68da40d9028b36359534d93588021459d519409d`. The tree is byte-identical to
reviewed #156 head `0d1c1bc303814c70016b7cb048f0a5d74c83bcf3`.
PR #152 still targets `v0.28.1-dev`; this operation did not merge #152 there.
DCP was still separate in #154 at this earlier PCP-only gate.

Independent read-only review of the combined production diff found no Critical
or Important blocker. One non-blocking wording issue remains: shared PCP
validation errors in `patch_vllm_config.py` still name GLM-5.2 for some Hy4
rejections. The eager-only rejection itself is correct.

The complete set of 16 test files changed by the combined PR passed
**331 tests, 1 skipped** in 144.82 seconds. This is not a full-repository pass:
the prior LightOp and W4A8 failures remain documented in the PCP record,
including the identical W4A8 base/candidate numerical failure.

## Paired artifact evidence

- vLLM: `0.28.1+dtk2604.torch2110.2609171627.g77acaf`, downloaded wheel
  SHA-256 `2d6b392dcff0d5064c754e838d9ac5ea3ffd5d6c7cac86ed2b5b1b2811cb9b3b`.
  All 3,008 packaged files under `vllm/` match the preinstalled runtime
  byte-for-byte. The artifact is now installed separately with `--no-deps
  --target`; no global vLLM installation was changed.
- Plugin: clean `791be82` wheel, SHA-256
  `58135f06295ad7173bbf4aeb7f4564d9fd00bedffc39db4d125353ebd55c4b29`.
  All 298 tracked non-generated Python files match the combined branch.
  There are no production-file changes between `791be82` and the combined
  tree; subsequent changes are tests and documentation only.
- Live imports resolve vLLM to `vllm-wheel/vllm` and both `vllm_hcu` and
  compiled `hcu_ops` to the isolated plugin wheel. Launch working directory
  is `/tmp`; model is `/models/Hy4-preview-Channel-FP8-w8a8-v2`.
- Evidence directory `E` is
  `/models/vllm-plugin-das/.worktrees/feat-hy4-pcp-v0281/.superpowers/sdd/2026-09-26-hy4-152-combined`.
  Prior PCP evidence remains under the sibling `2026-09-25-hy4-pcp-v0281`.

## Fresh paired-wheel regression

Requests retain the committed HumanEval/0-7 prompt, temperature 0, seed 0,
max tokens 2048 and `reasoning_effort=no_think`. Generated code is inspected
before EvalScope 1.11 functional execution. Concurrent checks use eight
requests; source and request logs are retained under `E`.

| Mode | Result | Evidence prefix under E |
| --- | --- | --- |
| DP8/TP1/EP8, MTP3, E4M3, DeepEP LL/DeepGEMM, default Graph | 8/8, all stop; main/draft Graph capture | `dp8-mtp3-fp8-graph` |
| DP2/TP4/EP8, MTP3, BF16 KV, DeepEP LL/DeepGEMM, default Graph | 8/8, all stop; all three draft positions accepted tokens | `dp2-tp4-mtp3-bf16-graph` |
| TP8, MTP3, E4M3, AITER selector, default Graph | 8/8, all stop; accepted tokens by position: 252/244/228 | `tp8-mtp3-fp8-graph` |
| TP4/PCP2/EP8, MTP3, BF16 KV, AITER selector, eager | 8/8, all stop; accepted tokens by position: 248/244/227 | `tp4-pcp2-mtp3-bf16-eager` |
| PP2/TP1/PCP4/EP4, MTP2, E4M3, DeepEP HT/DeepGEMM, eager | 8/8, all stop; accepted tokens by position: 319/308 | `pp2-pcp4-mtp2-fp8-eager` |

All five runs completed eight concurrent HTTP-successful requests and passed
functional scoring. The three non-PCP services captured main and draft
Graphs under the default policy; PCP services explicitly used eager.
The AITER selector retains per-shape Triton fallback; these results do not
certify an AITER config hit for every shape. Detailed acceptance metrics were
retained for the last four modes; the DP8 run retained capture/request evidence
but was stopped before a metrics snapshot was saved.

Owned services were stopped between modes. After the final PP2 service,
no owned worker remained and all eight HCUs returned to 2 MiB used. No global
runtime installation was replaced and no unrelated process was signalled.

The prior missing-artifact Draft blocker is closed by the byte comparison and
fresh paired-wheel gate. The combined PR is ready for review, not merged or
certified for every configuration. A per-call FP8 writer trace, full-dataset
accuracy, and a green full-repository test baseline remain unclaimed.

Prior same-production PCP gates remain: TP4/PCP2/EP8 target-only and
MTP1/2/3 each 8/8; PP2/TP1/PCP4/EP4 target-only and MTP2 each 32/32.
These subset results do not establish full-dataset accuracy. PCP+Graph and
PCP+DCP remain unsupported; other PP2 backend combinations are not certified.
