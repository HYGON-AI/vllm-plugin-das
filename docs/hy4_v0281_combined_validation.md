# Hy4 combined PR #152 / #156 validation (2026-09-26)

## Integration and review

At the user's request, #156 was squash-merged into #152's feature branch
as `68da40d9028b36359534d93588021459d519409d`. The tree is byte-identical to
reviewed #156 head `0d1c1bc303814c70016b7cb048f0a5d74c83bcf3`.
PR #152 still targets `v0.28.1-dev`; this operation did not merge #152 there.
DCP remains separate in #154.

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
