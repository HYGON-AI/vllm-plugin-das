# Hy4 FP8 DCP2 validation on v0.28.1-dev

## Frozen artifacts

- Date: 2026-09-26 UTC.
- Model: `/models/Hy4-preview-Channel-FP8-w8a8-v2` (`HYV4ForCausalLM`).
- Model `config.json` SHA-256: `d4a648cb09bb89f4b8778e60629e43618f1abb581ef3aa38bd67b2b2cd998441`.
- Installed OpenDAS vLLM: `0.28.1+dtk2604.torch2110.2609171627.g77acaf`, imported from `/usr/local/lib/python3.10/dist-packages/vllm`.
- Plugin code under test: commit `627ceda105459ac1968feceaa4c1e436f915aac1`.
- Isolated plugin wheel: `vllm_hcu-0.28.1rc1.dev491+das.627ceda.dtk2604-cp310-cp310-linux_x86_64.whl`.
- Plugin wheel SHA-256: `8ef2e52009b1ce290656f8e4cf2eaaeee7b57e61083923cc7c96bb4bf60fa617`.
- The wheel was installed with `pip --target` into the plan's ignored validation workspace. `vllm_hcu.__file__` resolved below that target when launched from `/models`, not from a source checkout.
- HumanEval source: `/models/datasets/humaneval/HumanEval.jsonl.gz`; scorer: EvalScope 1.11.0 installed only in an ignored scoring target.

The installed vLLM distribution exposes the `g77acaf` lineage, but its original
wheel file and delivery checksum are unavailable. Therefore this is an isolated
plugin-wheel gate against the installed target, not a complete paired-wheel
release gate.

## Exact service commands

All commands were run from `/models`. Upper- and lower-case proxy variables
were unset, both `NO_PROXY` forms were set for localhost, and `VLLM_PLUGINS`
was unset so all plugin entry points could load. The resolved plugin package
and compiled `hcu_ops` came from the isolated wheel target below, not the
source checkout.

The DCP2+MTP3 acceptance command was:

```bash
env -u VLLM_PLUGINS -u VLLM_USE_BREAKABLE_CUDAGRAPH \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_NN=1 \
  VLLM_DCP_Q_REPLICATE=0 VLLM_HCU_USE_CUSTOM_OPS=1 \
  VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM=1 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/models/vllm-plugin-das/.worktrees/feat-hy4-dcp-pcp-v0281/.superpowers/sdd/2026-09-25-hy4-dcp-fp8-v0281/wheel-site-627ceda \
  python -m vllm.entrypoints.cli.main serve \
  /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --host 127.0.0.1 --port 8000 --trust-remote-code \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --moe-backend aiter --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching --max-model-len 4096 --block-size 64 \
  --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --seed 0 \
  --decode-context-parallel-size 2 --dcp-comm-backend ag_rs \
  --cp-kv-cache-interleave-size 1 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The DCP target-only command omits `--speculative-config`; the eager diagnostic
also adds `--enforce-eager`. The no-DCP controls additionally omit all three
DCP flags. None of the Graph runs set `--enforce-eager`, a manual graph mode,
PLE prefetch, or an AITER-shuffle override. The later concurrent gate adds
only `--per-request-spec-decode-metrics detailed` to the acceptance command.

## Source-test status

The complete set of changed test files passed `277` tests, with one skip and
`14` warnings, after the launcher and concurrent gate edits. The
full source suite is not green: after correcting its external v0.25.1 source
root, a fail-fast run stopped at
`tests/accuracy/test_unified_aiter_moe_operator.py::test_auto_w4a8_shared_storage_feeds_ht_and_ll_with_empty_expert`
after `204` passes. The branch does not change unified AITER MoE code. A prior
non-fail-fast run reached 88% and then hung in a hardware test, so no full-suite
pass is claimed.

## Live validation matrix

| Mode | Graph policy | HumanEval 0-7 | HumanEval 0-31 | Status |
| --- | --- | ---: | ---: | --- |
| no-DCP target, FP8 E4M3 | default `FULL_AND_PIECEWISE` | 8/8 | 32/32 | passed |
| no-DCP MTP3, FP8 E4M3 | default | 8/8 | n/a | passed |
| DCP2 target, FP8 E4M3 | eager diagnostic | 8/8 | n/a | passed |
| DCP2 target, FP8 E4M3 | default `PIECEWISE` | 8/8 | 32/32 | passed |
| DCP2 MTP3, FP8 E4M3 | default `PIECEWISE` | 8/8 | 32/32 | passed |

The completed no-DCP target run constructed eight
`HcuGPUModelRunnerV2` workers, resolved `FULL_AND_PIECEWISE`, selected native
FP8 E4M3 KV, Channel-wise FP8 dense linear, sparse MLA and expert parallelism,
captured PIECEWISE and FULL graphs on every worker, and served 40 HTTP 200
completions. All finish reasons were `stop`; the log contained no traceback or
error marker. The repeated prompts raised the final prefix-cache hit rate to
17.5%. Some MoE shapes had no supported AITER solution and explicitly fell
back to vLLM Triton; the tuned channel-shuffle CSV also loaded. This result is
reported as ordered AITER lookup with per-shape fallback, not universal AITER
kernel execution.

The no-DCP MTP3 control also constructed eight Model Runner V2 workers and
captured the default graphs. HumanEval 0–7 passed 8/8 with all finish reasons
`stop`. Its four reported speculative-decoding windows accepted 91.1%–94.4%
of draft tokens. The installed target reports fused multi-step execution as
unsupported for the sparse indexer/attention pair and rebuilds attention
metadata between draft steps; that explicit fallback is the control behavior
used for the DCP2+MTP3 comparison.

The first DCP eager attempt started but produced incoherent repeated text.
The plugin NN weight path had skipped the target attention backend's post-load
hook, so DCP sink heads were not gathered. After correcting that, a native
FlashMLA probe showed the FP8 kernel accepts scattered `-1` index holes and
rejects dynamic `topk_length`; empty shards require explicit LSE
normalization. The remaining precision defect was the sparse indexer: the
plugin path did not merge rank-local DCP candidate scores and global positions
before top-k selection. Commit `627ceda` ports that HCU-specific merge and
handles short local prefills, empty shards and decode rows. Its LightOp fused
top-k route remains gated by `VLLM_HCU_USE_CUSTOM_OPS`.

The corrected DCP eager target run passed 8/8. DCP Graph target and MTP3 each
constructed eight `HcuGPUModelRunnerV2` workers and completed 16 graph-capture
markers (target and draft in MTP3), with `enforce_eager=False`. The resolved
mode was `PIECEWISE`; this is a non-eager Graph pass, not a FULL-graph claim.
The MTP3 log reports the official metadata rebuild between draft steps and
nonzero draft acceptance, ranging roughly 87%–100% across logged windows.
For these request shapes, AITER MoE lookup found no compatible tuned solution
and selected the documented vLLM Triton MoE fallback.

All 32 responses in each DCP Graph run finished with `stop` and passed the
EvalScope functional checker. Exact generated text matched the no-DCP target
for 24/32 DCP target tasks and 23/32 DCP MTP3 tasks; the two DCP modes matched
for 23/32 tasks. The raw completions and per-task scores are retained, so
functional parity does not imply byte-identical generation.

The MTP3 raw/score SHA-256 values are `5506cb80fb84e962b5c6c012304f9b885e7175306ecaf77d9c0f78d78d7d220c`
and `257cd3932ba9983c1b900dc8957f7c096e5641d5daef7bfc18170b82aa0fd2d0`
for 8, and `069d9c29a21542415ba0320d29ec54871e5ec8c51d69e99ae6239409602e6f89`
and `8fc7e00027ffe6680b7b2a93b4e6f3ba3d3d765931cd3a98da59693b833c125d`
for 32. The MTP3 server log SHA-256 is
`0d5c6aac9b92fb57987ad24d9d9f5f6ac32f5fcaff7058fe399c0c3ac8be6b67`.

Raw predictions, score JSON, server logs, marker extracts and their checksums
are retained below the ignored SDD workspace for
`2026-09-25-hy4-dcp-fp8-v0281`.

The concurrent MTP3 gate used the same DCP Graph command with
`--per-request-spec-decode-metrics detailed` added. Three simultaneous
HumanEval/0, /3 and /7 requests were active together (`Running: 3 reqs`),
returned HTTP 200 with `stop`, and passed the functional checker 3/3.
Detailed per-request acceptance reported 140/150, 42/42 and 79/81 draft
tokens. Their respective seventh request-local verification steps accepted
`[0, 3, 1]` draft tokens, showing different acceptance patterns while the
requests overlapped; no global step IDs were recorded. The raw response SHA-256 is
`60f9f12444a5c3c0537e8e53bd215052bd1b2d21cde351c07fc6b671b05886b7`;
the server log SHA-256 is
`35a85c63df68f53b79acda6e6498aff8e141bee3d18e24b3f1bb587714b6c7ce`.
The server was shut down and all eight HCUs returned to 0% VRAM usage.

## Current release decision

The target-only, MTP3 and concurrent accuracy gates pass. Independent
complete-diff review found no blocking runtime issue; its one minor finding
about request-local versus global MTP step IDs was corrected in the test and
this record, then the changed-test-file suite passed again. Keep the separate
MR in Draft because the original paired OpenDAS vLLM wheel file/checksum is
unavailable, even though the installed target lineage and isolated plugin
wheel were validated.
