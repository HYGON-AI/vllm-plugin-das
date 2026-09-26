# Hy4 PCP on v0.28.1-dev: validation record

This eager-only work was initially submitted as PR #156, based on PR #152
commit `1b51a2bccf36437f0638a74887bffb9167a7e294`. At the user's request,
PR #156 was squash-merged into PR #152 as `68da40d9028b36359534d93588021459d519409d`
on 2026-09-26. The resulting tree is identical to the reviewed #156 head
`0d1c1bc`. It does not claim PCP+CUDA Graph or PCP+DCP. See the
[combined validation record](hy4_v0281_combined_validation.md) for the latest gate.

## Frozen wheel and model provenance

- Historical vLLM baseline: `0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a`, wheel SHA-256 `8e69b61d591cd4b0ae38b0c725eca10232983f9bad8767889908cbcc3e944f46`; source checkout `/models/zb/vllm_025/vllm` at `7b108ad1a51b217e9abec0ddc047978405481bae`. This wheel was downloaded for comparison, not installed over the target.
- Target vLLM: installed `0.28.1+dtk2604.torch2110.2609171627.g77acaf`, downloaded wheel SHA-256 `2d6b392dcff0d5064c754e838d9ac5ea3ffd5d6c7cac86ed2b5b1b2811cb9b3b`; import path `/usr/local/lib/python3.10/dist-packages/vllm`. The version carries the `77acaf` source lineage, but no exact full source SHA is asserted here.
- First PCP candidate: plugin commit `b66e8188e7702b55028408ee1f4977807e39093e`, built as `vllm_hcu-0.28.1rc1.dev491+das.b66e818.dtk2604-cp310-cp310-linux_x86_64.whl`, SHA-256 `ad4dc4f49376301b77a7ed985d4edf0e69472dfaf88ee326fbd1710b728ef5c6`. Installed only under `.superpowers/sdd/2026-09-25-hy4-pcp-v0281/plugin-wheel`; the live import check loaded both `vllm_hcu` and `hcu_ops` from that isolated directory.
- PP2 fix candidate: plugin commit `791be82e058cc8389e685e183c9bf063bc95376f`, built as `vllm_hcu-0.28.1rc1.dev491+das.791be82.dtk2604-cp310-cp310-linux_x86_64.whl`, SHA-256 `58135f06295ad7173bbf4aeb7f4564d9fd00bedffc39db4d125353ebd55c4b29`. Installed only under `.superpowers/sdd/2026-09-25-hy4-pcp-v0281/plugin-wheel-791be82`; both `vllm_hcu` and compiled `vllm_hcu.hcu_ops` were imported from that isolated directory with `/tmp` as the working directory to avoid source-tree shadowing.
- Model: `/models/Hy4-preview-Channel-FP8-w8a8-v2`, 78 backbone layers. Its indexer layer 41 is a `full` producer, which is why the PP2 partition is `41,37`.

## Source and environment checks

- New launcher command tests: 8 passed.
- Hy4/PCP focused source suite: 299 passed, 1 skipped. A separate runner/manager run: 77 passed.
- Post-runtime review: every changed test module plus the PCP manager selection passed 143 tests. The final fixture correction then passed all 18 runner tests, covering non-final PP with zero/two speculative steps and final PP with unequal draft counts and concrete global block/slot rows. Production files remain identical to the validated `791be82` wheel.
- Full `pytest -q tests` was attempted. With `VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm`, it stopped after 5 failures, 645 passes and 64 skips. Two clean-process bootstrap failures were caused by leaking the old source root into their subprocess imports; both pass when rerun without that variable. Two non-PCP LightOp API/patch-coverage failures reproduce on PR #152's base. The W4A8 shared-storage numerical failure also reproduces identically on that base in the same environment (details below). This full-suite attempt is not a pass.

After server teardown, the exact W4A8 case was run sequentially on stacked
base `1b51a2b` and PCP candidate `791be82` (later edits are tests/docs only):

```bash
env -u VLLM_V0251_SOURCE_ROOT PYTHONNOUSERSITE=1 PYTHONPATH=. \
  HIP_VISIBLE_DEVICES=0 pytest -q \
  tests/accuracy/test_unified_aiter_moe_operator.py::test_auto_w4a8_shared_storage_feeds_ht_and_ll_with_empty_expert
```

Both failed at `deepseek_v4_dspark_ops_cases.py:201`, comparing the HT output
with its reference: 1,572,632/1,572,864 mismatches, greatest absolute error
2,211,653.5 and relative error 15.082396507263184. This is a reproduced
baseline issue, not introduced by this PCP diff; it remains unresolved.
Logs: `w4a8_base_comparison.log` and `w4a8_candidate_comparison.log` in the
isolated validation directory. After both tests, all eight HCUs used 2 MiB.

## TP4/PCP2/EP8 target-only BF16 KV (passed)

The eight-card service used the isolated plugin wheel above, `HcuGPUModelRunnerV2`, `FLASHMLA_SPARSE`, eager mode, and the `aiter` MoE selector. AITER lacked a tuned solution for some shapes and logged a per-shape fallback to vLLM Triton; this run does not prove every MoE call used AITER. The exact server command was:

```bash
env -u VLLM_PLUGINS -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  VLLM_USE_V2_MODEL_RUNNER=1 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/models/vllm-plugin-das/.worktrees/feat-hy4-pcp-v0281/.superpowers/sdd/2026-09-25-hy4-pcp-v0281/plugin-wheel \
  python -m vllm.entrypoints.cli.main serve \
  /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --host 127.0.0.1 --port 8000 --trust-remote-code \
  --tensor-parallel-size 4 --prefill-context-parallel-size 2 \
  --enable-expert-parallel --enforce-eager \
  --attention-backend FLASHMLA_SPARSE --moe-backend aiter \
  --enable-prefix-caching --max-model-len 4096 --block-size 64 \
  --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --seed 0
```

The service reached `/health` HTTP 200; all eight HumanEval/0–7 requests
returned HTTP 200 with `finish_reason=stop`. Their generated code was inspected
before running EvalScope 1.11.0's functional checker in an isolated target:
**8/8 passed**. The raw completions, score and server log are under
`.superpowers/sdd/2026-09-25-hy4-pcp-v0281/` as
`tp4_target_humaneval8.json`, `tp4_target_humaneval8_score.json`, and
`tp4_target_server.log`.

A delayed mixed test kept one decode active while two new long prompts of
3,072 and 3,497 tokens arrived. The log showed 3 running requests and nonzero
prompt throughput; all three returned HTTP 200 with `finish_reason=stop`, and
the service remained healthy. The owned process group was terminated and all
eight HCU memory readings returned to zero.

## TP4/PCP2/EP8 MTP1 BF16 KV (passed)

The same command with `--speculative-config
'{"method":"mtp","num_speculative_tokens":1}'` reached `/health` HTTP 200.
The server reported speculative decoding with approximately 93–100% per-position
acceptance during the eight requests. All HumanEval/0–7 responses ended with
`finish_reason=stop`; the completions were inspected before EvalScope scored
**8/8 passed**. Evidence is `tp4_mtp1_server.log`,
`tp4_mtp1_humaneval8.json`, and `tp4_mtp1_humaneval8_score.json` in the
isolated validation directory above. The process group was stopped before the
next configuration.

## TP4/PCP2/EP8 MTP2 BF16 KV (HumanEval passed)

The same command with `--speculative-config
'{"method":"mtp","num_speculative_tokens":2}'` reached `/health` HTTP 200.
All eight HumanEval responses ended normally, generated code was inspected,
and EvalScope scored **8/8 passed**. The log reports two speculative positions
with real draft acceptance. Evidence is `tp4_mtp2_server.log`,
`tp4_mtp2_humaneval8.json`, and `tp4_mtp2_humaneval8_score.json` in the
isolated validation directory.

A concurrent MTP2 run held one decode while two new prefills of 2,230 and
3,630 prompt tokens arrived. The engine log showed `Running: 3 reqs` and
nonzero prompt throughput. All three requests returned HTTP 200; the two
prefills ended with `stop` after two completion tokens each, while the
deliberately capped decode reached its configured 600-token `length` limit.
The service stayed healthy and its owned process group released all eight HCUs
before the MTP3 run.

## TP4/PCP2/EP8 MTP3 BF16 KV (passed)

The same command with `--speculative-config
'{"method":"mtp","num_speculative_tokens":3}'` reached `/health` HTTP 200.
The log recorded three speculative positions with real accepted drafts. All
eight HumanEval responses ended with `stop`, the generated code was inspected,
and EvalScope scored **8/8 passed**. Evidence is `tp4_mtp3_server.log`,
`tp4_mtp3_humaneval8.json`, and `tp4_mtp3_humaneval8_score.json` in the
isolated validation directory. Teardown returned all eight HCUs to 0% VRAM.

## PP2/TP1/PCP4/EP4 FP8 E4M3: diagnosis and verified target/MTP2 gates

The first PP2 launch stopped at CLI parsing because `--moe-backend deepgemm`
is invalid in installed 0.28.1; `deep_gemm` is the accepted name. The corrected
service passed `/health` but stalled on its first request. PP0/PCP workers
waited in a synchronous H2D copy at `pcp_manager.py:490`; PP1/PCP workers
waited in `irecv_tensor_dict`. A one-variable AITER MoE control stalled at the
same place, excluding DeepGEMM alone as the cause. The installed upstream
`GPUModelRunner.sample_tokens()` calls `PPHandler.receive(input_batch)` on
non-final PP before its final-stage-only PCP restore. The v0.25.1 HCU runner
restored the global batch on non-final PP; the initial v0.28.1 adapter did so
only when a speculator existed. A new regression test failed on that mismatch.
Commit `791be82` restores the global batch before non-final PP receive even
without MTP, without a hidden-state collective; all 17 runner tests then
passed. The historical `VLLM_KV_CACHE_LAYOUT=NHD` aliases to `LBNHC` in this
vLLM and was not a meaningful variable in this diagnosis.

The exact eight-card target-only command on the isolated `791be82` wheel is
below; run it from `/tmp` or another directory outside the plugin source tree
so the source checkout does not shadow the isolated install:

```bash
env -u VLLM_PLUGINS -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  VLLM_USE_V2_MODEL_RUNNER=1 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  VLLM_PP_LAYER_PARTITION=41,37 \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/models/vllm-plugin-das/.worktrees/feat-hy4-pcp-v0281/.superpowers/sdd/2026-09-25-hy4-pcp-v0281/plugin-wheel-791be82 \
  /usr/bin/python -m vllm.entrypoints.cli.main serve \
  /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --host 127.0.0.1 --port 8000 --trust-remote-code \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --prefill-context-parallel-size 4 --enable-expert-parallel \
  --moe-backend deep_gemm --all2all-backend deepep_high_throughput \
  --attention-backend FLASHMLA_SPARSE --kv-cache-dtype fp8_e4m3 \
  --enforce-eager --enable-prefix-caching --max-model-len 4096 \
  --block-size 64 --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --seed 0
```

The target-only service selected `HcuGPUModelRunnerV2`, DeepEP HT,
`DeepEPDeepGemmContiguousExperts` and FP8 E4M3 KV; `/health` returned HTTP
200, then a short request returned `OK` with `finish_reason=stop`. All eight
HumanEval/0–7 requests returned `stop`. Generated code was inspected before
EvalScope 1.11.0 functional scoring: **8/8 passed**. Evidence:
`pp2_target_791be82_server.log`, `pp2_target_791be82_humaneval8.json`,
`pp2_target_791be82_humaneval8_score.json` in the isolated validation
directory. The owned server exited and all eight HCUs returned to 2 MiB used
before the MTP2 launch.

For MTP2, append
`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
to the command above. The service passed `/health`, returned a short `OK`,
and logged real accepted drafts at both speculative positions. HumanEval/0–7
returned `stop` and scored **8/8** after inspection. HumanEval/0–31 then
returned 32 `stop` completions; all generated code was inspected before
functional scoring and **32/32 passed**. Evidence: the
`pp2_mtp2_791be82_server.log`, `pp2_mtp2_791be82_humaneval8.json`,
`pp2_mtp2_791be82_humaneval8_score.json`,
`pp2_mtp2_791be82_humaneval32.json`, and
`pp2_mtp2_791be82_humaneval32_score.json` files.

A delayed mixed request held one 500-token decode while two new prompts of
2,133 and 3,033 tokens prefetched. Both prefill responses returned HTTP 200
with `finish_reason=stop`; the capped decode returned HTTP 200 with its
expected `length` finish. Engine logs showed `Running: 3 reqs`, nonzero
prompt throughput during overlap, and a speculative window with 50% draft
acceptance. `/health` remained HTTP 200. Evidence:
`pp2_mtp2_791be82_mixed.log` and the MTP2 server log. The owned process
group was stopped and all eight HCU memory readings returned to 2 MiB.

The config gate restricts topology and structural prerequisites; KV dtype,
all-to-all and MoE backend selectors remain independent. Only the exact
FP8 E4M3/DeepEP HT/DeepGEMM command above has PP2 accuracy evidence.
During the target-only 32-sample run, two additional delayed long prompts
(2,133 and 3,033 input tokens) were submitted concurrently with active
HumanEval generation. Both returned HTTP 200 and `stop`; evidence is
`pp2_target_791be82_mixed.log`. The requester was interrupted after the first
eight completed samples; it resumed from HumanEval/8 with the same payload
and sampling protocol, preserving all original completed records.
Target-only HumanEval/0–31 completed with 32 `stop` responses. After code
inspection, EvalScope functional scoring passed **32/32**, matching the
MTP2 score on this same subset (not a full-dataset accuracy claim). Evidence:
`pp2_target_791be82_32_server.log`, `pp2_target_791be82_humaneval32.json`,
`pp2_target_791be82_humaneval32_score.json`, and
`pp2_target_791be82_resume32.log`. The final health check returned HTTP 200
and the owned server process group was stopped. PCP+CUDA Graph and PCP+DCP
are not claimed.
