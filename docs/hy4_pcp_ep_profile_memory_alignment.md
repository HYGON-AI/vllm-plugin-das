# Hy4 PCP + EP profile-memory alignment

## Scope and provenance

This change adapts the remaining PCP+EP profile-memory behavior from
`origin/v0.25.1@6ea7b12` commits `3f30e7c` and `2b7de5f` to the current MRV2
owners on `origin/v0.28.1-dev@293b3ee`. It does not port the v0.25.1 V1
communicator monkey patch or claim PCP CUDA Graph support.

Candidate source commit: `2403dff5aaa8225f14625c0c6215728b3114cb65`.
Candidate wheel:
`vllm_hcu-0.28.1rc1.dev491+das.2403dff.dtk2604-cp310-cp310-linux_x86_64.whl`,
SHA-256 `0c953d21e86d5b4aeef00129dd3a22dc87526dddd7767e1299f02173e4fe781f`.

The baseline wheel was built from `293b3ee`:
`vllm_hcu-0.28.1rc1.dev491+das.293b3ee.dtk2604-cp310-cp310-linux_x86_64.whl`,
SHA-256 `3b37c395de92114d8337d885fdaf0635ff83621e2898d509809917bdf3a20225`.
Both wheels were installed with `--no-deps --target` into separate directories.

## Changes

- `HcuGPUModelRunnerV2.profile_run()` profiles PCP ranks with
  `max(1, (max_num_tokens // pcp_size) * 9 // 8)` and restores the configured
  value in `finally`. PCP1 delegates without mutation.
- Fixed DeepEP high-throughput construction now derives `num_nvl_bytes` from
  `VLLM_DEEPEP_BUFFER_SIZE_MB`; the auto manager's existing maximum-size rule
  remains unchanged.

Focused red tests failed on the unmodified implementation. After the change,
the complete relevant selection passed `155` tests with `14` existing warnings:

```text
tests/runtime_patch/test_glm52_pcp_runner.py
tests/runtime_patch/test_worker_framework_opt.py
tests/runtime_patch/test_moe_deepep.py
```

A repository-wide `pytest -q tests` attempt reached 90% with five failures,
then remained inside one waiting test for more than 11 minutes and did not
respond to repeated terminal interrupts. The owned pytest process was stopped
and all device memory was released. A bounded rerun with `--maxfail=5`
completed with `803 passed, 64 skipped, 5 failed, 14 warnings`; the failures
were the existing W4A8 shared-storage numerical case, LightOp public-category
audit, patch-module coverage audit, and two fresh-process worker-dispatcher
state assertions. None names either changed production file or a new test.
The W4A8 mismatch count and maximum error exactly match the recorded baseline.
This is not a green repository-wide suite and remains an explicit limitation.

## Eight-card service command

The candidate and baseline used the same command. Only `PYTHONPATH` and
`VLLM_CACHE_ROOT` selected the isolated install and cache directory.

```bash
env -u VLLM_PLUGINS -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_NN=1 \
  VLLM_DEEPEP_BUFFER_SIZE_MB=256 \
  VLLM_PP_LAYER_PARTITION=41,37 \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONNOUSERSITE=1 \
  VLLM_CACHE_ROOT=<isolated-cache> \
  PYTHONPATH=<isolated-plugin-wheel> \
  /usr/bin/python -m vllm.entrypoints.cli.main serve \
  /models/Hy4-preview-Channel-FP8-w8a8-v2 \
  --served-model-name hy4-pcp-ep-mtp2 \
  --host 127.0.0.1 --port 8015 --trust-remote-code \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --prefill-context-parallel-size 4 --enable-expert-parallel \
  --moe-backend deep_gemm --all2all-backend deepep_high_throughput \
  --attention-backend FLASHMLA_SPARSE --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --enforce-eager --enable-prefix-caching --max-model-len 4096 \
  --block-size 64 --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --seed 0
```

Both runs constructed eight `HcuGPUModelRunnerV2` workers, selected
`DeepEPHTAll2AllManager` and
`DeepEPDeepGemmContiguousExperts with DeepGEMM HT path`, used FP8 E4M3 KV,
and resolved CUDA Graph mode to `NONE` because this PCP topology remains eager.

## A/B results

| Gate | Candidate | Baseline |
| --- | ---: | ---: |
| HumanEval/0-7 | 8/8 | 8/8 |
| Concurrent stress | 320/320 HTTP 200, all `stop` | 320/320 HTTP 200, all `stop` |
| Preemptions / request errors | 0 / 0 | 0 / 0 |
| PP0 peak activation | 1.93 GiB | 2.38 GiB |
| PP0 available KV cache | 17.75 GiB | 15.67 GiB |
| PP1 available KV cache | 21.34 GiB | 18.73 GiB |
| Stress completion throughput | 22.40 tok/s | 22.74 tok/s |
| Stress p95 latency | 1.605 s | 1.507 s |

The memory deltas are the intended outcome: candidate PP0 exposed about
2.08 GiB more KV cache under the same 256 MiB DeepEP setting. The single-run
short-output throughput delta was about -1.5% and is not evidence of a speedup
or a regression. No `Traceback`, OOM, timeout, reallocation, or NCCL timeout
marker appeared in either service log. Both services remained healthy after
the workload and all eight cards returned to zero reported memory use after
shutdown.

The stress workload used 16 concurrent clients, a 16-request warmup, and 320
requests with varied prompt lengths. HumanEval was run first in both services
with identical requests; both functional score reports passed all eight tasks.
Six of eight raw completions were byte-identical, so functional scoring rather
than raw-text identity is the precision criterion for this concurrent MTP run.
