# Task 3 Report: Real GLM-5.2 Runtime Diagnosis

## Status

**BLOCKED.** The real high-throughput runtime now reaches an eight-rank
DeepEP profile forward with the intended `DeepEPHTPrepareAndFinalize` and
HCU `DeepGemmExperts`, but fails because the HCU LightOP path still calls
vLLM's unavailable upstream DeepGEMM alignment API from
`deepgemm_moe_permute()`. The three-hypothesis stop rule was reached, so no
fourth production fix was attempted. Low-latency, deterministic API parity,
and four-request concurrency were not run after deterministic correctness
failed.

Scoped partial runtime fixes were committed as `a38f27d` (`fix(hcu): advance
GLM52 DeepEP runtime setup`).

## Software and device baseline

Baseline command:

```bash
python -m pip list | rg -i 'vllm|deep-ep|deepgemm|lightop|torch'
rocm-smi --showmeminfo vram
```

Relevant packages:

- `vllm 0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a`
- `vllm_hcu 0.25.1+das.f1f7a06.dtk2604`
- `deep-ep 1.1.0+das185.dtk2604.torch2110.2608181058.gb5b9ab`
- `deepgemm 2.1.0+das185.dtk2604.torch2110.2608171132.g493d80`
- `lightop 0.6.0+das.dtk2604.torch2110.2608171227.g8c835c`
- `torch 2.11.0+das.opt1.dtk2604.202604021232.g1175f0`

All eight HCUs were idle before the first launch: each reported 147440 MiB
total and 2 MiB used. No stale vLLM, EngineCore, or worker process was present.

Repository baseline was commit `81f865ceb02fea8f04fd210bf56e849827d38544`
on `feat/glm52-deepep-deepgemm-v0251`; the worktree was four commits ahead of
`origin/v0.25.1` before Task 3 changes.

## Exact two-case command and initial outcome

The required command was run exactly:

```bash
VLLM_HCU_GLM52_CHANNEL_INT8_MODEL=/models/GLM-5.2-Channel-INT8-w8a8 \
VLLM_USE_DEEP_GEMM=1 \
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=$PWD \
python -m pytest -q -s \
  --strict-test-resources \
  tests/integration/parallel/test_tp_ep_models.py -k glm52
```

Outcome: `2 failed, 11 deselected in 55.98s`. Both cases failed before model
or worker creation because `tests.integration.model_runtime` called
`LLM(data_parallel_size=8)` in one process. vLLM raised:

```text
ValueError: LLM(data_parallel_size=8) is not supported for single-process
usage and may hang. Please use the explicit multi-process data-parallel
example at 'examples/features/data_parallel/data_parallel_offline.py'.
```

Logs:

- HT: `/tmp/vllm-hcu-integration/logs/20260826_001240_GLM-5.2-Channel-INT8-w8a8_glm52-int8-deepep_high_throughput.log`
- LL: `/tmp/vllm-hcu-integration/logs/20260826_001301_GLM-5.2-Channel-INT8-w8a8_glm52-int8-deepep_low_latency.log`

## Systematic RED/GREEN diagnosis

### 1. Unsupported single-process DP harness

Root-cause hypothesis: DP8 offline inference must follow vLLM 0.25.1's
explicit multiprocess example, set `VLLM_DP_*` per rank, and omit
`data_parallel_size` from each rank-local `LLM` constructor.

RED:

```bash
python -m pytest -q \
  tests/integration/test_model_runtime_cli.py::test_tp_ep_dp_uses_explicit_multiprocess_launcher
```

The test failed because the existing path constructed a single-process LLM
with `data_parallel_size=8`.

GREEN after the scoped harness launcher change:

```text
2 passed in 0.02s
```

for the new launcher regression plus the adjacent CLI forwarding contract.

### 2. HCU expert classes retained an NVIDIA-only device gate

The isolated HT rerun formed the eight-rank NCCL DP/EP group and logged
`Using DeepEPHTAll2AllManager all2all manager`, then oracle selection rejected
both classes:

```text
ValueError: dpsk_deep_gemm is required by HCU sidecar but does not support
this INT8 MoE configuration: DeepGemmExperts: kernel does not support current
device hip; BatchedDeepGemmExperts: kernel does not support current device hip
```

Full log:
`/tmp/vllm-hcu-integration/logs/20260826_001545_GLM-5.2-Channel-INT8-w8a8_glm52-int8-deepep_high_throughput.log`.

Root-cause hypothesis: the HCU-owned classes contain explicit ROCm INT8/FP8
LightOP branches but retained upstream `_supports_current_device()` logic,
which delegates to `HCUPlatform.support_deep_gemm()` and its inherited `False`.

RED: the new real-class regression failed with
`DeepGemmExperts._supports_current_device() == False` under a ROCm platform
when upstream DeepGEMM support was false.

GREEN: both HCU classes accept `current_platform.is_rocm()` while retaining
the upstream support gate for non-HCU devices; the regression and adjacent
oracle test passed (`2 passed`).

### 3. DPSK dynamic INT8 was downgraded to W8A16

The next HT rerun passed expert selection and loaded all 282 shards, then
failed constructing `DeepGemmExperts`. The wrapper had allowed only AITER to
use `int8_w8a8_moe_quant_config`; DPSK fell through to upstream, which treats
absent static activation-scale tensors as W8A16. Consequently the HCU LightOP
branch was not entered and constructor code called an unavailable upstream
DeepGEMM alignment function.

Full log:
`/tmp/vllm-hcu-integration/logs/20260826_002023_GLM-5.2-Channel-INT8-w8a8_glm52-int8-deepep_high_throughput.log`.

RED: the oracle regression returned the literal upstream W8A16 sentinel for
`DPSK_DEEPGEMM` with `per_act_token_quant=True`.

GREEN: DPSK now shares the canonical dynamic-W8A8 construction path with
AITER. The oracle and HCU device regressions passed (`2 passed`).

### Stopped fourth boundary

The final HT rerun passed all three earlier boundaries, loaded all weights,
logged the intended manager and prepare/finalize implementation, and entered
the profile forward:

```text
Using DeepEPHTAll2AllManager all2all manager.
Using DeepEPHTPrepareAndFinalize
...
vllm_hcu/.../experts/deep_gemm_moe.py:394 in apply
vllm_hcu/.../deep_gemm_utils.py:551 in deepgemm_moe_permute
block_m, block_k = get_mk_alignment_for_contiguous_layout()
RuntimeError: DeepGEMM backend is unavailable in the current vLLM environment,
or the available DeepGEMM package does not provide the required APIs for these
kernels.
```

This traceback provides runtime evidence that `DeepGemmExperts` was selected,
even though the class name is not emitted as a standalone INFO log. Full log:
`/tmp/vllm-hcu-integration/logs/20260826_002427_GLM-5.2-Channel-INT8-w8a8_glm52-int8-deepep_high_throughput.log`.

No fourth hypothesis/fix was attempted. Architecture must be revisited before
deciding whether HCU LightOP should own its alignment contract or the HCU
DeepGEMM utility wrapper should supply a compatible alignment API.

## Mode outcomes and intended expert evidence

| Mode | Outcome | Manager evidence | Expert evidence |
|---|---|---|---|
| `deepep_high_throughput` | BLOCKED during profile forward | `DeepEPHTAll2AllManager`, `DeepEPHTPrepareAndFinalize` | Traceback enters HCU `DeepGemmExperts.apply`; then unavailable upstream alignment API |
| `deepep_low_latency` | Initial harness RED only; not rerun after correctness remained blocked | None after fix | Required `BatchedDeepGemmExperts` evidence not obtained |

The full two-case command therefore does not pass. Direct LL validation was
not justified after HT deterministic correctness failed and the stop rule was
reached.

## Deterministic parity and concurrency

Not run. No mode reached a healthy generation endpoint, so it was impossible
to issue the required temperature-zero request for `"The capital of France
is"` with 32 output tokens, seed 0, and logprobs 5. Text, token-ID, and
first-token-logprob parity against the v0.25.1 default result are therefore
**not established**.

The prior reference/HumanEval artifacts under
`/tmp/glm52-dcp2-he32-pr28-20260825/` were inspected only for availability and
were not modified. They do not substitute for a live candidate-mode parity
result.

Four simultaneous 24-token API requests were not sent; the required four HTTP
200 responses are **not established**. Task 4 performance/HumanEval was not
run.

## Focused automated verification

Fresh final command:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=$PWD \
python -m pytest -q \
  tests/runtime_patch/test_quant_gemm_aiter.py \
  tests/runtime_patch/test_moe_deepep.py \
  tests/runtime_patch/test_platform_hcu_config.py \
  tests/patch/test_worker_dispatcher.py \
  tests/patch/test_module_exchange.py \
  tests/integration/test_model_runtime_cli.py
```

Result: `227 passed, 14 warnings in 162.02s`. The warnings are existing torch
`jit.script_method` deprecations. `git diff --check` also passed before commit.

## Changes and commit

Commit `a38f27d` contains:

- the supported explicit multiprocess offline DP launcher for DP>1;
- a RED/GREEN regression preventing single-process DP construction;
- ROCm acceptance for the two HCU LightOP-backed expert classes;
- DPSK dynamic-W8A8 quant-config preservation;
- focused regressions for both oracle/runtime gates.

No external vLLM source, model files, or wheels were modified.

## Final resource release

The last failed profile run orphaned several EngineCore/worker grandchildren in
process group `2078704`; SIGTERM did not release the remaining workers, so the
validated task-owned group was released with SIGKILL. After five seconds:

- no `tests.integration.model_runtime`, vLLM entrypoint, EngineCore, or
  `VLLM::Worker` process remained;
- every HCU again reported exactly 2 MiB VRAM used.

All eight HCUs were idle when this report was finalized.
