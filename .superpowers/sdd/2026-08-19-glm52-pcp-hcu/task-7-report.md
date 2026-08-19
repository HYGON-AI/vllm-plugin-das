# Task 7 Report: Reproducible GLM-5.2 PCP Acceptance Coverage

## Status

Implemented CPU-verifiable launch, configuration, and payload contracts plus
the marked eight-HCU smoke, 32K/64K long-context, and EvalScope HumanEval
acceptance cases. No server or accelerator workload was started in this task.

## Delivered contracts

- The default model is `/models/GLM-5___1-Channel-FP8-w8a8`, with
  `VLLM_HCU_GLM52_MODEL` consumed by the server command builder as an override.
- The launch environment sets `VLLM_USE_V2_MODEL_RUNNER=1` through the shared
  EvalScope server harness.
- The candidate server uses TP=4, PCP=2, EP, eager execution, Triton MoE, and
  `max_model_len=69632`, with thinking disabled by default.
- The command contract rejects speculative decoding, PP, DCP, compilation,
  and explicit quantization flags (including `slimquant_marlin`).
- Fixed-ID OpenAI-compatible smoke requests use `temperature=0`, return token
  IDs, and pass `chat_template_kwargs.enable_thinking=false`.
- The 32K and 64K cases submit exact token-count prompts, require at least one
  decode token, and require the server worker group to remain live. JSON
  artifacts contain token IDs, latency, TTFT, throughput, and peak observed
  device-memory fields.
- EvalScope uses OpenAI API evaluation, HumanEval limit 32, batch size 1,
  `temperature=0`, `do_sample=false`, `max_tokens=2048`, and a request-level
  thinking-off option.
- All real acceptance cases carry `hcu_count(8)`, `multi_hcu`, `slow`, and
  `nightly`; the EvalScope case also carries `external_service("evalscope")`.

## TDD evidence

RED was captured before creating the YAML configuration or extending the
launch environment consumer:

```text
pytest -q tests/integration/server/test_evalscope_report_threshold.py \
  tests/integration/server/test_evalscope_glm52_pcp_humaneval.py \
  -k 'command or payload or config'

3 failed, 4 deselected
```

All three failures were the expected `FileNotFoundError` for the not-yet-added
`tests/models/glm52_pcp_humaneval_evalscope.yaml`; they demonstrated that the
command, config, and payload consumers could not satisfy the new contract.
After adding the YAML and the smallest required shared-harness extension, the
same focused command passed.

## Verification

```text
VLLM_HCU_GLM52_MODEL=/models/GLM-5___1-Channel-FP8-w8a8 \
pytest --collect-only -q \
  tests/integration/models/test_glm52_pcp_mrv2.py \
  tests/integration/server/test_evalscope_glm52_pcp_humaneval.py

8 tests collected
```

```text
pytest -q tests/integration/server/test_evalscope_report_threshold.py \
  tests/integration/server/test_evalscope_glm52_pcp_humaneval.py \
  -k 'command or payload or config'

3 passed, 4 deselected
```

Additional CPU-safe verification before commit:

```text
pytest -q -m 'not hcu' \
  tests/integration/models/test_glm52_pcp_mrv2.py \
  tests/integration/server/test_evalscope_report_threshold.py \
  tests/integration/server/test_evalscope_glm52_pcp_humaneval.py

7 passed, 4 deselected
```

`python -m compileall` for the modified Python files and `git diff --check`
also completed successfully. `ruff` is not installed in this environment.

## Concerns and deferred evidence

- The real GLM-5.2 server, long-context requests, rank convergence, HumanEval
  results, and runtime metrics remain unexecuted; they require the Task 8
  eight-HCU run.
- Peak device memory is sampled from aggregate live device usage during each
  request, so Task 8 should run on an otherwise idle eight-HCU host for a clean
  measurement.

## Fix Round 1

Review identified two integration gaps: EvalScope addressed the checkpoint
path while the server advertised a basename, and post-request liveness relied
only on the parent process plus deterministic output parity.

### RED: served/request model identity

The new builder contract executed both consumers with the default checkpoint
and an overridden `VLLM_HCU_GLM52_MODEL`:

```text
pytest -q tests/integration/server/test_evalscope_glm52_pcp_humaneval.py \
  -k 'request_model_matches_served_model'

2 failed, 4 deselected
```

The default case emitted `/models/GLM-5___1-Channel-FP8-w8a8` to EvalScope
while advertising `GLM-5___1-Channel-FP8-w8a8`; the override case emitted the
override checkpoint path while retaining that advertised basename.

### GREEN: stable served/request model identity

`server.served_model_name` is now the single stable request-facing identity.
The server builder adds `--served-model-name` from it and the EvalScope builder
uses the same value, while `VLLM_HCU_GLM52_MODEL` continues to select the
checkpoint loaded by `vllm serve`.

```text
pytest -q tests/integration/server/test_evalscope_glm52_pcp_humaneval.py \
  -k 'command_request_model_matches_served_model'

2 passed, 4 deselected
```

### RED: acceptance-path worker-group health check

A CPU integration test started a real loopback SSE completion endpoint and a
real local `/health` endpoint. With the acceptance path's health call absent,
the completion succeeded but the health endpoint was never reached:

```text
pytest -q tests/integration/models/test_glm52_pcp_mrv2.py \
  -k 'health'

1 failed, 4 deselected
AssertionError: assert 0 == 1
```

### GREEN: post-request `/health`

`_stream_completion` now requires an HTTP 200 from the existing OpenAI server
`/health` endpoint after every smoke and 32K/64K completion. Deterministic token
parity remains separately asserted; it is not represented as individual-rank
telemetry.

```text
pytest -q tests/integration/models/test_glm52_pcp_mrv2.py \
  -k 'health'

1 passed, 4 deselected
```

The loopback test declares `NO_PROXY` so the environment's configured HTTP
proxy cannot intercept the local health contract.

Individual-rank failure evidence remains a Task 8 hardware-log check.

### Fix Round 1 verification

```text
VLLM_HCU_GLM52_MODEL=/models/GLM-5___1-Channel-FP8-w8a8 \
pytest --collect-only -q \
  tests/integration/models/test_glm52_pcp_mrv2.py \
  tests/integration/server/test_evalscope_glm52_pcp_humaneval.py

11 tests collected
```

```text
pytest -q \
  tests/integration/models/test_glm52_pcp_mrv2.py \
  tests/integration/server/test_evalscope_report_threshold.py \
  tests/integration/server/test_evalscope_glm52_pcp_humaneval.py \
  -k 'command or payload or config or health'

7 passed, 7 deselected
```

```text
pytest -q -m 'not hcu' \
  tests/integration/models/test_glm52_pcp_mrv2.py \
  tests/integration/server/test_evalscope_report_threshold.py \
  tests/integration/server/test_evalscope_glm52_pcp_humaneval.py

10 passed, 4 deselected
```

The modified Python files compile successfully and `git diff --check` is
clean. No hardware or model server was started.
