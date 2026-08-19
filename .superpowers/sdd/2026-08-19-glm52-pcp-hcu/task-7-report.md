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
