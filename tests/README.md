# vLLM-HCU Tests

The detailed test architecture is documented in
[`docs/test_architecture_design_v0251.md`](../docs/test_architecture_design_v0251.md).
This file is the quick entry point for running the current test suites.

## Test layers

- `patch/`: patch inventory, lifecycle, compatibility, and isolation.
- `runtime_patch/`: CPU/mock runtime contracts.
- `accuracy/`: portable operator/reference comparisons and live HCU kernel
  checks.
- `integration/`: real model, graph, LoRA, speculative decoding, KV transfer,
  and OpenAI server/EvalScope integration.
- `distributed/single_node/`: up to eight local HCU devices.
- `distributed/multi_node/`: reserved until a real multi-node environment is
  available.
- `stress/`: long-context, memory-pressure, and soak workloads.
- `models/`: model and dataset configuration files, never checkpoints.
- `fixtures/`: small CPU-safe support modules.

Do not describe a smoke test as an accuracy test, or a single-node
producer/consumer test as multi-node validation.

## Runner

Install the lightweight test dependency:

```bash
python -m pip install -r requirements-test.txt
```

Run the standard suites from the repository root:

```bash
python tools/run_patch_tests.py --suite inventory
python tools/run_patch_tests.py --suite accuracy
python tools/run_patch_tests.py --suite contract
python tools/run_patch_tests.py --suite integration-smoke
python tools/run_patch_tests.py --suite distributed-single-node
python tools/run_patch_tests.py --suite full
```

The runner prefers the installed vLLM package root, typically
`/usr/local/lib/python3.10/dist-packages`, and then falls back to local
`vllm_0251` checkouts. Use `--vllm-source /path/to/vllm/root` or set
`VLLM_V0251_SOURCE_ROOT` when testing a different vLLM tree. Extra pytest
arguments can be appended after `--`.

Useful dry runs:

```bash
python tools/run_patch_tests.py --suite full --collect-only
python tools/run_patch_tests.py --suite model --collect-only
python tools/check_patch_test_coverage.py --json
```

`tools/check_patch_test_coverage.py` is a pytest-independent preflight. It
fails when a new `patch_*.py` module lacks the standard adapter contract or a
direct test reference.

## Suite intent

- `inventory`: direct patch inventory and lifecycle checks.
- `accuracy`: portable CPU/reference comparisons, excluding live HCU tests.
- `accuracy-hcu`: live HCU lightop and Triton-kernel comparisons.
- `contract`: portable patch/runtime contracts, excluding live HCU tests.
- `integration-smoke`: non-slow, non-multi-HCU, non-multi-node integration
  tests.
- `model`: real HCU model integration tests under `tests/integration`.
- `distributed-single-node`: single-machine distributed tests.
- `distributed-multi-node`: tests marked `multi_node`.
- `nightly`: scheduled hardware tests marked `hcu`, `model`, or `nightly`.
- `full`: all pytest tests under `tests/` with no marker filter.

## Hardware CI operations

The repository keeps the expected Python, Torch, vLLM, AITER, EvalScope,
pytest, HIP, and DTK versions in
`.github/workflows/configs/hcu-runner-environment.json`. Every hardware job
checks this lock before running tests. Update the lock in the same reviewed
change as an intentional Runner image or package upgrade; an unreviewed Runner
drift fails before pytest starts.

Available manual workflows:

- `HCU full-enabled` accepts a branch, tag, or SHA and runs every job declared
  in `hcu-test-map.yaml`.
- `HCU release wheel` builds the requested checkout, installs the wheel in an
  isolated virtual environment, verifies the imported package and compiled
  extension originate from that wheel, then runs an OpenAI server smoke and a
  TP4/EP4 smoke.
- `HCU nightly` accepts an optional ref for manual runs and uses `v0.25.1` for
  scheduled runs. Its matrix is generated from the `nightly_jobs` list in
  `hcu-test-map.yaml`.

Nightly failures create or update one GitHub tracking issue and close it after
recovery. Set the repository variable `HCU_CI_NIGHTLY_OWNER` to a GitHub login
to assign that issue; leaving it unset keeps the issue unassigned. Per-job
request, environment, JUnit, pytest, server, and EvalScope logs remain on the
self-hosted Runner under
`/tmp/vllm-hcu-ci/<run-id>/<attempt>/<job-id>/`; each job summary records the
exact Runner and path. These files follow the host's `/tmp` cleanup policy and
are not uploaded to GitHub. Release validation logs use the same local layout;
only the built Wheel is retained as a GitHub Artifact for 30 days so downstream
validation jobs can receive it when they run on a different host.

Intentional HCU quarantine is controlled by
`.github/workflows/configs/hcu-quarantine.json`; do not add an untracked
`xfail` as a substitute. Each entry must name an existing HCU job and declare
`id`, `owner`, `reason`, `issue`, `nodeid`, focused `pytest_args`,
`retest_after`, and `expires`. It may override `timeout_minutes`. Due entries
are run by nightly with the same zero-test/skip/xfail fail-closed rules as
normal hardware jobs. An expired entry or an `xfail` without a matching
quarantine entry fails matrix preparation until it is removed or deliberately
renewed.

## HCU model and accuracy tests

Common model test entry points:

```bash
# Qwen3.5-9B model smoke/graph coverage.
python tools/run_patch_tests.py --suite model -- -k qwen35_9b

# OpenAI server + EvalScope accuracy gates.
python tools/run_patch_tests.py --suite model -- -k qwen3_8b_gsm8k_evalscope_server
python tools/run_patch_tests.py --suite model -- -k qwen35_9b_gsm8k_evalscope_server
python tools/run_patch_tests.py --suite model -- -k qwen3_vl_8b_mmmu_evalscope_server
python tools/run_patch_tests.py --suite model -- -k deepseek_r1_channel_fp8_gsm8k_evalscope_server

# Feature smoke tests.
python tools/run_patch_tests.py --suite model -- -k qwen3_4b_engine_features
python tools/run_patch_tests.py --suite model -- -k tp_ep
python tools/run_patch_tests.py --suite model -- -k qwen3_4b_lora_switching
python tools/run_patch_tests.py --suite model -- -k llama2_7b_eagle
python tools/run_patch_tests.py --suite model -- -k example_connector
```

Model integration tests force `VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1` in the test
environment. Server/EvalScope cases also clear `VLLM_PLUGINS` before launching
`vllm serve`, so vLLM can load the HCU entry points normally.

Logs and reports:

- Server/EvalScope cases write to `/tmp/vllm-hcu-evalscope/<case>/`.
- Regular integration helpers write to `/tmp/vllm-hcu-integration/logs`.
- Clear the corresponding `/tmp` case directory before a rerun if you want a
  clean report set.

Accuracy gates are asserted from EvalScope JSON reports:

- GSM8K uses `mean_acc`, displayed as `Pass@1`, and must be at least `0.95`.
- Qwen3-VL MMMU smoke uses the `Art` subset and requires `mean_acc >= 0.55`.
- Qwen3.5 GSM8K disables thinking through chat-template kwargs so EvalScope
  answer extraction stays deterministic.

## Model configuration files

The current server/EvalScope configs live under `tests/models/`:

- `deepseek_r1_gsm8k_evalscope.yaml`
- `qwen3_8b_gsm8k_evalscope.yaml`
- `qwen35_9b_gsm8k_evalscope.yaml`
- `qwen3_vl_8b_mmmu_evalscope.yaml`

Each config declares the model path, server arguments, EvalScope work directory,
generation settings, dataset settings, and pass criteria. Override a model path
with the corresponding environment variable named by the test file, for example
`VLLM_HCU_QWEN35_9B_MODEL`.
