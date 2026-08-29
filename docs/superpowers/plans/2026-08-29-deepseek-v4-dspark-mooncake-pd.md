# DeepSeek V4 DSpark Mooncake P/D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable and validate DeepSeek-V4-Flash-0731 FP8/INT8 DSpark serving with Mooncake P/D disaggregation on one eight-HCU node split into DP4+EP prefill and DP4+EP decode pools.

**Architecture:** Replace the blanket DSpark/KV-transfer rejection with an exact DeepSeek-V4 plus Mooncake allowlist while preserving PCP and unsupported-connector guards. Add a config-driven integration runner that owns only its three process groups (prefill, decode, proxy), routes HumanEval through the official vLLM Mooncake proxy, and asserts real Mooncake, DSpark, DeepEP, and DeepGEMM evidence before accepting accuracy.

**Tech Stack:** Python 3.10, pytest, vLLM 0.25.1, vllm-plugin-das, Mooncake Transfer Engine, FastAPI Mooncake proxy, EvalScope HumanEval, HCU/ROCm multiprocessing.

**Spec:** `docs/plans/2026-08-29-deepseek-v4-dspark-mooncake-pd-design.md`

## Global Constraints

- Base all work on stacked dependency commit `ae6a19481a71241243971d4f6f1162b88eea091e`.
- Support `/models/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8` and `/models/DeepSeek-V4-Flash-0731-Channel-INT8-w8a8`.
- Prefill owns HCU 0-3 and decode owns HCU 4-7; each service uses DP=4 and EP.
- Use `--kv-cache-dtype fp8`, `MooncakeConnector`, and the standard `--speculative-config` DSpark interface.
- Use one `--all2all-backend deepep_auto` request per service; do not expose internal high-throughput or low-latency flags.
- Do not add `--moe-backend`; the existing HCU `deepep_auto` oracle selects the paired DeepGEMM experts.
- Do not enable PCP+P/D, pipeline-parallel P/D, other KV connectors, or cross-node transport.
- HumanEval acceptance requires exactly 32 predictions, 32 reviews, and normalized pass@1=1.0 for each checkpoint.
- Never treat startup alone as a pass: require proxy traffic, Mooncake send/receive evidence, and positive DSpark draft/accepted counters.

---

### Task 1: Narrow the DSpark P/D configuration gate

**Files:**
- Modify: `tests/runtime_patch/test_platform_hcu_config.py:1009-1090`
- Modify: `vllm_hcu/patch/platform/core_fix/patch_vllm_config.py:185-210`

**Interfaces:**
- Consumes: `VllmConfig.speculative_config.method`, `model_config.architectures`, and `kv_transfer_config.kv_connector`.
- Produces: `_validate_dspark_pd_scope(vllm_config: object) -> None`, called by `validate_and_update_hcu_config` before runtime binding.

- [ ] **Step 1: Replace the old rejection test with failing allowlist tests**

Add these behaviors next to the existing DSpark P/D test:

```python
def _dspark_pd_config(connector: str, architecture: str = "DeepseekV4ForCausalLM"):
    config = _validation_config(HcuFeatureConfig())
    config.model_config.architectures = [architecture]
    config.speculative_config = SimpleNamespace(method="dspark")
    config.kv_transfer_config = SimpleNamespace(kv_connector=connector)
    return config


def test_deepseek_v4_dspark_allows_mooncake_pd_before_model_loading() -> None:
    patch_vllm_config.validate_and_update_hcu_config(
        _dspark_pd_config("MooncakeConnector")
    )


@pytest.mark.parametrize("connector", ["NixlConnector", "ExampleConnector"])
def test_deepseek_v4_dspark_rejects_unvalidated_pd_connectors(connector: str) -> None:
    with pytest.raises(ValueError, match=f"DSpark.*{connector}"):
        patch_vllm_config.validate_and_update_hcu_config(
            _dspark_pd_config(connector)
        )


def test_non_deepseek_dspark_does_not_gain_mooncake_pd_support() -> None:
    with pytest.raises(ValueError, match="DeepSeek-V4"):
        patch_vllm_config.validate_and_update_hcu_config(
            _dspark_pd_config("MooncakeConnector", "Qwen3ForCausalLM")
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
pytest -q tests/runtime_patch/test_platform_hcu_config.py \
  -k 'dspark and (mooncake or pd_connector or non_deepseek)' -vv
```

Expected: the Mooncake allow test fails with the existing blanket
`DeepSeek-V4 DSpark with P/D disaggregation is not supported on HCU` error.

- [ ] **Step 3: Implement the minimal allowlist helper**

Add this helper above `validate_and_update_hcu_config` and replace the blanket
guard with a call to it:

```python
def _validate_dspark_pd_scope(vllm_config: object) -> None:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    use_dspark = getattr(speculative_config, "use_dspark", None)
    dspark_enabled = (
        bool(use_dspark())
        if callable(use_dspark)
        else getattr(speculative_config, "method", None) == "dspark"
    )
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    connector = getattr(kv_transfer_config, "kv_connector", None)
    if not dspark_enabled or connector is None:
        return
    architectures = getattr(
        getattr(vllm_config, "model_config", None), "architectures", ()
    )
    if connector == "MooncakeConnector" and "DeepseekV4ForCausalLM" in architectures:
        return
    raise ValueError(
        "DeepSeek-V4 DSpark P/D disaggregation on HCU supports only "
        f"MooncakeConnector; got connector={connector!r}, "
        f"architectures={architectures!r}."
    )
```

- [ ] **Step 4: Verify GREEN and the surrounding configuration suite**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
pytest -q tests/runtime_patch/test_platform_hcu_config.py -vv
```

Expected: all tests pass, including existing PCP configuration tests.

- [ ] **Step 5: Commit the configuration change**

```bash
git add vllm_hcu/patch/platform/core_fix/patch_vllm_config.py \
  tests/runtime_patch/test_platform_hcu_config.py
git commit -m "feat(hcu): allow DeepSeek V4 DSpark Mooncake PD"
```

---

### Task 2: Define the two-pool command contract

**Files:**
- Create: `tests/integration/server/pd_evalscope_server.py`
- Create: `tests/models/deepseek_v4_flash_0731_dspark_mooncake_pd_humaneval.yaml`
- Create: `tests/integration/server/test_evalscope_deepseek_v4_dspark_mooncake_pd.py`

**Interfaces:**
- Consumes: `load_profiled_config`, existing EvalScope configuration schema, `VLLM_V0251_SOURCE_ROOT`, and the model override environment variable.
- Produces: immutable `PDCommands` and `pd_commands(config: dict, *, model_env: str) -> PDCommands` with prefill/decode/proxy argv, per-service environments, endpoints, and timeouts.

- [ ] **Step 1: Write command-contract tests before the helper exists**

Create the new test module with profiles `fp8` and `int8`. Assert for each
profile:

```python
commands = pd_commands(config, model_env=MODEL_ENV)
assert commands.prefill_env["HIP_VISIBLE_DEVICES"] == "0,1,2,3"
assert commands.decode_env["HIP_VISIBLE_DEVICES"] == "4,5,6,7"
for command in (commands.prefill, commands.decode):
    assert _option_value(command, "--data-parallel-size") == "4"
    assert _option_value(command, "--tensor-parallel-size") == "1"
    assert "--enable-expert-parallel" in command
    assert _option_value(command, "--all2all-backend") == "deepep_auto"
    assert _option_value(command, "--kv-cache-dtype") == "fp8"
    assert "--moe-backend" not in command
    assert json.loads(_option_value(command, "--speculative-config")) == DSPARK_CONFIG
assert json.loads(_option_value(commands.prefill, "--kv-transfer-config"))["kv_role"] == "kv_producer"
assert json.loads(_option_value(commands.decode, "--kv-transfer-config"))["kv_role"] == "kv_consumer"
assert commands.proxy[-9:] == [
    "--prefill", "http://127.0.0.1:10141", "18998",
    "--decode", "http://127.0.0.1:10142",
    "--host", "127.0.0.1", "--port", "10140",
]
```

Also assert the FP8 and INT8 model paths and served-model names match their
selected profile.

- [ ] **Step 2: Run the new contract tests and verify RED**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
pytest -q tests/integration/server/test_evalscope_deepseek_v4_dspark_mooncake_pd.py \
  -k contract -vv
```

Expected: collection fails because `pd_evalscope_server` and `pd_commands` do
not yet exist.

- [ ] **Step 3: Add the profiled YAML configuration**

The shared `pd` block must declare:

```yaml
pd:
  host: 127.0.0.1
  proxy_port: 10140
  startup_timeout_s: 7200
  shutdown_timeout_s: 120
  prefill:
    port: 10141
    bootstrap_port: 18998
    data_parallel_rpc_port: 29551
    visible_devices: "0,1,2,3"
  decode:
    port: 10142
    bootstrap_port: 18999
    data_parallel_rpc_port: 29552
    visible_devices: "4,5,6,7"
```

Put the common vLLM arguments once under `pd.common_args`; append role-specific
Mooncake JSON under each role. Include `VLLM_HCU_MOONCAKE_TTFT_TRACE=1` and
`VLLM_LOGGING_LEVEL=DEBUG` in both role environments. Profiles override only
`model`, `served_model_name`, and `evalscope.work_dir`.

- [ ] **Step 4: Implement `PDCommands` and `pd_commands` minimally**

Use this public shape:

```python
@dataclass(frozen=True)
class PDCommands:
    prefill: list[str]
    decode: list[str]
    proxy: list[str]
    prefill_env: dict[str, str]
    decode_env: dict[str, str]
    proxy_env: dict[str, str]
    host: str
    proxy_port: int
    prefill_port: int
    decode_port: int
    startup_timeout_s: int
    shutdown_timeout_s: int
```

Build each server argv as `vllm serve MODEL`, common args, role args, explicit
port, and explicit DP RPC port. Resolve the proxy script only from
`$VLLM_V0251_SOURCE_ROOT/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py`
and raise `FileNotFoundError` if absent. Build environments by copying
`_server_environment(config)`, adding `HIP_VISIBLE_DEVICES`, and assigning the
role's `VLLM_MOONCAKE_BOOTSTRAP_PORT`.

- [ ] **Step 5: Run command-contract tests and verify GREEN**

Run the Step 2 command. Expected: both FP8 and INT8 contract cases pass.

- [ ] **Step 6: Commit the command contract**

```bash
git add tests/integration/server/pd_evalscope_server.py \
  tests/integration/server/test_evalscope_deepseek_v4_dspark_mooncake_pd.py \
  tests/models/deepseek_v4_flash_0731_dspark_mooncake_pd_humaneval.yaml
git commit -m "test: define DeepSeek V4 Mooncake PD topology"
```

---

### Task 3: Add owned P/D lifecycle and runtime-evidence gates

**Files:**
- Modify: `tests/integration/server/pd_evalscope_server.py`
- Modify: `tests/integration/server/test_evalscope_deepseek_v4_dspark_mooncake_pd.py`

**Interfaces:**
- Consumes: `PDCommands`, `_terminate_process_group`, `_reset_evalscope_artifacts`, `evalscope_command`, and `_assert_pass_criteria`.
- Produces: `assert_pd_runtime_evidence(prefill_log: Path, decode_log: Path, decode_metrics: str) -> None` and `run_evalscope_pd_server_test(config: dict, *, model_env: str, model_label: str, required_hcu_count: int) -> None`.

- [ ] **Step 1: Write a failing evidence-parser test**

Create temporary P and D logs containing the required positive markers and a
metrics string containing:

```text
vllm:spec_decode_num_draft_tokens_total{engine="0"} 128
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 64
```

The positive test must include `event=p_send_kv_done` in P, `event=d_kv_ready`
in D, the contiguous DeepEP/DeepGEMM marker in P, and the masked marker in D.
Add parameterized negative cases for zero draft tokens, zero accepted tokens,
missing Mooncake receive evidence, `Sending to ... failed`, and
`MooncakeXferMetadata transfer failed`.

- [ ] **Step 2: Run the evidence tests and verify RED**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
pytest -q tests/integration/server/test_evalscope_deepseek_v4_dspark_mooncake_pd.py \
  -k runtime_evidence -vv
```

Expected: import or attribute failure because the parser does not exist.

- [ ] **Step 3: Implement the minimal evidence parser**

Require all of these strings:

```python
required_prefill = (
    "Mooncake TTFT_EVENT event=p_send_kv_done",
    "DeepEP auto selected contiguous high-throughput experts for this forward.",
    "Using DeepEPDeepGemmContiguousExperts with DeepGEMM HT path.",
)
required_decode = (
    "Mooncake TTFT_EVENT event=d_kv_ready",
    "DeepEP auto selected masked low-latency experts for this forward.",
    "Using DeepEPDeepGemmMaskedExperts with DeepGEMM LL path.",
)
```

Reject known Mooncake failure strings in either log. Parse and sum every
Prometheus sample for `spec_decode_num_draft_tokens_total` and
`spec_decode_num_accepted_tokens_total`; require each sum to be greater than
zero.

- [ ] **Step 4: Verify the evidence tests pass**

Run the Step 2 command. Expected: all positive and negative cases pass.

- [ ] **Step 5: Write a failing process-order test with fake processes**

Monkeypatch `subprocess.Popen`, server/proxy wait helpers, EvalScope execution,
metrics download, and `_terminate_process_group`. Assert the runner:

1. starts prefill;
2. waits for prefill health;
3. starts decode;
4. waits for decode health;
5. starts proxy and waits for a successful routed smoke response;
6. runs EvalScope against `proxy_port`;
7. checks accuracy and runtime evidence;
8. terminates proxy, decode, then prefill in `finally`.

- [ ] **Step 6: Run the process-order test and verify RED**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
pytest -q tests/integration/server/test_evalscope_deepseek_v4_dspark_mooncake_pd.py \
  -k process_order -vv
```

Expected: failure because `run_evalscope_pd_server_test` is missing.

- [ ] **Step 7: Implement owned lifecycle in the tested order**

Write logs under `WORK_DIR/logs/{prefill,decode,proxy,evalscope}.log`. Use
`start_new_session=True` for all three long-lived processes and stop only the
captured `Popen` objects with `_terminate_process_group`. The routed smoke
request must target `/v1/completions`, use the configured served model,
`temperature=0`, `max_tokens=16`, and retry HTTP 503 until the proxy has loaded
the P-side DP engine IDs. After EvalScope returns zero, fetch decode `/metrics`,
then call `_assert_pass_criteria` and `assert_pd_runtime_evidence`.

- [ ] **Step 8: Verify the whole non-HCU P/D test module**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
pytest -q tests/integration/server/test_evalscope_deepseek_v4_dspark_mooncake_pd.py \
  -m 'not hcu' -vv
```

Expected: every command, evidence, and lifecycle contract test passes.

- [ ] **Step 9: Commit lifecycle support**

```bash
git add tests/integration/server/pd_evalscope_server.py \
  tests/integration/server/test_evalscope_deepseek_v4_dspark_mooncake_pd.py
git commit -m "test: run Mooncake PD accuracy with owned processes"
```

---

### Task 4: Register and document the acceptance entry points

**Files:**
- Modify: `tests/hcu_ci_registry.py:250-270`
- Modify: `.github/workflows/configs/hcu-test-map.yaml:445-490,700-735`
- Modify: `tests/patch/test_hcu_ci_selector.py:575-610`
- Modify: `tests/integration/server/README.md`
- Create: `docs/deepseek_v4_flash_0731_dspark_mooncake_pd_validation.md`

**Interfaces:**
- Consumes: the two runtime pytest node IDs from Task 3.
- Produces: two new nodes in the existing `deepseek-v4-dspark-humaneval` CI job and copy/paste P, D, proxy, smoke-client, and HumanEval commands.

- [ ] **Step 1: Add a failing registry-coverage assertion**

Extend the existing DeepSeek-V4 selector parameter table with the new P/D test
module and YAML, both mapped only to `deepseek-v4-dspark-humaneval` and a new
group named `deepseek-v4-dspark-pd-eval`; run:

```bash
pytest -q tests/patch/test_hcu_ci_selector.py -k registry -vv
```

Expected: failure because the new paths are not yet mapped to the expected
DeepSeek-V4 HumanEval job/group.

- [ ] **Step 2: Register and map the P/D test module**

Add one literal registration under the existing job:

```python
register_hcu_ci(
    job="deepseek-v4-dspark-humaneval",
    target=(
        "tests/integration/server/"
        "test_evalscope_deepseek_v4_dspark_mooncake_pd.py"
    ),
    est_time=43200,
)
```

Add `tests/integration/server/pd_evalscope_server.py`, the P/D test module, and
the P/D YAML to a `deepseek-v4-dspark-pd-eval` group in
`.github/workflows/configs/hcu-test-map.yaml`; map that group only to
`deepseek-v4-dspark-humaneval`. The existing job's `-k
deepseek_v4_dspark_humaneval` selector already matches the two runtime node
names defined in Task 3.

- [ ] **Step 3: Verify registry tests pass**

Run Step 1 again and then:

```bash
python tools/run_patch_tests.py --suite inventory
```

Expected: selector and inventory suites pass.

- [ ] **Step 4: Add operator documentation**

Document exact FP8 and INT8 commands generated by the YAML contract, including
the model override variable, role-specific `HIP_VISIBLE_DEVICES`, distinct DP
RPC/bootstrap ports, official Mooncake proxy command, curl smoke request, and
both pytest node IDs. Explicitly state that PCP+P/D and non-Mooncake connectors
remain unsupported, and that `deepep_auto` is one public selector rather than
separate HT/LL CLI flags.

- [ ] **Step 5: Commit registry and initial documentation**

```bash
git add tests/hcu_ci_registry.py tests/patch/test_hcu_ci_selector.py \
  tests/integration/server/README.md \
  docs/deepseek_v4_flash_0731_dspark_mooncake_pd_validation.md
git commit -m "docs: add DeepSeek V4 DSpark Mooncake PD validation"
```

---

### Task 5: Run FP8 4P+4D Mooncake and HumanEval-32

**Files:**
- Modify: `docs/deepseek_v4_flash_0731_dspark_mooncake_pd_validation.md`

**Interfaces:**
- Consumes: profile `fp8` and runtime runner from Tasks 2-3.
- Produces: fresh FP8 logs, exact accuracy artifacts, and recorded runtime evidence.

- [ ] **Step 1: Confirm the node is clean and dependencies are present**

Run:

```bash
hy-smi --showmeminfo vram
python -c 'from mooncake.engine import TransferEngine; print(TransferEngine)'
test -f /models/zb/vllm_025/vllm/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py
```

Expected: all eight devices have no model allocation, Mooncake imports, and the
proxy exists.

- [ ] **Step 2: Execute the FP8 P/D acceptance node**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
pytest -q -s \
  tests/integration/server/test_evalscope_deepseek_v4_dspark_mooncake_pd.py::test_deepseek_v4_dspark_humaneval_fp8_mooncake_pd
```

Expected: P and D become healthy, routed smoke succeeds, runtime-evidence gate
passes, EvalScope writes exactly 32 predictions/reviews, normalized score is
32/32, and pytest passes.

- [ ] **Step 3: Inspect fresh FP8 evidence**

Run targeted searches over the generated P/D logs for Mooncake TTFT events,
positive DeepEP HT/LL markers, DeepGEMM contiguous/masked class markers, and
absence of Mooncake failure strings. Inspect the normalized HumanEval JSON and
the raw EvalScope report.

- [ ] **Step 4: Record the exact FP8 result**

Add the pytest duration, raw `mean_acc`, normalized passed count, artifact
paths, positive draft/accepted counter totals, and P/D log paths to the
validation document.

- [ ] **Step 5: Commit FP8 validation evidence**

```bash
git add docs/deepseek_v4_flash_0731_dspark_mooncake_pd_validation.md
git commit -m "test: validate FP8 DSpark Mooncake PD on 4P4D"
```

---

### Task 6: Run INT8 4P+4D Mooncake and HumanEval-32

**Files:**
- Modify: `docs/deepseek_v4_flash_0731_dspark_mooncake_pd_validation.md`

**Interfaces:**
- Consumes: profile `int8` and the same quantization-independent P/D implementation.
- Produces: fresh INT8 logs, exact accuracy artifacts, and a direct FP8/INT8 comparison.

- [ ] **Step 1: Confirm FP8 processes are gone and HCU memory is released**

Run `hy-smi --showmeminfo vram` and inspect only process IDs recorded by the
FP8 runner. Expected: all eight devices are available before INT8 startup.

- [ ] **Step 2: Execute the INT8 P/D acceptance node**

Run:

```bash
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
pytest -q -s \
  tests/integration/server/test_evalscope_deepseek_v4_dspark_mooncake_pd.py::test_deepseek_v4_dspark_humaneval_int8_mooncake_pd
```

Expected: the same topology/evidence gates pass and HumanEval normalizes to
32/32 with exactly 32 predictions and reviews.

- [ ] **Step 3: Inspect and record fresh INT8 evidence**

Record the same fields as FP8 and add a compact comparison table for raw score,
normalized score, latency, output token rate, DSpark counters, and Mooncake
failure count.

- [ ] **Step 4: Commit INT8 validation evidence**

```bash
git add docs/deepseek_v4_flash_0731_dspark_mooncake_pd_validation.md
git commit -m "test: validate INT8 DSpark Mooncake PD on 4P4D"
```

---

### Task 7: Full verification, review, and separate stacked MR

**Files:**
- Review all files changed since `ae6a19481a71241243971d4f6f1162b88eea091e`.

**Interfaces:**
- Consumes: all implementation and evidence commits.
- Produces: one pushed branch and one separate stacked GitHub MR based on `feat/deepseek-v4-flash-0731-dspark-hcu-v0251`.

- [ ] **Step 1: Run the focused regression suites fresh**

```bash
python tools/run_patch_tests.py --suite contract -- \
  tests/runtime_patch/test_platform_hcu_config.py \
  tests/runtime_patch/test_hcu_mooncake_contract.py
VLLM_V0251_SOURCE_ROOT=/models/zb/vllm_025/vllm \
PYTHONPATH=/models/zb/vllm_025/vllm:. \
pytest -q tests/integration/server/test_evalscope_deepseek_v4_dspark_mooncake_pd.py \
  -m 'not hcu'
python tools/run_patch_tests.py --suite inventory
git diff --check ae6a19481a71241243971d4f6f1162b88eea091e..HEAD
```

Expected: every command exits zero.

- [ ] **Step 2: Perform code review against compatibility boundaries**

Confirm the diff does not alter Mooncake transfer algorithms without a failing
test, does not weaken PCP rejection, does not enable other connectors, does not
introduce broad process kills, does not hardcode credentials, and does not add
internal HT/LL or MoE-backend CLI flags.

- [ ] **Step 3: Verify branch shape**

```bash
git status --short
git log --oneline --decorate ae6a19481a71241243971d4f6f1162b88eea091e..HEAD
git diff --stat ae6a19481a71241243971d4f6f1162b88eea091e..HEAD
```

Expected: clean worktree and only P/D-specific commits.

- [ ] **Step 4: Push and create the separate stacked MR**

Push `feat/deepseek-v4-dspark-mooncake-pd-v0251`, then create an MR with base
`feat/deepseek-v4-flash-0731-dspark-hcu-v0251`. Include upstream PR #46807,
the reason the blanket guard existed, FP8/INT8 4P+4D evidence, HumanEval-32
results, and the exact service/client commands.

- [ ] **Step 5: Re-read the remote MR and verify uploaded SHA**

Compare the remote branch SHA with local `HEAD`, inspect the rendered title,
base/head branches, commit list, and diff. Report the MR URL, commit SHA,
accuracy scores, and copy/paste commands to the user.
