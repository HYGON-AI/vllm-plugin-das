# HYV4 v0.25.1 clean integration validation

This is the Task 10 execution protocol, prepared before real model runs.
CPU tests establish contracts; they do not establish checkpoint loading,
accelerator arithmetic, graph correctness, distributed serving, or accuracy.
Task 11 must replace each state only after retaining its evidence.

## Hardware result matrix

| Gate | State | Required evidence |
| --- | --- | --- |
| TP8 Channel-FP8 target, AITER, default graphs | not run | Roots, load, backend, graph capture, health, short/long chat, teardown |
| TP8 MTP3, AITER, default graphs | not run | Target/draft capture, drafted/accepted tokens and acceptance by position |
| TP8 MTP3, FP8 E4M3 KV, repeated prefix | not run | Three requests, rising cache hits, coherent output, healthy service |
| PP2/TP1/PCP4/DP1/DCP1/EP4, DeepEP HT, DeepGEMM, eager | not run | Two stages, four stage-local ranks, short and 3000+ token prefill |
| DP8/TP1/EP8, DeepEP HT, DeepGEMM | not run | All eight ranks, HTTP 200, correct backend and clean teardown |
| DP8/EP8 static offline EPLB load | not run | Same plan digest on ranks, pre-load binding, zero post-load rearrangements |
| HumanEval/0–31 target-only | not run | 32 predictions/reviews, accuracy, Pass@1, errors and lengths |
| HumanEval/0–31 MTP3 | not run | Same tasks/options, per-task flips, accuracy and acceptance |
| Custom packed W4A8 | not run | Compatible checkpoint and device numerical/runtime evidence |
| Native packed W4A8 | not run | Compatible checkpoint and device numerical/runtime evidence |
| Two-node Mooncake P/D | not run | Two-node transfer and generation evidence |

Custom/native W4A8 have no compatible checkpoint or hardware validation.
The supplied checkpoint is Channel-FP8, not a W4A8 substitute. Mooncake has
no two-node runtime validation. Static SlimQuant EPLB, HYV4 dynamic/record
EPLB, and MTP+PCP are unsupported. Generic offline record tests do not
authorize HYV4 record-mode serving. Do not replace failures by changing the
model, backend, topology, quantization, or graph mode.

## Pinned environment

Run from the worktree below. Keep this prefix for every Python, server,
client, and evaluator command. Never edit the installed vLLM or checkpoint.

```bash
export PLUGIN_ROOT=/models/vllm-plugin-das/.worktrees/feat-hy4-v0251-clean
export VLLM_TARGET_ROOT=/models/.installs/vllm-0.25.1-das185-g7b108a-hy4-clean
export VLLM_V0251_SOURCE_ROOT="$VLLM_TARGET_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VLLM_TARGET_ROOT:$PLUGIN_ROOT"
export MODEL=/models/Hy4-preview-Channel-FP8-w8a8
cd "$PLUGIN_ROOT"
unset VLLM_PLUGINS
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
unset VLLM_PP_LAYER_PARTITION
export RUN_ROOT="$(mktemp -d /tmp/hy4-v0251-validation.XXXXXX)"
python3 - <<'PY'
import importlib.metadata as md
import os, pathlib, sys
import torch, vllm, vllm_hcu
assert "VLLM_PLUGINS" not in os.environ
assert pathlib.Path(vllm.__file__).resolve().is_relative_to(
    pathlib.Path(os.environ["VLLM_TARGET_ROOT"]).resolve())
assert pathlib.Path(vllm_hcu.__file__).resolve().is_relative_to(
    pathlib.Path(os.environ["PLUGIN_ROOT"]).resolve())
assert md.version("vllm") == "0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a"
assert torch.__version__.split("+")[0] == "2.11.0"
assert torch.version.hip == "6.3.26113"
print(sys.version, sys.executable)
print(vllm.__file__, md.version("vllm"))
print(vllm_hcu.__file__, md.version("vllm-hcu"))
print(torch.__version__, torch.version.hip)
PY
sha256sum /models/artifacts/hy4-v0251-clean/vllm-0.25.1+das185.dtk2604.torch2110.2608171710.g7b108a-cp310-cp310-linux_x86_64.whl
cat /opt/dtk/.info/rocm_version
git rev-parse HEAD origin/v0.25.1
python3 -m vllm_hcu.doctor
```

Expected wheel SHA-256:
`8e69b61d591cd4b0ae38b0c725eca10232983f9bad8767889908cbcc3e944f46`.
The frozen baseline used Python 3.10.12, DTK 26.04, Torch 2.11.0/HIP
6.3.26113. Installed plugin distribution metadata still says
`0.28.1rc1.dev491+das.0e8d794.dtk2604`; source-root assertions above
are essential and the metadata discrepancy remains a packaging concern.

## Automated gates

Enumerate again after the candidate commit; keep helper and `__init__.py`
files in the explicit list. No marker, name, node or module exclusions are
allowed in the changed-module gate.

```bash
git diff --name-only origin/v0.25.1..HEAD | rg '^tests/.*\.py$' > "$RUN_ROOT/changed-tests.txt"
wc -l "$RUN_ROOT/changed-tests.txt"
cat "$RUN_ROOT/changed-tests.txt"
mapfile -t changed_tests < "$RUN_ROOT/changed-tests.txt"
env -u VLLM_PLUGINS python3 -m pytest -q "${changed_tests[@]}"
env -u VLLM_PLUGINS python3 tools/run_patch_tests.py \
  --vllm-source "$VLLM_TARGET_ROOT" --suite contract -- -q
env -u VLLM_PLUGINS python3 -m pytest -q -m 'not hcu' \
  tests/patch tests/runtime_patch tests/accuracy tests/gemma4_test
env -u VLLM_PLUGINS python3 -m pytest -q tests/hcu_ci_registry.py \
  tests/integration/test_model_runtime_cli.py tests/patch/test_hcu_ci_selector.py
env -u VLLM_PLUGINS python3 -m vllm_hcu.doctor
python3 -m compileall -q vllm_hcu tests
git diff --check origin/v0.25.1..HEAD
git diff --check
```

The runner ignores `VLLM_TARGET_ROOT` unless passed as `--vllm-source`
(or through `VLLM_V0251_SOURCE_ROOT`). It also forces its child
`VLLM_PLUGINS=__disabled__`. The direct invocation is additional evidence,
but `tests/conftest.py` itself defaults that variable during collection.
Neither pytest command proves unrestricted plugin discovery in a real service.
The explicit environment assertion, doctor and foreground server supply
separate discovery/root evidence.

CI job `hy4-contract` runs all registered HYV4 contract modules and the
runtime CLI tests with `suite=full`, no pytest filters, and no checkpoint
requirements, on the existing bw18/gfx936 runner. The tests are portable;
the runner's device allocation does not make these hardware validation.
HYV4 model/parser/quantization paths and the runtime helper route to this
job; full/nightly matrices include it. Existing model jobs remain registered.

## Service ownership and foreground lifecycle

Before each launch set `SERVED_MODEL` to the name in that launch and save the
process and memory baseline. Only proceed when
all eight selected devices are free; preserve an occupancy conflict as a
blocked run. Do not stop another service.

```bash
rocm-smi --showmeminfo vram --showuse | tee "$RUN_ROOT/${SERVED_MODEL:?}-memory-before.txt"
ps -eo pid,ppid,pgid,user,stat,lstart,cmd --sort=start_time > "$RUN_ROOT/$SERVED_MODEL-processes-before.txt"
ss -ltnp 'sport = :8000'
```

Use a dedicated interactive PTY for the server. In that PTY set
`RUN_ROOT` to the same newly created directory and run the environment
block's exports (do not create a second run directory). Execute one launch
below in the foreground, with its output retained by the PTY/session
recorder. No background service, nohup, or detached shell. If using a tool,
allocate `tty=true` and retain the returned session ID. Record each
service's PID, process group, descendants and start times in another terminal
while it is alive. Send Ctrl-C to that exact PTY after the requests; wait for
the foreground command to exit and record its exit status.

```bash
# Common foreground arguments; define in the server PTY.
common=(--host 127.0.0.1 --port 8000 --trust-remote-code
  --enable-prefix-caching --gpu-memory-utilization 0.95
  --max-model-len 4096 --max-num-seqs 16 --max-num-batched-tokens 4096
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}'
  --reasoning-parser hy_v4 --enable-auto-tool-choice --tool-call-parser hy_v4
  --seed 0)
```

TP8 target-only, default graph behavior:

```bash
export SERVED_MODEL=hy4-v0251-target
env -u VLLM_PLUGINS python3 -m vllm.entrypoints.cli.main serve "$MODEL" \
  "${common[@]}" --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 8 --moe-backend aiter
```

After complete teardown, TP8 MTP3, default graph behavior:

```bash
export SERVED_MODEL=hy4-v0251-mtp3
env -u VLLM_PLUGINS python3 -m vllm.entrypoints.cli.main serve "$MODEL" \
  "${common[@]}" --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 8 --moe-backend aiter \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

After complete teardown, TP8 MTP3 and native FP8 KV:

```bash
export SERVED_MODEL=hy4-v0251-fp8-kv
env -u VLLM_PLUGINS python3 -m vllm.entrypoints.cli.main serve "$MODEL" \
  "${common[@]}" --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 8 --moe-backend aiter --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Exact PP2/TP1/PCP4/DP1/DCP1 with stage-local EP4, FP8 KV and eager execution.
There is no speculative configuration on this command.

```bash
export SERVED_MODEL=hy4-v0251-pp2-pcp4
env -u VLLM_PLUGINS VLLM_PP_LAYER_PARTITION=41,37 \
  python3 -m vllm.entrypoints.cli.main serve "$MODEL" \
  "${common[@]}" --served-model-name "$SERVED_MODEL" \
  --pipeline-parallel-size 2 --tensor-parallel-size 1 \
  --prefill-context-parallel-size 4 --data-parallel-size 1 \
  --decode-context-parallel-size 1 --enable-expert-parallel \
  --all2all-backend deepep_high_throughput --moe-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 --enforce-eager
```

Representative DP8/TP1/EP8 (run separately from static EPLB):

```bash
export SERVED_MODEL=hy4-v0251-dp8-ep8
env -u VLLM_PLUGINS python3 -m vllm.entrypoints.cli.main serve "$MODEL" \
  "${common[@]}" --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
  --all2all-backend deepep_high_throughput --moe-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 --enforce-eager
```

For static load, create a deterministic identity placement from the supplied
configuration, explicitly not a recorded/optimized map. With 256 experts and
EP8 each rank owns 32 physical experts. There are 77 sparse layers, no
redundant experts, and the exact model key is `HYV4ForCausalLM`.

```bash
export STATIC_MAP="$RUN_ROOT/hy4-static-map.json"
python3 - <<'PY'
import json, os, pathlib
cfg = json.loads((pathlib.Path(os.environ["MODEL"]) / "config.json").read_text())
layers = sum(kind == "sparse" for kind in cfg["mlp_layer_types"])
experts = cfg["n_routed_experts"]
assert (cfg["num_hidden_layers"], layers, experts) == (78, 77, 256)
entry = dict(model_class="HYV4ForCausalLM", num_moe_layers=layers,
             num_logical_experts=experts, num_physical_experts=experts,
             num_redundant_experts=0,
             physical_to_logical_map=[list(range(experts)) for _ in range(layers)])
path = pathlib.Path(os.environ["STATIC_MAP"])
with path.open("x") as f:
    json.dump({"model_maps": {"HYV4ForCausalLM": entry}}, f)
from vllm_hcu.model_executor.layers.fused_moe.static_eplb import load_static_eplb_plan
plan = load_static_eplb_plan(path, model_key=entry["model_class"],
    expected_shape=(77, 256), num_logical_experts=256, num_redundant_experts=0)
print(plan.fingerprint())
PY
sha256sum "$STATIC_MAP"
export EPLB_CONFIG="$(python3 -c 'import json,os; print(json.dumps(dict(expert_map_path=os.environ["STATIC_MAP"],num_redundant_experts=0,static_dispatch_policy="nearest")))')"
export SERVED_MODEL=hy4-v0251-static-eplb
env -u VLLM_PLUGINS python3 -m vllm.entrypoints.cli.main serve "$MODEL" \
  "${common[@]}" --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
  --all2all-backend deepep_high_throughput --moe-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 --enforce-eager --enable-eplb \
  --eplb-config "$EPLB_CONFIG"
```

`--eplb-config` accepts HCU `expert_map_path` and
`static_dispatch_policy`; these become HCU sidecar controls while official
EPLB fields retain upstream validation. EP weight filtering must stay off
(the pinned default). HYV4 record/dynamic EPLB remains unsupported.
Task 11 must capture actual worker plan digests and binding order plus an
observed rearrangement count of zero. The current adapter does not emit a
dedicated count log: absence of a rearrangement message is insufficient.
Collect worker instrumentation if needed and retain it with the run; an HTTP
200 alone does not close the static EPLB evidence row.

## Requests and metric evidence

In the client terminal use the same pinned environment, `RUN_ROOT`, and the
exact running `SERVED_MODEL`. Wait for the foreground startup log to announce
readiness, then run:

```bash
curl --noproxy '*' --fail-with-body --max-time 30 http://127.0.0.1:8000/health
curl --noproxy '*' --fail-with-body --max-time 30 http://127.0.0.1:8000/v1/models
python3 - <<'PY' | curl --noproxy '*' --fail-with-body --max-time 300 \
  -H 'Content-Type: application/json' -d @- http://127.0.0.1:8000/v1/chat/completions
import json, os
print(json.dumps(dict(model=os.environ["SERVED_MODEL"],
    messages=[dict(role="user", content="What is the capital of France? Answer briefly.")],
    temperature=0, seed=0, max_tokens=128,
    chat_template_kwargs=dict(reasoning_effort="no_think"))))
PY
curl --noproxy '*' --fail-with-body http://127.0.0.1:8000/metrics \
  > "$RUN_ROOT/$SERVED_MODEL-metrics-before.txt"
python3 - <<'PY'
import json, os, pathlib
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL"], trust_remote_code=True)
prefix = "Paris is the capital of France. The Seine flows through Paris. "
while len(tokenizer.encode(prefix)) < 3100:
    prefix += "Paris has museums and bridges. Visitors travel along the Seine. "
for i in range(3):
    message = prefix + f"\nQuestion {i}: Name the city described above."
    messages = [dict(role="user", content=message)]
    token_count = len(tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, reasoning_effort="no_think"))
    assert 3000 <= token_count < 3800, token_count
    payload = dict(model=os.environ["SERVED_MODEL"], messages=messages,
        temperature=0, seed=0, max_tokens=128,
        chat_template_kwargs=dict(reasoning_effort="no_think"))
    path = pathlib.Path(os.environ["RUN_ROOT"]) / f"prefix-{i}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(i, token_count)
PY
for i in 0 1 2; do
  curl --noproxy '*' --fail-with-body --max-time 1800 \
    -H 'Content-Type: application/json' --data-binary "@$RUN_ROOT/prefix-$i.json" \
    http://127.0.0.1:8000/v1/chat/completions \
    > "$RUN_ROOT/$SERVED_MODEL-prefix-$i-response.json" || break
  curl --noproxy '*' --fail-with-body http://127.0.0.1:8000/metrics \
    > "$RUN_ROOT/$SERVED_MODEL-metrics-$i.txt"
done
curl --noproxy '*' --fail-with-body http://127.0.0.1:8000/health
rg 'vllm:(prefix_cache_hits|prefix_cache_queries|spec_decode_num_)' \
  "$RUN_ROOT/$SERVED_MODEL"-metrics-*.txt
```

Retain HTTP status, complete response JSON, token usage, finish reason and
actual text for every request. Later prefix-hit deltas must be positive.
For MTP retain drafted and accepted token deltas greater than zero and
`spec_decode_num_accepted_tokens_per_pos` for all three positions; compute
each position's rate against `spec_decode_num_drafts`. Graph logs must show
target capture and, for MTP, draft capture; a default CLI option is not
capture evidence. Require model class `HYV4ForCausalLM`, draft
`HYV4MTPModel`, Channel-FP8 selection and AITER lookup/final backend.
For PP require partition 41/37 and four PCP/EP ranks within each stage,
DeepEP high-throughput, DeepGEMM, and native FP8 KV. Preserve any missing
kernel, OOM, topology rejection, dtype/shape/slot error as a failed or blocked
run, with full traceback.

Optional offline harness commands use raw completion prompts, not the chat
protocol or EvalScope. They repeat one long prefix three times and close the
engine even if generation fails. Output alone does not prove cache hits or
MTP acceptance:

```bash
env -u VLLM_PLUGINS python3 -m tests.integration.model_runtime hy-v4-smoke \
  --model "$MODEL" --gpu-memory-utilization 0.95
env -u VLLM_PLUGINS python3 -m tests.integration.model_runtime hy-v4-mtp3-smoke \
  --model "$MODEL" --gpu-memory-utilization 0.95
env -u VLLM_PLUGINS python3 -m tests.integration.model_runtime hy-v4-fp8-kv-smoke \
  --model "$MODEL" --gpu-memory-utilization 0.95
```

Run each only after the preceding service/engine teardown and memory check.

## HumanEval first 32 numeric tasks

Prepare one immutable subset for both services, retain its source commit and
SHA-256, and verify exact numeric IDs. This downloads data, not model weights.
The installed EvalScope HumanEval adapter uses subset `openai_humaneval`,
test split, and `acc` aggregated as mean accuracy and Pass@1.

```bash
export HE_ROOT="$RUN_ROOT/humaneval32"
export HE_COMMIT="$(git ls-remote https://github.com/openai/human-eval.git HEAD | cut -f1)"
python3 - <<'PY'
import gzip, hashlib, json, os, pathlib, urllib.request
root = pathlib.Path(os.environ["HE_ROOT"])
root.mkdir()
commit = os.environ["HE_COMMIT"]
assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
url = f"https://raw.githubusercontent.com/openai/human-eval/{commit}/data/HumanEval.jsonl.gz"
with urllib.request.urlopen(url, timeout=60) as response:
    raw = response.read()
rows = sorted((json.loads(line) for line in gzip.decompress(raw).splitlines()),
              key=lambda row: int(row["task_id"].split("/")[-1]))[:32]
assert [row["task_id"] for row in rows] == [f"HumanEval/{i}" for i in range(32)]
(root / "test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
(root / "README.md").write_text(
    "---\nconfigs:\n- config_name: openai_humaneval\n"
    "  data_files:\n  - split: test\n    path: test.jsonl\n---\n")
(root / "source.json").write_text(json.dumps(dict(url=url, commit=commit,
    source_sha256=hashlib.sha256(raw).hexdigest())))
PY
sha256sum "$HE_ROOT/test.jsonl"
python3 -c 'import importlib.metadata as m; print("evalscope",m.version("evalscope"))'
```

Start a fresh TP8 target-only service using the launch above. After it passes
health, evaluate with the exact target name. Then stop it, verify teardown,
start a fresh TP8 MTP3 service, and repeat this block with
`SERVED_MODEL=hy4-v0251-mtp3`. Never reuse an output directory or predictions.

```bash
export FRESH_RESULT_DIR="$(mktemp -d "$RUN_ROOT/$SERVED_MODEL-eval.XXXXXX")"
export HE_DATASET_ARGS="$(python3 -c 'import json,os; print(json.dumps({"humaneval":{"local_path":os.environ["HE_ROOT"]}}))')"
env -u VLLM_PLUGINS -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  evalscope eval --model "$SERVED_MODEL" --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY --eval-type openai_api \
  --generation-config '{"temperature":0,"do_sample":false,"seed":0,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \
  --eval-batch-size 1 --timeout 1800 --limit 32 --datasets humaneval \
  --dataset-args "$HE_DATASET_ARGS" --work-dir "$FRESH_RESULT_DIR" --no-timestamp
python3 - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ["FRESH_RESULT_DIR"])
for kind in ("predictions", "reviews"):
    paths = list((root / kind).rglob("*.jsonl"))
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()
            if line.strip()]
    assert len(rows) == 32, (kind, len(rows), paths)
    print(kind, len(rows), paths)
print("reports", list((root / "reports").rglob("*.json")))
PY
curl --noproxy '*' --fail-with-body http://127.0.0.1:8000/metrics \
  > "$FRESH_RESULT_DIR/metrics-after.txt"
```

Check all review/prediction task IDs against HumanEval/0–31, unique and
complete. Record accuracy, Pass@1, HTTP/evaluator errors, finish reasons,
output lengths, and all per-task pass/fail flips target versus MTP.
Save both report trees and compare the same subset digest and generation
options. Nonempty output and HTTP 200 are not accuracy results.

## Exact teardown checks

During each service run record its ownership set in the client terminal:
the listening PID must be inspected and confirmed to be this run's
`SERVED_MODEL` before setting `SERVICE_PID`.

```bash
ss -ltnp 'sport = :8000'
# Set SERVICE_PID to the inspected listening PID, not a name-based match.
read -r -p 'Inspected listening PID for this service: ' SERVICE_PID
export SERVICE_PID
python3 - <<'PY'
import json, os, pathlib, psutil
process = psutil.Process(int(os.environ["SERVICE_PID"]))
assert os.environ["SERVED_MODEL"] in process.cmdline(), process.cmdline()
owned = [process, *process.children(recursive=True)]
records = [dict(pid=p.pid, created=p.create_time(), pgid=os.getpgid(p.pid),
                command=p.cmdline()) for p in owned]
path = pathlib.Path(os.environ["RUN_ROOT"]) / (os.environ["SERVED_MODEL"] + "-owned.json")
path.write_text(json.dumps(records, indent=2))
print(path, records)
PY
```

Refresh that ownership snapshot before Ctrl-C. Send Ctrl-C only to the
recorded foreground PTY, wait for its return, and retain the exit code.
Then verify PID/start-time identity rather than treating PID reuse as an
orphan. The following checks are read-only; do not kill broad name matches.

```bash
python3 - <<'PY'
import json, os, pathlib, psutil
path = pathlib.Path(os.environ["RUN_ROOT"]) / (os.environ["SERVED_MODEL"] + "-owned.json")
survivors = []
for row in json.loads(path.read_text()):
    try:
        p = psutil.Process(row["pid"])
        if abs(p.create_time() - row["created"]) < 0.01:
            survivors.append(row)
    except psutil.NoSuchProcess:
        pass
assert not survivors, survivors
PY
ss -ltnp 'sport = :8000'
rocm-smi --showmeminfo vram --showuse | tee "$RUN_ROOT/$SERVED_MODEL-memory-after.txt"
ps -eo pid,ppid,pgid,user,stat,lstart,cmd --sort=start_time \
  > "$RUN_ROOT/$SERVED_MODEL-processes-after.txt"
```

Require no listener on port 8000, no owned API/EngineCore/Worker/DP-rank
processes, and VRAM returned to the recorded free-device baseline. Inspect
the before/after process lists for reparented workers. If a process or memory
allocation survives, retain its ownership evidence and resolve that exact
service before the next run. A closed HTTP port alone is not teardown.

## Deferred review observations

These are carried forward for final review; Task 10 does not claim fixes:

- Task 2: unused alternate tool-stream path.
- Task 7: record-file publication lacks parent-directory fsync.
- Task 8: explicit Mooncake layer-count boundaries 78/41/77 need coverage review.
- Task 9: linear weight-loader bound-owner introspection.
- Inherited Torch/JIT deprecation warnings and installed plugin version metadata
  concern described above.
