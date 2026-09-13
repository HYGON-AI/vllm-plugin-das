# HYV4 v0.25.1 clean integration validation

This contains the Task 10 execution protocol, Task 11 observed results, and
the focused Task 11A static-loader, Task 11B PCP, Task 11C static-memory,
and Task 13 exact PP2/PCP4 native MTP3 fixes/reruns.
CPU tests establish contracts; they do not establish checkpoint loading,
accelerator arithmetic, graph correctness, distributed serving, or accuracy.
Each result state is updated only after retaining its evidence. Task 11B fixes
the PP2/PCP4 correctness failure and passes its original exact gate. DP8 memory
failures mean this matrix is still not an overall integration pass. Task 11A
fixes the static loader-owner rejection; Task 11C removes the false KV budget
and changes the exact static gate to an early, safe capacity failure.
The separate plain-PP diagnostic stall below
remains unresolved.

Task 11 provenance capture began at 2026-09-12T21:11:05Z (the host renders
local log timestamps as 2026-09-13 UTC+08:00), from candidate
`d1dc00f31fba9e850d6e1c8f1ac8d14870afa5ad`. Artifacts are under
`/models/validation-logs/hy4-v0251-clean-20260912T211105Z` and
`/models/eval-results/hy4-v0251-clean-20260912T211105Z`.

The provenance-review Minor is resolved: every launch captures
`aiter=0.1.5+dtk2604.torch2110.2609021352.gab70b0`,
`lightop=0.6.0+das.dtk2604`, and
`deep-ep=1.1.0+dtk2604.torch2110.2609022105.gbb2ba6`.
Model `config.json` SHA-256 is
`d4a648cb09bb89f4b8778e60629e43618f1abb581ef3aa38bd67b2b2cd998441`;
`model.safetensors.index.json` SHA-256 is
`9cc56bfc4527ee64d6e2cf05d6c4eb7c2d658a6c7431bc52265cf90e93c8a715`.
This host has no `ss`; the executable listener checks below use `psutil`.

## Hardware result matrix

| Gate | State | Required evidence |
| --- | --- | --- |
| TP8 Channel-FP8 target, AITER, default graphs | PASS | 131 shards loaded; AITER configs; PIECEWISE/FULL capture; HTTP 200 short and 3×3145-token prompts; clean teardown |
| TP8 MTP3, AITER, default graphs | PASS | Target/draft capture; HTTP 200; 114 drafted, 81 accepted tokens; clean teardown |
| TP8 MTP3, FP8 E4M3 KV, repeated prefix | PASS | 3×3145-token prompts; hits 0/3008/6016; coherent output; HTTP 200; clean teardown |
| PP2/TP1/PCP4/DP1/EP4, DeepEP HT, DeepGEMM, eager | PASS — Task 11B | Original short + three 3145-token requests correct, HTTP 200/stop; exact 41,37/FP8 KV topology; health 200; clean teardown; DCP disabled/default |
| Exact PP2/TP1/PCP4/DP1/EP4 + native MTP3 | PASS — Task 13 | No-observer exact 41,37/eager/HT/DeepGEMM/FP8 command; short + three identical 3145-token requests and concurrent short/long all match target-only; 18 drafted/8 accepted, positions 3/3/2; health 200 and clean teardown; shutdown warning retained below |
| DP8/TP1/EP8, DeepEP HT, DeepGEMM | FAIL — capacity | All eight ranks loaded; KV budget −0.7 GiB; no readiness; exit 1; clean teardown |
| DP8/EP8 static offline EPLB load, exact 0.95/4096 gate | SAFE CAPACITY FAIL — Task 11C | All eight ranks loaded 125.69 GiB; positive 3.19 GiB non-torch and 8.62 GiB peak; KV −0.71 GiB; rejected before cache allocation; zero NIXL registration/late OOM; exit 1; clean teardown |
| DP8/EP8 static offline EPLB, constrained batch tokens 2048 | PASS — constrained Task 11C | Model length 4096/0.95/max sequences 16 unchanged; all-rank map/fingerprint/Gloo owner and zero rearrangement; observed and no-observer 36-request runs pass; final teardown passes after one owned-worker cleanup described below |
| HumanEval/0–31 target-only | PASS | 32 predictions/reviews; Accuracy=Pass@1=100%; 0 errors; all stop; clean teardown |
| HumanEval/0–31 MTP3 | PASS | 32 predictions/reviews; Accuracy=Pass@1=100%; 0 errors/flips; acceptance 93.03%; clean teardown |
| Custom packed W4A8 | not run | Compatible checkpoint and device numerical/runtime evidence |
| Native packed W4A8 | not run | Compatible checkpoint and device numerical/runtime evidence |
| Two-node Mooncake P/D | not run | Two-node transfer and generation evidence |

Custom/native W4A8 have no compatible checkpoint or hardware validation.
DCP is not tested, per the updated Task 11 scope. A recorded
`decode_context_parallel_size=1` is only the disabled/default setting;
there are no DCP hardware runs, metrics, or expanded runtime combinations.
Configuration rejection tests do not establish DCP runtime support.
The supplied checkpoint is Channel-FP8, not a W4A8 substitute. Mooncake has
no two-node runtime validation. Static SlimQuant EPLB and HYV4 dynamic/record
EPLB are unsupported. MTP+PCP is supported only for the exact Task 13
PP2/TP1/PCP4/DP1/EP4/41,37/eager/DeepEP HT/DeepGEMM/FP8/native-MTP3
configuration; all neighboring HYV4 speculative PCP configurations remain
fail-closed. Generic offline record tests do not
authorize HYV4 record-mode serving. Do not replace failures by changing the
model, backend, topology, quantization, or graph mode.
Separate merge blockers remain: packed W4A8 expert extent validation and
native BF16/FP16 MTP source-format adaptation. This checkpoint has
`mtp_quant_algo=None`; Task 13 does not fix or validate those paths.

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
for distribution in ("aiter", "lightop", "deep-ep"):
    print(distribution, md.version(distribution))
PY
sha256sum "$MODEL/config.json" "$MODEL/model.safetensors.index.json"
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
python3 -c 'import psutil; print([(c.pid, c.laddr) for c in psutil.net_connections(kind="tcp") if c.status == "LISTEN" and c.laddr.port == 8000])'
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

Exact PP2/TP1/PCP4/DP1 with stage-local EP4, FP8 KV and eager execution.
The explicit decode-context size of 1 only records DCP disabled/default;
DCP is not tested.
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

After complete PID/port/device teardown, the exact Task 13 MTP3 command
adds only the speculative configuration to the preceding PP2/PCP4 launch.
Run in the same foreground-owned PTY lifecycle; do not leave both services
running. DCP remains disabled/default and NIXL/UCX is excluded.

```bash
export SERVED_MODEL=hy4-v0251-pp2-pcp4
env -u VLLM_PLUGINS VLLM_PP_LAYER_PARTITION=41,37 \
  python3 -m vllm.entrypoints.cli.main serve "$MODEL" \
  "${common[@]}" --served-model-name "$SERVED_MODEL" \
  --pipeline-parallel-size 2 --tensor-parallel-size 1 \
  --prefill-context-parallel-size 4 --data-parallel-size 1 \
  --decode-context-parallel-size 1 --enable-expert-parallel \
  --all2all-backend deepep_high_throughput --moe-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 --enforce-eager \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
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
    token_count = len(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
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
python3 -c 'import psutil; print([(c.pid, c.laddr) for c in psutil.net_connections(kind="tcp") if c.status == "LISTEN" and c.laddr.port == 8000])'
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
python3 -c 'import psutil; print([(c.pid, c.laddr) for c in psutil.net_connections(kind="tcp") if c.status == "LISTEN" and c.laddr.port == 8000])'
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

## Task 11 observed runs

### TP8 target, default graphs — PASS

Evidence directory: `01b-target` under the validation-log root above.
The service ran from 2026-09-12T21:13:04Z through 21:26:53Z, API PID 1017407.
All eight TP workers loaded the 131-shard checkpoint. The log identifies
`ChannelWiseTorchFP8ScaledMMLinearKernel`, the AITER FP8 MoE backend, and
AITER `gfx938/fp8_w8a8/E=256,N=256,dtype=fp8_w8a8.json` configurations
(including the bottom-layer variant). PIECEWISE and FULL graph capture
completed; each worker reported 24 seconds and 0.28 GiB for capture.

The final request batch in `requests-final/` returned HTTP 200 for health,
models, metrics, short chat, and three long prompts. Short chat answered
“Paris.” (34 prompt tokens, 3 output tokens). All long prompts had 3145
tokens; outputs identified Paris with 2, 53 and 60 completion tokens, all
with finish reason `stop`. Prefix-cache hits were 0, 3072 and 6144 after
the three requests. Health remained HTTP 200. Full payloads, responses,
headers and timing are retained, with `request-summary.json` aggregating them.

The initial client-side long-prompt check counted the two keys of this
Transformers build's default `BatchEncoding`; explicit `return_dict=False`
corrected the token-count check. Its original evidence remains intact and
the server configuration was unchanged. A preliminary wrapper attempt in
`01-target` was stopped before worker initialization and is separately
recorded in the Task 11 report; it is not a model failure.

Ctrl-C was sent only to the owned foreground PTY at 21:26:51Z. The pinned
runtime used its default abort shutdown, including its own process-manager
cleanup. The command exited 0. `teardown.log` reports no owned PID/start-time
survivors, no listener on port 8000, and all eight devices at 0% and 2 MiB.

### TP8 MTP3, default graphs — PASS

Evidence: `02-mtp3`; API PID 1056982; UTC 21:27:32–21:32:09 on
2026-09-12. `HYV4MTPModel` resolved; all eight workers completed target
PIECEWISE/FULL and speculator prefill/decode graph capture (42 seconds,
0.88 GiB each). Channel-FP8 linear and AITER FP8 MoE selection were retained.
Health, models, short chat, three 3145-token prompts and post-reuse health
all returned HTTP 200. Outputs coherently identified Paris; completion
lengths were 3, 2, 54 and 60, all `stop`.

Final cumulative metrics: 38 drafts, 114 drafted tokens, 81 accepted tokens
(71.05%); accepted by position 34/26/21, or 89.47%/68.42%/55.26% of drafts.
These are cumulative counters, distinct from the service's interval rates.
Prefix hits reached 6016. Ctrl-C targeted only this foreground PTY; exit 0,
no owned process survivors or listener, and all eight cards back to 0%/2 MiB
(`teardown.log`).

### TP8 MTP3, FP8 KV and repeated prefix — PASS

Evidence: `03-mtp3-fp8-kv`; API PID 1074407; UTC 21:32:32–21:36:30.
The configuration retained TP8, AITER, MTP3 and default graphs and explicitly
selected `fp8_e4m3` KV. Target/draft capture completed on all eight workers
(41 seconds, 0.90 GiB each). All readiness, short, long-prefix and health
checks returned HTTP 200. Long prompts were 3145 tokens each; outputs were
Paris/Paris/a coherent Paris explanation (2/2/52 tokens, all `stop`). Prefix
hits rose 0→3008→6016; no cache shape, dtype or slot error appeared.

Final counters: 22 drafts, 66 drafted tokens, 36 accepted (54.55%); accepted
by position 15/11/10 (68.18%/50.00%/45.45% of drafts). Short chat used 3 output
tokens and stopped normally. Owned-PTY Ctrl-C at 21:36:27 was followed by
exit 0 and `TEARDOWN_PASS`: no owned survivors/listener, eight cards 0%/2 MiB.

### PP2+PCP4+EP4 eager — FAIL, long-prefill correctness

Evidence: `04-pp2-pcp4`; API PID 1092968; UTC 21:36:56–21:41:27.
The exact requested topology/backend ran: partition `41,37`, two PP stages,
four stage-local PCP/EP workers each, TP1/DP1, DeepEP HT, DeepGEMM and FP8 KV.
Logs name `DeepEPHTAll2AllManager` and
`DeepEPDeepGemmContiguousExperts with DeepGEMM HT path`. All eight workers
loaded; stage-0 workers reported 110.36 GiB, stage-1 workers 101.58 GiB.
DCP was disabled/default and was not tested.

Short chat correctly answered Paris (3 tokens, `stop`). All three long
requests returned HTTP 200 but generated unrelated/repetitive mixed-language
text, each 128 tokens with `finish_reason=length`; this is a correctness
failure, not a passing service gate. Post-request health remained 200.
`baseline-comparison.json` retains exact texts, hashes and usage and proves
each corresponding request is identical to the successful TP8 baseline
except the served-model name (same template kwargs, seed, temperature,
128-token limit and 3145-token prompt). The three controlled suffixes are
different from one another, but each matches its baseline counterpart.

The launch partition and all eight PP/PCP/EP worker identities are recorded.
No explicit NaN, collective, index/cache error, traceback or ERROR was found
in the pre-shutdown log; this does not prove numerical intermediates are
valid. Root cause is not established, and no implementation, topology or
backend was changed. Owned-PTY Ctrl-C at 21:41:24 led to exit 0 and full
`TEARDOWN_PASS` (no owned survivors/listener, eight cards 0%/2 MiB).

### DP8+EP8 eager — FAIL, environment capacity

Evidence: `05-dp8-ep8`; launcher PID 1110663; UTC 21:42:06–21:44:56.
All eight DP/EP workers loaded 125.69 GiB each. DeepEP HT prepare/finalize
and DeepGEMM HT experts were selected. At the required 0.95 GPU utilization,
4096-token model/batch limits and 16 sequences, the log reports
`Available KV cache memory: -0.7 GiB`, followed by
`ValueError: No available memory for the cache blocks` on all eight engines.
The service exited 1 before readiness; no model request was sent.

This is the observed capacity limit for this exact checkpoint/environment,
not evidence that DeepEP itself rejected the topology. No memory, batch,
length or backend setting was changed. Runtime failure cleanup left no
owned survivors/listener and all eight cards at 0%/2 MiB (`teardown.log`).
The owned readiness helper was stopped by its verified exact PID after the
service exited; its identity and SIGTERM are retained in `readiness-stop.json`.

### DP8/EP8 static EPLB load — FAIL, implementation loader-owner guard

Evidence: `06-static-eplb`; launcher PID 1132324; UTC 21:45:44–21:47:41.
The retained synthetic identity map has 77×256 entries and SHA-256
`4a37b1fe4de680fe0fd4d630156c07433052796828d0a1a1744b4892b70768e8`;
current-loader schema validation passed in `static-map-preflight.log`.
It is not a recorded or optimized map. The exact DP8 launch added only
`--enable-eplb` and the documented static-load configuration.

At 21:47:29 all eight workers raised
`ValueError: Static EPLB parameter has an unsupported loader owner` from
`bind_static_eplb_plan` (`static_eplb.py:252`), reached through the
`initialize_model` pre-load hook. Thus binding failed before checkpoint
weight loading and KV initialization; this is independent of the previous
DP8 capacity failure. No ready service, HTTP inference, successful runtime
plan binding, or zero-rearrangement evidence exists for that Task 11 run.
The offending parameter identity was not logged then; Task 11A establishes
the source/owner reproduction below.

The process exited 1 without an assistant-issued service signal. Teardown
verified all 24 logged API/engine/worker PIDs absent, no owned identity
survivors/listener, and eight cards 0%/2 MiB. No code or backend was changed.

### Task 11A static-loader fix — PASS for binding/load; service FAIL at KV allocation

Evidence root:
`/models/validation-logs/hy4-static-eplb-fix-20260912T222135Z`.
The sole production change selects loader-owning Parameters from current
`RoutedExperts.get_expert_weights()` storage-sharing views. HYV4's
`expert_bias` is registered again as `e_score_correction_bias`, with no
`weight_loader`; the pinned owner explicitly excludes this router parameter
from rearrangeable expert weights. Bias/global-state identity, values,
loaders and checkpoint-ledger ownership remain unchanged. Actual expert
weights still fail closed on foreign/unbound/missing loaders, now naming the
rejected parameter. All targets are checked before cross-rank verification
and publication. No backend owner or installed/model file was changed.

Observed TDD: 10 failed/37 passed before the adapter edit, then 47 passed.
Expanded owner suites: 375 passed. All 35 branch-changed Python test files,
including helpers, ran unfiltered: 1,076 passed. The explicit pinned-root
contract runner passed 2,160 tests (0 failures/errors/skips); doctor had
7 PASS checks; compileall and diff checks were clean. The initial unfiltered
attempt's missing legacy source-root environment variable is retained in
`changed.log`; the identical full list passed in `changed-final.log` after
setting `VLLM_V0251_SOURCE_ROOT` to the pinned root, without test changes.

The foreground rerun (`static-rerun/service.typescript`) ran at UTC
**2026-09-12 22:29:09–22:32:02**, PID/PGID **1223293**. It used baseline
`7f68fac3bd6a2b0960e426ac96705e6325024572` plus the retained
`candidate-source.diff`; the adapter SHA-256 was
`f7309c268b2f52feba39fa0796d02f6245830551b4d2877b9e15a4293a2e1420`.
`candidate-source.sha256` identifies all production/test inputs.
`rerun-comparison.json` verifies the original Task 11 ARGV and recorded
environment are identical. The 77×256 identity map and its hash are unchanged.
Both preflight and launcher observed all eight cards free (0%/2 MiB), no
service/listener was present, and the launcher reasserted pinned ABI,
dependency versions, source roots and checkpoint hashes. DCP was not tested;
size 1 remains disabled/default.

Shard loading began at 22:30:54 and completed 131/131. At 22:31:34–35,
**all DP/EP ranks 0–7** reported **125.69 GiB loaded** (43.67–44.08s).
The old owner rejection occurred on zero ranks. DeepEP HT prepare/finalize
and DeepGEMM HT experts were selected; EPLB initialized its
`NixlEplbCommunicator`. This satisfies the focused loader-fix hardware gate.

The service nevertheless failed before readiness. At 22:31:48, the profiler
reported **88.18 GiB available KV memory** and engines planned
**1,755,200–1,755,264 cache tokens**. All eight workers then raised
`torch.OutOfMemoryError` while the pinned MRV2
`gpu/attn_utils.py:186` called `torch.zeros` for a **1.07 GiB** KV tensor.
Diagnostics simultaneously reported **133.75–134.82 GiB PyTorch allocated**
and **88.59–89.30 GiB free**, on 143.98 GiB devices. This is a distinct
KV-allocation/memory-accounting follow-up, **not** the earlier DP8
`No available memory for the cache blocks`/−0.7 GiB failure. Its deeper cause
is not established; communicator initialization preceding it does not prove
causation. No memory limit, allocator, topology or backend was changed to
hide the failure. There is no static-serving accuracy or runtime
zero-rearrangement claim.

All 17 bounded proxy-free health probes failed to connect; the service
exited 1 naturally, with no assistant-issued service signal. The readiness
helper exited on the owned process's disappearance. `teardown.log` verifies
all 24 logged API/engine/worker PIDs and all saved owned process identities
absent, no port-8000 listener, and every card at 0%/2 MiB. Raw traces for all
eight KV-allocation failures are retained in `rerun-comparison.json` and the
full typescript. The original Task 11 failure evidence is preserved.

### HumanEval/0–31 target-only — PASS for this 32-task slice

Fresh-service evidence: `07-humaneval-target`; PID 1144718. Evaluation
results are in `target/` under the evaluation root; `target-summary.json`
independently checks 32 unique predictions and 32 reviews for exactly tasks
HumanEval/0 through HumanEval/31. Retained input JSONL SHA-256:
`6906d173a121e247cc03340ec685d66019a6aa8949276a368b796b819540bc2c`.
The dataset comes from openai/human-eval commit
`6d43fb980f9fee3c892a914eda09951f772ad10d`; original compressed data is retained.

EvalScope 1.9.1 ran at UTC 21:51:36–22:00:48 with fresh outputs, no_think,
temperature/seed 0, max_tokens 2048 and batch size 1. Accuracy and Pass@1
were both 1.0 (32/32). There were zero model/API errors or judge failures;
all 32 finish reasons were `stop`. Output tokens: total 4328, mean 135.25,
median 122.5, range 46–285. Per-task outputs, lengths and execution results
are preserved in predictions/reviews and the independent summary.

The fresh service retained TP8/AITER/default graphs and default KV, completed
graph capture and returned HTTP 200 for pre/post checks. Ctrl-C targeted only
its PTY; exit 0 and full process/listener/device teardown passed. This small
slice is not a full HumanEval or general accuracy claim; MTP comparison follows.

### HumanEval/0–31 MTP3 and comparison — PASS for this 32-task slice

Fresh-service evidence: `08-humaneval-mtp3`; PID 1173421; UTC
22:01:22–22:08:58. Target and draft graph capture completed under the same
TP8/AITER/default-KV/default-graph conditions as the validated MTP3 smoke.
EvalScope ran at 22:04:50–22:08:45 into fresh `mtp3/` results. Independent
`mtp3-summary.json` and `comparison.json` verify the complete data below.

| HumanEval/0–31 result | Target-only | MTP3 |
| --- | ---: | ---: |
| Unique predictions / reviews | 32 / 32 | 32 / 32 |
| Accuracy / Pass@1 | 100% / 100% | 100% / 100% |
| Model/API errors / judge failures | 0 / 0 | 0 / 0 |
| Finish reason `stop` / length-limited | 32 / 0 | 32 / 0 |
| Total output tokens | 4328 | 4409 |
| Mean / median output tokens | 135.25 / 122.5 | 137.78125 / 125 |
| Minimum–maximum output tokens | 46–285 | 47–299 |

Both server logs contain exactly 32 chat HTTP-200 responses. Evaluation
arguments differ only in served-model name and output directory; all retained
user messages match after excluding EvalScope's internal generated message
ID. Per-sample correctness flips: **0** (neither pass→fail nor fail→pass).
Exact output text matches on 25/32 tasks; seven differ in text while both
outputs pass. All 32 per-task comparisons, output hashes, lengths, finish
reasons and judge results are retained; this is not a bitwise-output claim.

MTP cumulative metrics began at zero: 1171 drafts, 3513 drafted tokens,
3268 accepted tokens, overall acceptance 93.0259%. Accepted by positions
0/1/2: 1149/1093/1026, corresponding to 98.1213%/93.3390%/87.6174% of drafts.
These are full-evaluation counters, not interval-rate snapshots. Post-run
metrics and health returned 200; owned-PTY Ctrl-C at 22:08:55 led to exit 0
and `TEARDOWN_PASS` with no owned/logged survivors, no listener and eight
cards 0%/2 MiB. These results do not close the independent distributed
implementation failures, capacity blocker, W4A8 or Mooncake gaps above.

## Task 11 closing checks

Only this validation document changed in the tracked repository. Pinned-env
`python3 -m compileall -q vllm_hcu tests`, `git diff --check`, and
`git diff origin/v0.25.1 --check` returned 0. All 16 Bash command blocks passed
`bash -n`. The final evidence audit rechecked launch roots/dependencies/hash
provenance, the eleven matrix states, all owned/logged service PIDs, port 8000
and all eight devices. Every service teardown passed; the final devices were
0%/2 MiB. These static/evidence checks do not turn the three failed hardware
rows into passes. Full details and historical setup/audit-helper corrections
are retained in the Task 11 report and validation-log directory.

## Task 11B PCP long-prefill diagnosis

Baseline `7205e8d71f173ecce5175a1947779495eec77065`; pinned ABI, checkpoint,
max length 4096 and original `41,37` partition were preserved. Artifacts:
`/models/validation-logs/hy4-pp-pcp-quality-20260912T224350Z`.
The directory suffix is an identifier; authoritative UTC starts are in the
service logs, beginning at 2026-09-12T22:42:01Z. DCP is not tested; size 1
remains disabled/default. These controls are diagnostic, not substitute gates.

| Diagnostic | Observation |
| --- | --- |
| Exact PP2/TP1/PCP4/EP4, DeepEP HT/DeepGEMM/FP8/eager | Cold sampled prompts of 34–2106 tokens correct; 2310/2802/3150 and original 3145 fail with repetitive 128-token/length outputs. Warm cache shifts the observed transition. |
| PP2/TP4, PCP1, no EP | Cold sampled prompts through 2802 correct; 3150 stalls after 127 generated tokens. PP0 waits on executor queue; PP1 waits on PP receive. This separate runtime stall is inconclusive for long quality, not a pass or a fixed issue. |
| PP1/TP2/PCP4/EP8, same backends | Reproduces the cold 2310+ failure; PP transfer and stage 41 boundary are not necessary for it. |
| Exact PP/PCP topology, AITER MoE only | Original short correct; all three original 3145-token prompts fail with 128/length. DeepGEMM alone is not necessary for failure. |
| Real CPU PCP planner/restore | Four ranks at 34/2106/2310/3145, cached 498 and decode preserve token ordering/restore. |
| Standalone gfx938 MQA/top-k | 52 sampled real-layout rows, max absolute reference-logit error 0.00048828125, top-k overlap 1.0, invalid indices 0. |
| Read-only runtime owner/cache observer | All 8 workers select base `SparseAttnIndexer.forward_hip`; 336 indexer calls/1008 sampled rows expose missing cache keys, max own-K/cache error 5.0. |

Root cause: HYV4 constructed the base indexer rather than the existing
PCP-aware V32 owner. The base receives local K with global rank-ordered cache
slots and omits the PCP K/slot gather. Short contexts mask the problem because
top-k 2048 selects every valid token; longer contexts require correct key scores.
The change selects the existing V32 owner only for PCP>1, through the canonical
module exchange; PCP1 dispatch, PP transport, layer ownership, metadata
algorithms, kernels and MoE policies are unchanged.

The first observer filtered V32 only and therefore captured no indexer rows;
it is retained as an explicitly corrected diagnostic, not numerical evidence.
All diagnostic services exited 0 with full 8-device/process/port teardown.
For the plain-PP stalled request and deliberately interrupted first-observer
request, only identity-verified task-owned curl clients were terminated after
owned-PTY shutdown; no unrelated process was signalled. Raw failed/transient
checks and subsequent complete teardown evidence are retained.

Constructed-Indexer regression: corrected behavioral RED 17 failed/4 passed;
GREEN 21 passed. Tests use real model construction, HIP dispatch and PCP gather,
doubling only accelerator allocation/kernel leaves in an isolated process.

### Original exact post-fix gate — PASS

Evidence: `06-exact-final/` under the Task 11B artifact root. Foreground service
PID/PGID 1389576 ran at 2026-09-12T23:27:55Z–23:31:25Z. Its complete ARGV is
identical to the original failing run: PP2/TP1/PCP4/DP1, stage-local EP4,
`41,37`, DeepEP HT, DeepGEMM, FP8 E4M3 KV, eager, max length 4096 and memory
utilization 0.95. No diagnostic hooks or backend/topology substitutions.
All eight worker ranks and source partition `[0,41)`/`[41,78)` are verified.

| Original request | Prompt tokens | Completion tokens | Finish | Reviewed output |
| --- | ---: | ---: | --- | --- |
| Short | 34 | 3 | stop | `Paris.` |
| Prefix 0 | 3145 | 2 | stop | `Paris` |
| Prefix 1 | 3145 | 2 | stop | `Paris` |
| Prefix 2 | 3145 | 60 | stop | Correctly identifies Paris and coherently explains the source text. |

All four requests returned HTTP 200 and match their corresponding retained
TP8 requests exactly except served-model name. The three original long cases
retain their original question ordinals; this is not a claim that their
request bodies are identical to one another. Full raw text, request/response
hashes, token counts and comparison are in `baseline-comparison.json` and
`acceptance-audit.json`. This is correctness evidence, not bitwise output parity.
Post-reuse health returned 200; no NaN/collective/index/cache errors were found.
Owned-PTY Ctrl-C at 23:31:23Z led to exit 0. `teardown.log` verifies no saved
or logged PIDs, no port 8000 listener, and all eight devices at 0%/2 MiB.
DCP remains disabled/default and was not tested.

Final gates: focused 21 passed; neighbors 692 passed; all 36 changed Python
test modules unfiltered, 1097 passed; complete pinned contract 2160 passed;
CI selector 70 passed; doctor 7 PASS/49 callbacks. The initially missing new
test-module CI registration was observed RED, added as one literal entry and
verified by the full selector and contract rerun. Compileall and diff checks
pass. Remaining DP8/static-KV capacity failures, the plain-PP diagnostic stall,
W4A8/Mooncake hardware gaps, unsupported modes and deferred review observations
are unchanged; this focused pass is not an overall integration pass.

## Task 11C static EPLB memory diagnosis

Baseline `eff75c6f2062cf69c30847a2ea4f80324424c719`; evidence root:
`/models/validation-logs/hy4-static-memory-20260912T234254Z`.
The unchanged static DP8/EP8 command with a read-only call/return observer
reproduced the late KV OOM on all eight ranks (`01-observed`). At NIXL/RIXL
`_init_registered_buffers`, reported free memory increased by **88.890625 GiB**
while the device index, total memory and PyTorch allocated/reserved/peak
counters stayed unchanged. For device 2, free memory jumped from 10.597656
to 99.488281 GiB with 123.865174 GiB still allocated by PyTorch.

The resulting negative non-PyTorch increase inflated the KV budget. Device 0
reported weights 125.689453 GiB, activation peak increase 8.619878 GiB,
non-PyTorch increase −85.703125 GiB and approximately 88.18 GiB available KV.
Ordinary DP8 under the identical observer (`02-plain-observed`) had no
registration jump: device 1 reported a positive 3.179688 GiB non-PyTorch
increase, an 8.620533 GiB activation increase and a −756469760-byte KV budget.
Both diagnostic services exited 1 naturally and passed complete teardown.
`diagnosis.json` retains all 16 rank records and raw snapshots;
`diagnosis-audit.log` verifies their formulas and registration boundaries.

The static adapter now selects the **official Gloo staged communicator** only
while constructing an already-validated direct-load state. Its constructor
does not register all expert weights or allocate transfer staging; static
step/rearrange guards make transfer execution unreachable. The official state,
one-layer expert buffers, maps and shared tensors are retained. No fake
communicator, state-constructor copy, memory clamp or guessed margin is used.
Even an explicit static transfer preference uses this construction policy;
its original value is restored in `finally`. Unresolved/unknown preferences
fail closed. Nonstatic, dynamic and record configurations retain their
original communicator, including explicit choices. This does not fix generic
dynamic/record NIXL device accounting or authorize HYV4 dynamic/record mode.
NIXL/UCX functionality is not added or accepted in this scope; no dynamic
NIXL repair or runtime validation is performed. The pinned generic implementation
is retained untouched. Registration observations above are causal diagnostics,
and the post-fix zero-registration check validates only the static Gloo path.

The observer records scalar counters and CPU plan metadata without replacing
methods, synchronizing devices or retaining tensors. Instrumented diagnostics
and no-observer acceptance runs are recorded separately. DCP remains disabled
by default and is not tested.

### Exact static post-fix gate — SAFE CAPACITY FAIL

`03-exact-final/`: no observer, original exact ARGV, PID/PGID 1489731,
2026-09-13T00:51:12Z–00:54:00Z. All eight ranks load the supplied weights
and report non-KV usage 137.5 GiB: weights 125.69 GiB, activation increase
8.62 GiB, positive non-torch increase 3.19 GiB. Requested memory remains
136.78 GiB (143.984375 × 0.95); returned KV budget is −0.71 GiB.
The unchanged parent formula preserves all positive reservations, rejects
before KV allocation and never late-OOMs. This is not a serving PASS.
The actual communicator is official Gloo; NIXL/UCX registration log count
is zero. No memory clamp, explicit KV override or EP weight filtering.
Natural exit 1; no assistant-issued service signal. `teardown.log` confirms
no saved/logged processes or port-8000 listener, all eight cards 0%/2 MiB.

Only after this genuine capacity result, a separate bounded runtime probe
reduces `max-num-batched-tokens` from 4096 to 2048. Model length 4096,
max sequences 16, utilization 0.95, map, DP8/TP1/EP8 topology, DeepEP HT,
DeepGEMM, FP8 KV and eager mode remain unchanged. Its result is recorded
separately; it cannot replace the failed exact resource gate.

### Constrained static semantics (2048 batch tokens)

`04-constrained-observed/`: PID/PGID 1510994, UTC 00:54:11–01:01:06,
readiness 00:57:42. Instrumented evidence verifies all eight bound 77×256
identity maps and the same original SHA-256/fingerprint, official Gloo state,
disabled async and no pending/rebalanced state. Each rank has 146 static
step calls and 148–150 model-execution calls. Actual upstream rearrangements,
weight moves and communicator transfers are **zero on every rank**.
CPU bound-row values and GPU shape metadata are observed; the observer does
not copy GPU map values or synchronize devices. The real map-commit owner
is covered by the behavioral tests and coherent distributed inference.

All eight raw profiles have nonnegative snapshots/non-torch/peak terms and
KV budgets 2,978,983,936–2,982,719,488 bytes, each no greater than requested
minus weights minus the measured positive peak/non-torch reservations.
Reported model usage is 125.53 GiB, peak 5.29–5.30 GiB and non-torch 3.18 GiB;
55,168–55,296 cache tokens are available. The official one-layer expert
buffer (1,209,270,272 bytes per rank) remains allocated; no ABI was removed.

Short and original three 3145-token prompts returned coherent Paris answers,
completion lengths 3/2/55/59, all HTTP 200/stop. Another 32 concurrent short
requests all returned 200/stop; every rank executed real scheduled work.
Health returned 200 after reuse. `runtime-audit.json` contains every text,
hash, token count, full rank profile and counters. No NIXL/UCX registration,
NaN, illegal access or OOM was observed. Owned-PTY Ctrl-C exited 0, followed
by `TEARDOWN_PASS` (no owned/logged PIDs/listener, all cards 0%/2 MiB).

The corresponding uninstrumented foreground command, after the pinned
environment/map setup and clean-device checks above, is:

```bash
env -u VLLM_PLUGINS python3 -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host 127.0.0.1 --port 8000 --trust-remote-code --enable-prefix-caching \
  --gpu-memory-utilization 0.95 --max-model-len 4096 --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --reasoning-parser hy_v4 --enable-auto-tool-choice --tool-call-parser hy_v4 \
  --seed 0 --served-model-name hy4-v0251-static-eplb \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
  --all2all-backend deepep_high_throughput --moe-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 --enforce-eager --enable-eplb \
  --eplb-config "$EPLB_CONFIG"
```

This is the first bounded resource candidate, not an exhaustive search for
the largest possible batch budget. The no-observer rerun is recorded below.

### No-observer constrained rerun — PASS, with shutdown cleanup concern

`05-constrained-final/`: PID/PGID 1541953, UTC 01:01:15–01:05:16,
readiness 01:04:03. No diagnostic extension or artifact import path. The
complete ARGV differs from the exact gate only in batch tokens 4096→2048;
all 36 request bodies match the instrumented run exactly. All eight ranks
report normal positive profiles and approximately 2.78 GiB KV. The actual
Gloo owner initializes; NIXL/UCX registration remains zero.

Short and original three 3145-token requests return 200/stop with completion
lengths 3/2/53/55; every answer coherently identifies Paris. Another 32 short
requests return 200/stop, and post-reuse health/metrics return 200. Full texts,
hashes, usage and observed/final differences are in `constrained-comparison.json`
and each run's `runtime-audit.json`. This is not bitwise parity or performance
evidence; instrumented all-rank state/counters and uninstrumented serving
results are deliberately separate.

Owned-PTY Ctrl-C at 01:05:11 led to parent exit 0, but the first teardown
failed: one DP6 worker remained and emitted shutdown TCPStore reset/broken-pipe
warnings. Its exact saved PID 1551995, creation time 1789261352.96, PGID 1541953
and command were checked before each signal. A targeted SIGTERM at 01:05:50
did not exit within ten seconds; targeted SIGKILL at 01:06:00 removed only
that owned worker. No unrelated PID/group was signalled. The original
`teardown.log` failure and `owned-dp6-cleanup.log` are retained; this manual
cleanup remains a shutdown concern, not an automatic graceful-exit claim.

After all regression processes finished, `teardown-final.log` was regenerated
for all five Task 11C services at 01:11:36 UTC: no owned/logged processes,
no port-8000 listener, every card 0%/2 MiB. No further model was launched.

Final source gates: 50 focused owner tests; 810 neighbors; all 36 changed
Python modules unfiltered, 1113 tests; complete pinned contract 2176 tests
with zero failures/errors/skips; CI selector 70 tests; doctor 7 PASS/49 callbacks.
Compileall, both diff checks and 17 Bash-block syntax checks pass. All 1970
pinned Python/shared-library hashes, candidate production/test hashes, model
config/index hashes and checkpoint size/mtime records remain unchanged.
The two exact DP8 capacity gates, separate plain-PP stall, cleanup concern,
W4A8/Mooncake gaps, unsupported modes and deferred observations remain open.
DCP and NIXL/UCX functionality are not tested/accepted by this work.

## Task 13 exact PP2/PCP4/EP4 native MTP3

Baseline `354bd1e230b3d6801f8ac15df7ad4a5f257a94a7`; artifacts:
`/models/validation-logs/hy4-pp-pcp-mtp-20260913T030102Z`.
The pinned installation/checkpoint and all original serving parameters stay
unchanged. This is one narrow supported topology, not general PCP speculation.
The new configuration contract requires the same checkpoint, one native MTP
layer, exactly three draft tokens, the exact target/draft architectures,
DeepEP HT/DeepGEMM/FP8 and no EPLB. Existing PP/TP/PCP/DP/partition, MRV2,
eager, LoRA, multimodal, offload, P/D and multi-layer sidecar guards remain.
PCP1/TP8 MTP3 and GLM PCP MTP1/2 retain their prior paths.

The baseline `01-baseline-rejection` reproduced the expected configuration
rejection before weights, exit 1, clean teardown. Initial RED was one exact
acceptance failure with 186 passing controls. Actual pinned sampling-method
CPU probes also passed: global PCP batch/attention/slot restoration,
last-stage proposal under replicated scope, exception restoration, and
real width-3 draft IDs at shuffled request slots on earlier stages.
Constructed HYV4 V32 indexers on all four ranks use effective PCP1 inside
the replicated draft scope, preserve shared top-k and do not gather again.

The first diagnostic observer failed on its own positional-argument
assumption during a keyword profile call; its log is retained as an observer
defect, not a model failure. Corrected `02b-observed` exposed a candidate
validation issue: native sparse draft loading canonicalizes public
`fp8_e4m3` to internal `fp8_ds_mla`, then PCP-manager initialization
revalidates the same config. A second focused RED reproduced that rejection;
recognizing the existing canonical FP8 layout gives 146 config tests passing.
No attention/model/runner/PCP/DeepEP/loader implementation was changed.
The launch still requests `fp8_e4m3`; this is not a precision/backend change.

Successful `02c-observed` retains all-eight-rank evidence:

- PP0 ranks 0–3 own layers [0,41), no speculator.
- PP1 ranks 4–7 own layers [41,78), native `MTPSpeculator` with layer 78
  only; stage-local EP groups are [0,1,2,3] and [4,5,6,7], 64 local experts.
- On each last-stage rank, target/draft/indexer/MLA top-k pointers are equal.
  The constructed indexer remains the current `V32SparseAttnIndexer`.
- Proposal metadata and effective PCP width are 1; scope exit restores 4.
- Each PP pair has matching sampled [N,4], counts [2,N], draft [N,3]
  collectives and non-placeholder draft IDs, including N=2 mixed requests.
  Local request allocator indices differ across stages; each side uses its
  own mapping. The initial analyzer's invalid cross-stage slot-equality
  assertion and corrected wire-payload audit are both retained.

Observer execution SHA-256:
`68308869ded884ddb2f00867f0e236455bc2f03a86d5a8584ddbe837cbeb9364`
(`task13_observer_executed.py`). It records bounded scalar/CPU values,
preserves calls/results, and samples draft IDs after the owner's existing
stream synchronization. The worktree observer was removed before acceptance.

### Exact uninstrumented acceptance

`03-exact-final`: foreground PID/PGID 1667106, start 03:18:27 UTC,
readiness 200 at 03:20:39 UTC. ARGV comparison proves the only addition to
the accepted Task 11B target-only command is native MTP3. No worker extension,
observer source or observer import is present. All eight cards were 0%/2 MiB
before launch.

The original short prompt plus three **byte-identical**, tokenizer-verified
3145-token requests and concurrent short/long controls all return HTTP
200/`stop`. Short output `Paris.`, every long output `Paris`; all six
match the retained target-only control text exactly. The identical request
SHA-256 is `963996620b60d947ea011ddf49746ea597c3c8f7c7fdff82813d04fadefaa9df`.
Raw requests/responses/headers, token counts, text hashes and control
comparisons are in `quality-summary.json` and `acceptance-summary.json`.
Final metrics: 18 drafted, 8 accepted, accepted positions 0/1/2 = 3/3/2;
prefix hits 0→3072→6144→9216. Health remains 200 after the requests.
These short completions establish functionality, not broad accuracy or
throughput; no new HumanEval or other benchmark score is claimed.

The pre-shutdown audit finds no runtime exception, NaN/OOM, cache/index error
or collective mismatch. Owned PTY Ctrl-C at 03:21:24 UTC closes the service.
The API output handler logs `EngineDeadError` during shutdown. The first
teardown poll catches workers still exiting; a subsequent check passes with
no extra process signals: no owned/logged PIDs or port 8000 listener and all
eight cards 0%/2 MiB. Both initial and final teardown records are preserved;
the shutdown warning is not represented as a warning-free exit.

NIXL/UCX and DCP are not tested or newly accepted. W4A8 extent validation,
native BF16/FP16 MTP source-format adaptation and parser review Minors remain
separate merge blockers/triage items. Prior DP8 capacity failures, plain-PP
stall, W4A8/Mooncake hardware gaps and other deferred observations remain.

Final Task 13 verification: all 36 branch-changed Python test modules
unfiltered, 1164 passed; HYV4/MTP/PCP/PP/GLM/DeepEP/DeepGEMM neighbors,
796 passed; explicit pinned contract, 2223 tests with zero failures/errors/
skips; CI selector 70 passed; doctor 7 PASS/49 callbacks. Compileall, target
diff check, 77-file AST and 18 Bash-block checks pass. All 1970 pinned source/
library hashes and checkpoint config/index hashes plus full file size/mtime
manifest remain unchanged. Post-test checks reconfirm all five Task 13
service lifecycles have no remaining owned/logged PID or listener and all
eight cards 0%/2 MiB.
