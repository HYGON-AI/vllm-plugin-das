# vLLM 0.32 upstream-first Plugin validation

Validation date: 2026-10-29

## Result

The v0.32 Plugin keeps only LightOp ordinary and fused-add RMSNorm as an
explicit opt-in. The evaluated LightOp `SiluAndMul` override was slower than
official vLLM for every measured Qwen dense shape, so vLLM owns that operator
in both Plugin ON and OFF configurations.

This record is scoped to the exact source, wheel, toolchain, and `gfx936`
hardware below. It does not imply HumanEval, multi-model, long-running, or
production approval.

## Frozen identities

| Component | Identity |
| --- | --- |
| official vLLM upstream base | `vllm-project/vllm:v0.32.0`, `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| HCU vLLM build source | `OpenDAS/vllm:v0.32.0`, `7c13676e497262b71a8b107a3186b0f6641dcbf1` (DTK/gfx936 build patch on the upstream base) |
| installed HCU vLLM wheel | `vllm-0.32.0+rocm723-cp312-cp312-linux_x86_64.whl`, SHA256 `d7f084b8754288893dd9ba28d24784ec4b4e59350c9bfb5038ec24ac3af9572b` |
| Adapter source | `adapter-das`, `c99886d41f435e594050d47663e03ea94081fdc6` |
| Adapter wheel | `sgl_kernel-0.1.0-py3-none-any.whl`, SHA256 `69b655ca747718eed3c4487e984fe749b73880369f20ab0288fc76799db0c6db` |
| Plugin upstream base | `HYGON-AI/vllm-plugin-das:v0.32.0`, `e76220d9c0006cc3c9d886587a4771a31e7d83e8` |
| Plugin candidate | `feat/vllm032-plugin-onoff-gates`, `46923fbcf2068c1491090ca4d00e519b2abad91f` |
| Plugin wheel | `hcu_vllm_das_plugin-0.2.0.dev0-py3-none-any.whl`, SHA256 `b57733fbde63fb252e74492596aaed4c69b40b724eefb14e97eb51324ae11e14` |

The runtime was Python 3.12.12, PyTorch 2.11.0, HIP 7.2.26365, DTK 26.10,
and LightOp `0.6.0+dtk2610.torch2110.2610231814.gb2f8c9`. The device reported
`gfx936:sramecc+:xnack-`.

## Build and install

The Adapter wheel was built from a clean `git archive` of its frozen commit.
The Plugin wheel was built from a clean worktree after deleting the ignored
setuptools build directory:

```bash
env -u PYTHONPATH HCU_ADAPTER_HOST=none \
  /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python \
  -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir "$WHEEL_DIR" ./adapter_extension

env -u PYTHONPATH HCU_ADAPTER_HOST=none \
  /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python \
  -m pip install --no-deps --force-reinstall \
  /wheels/sgl_kernel-0.1.0-py3-none-any.whl \
  /wheels/hcu_vllm_das_plugin-0.2.0.dev0-py3-none-any.whl
```

Wheel inspection found exactly one Plugin entry point:

```text
[hcu_adapter.extensions]
das_hcu = hcu_vllm_das.extension:extension_spec
```

There is no Plugin-owned `vllm.platform_plugins` or `vllm.general_plugins`
entry point and no `hcu_vllm_das/silu_and_mul.py` in the wheel.

## Portable and installed gates

Portable suite:

```bash
env -u PYTHONPATH \
  HCU_ADAPTER_HOST=none \
  VLLM_TARGET_ROOT=/dev/shm/hcu-vllm032-work/src/vllm \
  PYTHONPATH=/dev/shm/adapter-das-main-20261029 \
  /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python \
  -m pytest -q tests/vllm032
```

Result: `61 passed, 3 skipped in 162.52s`.

Installed ownership and activation gate:

```bash
env -u PYTHONPATH \
  HCU_ADAPTER_HOST=vllm \
  VLLM_PLUGINS=hcu_adapter \
  /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python -S -c \
  'import runpy,sys; sys.path[:0]=["/dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/lib/python3.12/site-packages","/usr/local/lib/python3.12/site-packages"]; import sitecustomize; runpy.run_path("adapter_extension/tools/verify_installed.py", run_name="__main__")'
```

Result: pass. All three imported roots were owned by their installed wheel
RECORDs; the platform was `hcu_adapter.bridges.vllm.HcuPlatform`; the extension
was active with only `lightop-rms-norm`; the only required OOT owner was
`hcu_vllm_das.rms_norm.HcuRMSNorm`. The explicit venv-first launcher isolates
the same node-level system-package ordering issue described in the service
section below.

## Warmed compiled operator A/B

The takeover audit used the pre-removal Plugin commit `3f8f510e` and wheel
SHA256 `b10bc1d8910803e1a0595d6e52889ecbd77699dea698350ea87ea7573b04e729`,
because that is the last candidate containing both LightOp implementations.
Immediately before measurement, eager Silu/RMSNorm numerics and changed-input
compiled fused RMSNorm passed; the corrected combined Graph test then passed in
149.24 seconds with no mutable-output alias warning.

The checked-in
[`benchmark_operator_takeover_hcu.py`](../../tests/vllm032/benchmark_operator_takeover_hcu.py)
is the extracted equivalent of the inline probe. It was run in two fresh
processes with `HCU_BENCH_MODE=off`, master OFF, and then
`HCU_BENCH_MODE=on`, master plus both candidate children ON. Both used the
installed candidate wheel, Adapter `c99886d`, `HIP_VISIBLE_DEVICES=0`, and
`VLLM_PLUGINS=hcu_adapter`:

```bash
env -u PYTHONPATH HCU_ADAPTER_HOST=vllm VLLM_PLUGINS=hcu_adapter \
  HIP_VISIBLE_DEVICES=0 PYTHONPATH=/dev/shm/adapter-das-main-20261029 \
  HCU_BENCH_MODE=off VLLM_HCU_USE_CUSTOM_OPS=0 \
  VLLM_HCU_USE_CUSTOM_RMS_NORM=1 \
  VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=1 \
  /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python \
  tests/vllm032/benchmark_operator_takeover_hcu.py

# Repeat in a fresh process with:
# HCU_BENCH_MODE=on VLLM_HCU_USE_CUSTOM_OPS=1
```

Inputs were BF16. Silu shape was `[tokens, 24576]`; RMSNorm hidden size was
4096. Token/inner-iteration pairs were `1/200`, `16/100`, and `128/50`.
Each compiled callable received 10 warmups, followed by three
`time.perf_counter` measurements; one device synchronization occurred before
sampling and after each inner loop. Cold compilation was excluded.

Raw microseconds per call:

| Operator | tokens | official vLLM OFF | LightOp ON |
| --- | ---: | --- | --- |
| SiluAndMul | 1 | 99.0434, 99.6859, 99.3319 | 128.5461, 127.8036, 127.8883 |
| SiluAndMul | 16 | 103.8643, 104.6354, 104.0923 | 132.4756, 131.6108, 130.9058 |
| SiluAndMul | 128 | 103.9347, 104.1747, 103.4883 | 131.4280, 130.9786, 131.7620 |
| ordinary RMSNorm | 1 | 295.9525, 298.2647, 298.1261 | 151.7041, 150.5817, 150.2294 |
| ordinary RMSNorm | 16 | 303.5558, 312.7235, 314.2194 | 166.1581, 165.5552, 165.4851 |
| ordinary RMSNorm | 128 | 307.2379, 309.3152, 308.3241 | 165.5092, 165.7428, 164.5185 |
| fused-add RMSNorm | 1 | 353.7063, 351.4630, 350.9665 | 116.4316, 116.2898, 117.4073 |
| fused-add RMSNorm | 16 | 365.4227, 366.7266, 364.6846 | 127.0519, 126.7323, 126.9994 |
| fused-add RMSNorm | 128 | 362.1449, 371.8136, 371.7494 | 126.1129, 128.5817, 126.8483 |

| Operator | official vLLM | LightOp | Disposition |
| --- | ---: | ---: | --- |
| `SiluAndMul` | about 99--104 us | about 128--132 us | remove Plugin override; official wins every measured shape |
| ordinary RMSNorm | about 296--314 us | about 150--166 us | retain explicit opt-in |
| fused-add RMSNorm | about 351--372 us | about 116--129 us | retain explicit opt-in |

Dispatch was `forward_native` for both classes in OFF and `forward_hip` for
both candidate classes in ON. These microbenchmarks establish the ownership
decision only. They are not a serving-throughput claim. RMSNorm remains an
explicit, default-OFF opt-in after the final-wheel gates below passed.

## Live Graph and service gates

The final-wheel live Graph and small-model non-eager ON/OFF commands ran only
after the shared node became idle. Earlier, at `2026-10-29T07:06:38Z`, every
visible card was at 0% compute but held 58,088--58,216 MiB of 65,520 MiB VRAM
from a process outside this container's PID namespace. No process was
terminated and no test was started against that occupied baseline. At
`2026-10-29T07:32:14Z`, all eight cards had returned to 0% compute and 2 MiB
used, so the final gates proceeded on card 0.

Final-wheel eager, compile-replay, and ON/OFF Graph command:

```bash
env -u PYTHONPATH \
  HCU_ADAPTER_HOST=vllm VLLM_PLUGINS=hcu_adapter \
  HIP_VISIBLE_DEVICES=0 \
  HCU_VLLM032_RMSNORM_LIVE=1 HCU_VLLM032_GRAPH_LIVE=1 \
  /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python \
  -m pytest -q \
  tests/vllm032/test_rms_norm_hcu.py \
  tests/vllm032/test_operator_graph_hcu.py -vv
```

Result: `3 passed, 14 warnings in 98.17s`. This covers ordinary and fused-add
RMSNorm numerics, changed-input `torch.compile` replay without a mutable-output
alias warning, and fresh-process ON/OFF Graph dispatch. Explicit ON selected
`HcuRMSNorm.forward_hip`; master OFF selected `forward_native`; official
`SiluAndMul.forward_native` remained fixed in both rows.

The service used `/model/Qwen2-0.5B-Instruct` with BF16, max model length 512,
one sequence, `TRITON_ATTN`, and 20% device-memory utilization. It deliberately
did not pass `--enforce-eager`. Both fresh processes reported
`enforce_eager=False`, `VLLM_COMPILE`, `FULL_AND_PIECEWISE`, completed PIECEWISE
and FULL graph capture, and exposed vLLM `0.32.0`.

The validation venv uses `--system-site-packages`, and this node also contains a
global vLLM 0.28.1 ahead of the venv on normal startup. To prevent that global
package from contaminating the 0.32 service gate, both modes used the following
explicit venv-first launcher. ON used port 18015 and an isolated `on-cache`:

```bash
env -u PYTHONPATH \
  HCU_ADAPTER_HOST=vllm VLLM_PLUGINS=hcu_adapter \
  HIP_VISIBLE_DEVICES=0 HSA_VISIBLE_DEVICES=0 \
  VLLM_HCU_USE_CUSTOM_OPS=1 \
  VLLM_HCU_USE_CUSTOM_RMS_NORM=1 \
  XDG_CACHE_HOME=/dev/shm/vllm032-plugin-service-final/on-cache \
  TRITON_CACHE_DIR=/dev/shm/vllm032-plugin-service-final/on-cache/triton \
  /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python -S -c \
  'import sys; sys.path[:0]=["/dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/lib/python3.12/site-packages","/usr/local/lib/python3.12/site-packages"]; import sitecustomize; from vllm.entrypoints.cli.main import main; main()' \
  serve /model/Qwen2-0.5B-Instruct \
  --served-model-name Qwen2-0.5B-Instruct \
  --host 127.0.0.1 --port 18015 \
  --dtype bfloat16 --max-model-len 512 \
  --gpu-memory-utilization 0.20 --max-num-seqs 1 \
  --disable-custom-all-reduce --attention-backend TRITON_ATTN
```

OFF repeated that command in a fresh process with port 18016, an isolated
`off-cache`, `VLLM_HCU_USE_CUSTOM_OPS=0`, and the child deliberately left at
`VLLM_HCU_USE_CUSTOM_RMS_NORM=1` to prove that the master switch wins. Each
mode received proxy-bypassed `/health`, `/version`, `/v1/models`, and
temperature-zero chat probes. All four endpoints returned HTTP 200 in both
modes; both chats returned exactly `42` with identical 28 prompt, 3 completion,
and 31 total token counts. The sampled first chat times were 57.9 ms ON and
60.6 ms OFF. They are correctness samples, not a serving-performance claim.

Each owned service was stopped through its PTY with SIGINT. No vLLM or
EngineCore process remained, all cards returned to 0% compute and 0% reported
memory use, and no external process was signalled.

## HumanEval/0-63 accuracy

The final Adapter/Plugin wheel pair was also evaluated on the ordered
HumanEval/0-63 prefix with `/model/Qwen3-8B`. ON and OFF used fresh processes,
independent compile caches, BF16, max model length 4096, max sequences 1,
`TRITON_ATTN`, temperature 0, seed 0, max output 2048, thinking disabled, and
the same non-eager `VLLM_COMPILE` plus `FULL_AND_PIECEWISE` Graph policy used by
the service gate. The HumanEval repository was pinned to
`6d43fb980f9fee3c892a914eda09951f772ad10d`.

Generation command, repeated against ports 18017 (master/RMS ON) and 18018
(master OFF, RMS child deliberately ON):

```bash
env -u PYTHONPATH \
  TMPDIR=/dev/shm/vllm032-plugin-humaneval-current/on-tmp \
  python3 -S /dev/shm/hcu-vllm032-work/tools/humaneval64_client.py \
  --human-eval-root /dev/shm/hcu-vllm032-work/src/human-eval \
  --output-dir /dev/shm/vllm032-plugin-humaneval-current/on \
  --base-url http://127.0.0.1:18017 \
  --model Qwen3-8B --limit 64 --max-tokens 2048 --timeout 600
```

Generated code was not executed by the service client or directly on the host.
Scoring used official `human_eval.execution.check_correctness`, one sample per
task and a 10-second timeout, inside Bubblewrap 0.6.1 with unshared network,
PID, IPC, and UTS namespaces, UID/GID 65534, all capabilities dropped,
read-only host binds, tmpfs `/tmp`, and 2 GiB address-space, 128-process,
256-file, and zero-core limits. Both scoring stderr files were empty.

| Mode | Passed | pass@1 | Failed tasks | Timeouts | Generation wall time |
| --- | ---: | ---: | --- | ---: | ---: |
| master ON, RMS ON | 61/64 | 0.953125 | 26, 32, 54 | 0 | 167.67 s |
| master OFF, RMS child ON | 60/64 | 0.937500 | 18, 26, 32, 54 | 0 | 169.47 s |

Fifty-eight completions were byte-identical. The six differing tasks were 12,
18, 25, 32, 38, and 59; only task 18 changed pass/fail status. The ON answer
used a regular-expression lookahead to count overlapping substrings and passed,
while OFF used non-overlapping `finditer` and failed. Therefore this run proves
no HumanEval pass@1 regression from enabling the retained RMSNorm path, but it
does not claim token-exact numerical equivalence.

Node-local raw evidence hashes:

| Artifact | ON SHA256 | OFF SHA256 |
| --- | --- | --- |
| `problems.jsonl` | `3a64d077946f2c1b8707e13217e7069cc4bcefc034a6992f7ded9104c2dc59bd` | `3a64d077946f2c1b8707e13217e7069cc4bcefc034a6992f7ded9104c2dc59bd` |
| `responses.jsonl` | `ff47e25b6ebd7d07d92a30df8bf88b26abbfdd3957004b232eef6eda163e4fef` | `a6e46c0ad793441f6cb68e9f4d4067824f54a739f2341a27f8100aa75cf0a228` |
| `samples.jsonl` | `c4b09fd5a4479485b11080f2e70ebf450034e9360362145c04216794ac049521` | `6b213e3fd51d45d1b3ebc2f0bc4a5a438bb403ea29b694c687a3978ef2d8f775` |
| `score-results.jsonl` | `2d94ef218efc3dc949da20beb56d9c89b23b0bff8ff8cb9c6b8088f4a472a931` | `755add454228c2afac9b808553031ee4a3472746b98858fb1f1b271136526711` |
| `generation-summary.json` | `02a74d15dbc7a3cef274eff04db48b4a95a7cae904718563535ca1c642158678` | `bf828927a8b68a929ca17f1be3f3bc146fd2094bb36f34440a3221db0876f986` |

Both services remained healthy after all requests, stopped cleanly through
their owned PTY sessions, and all cards returned to 0% reported memory use.

## Qwen2 MoE HumanEval/0-63

The same installed wheel pair was then exercised with the BF16
`/model/Qwen2-57B-A14B-Instruct` checkpoint (`Qwen2MoeForCausalLM`, 64 experts,
top 8) on four gfx936 cards. Adapter source
`c99886d41f435e594050d47663e03ea94081fdc6`, which was subsequently merged by
[HYGON-AI/adapter-das#11](https://github.com/HYGON-AI/adapter-das/pull/11),
reported `runtime: no ops installed (pure upstream)`.
The active Plugin extension advertised only `lightop-rms-norm`; it did not
register or replace a MoE expert implementation.

Both modes used TP4, BF16, max model length 4096, max sequences 1, memory
utilization 0.75, `TRITON_ATTN`, disabled custom all-reduce, and the upstream
default non-eager `VLLM_COMPILE` plus `FULL_AND_PIECEWISE` Graph policy. Expert
parallelism was not enabled. No `--moe-backend` override was supplied: vLLM
resolved `moe_backend='auto'` to its unquantized Triton route with
`MoEPrepareAndFinalizeNoDPEPModular` and `TritonExperts`.

Representative server arguments, with ports 18020 (master OFF) and 18021
(master/RMS ON), were:

```bash
serve /model/Qwen2-57B-A14B-Instruct \
  --served-model-name Qwen2-57B-A14B-Instruct \
  --host 127.0.0.1 --port 18021 \
  --dtype bfloat16 --max-model-len 4096 \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.75 \
  --max-num-seqs 1 --disable-custom-all-reduce \
  --attention-backend TRITON_ATTN
```

ON set both `VLLM_HCU_USE_CUSTOM_OPS=1` and
`VLLM_HCU_USE_CUSTOM_RMS_NORM=1`. OFF set the master to `0` while deliberately
leaving the RMSNorm child at `1`. Health, version, model discovery, arithmetic
chat, changed-input Graph replay, PIECEWISE capture, and FULL capture passed in
both modes. Each rank loaded about 26.78 GiB of weights and used about 49.7 GiB
after KV-cache allocation.

Generation reused the pinned HumanEval commit and isolation protocol described
above, with temperature 0, seed 0, max output 2048, and thinking disabled. All
128 requests returned `finish_reason=stop`; there were no HTTP failures,
truncations, or evaluator timeouts. A fresh Bubblewrap rescore produced:

```bash
env -u PYTHONPATH \
  TMPDIR=/dev/shm/vllm032-qwen2moe-RyvxTJ/on2-tmp \
  /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python -S \
  /dev/shm/hcu-vllm032-work/tools/humaneval64_client.py \
  --human-eval-root /dev/shm/hcu-vllm032-work/src/human-eval \
  --output-dir /dev/shm/vllm032-qwen2moe-RyvxTJ/on-humaneval64 \
  --base-url http://127.0.0.1:18021 \
  --model Qwen2-57B-A14B-Instruct \
  --limit 64 --max-tokens 2048 --timeout 600
```

OFF repeated the client with port 18020, `off-tmp`, and
`off-humaneval64`. The scorer was run separately for each mode; the following
is the ON command before its JSON lines were summarized:

```bash
ulimit -v 2097152
ulimit -u 128
ulimit -n 256
ulimit -c 0
/dev/shm/hcu-vllm032-work/tools/bubblewrap/usr/bin/bwrap \
  --unshare-all --die-with-parent --new-session --cap-drop ALL \
  --uid 65534 --gid 65534 --clearenv \
  --setenv PATH /usr/bin --setenv PYTHONDONTWRITEBYTECODE 1 \
  --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
  --proc /proc --dev /dev --tmpfs /tmp \
  --ro-bind /dev/shm/hcu-vllm032-work/src/human-eval /human-eval \
  --ro-bind /dev/shm/vllm032-plugin-humaneval-current/score_humaneval.py \
    /score_humaneval.py \
  --ro-bind /dev/shm/vllm032-qwen2moe-RyvxTJ/on-humaneval64 /eval \
  --chdir /tmp \
  /usr/bin/python3 /score_humaneval.py \
    --human-eval-root /human-eval \
    --problems /eval/problems.jsonl \
    --samples /eval/samples.jsonl --timeout 10 \
  > /dev/shm/vllm032-qwen2moe-RyvxTJ/on-humaneval64/score-results.jsonl
```

| Mode | Passed | pass@1 | Failed tasks | Completion tokens | Generation wall time |
| --- | ---: | ---: | --- | ---: | ---: |
| master ON, RMS ON | 56/64 | 0.875000 | 9, 11, 19, 26, 32, 33, 41, 54 | 10,142 | 131.60 s |
| master OFF, RMS child ON | 55/64 | 0.859375 | 9, 11, 19, 26, 32, 33, 41, 54, 55 | 9,865 | 126.14 s |

Forty-four completions were byte-identical and twenty differed. HumanEval/55
was the only pass/fail difference. Repeating the OFF prewarm prefix within one
service had already produced different completion lengths for tasks 1, 4, and
7 despite temperature 0 and seed 0. The one-task ON advantage therefore is not
attributed to RMSNorm. This evidence establishes no observed ON accuracy
regression; it does not establish token-exact equivalence or an accuracy gain.

The observed completion-token rates were 77.07 tokens/s ON and 78.22 tokens/s
OFF. They are not a performance comparison because the modes had different
cache and prewarm histories, generated different token counts, and were each
measured only once. A performance claim requires matched warm state and at
least three independent repetitions.

Node-local raw evidence hashes:

| Artifact | ON SHA256 | OFF SHA256 |
| --- | --- | --- |
| `problems.jsonl` | `3a64d077946f2c1b8707e13217e7069cc4bcefc034a6992f7ded9104c2dc59bd` | `3a64d077946f2c1b8707e13217e7069cc4bcefc034a6992f7ded9104c2dc59bd` |
| `responses.jsonl` | `19ccb5755c35d2deeb2efdbdb960dd5d3f1bc7c445072e1eab71b30c6cd08e85` | `c7ec5c0fda2bfde5b86493a652bd30d5d9dbeeee571b6fd4d7e2899ae612f93a` |
| `samples.jsonl` | `25df7bb9152bb2461d8370fffac1771345978c47cad3dbe9651f3579b708da67` | `42900b4b45b265134a6dadbb7e31315f40cab1433c58ce00e5e97ccd4a6efcd4` |
| `score-results.jsonl` | `59481093251d639929223038237bdfe9a1bc978d310b83b3a99225d17c5ada9d` | `b7d8f6988c52e8bbeaf79eb1acdfabb9ee41a1dd5bec82fb7cbbeb9244bbf959` |
| `generation-summary.json` | `48486dc5ddc6d221b5711a5aef6e1c6124dd3000a241838122eb2aa56d433d14` | `ef81547e657f2dd5016d32ae8432b4a9ac218898e9b7f342ad156f24b43f8960` |

Upstream emitted two actionable MoE observations. It found no tuned Triton
configuration named `E=64,N=640,device_name=BW.json`, so it used its default
configuration, and the first real request still JIT-compiled
`fused_moe_kernel`. These are upstream configuration and warmup opportunities,
not evidence for adding a duplicate Plugin-owned MoE implementation.

All owned processes exited with status zero, no workers remained, and all eight
cards returned to 0% compute and 2 MiB reported memory. Python emitted a
`resource_tracker` warning about one leaked shared-memory object during each
service shutdown. The resources were reclaimed, but the warning remains an
explicit teardown follow-up rather than being classified as clean shutdown.

Broader multi-model precision remains pending for this candidate.
