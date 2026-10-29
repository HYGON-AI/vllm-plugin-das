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

Result: `61 passed, 3 skipped in 135.05s`.

Installed ownership and activation gate:

```bash
env -u PYTHONPATH \
  HCU_ADAPTER_HOST=vllm \
  VLLM_PLUGINS=hcu_adapter \
  /dev/shm/adapter-das-pr11-vllm032-qwen2-05b/venv/bin/python \
  adapter_extension/tools/verify_installed.py
```

Result: pass. All three imported roots were owned by their installed wheel
RECORDs; the platform was `hcu_adapter.bridges.vllm.HcuPlatform`; the extension
was active with only `lightop-rms-norm`; the only required OOT owner was
`hcu_vllm_das.rms_norm.HcuRMSNorm`.

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
decision only. They are not a serving-throughput claim, and the retained final
RMSNorm wheel remains default OFF until its pending final-wheel gates execute.

## Live Graph and service gates

The final-wheel live Graph and small-model non-eager ON/OFF commands must only
run on an idle shared node. At `2026-10-29T07:06:38Z`, every visible card was at
0% compute but held 58,088--58,216 MiB of 65,520 MiB VRAM from a process outside
this container's PID namespace. No process was terminated and no new HCU test
or service was started against that occupied baseline.

Pending rows:

- final-wheel ordinary and fused-add RMSNorm eager numerics;
- final-wheel changed-input `torch.compile` replay with no mutable-output alias
  warning;
- final-wheel ON/OFF Graph dispatch (`HcuRMSNorm.forward_hip` versus
  `forward_native`, while official `SiluAndMul.forward_native` remains fixed);
- `/model/Qwen2-0.5B-Instruct` ON/OFF service in default non-eager
  `FULL_AND_PIECEWISE` mode, without `--enforce-eager`;
- clean owned-process teardown and return to the node's pre-run VRAM baseline.

HumanEval-64 and broader model precision remain not run for this candidate.
