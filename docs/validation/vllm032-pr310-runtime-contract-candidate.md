# vLLM 0.32 PR #310 runtime-contract candidate

Validation date: 2026-10-30

## Scope and result

This record identifies the first PR #310 candidate that contains both declared
Plugin capabilities and the three reviewed runtime-contract fixes. Portable,
package-boundary, fresh installed-state, single-card live HCU numerics, and HIP
Graph gates passed. Two-rank expert-parallel communication, model accuracy,
serving, and throughput have not been rerun for this wheel, so the release
status remains `pending`.

## Frozen identities

| Component | Identity |
| --- | --- |
| official vLLM source | `v0.32.0`, `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| HCU vLLM wheel | `vllm-0.32.0+rocm723-cp312-cp312-linux_x86_64.whl`, SHA256 `d7f084b8754288893dd9ba28d24784ec4b4e59350c9bfb5038ec24ac3af9572b` |
| Adapter source | `c99886d41f435e594050d47663e03ea94081fdc6` |
| Adapter wheel | `sgl_kernel-0.1.0-py3-none-any.whl`, SHA256 `69b655ca747718eed3c4487e984fe749b73880369f20ab0288fc76799db0c6db` |
| Plugin source/build | `a7f5c2bd15c19296ab00219f188b3b7d65b45f50` |
| Plugin wheel | `hcu_vllm_das_plugin-0.2.0.dev0-py3-none-any.whl`, SHA256 `67db8901f289e5e9cc7ee8476be2e9c13980225d185cfe32665d9af8873893b9` |

The Plugin wheel was built from a clean `git archive` of the frozen Plugin
commit with:

```bash
python -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir "$WHEEL_DIR" "$SOURCE_ROOT/adapter_extension"
```

The wheel has one `hcu_adapter.extensions` entry point, contains RMSNorm and
AITER INT8 MoE modules, contains no Plugin-owned Silu override, and declares
both `lightop-rms-norm` and `aiter-int8-moe`.

## Portable and installed gates

The complete portable suite used Adapter source `c99886d`, official vLLM
source `5df63a7`, and no live-HCU switches:

```bash
PYTHONPATH="$ADAPTER_SOURCE" \
HCU_ADAPTER_HOST=none VLLM_TARGET_ROOT="$VLLM_SOURCE" \
python -m pytest -q tests/vllm032
```

Result: `100 passed, 6 skipped in 153.84s`. The skipped tests require explicit
live-HCU switches and are not counted as hardware passes.

A fresh `--system-site-packages` virtual environment then installed the three
locked wheels with `pip install --no-deps`. The installed verifier passed with
schema 3 and reported:

- platform `hcu_adapter.bridges.vllm.HcuPlatform`;
- active `das_hcu` entry point;
- capabilities `aiter-int8-moe` and `lightop-rms-norm`;
- AITER selector `hcu_vllm_das.moe.int8_oracle.HcuAiterInt8Experts`;
- OOT owner `hcu_vllm_das.rms_norm.HcuRMSNorm`.

An installed-module probe additionally verified official TRITON, HUMMING, and
CPU factory dispatch; TRITON/CPU weight-format pass-through; canonical EP map
to AITER mask conversion; official custom-op name `rms_norm`; and OOT registry
key `RMSNorm`.

## Live HCU operator and Graph gate

After all eight `gfx936` cards reported 0% compute and 0% memory, card 0 ran the
installed wheel with the RMSNorm, ON/OFF operator Graph, and AITER INT8 live
switches enabled:

```bash
HSA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
HCU_VLLM032_RMSNORM_LIVE=1 HCU_VLLM032_GRAPH_LIVE=1 \
HCU_VLLM032_AITER_INT8_LIVE=1 \
python -m pytest -q -s -rs \
  tests/vllm032/test_rms_norm_hcu.py \
  tests/vllm032/test_operator_graph_hcu.py \
  tests/vllm032/test_aiter_int8_hcu.py
```

Result: `6 passed, 14 warnings in 112.02s`. AITER M=1, 16, and 128 each
selected native ASM W8A8 configurations, stayed within the checked NMAE,
NRMSE, and max-error bounds, and replayed Graph with changed input. RMSNorm
eager/compile and fresh-process ON/OFF dispatch passed. After teardown all
eight cards returned to 0% compute and 0% memory.

## Unfinished gates

The current wheel has not run two-rank EP communication, model serving,
HumanEval, long-context, stability, or throughput gates. The earlier
`46923fbc` RMSNorm and `c40a2c67` AITER records remain historical evidence only
and do not approve this candidate.
