# Compressed-tensors channel W4A8 on HCU

## Problem and scope

An INT4/channel `pack-quantized` checkpoint with dynamic per-token symmetric
INT8 activations selects upstream `CompressedTensorsW4A8Int8MoEMethod`. In the
audited vLLM 0.28.1 build, that selector only has a CPU backend. Explicitly
selecting `--moe-backend aiter` does not connect it to the separate HCU
SlimQuant W4A8 implementation.

The new worker callback fills this connection for supported HCU/AITER
configurations. It does not edit the checkpoint configuration or relabel
W4A8 as W4A16. Other platforms, unsupported quantization schemes, and unrelated
errors retain the upstream behavior. If upstream implements this backend in
the future, its successful result takes priority over this fallback.

Supported scheme: symmetric static INT4 channel weights, dynamic symmetric
INT8 token activations, SiLU, no bias or input-side router weighting. Existing
WNA16 and HCU runtime restrictions still apply.

## Files

- `vllm_hcu/patch/worker/op_opt/patch_compressed_tensors_moe_w4a8_int8.py`:
  signature-checked, idempotent selection callback.
- `vllm_hcu/model_executor/layers/quantization/compressed_tensors_w4a8_runtime.py`:
  checkpoint representation conversion and HCU runtime delegation.
- `vllm_hcu/patch/worker/__init__.py`: lazy worker callback registration.
- `tests/runtime_patch/test_compressed_tensors_w4a8.py`: selection, isolation,
  packing, scale, buffer ownership, and compatibility contracts.

## Representation and loader ownership

The adapter reuses the existing WNA16 object's checkpoint loading behavior.
Its exact class identity is retained because upstream RoutedExperts uses that
identity to transpose packed checkpoint tensors and supply full intermediate
dimensions. The WNA16 constructor is used for storage only; the original INT8
activation descriptor is retained and execution is delegated to
`SlimQuantW4A8Int8AiterMoEMethod`.

CT packing stores eight unsigned `q + 8` nibbles in each INT32, least
significant nibble first. HCU W4A8 expects two signed two's-complement nibbles
per INT8, high nibble first. Post-load conversion preserves every signed
INT4 value. CT scales are divided by 16 for the HCU high-nibble scale domain;
the existing AITER runtime and Triton fallback apply their established
compensation. Final loader aliases reference the final runtime Parameters,
so the intermediate packed storage is not retained as a second weight copy.

## Validation and deployment status

CPU contract tests and the repository patch coverage gate were run on nmz10
in `gf_test_vllm_028_1022`. A separate process imported this source tree and
checked small DCU MoE outputs at 1, 9, and 64 tokens against the existing
Triton INT8 reference; outputs were equal for these cases.

The earlier isolated launcher was also validated with GLM-sized experts
(32 local experts, H=6144, I=2048, top-k=8), full TP8/EP startup, FP8 DS MLA KV,
MTP=3, CUDA graphs, and short arithmetic/code/concurrent generation requests.
Those full-model checks exercised the isolated launcher, not a rebuilt wheel
containing this source change. No full model benchmark accuracy claim is made.

Source changes alone do not modify an already-installed wheel or a running
server. Rebuild/install the plugin following the repository's standard build
procedure, then validate a new standard `vllm serve` process without the old
launcher. The running isolated-launcher service was left untouched during
source integration.

Focused checks:

```bash
python -m pytest -q tests/runtime_patch/test_compressed_tensors_w4a8.py
python tools/check_patch_test_coverage.py --json
```

## Source integration on nmz30 (2026-10-26)

Applied to `/public/home/gaofang/gf_proj/GLM-5.3-Channel-INT4-w8a8/vllm-plugin-das`
on base commit `2916920` (`v0.28.1-dev`). Validation used the source tree via
`PYTHONPATH` in container `gf_test_vllm_028_1022`:

- New W4A8 contract tests: 19 passed.
- Related worker-dispatch, patch-coverage, WNA16/W4A8 and INT8 post-load
  regressions: 96 passed, 221 deselected.
- Static patch coverage gate: passed, no missing contracts or untested adapters.
- `git diff --check`: passed.

This integration changed source files only. No plugin wheel was rebuilt or
installed, and no full-model service or accuracy evaluation was launched.
