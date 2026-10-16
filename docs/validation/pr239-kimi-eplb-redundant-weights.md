# PR #239: Kimi-K3 redundant expert checkpoint loading

## Scope and root cause

Base: `7d52be2dde99519d103afed9de94eeed74eaba1d`.

With EPLB enabled and redundant experts allocated, the Kimi main model's
inline checkpoint loader only constructs mappings for logical expert IDs.
It also breaks after the first mapping match. Consequently, the physical
replicas can enter routing with uninitialized weights and quantization scales.
Passing the redundant count alone would still leave the first-match bug.

## Fix

The plugin's existing Kimi EPLB adapter wraps `KimiLinearModel.load_weights`.
For layers with redundant storage, it delegates expert tensors to the owning
`RoutedExperts.load_weights` implementation. That loader visits every physical
mapping and uses each parameter's loader for rank-local placement. Bound static
EPLB loaders retain their logical-to-physical fanout.

The adapter translates Kimi checkpoint `w1/w2/w3` names to the owner's configured
projection names and preserves the existing `weight_packed` to `weight` alias
when storage is unpacked. Other tensors and PP missing-layer filtering continue
through the Kimi loader. Layers without redundant storage use the original
loading path. The patch remains in the plugin; no installed package is changed.

## CPU regression coverage

`tests/runtime_patch/test_kimi_k3_eplb_model.py` includes:

- A pre-fix probe using the installed Kimi loader: redundant `w13_weight`,
  `w2_weight`, and both scales retain their sentinel values.
- EP1, EP2, and EP16 placement, iterating every simulated rank, with default
  initial placement and an alternate static placement.
- FP32 weights and synthetic INT8 checkpoint bytes with `weight_packed` names,
  plus per-channel weight scales for all w1/w2/w3 projections.
- Exact comparison of every local physical slot against its logical checkpoint
  weights/scales; a CPU SiLU MLP calculation through every FP32 physical slot.
- No-redundancy behavior, non-expert weights, PP missing-layer filtering, adapter
  registration and idempotence.

Mapping construction, expert loading, parameter weight loaders and Kimi inline
loading use the installed vLLM implementations. Fixtures bypass accelerator
allocation and supply CPU placement/configuration. These are small synthetic
weights: they do not validate HIPC packing, device MoE kernels, distributed
communication, or full-model HumanEval.

## Reproduce

Run in `xiesl_kimi3_plugin_028`, from
`/home/xiesl/inference/kimi3_plugin_vllm/plugin_dir/plugin-v0.28.1-dev-kimi-k3-xiesl`:

```bash
export VLLM_PLUGINS=__disabled__
export CUDA_VISIBLE_DEVICES=
export HIP_VISIBLE_DEVICES=
python -m pytest -q --disable-warnings \
  tests/runtime_patch/test_kimi_k3_eplb_model.py \
  tests/runtime_patch/test_kimi_k3_reference_adapters.py \
  tests/runtime_patch/test_model_loader_static_eplb.py \
  tests/model_executor/layers/fused_moe/test_static_eplb.py \
  tests/runtime_patch/test_moe_deepep.py
python tools/check_patch_test_coverage.py
python tools/check_production_boundary.py
```

No service is started and no accelerator is exposed to these commands.

## Results (2026-10-16)

- Final combined regression: **232 passed**, 14 warnings, 64.07 seconds.
- Focused redundant-expert regression: **17 passed**.
- Patch coverage: 171 files, 170 adapters, 0 untested, 0 invalid contracts.
- Production boundary: 401 Python files clean.
- `git diff --check`: clean.

Host artifacts: `task_dir/debug_028/eplb_redundant_fix_tests.log` and
`task_dir/debug_028/eplb_redundant_fix_regression.log`.
No new full-model HumanEval result is claimed for this fix.

## Runtime provenance and PP fixture correction

The 17-test and 232-test runs above used `xiesl_kimi3_plugin_028`, with installed
vLLM version `0.28.1+das.77acaf6.dtk2604` at
`/usr/local/lib/python3.10/dist-packages/vllm`. This is the `77acaf6` source
lineage, not the repository-declared OpenDAS baseline `462fdb097`. Installed
source may contain local changes; the version alone is not an exact tree ID.
The installed source SHA-256 values checked for the fixture correction are:

- `models/kimi_k3/amd/linear.py`:
  `dc32aebceb1c2a19560a404f6468c8316892831ce3d3062981d3612ecddf1b65`.
- `model_executor/models/utils.py`:
  `a2e26518da6b49ec861a580cd1195fe38fce57b4484ee457627093d290e3cf68`.

The PP skip fixture now preserves all four layer indices, uses `PPMissingLayer`
for nonlocal layers 1–3, and keeps `num_hidden_layers=4`. Layer 3 therefore
exercises the native PP missing-layer contract, without relying on an
out-of-range tensor filter present in the installed Kimi loader.

Validation of this fixture correction:

- Installed runtime, same 17-test file: **17 passed**, 10.86 seconds.
- An isolated CPU probe executes `KimiLinearModel.load_weights`,
  `get_spec_layer_idx_from_weight_name`, `get_pp_missing_layer_names`, and
  `is_pp_missing_parameter` extracted from OpenDAS commit
  `462fdb097c66b487ef4826e8009431c10fe88fb8`. The original fixture reproduces
  `KeyError('layers.3.mlp.experts.routed_experts.w13_weight')`; the corrected
  fixture's 17-test file passes. Other runtime classes and dependencies remain
  those of the installed environment, so this is a baseline-method probe,
  not a full pytest run in an installed `462fdb097` environment.

Host probe and logs are under `task_dir/debug_028/`:
`pp_missing_layer_baseline_probe.py`, `pp_missing_layer_baseline_probe.log`,
and `pp_missing_layer_fixture_tests.log`. No dependency or service changes
are involved in this fixture correction.
