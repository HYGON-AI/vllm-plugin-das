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

## PR #299 reload review follow-up (2026-10-30)

Replaced the method-lifetime conversion guard with validation of both packed
checkpoint tensors. Native checkpoint-format reload and weight-transfer updates
restore INT32 checkpoint Parameters while retaining the quantization method.
Each restored checkpoint is now converted again; already-converted INT8 runtime
weights and mixed checkpoint/runtime storage are rejected before conversion.
The execution-ready flag is cleared during processing and set only after the
delegate finishes, so failed processing cannot leave the method marked ready.

An omitted channel `group_size` is normalized to `-1` on a private copy before
WNA16 construction. Current native WNA16 `create_weights` already performs this
normalization before HCU qzeros creation, so the reported failure was not
reproduced on this vLLM revision. Tests cover both descriptors through the real
constructor and weight creation without modifying the caller's configuration.

Validation on nmz30 in `gf_test_vllm_028_1022`, using plugin base
`be55538626ba2e3f79ceccbc186397653f7d1e5b` and vLLM source
`77acaf633d4b1bfdc31390a012cfc470441df88f`:

- The new native reload regression fails against the old adapter at
  `online_process_loader -> _layerwise_process -> process_weights_after_loading`
  with `post-load conversion must run only once`.
- The W4A8 file and related WNA16/INT8 regressions: 104 passed, 220 deselected.
  This includes six cases loading three different checkpoints each, covering
  both group-size descriptors and AITER, native fallback, and packed Triton
  post-load boundaries. Assertions check weight values, scale compensation,
  final aliases, and preservation of original kernel Parameter objects.
- Patch coverage gate and `git diff --check`: passed.

These are CPU regressions using the real compressed-tensors package, WNA16
loader, SlimQuant delegate post-load, and native layerwise reload helpers.
Accelerator backend selection and kernel preparation are replaced in these
tests. Actual DCU kernel execution, IPC transport, a rebuilt wheel, and full
model accuracy/performance were not tested in this follow-up.

## Mixed-checkpoint MTP prefix compatibility (2026-10-30)

The checkpoint now in `/llm-models/hygon/GLM-5.3-Channel-INT4-w4a8` has separate
INT4 and INT8 groups. Its MTP INT4 regex targets `model.decoder.mlp.experts`,
while HCU constructs that block with the quantization prefix
`model.layers.78.mlp.experts`. Without translation this falls through to the
generic INT8 group, creates `w2_weight`, and then fails when loading the actual
INT4 `weight_packed` checkpoint into `w2_weight_packed`.

When explicitly enabled, `vllm_hcu/models/deepseek_mtp.py` adds aliases for canonical decoder targets
and ignores to a per-layer copy of the compressed-tensors configuration before
constructing the decoder. Anchored/unanchored canonical regex targets and literal
paths are supported. The engine's shared config, existing actual-prefix targets,
legacy MTP ignores, model files, and other quantization backends are preserved.
The exact MTP module hierarchy and expert weight loader are unchanged.

Validation:

- Combined MTP, W4A8, WNA16 and INT8 regression selection: 131 passed,
  219 deselected. This includes real compressed-tensors matching, constructor
  propagation, main-config isolation, legacy ignores, and multiple MTP indices.
- A real DCU probe on otherwise-idle device 2 constructed the MTP model through
  native `initialize_model`, read its 2336 checkpoint entries from 5 shards,
  called HCU `load_weights` and native `process_weights_after_loading`, and
  completed with `MTP_CHECKPOINT_AND_POSTLOAD_OK` (about 8.56 GiB allocated).
- A control using the same model, configuration, device and loader with only
  the new alias helper disabled in that process reproduced the original
  `KeyError: model.layers.78.mtp_block.mlp.experts.routed_experts.w2_weight_packed`.
- Probe artifacts are under
  `/public/home/gaofang/gf_proj/GLM-5.3-Channel-INT4-w8a8/run/20261030_mtp_fix_validation/`:
  `mtp_load_probe.py`, `mtp_load_probe.log`, and `mtp_load_original_prefix.log`.
  A repeat after adding the opt-in gate also passed; see `mtp_load_optin.log`.

The probe uses TP1/eager mode and BF16 KV configuration to validate actual MTP
loading and post-load conversion without loading the full target model. It does
not execute MTP forward passes, allocate KV caches, or measure accuracy. Full TP8
service startup and accuracy were not run because six cards were already using
about 132 GiB each for existing workloads. Probe processes exited and released
their device allocations. No existing model service was stopped.

The fix is enabled by default through the registered boolean environment
variable `VLLM_HCU_GLM_MTP_QUANT_PREFIX_FIX` as of 2026-11-05. Set it to `0`
to disable the compatibility mapping. It requires the target HF `model_type`
to be `glm_moe_dsa` and a compressed-tensors quantization config; other
architectures retain the original engine config. Regressions cover an unset
flag, explicit enable/disable, and non-GLM models with matching decoder rules.

The project `test.sh` explicitly exports
`VLLM_HCU_GLM_MTP_QUANT_PREFIX_FIX=1` and prepends this source tree to
`PYTHONPATH`, so its next launch enables the fix without rebuilding a wheel or
changing model config. Its pre-change backup is
`test.sh.bak.mtp-prefix-optin.20261030_170721` beside the script. Other launch
scripts using this source tree now get the GLM mapping without an explicit export.

## Original target identity under Eagle loading (2026-11-02)

The real TP8 launch in `run/20261102_111240` still failed with the original
`w2_weight_packed` KeyError despite having the opt-in flag enabled. Native
`SpeculativeConfig.hf_config_override` changes GLM DSA's draft `model_type` to
`deepseek_mtp`, and the Eagle wrapper supplies that draft as the current
`model_config`. The previous architecture gate checked that converted config,
so it incorrectly skipped the GLM alias fix. The earlier direct MTP probe had
kept the original target as its current config and did not cover this case.

The gate now identifies the original target through
`speculative_config.target_model_config`, falling back to the current config
only when no original target is supplied. The default-off flag and original
`glm_moe_dsa` target restriction are retained; the fix is not enabled for every
model using a `deepseek_mtp` draft.

- Regression selection: 135 passed, 219 deselected. Added coverage uses the real
  HF-config conversion, MTP construction with the converted config, and the HCU
  Eagle wrapper. Converted DeepSeek V3/V3.2 targets remain excluded.
- A real single-DCU MTP probe now also sets the current config to the converted
  draft, retaining the original target only in `SpeculativeConfig`. It printed
  `TARGET_MODEL_TYPE glm_moe_dsa`, `DRAFT_MODEL_TYPE deepseek_mtp`, and
  `MTP_CHECKPOINT_AND_POSTLOAD_OK` after reading 2336 entries from 5 shards.
  Artifacts are in `run/20261102_mtp_fix_validation/`.
- Full TP8 launches in `run/20261102_112638_mtpfix` and
  `run/20261102_113138_mtpfix` stopped earlier, during `init_device`: startup
  memory snapshots reported 75.58--99.5 GiB free, below the configured
  0.92 utilization target of 132.47 GiB. These attempts did not reach MTP loading
  or establish service readiness. The script's GPU-memory settings were not
  changed, and this initialization-memory issue remains unresolved.

This validates corrected MTP matching/loading and post-load conversion with the
converted config. Full-service readiness and end-to-end accuracy are still
unverified. The probe and failed validation workers have exited.


## PR #299 submission checks (2026-11-04)

Re-ran the current source tree on nmz30 in `gf_test_vllm_028_1022` with
accelerator visibility disabled and automatic vLLM plugin activation disabled:

```bash
python3 -m pytest -q tests/runtime_patch/test_compressed_tensors_w4a8.py tests/runtime_patch/test_glm51_mtp_contract.py
```

Result: 58 passed. This includes checkpoint reload lifecycle, channel group-size
normalization, opt-in MTP prefix matching, and converted draft target identity
regressions. These CPU tests do not establish full-model service readiness,
DCU kernel numerical accuracy, IPC transport correctness, or benchmark scores.

## Final scale ownership and default MTP aliases (2026-11-05)

Native layerwise reload restores persistent kernel Parameters after quantization
post-load. The W4A8 adapter now refreshes both AITER scale references before
execution, retaining the quantization config and its descriptors so consumers
observe the final layer-owned scales. CPU regressions cover repeated reloads,
in-place and replacement updates, and full-graph tracing.

GLM DSA compressed-tensors MTP decoder target aliases are enabled by default.
`VLLM_HCU_GLM_MTP_QUANT_PREFIX_FIX=0` remains an explicit rollback; the original
model-type and quantization-config checks still isolate unrelated models.

A user-run TP8/EP evaluation with AITER, MTP3, `reasoning_effort=low`, BF16 KV,
seed 42, batch size 8 and `max_tokens=30000` passed 100/100 on the first 100
HumanEval tasks (accuracy and pass@1 both 1.0). This is one first-100 evaluation,
not a result for all 164 tasks. The run used regular compilation and graph
capture without forcing `VLLM_USE_AOT_COMPILE=0`.

Run: `run/20261105_160550/20261105_161204`, report:
`reports/default_model/humaneval.json` in the GLM-5.3 validation project.
