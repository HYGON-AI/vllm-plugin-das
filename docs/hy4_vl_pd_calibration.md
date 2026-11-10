# HYV4-VL experimental calibration and balance contract

The frozen helpers are in [tools/hy_v4/vl_pd_validation](../tools/hy_v4/vl_pd_validation/README.md). See [tested launch commands](hy4_vl_pd_serving.md) for staging and actual paths.

## Fixed scope

- Checkpoint: `/data_hy4_w4a8/Hy4-testing-Channel-INT4-w4a8`.
- Exact class/key: `HYV4VLForConditionalGeneration`; 77 routed MoE rows, 256 logical experts, top-k 8, no draft/speculative configuration.
- P: PCP16/EP16/TP1, fixed HT. Calibration bootstraps from the newly generated VL identity R0. Candidate R16 is 272 physical slots, 17 per rank, exactly 16 extra slots per row, with locality dispatch `nearest` in the serving launch configuration.
- D: no EPLB. Generator may execute on D's CPU; it never invokes a server or accelerator.
- Old text/MTP capture JSON, old maps, and old prompt-token calibration helpers are rejected or outside this workflow. Root owns requests, frozen representative real image workloads, process lifecycle, quality gates, and timing.

## Recorder integration and lifecycle

Put both `vl_calibration_probe.py` and `build_vl_maps.py` on the P process's PYTHONPATH, and opt in at launch with:

```text
--worker-extension-cls vl_calibration_probe.WorkerExtension
--middleware vl_calibration_probe.CalibrationMiddleware
```

The loopback-only middleware exposes POST `/hy4/vl-eplb/info`, `/hy4/vl-eplb/start`, and `/hy4/vl-eplb/stop`. Use `{}` for info/start. For stop, use `{"output_dir":"/workspace/hy4_mm_20261108/capture-fit-UNIQUE"}`. Save the complete stop response as a manifest: `{"workers":[{"rank":0,"file":".../rank0.json","sha256":"...",...}, ...]}`. Use different new directories for fit, heldout, and actual R16 validation; `rankN.json` opens exclusively and cannot overwrite a prior capture. Outputs are restricted to absolute paths inside `/workspace/hy4_mm_20261108/`, including resolved symlink checks.

Worker RPC names are `hy4_vl_eplb_info`, `hy4_vl_eplb_start`, and `hy4_vl_eplb_stop`. Start rejects wrong class/layer count, any draft/speculator, non-null speculative config, duplicate routers/layers, simulated router names, existing capture callbacks, and absent static plans. It validates every router before installing hooks. Logical capture occurs before mapping; the mapper wrapper clones logical IDs before calling the saved mapper, including an in-place mapper. It checks every actual physical selection maps back to the corresponding logical expert. Only sentinel `-1` is excluded and counted; other invalid IDs fail clearly. Logical and physical totals and ignored counts must agree.

Stop restores the original capture function and mapper before importing native rank helpers, validating paths, moving histograms to CPU, serializing, or writing. A pre-existing instance mapper override is restored with its original object; an override introduced over a class method is deleted. Start-installation, callback, and mapper errors also restore all hooks. Middleware attempts collective stop if collective start fails, so workers which already started are cleaned up. After any distributed RPC failure, inspect info on all workers before continuing. Warmup belongs outside capture. No hook is permitted in timed performance runs; stop returns `capture_disabled:true`, and subsequent info must show false on every worker.

Capture records contain both 256-bin logical histograms and P-bin actual physical histograms, with calls, mapped calls, rows, top-k, map rows, and ignored-slot counts. The verifier scatter-adds physical counts through the map and requires exact logical reconstruction, including uneven replica dispatch. It does not assume a permutation or equal replica shares.

The scope label is **executed real top-k including PCP padding; -1 slots excluded**. PCP padding/duplicate execution is included. These counts are not real unpadded token counts, GPU utilization, grouped-GEMM padded FLOPs, communication bytes, or TTFT.

## CPU map generation

Run on the native CPU environment with torch and numpy available. The native policy source is read and SHA checked, then its `DefaultEplbPolicy` class body executes verbatim with CPU torch/numpy and a neutral abstract base. No native GPU/platform import is needed in `--policy-source` mode. Alternatively omit that flag to use the installed native policy on D's CPU. Accelerator visibility and plugins are disabled before policy loading.

```bash
python /workspace/hy4_mm_20261108/probe/build_vl_maps.py \
  --fit-manifest /workspace/hy4_mm_20261108/fit-stop.json \
  --heldout-manifest /workspace/hy4_mm_20261108/heldout-stop.json \
  --provenance /workspace/hy4_mm_20261108/provenance.json \
  --policy-source /workspace/hy4_ttft_resume_20261101_0912/pd-eplb-20261103-0452eab/core/vllm/distributed/eplb/policy/default.py \
  --redundancy 16 \
  --output /workspace/hy4_mm_20261108/maps-UNIQUE/vl-r16.json
```

`--policy-sha256` defaults to pinned `3b54229e6240d050480390f6b29d4744fd38437d23719f78a6decf7002296a29`. `--redundancy 0` generates a fresh optimized R0 comparison; it does not replace the identity bootstrap.

Each manifest must contain exactly the 16 distinct source ranks 0–15. Every capture file SHA is checked, and its internal rank must match. Every rank must provide exactly 77 distinct layer records, exact VL key, integer/nonnegative/int64 histograms, positive calls/rows, calls equal mapping calls, and total valid counts equal `rows*8 - ignored_invalid_slots`. Every row must have complete logical coverage and a valid physical-to-logical map. Bound maps must agree across all source ranks. Physical scatter reconstruction is repeated by the generator. Duplicate/missing layers or ranks, text/MTP records, incorrect types, inconsistent totals, bad IDs, and hash mismatches fail.

The native policy receives CPU int64 fit weights only and `num_replicas=256|272`, `num_groups=1`, `num_nodes=1`, `num_ranks=16`, and a corresponding initial map. Heldout is never passed to policy. The generated runtime JSON has only the VL model key and the existing version-2 generic loader fields. `static_dispatch_policy:'nearest'` is recorded in the sidecar; the serving command must still set it in `--eplb-config` alongside the generated map path and `num_redundant_experts:16`. Set `HY4_EXPERIMENTAL_STATIC_R16=1` on the P launch. D keeps EPLB off.

The sidecar is `vl-r16.report.json`. Both map and sidecar refuse overwrite. It saves map SHA, input-manifest/provenance hashes, all source capture files/hashes, generator SHA, native policy source SHA, complete supplied provenance, actual captured balance, and ideal equal-split predictions for identity and candidate on fit and heldout. Policy prediction assumes each logical expert load divides equally among its replicas; it does not claim actual locality/hash dispatch behavior. Fresh R16 heldout physical capture is required for actual balance.

## Provenance JSON contract

Required top-level fields:

```json
{
  "model_path": "/data_hy4_w4a8/Hy4-testing-Channel-INT4-w4a8",
  "model_class": "HYV4VLForConditionalGeneration",
  "speculative_config": null,
  "config_sha256": "738902ee7f32d7f4db373c7133e98a2b284dc395784144d04c677cf5bec419b7",
  "index_sha256": "b7019fa49f4c9eaf1a629fd6ac06daa463ce911efe9dd882232b9aa00568f4be",
  "corpus_sha256": "ACTUAL_64_HEX_SHA256_OF_FROZEN_CORPUS_MANIFEST",
  "core_sha256": "ACTUAL_64_HEX_SOURCE_FINGERPRINT",
  "plugin_sha256": "ACTUAL_64_HEX_SOURCE_FINGERPRINT",
  "fit_item_ids": ["realqa-fit-item"],
  "heldout_item_ids": ["realqa-heldout-item"],
  "item_image_hashes": {
    "realqa-fit-item": {"item_sha256":"ACTUAL_64_HEX","image_sha256":"ACTUAL_64_HEX"},
    "realqa-heldout-item": {"item_sha256":"ACTUAL_64_HEX","image_sha256":"ACTUAL_64_HEX"}
  },
  "preprocessing_limits": {"save_actual_limits_here":true},
  "input_output_lengths": {"save_actual_token_limits_and_observed_lengths_here":true},
  "concurrency": 1,
  "prefix_cache_state": "save actual state and warmup here",
  "prompt_templates": "save actual template bytes or immutable content reference here",
  "source_files": [{"file":"/absolute/native/source/or/manifest/file","sha256":"ACTUAL_64_HEX"}]
}
```

The example placeholders are not accepted hashes. Save every selected item and every image hash (multiple image hash keys are permitted), the frozen corpus manifest, concrete launch settings, and source fingerprints. Fit/heldout IDs must be nonempty, unique, disjoint, and covered by item/image hashes. Source files are rehashed when generating. Config/index SHA values are pinned to the P/D checkpoint identity. Corpus/core/plugin aggregate hashes and item/image hashes are provenance assertions supplied by the root; include their actual manifests/files in `source_files` to independently rehash those bytes. No image decoding or requests are performed by this generator.

## Actual balance printing

After a fresh candidate heldout capture, run:

```bash
python /workspace/hy4_mm_20261108/probe/build_vl_maps.py \
  --analyze-manifest /workspace/hy4_mm_20261108/r16-heldout-stop.json
```

This path does not import torch, numpy, policy, or vLLM. It verifies all records/hashes and prints actual destination loads and metrics as JSON. It can analyze either P=256 or272 and uses 16 or17 consecutive physical slots per destination rank respectively.

Exposed pure functions: `validate_map`, `scatter_logical`, `destination_loads`, `balance_metrics`, `validate_capture`, `load_capture_manifest`, `ideal_equal_split`, and `validate_provenance`. `generate` and `load_policy` are CPU policy helpers.

Metric definitions: weighted layer max/mean is sum of layer maxima divided by sum of layer means; worst-layer ratio identifies its layer; mean per-layer CV uses population standard deviation. Cumulative-rank CV/max/mean is computed separately after summing across layers. All-zero layers are listed explicitly and excluded from per-layer ratios; an entirely zero capture reports null ratios, not perfect balance.

The unchanged helper CPU regression suite is included beside the tools. The actual observed heldout results are in the [deployment report](hy4_vl_pd_serving.md#results-and-limits-2026-11-08).
