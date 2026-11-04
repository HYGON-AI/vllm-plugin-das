# HYV4VL image support (experimental)

The v0.25.1 baseline at `70a25cd5a2b5fe26427aad1cfc4f85124d53dd5e`
fails to load `Hy4-testing-Channel-FP8-w8a8`: its `hy_v4_vl` model type is
unregistered. This change adds configuration/model registration, the checkpoint's
unquantized vision tower with RMSNorm/Conv2d projection and image row separators,
and interleaved three-axis position encoding. It retains the HYV4 language
backbone and strict language/vision weight completeness checks.

Images are supported by the adapter; video, precomputed image embeddings and
speculative decoding are explicitly unsupported. End-to-end image generation
has **not** been validated; this implementation remains experimental.

## Validation

On 2026-11-04, using vLLM `0.25.1+das185`, PyTorch `2.11.0+dtk2604`,
Transformers `5.20.1`, and eight BW1101 HCUs:

- Baseline model loading reproduced the unknown `hy_v4_vl` error.
- 114 tests passed across `test_vl.py`, `test_registration.py`,
  `test_runtime_config.py`, and `test_weight_loading.py` in `tests/models/hy_v4/`.
- Actual checkpoint/tokenizer preprocessing of a 384x256 image produced grid
  `[1,16,24]`, pixels `[384,768]`, and 104 image tokens (8 rows x 13 positions).
- The vLLM multimodal processing pipeline passed text-only, single-image,
  two-image and repeated-image cases; image placeholder lengths were 104 each.
- Loaded all 489 vision parameters from the actual checkpoint on HCU0 and
  forwarded red, blue, counting, OCR and photographic fixtures. Outputs were
  finite with shapes `[104,6144]`, `[104,6144]`, `[136,6144]`, `[136,6144]`,
  and `[260,6144]` respectively. This checks the encoder only, not answer accuracy.
- The adapted TP8 server resolved `HYV4VLForConditionalGeneration`, then failed
  the device memory check: approximately 11 GiB free of 143.98 GiB per HCU,
  versus 129.59 GiB requested. Other workloads occupied all eight HCUs.
  No image answers or accuracy pass rate are claimed.

A previous default-backend attempt selected a DeepGEMM implementation that
asserted `not quant_config.per_out_ch_quant`. Explicitly select a channel-capable
backend for the next run; the command below requests Triton. This rerun still
needs idle hardware and must pass loading, generation and answer checks.

## Reproduction on idle hardware

From this checkout with the plugin dependencies installed:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /hy4_model/Hy4-testing-Channel-FP8-w8a8 \
  --served-model-name hy4-vl \
  --tensor-parallel-size 8 --moe-backend triton \
  --gpu-memory-utilization 0.90 --max-model-len 4096 \
  --max-num-seqs 2 --max-num-batched-tokens 1024 --enforce-eager \
  --mm-processor-kwargs '{"max_pixels":262144}' \
  --limit-mm-per-prompt '{"image":2}' \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --host 127.0.0.1 --port 8010
```

Once the server is ready:

```bash
python tools/hyv4_vl_image_smoke.py --output-dir /tmp/hy4-images
# Optionally add --photo /path/to/photo.jpg for manual description review.
```

The client creates red/blue, counting and OCR fixtures and checks text, single
image, multiple images and repeated-image requests at temperature zero. It saves
raw responses and heuristic answer checks to JSON, and exits nonzero for failed
HTTP requests or incorrect synthetic answers. Photo descriptions require human
review. These smoke checks do not replace a reference-model accuracy evaluation.
