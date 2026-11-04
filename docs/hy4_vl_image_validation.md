# HYV4VL image support (experimental)

For current standalone server/client commands, see the
[serving guide](hy4_vl_serving.md). Later graph-mode evaluations have their own
scoped results in the [decode](hy4_vl_graph_performance.md) and
[full/piecewise](hy4_vl_piecewise_performance.md) reports; this page records the
initial eager validation.

The v0.25.1 baseline at `70a25cd5a2b5fe26427aad1cfc4f85124d53dd5e`
fails to load `Hy4-testing-Channel-FP8-w8a8`: its `hy_v4_vl` model type is
unregistered. This change adds configuration/model registration, the checkpoint's
unquantized vision tower with RMSNorm/Conv2d projection and image row separators,
and interleaved three-axis position encoding. It retains the HYV4 language
backbone and strict language/vision weight completeness checks.

Images are supported by the adapter; video, precomputed image embeddings and
speculative decoding are explicitly unsupported. End-to-end eager TP8 image generation passes the synthetic smoke checks below.
The adapter remains experimental: reference-model accuracy parity, video and
non-eager execution have not been validated.

## Validation

On 2026-11-04, using vLLM `0.25.1+das185`, PyTorch `2.11.0+dtk2604`,
Transformers `5.20.1`, and eight BW1101 HCUs:

- Baseline model loading reproduced the unknown `hy_v4_vl` error.
- 115 tests passed across `test_vl.py`, `test_registration.py`,
  `test_runtime_config.py`, and `test_weight_loading.py` in `tests/models/hy_v4/`.
- Actual checkpoint/tokenizer preprocessing of a 384x256 image produced grid
  `[1,16,24]`, pixels `[384,768]`, and 104 image tokens (8 rows x 13 positions).
- The vLLM multimodal processing pipeline passed text-only, single-image,
  two-image and repeated-image cases; image placeholder lengths were 104 each.
- Loaded all 489 vision parameters from the actual checkpoint on HCU0 and
  forwarded red, blue, counting, OCR and photographic fixtures. Outputs were
  finite with shapes `[104,6144]`, `[104,6144]`, `[136,6144]`, `[136,6144]`,
  and `[260,6144]` respectively. This checks the encoder only, not answer accuracy.
- Built the native extension with `MAX_JOBS=8 python setup.py build_ext --inplace`.
- With idle HCUs, loaded all 246 checkpoint shards (738.74 GiB) in 422.57 seconds.
  TP8 serving started successfully with the command below. Per-rank model
  weights consumed approximately 95.78 GiB; the run used the channel FP8 linear
  kernel and Triton FP8 MoE backend.
- Fixed a serving-path processor defect: vLLM tokenizes cached multimodal
  prompts without supplying images, so those placeholders must remain intact.
  The new regression failed before the fix and passes after it.
- All eight final HTTP requests returned 200 and finished with `stop`.
  Seven constrained checks passed, including the text baseline:

| Case | Expected | Actual | Result |
| --- | --- | --- | --- |
| Text | 2 | 2 | pass |
| Red rectangle | red | 红色 | pass |
| Blue rectangle | blue | 蓝色 | pass |
| Three red circles | 3 | 3 | pass |
| OCR | HY4 7285 | HY4 7285 | pass |
| Two images | red, then blue | 第一张图片中的矩形是红色，第二张图片中的矩形是蓝色。 | pass |
| Repeated red image | red | 红色 | pass |

The initial OCR prompt allowed explanations: the answer read the text correctly
but failed the client's exact-match rule. The final prompt explicitly requests
only the original text; the exact-match rule was retained. No answer-checking
threshold was relaxed.

The optional photo produced a coherent description of pink cherry blossoms and a
blue sky, but also named the landmark “广州塔”. That identity is not established
by this evaluation and must not be counted as an accuracy pass. The photo case
is manual review only (`correct: null`). Synthetic checks are not a general
multimodal benchmark or a comparison with the reference checkpoint implementation.

Earlier attempts with occupied hardware stopped at the memory check. A default
MoE backend also selected DeepGEMM, which rejected per-channel quantization;
explicit `--moe-backend triton` avoids that incompatibility for this checkpoint.

## Reproduction on idle hardware

From this checkout with the plugin dependencies installed:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /module/Hy4-testing-Channel-FP8-w8a8 \
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
