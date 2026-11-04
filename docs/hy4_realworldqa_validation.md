# HYV4VL RealWorldQA evaluation

The subsequent ViT/M-RoPE performance changes and full accuracy regression are
documented in [the performance report](hy4_vl_performance.md). This page records
the original working-image baseline.

A separate full-split concurrency-1 graph/eager comparison is recorded in the
[decode graph report](hy4_vl_graph_performance.md).

This evaluation uses the full 765-question `test` split of the original
[xAI RealWorldQA dataset](https://huggingface.co/datasets/xai-org/RealworldQA),
revision `17e7f75e092e47169732462ea3cdfebe911105dd`. The split contains
438 multiple-choice questions and 327 open-ended questions. xAI describes the
benchmark and its CC BY-ND 4.0 license in the
[release announcement](https://x.ai/news/grok-1.5v).

## Results (2026-11-04)

All 765 questions completed in 793.98 seconds using the protocol below.

| Metric | Result |
| --- | --- |
| Overall reference-scored accuracy | **560/765 (73.20%)** |
| Multiple-choice accuracy | 344/438 (78.54%) |
| Open-ended accuracy | 216/327 (66.06%) |
| Incorrect answers | 205/765 |
| Raw case-sensitive exact matches | 545/765 (71.24%) |
| Nominal Wilson 95% interval | 69.95%–76.22% |
| HTTP errors / empty answers / truncations | 0 / 0 / 0 |
| Median request latency (concurrency 2) | 2.03 seconds |

The 15 additional reference-scored matches arise from the documented answer
normalization/extraction rules; raw predictions are retained. No failed questions
were removed or retried with different prompts. The runtime remained stable
across the real images and their varying aspect ratios, but answer accuracy is
not perfect. For example, dataset item 36 asks for right-pointing arrows: the
standard answer is 3 and the model answered 2. Item 44 asks for the lane count:
the standard answer is 3 and the model answered 2. Direction errors also occur
(e.g. item 8: downhill versus the model's uphill selection).

These are benchmark mismatches, not evidence by themselves of a plugin defect.
Reference-checkpoint parity has not been measured. This report does not claim
that a synthetic smoke pass guarantees real-world correctness, nor does it
compare this configuration directly with published model rankings.

## Protocol

- Checkpoint: `/module/Hy4-testing-Channel-FP8-w8a8`.
- Plugin serving revision: `03841f4` (includes the cached-placeholder fix).
- Hardware: eight BW1101 HCUs; TP8, eager execution, Triton FP8 MoE.
- Maximum image pixels: 1,048,576. The complete original image is supplied;
  the model's image processor performs resizing. No crops or image augmentation.
- Original English question, including its existing answer-format instruction;
  no additional prompt, demonstrations, system message or answer hints.
- Temperature 0, top-p 1, seed 0, `reasoning_effort=no_think`, maximum 16 output
  tokens; concurrency 2. Generation settings follow the pinned lmms-eval task.
- Scoring follows `realworldqa_process_results` from
  [lmms-eval revision 7847ad8](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/7847ad8539889c87533b78c93e280ddfc117d16b/lmms_eval/tasks/realworldqa).
  Multiple-choice answers use its unmodified `extract_mcq_answer`. Open-ended
  answers are lowercased, stripped of outer whitespace and trailing periods,
  then compared exactly. No synonyms, number-word conversion or LLM judge.
- The evaluator's scorer was checked against the pinned reference function on
  160 boundary cases; all agreed. Prompt and scoring were fixed before the run.

HTTP failures, empty answers and truncations are reported separately. All
questions remain in the accuracy denominator. The evaluator saves every raw
response and data/scorer/evaluator hashes, and can resume with matching settings.

## Reproduction

Start the server from the plugin checkout after building the native extension:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /module/Hy4-testing-Channel-FP8-w8a8 \
  --served-model-name hy4-vl \
  --tensor-parallel-size 8 --moe-backend triton \
  --gpu-memory-utilization 0.90 --max-model-len 4096 \
  --max-num-seqs 2 --max-num-batched-tokens 1024 --enforce-eager \
  --mm-processor-kwargs '{"max_pixels":1048576}' \
  --limit-mm-per-prompt '{"image":2}' \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --host 127.0.0.1 --port 8010
```

Download the pinned public dataset and reference parser (the evaluator requires
PyArrow, Pillow and Requests; downloading uses `huggingface_hub`):

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "xai-org/RealworldQA", repo_type="dataset",
    revision="17e7f75e092e47169732462ea3cdfebe911105dd",
    local_dir="/tmp/realworldqa",
)
PY
curl -fL \
  https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/7847ad8539889c87533b78c93e280ddfc117d16b/lmms_eval/tasks/_task_utils/mcq_extract.py \
  -o /tmp/realworldqa_mcq_extract.py
python tools/hyv4_realworldqa_eval.py \
  --data-dir /tmp/realworldqa/data \
  --mcq-extractor /tmp/realworldqa_mcq_extract.py \
  --output-dir /tmp/hy4-realworldqa-results --max-pixels 1048576
```

The CLI's `--max-pixels` records the server setting; it must match the serving
command. Results are specific to this checkpoint, image resolution, prompt and
runtime. Without a reference implementation or unquantized checkpoint comparison,
this evaluation cannot isolate errors caused by the base model, FP8 quantization
or the plugin adaptation.
