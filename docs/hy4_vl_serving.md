# HYV4VL server and client commands

Run these commands from the PR checkout in the tested DAS vLLM 0.25.1 / DTK2604
image, after installing plugin dependencies and building its native extension
(`MAX_JOBS=8 python setup.py build_ext --inplace`). The tested checkpoint is
`/module/Hy4-testing-Channel-FP8-w8a8` on eight idle BW1101 HCUs. These are image
serving commands; video, encoder-only mode and speculative decoding are
unsupported.

Vision tower/connector LoRA (`enable_tower_connector_lora=True`) is rejected
before model allocation because the multimodal LoRA token-count interfaces
are not implemented. This restriction does not disable language-only LoRA.

## Server

The [sparse-index order fix](hy4_vl_validation.md#precision-fixes) resolves the
reproduced long-context variation, including the eager-mode case. Restart the
server from the updated checkout and capture new graphs. The fix is automatic
for HYV4-VL on HCU; no additional flag is required. Full/piecewise retains the
padding guard described below. The complete 765-question rerun matches the
prior control's raw answers, and the extended image repeat checks pass.

Choose one execution mode. `eager` is the default fallback. The
[validation summary](hy4_vl_validation.md) records the tested graph modes,
precision guards, accuracy, performance and configuration limits. At DP=1,
PIECEWISE replays only when its token descriptor exactly matches the input
count; other prefill chunks run at their original eager size. Stop the previous
server before starting another mode.

```bash
# Run in the plugin checkout. Choose: eager, decode, or full-piecewise.
HY4_EXECUTION=eager
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
case "$HY4_EXECUTION" in
  eager)
    HY4_EXECUTION_ARGS=(--enforce-eager)
    ;;
  decode)
    HY4_EXECUTION_ARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}')
    ;;
  full-piecewise)
    HY4_EXECUTION_ARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,128,256,512,768,1024]}')
    ;;
  *)
    echo "Unknown HY4_EXECUTION: $HY4_EXECUTION" >&2
    exit 2
    ;;
esac
vllm serve /module/Hy4-testing-Channel-FP8-w8a8 \
  --served-model-name hy4-vl --tensor-parallel-size 8 --moe-backend triton \
  --gpu-memory-utilization 0.90 --max-model-len 4096 \
  --max-num-seqs 2 --max-num-batched-tokens 1024 \
  "${HY4_EXECUTION_ARGS[@]}" \
  --mm-processor-kwargs '{"max_pixels":1048576}' \
  --limit-mm-per-prompt '{"image":2}' --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching \
  --default-chat-template-kwargs '{"reasoning_effort":"no_think"}' \
  --host 127.0.0.1 --port 8010
```

In this DAS build, `VLLM_USE_BREAKABLE_CUDAGRAPH=1` disables the vLLM
`torch.compile` pipeline. The graph modes above use that implementation; they
are not an Inductor-compiled experiment. Decode-only leaves prefill eager.
Full/piecewise additionally dispatches eligible prefill/mixed batches to
PIECEWISE. Capturing only `[1,2]` would cover very little image prefill, so the
latter configuration includes larger token buckets. With the precision guard,
1024-token chunks can replay while short tails that would be padded fall back to
NONE. This guard covers DP=1; DP/SP/CP precision is not established by this test.
ViT remains eager in both.

The benchmark disables prefix and CPU processor caches. The default GPU encoder
embedding LRU remains enabled; repeated images can reuse embeddings. Start a
fresh process for each independent mode comparison and run the same client
sequence. A complete encoder-cache reset between diagnostic runs was performed
in the reports; it is not part of each normal inference request.

The examples omit `VLLM_SERVER_DEV_MODE` and diagnostic worker extensions.
Internal mode-switching RPCs used for same-weight controls are documented with
the result artifacts, not required for normal serving. Run clients from another
terminal after readiness:

```bash
curl --fail --silent --show-error http://127.0.0.1:8010/health
```

## Prepare the real-image dataset

Clients require PyArrow, Pillow and Requests. Downloading additionally requires
`huggingface_hub`. The original test split and scorer are pinned:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "xai-org/RealworldQA", repo_type="dataset",
    revision="17e7f75e092e47169732462ea3cdfebe911105dd",
    allow_patterns=["data/test-*.parquet"],
    local_dir="/tmp/realworldqa",
)
PY
curl --fail --location \
  https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/7847ad8539889c87533b78c93e280ddfc117d16b/lmms_eval/tasks/_task_utils/mcq_extract.py \
  --output /tmp/realworldqa/mcq_extract.py
```

The dataset is CC BY-ND 4.0. These commands download it locally; the PR does not
redistribute the images. The dataset/scorer protocol is described in the
[validation summary](hy4_vl_validation.md#dataset-and-scoring).

## Client: image smoke checks

```bash
python tools/hyv4_vl_image_smoke.py \
  --base-url http://127.0.0.1:8010 --model hy4-vl \
  --output-dir /tmp/hy4-smoke
# Optionally add --photo /path/to/photo.jpg for manual description review.
```

This generates red/blue, counting and OCR fixtures and checks text, one image,
two-image order and repeated input. The seven constrained checks must pass.
The optional photographic description is unscored manual review.

## Client: streaming performance

Use a new output directory for each mode/run. For an independent comparison,
restart the server immediately before each measured client; do not run the
smoke, performance and accuracy commands as one implicit warmup sequence. This
starts with an empty encoder cache without enabling diagnostic reset RPCs.
`--samples` must be at least 8 for the four two-image pairs; smaller values
are rejected before data loading or warmup requests.

```bash
python tools/hyv4_vl_perf.py \
  --base-url http://127.0.0.1:8010 --model hy4-vl \
  --data-dir /tmp/realworldqa/data \
  --samples 32 --rounds 2 \
  --output-dir /tmp/hy4-perf-eager
```

For `decode` or `full-piecewise`, restart the server in that mode and change the
output directory accordingly. The client uses seed 338, two warmups, 64 QA
requests at concurrency 1 and 2, and four single/two-image 32-token captions.
It writes per-request answers/usage/TTFT/latency, a manifest and summaries.
`content_chunk_interval_seconds` measures nonempty text-delta intervals, not
time per token; it is null for a single content delta. The summary reports
`content_chunk_interval_median_seconds`. These replace the earlier misleading
`decode_seconds_per_token` / `decode_median_seconds_per_token` fields.
Images are encoded before timing. Keep model limits, resolution, cache flags,
prompts and generation budgets identical across runs. This is a fixed
repeated-image workload, not a maximum-capacity or forced-cold encoder test.

## Client: complete accuracy evaluation

For the empty-cache full-split protocol, restart the server before this client,
even if the smoke/performance client has already passed on the same mode.

```bash
python tools/hyv4_realworldqa_eval.py \
  --base-url http://127.0.0.1:8010 --model hy4-vl \
  --data-dir /tmp/realworldqa/data \
  --mcq-extractor /tmp/realworldqa/mcq_extract.py \
  --max-pixels 1048576 --max-tokens 16 --concurrency 1 \
  --output-dir /tmp/hy4-accuracy-eager
```

`--max-pixels` records the server setting; it does not reconfigure the server.
Use the same value on both sides. The evaluator requires all 765 original items,
preserves original prompts, uses temperature 0 / top-p 1 / seed 0 / no_think,
and reports HTTP failures, empty answers and truncations without dropping them.
Use a new output directory for each mode. Resuming requires matching code,
dataset, scorer and settings; do not mix modes in one results directory.

Compare original predictions as well as the aggregate score. For two completed
runs, after changing the paths to the directories of interest:

```bash
python - <<'PY'
import json
from pathlib import Path
runs = [Path("/tmp/hy4-accuracy-eager"), Path("/tmp/hy4-accuracy-full-piecewise")]
rows = []
for run in runs:
    summary = json.loads((run / "summary.json").read_text())
    assert summary["complete"] and summary["completed"] == 765
    assert all(summary[k] == 0 for k in ("http_errors", "empty_answers", "truncated"))
    print(run.name, summary["correct"], "/ 765")
    rows.append({x["index"]: x for x in map(json.loads, (run / "results.jsonl").read_text().splitlines())})
assert rows[0].keys() == rows[1].keys()
changed = []
for i in sorted(rows[0]):
    a, b = rows[0][i], rows[1][i]
    assert (a["image_sha256"], a["question"], a["answer"]) == (b["image_sha256"], b["question"], b["answer"])
    if a["prediction"] != b["prediction"]:
        changed.append(i)
print("Changed raw predictions:", changed)
PY
```

An equal score alone does not establish prediction or numerical equivalence.
This public-command comparison uses separate processes; the validation summary
distinguishes it from diagnostic same-weight runtime controls.

## Standalone TP8 ViT comparison

Stop the inference server first so all eight HCUs are idle. Run from the updated
PR checkout. Prepare the pinned dataset above, then create a detached checkout
of the reference version for the comparison:

```bash
# Canonical channels-last stride comparison (latest incremental change):
HY4_VIT_BASELINE=9ee4c46cce0c34fd8ad4bc19d687c3263f984e33
# Earlier spatial projection: 93e32720287a6b90b1938cf730d792e484fb726d
# Earlier activation copies: 94810016cb414f5b117d88bce8958541842152f4
git worktree add --detach /tmp/hy4-vit-reference "$HY4_VIT_BASELINE"
python - <<'PY'
import io
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

root = Path("/tmp/hy4-vit-inputs")
root.mkdir(parents=True, exist_ok=True)
small = Image.new("RGB", (384, 256), "white")
ImageDraw.Draw(small).rectangle((80, 40, 304, 216), fill="red")
small.save(root / "small.png")
rows = [row for p in sorted(Path("/tmp/realworldqa/data").glob("test-*.parquet"))
        for row in pq.read_table(p).to_pylist()]
assert len(rows) == 765
for index in (552, 571):
    image = Image.open(io.BytesIO(rows[index]["image"]["bytes"])).convert("RGB")
    image.save(root / f"realworldqa-{index}.png")
PY
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
torchrun --standalone --nproc-per-node=8 tools/hyv4_vit_bench.py \
  --model /module/Hy4-testing-Channel-FP8-w8a8 \
  --reference-checkout /tmp/hy4-vit-reference --tp-size 8 \
  --case small /tmp/hy4-vit-inputs/small.png \
  --case large /tmp/hy4-vit-inputs/realworldqa-552.png \
  --case two /tmp/hy4-vit-inputs/realworldqa-552.png /tmp/hy4-vit-inputs/realworldqa-571.png \
  --case mixed /tmp/hy4-vit-inputs/small.png /tmp/hy4-vit-inputs/realworldqa-552.png \
  --output-dir /tmp/hy4-vit-comparison
```

This repository tool loads actual vision weights into both towers, uses five
warmups and 15 CUDA-event samples per version/case, and writes
`vit-comparison.json` with medians, complete-output error metrics and joint vs
individual image checks. For this layout change, check that `relative_l2`
and `max_abs` are zero. It measures the encoder without language decoding or
HTTP overhead. Use a new output directory on each run.

The published ViT tables use separate host-side validation scripts with 30
samples per version/case and alternating execution order. The encoder gate
asserts exact complete outputs on every rank for all 765 dataset image rows.
The preceding activation-copy comparison used baseline `9481001`; the command
above compares canonical channels-last strides against `9ee4c46`. Select the
commented baseline for either earlier optimization. All serving flags remain
the same; the guarded layout choice is automatic. The command above is the
reusable shorter comparison; it does not rerun that full-split gate or the
765-question HTTP evaluation.
