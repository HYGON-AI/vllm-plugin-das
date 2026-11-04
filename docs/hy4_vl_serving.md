# HYV4VL server and client commands

Run these commands from the PR checkout in the tested DAS vLLM 0.25.1 / DTK2604
image, after installing plugin dependencies and building its native extension
(`MAX_JOBS=8 python setup.py build_ext --inplace`). The tested checkpoint is
`/module/Hy4-testing-Channel-FP8-w8a8` on eight idle BW1101 HCUs. These are image
serving commands; video and speculative decoding are unsupported.

## Server

Choose one execution mode. `eager` is the default fallback. Graph configurations
are experimental: see the [decode graph report](hy4_vl_graph_performance.md) and
[full/piecewise report](hy4_vl_piecewise_performance.md) for precision, stability
and performance limits. Full/piecewise fails this checkpoint's documented
accuracy gate; its commands reproduce that experiment. Stop the previous server
before starting another mode.

```bash
# Run in the plugin checkout. Choose: eager, decode, or full-piecewise.
HY4_EXECUTION=eager
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
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
latter configuration includes larger token buckets. ViT remains eager in both.

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
[RealWorldQA report](hy4_realworldqa_validation.md).

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
This public-command comparison uses separate processes; the reports distinguish
it from diagnostic same-weight runtime controls and retain observed differences.
