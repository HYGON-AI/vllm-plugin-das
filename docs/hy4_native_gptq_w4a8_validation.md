# Native HY4 GPTQ W4A8 validation

Checkpoint: `/data/models/hy4-preview-channel-int4a8-gptq-911-2`.
The native `hy4_w4a8_v1` format stores packed UINT8 INT4 weights and FP32
output-channel scales. Its MTP has 771 quantized projections (768 routed,
3 shared). The plugin adapts names and reuses the existing aiter W4A8
Linear/MoE methods for target and draft models.

## Scope and FP8 compatibility

The adapter is selected only with `quantization=slimquant_w4a8` and explicit
`checkpoint_format=hy4_w4a8_v1`. Native name normalization in target/MTP loaders
is guarded by that format. Existing channel/block FP8 quantization and launch
commands are unchanged. Do not apply the INT4A8 `--hf-overrides` below to FP8
weight checkpoints. KV cache dtype is a separate launch option.

## Serve

From the repository root, using the tested vLLM 0.25.1 DAS / aiter 0.1.6 DTK
runtime and 8 BW200 cards:

```bash
# Default KV cache, graph mode, aiter W4A8, MTP3
bash tools/hy_v4/serve_native_w4a8.sh

# FP8 KV cache comparison (run after stopping the first service)
bash tools/hy_v4/serve_native_w4a8.sh --kv-cache-dtype fp8_e4m3
```

HY4 sparse attention normalizes `fp8_e4m3` to `fp8_ds_mla` cache layout.
Neither command enables eager execution.

## HumanEval 32

Use the official first 32 numeric task IDs, one sample per task, temperature 0,
no_think, max_tokens 2048, batch size 1. Prepare the local dataset:

```bash
python - <<'PYDATA'
import gzip, json, pathlib, urllib.request
p = pathlib.Path('/tmp/hy4-humaneval32')
p.mkdir(exist_ok=True)
url = 'https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz'
rows = [json.loads(x) for x in gzip.decompress(urllib.request.urlopen(url).read()).splitlines()]
rows.sort(key=lambda x: int(x['task_id'].split('/')[-1]))
(p / 'test.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows[:32]))
(p / 'README.md').write_text('---\nconfigs:\n- config_name: openai_humaneval\n  data_files:\n  - split: test\n    path: test.jsonl\n---\n')
PYDATA
```

Run against the service (use a fresh output directory for each KV setting):

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost evalscope eval \
  --model hy4-gptq911 --api-url http://127.0.0.1:8000/v1 --api-key EMPTY \
  --eval-type openai_api \
  --generation-config '{"temperature":0,"do_sample":false,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"reasoning_effort":"no_think"}}}' \
  --eval-batch-size 1 --timeout 1800 --limit 32 --datasets humaneval \
  --dataset-args '{"humaneval":{"local_path":"/tmp/hy4-humaneval32"}}' \
  --work-dir /tmp/hy4-humaneval32-kv-fp8-result --no-timestamp
```

EvalScope 1.9.1 uses its default HumanEval prompt, code extraction and execution
checker, with a 4-second execution timeout. The canonical solutions passed
32/32 in the same environment. This subset result is not the full 164-task score.

## Results

| KV cache | HumanEval pass@1 | Accepted draft tokens | GPU KV cache capacity |
| --- | --- | --- | --- |
| auto (default) | 32/32 (100%) | 3013/3264 (92.31%) | 70,144 tokens |
| fp8_e4m3 (sparse fp8_ds_mla) | 32/32 (100%) | 3037/3300 (92.03%) | 120,256 tokens |

Both runs used TP8, aiter W4A8 target/MTP, MTP3, graph mode, and the same
32-task dataset and generation settings. The higher cache capacity was observed
with the same 0.95 GPU memory utilization setting. This is not a controlled
throughput benchmark or validation of FP8 weight checkpoints on hardware.

Regression validation: 205 tests passed across HY4 and FP8 runtime suites;
a separate 6-test native-format run passed, including two added channel/block
FP8 loader isolation cases. Production boundary and patch coverage checks passed.
