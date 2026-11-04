# LocateAnything Slow / Fast on vLLM-HCU

The HCU wheel contains the complete LocateAnything model, processor, connector,
configuration registration, Fast sampler, and Fast execution integration.
The existing `hcu_model` entry point registers both architectures. No separate
`locateanything-vllm-native` installation or external Fast `PYTHONPATH` overlay
is needed. Original model weights and tokenizer files remain in the checkpoint.

## Build and install

Use the matching vendor DTK, PyTorch and HCU vLLM runtime described in README.
Build a real native extension, then install the exact wheel without resolving
device dependencies:

```bash
source /opt/dtk/env.sh
export ROCM_PATH=/opt/dtk MAX_JOBS=8
python3 setup.py bdist_wheel
python3 -m pip install --no-deps --no-build-isolation \
  --index-url https://pypi.mirrors.ustc.edu.cn/simple/ dist/<exact-wheel-name>.whl
vllm-hcu-doctor --no-arm --json
```

Restart the Python service after upgrading. Keep `VLLM_PLUGINS` unset, or include
all of `hcu,hcu_model,hcu_ops` in an explicit allow-list.

## Serve

Select an idle device. From any directory outside the source checkout:

```bash
VLLM_PLUGINS=hcu,hcu_model,hcu_ops HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
python3 -m vllm_hcu.models.locateanything.serve --mode slow \
  --model /workspace/LocateAnything --host 127.0.0.1 --port 33345 \
  --max-num-seqs 16 --gpu-memory-utilization 0.6

VLLM_PLUGINS=hcu,hcu_model,hcu_ops HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 \
python3 -m vllm_hcu.models.locateanything.serve --mode fast \
  --model /workspace/LocateAnything --host 127.0.0.1 --port 33346 \
  --max-num-seqs 16 --gpu-memory-utilization 0.6
```

The wrapper only supplies the model launch profile and execs the standard vLLM
API server. It sets `VLLM_HCU_LOCATEANYTHING_MODE` before any vLLM imports.
For direct `vllm serve`, set this variable to `slow` or `fast` before startup and
use the same flags as the wrapper. Mode is process-wide: use independent servers
for Slow/Fast. Do not enable Fast for other model families.

The model names are `locateanything-native-slow` and `locateanything-native-fast`.
Use one image and the original detection question with `/v1/chat/completions`.
The original special-token output is interpreted by the official box parser.

## Ownership and execution

- `models/locateanything/`: exact image processing, prompt template, composite
  config, MoonViT + original connector + native Qwen2, and weight loading.
- `v1/locateanything/`: six-position input/RoPE, block sampling/rules, real KV
  accounting, scheduler, runner, attention mask, startup and API contracts.
- `patch/locateanything.py`: mode selection, qualified-source checks and seven
  exact cold module exchanges through the standard HCU import coordinator.

Slow uses the shared vLLM scheduler and paged KV. Fast consumes one to six real
tokens plus six temporary probe positions each decode step, and commits only
real KV. Both support continuous batching and decode graphs. Fast sampling is
outside the model-forward graph; the vision encoder uses ordinary packed FA2.

Fast preserves original block-level EOS/budget semantics, including up to five
tokens beyond the requested budget. It disables async scheduling/chunked
prefill and rejects unsupported streaming, multi-output and grammar requests.
The validated target is one BF16 HCU, TP=1 and one image per request.

Startup checks exercise request reordering and shrinking batches while checking
physical token IDs, shifted RoPE positions and committed KV progress. Synthetic
prompt lengths are bounded by the runner token budget; KV blocks cover each
prompt's generation budget and six probe positions. This keeps the startup
checks valid at capacities such as 128 and 256 without disabling assertions or
decode graphs.

The seven substantial Fast modules retain the original upstream vLLM license
headers. They are qualified against the exact base files listed in the packaged
`source_contract.json`. Fast fails before registration on source drift or late
module loading; upgrades require rebasing and retesting. No installed vLLM file
is rewritten. Slow and other model processes do not activate these exchanges.

## Verification

```bash
python3 tools/check_production_boundary.py --json
python3 tools/check_patch_test_coverage.py --json
python3 -m pytest tests/runtime_patch/test_locateanything.py \
  tests/runtime_patch/test_locateanything_warmup_budget.py -q
```

Hardware validation must check actual graph dispatch, changing request batches,
official prompt IDs, source/module origins, complete output records and official
COCO scoring. Diagnostic teacher forcing is not an accuracy or performance run.
