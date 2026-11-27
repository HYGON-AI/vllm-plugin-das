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

Fast interprets `--max-model-len` as the physical capacity, including six probe
slots. Its logical prompt/output limit is `max-model-len - 6` throughout API
validation, default output-budget calculation, scheduling and stopping. For
example, `--max-model-len 4096` allows 4090 logical tokens. Requests that exceed
this logical budget are rejected before generation; the final Fast block keeps
its original up-to-five-token budget overshoot and finishes before another
physical query is needed. Startup synthetic requests also respect this context
limit. The default physical/logical pair remains 16390/16384.

`--model` accepts either a downloaded directory or a Hugging Face model ID, such
as `nvidia/LocateAnything-3B`. The processor reuses vLLM's resolved image config;
fallback and standalone loading use the standard Transformers Hub resolver,
preserving revision, authentication and cache options. Offline use requires the
requested revision and files to be present in the Hub cache.

The model names are `locateanything-native-slow` and `locateanything-native-fast`.
Use one image and the original detection question with `/v1/chat/completions`.
The original special-token output is interpreted by the official box parser.
For this model, Chat in Slow/Fast and Completions in Slow preserve special tokens
and disable spaces between them. Fast image requests use `/v1/chat/completions`;
its Completions route returns HTTP 400 with that instruction. The adapter sets
`skip_special_tokens=False` and `spaces_between_special_tokens=False`, including
when a client sends `True`: BOX/REF markers and coordinate tokens are required
output syntax. Other model families retain their normal API behavior.

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

The Fast startup contract enforces this profile in both Scheduler and worker:
TP/PP/DP/PCP/DCP are all 1, Model Runner V2, TRITON_ATTN, unquantized BF16
weights/KV, FULL decode graphs, no prefix caching, no LoRA, and no KV/encoder
transfer or DBO/microbatching. Chunked/async scheduling, positive long-prefill
thresholds and token budgets too small for a complete query are rejected.
Global logits processors, replay/thinking/routed-expert debugging and prompt
embeddings are not implemented by Fast. These restrictions apply to Fast;
Slow keeps the upstream execution and sampling paths.

Startup checks exercise request reordering and shrinking batches while checking
physical token IDs, shifted RoPE positions and committed KV progress. Synthetic
prompt lengths are bounded by the runner token budget; KV blocks cover each
prompt's generation budget and six probe positions. This keeps the startup
checks valid at capacities such as 128 and 256 without disabling assertions or
decode graphs.
The synthetic request count is also capped by actual KV blocks across all cache
groups, reserving block 0. Fully using IDs through `num_blocks - 1` is valid.
Fast rejects vLLM speculative decoding before scheduler/worker setup; its own
six-position generation must not be combined with `--speculative-config`.
Fast requires `pipeline_parallel_size=1`: the launcher, scheduler and worker
reject pipeline parallelism before Fast warmup or device stream setup. Slow
does not use this Fast restriction.
Fast does not support resumable/session input. The client rejects both keyword
and positional `resumable` / `session_id` arguments before rendering or IPC;
the scheduler also rejects these flags and reuse of an active request ID.
It never merges input chunks into retained token history. Completed request
IDs can still be reused once the previous request has been removed.

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
  tests/runtime_patch/test_locateanything_warmup_budget.py \
  tests/runtime_patch/test_locateanything_limits_loading.py \
  tests/runtime_patch/test_locateanything_scheduler_edges.py \
  tests/runtime_patch/test_locateanything_input_boundaries.py \
  tests/runtime_patch/test_locateanything_contract_matrix.py -q
python3 examples/locateanything/validate_fast_rules.py --model-dir /path/to/trusted/checkpoint
```

Hardware validation must check actual graph dispatch, changing request batches,
official prompt IDs, source/module origins, complete output records and official
COCO scoring. Diagnostic teacher forcing is not an accuracy or performance run.

### Validation source trust

The standalone `validate_model_contract.py`, `validate_fast_rules.py` and
`validate_fast_contract.py` scripts execute selected, unchanged AST definitions
from local sources to compare the adapter with its reference implementation.
They are offline developer tools, never imported by a serving endpoint. Their
inputs are fixed package paths or the developer's explicit `--model-dir` and
`--vllm-source` directories. Use only trusted, reviewed local source trees;
these scripts are not a sandbox for untrusted checkpoints. The model-contract
validator also loads checkpoint Python through the Transformers local-code
interface, so its entire checkpoint directory must be trusted.
The Fast rules oracle resolves both `generate_utils.py` and
`mask_sdpa_utils.py` from `--model-dir`, loads the mask reference once, and
reports both reference SHA-256 digests for reproducibility. Missing reference
files produce an explicit CLI error; no machine-specific source path is used.

Fast scheduler admission validates the same logical prompt/output limits as
Chat, including offline engine requests. Waiting multimodal requests stay
queued if encoder or token budgets clip the full suffix plus six probes;
sampling starts only once the complete query and required image encoding fit.
Fast also rejects legacy `guided_choice` at the API boundary, consistently
with other unsupported structured-output options, rather than silently
ignoring a client's requested constraint.
Offline sampling parameters are validated before the scheduler queues or
registers a request. A Fast-only InputProcessor callback rejects them in the
client before EngineCore IPC, so ordinary invalid offline calls do not fail
inside the engine process. InputProcessor, admission and the worker share one sampling contract,
so unsupported stop strings/token IDs, penalties, logprobs and structured
outputs fail before request, model or block-table state is inserted.
Custom per-request logits processors are unsupported. The qualified vLLM
constructor already rejects the removed `logits_processors` field; the Fast
contract additionally rejects nonempty fields supplied by older layouts or
SamplingParams subclasses, instead of silently ignoring their constraints.

The finalized InputProcessor request also retains detection special tokens
with no added special-token spacing for both Slow and Fast, including ordinary
offline `LLM.generate` calls. Normalization owns a separate SamplingParams
struct even with `skip_clone=True`; the caller's format flags are not changed.
Fast validates again after generation-config processing and image expansion,
before the request reaches detokenization or EngineCore IPC. An unset output
budget is capped to the logical context; explicit overflows are rejected in
the client. Explicit `detokenize=False` remains a supported token-only request.
Thinking budgets, repetition detection, trace replay, selected-token logprobs
and custom extra args are rejected instead of being silently ignored.

| Interface / option | Slow | Fast |
| --- | --- | --- |
| HTTP Chat and offline generation | Detection tokens retained | Detection tokens retained |
| Dynamic batch and graph | Native vLLM | Qualified single-HCU FULL decode graph |
| Seeded sampling, temperature, top-p/top-k, repetition penalty | Native vLLM | Original Fast six-position rules |
| Unseeded requests | Native vLLM | Supported on the enforced single device |
| `max_tokens=None` | Native model limit | Physical context minus six probes |
| Streaming/session, custom processors, grammar/logprobs | Native capabilities | Rejected before IPC/admission |
| TP/PP/DP/context parallelism, LoRA, quantized KV, prefix cache | Native capabilities where supported | Rejected at startup |
| Direct internal EngineCore injection | Internal vLLM API | Caller must use the validated preprocessing contract |

The three inline `nosemgrep: rules.hygon.python.dynamic-eval` exceptions are
limited to these oracle calls. They do not suppress dynamic execution in the
production package or across directories. The ordinary unit tests import the
CPU-only warmup geometry normally and use a real fixture module for the globals
contract, with no `exec`. The cold-import regression runs in a child interpreter
and checks reentry as well as rejection of already imported targets, independent
of full-suite collection order.

Fast owns request RNG state by request ID across preemption, and releases it on
completion or abort, including cancellation while preempted. Slot-indexed RNG
and diagnostic references are released whenever a slot is removed. Reusing a
completed request ID creates a generator from the new request's seed.
