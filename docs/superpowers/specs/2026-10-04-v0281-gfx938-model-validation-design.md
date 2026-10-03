# vLLM 0.28.1 gfx938 model validation and compatibility design

## Objective

Validate the vLLM 0.28.1 DTK wheel and the stacked HCU plugin changes on the
local eight-card BW1100/gfx938 host, fix any plugin-owned compatibility or
accuracy failures, and submit exactly one new plugin PR on top of the existing
PR stack. The result must cover Model Runner V2, the default CUDA Graph policy,
tensor parallelism, supported data/expert parallel paths, MTP, prefix caching,
HND/BHSD FlashAttention cache layout, and native `fp8_e4m3` KV cache.

Every supported full checkpoint under `/models` receives a fresh HumanEval16
run on this gfx938 machine. Results obtained on a different HCU generation are
historical context, not acceptance evidence.

## Frozen inputs and provenance

The target framework distribution is:

```text
vllm==0.28.1+dtk2604.torch2110.2609171627.g77acaf
index=https://pypi.sourcefind.cn/nightly/dtk/
```

The validation record must include:

- the exact downloaded vLLM wheel filename and SHA256;
- installed vLLM and plugin versions and module paths;
- DTK, Torch, Python, AITER, BoltOPs, FlashAttention, FlashMLA, LightOp,
  DeepEP, DeepGEMM, and LMSlim versions;
- the plugin base, PR head, and candidate source SHAs;
- the model configuration and checkpoint-index checksum or stable metadata;
- the final launch and evaluation commands.

The wheel name declares the Torch 2.11/DTK 26.04 ABI line, while the container
currently has a later Torch 2.11 DTK build. Record both values rather than
claiming that their complete build identifiers are identical.

Final evidence must come from clean wheel installs in isolated targets with
`PYTHONNOUSERSITE=1` and a branch-specific `VLLM_CACHE_ROOT`. Imports from the
plugin checkout or a source `PYTHONPATH` are invalid as release evidence.

## Branch and PR topology

Use this stack:

```text
v0.28.1-dev
  -> PR #180: fix/flash-attn-unquantized-cache-dtype
    -> rebased PR #181: gfx938 FP8 sparse-indexer reader
      -> one new validation/fix PR
```

Rebase the existing PR #181 commits onto the exact PR #180 head. Resolve its
overlap in `patch_glm5next_channel_fp8.py` and
`rocm_aiter_mla_sparse.py` by current ownership and behavior, not by choosing
one side wholesale. Re-run the complete changed-test set after the rebase.

Create the only new PR from the rebased #181 head and set its base to the #181
branch. The new PR contains only evidence-backed compatibility fixes, focused
regressions, and validation documentation needed by this task. Do not add a
second vLLM-core PR. If an official interface is genuinely missing, record it
as a blocker before changing framework code.

Review the complete diff before every commit, review the committed diff again,
and review the final remote range relative to the rebased #181 head before
requesting merge. Do not store the supplied GitHub credential in a repository,
log, command transcript, or documentation.

## Model scope

DeepSeek-V4.1 is explicitly excluded because the current plugin does not yet
support it. Record the exclusion; do not treat it as an attempted failure.
`/models/dspark_qwen3_8b_block7` is a five-layer component artifact without a
complete tokenizer, not a standalone serving checkpoint. Cover it only with
component contract tests.

The following 15 full checkpoints require HumanEval16 on gfx938:

| Model | Primary path | TP gate |
| --- | --- | ---: |
| `DeepSeek-V4-Flash-0731-FP8-Channel` | sparse MLA, channel FP8, MoE, embedded draft | 4 |
| `GLM-5-W8A8` | sparse MLA, W8A8 MoE, embedded draft | 8 |
| `GLM-5.3-Channel-FP8-w8a8` | sparse MLA/indexer, channel FP8, embedded draft | 8 |
| `GLM-5___1-Channel-FP8-w8a8` | sparse MLA/indexer, channel FP8, embedded draft | 8 |
| `Hy4-preview-Channel-FP8-w8a8` | sparse MLA, channel FP8, embedded draft | 8 |
| `MiniMax-M2.5-Channel-INT8-w8a8` | FlashAttention, channel INT8, Eagle3 capability | 4 |
| `Qwen2-57B-A14B-Instruct` | FlashAttention, BF16 MoE | 2 |
| `Qwen3-30B-A3B-Channel-INT8-w8a8` | FlashAttention, channel INT8 MoE | 2 |
| `Qwen3-8B` | dense FlashAttention control | 2 |
| `Qwen3.5-35B-A3B` | hybrid GDN/FlashAttention, BF16 MoE, MTP | 2 |
| `Qwen3.5-35B-A3B-W8A8` | hybrid GDN/FlashAttention, INT8 MoE, MTP | 2 |
| `Qwen3.6-27B-W8A8` | hybrid GDN/FlashAttention, W8A8 | 2 |
| `Qwen3.8-27B-Channel-INT8-w8a8` | hybrid GDN/FlashAttention, channel INT8 | 2 |
| `Qwen3.8-Flash-Next-FP8-Channelwise` | Qwen4Exp hybrid, channel FP8, MTP | 4 |
| `Qwen3.8-Flash-Next-w4a8-slimquant` | Qwen4Exp hybrid, SlimQuant W4A8, MTP | 4 |

These TP values are starting gates based on model size, head divisibility, and
available memory. Adjust only when a measured loading or model-contract
constraint requires it, and record the reason. Do not use TP8 for a model that
has a meaningful smaller TP gate. A DP-only run never substitutes for the TP
gate.

## Runtime contracts

### Model Runner V2

Require the HCU worker to construct `HcuGPUModelRunnerV2`, which subclasses the
official runner from `vllm/v1/worker/gpu/model_runner.py`. The generic log
`Initializing a V1 LLM engine` is not runner-generation evidence. Record the
worker class, runner class, and imported runner file. A false
`use_v2_model_runner` must remain fail-closed.

### CUDA Graph

Omit `--compilation-config` for the primary acceptance command. Record the
resolved policy and require the graph modes used by real target and draft
requests to capture and replay successfully. An eager run or forced
`PIECEWISE` run is diagnostic only and cannot replace the default-Graph gate.

DeepEP high-throughput may officially select `cudagraph_mode=NONE`; do not
override that decision. This task's distributed priority is
`deepep_low_latency`, where the default graph path must be validated.

### KV layout and dtype

For HCU FlashAttention, set `VLLM_KV_CACHE_LAYOUT=HND` in the dedicated layout
gate and verify that the plugin forwards vendor layout `bhsd`. The NHD alias
maps to `bshd` and remains a control. Verify the complete writer-to-reader path,
including page shape, physical strides, storage offset, cache overwrite,
prefix-cache reuse, and graph replay.

Do not force HND/BHSD onto FlashMLA or sparse-indexer caches. Record their
resolved native layout and validate the layout expected by each writer and
reader. Keep official allocation and pass a restored physical-page view only
to an HIPC callsite that requires it.

Native `fp8_e4m3` coverage must include:

- dense FlashAttention with `Qwen3-8B`;
- hybrid/GDN FlashAttention with Qwen3.5 and Qwen3.8 Flash-Next;
- sparse MLA/indexer with `GLM-5.3-Channel-FP8-w8a8`;
- `DeepSeek-V4-Flash-0731-FP8-Channel` when its current attention contract
  accepts ordinary E4M3 rather than a model-specific format.

Use BF16/auto KV cache as the paired control. Do not claim E4M3 support from a
cache write, unit kernel, or graph capture alone; require correct repeated
output, prefix hits, and HumanEval.

### MoE, quantization, and providers

Treat the container AITER as a proprietary HCU ABI. For explicit
`--moe-backend aiter`, record the tuned configuration lookup, shape key, lookup
result, and concrete expert implementation. Missing configurations must fall
back to the official Triton experts before any incompatible weight mutation.

Keep AITER MoE, dense LightOp/official scaled-mm, BoltOPs FLA, SlimQuant,
DeepEP, and DeepGEMM as separate backend decisions. Never infer one provider
from another provider's CLI flag. A provider that changes packed weight layout
must prove complete reachable-shape support before loading the model.

## Validation sequence

### Gate 1: source and wheel tests

1. Run every test file changed by the full #180 plus rebased #181 range.
2. Run the full `tests/runtime_patch` suite and packaging/lifecycle tests.
3. Run focused cache-writer, cache-layout, FP8 sparse-indexer, MTP, DeepEP,
   quantization, and custom-op schema tests.
4. Build a clean plugin wheel and validate parent and child-process import
   roots from an isolated install.

When a failure is found, add or tighten the smallest reproducing test before
changing production code. Keep fixes at the current official extension
boundary and preserve complete current signatures.

### Gate 2: per-model TP and HumanEval16

For every in-scope checkpoint:

1. start the canonical TP topology with the model's required attention and
   MoE/quantization routes;
2. verify MRV2, resolved cache layout/dtype, provider selection, default Graph,
   and API readiness;
3. send a deterministic smoke request;
4. repeat a sufficiently long shared prefix and require nonzero hits plus
   coherent output;
5. run HumanEval samples 0 through 15;
6. collect service metrics and terminate the complete service process group.

Use `--reasoning-parser minimax_m2` for MiniMax. Inspect each GLM chat template
and use `glm45` when generation begins inside a think section. GLM-5 variants
may use `enable_thinking=false`; GLM-5.3 uses `reasoning_effort=low`. Freeze the
actual request body per model before baseline/candidate comparison.

### Gate 3: distributed paths

TP is mandatory and independent of distributed replication. After the TP gate:

- Prefer `DP8 + TP1 + EP8 + MTP3 + deepep_low_latency` for a one-card-capable
  MoE model with real MTP support, starting with
  `Qwen3.5-35B-A3B-W8A8`.
- Also validate at least one topology that combines TP, DP, EP, MTP, and
  low-latency DeepEP: first `DP4 + TP2 + EP8 + MTP3`, then
  `DP2 + TP4 + EP8 + MTP3` if required.
- Approximately 700 GiB checkpoints use TP8/EP8 and do not claim DP2.
- Confirm every DP rank receives real requests and record per-rank backend,
  prefix-cache, graph, and MTP evidence.
- Compare MTP against the same topology without MTP. For DP plus MTP, run the
  same HumanEval16 protocol twice in one healthy service before assigning a
  one-sample difference to a code regression.

DP8 is a preferred evidence point, not an excuse to alter model semantics or a
hard blocker when memory or an audited framework capability prevents it.

## HumanEval protocol and acceptance

Use the same 16 sample IDs, prompt template, deterministic sampling, API
protocol, parser, batch size, request body, and token limit for baseline and
candidate comparisons. Clear all upper- and lower-case proxy variables and set
both `NO_PROXY` and `no_proxy` for localhost.

The target result is 16/16 for every in-scope model. Preserve every first-run
failure and its completion, finish reason, and generated-token count. A
reasoning completion that ends at `max_tokens` is an evaluation-protocol issue
until a predeclared larger budget disproves it; do not silently add a stop
sequence or change extraction only for the candidate.

Keep a fresh output directory for every run. EvalScope correctness helpers must
run from a real guarded Python file rather than a stdin heredoc so
`multiprocessing.spawn` can re-import the harness.

## Failure isolation and repair policy

Classify failures at the owning boundary before editing:

- model/config import and weight loading;
- backend selection or tuned-config lookup;
- writer/cache layout/reader data flow;
- MTP metadata, cache ownership, or sampling;
- TP/DP/EP collective ownership;
- graph capture/replay;
- evaluation parser, prompt, or truncation.

Use baseline/candidate A/B runs with identical commands. Read the failure
ledger before applying a fix that resembles a recorded failure. Do not disable
prefix caching, CUDA Graph, MTP, FP8 KV, MRV2, or a requested backend to declare
success. Do not introduce a model-specific scheduler override for a generic
cache or collective contract.

## Conditional performance work

Correctness and compatibility are the primary scope. Compare warmed,
synchronized steady-state measurements against the rebased #181 baseline.
Consult SGLang-DAS only when there is a measured regression or a current,
contract-compatible operator route with a repeatable improvement of roughly
five percent or more on critical model shapes.

Port one owned operator at a time. Include layout conversions, copies, metadata
construction, and graph replay in the measurement. Retain the current provider
when a candidate wins only in an isolated eager microbenchmark or changes the
current vLLM ABI.

## Evidence, teardown, and deliverables

Start every validation service as the foreground process of a PTY session.
Never use `nohup` in a one-shot executor. Use unique ports, cache roots, temp
directories, logs, and EvalScope result directories. On shutdown, terminate the
owned process group, confirm all workers exit, inspect shared-memory cleanup,
and require HCU memory to return to its idle baseline.

The final handoff includes:

- the rebased and revalidated PR #181;
- exactly one new stacked plugin PR;
- a per-model matrix with commands, versions, paths, resolved settings,
  HumanEval16 results, prefix hits, graph evidence, and MTP metrics;
- the pre-commit, post-commit, and final remote-range code review;
- explicit remaining limitations, including the DeepSeek-V4.1 exclusion;
- evidence-backed updates to `/models/upgrading-vllm-hcu/SKILL.md` and its
  validation/failure references.

The local skill update is not placed in the plugin PR. Record verified facts,
commands, failures, and limitations; do not turn an unverified workaround into
guidance.
