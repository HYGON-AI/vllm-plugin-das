# HYV4 Flash on vLLM 0.33 (candidate)

The plugin registers `HYV4FlashForCausalLM` and
`HYV4FlashVLForConditionalGeneration`. Model registration and static import
closure are not evidence that a checkpoint can be served on HCU.

## Repository boundary and dependency

- Reuse the vLLM/HYV4 loading and projection paths where their ABI matches.
- Keep Flash IndexPool/SWA selection, cache writes and sinks in this plugin.
- On HCU, require `hcu_adapter.bridges.vllm_paged_mqa.get_paged_mqa_provider`
  from the companion adapter-das change. Older adapters fail with an actionable
  dependency error; do not silently fall through to a CUDA DeepGEMM provider.
- On CUDA, continue using vLLM's own DeepGEMM wrappers. No framework source
  files or global DeepGEMM bindings are rewritten.

## IndexPool contract

HCU uses E4M3FN query values `[rows, 1, heads, 128]`; query scales are folded
into FP32 head weights. The page64 uint8 cache is physically laid out as
`[64*128 key bytes | 64 FP32 scales]` per page, NOT interleaved token records
and NOT the SGLang preshuffled layout. The adapter returns FP32 logits and
uses each query's block-table row independently, including prefill chunks.
The bridge currently supports `next_n=1` only. It is a Torch correctness
candidate, not a performance-qualified native kernel.

The FlashMLA path runs indexer and attention in a single eager break on the
same stream. It must not call ROCm-AITER's private logical-topk event hook.
The current runtime explicitly rejects speculative decoding.

## Validation gates

Quantized checkpoint completeness includes every routed layer's FP32
`expert_bias`, including the `gate.e_score_correction_bias` checkpoint alias;
do not rely on Core's optional quantized tracker to detect a whole missing bias.
For visual 2D RoPE, the learned position-embedding edge is a training grid,
while preprocessing can admit longer rectangular grids. The tower retains
Core's original table and uses its inverse frequencies for a local,
nonpersistent extension when needed. CPU tests cover real admitted profiling
grids, repeated growth without a construction config context, and unchanged
small-grid values. Encoder Graph and installed VL model validation remain open.

Image profiling follows the processor's input aspect check before rounding.
For example, legal `8192x41` input becomes `8192x32`, with 512 tokens when
row newlines are enabled. A dummy image can therefore have unaligned raw
dimensions; token budgets use its actual resized grid. The search also covers
downscale floor grids below `min_pixels`, both closed input-aspect endpoints,
and the separate alignment/max-pixel contract when resizing is disabled.
It does not tighten request admissibility or change image preprocessing.
Regression coverage includes real resize/profile/token-count function bodies,
an exhaustive finite raw-input family and independent sampled raw inputs.
This CPU coverage does not qualify the installed multimodal dispatcher,
checkpoint loading, encoder Graphs or model quality.

CPU-only seam tests:

```sh
python -m pytest --noconftest tests/models/hy_v4/test_flash_runtime_contracts.py
```

These test provider selection, the v0.33 three-argument MX helper ABI and
FlashMLA orchestration. They do NOT import the complete model or execute
HCU kernels. Before this change is ready to merge, run the companion
adapter's numerical tests on HCU, changing-input graph replay, real vLLM
model import/config/cache setup, checkpoint loading, multi-request short
and long prefill/decode, sink accuracy and text/VL end-to-end comparisons.
Verify the installed wheels and selected backend; report hardware, DTK,
PyTorch and vLLM versions with results. PP/TP/DCP and quantized cache
configurations need their own qualification. Do not infer those from CPU
contracts. FlashMLA/sink provider ABI remains a device validation gate.

## Recorded focused HCU validation (2026-11-17)

On BW1100/gfx938, Python 3.12.12, Torch 2.13.0, HIP 7.2.26365 and
vLLM 0.33.0+dtk2610.torch2130.2611101131.g7364da:

```sh
python -m pytest --noconftest tests/models/hy_v4/test_flash_runtime_contracts.py tests/models/hy_v4/test_registration.py tests/models/hy_v4/test_flash_indexpool_provider.py
```

**20 passed** (14 upstream Torch deprecation warnings). The integration test
uses the actual Flash FWHT quantizer and Triton page writer, then scores the
produced shared/offset cache through adapter-das against independent FP32
math, including Q-scale folding and inactive writes. The companion adapter's
focused suite and existing CPU ops reported **76 passed**, including changing
input Graph replay. Text/VL/config module import smoke also passed; it emitted
an AITER HIP-runtime discovery warning, so AITER JIT availability is not claimed.

Source overlays did not replace installed packages. These results do not
establish real checkpoint loading, full model graph execution, text/VL
end-to-end accuracy, distributed serving, wheel installation or throughput.


## IndexPool no-pruning optimization

When the CPU-known sequence upper bound is at most
`(compressed_indexer_group_topk) * compressed_indexer_compress_ratio`, all
completed historical groups fit in the budget reserved before the current tail
group. Sorted expansion is exactly the causal range `[0, position]` with `-1`
padding. The plugin can skip query projection, query RoPE/FWHT/quantization,
head weights, paged logits and group top-k. **All pooled-key and tail cache
writes must remain before this early return** so decode can cross the boundary.
FULL graph mode uses configured `max_model_len`, never the first replay's
sequence length; no device scalar is read back to choose the branch.

Tests cover exact index equality, inactive rows, runtime/static graph bounds,
and a real HCU cache transition from the unpruned to scored route, comparing
both indices and cache bytes. Longer-context scoring can opt into the adapter's
existing-Core Triton provider using public HCU_CONFIG; its default is unchanged.
Operator Graph qualification does not certify full-model graph execution.


## Community PR alignment and DFly qualification (2026-11-20)

Reference: vllm-project/vllm PR #68296, merged commit
`b067b6f6fa509a8c98c5a79d75b2aa3742d539c6`. Compare that pinned source,
not a moving main branch. The existing Flash target is not feature-equivalent
to this PR: the community implementation also adds a DFly draft model,
target-layer logical-index snapshots, writable MLA KV aliases, configuration
registration and draft-specific checkpoint loading.

The required qualification configuration is:

```text
--max-model-len 8192
--speculative-config '{"num_speculative_tokens":5,"method":"dfly"}'
--attention-backend FLASHMLA_SPARSE
```

This is a **qualification target, not a supported launch recipe yet**. The
unmodified v0.33 runtime rejects `method="dfly"` before loading a model. Its
speculator factory and configuration schema lack the community DFly entries.
The pinned community DFly quantization validator also rejects the channel-FP8
weights/dynamic per-token activation recipe used by the HCU checkpoint.
Do not bypass these guards or substitute DFlash/MTP while advertising DFly.

Three independent prerequisites are now aligned with the community behavior:

- Snapshot logical target indices immediately after the indexer and before
  attention can overwrite/remap the shared buffer. Preserve unused rows.
- Capture IndexPool metadata with the full configured context bound rather
  than the short dummy batch's bound.
- Return invalid Engram hashes for all-padding/no-request metadata without
  reading checkpoint tables or lookback state.

The prerequisites alone do not enable speculative decoding. An explicit,
experimental `dfly_compat.install_v033_dfly()` path now owns configuration,
speculator-factory and lookahead hooks without editing installed packages or
replacing the upstream configuration class. It is **not enabled by the standard
platform dispatcher**. Non-DFly methods delegate to their original constructors
and factory. First qualification is eager, text-only, TP-local and BF16 compute.
The draft derives from the same checkpoint, uses the original channel/block-FP8
recipes, and preserves separate q_a/kv_a projections in the v0.33 HCU layout.
For the owned DFly configuration only, the v0.33 compatibility layer also
backports MLA-aware KV-sharing discovery from the pinned PR. Shared layers are
inserted idempotently into target groups; heterogeneous groups inherit the
target's per-layer spec. Ordinary methods retain the upstream discovery/group
helpers. KV storage identity/layout validation remains mandatory before writes.
Source-overlay configuration tests do not establish model-forward accuracy.
The sink-capable plugin backend retains the
`FLASHMLA_SPARSE` name and its HCU-specific scheduler/provider wiring.

Before submitting or updating an MR, qualify the exact flags above on HCU:
load all required draft weights with their original scale axes; verify target
layer selection, per-layer logical snapshots, KV alias geometry and lifetime;
exercise accept/reject, all-padding and 8192 boundary cases; then run complete
HumanEval with EvalScope and audit truncation. Record draft acceptance counters
and demonstrate that the selected DFly speculator actually executes. A score
from a non-speculative service is not evidence for DFly accuracy. Full-model
Graph, multimodal and independent image/wheel qualification remain separate
from operator and source-overlay tests.


### Sparse index readiness (local candidate, 2026-11-20)

Flash publishes logical top-k through Core's `record_logical_topk_ready()`
before MLA attention. Core's sparse index group can remap on a side stream,
including mixed FP8 batches classified as decode; sharing the tensor does not
make that stream wait for the producer. Keep this event even when an external
producer skips the indexer. DFly snapshots precede the ready event and attention.

The focused source-overlay regression has 264 passing tests, including a real
Core-group delayed-producer stream regression. This does not qualify FULL/
PIECEWISE graphs, a wheel/image, standard dispatch, or end-to-end accuracy.
New full HumanEval validation is separate; cache-miss/hit token equivalence has
not been established. No environment or global synchronization workaround is
part of this fix.

### Historical no-thinking validation (2026-11-20)

The historical no-thinking source-overlay EvalScope HumanEval DFly run scored
151/164 (92.0732%), with 164 normal stops and no truncation, request errors,
or syntax failures. This is lower than the historical changed-source 153/164
run, not a passed accuracy release gate. The completed matched-current-source non-DFly full control scores 153/164
(93.2927%), a net two-problem gap. DFly adds failures 93/115/160 and recovers 83;
89 answers are text-identical and 75 differ under independent batch2 schedules.
Both runs have no truncation or syntax failures; token equivalence is not established.
Final DFly counters before boundary traffic are 3498 draft rounds, 17490 drafted
tokens and 14566 accepted tokens (83.2819%).

Three synthetic exact-token context smokes completed: 4096+16, 8176+16 and
8191+1 input+output budgets. The last two reach 8192 total tokens; they test
functional bounds only, not long-context semantic accuracy. No MR publication,
wheel/image qualification or standard-dispatch readiness is implied.

A separate fixed-input native diagnostic found reproducible query-row-dependent
rounding for 1/2/6 rows versus a 205-row suffix (max absolute difference
0.00048828125). Padding to 16 rows with masked extra indices eliminated that
captured-input difference, but this remains an experimental probe only: no
independent math reference, full-model cause attribution, production default,
or adapter dispatch change is qualified by it.


## Draft validation update (2026-11-21)

The explicit v0.33 DFly compatibility path now completes real text-only TP8
source-overlay/eager inference with the requested five-token DFly method and
`FLASHMLA_SPARSE`. This does not enable the standard platform dispatcher.
The runtime is specifically
`vllm-0.33.0+dtk2610.torch2130.2611101131.g7364da`, Torch 2.13/HIP 7.2,
on eight BW1100/gfx938 devices. Do not assume an arbitrary upstream 0.33 wheel
has the same reasoning-budget or worker APIs. Relevant installed runtime files
match their wheel RECORD; this qualification did not rewrite installed files.

EvalScope 1.15, official HumanEval 164 tasks, one sample per task, temperature 0,
seed 0, batch 2, high reasoning and native `thinking_token_budget=4096`:

| Lane | Official pass@1 | Truncation / empty final / syntax / request errors |
| --- | --- | --- |
| DFly, five speculative tokens, total length 8192 | **161/164 (98.1707%)** | **0 / 0 / 0 / 0** |
| Non-speculative target, identical 4096 thinking budget | Running; no final score yet | Audit pending |

DFly failed tasks 47, 116 and 145 are semantic failures under the unchanged
official tests. All 164 responses stopped normally; maximum total usage was
4800 tokens, with reasoning at most 4096. Generated code was scored in a
separate non-root, read-only, networkless sandbox, not the serving container.
Full-run metric deltas were 42,569 draft rounds, 212,845 drafted tokens and
83,887 accepted tokens (39.41% accepted/drafted). These prove DFly execution,
not speedup. No raw reasoning or private checkpoint assets are published here.

A previous high-reasoning target run **without** a thinking budget scored
155/164; eight tasks exhausted the total token budget and had no final answer.
It is not a matched control and cannot establish a DFly accuracy gain. The
historical no-thinking regression above is not considered fixed by changing
the reasoning protocol. Total length 8192 includes prompt, reasoning and final
answer, not 8192 final-answer tokens.

Keep the MR **Draft / experimental** until matched-budget comparison,
standard-dispatch/spawn lifecycle, independent wheel/image installation,
full-model graphs, long-context semantic accuracy and non-text/other parallel
configurations are qualified. Existing synthetic boundary smokes do not prove
long-context accuracy. New performance candidates must be isolated from the
accuracy snapshot, unit-tested against independent references and re-evaluated
end to end before default activation.


### Video spatial compression alignment

Real-video resizing and `do_resize=False` validation must use
`patch_size * merge_size * spatial_patch_size`, matching the dummy builder and
`vision_token_grid`. For example, a two-frame224×224 input with patch16,
merge2 and spatial compression2 must resize to192×192, not remain224×224.
The video alignment regressions exercise production preprocessing definitions
with real CPU Torch frames and isolated resize/result-container shells; they
do not qualify the installed processor, multimodal dispatcher or model.
