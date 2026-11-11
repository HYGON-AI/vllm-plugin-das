# V41 PCP1 HCU FP8 indexer correctness route

> Historical scope: the sections below record the initial `bbb99f3` eager,
> default-OFF implementation. The current uncommitted 2026-11-06 candidate is
> default-ON by explicit user choice and supports PIECEWISE breakable graphs.
> The published environment/operator names retain `REFERENCE` for compatibility;
> the device-only Triton decode is an **active candidate**, while the scalar CPU
> implementation remains an independent replay reference. It is not certified
> for deployment by its name, registration or component results.
>
> Current validation and remaining distributed blockers are tracked in
> [v41-graph-replay-stall.md](v41-graph-replay-stall.md): 200 real-HCU numeric
> cases and 400 changing-metadata capture/replays passed, alongside the existing
> CPU 6/6 replay. Full-model DP8/EP8 graph completion and HumanEval acceptance are
> still pending. The Adapter C4 bridge is ratio-4/NORMAL-only and cannot replace
> this ratio-1/2 SHUFFLE candidate merely by widening an architecture gate.

## Scope and actual source findings

Start Plugin3278f1e; Core read-only368701db06; Adapter read-only3257a934.
Only the installed new package `adapter_extension/src/hcu_vllm_das` is active;
legacy module_exchange incompatibilities are not blockers for this path.

Core `SparseAttnIndexer.forward_hip:1163-1208` refuses AITER-off gfx936/gfx938.
The lower ROCm operator carries the current candidate ABI, but bypassing the gate
is unsafe: `rocm_aiter_mla_sparse.py:55-57` identifies V41 ratio1/2 SHUFFLE while
its paged Torch fallback441-539 reads NORMAL/block-flat (next_n1) or per-row
data/scale(next_n>1). SHUFFLE uses16x16 tiles with all per-token FP32 scales at
the end of each block (kernel217-296). Its torch schema is also not guaranteed
registered when AITER/architecture support is absent (_aiter_ops3023-3045).

This increment therefore publishes an owned OOT SparseAttnIndexer and a unique
operator. It is explicit opt-in `VLLM_HCU_DSV41_FP8_INDEXER_REFERENCE=1` and only
handles HCU V41 FP8, ratio1/2, source-owned K insertion, PCP1/DCP1 and eager mode.
It does not spoof hardware, enable AITER or change native classes. Other models
and disabled policy retain native routing. The owned decode reads the actual
SHUFFLE cache, keeps per-query causal limits and -1 padding. Prefill reuses the
native layout-aware K gather and dense Torch logits function. Candidate producer/
mask kernels remain native; local top-k preserves short-row order and does not
resurrect masked scores. Cache writes/source ownership stay in native V41.

This is a slow reference/correctness route, not a tuned kernel replacement.
Host metadata reads make it eager-only. Graph/PCP8/DCP/MXFP4 remain separate work.
The raw compressor has quant_config=None and native unquantized weights remain
NT; don't restore the old global transpose. Native attention, inverse RoPE,
metadata topology, SWA and compressor ownership are preserved.

## Failure modes before implementation

| ID | Failure | Fixed replay boundary |
|---|---|---|
| I01 | HCU gate bypass selects wrong byte layout or missing native torch op. | Actual native class construction via OOT, unique schema and source API. |
| I02 | SHUFFLE values/scale tail or page stride interpreted as NORMAL. | 200 seeds independently pack16x16 cache and compare bytes/decode logits; page permutation and block1 boundary. |
| I03 | next_n or ragged padding allows future keys or wrong row order. | Ordinary/uniform/per-query limits and heterogeneous decode_lens; exact -inf masks, -1 outputs. |
| I04 | Short topk or candidates resurrect masked rows. | Exact short-row indices/-1 sentinel and producer/consumer candidate forwarding. |
| I05 | Native source K writes duplicate. | Owned path requires skip_k_cache_insert and k=None; cache bytes remain unchanged. |
| I06 | Prefill chunks/global offsets are lost. | Native metadata/chunk loop with actual Torch logits; CPU gather/candidate kernels explicitly doubled. |
| I07 | Unrelated models or disabled policy are intercepted. | Same native constructor and exact super forwarding; flags are opt-in. |
| I08 | Unsupported Graph/PCP/DCP/FP4 accepted then fails late. | Reject at owned construction with concrete reason, no width whitelist. |
| I09 | Partial registry/preparation or policy drift reported healthy. | Foreign owner preflight, sticky fatal, owned constructor health and frozen policy. |
| I10 | Compressor layout changed unnecessarily. | Actual unquantized merged loader/postprocess retains NT and native raw FP32 projection consumer. |
| I11 | CPU doubles overclaimed as GPU/model acceptance. | Source hashes/SHAs/CUDA false, exact doubled boundaries logged. |

## Implementation and usage

`das_hcu_v41_indexer` uses the public exact-name OOT registry and a unique
`hcu_v41_fp8_indexer_reference` operator. It never assigns native AITER flags,
registers under a native schema or replaces native classes/methods. Construction
checks the frozen policy and rejects unimplemented modes before forward.
The fake and mutable-buffer declarations follow the native candidate/topk ABI.
Disabled policy/non-V41 models call the original `forward_hip`; the separate
native MXFP4 indexer class is not replaced.

Enable the correctness route only in a fresh process with:

```bash
export VLLM_HCU_DSV41_FP8_INDEXER_REFERENCE=1
# Model startup must also use --enforce-eager and FP8 indexer cache.
```

Default is OFF. The construction log prefix is
`HCU_V41_FP8_INDEXER_REFERENCE`; that log proves selection, not GPU success.
Registered capability `hcu-v41-fp8-indexer-eager-reference` means the public
adapter is available, not model correctness or a fast kernel.

Decode gathers only each request's referenced pages before reversing16x16 tile
order. It reads the separate FP32 scale tail and forms effective per-query
causal bounds. Q/output rows remain in actual query order while candidate-bank
rows retain Core's padded request-major ABI when `requires_padding=True`:
request r uses rows r*max_decode_len through its valid query count. Padding
candidate rows are reset to-1 on publication. The per-request computation avoids
allocating padded Q/weights while performing equivalent row remapping; final
outputs stay compact. Candidate mode with padded decode and simultaneous prefill
is explicitly rejected because the native shared-bank ranges overlap. Monokernel
mode is rejected because it can bypass this owned route. The fixed replay
exercises producer then consumer in the native physical bank layout.
Ratio2 multi-query decode requires explicit
effective per-query metadata; a compressed final length alone cannot recover
individual query rounding. Short rows sort valid indices by original position;
masked -inf entries remain-1. Shared output buffers may be wider than current
K; only the selected-width slice is scored. Cache Parameters/storage are never
written. Prefill retains native layout-aware gather, chunk offsets and dense
Torch logits; existing candidate select/mask kernels are reused.

Profiling without runtime metadata delegates only the native reservation/fake
branch, never its faulty paged decode. This preserves the existing upper-bound
reservation contract but is not a real-device workspace acceptance result.
Reference decode uses host `.item()`/`.tolist()` and Torch temporaries; it is
intentionally eager-only, slow, and not a replacement for a tuned HCU provider.

## Results

Before implementation, fixed replay `red-fixed/` recorded **0/5 passed** for
missing production modules. An earlier attempt failed at an unused fixture
import before running cases; it remains in `red/`. After production import,
the replay passed **5/5** (`green/`, then `bounded/`). Review identified that
candidate banks must retain native padded request-major rows; production now
maps those rows explicitly. The existing ragged fixture was corrected to inspect
second actual query at physical row2 and supply consumer candidates over all4
padded rows. Logical selection/count/causal assertions and thresholds were not
weakened and no new scenarios were added. The initial passing compact-bank snapshot is not acceptance of native bank
interoperability. After correction, `padded-final/` passed **5/5**. Independent
read-only re-review confirmed native padded-bank mapping, zero-count slots,
identical source/consumer slices and compact final output order, with no further
concrete blocker. No production code was committed before resolving this review.

The200-seed differential independently writes packed cache bytes using the
writer's offset formula, then compares actual decode math to a logical-key
reference over shuffled page tables, next_n1/2/3, final1D/effective2D lengths,
scale tails, causal -inf masks and block1. Logits match with atol=rtol=0 for
these binary-exact FP8/scale fixtures. This is not a real GPU arithmetic result.
Other cases construct the actual native SparseAttnIndexer through OOT,
instantiate native metadata dataclasses, exercise ragged decode candidates and
prefill chunks, verify cache immutability/-1 padding, and verify native delegation
for disabled policy/other models plus unsupported-mode rejection.

Existing portable tests passed **312 passed,66 skipped,14 warnings in178.85
seconds** (`regression.log`/`regression.xml`). After shared-buffer width,
constructor policy and native op-control-name adjustments, the final tree also
passed **312 passed,66 skipped,14 warnings in166.76 seconds**
(`regression-final.log`/`regression-final.xml`). After the padded-bank ABI review
fix, the exact final tree passed **312 passed,66 skipped,14 warnings in176.00
seconds** (`regression-padded.log`/`regression-padded.xml`). Final fixed replay/source
hashes and actual native module paths are retained in `padded-final/report.json`.
No scenarios or numerical thresholds were changed after implementation; metadata
representation mistakes were corrected as described above and environment-path
reporting fields were added to the existing report only. Native GPU K gather/candidate kernels are CPU
doubles; layout/logit/selection math is actual CPU Torch. No GPU/model forward,
AITER, native writer/quantizer, Graph replay, distributed progress, model accuracy,
performance or gfx938 execution is certified. Constructor/profiling/layout/source
interfaces not reached by the fixed replay remain source-reviewed boundaries.

Reproduce using existing Python3.12 with pinned Core/Adapter siblings:

```bash
VLLM_TEST_PYTHON=/path/to/python3.12 \
  bash tests/vllm033/run_v41_fp8_indexer_cpu.sh /path/to/output
```

Evidence root `tests/vllm033/.cache/v41-fp8-indexer/` stores JSON, stdout and
JUnit; source paths/SHAs/hashes and `cuda_initialized=false` are recorded. The
existing bw47 container was reused with installed metadata `vllm0.33.0+rocm72`;
Python source is explicitly pinned/selected, not inferred from metadata. Missing
`_C_AVX2` warning is retained and not counted as native-ABI acceptance.

No compressor/wo_a layout change was made: native compressor BF16 NT weights and
raw FP32 projection already agree. Channel-FP8 wo_a remains explicitly native;
CT W4A8 wo_a still follows actual checkpoint targets/ignores, which this replay
does not certify. No README/Core/Adapter/legacy package changes or remote Git
operations. PCP8 attention admission, metadata/gather/cache-write ownership and
Engram sharding remain unfinished and must not be inferred from this PCP1 route.
