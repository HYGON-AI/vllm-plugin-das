# Opt-in HYV4 static-HT W4A8 scratch bounds

The conservative HCU workspace bound covers all routed tokens plus one padded
tail per local expert. For eager HYV4 static EPLB with the fixed DeepEP HT
communicator, the dispatch result already contains native per-expert counts.
This optional path rounds each validated count to 256 rows and shares that
bound between workspace allocation and the existing contiguous W4A8 apply.

```bash
export VLLM_HCU_STATIC_HT_W4A8_COUNT_SCRATCH=1
export VLLM_HCU_STATIC_HT_W4A8_COUNT_DIAGNOSTIC=0
```

Both flags default to zero. The owner is attached only for explicit eager
execution, HYV4/HYV4-MTP, static EPLB and fixed HT. Dynamic routing placement,
unsupported owners, invalid counts, zero receives, noncanonical physical maps
and unsupported alignment retain the conservative workspace bound.

The native Python count list is preserved before tensor conversion. A plan is
attached to the current receive metadata after checking the owner, rank offset
and canonical physical map. Map identity/version changes invalidate it; a map
change after allocation is rejected before scatter. Versionless inference maps
must belong to the static expert-map manager, whose supported topology update
replaces the map object. In-place mutation of such a frozen map is outside this
contract. Quantization, scatter/GEMM kernels and model weights are unchanged.

For idle diagnostics, set `VLLM_HCU_STATIC_HT_W4A8_COUNT_DIAGNOSTIC=1` to compare
the post-map GPU counts with each bound before scatter. This adds an explicit
GPU synchronization; keep it disabled during performance measurement.

Run the portable contract tests in the matching PyTorch/vLLM environment:

```bash
python -m pytest -q tests/runtime_patch/test_static_ht_scratch.py
```

Native validation includes dispatch/GEMM replay with the flag disabled and
enabled, skew/duplicate destinations, alignment boundaries and zero receives.
It is bounded coverage, not a proof for arbitrary backends or graph execution.
The recorded HTTP results use a deployment combining other frontend,
scheduling, attention, tokenizer, Mooncake and R16 changes; no isolated HTTP
speedup is attributed to this patch. R16 enablement is a separate change; the
scratch optimization also supports the existing zero-redundancy static path.
