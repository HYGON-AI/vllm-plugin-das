# Sparse indexer: empty prefill chunk companion fix

The vLLM v0.25.1 sparse indexer helper can set `max_q=0` for a chunk
whose sequence and query lengths are both zero. Its subsequent
`range(0, chunk_m, max_q)` raises
`ValueError: range() arg 3 must not be zero` before returning metadata.
The HCU adapter's post-return empty-slice filter cannot intercept this error.

This was observed during a HYV4 W4A8 PCP16 uniform routing benchmark with
146 distinct 79,050-token prefixes, 5,950-token suffixes and shuffled
requests. The initial 146-request round stalled after P reported one success;
D reported zero successes. The 2920-request round did not start, so this observation
does not provide a valid TTFT or throughput result.

`patches/vllm-0.25.1-indexer-zero-chunk.patch` makes the zero-N fallback
step positive. A zero-query chunk then emits no query slices; positive-query
chunk boundaries and request offsets retain their existing behavior.
It changes no model weights, routing policy, attention kernels or logits budget.
It does not change the existing workspace behavior for a nonempty request
that exceeds the workspace or logits budget.

## Apply to the matching core

Installing this plugin **does not automatically apply this core patch**.
Use a separate vLLM core checkout or deployment copy, preserve the original,
then inspect and apply the patch:

```bash
git -C "$VLLM_SOURCE" apply --check \
  "$PLUGIN_SOURCE/patches/vllm-0.25.1-indexer-zero-chunk.patch"
git -C "$VLLM_SOURCE" apply \
  "$PLUGIN_SOURCE/patches/vllm-0.25.1-indexer-zero-chunk.patch"
```

Build/install that core using the deployment's existing procedure. Check the
worker's actual import path rather than relying only on package metadata.
Other core revisions require a compatibility review; the patch should not be
force-applied when its context no longer matches.

## CPU regression checks

```bash
VLLM_SOURCE=/path/to/matching/vllm \
  python -m pytest -q tests/runtime_patch/test_indexer_zero_chunk_companion.py
```

The tests copy the actual core file to a temporary directory and verify patch
application using git. An already-patched input is reversed only in that
temporary copy. They execute the original/patched helper AST without importing
vLLM, Torch or GPU kernels. Cases cover empty metadata, zero-only rows,
over-budget positive prefill followed by zero padding, request offsets and
2,000 randomized metadata inputs. Positive query coverage is preserved, and
every case where the original returns normally must retain identical slices.

The source inspected for the observed run was core `752a3a5` plus existing
deployment adaptations; original `indexer.py` SHA256 was
`58b8328e3f2e09ad4e3e98decb5181729806024777d04ea05db6ae44bada9814`.
With only this guard changed, it was
`eafef2f2510b010f3d852fa55383a23f72adce1909fd5ce3026763fdd3049674`.
These are validation provenance, not hashes required of every compatible core.
The live failing length arrays were not captured; synthetic CPU reproductions
must not be presented as the exact live inputs.

CPU checks do not establish HCU integration, output accuracy or performance.
The matching P/D 146→2920 rerun remains pending node availability. No performance
improvement is claimed for this patch.
