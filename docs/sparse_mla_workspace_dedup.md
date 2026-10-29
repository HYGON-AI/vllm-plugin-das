# Sparse MLA prefill workspace deduplication

This plugin implementation targets `fix/hy4-compressed-tensors-w4a8-pp8-mtp3`
(base `b6040197`). It runs on the supported vLLM 0.25.1 base without applying
the earlier experimental deduplication changes to vLLM core.

Rebuild the plugin's native operators in your prepared HCU environment:

```bash
uv pip install --python .venv/bin/python -e . --no-build-isolation
```

Enable before starting workers:

```bash
export VLLM_HCU_SPARSE_MLA_PREFILL_WORKSPACE_DEDUP=1
vllm serve /path/to/model --attention-backend FLASHMLA_SPARSE \
    --kv-cache-dtype fp8_ds_mla
```

The switch defaults to off. It applies to the separate FP8-cache/BF16-prefill
path, with 512 NoPE and 64 RoPE elements and DCP=1. Existing mixed-FP8 batches,
decode-only batches, capture/dummy metadata without host tables, and the legacy
Lightly-CP path use the original implementation. PCP through the HCU Model
Runner V2 is supported. Pipeline stages plan against their own local caches;
there is no cross-rank or cross-stage workspace sharing.

## Implementation

The registered HCU sparse-MLA backend supplies its own metadata builder and
separate-prefill implementation. After the existing PCP phase classification,
the builder groups consecutive prefill rows by the number of unique physical
KV rows they require. Shared pages contribute their maximum valid extent in
that group. A descriptor records the physical page, valid token count, and
compact workspace start. A second table maps each request's logical pages to
those compact starts.

The plugin's HIP gather reads each unique page once per group, dequantizes its
512 FP8 values with four FP32 scales, and copies the 64 BF16 RoPE values. It
uses 256 threads, with eight 32-lane token groups per tile. The existing sparse
MLA call consumes the compact BF16 workspace. The plugin's Triton remapper
performs the page lookup in the same pass that converts selected indices; it
preserves selection order, repeated indices, invalid padding, and decode-cache
addresses. No extra pass over top-k selections is added.

Both HCU runners supply host block tables. The original runner reads its
existing CPU storage. Model Runner V2 installs an optional mirror of canonical
block-table append/overwrite operations before request allocation. It expands
allocation pages into kernel pages and snapshots the current batch mapping,
including repeated PCP virtual rows. Snapshots are attached to the device table
objects and captured by the HCU CommonAttentionMetadata adapter. Replacement,
unpadding, and microbatch slicing preserve the matching row range; replacing a
device table drops an obsolete host map. Replicated and local MTP table-refresh
paths also attach snapshots. Decode-only steps do not create snapshots.

The planner runs once per metadata build, and all layers using that metadata
reuse the plan. Each layer gathers its own KV values. Host planning does not
copy GPU block tables back to the CPU. The builder retains the core metadata
for decode and fallback and adds the plugin-owned compact prefill plan.

The allocation budget is unchanged. Without the separate workspace-capacity
backport, it is five times max_model_len. When that backport is installed, the
planner honors the resolved sparse_mla_prefill_workspace_factor field. Fewer
used rows reduce staging work and may reduce group count; they do not release
the reserved BF16 workspace. See [workspace sizing](sparse_mla_workspace.md).

## Validation

CPU planner, metadata, and host-mirror tests:

```bash
.venv/bin/python -m pytest \
    tests/runtime_patch/test_sparse_mla_workspace_planner.py \
    tests/runtime_patch/test_sparse_mla_workspace_metadata.py -q
```

GPU gather, remapping, graph replay, and real HCU-backend equivalence:

```bash
.venv/bin/python -m pytest tests/accuracy/test_sparse_mla_workspace_dedup.py -v
```

Benchmark a shared prefix and then private prefixes:

```bash
.venv/bin/python benchmarks/bench_sparse_mla_workspace_dedup.py
.venv/bin/python benchmarks/bench_sparse_mla_workspace_dedup.py \
    --shared-prefix-len 0
```

Use `--plan-only` for host measurements without a GPU. A cache-hit percentage
alone does not establish shared physical pages. The benchmark's `--prefix-len`
controls already-computed tokens, while `--shared-prefix-len` independently
controls physical sharing.

On October 29, 2026, in pod2c2's `xiabo_vllm_hy4_1024` container, device 15
(BW1301_LC/gfx938, PyTorch 2.11.0, HIP 6.3.26113):

- The plugin HIP kernel compiled in an isolated test library using the production
  operator name. The installed plugin and vLLM packages were not overwritten.
- All 51 new planner, metadata, kernel, and backend tests passed across runs.
  This includes exact dequantization against PyTorch, partial pages, strided
  buffers, buffer guards, graph replay, and real backend output comparisons
  for pure prefill and mixed decode/prefill against the installed core path.
- Related existing attention and PCP tests passed, including the added runner
  host-map assertions and metadata replacement/unpadding/slicing checks.
- A full plugin-wheel build, full HY4 serving run, and distributed PCP/PP/MTP
  execution with dedup enabled were not tested.

For ten 85,000-token requests with a physically shared 79,040-token prefix,
PCP16, 128 query heads, top-k 2,048, and 450,560 workspace rows:

| Metric | Installed vLLM gather / original mapping | Plugin dedup |
| --- | ---: | ---: |
| Local query tokens / virtual rows | 3,500 / 20 | 3,500 / 20 |
| Staged KV rows | 1,642,270 | 138,640 |
| Gather/MLA groups per layer | 4 | 1 |
| Index conversion | 215.1 us | 216.0 us |
| Total gather time per layer | 2.920 ms | 0.185 ms |
| Index + gather + MLA | 38.597 ms | 34.344 ms |

The baseline calls the installed vLLM operation
`cp_gather_and_upconvert_fp8_kv_cache`. Host planning took 1.115 ms per metadata
build and is reused across layers. GPU timings exclude host planning, host
snapshots, and metadata transfer. These are synthetic pipeline measurements on
a shared host whose existing services held about 85% of device memory; they do
not establish an end-to-end model speedup.

With private prefixes at the same PCP16 workload, the two segments of each
request still overlap: 4 → 2 groups, 1,642,270 → 850,000 staged rows,
2.920 → 1.082 ms gathering, and 38.570 → 35.995 ms for the pipeline. Host
planning took 1.409 ms. Both benchmark cases passed attention-output comparison.
