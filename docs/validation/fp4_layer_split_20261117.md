# FP4 Indexer + Layer Split integration, 2026-11-17

## Scope and ownership

This follow-up composes the migrated main/SWA Layer Split feature with the
official FP4 Indexer implementation. Its bases are Adapter main `4b9ea6d`
and Plugin v0.33.0 `89ee352`. The pre-integration feature heads are retained
under `codex/layer-split-before-fp4-20261117` (Adapter `3d2ae32`, Plugin `8929c37`).
Both active branches are `codex/layer-split-official-20261117`.

The Adapter exposes one shared PCP transfer-admission function. FP4 and FP8
call the same registered owner validator. Missing validators, non-None results,
foreign connector modules and unsupported roles remain rejected.

The Plugin owns the composition:

- `LayerSplitIndexerMixin` filters writes by source owner and broadcasts after
  the native indexer forward, including its short-context/zero-query return.
  FP4 has a distinct `_produce_k`; intercepting only the FP8 producer was
  insufficient. `Fp4LayerSplitIndexer` now wraps the actual FP4 producer.
- `LayerSplitAttentionMixin` retains main/SWA broadcasts, non-owner SWA write
  masking and the private forward context used by bounded decoder replay.
  The native FP4 constructors, quantization kernels and reader are unchanged.
- The model selector selects `Fp4LayerSplitPcpAttention` only when both features
  are enabled. With Layer Split disabled it still selects `Fp4PcpAttention`.
- Existing independent main/SWA/compressor pools and scheduler remain the
  allocation owners. FP4 specs come from the real FP4 cache constructor;
  physical pages have 68-byte records, including payload and scale.
- Mooncake owner protocol v3 already describes physical cache bytes and shapes.
  It carries the new FP4 pages without a protocol change and rejects a peer
  exposing an incompatible FP8 schema.

Official Core is not modified. No provider packages are replaced or rebuilt.

## Configuration boundary

Keep the existing model configuration and select the combination explicitly:

```bash
export VLLM_HCU_DSV41_LAYER_SPLIT=1
export VLLM_HCU_DSV41_PCP_ATTENTION=1
export HCU_VLLM_PCP_IMPL=native
export HCU_VLLM_FP4_INDEXER_IMPL=lightop
export HCU_VLLM_FP4_PACKER_IMPL=triton
# Set attention_config.indexer_kv_dtype=mxfp4 in the serving configuration.
# Add --swa-bounded-replay only for the replay-enabled test arm.
```

The FP4 combination remains eager TP1/DP1/PP1/DCP1/PCP8 with EP and
`deepep_high_throughput`; no dSpark, EPLB, DBO or LoRA. These are retained
admission limits, not newly qualified device combinations. Official
`deep_gemm` admits SWA bounded replay on this baseline; MegaMoE does not.
Layer Split, FP4 and Replay remain separate switches.

For PD, the P endpoint must use the exact
`HcuLayerSplitMooncakeConnector` / `hcu_vllm_das.mooncake_layer_split`
producer contract. Both peers must expose compatible cache specs, including
FP4 format. An ordinary Mooncake connector is not silently accepted.

## CPU evidence and remaining device work

The Shenzhen validation container uses the installed Core wheel and Python 3.12,
with GPU visibility empty,
`HCU_ADAPTER_HOST=none` and `VLLM_PLUGINS=__disabled__`.

`tests/vllm033/test_fp4_layer_split.py` checks native selectors/constructors,
CR1/CR2 and eight source-owner ranks, native allocation views, scheduler-backed
capacity, PD descriptors across different group counts, and peer-schema
refusal. Its Gloo test executes real two-process broadcasts of noncontiguous
live FP4 pages, including scale bytes, with an empty-query owner, repeated
scratch reuse and protected SWA/compressor bytes. Device leaf kernels in the
constructor tests are explicit recorders; Gloo byte transport is real CPU work.

`test_swa_cache_layer_split.py` runs the native replay preparation and
full-batch/trimmed-write regressions against both FP8 and FP4 attention classes.
`test_pcp_transfer_contract.py` verifies the common owner callback contract.
The migration CPU driver selects these new cases as well as the existing
FP8/pool/scheduler/PD/replay regression suite.

Detailed results and exact final commits are recorded in the workspace's
`validation/fp4_layer_summary.json`:

- `validation/fp4_layer_cpu`: 288 pytest cases passed and 13/13 PCP source
  replays passed; CUDA not initialized.
- `validation/fp4_layer_upstream.xml`: 104 existing FP4 constructor, workspace,
  admission and FP8-isolation cases passed.
- Adapter L0 structural gate passed. No tests were skipped in these two suites.

The first run's failed test fixture is
retained in `validation/fp4_layer_01.log`: it supplied a bare object instead
of a valid dSpark config. Correcting its required fields retains the expected
speculation rejection. `fp4_layer_02.log` records 37 passing focused cases.

No GPU service, GPU quantization comparison, actual RDMA transfer, model
accuracy or capacity/performance claim is established by these CPU tests.
Next device validation must compare the same FP4 deployment with Layer Split
off/on, then Replay off/on, recording actual memory budgets, token capacity,
output tokens/logprobs, owner logs and clean shutdown. PD accuracy follows
standalone correctness. Do not reuse the historical FP4-only or legacy
Layer Split model scores as results for this revision.

The separate prior upstream regression failure caused by the container's
missing `deepgemm.mega` remains unresolved; this integration neither removes
that test nor claims MegaMoE support. AI-assisted implementation and review.
