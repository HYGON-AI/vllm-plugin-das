# HY4 Flash HiSparse on HCU: experimental integration

**Status: Draft; plugin model/installation/Graph not qualified.** This is a separate
HiSparse workstream, stacked on the Flash and Channel W4A8 support changes.
Existing DFly inference without HiSparse remains unchanged.

**Delivery caveat (2026-11-22):** this branch preserves the latest #493
Adapter-extension baseline. Its active wheel currently delivers Flash config
metadata, not the legacy `vllm_hcu` model implementations or their registrations.
The integration below is therefore a source candidate until the model is
ported to `hcu_vllm_das` through the Adapter lifecycle. Neither adding this
checkout to `PYTHONPATH` nor restoring the retired root build closes that gate.
See `docs/hy4_flash.md` for the separate model-delivery blocker.

## Scope and upstream evidence

HiSparse is host-resident sparse-MLA KV plus a GPU hot cache. Selecting
`FLASHMLA_SPARSE` alone does not enable it. The explicit entry point is
`HiSparseConnector` with Core-resolved `attention.hisparse_config`.

Upstream vLLM PR #69026 (`0f92ded` head, merged 2026-11-21) excludes sliding
window attention from HiSparse cache ownership and admits HiSparseConnector.
This plugin implements the same ownership rule with a **model-scoped window
index-group builder**, instead of rewriting Core or globally patching its
builder. Indexed layers continue to use Core's normal HiSparse registration;
window layers continue to use the window manager.

Upstream #69057 (`2f459ea` head inspected, still Draft on 2026-11-21) additionally
packs valid attention indices and changes DFly slot mappings and shared-hot
cache ownership/stride. It is **not** a configuration-only change. The pinned
v0.33 runtime lacks that owner-binding ABI. This first integration deliberately
rejects HiSparse with speculative decoding, including DFly, rather than
aliasing host tensors with independent/stale GPU residency metadata. Its
shared-hot-cache backport and real DFly5 acceptance remain follow-up gates.

## Admission and companion adapter

- Ordinary inference does not import/load the HiSparse adapter extension.
- Only direct `HiSparseConnector` is admitted. NIXL, Mooncake, MultiConnector,
  and other P/D connectors are not enabled by this change.
- Core must resolve HiSparseConfig; an orphan config/connector fails early.
- Initial lane is target-only and `enforce_eager`; Graph is not qualified.
- HCU explicitly calls `hcu_adapter.kernels.hisparse.ensure_hisparse_ops()` from
  the companion adapter [Draft #159](https://github.com/HYGON-AI/adapter-das/pull/159).
  Older adapters fail with a dependency error.
- All four `_C_cache_ops` names **and CUDA/HIP dispatch kernels** must exist;
  schema-only registration and CPU fallback are not accepted as support.
- CUDA uses its Core-native operators and does not load the HCU extension.
- Existing Flash TP/EP, parallelism, BF16-compute, LoRA, quantization and KV
  dtype checks remain in force. This change does not certify every combination.

The current test container's installed vLLM 0.33 ROCm extension had none of the
four native HiSparse operators when probed on 2026-11-21. This is why simply
relaxing the Flash runtime guards would not make HiSparse run.

## Compressed IndexPool block units

The pinned Core resolves HiSparse before generic cache-group planning. Its
sparse and indexer MLA specifications must already have the same scheduler
block size at that point. Flash's pool page holds 64 **compressed states**;
with compression ratio 4 this is 256 original tokens, whereas ordinary sparse
MLA defaults to 64 tokens. The model-scoped `FlashMLAAttention` specification
uses `lcm(requested_block_size, ratio * 64)` only for HiSparse indexed layers,
matching IndexPool's specification before Core resolution. It does not mutate
the shared CacheConfig or change SWA/non-HiSparse specs. Core then selects the
common kernel size; IndexPool still binds exactly 64-state physical pages.
This source-contract correction is not installed-runtime qualification.

## Validation and release gates

```sh
python -m pytest --noconftest tests/models/hy_v4/test_flash_hisparse.py
```

The tests execute production admission/window-registration code. Native
extension/dispatch checks use explicit doubles. Optional installed-Core AST
checks execute its real registration/dataclass code with real Torch CPU
tensors; streams/events are inert. These are not GPU/native operator tests.

### Recorded source and isolated native evidence (2026-11-22)

- Immutable Linux plugin source snapshot at tree
  `91c10ceeb75d6f2bf191383130e4fb944ae2dc12`: **282 passed /
  15 accelerator-only skipped**, with14 Torch deprecation warnings. This is
  source/CPU coverage, not installed plugin support.
- Windows source/package/runtime/HiSparse lane: **106 passed /2 optional
  Torch+installed-Core skipped**. A separate45-case probe imported the actual
  pinned-Core specification methods/dataclasses and common-block selector with
  lightweight heavyweight-constructor shells; it does not allocate or bind KV.
- Companion adapter #159: fresh isolated gfx938 HIP build from exact source
  `371360626b16`,922 tracked source files verified; fresh binary SHA256
  `db27ecdb0b1dac6713590299b2aaa99191081b710625d85d234832b66002f48f`.
  **44 real HCU eager native tests passed /0 failed /0 skipped** (21.04s).
  Covers resolve/LRU/state/stats, invalidation/recopy, holes, multi-request,
  resident/mixed/padding, pinned interior offsets, physical pitch/guards,
  block64/128/256 and plan/compact copies. Eight added cases verify BF16 and
  opaque FP8-DS byte-copy layout only, not attention/quantization numerics.
- These native results do not establish this plugin's active model delivery,
  cache-to-attention replay, clean installation, full-model accuracy, Graph,
  DFly or performance. No running service was hot-patched or package installed.

Before Ready/merge: paired clean Core/plugin/adapter installation and model
registration/profile/serve; full cache-to-attention backend replay including
sinks, state ownership and stream/host lifetime; independent FP8 and Channel
W4A8 model accuracy. DFly requires its owner/stride and hole-packing backport
first, then `num_speculative_tokens=5`, `FLASHMLA_SPARSE` and complete
HumanEval164 with `max_model_len=8192` (prompt + thinking + final). Graph needs
changing-input replay qualification separately. Measure host DMA volume, GPU
memory, TTFT and TPOT before advertising a performance benefit at length8192.
The normal non-HiSparse DFly main lane is a separate acceptance target.

Do not replace an active evaluation's source or installed packages. CPU/source
checks and isolated builds may run concurrently; device tests must wait until
the existing accuracy job exits naturally.
