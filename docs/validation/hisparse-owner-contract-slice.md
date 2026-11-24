# HiSparse owner hot-cache contract slice — NOT joint support

Date: 2026-11-22. Status: **local, unwired contract utility + CPU tests**.
No production runtime/binding calls this provider. Both admission guards remain
unchanged. This is not HiSparse+DFly enablement, clean-installed qualification,
model delivery or a replacement for the native HiSparse provider.

## Evidence and scope

Read the existing `hisparse-dfly-dflash-20261122T062442Z/report.zh-CN.md`
and its fixed primary-source snapshots before implementation:

- Core033 `4ba787ec9959e566d3b587a796dff07673324da5`, runtime lines
  672–807 and 881–889, binding lines 181–298: int16 LRU, no owner keyword,
  no reader discovery/owner-aware hot binding.
- Official #69057 head `2f459ea8c1355bcb8428dc7544d87700c2cb3fde`, runtime
  lines 723–764: capacity check, owner stride and `lru_init`, shared residency,
  LRU slots and device stats. Its int32 LRU is **not** backported here.
- Upstream dependency `e72fc9ae50c221180c4b8f4a94c5009714a4b199`, binding
  lines 263–423: reader discovery and resident/hot/host/source-slot binding.
  That complete binding integration is deliberately NOT implemented here.
- #69057 DFly speculator lines 405–461: real BlockTables slot mappings,
  unlike the candidate plugin's private DFly arrays. No slot code is changed.
- Adapter #159 HiSparse native provider: explicit/lazy, process-local,
  fail-closed resolution and original Core033 schemas. No native registration,
  schema, site-package or existing provider is modified by this slice.

The model PR history skill was read and queried for `HiSparse DFly` and owner
hot-cache history. No direct history entry was found. The inspected Qwen3 Core
history lists `qwen3_dflash.py` with no direct PR-number commit; it is not
HiSparse/DFly owner evidence. Do not equate DFlash with DFly.

## Contract implemented, but not wired

`get_owner_provider()` lazily imports the real Core033/Torch and returns one
caller-owned provider. Module import itself imports neither. There is no global
patch, model registration, entrypoint, automatic seam installation or fallback.
Keep that provider alive for the entire lifetime of all its bound runtimes.

- `bind_owner`: bind the original Core033 hot view, invalidate old replacement
  state on a new binding. Identical repeated binding is a no-op, preserving hits.
- `bind_reader`: require a directly bound owner and sufficient owner capacity;
  share `region_stride`, `lru_init`, `device_global_indices`, `lru_slots`,
  `swap_stats`. Leave reader groups, top-k scratch and stream objects private.
- `reset`: route either owner or reader to the same owner's in-place state reset.
  CPU fixtures perform the three Core033 tensor operations explicitly because
  Core033 reset itself unconditionally obtains a GPU stream. Device reset would
  delegate to the original Core method; **that device path was not exercised by the CPU suite** (see the later HIP result below).
- `unbind_reader`: remove hot views and restore the reader's original private
  stride/state. An owner cannot detach with live readers.
- `rebind_owner`: fixed-layout replacement of owner AND all reader hot views;
  invalidate old hits. Roll back hot views if binding fails before reset. Reset
  failures latch the provider: fresh runtimes/process required, not silent retry.
- `unbind_owner`: detach only after every reader has detached. Reusing the
  runtime for a fresh owner binding invalidates old residency.

Every lifecycle call requires `quiescent=True`: the CALLER must stop/drain all
participants first. The lock only serializes lifecycle calls; it does not drain
GPU work or protect concurrent attention. Fork inheritance, staged swaps,
followers, owner chains, another provider taking over a live runtime, external
view/state replacement, invalid int16 ABI, overlapping independent owner ranges,
non-base raw views and logical byte overrun all fail closed. This slice accepts
only singleton leader groups and block-aligned, fixed layouts; resize is rejected.
Conservative hot-range overlap checks may reject interleaved disjoint layers in
one slab: general packed/interleaved layouts are not qualified here.

## Actual CPU results and limitations

The CPU tests ran in an internal v0.33 validation container with Torch 2.13.0.
Connection and container identifiers are retained in external validation evidence.
**45 passed / 0 failed / 0 skipped** in the final remote run.

The suite imports installed Core033 and calls its original bind/PagedCacheView
methods with real CPU tensors. It constructs the actual runtime/group types via
`__new__` to populate only the fields under test: it does NOT run the GPU/native
constructor, host pinning, model allocation or a complete Core binding flow.
Four guard cases execute complete original admission/constructor AST functions
with configuration namespaces, without constructing/importing a model.

Covered: 2176/4096 positive, 2176/2048 rejection, equality/one-row overflow,
int16 representable boundary, nonzero offsets/padded strides, two requests,
multiple readers, independent owner isolation, reset/template identity,
idempotence, detach/reuse/owner change, fixed-layout rebind, bind failure rollback,
reset failure latch, parallel serialized reader binding and both existing guards.
The 32768 boundary is a CPU dtype contract, NOT native shared-memory admission.

Final harness counters: 0 accelerator initialization attempts, 0 non-CPU tensor
operations, 0 native HiSparse operations; `cuda_initialized=false`. Four installed
Core source hashes and Core bind/reset function identities were unchanged.
First run was 29/30: a truncated tensor retained full underlying storage, so Core
`as_strided` accepted it. The sidecar now checks logical byte extent before bind;
the regression passes. No Core source was edited to fix that boundary.

Reproduce from the work root using the bundled Python:
`verification-reports/hisparse-owner-sidecar-20261122/run_remote_cpu.py`.
The launcher sends a tar on SSH stdin to a fresh remote `/tmp` staging directory,
not to site-packages. It does not start/stop/reconfigure any service or use GPUs.
Raw logs, source/input hashes and per-case JSON are retained in that directory
under the local work root. The temporary remote staging directories are retained
for inspection; they are not an installed plugin overlay.

## Still missing before actual joint adaptation

1. Freeze/review against the updated main MR baseline; this local checkout stays
   at `24175a34`, not a checkout/rebase to the newly pushed main-thread fixes.
2. A real lifecycle seam that calls the utility from allocation/binding/reset/
   teardown/rebind; stable owner relationships surviving DFly alias removal;
   reader discovery and non-singleton group/layout qualification.
3. Actual-Core resident + hot + host/source binding, host registration lifetime,
   request-state indices, source block tables and mirror slot mappings for readers.
4. DFly query KV writes, HiSparse mirror/write_target and attention metadata all
   using the SAME actual BlockTables slots, respecting Core033
   `group.kv_cache_spec.uses_slot_mapping` and slot-mapping-enabled contracts.
5. Active model/artifact delivery, exact native-provider qualification and an
   installed-package CPU joint binding test (not explicit-path test loading).
6. Separately authorized device cache-to-attention replay, stream/event teardown,
   invalidation/recopy and output correctness. No GPU or busy service was touched.

Only after those gates can narrow admission be considered in a separate review.
The main thread's reported 73/83/83 tests are separate and are NOT added to this
slice's 45 cases. No existing source was changed, no commit/push was performed.


## Subsequent real HIP lifecycle result (2026-11-22)

The opt-in `tests/vllm033/test_hisparse_owner_gpu_lifecycle.py` completed
**12 passed / 0 failed / 0 skipped** on physical GPU2 (PCI 0000:5e:00.0).
It used real HIP tensors, streams, events, original Core bind and GPU reset,
with seven original reset calls and fourteen original wait_stream edges.
Peak tensor allocation was 915456 bytes; allocator reservation was 32 MiB,
not including HIP context overhead. No Core method was replaced.

Runtime/group objects still use actual Core types with `__new__` plus the
fields under test, not native constructors. Reader-private copy streams are
explicitly joined by the caller before reset; the utility does not auto-drain
arbitrary private streams. Singleton leaders only: nonempty followers remain
unsupported. This verifies the hot-state lifecycle slice, NOT native runtime
allocation, host/resident/source binding, slots, sparse attention or HiSparse
plus DFly model execution. Both joint-admission guards remain unchanged.

The module SHA256 tested on CPU and HIP is
`0370f01c73c0c4c3e0cd9b8ebe5801fc7cc8be1ff9d622419fed00a2c605df6c`.
The GPU test SHA256 is
`7d3bedb023bf539c23370fbe21558496405076b472256f6face92f9f6264db28`.
These are exact source snapshots, not a full latest-head model qualification.
