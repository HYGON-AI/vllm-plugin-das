# V41 PCP Engram integration — failures fixed before implementation

Parent: 16e15b1. Scope: existing opt-in PCP model, MRV2 eager, DP1/PP1/DCP1, CPU-offloaded private tables. Core/Adapter/old source trees remain read-only; no GPU/model/service/install/remote writes.

## Failure modes

1. Native table `_get_shard_info` includes TP/Engram-DP only. Under PCP8 TP1 a complete table is replicated eight times. Own table must use (TP*PCP, TP_rank*PCP+PCP_rank), retain native bucket boundaries and loader `engram_vocab_start`, and reject layouts leaving any rank with no head.
2. Native gather is DP, not PCP. Local fragment hashes must be padded with DEAD_ID=-1 to a common slot, made contiguous and gathered rank-major before local-head lookup. Gather staged heads over PCP, slice this rank's token slot, then gather TP heads; never restore global token order inside Engram.
3. MRV2 native NgramHashState and local absolute-position/lookback semantics already exist. Keep tokenizer/vocab validation, hash kernel, lookback and local hash rows native. Do not manufacture a new hash kernel or move hashing to global query positions. Native hash num_warps=4 is retained; historical 2-warp issue has not been reproduced here.
4. Engram is constructed before attention; native `_create_embedding` resolves global ParallelEngramEmbedding. Reuse its exact raw inspect.getsource reviewed bytecode in private globals substituting owned table; add owned Engram only to existing private decoder namespace. No global monkeypatch/config mutation/copy of argument list.
5. Native `_init_staging` sizes max_tokens*DP; own staging also needs PCP factor. Prepare records this step's local count/slot only after successful lookup. Embed without prepare, mismatched token extent, repeat consumption or failed prepare must reject rather than use stale rows.
6. PCP0-query ranks participate in count/hash/head collectives whenever another rank has rows. Count gather gives every rank the same maximum slot. If the entire group is empty, all ranks perform count gather then uniformly skip zero-element hash/head gathers and lookup. Padding rows use DEAD_ID and zero values/scales; unequal hash head shards preserve TP-major ordering and truncate only final padded heads.
7. Large pin_memory allocations are historically unreliable on HCU. Use anonymous mmap plus HIP hipHostRegister (mapped flag), hipHostGetDevicePointer and hipHostUnregister. Validate page alignment, pinning and non-null device pointer. Do not guess that host_ptr is device-visible or fall back to pageable memory; do not allocate another private copy. No HIP call during CPU source discovery.
8. Native `get_accelerator_view_from_cpu_tensor` already dispatches the ROCm/HIPified `_C.get_cuda_view_from_cpu_tensor` and obtains the true device pointer. Reuse it; check cached weight/scale storage pointers against original registration to reject replaced storage. Mmap/registration lifetime must follow actual CPU tensor storage and UVA views, not module GC. Registration errors rollback; teardown errors are logged.
9. Nonempty Engram layout may be admitted only with an actual config: cpu_offload=True, dp_shared_memory=False, embedding_across_dp=False, use_thp=False, no SP and Engram-DP1. Shared-DP GPU/host table semantics remain native outside PCP1 and are not enabled by this stage.
10. Existing PCP1 model construction and optional PCP1 OOT scorer must be unchanged. Native parameter shapes/mapping/model-state remain owned by Core. Runtime feature opt-in stays frozen by existing PCP extension.
11. INT8 empty-row is a separate production scheme/loader concern. It must be checked before declaring complete model support; do not hide unknown INT8 scheme behavior under FP8 handling.

## Fixed replay (written before implementation)

- Construct actual native Engram via owned subclass, with projection/device/streams/host registration CPU doubles; assert actual owned embedding, Parameter loader spans, native Engram forward executes. Do not claim a complete model forward.
- 200 independently seeded head sizes/token extents across all eight ranks; TP1 and TP2; empty/all-empty/noncontiguous/padded hash rows. Compare independent logical full-table lookup with production prepare->embed output, byte/dtype/shape equality. Rank collective and GPU lookup are CPU doubles with explicit payload checks.
- Native constructor private-global identity and unchanged Core globals; admission failures, staging state/retry errors and PCP1 native branch.
- Registered host allocation lifecycle/rollback/device pointer/pinning/storage replacement with CPU HIP doubles; no CUDA initialization. Native loader bytes/scales exact.
- Existing PCP attention and plugin CPU regressions. Real HIP registration/UVA, hash/lookup GPU kernels, rank transport and complete model accuracy/performance remain unverified.

## Implemented route and evidence

The existing `VLLM_HCU_DSV41_PCP_ATTENTION=1` model opt-in now admits a nonempty Engram layout only when its resolved config selects private CPU offload, `dp_shared_memory=False`, `embedding_across_dp=False`, `use_thp=False`. Native MRV2/DP1/PP1/DCP1/eager/no-SP constraints remain; PCP width is not whitelisted. PCP1 construction remains native.

| Component | Final owner / behavior |
|---|---|
| Model decoder | Existing private native constructor namespace substitutes owned Engram and attention class names |
| Embedding constructor | Native `_create_embedding` bytecode, raw-source hash `ef2909d6...`, private global table name substitution |
| Hash map/kernel/lookback | Native NgramHashState with local absolute positions and MRV2 lookback; no hash kernel or launch configuration change |
| Table/loader | Native head bucket spans and Parameter loader, with TP-major `(TP*PCP, TP_rank*PCP+PCP_rank)` coordinates |
| Storage | One anonymous mmap for packed weight+scales, direct HIP register/device-pointer validation, Torch pinned recognition; numpy storage finalizer unregisters before mapping teardown |
| Accelerator views | Native Core UVA op, which queries the HIPified device pointer and retains CPU tensor storage |
| Staging | Eager synchronous local-head lookup on rank-major padded hash rows; PCP head gather -> this rank's token rows -> TP head gather |
| Whole model output restore | Native PCPManager, outside Engram |

There is no Core/Adapter/global method replacement. The table is still FP8 e4m3 with raw E8M0 scale bytes; scales/weight use the same native checkpoint bucket slicing. The owned Engram keeps native forward and residual gate. Host storage failure has no pageable-memory fallback. A changed registered weight/scale storage address/shape is rejected. The first version uses synchronous staging, rather than asynchronous lookups, to keep the eager PCP sequence explicit; no speedup is claimed.

`before-fixed/report.json` is the actual Engram pre-implementation red baseline: 0/3. The first launch accidentally imported another executable replay as a helper and ran the old attention replay; it is not Engram evidence. That script import was removed before production implementation. After implementation, fixture config injection was extended into the private constructor namespace, and the existing no-head-shard refusal was aligned with the native AssertionError type. No scenes or success thresholds were added/relaxed.

Final fixed Engram replay: `fixed-fixture/report.json`, 3/3 with CUDA uninitialized. It runs actual native Engram construction/forward, checks the owned nested table and retained native NgramHashState, compares native weight/scale loader bytes, and exercises 200 independent seeds of table head sizes and rank row counts for eight PCP ranks under TP1 and TP2. It includes noncontiguous hash inputs, empty ranks, all-empty groups, one token, padded heads, capacity/state consumption and storage-replacement refusal. HIP registration, projection, lookup/gate GPU kernels and rank transport are explicit doubles; native GPU hash output is not executed or numerically measured.

Host lifecycle evidence covers Parameter retention after the allocating tensor is deleted, unregister only after the last storage owner dies, and rollback after pin/device-pointer failures. These are CPU-double results, not real registered-memory measurements. The existing PCP attention replay passed 7/7 again under the new source.

Independent read-only review ended with no remaining concrete blocker. The initial documentation required zero-element hash/head collectives even for an entirely empty group; that wording was corrected to the uniform count-then-skip rule in failure 6. This was a documentation/contract clarification, with no numerical threshold or code change.

Evidence root: `tests/vllm033/.cache/v41-pcp-engram/`. Repeat with `VLLM_TEST_PYTHON=/usr/local/bin/python3.12 bash tests/vllm033/run_v41_pcp_engram_cpu.sh OUTPUT` in the existing CPU-capable container. Final combined existing CPU regression passed: **312 passed, 66 skipped, 14 warnings in 190.00s**, exit 0 (`regression.log` / `regression.xml`). No new pytest unit tests were added.

## Known limits

Real HCU library discovery/host registration/driver pinning/UVA lifetime, native GPU hash/lookup/gate, eight-process collective membership/order, checkpoint-scale full model loading, Graph, numerical accuracy and performance are not verified here. Native hash still uses four warps; the historical HCU two-warp workaround was not reintroduced without a current reproduction. THP and shared host tables are deliberately not enabled by this private storage adapter. Native head-sharding guards still reject any topology that leaves a rank without a hash head. INT8 empty-row support is separately documented in `v41-int8-empty.md` and applies only when native CT parsing actually selects its supported INT8 scheme.
