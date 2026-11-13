# V41 DP target private Engram host storage

Parent 5262be5. Core 368701db06 and Adapter 3257a934 read-only. No packages, GPU, service, container creation or remote Git writes.

## Failures fixed before implementation

1. Existing HCU registered allocation is selected only by the PCP model path. PCP1 DP targets still construct native ParallelEngramEmbedding, whose ROCm private allocator calls cudart.cudaHostRegister. Must select owned storage before native Engram allocation; post-load is too late.
2. Native table sharding and DP hash/head gather already exist. Preserve TP-major TP×Engram-DP coordinates and native `engram_vocab_start` loader ranges. Never add PCP into a PCP1 DP table or assume node-local Engram-DP width equals global DP width.
3. Native `engram_gathered_num_tokens` requires actual forward DP metadata and slices the contiguous local Engram-DP group within global DP. Native model supplies dummy hashes for empty/unbound-cache DP workers. Keep this real path, do not inject fake target counts or bypass missing metadata.
4. Native `_gather_engram_rows` gathers local-head data and selects this replica's token slot, then native embed completes TP head order. Keep tensor row/head order; GPU selection/lookup/hash kernels are not CPU-proven merely by preserving methods.
5. A new default-off host flag must share the existing public target model registration with PCP, with no competing registry owner. Track both flags in frozen policy and compile identity. Host-only activation must not implicitly enable PCP attention. Flag-off/PCP1-no-host remains fully native; owned draft must remain Engram-free.
6. Reuse reviewed native VL/backbone/decoder constructors in separate private namespaces, substituting only target Engram; preserve native attention selection for PCP1. Never change Core module globals or mutate model_config/Engram config to disable layers/sharing.
7. First supported storage is private CPU offload with `dp_shared_memory=False`, `use_thp=False`; support both native embedding_across_dp settings by retaining actual native group semantics. Shared mmap host tables, device tables and THP remain explicit refusals, not silent private copies. Require eager MRV2 and PCP1/DCP1/PP1, no DBO/EPLB/ElasticEP/LoRA.
8. Registration must reuse exactly the storage lifetime/pinning/device-pointer checks proven by PCP CPU replay. Extract allocation/view identity into shared helpers rather than duplicate the HIP ownership path or inherit PCP sharding by accident. Native `_storage` obtains HIPified UVA views; it must not fall back to pageable copies after storage replacement.
9. Preserve native hash/tokenizer/vocabulary/lookback and forward/loader, including dummy hash participation and canonical loaded names. No EngramBatch/peer path is enabled on AMD: it is absent from AMD target construction.
10. The public target entry must resolve and actually build owned native decoder→Engram→table objects before a native gate forward. CPU replay uses explicit projection/attention/hash/device/kernel doubles, not full-model numerics.

## Fixed source replay (before production)

- Public activation for host-only vs PCP-only/both/off; lazy registration, frozen drift, idempotence, native owner checks and unchanged Core globals.
- Real owned target/decoder/Engram constructor chain using reviewed native bytecode, with heavy device/model dependencies explicitly doubled; owned nested table identity and native forward executed.
- 200 independently seeded TP×EDP layouts, variable/empty hash rows and local-node DP slices; compare exact native parameter loader spans and retained native gather/head restoration with a logical full-table reference. Group/lookup/select kernels and HIP registration are CPU doubles. Missing DP metadata rejects.
- Shared storage/settings and wrong runtime configuration reject before allocator; PCP1 flag-off stays native, PCP allocation/gather replay remains 3/3 and attention replay 7/7.
- Whole existing CPU regression; no unit-test scenarios added after implementation. Real HCU UVA/host registration, interprocess DP transport, native GPU hash/lookup/gate, full checkpoint loading, Graph and whole-model accuracy/performance remain unverified.

## Implemented route and configuration

Set `VLLM_HCU_DSV41_ENGRAM_HOST=1` to select the owned PCP1 target host-storage route. Default is off. It shares the existing `das_hcu_v41_pcp` extension and public `DeepseekV41ForCausalLM` owner; there is no second competing model registration. `VLLM_HCU_DSV41_PCP_ATTENTION` remains independent: host-only does not enable PCP attention. Both flags are frozen and included as a two-boolean compile policy; flag drift or registry loss is fatal.

Requires a resolved Engram config selecting `cpu_offload=True`, `dp_shared_memory=False`, `use_thp=False`, eager MRV2 and PCP1/DCP1/PP1, no LoRA/EPLB/ElasticEP/DBO. Native `embedding_across_dp` settings are retained, not overwritten. Native group initialization for CPU offload derives a node-local Engram-DP width; this width can be smaller than global DP, regardless of that flag. The table still obtains its actual shard width from `get_engram_dp_size()`.

| Stage | Owner / final behavior |
|---|---|
| Public target selection | One lazy owner for PCP or private host opt-in; flag-off native |
| PCP1 host constructor | Reviewed native VL/backbone/decoder bytecode in separate private namespaces; only Engram class substituted |
| Attention / mHC / MoE | Original native target construction and forward; no host-route attention override |
| Table coordinates | Native TP-major TP×Engram-DP; no PCP coordinates |
| Allocation | Existing mmap/HIP register/get-device-pointer/pinning contract, shared helpers for both PCP and target |
| Weights / scales | Native FP8/E8M0 Parameters, bucket span loader and canonical names |
| Hash / lookup / gather | Native NgramHashState, local hashes, native padded DP gather and local-token/head restoration |
| Empty or unbound-cache DP worker | Native AMD dummy_hashes path and native collectives retained |
| Streams | Synchronous eager staging; no asynchronous host prefetch stream |

The small storage slicing/address-check code was extracted verbatim into `engram_host.allocate_table` and `check_table_storage`, keeping HIP registration/finalizer unchanged. PCP invokes the same helpers and keeps its original shard/gather semantics. DP target uses a different embedding subclass inheriting the original native sharding, rather than inheriting PCP behavior. Core private/sharing configuration is never changed or disabled on the user's behalf. Shared host tables and THP requests are refused before allocation.

AMD target does not construct EngramBatch/peer exchange; the owned target keeps its per-layer native prepare/embed path. DSpark draft has no Engram and is unchanged. A successful eager DP host constructor is not a Graph or whole-model enablement claim.

## CPU evidence

Evidence root: ignored `tests/vllm033/.cache/v41-dp-engram-host/`. Runner: `VLLM_TEST_PYTHON=/usr/local/bin/python3.12 bash tests/vllm033/run_v41_dp_engram_host_cpu.sh OUTPUT` in the existing bw47 CPU-capable container. No package installation or GPU visibility.

- `before/report.json`: **0/2**, exit 1, production module absent; failure modes and replay preceded implementation.
- First candidate: **1/2**. Native decoder construction required the fixture's missing `hc_eps`; only that config field was added, no new case or success threshold.
- `final/report.json`: **2/2**, exit 0, CUDA uninitialized. Public host-only lazy registration/idempotence/frozen drift executes. Real native decoder bytecode constructs actual owned Engram/table objects; attention, FFN, mHC, projections and outer VL allocation boundaries are explicit doubles. Native Engram forward is executed with its gate kernel doubled.
- The second case uses **200 independently seeded** head sizes and variable/empty local row counts, TP1/TP2, Engram-DP2/4/8 within a global DP group twice as large. It compares native weight/scale loader spans byte-for-byte, native `gather_engram_hashes` local-node metadata slicing and native head/token orchestration to an independent full-table reference. GPU lookup/select and rank transport are doubles; actual GPU hash output is not measured. All-empty, one/zero local rows and missing metadata are included.
- Existing PCP Engram replay **3/3** and PCP attention replay **7/7** passed after the shared-helper/activation changes, CUDA uninitialized. These retain earlier explicit kernel/provider doubles.
- Independent read-only final review found no concrete blocker in storage lifetime/rollback, native local-node DP semantics, dummy participation, private constructor chain or independent frozen flags. Core raw-source hashes are checked without dedenting.

Existing full CPU regression passed **312 passed, 66 skipped, 14 warnings in 167.47s**, exit 0 (`regression.log` / `regression.xml`). Changed production/replay files passed AST parsing, runner passed bash syntax and git diff --check passed. Final source hashes are in `source-manifest.json`. Accuracy and performance: not measured.

## Known limits

Real HIP library discovery/host registration/pinning/UVA, native GPU hash/lookup/gate, multi-process DP collectives and HCU kernel/provider ABI remain unverified. The full target/DSpark checkpoint was not loaded and a complete model was not run. Empty-worker fixtures cover forward-level hash/gather behavior only. Native MRV2 `gpu/model_runner.py:1792-1797` can return no-forward for zero scheduled local tokens before the later DP sync/all-zero branch (`:1832-1853`); actual engine dummy/coordination and collective participation require runtime verification and are not established by these CPU fixtures. No scheduler or DP metadata wrapper was added. Graph remains explicitly rejected, as do shared-memory host tables and THP. Native hash launch policy is unchanged; the historical two-warp workaround still needs a current reproduction before migration. Native head capacity, tokenizer/compressed-vocabulary and UVA capability checks remain. Meta materialization/storage replacement bypassing the registered allocator is not added as a supported reload path.
