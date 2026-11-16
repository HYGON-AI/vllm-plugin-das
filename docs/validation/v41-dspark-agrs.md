# V41 DSpark CT W4A8 AG-RS migration

Parent 7329d6f. Scope: owned CT W4A8 DSpark draft with TP1/PP1/PCP1/DCP1 and DP+EP, native MRV2 eager. Target HT/LL unchanged. Core/Adapter/old trees read-only, no installation/GPU/services/remote Git writes.

## Failure modes fixed before production implementation

1. Native dspark/utils.py36-69 retains target DP/EP/backend; native draft can therefore inherit the target's LL manager. Merely changing a backend string leaves global EP dispatch bound to the wrong transport. Draft must own its prepare/finalize and a native AG-RS manager bound to existing DP/EP groups, without changing global `_EP` or target device_communicator.
2. Existing native DSpark construction, forward, expert-count remap, weights loop and target embed/lm-head aliasing are retained. Scope only the owned CT W4A8 model class; non-CT/DP1 drafts keep their old construction. No config mutation of the target/speculator or copied constructor argument lists.
3. Shallow copy the construction VllmConfig and ParallelConfig, changing only all2all_backend. Keep scheduler, model, speculative config and compiler live tables identical. Native set_current_vllm_config context must restore after success/failure. Mark the private config as owned so arbitrary external AG-RS requests do not gain draft admission.
4. Native max_num_tokens comes from scheduler budget for speculator InputBuffers, hidden states, model top-k and MoE. For n draft tokens, query rows/request are n when sample_from_anchor=True else n+1; target verification needs n+1. Require both max_seqs*q and max_seqs*(n+1) <= original budget. Do not enlarge only one consumer or silently lower requests.
5. Native DFlash/DSpark dispatch_cg_and_sync_dp and _run_model already create current draft DP metadata from actual query rows. AG-RS uses those current nonuniform chunk sizes, not target token counts or a hardcoded uniform count. Reject missing/stale metadata, wrong group membership, oversized per-rank rows or malformed scales/topk.
6. Native naive modular prepare/finalize resolves global get_ep_group in prepare, finalize and combine allocation. Bind exact reviewed functions in a private per-instance namespace to the owned manager, with no global Core replacement. Keep native quantization, topk weighting/reduce, allocation and combine behavior.
7. Group coordinates/order must match TP1/PCP1 DP and EP rank lists, ranks and world sizes. Borrow process groups; never create/destroy/suspend/checkpoint target resources. Do not whitelist DP widths. Expert count must divide actual EP width and actual local expert count must agree.
8. Owned CT contiguous HIPC experts can process standard [gatheredM,K] activation after AG-RS prepare; shared experts stay local, and native runner final reduction must not double count. Reuse same signed packed INT4 weights and /16 scales; no LL N32 view on AG-RS contiguous path.
9. Native generic mapping must load draft CT expert weight_packed/weight_scale/weight_shape with local expert ids and canonical return names. Preserve Channel-FP8 scale aliases and all non-CT loading behavior. Never add duplicate Parameters.
10. Reject graph/non-eager target or draft for this increment; native graph buffers exist but this is not GPU Graph validation. Reject LoRA/EPLB/ElasticEP/DBO/SP, non-TP1/PP1/PCP1/DCP1, insufficient capacity or foreign ownership before destructive weight packing. No fallthrough into HT/LL on AG-RS failure.

## Fixed CPU source replay plan

Write and execute replay before production module creation. Construct actual owned/native draft entry with native constructor boundary doubled, observe scoped config identity/backend/restoration. Validate 200 independent capacities and DP widths without a width whitelist. Use actual native AG-RS manager and naive modular prepare/finalize with borrowed CPU collective doubles and actual metadata access; compare rank-major quantized input/scales/routes and reduce-scattered output against independent logical reference for 200 seeded variable/empty row sets. Check target manager/EP globals unchanged and failure recovery. Run actual modular CT packed weights/expert path with provider/quant/permutation doubles and native draft loader on tiny CT fixtures. Preserve and rerun existing DSpark scale-alias and CT target HT/LL replays. Record source hashes and CUDA initialization state.

## Inspected native seams

- `amd/dspark.py65-118,297-319,405-517`: actual draft ctor/forward/load.
- `dspark/utils.py36-69,93-121`: parallel copy and actual public model loader.
- `spec_decode/speculator.py117-153`: native budget/reqs/InputBuffers.
- `dspark/speculator.py46-64`: n vs n+1 query rows and native hidden capacity.
- `dflash/speculator.py315-338,555-578`: current query DP sync/forward context.
- `fused_moe/layer.py286-301,361-391`: scheduler-derived MoE capacity/expert map.
- `all2all_utils.py190-198,420-425`: public explicit manager and native naive PF.
- `prepare_finalize/naive_dp_ep.py121-132,173-284`: manager accesses and modular contract.
- `distributed/device_communicators/all2all.py55-200`: variable-size native AG-RS transport.

## Verification boundary

CPU doubles do not prove provider HIPC GEMM numerics, RCCL transport/rank order, real DSpark sampling/acceptance, GPU/Graph/full-model accuracy or performance. CPU pipeline construction is not a deployment claim. Existing legacy `vllm_hcu` wrapper ABI drift is outside the installed new-package path and is not modified.

## Implemented route

The existing public `DSparkV41DraftModel` entry selects the owned draft class. Only a matching `HcuDSV41CTW4A8Config` with DP>1 selects a private construction config; DP1, Channel-FP8 and other quantization configurations retain native construction. No new entry point or environment switch is added.

| Stage | Owner / behavior |
|---|---|
| Draft config | Shallow private config/parallel config, backend AG-RS; scheduler/speculator and compilation live tables retained |
| Native constructor | Same private config passed as explicit argument and via native current-config scope; restoration on failure |
| Capacity | Positive n/requests/budget; max_seqs*q and max_seqs*(n+1) fit the original budget; actual native buffers are not enlarged |
| Expert weights | Native DSpark weight loop/remap and public expert loaders; packed CT bytes/scales retained |
| Gather | Native AgRsAll2AllManager with owned group checks, actual current query DP chunk sizes; gather BF16 first, then owned per-token INT8 quantization |
| Expert compute | Existing contiguous HIPC experts with the native modular kernel; target LL retains its masked/N32 path |
| Combine | Reviewed native modular finalize and combine allocation, instance-bound manager; variable-size reduce-scatter into local output |
| Shared experts/reduction | Native local shared experts and TP reduction policy, TP1; no added EP reduction |

Three native PF source functions are copied as unchanged bytecode/defaults/closures into private instance globals replacing only get_ep_group. This is a version-bound implementation seam, not a public native PF dependency-injection API. Raw inspect.getsource SHA256 is used; no dedenting. The global EP object, target communicator/manager and native class methods are never rebound. No new process group is created or destroyed. A registered kernel makes supports_internal_mk true, preventing a second outer runner dispatch/combine.

The route is eager MRV2 only, TP1/PP1/PCP1/DCP1, DP>1 with actual EP ownership and expert divisibility. There is no DP width whitelist. Both target and draft enforce_eager must be true. Query metadata remains native; missing metadata is rejected by native manager checks. Shared-memory Engram behavior on a DP target is not changed by this draft-only stage.

## CPU evidence

- `before/report.json`: **0/3**, production module absent, exit 1; failure contract/replay were written before production implementation. Native CT draft weight loading was incorporated into the fixed pipeline before production implementation.
- `candidate/report.json`: **3/3**, exit 0, CUDA uninitialized. The first scenario covers 200 independently seeded n/request/budget settings, DP2/4/8/16 (samples, not an admission whitelist), scoped config identity/unchanged live tables and native constructor boundary restoration. The boundary double replaces the heavy native constructor; it does not prove full draft model allocation.
- Second scenario: 200 seeded variable/zero DP row sets over all participating ranks. Native manager and native modular prepare/finalize execute; rank collectives and quantization are explicit doubles. Inputs/routes and resulting INT8/scales match an independent logical reference exactly. A supplied sum of expert-rank contributions is reduce-scattered by the collective double; no real interprocess reduction is claimed.
- Third scenario: actual native DSpark load loop and expert mapping load six local packed weight/scale/shape Parameters from an mtp checkpoint fixture; actual native modular orchestration executes the owned AG-RS PF and existing contiguous experts. HIPC provider/permutation/quantization are CPU doubles, with exact comparison to an independent unclamped logical weight reference. This is a tiny partial checkpoint fixture, not full checkpoint tracking.
- `target-regression/report.json`: **7/7** existing CT target HT/LL source integration, CUDA uninitialized.
- `alias-current-constructor/report.json`: **17/17** existing DSpark alias source integration, CUDA uninitialized. Its initial run was 16/17 because an old assertion required constructor function identity. That preexisting assertion was synchronized to the preserved constructor parameter contract plus native forward identity; no scenario or numerical assertion was added/relaxed. Non-CT loader and canonical alias differentials remain native.
- Python AST parsing of changed/new production and replay files, runner bash syntax, and git diff --check passed. Existing CPU regression: **312 passed, 66 skipped, 14 warnings in 160.14s**, exit 0 (`regression.log` / `regression.xml`). Independent read-only final review found no concrete blocker in constructor scope, capacity, borrowed group membership/order, raw native PF fingerprints, current DP metadata, contiguous experts/finalization or unchanged target HT/LL behavior.

Evidence root is ignored `tests/vllm033/.cache/v41-dspark-agrs/`, including source-manifest.json, raw reports/logs and migration.patch. Run `VLLM_TEST_PYTHON=/usr/local/bin/python3.12 bash tests/vllm033/run_v41_dspark_agrs_cpu.sh OUTPUT` in the existing CPU-capable container. Core remains `368701db06`, Adapter `3257a934`, no package installation.

## Known limits

This stage does not enable Graph. An existing non-eager DSpark deployment is explicitly rejected by this owned CT draft route; CPU source evidence cannot replace Graph capture/replay verification. Full-model eager DP+DSpark still requires target attention/Engram/provider/ABI acceptance. Native FP8/MXFP4 draft experts are not routed through this CT W4A8 manager. Input tokens must fit the original scheduler budget; the adapter does not enlarge buffers or lower max_num_seqs silently. Native sampling/acceptance, GPU/transport/Graph accuracy and performance were not measured. Historical 0.29/0.32 HumanEval results are not results of this migration.
