# Qualified INT8 EP2 routing fusion

Runtime commit `f2258dec2e5376a6d8f9ef4f78aabd896b097680` preserves PR373's
`2a80b84875f64ec38ec4a231b78c3c07ff70a5fb` changes. It expands the previously
qualified native batch profiles and replaces only GPU canonical-map preparation
with one integer Triton kernel. CPU and ordinary TP/no-map preparation, CK,
native sorting/GEMMs, floating quantization/activation/reduction, architecture
and topology admission, and whole-call official fallback retain their behavior.
The new route module participates in ordinary/AOT content identity; an actual
FrozenFiles regression demonstrates late-file drift refusal (RED then GREEN).

This increment remains explicit opt-in. Static EP2 is qualified only on gfx936
for BF16, global256/local128, hidden2048/intermediate512/top8 and raw layouts.
It does not widen gfx938 EP admission. Earlier gfx938 TP1 evidence belongs to
its own frozen 20-module runtime, Core and Adapter combinations; no new gfx938
device result is claimed for this 21-module increment.

## Native profiles and public component

Genuine local vendor profiles retain precedence. Only after a local-E128 miss
may these actual, independently qualified global-E256 API tuples be used:

| M | SOL_ID1 | SOL_ID2 | BLOCK_SIZE_M |
| --- | --- | --- | --- |
|1|10010|20000|16|
|2,3,4,8,16|10013|20000|16|
|17|10009|20000|16|
|32,128,256|10011|20000|16|
|64|10008|20000|16|
|512|11005|21000|32|
|1024|12001|22000|64|

Unknown M, mismatched tuples and unsupported layouts retain complete official
fallback before native launch; native execution errors propagate. CSV dtype
labels are not a substitute for the actual vendor selector result.

New installed source170, installed170 and source-only CI6 pass. All21 runtime
Python files match source/wheel/install; plugin wheel SHA256 is
`57f6c685a66993af7594a70810b022261f03bfd1175c3c2a81cda4720f8e5f70`.
The initial baseline's three missing VLLM_TARGET_ROOT errors were harness
errors, retained separately from the corrected baseline and behavioral RED.

The new actual two-rank public CT factory/loader/shared RowParallel/NoDPEP
runner/final reduction component covers all13 M, two synthetic weight
generations,208 rank numerical observations,182 rank Graph observations and26
old-Graph observations after reload. All104 distributed output hashes agree.
Unchanged gates give max relativeL2 `0.0030538891442120075`, maxAbs `0.001953125`,
min cosine `0.9999953508377075`. Public workspace is profiled, locked at4MiB
and stable through reload; inputs/maps/parameters remain unchanged as required.
Each rank records13 full native selections with zero unexpected fallback.
Fresh gfx936 route ELF/metadata and native object bindings corroborate the
executed path; these are not exhaustive invocation traces.4420 frozen files
and106 private native seeds remain unchanged; root/ranks exit0 and resources
release. This synthetic component is not a full model reload certificate.

Independent component audit SHA256:
`b4ebb6dfed85fd1b899a6fa06784374ef70c1848028be62902e0a615b415723b`.

## New checkpoint pair

Actual runtime is DTK26.10, Torch2.11, Triton3.6 and gfx936. The serving Core is
`0.33.0+das.0b7b7bd23.dtk2610.epbase2`, including merged MR75/76/83; Core MR84
is not in this cohort. The paired Adapter contains the existing PR66 bridge;
this INT8 path does not depend on that unquantized hook. Exact latest-main
Adapter execution is not claimed here.

Both new arms use physical GPUs2,3, the same
`/model/Qwen3.5-35B-A3B-Channel-INT8-w8a8` checkpoint (total35B/active3B),
BF16/TP2/EP2/MTP0/TRITON_ATTN/default FULL_AND_PIECEWISE, maxlen8192,
batched1024/maxseq4/memory0.84 and identical prefix/Mamba settings. Requested
block64 resolves to manager block1088; page1114112B, four groups of10 layers,
hash/hit alignment64. Cache capacities differ: AITER3367 versus Triton3369
blocks. No exact allocation equivalence or unique capacity-effect claim.

Original HumanEval0–31, max2048, seed0/thinking off, request120s and isolated
scorer3s are unchanged:

| Result | AITER | Official Triton |
| --- | --- | --- |
|HTTP200/stop|32/32|32/32|
|Raw passed|32/32|30/32|
|Fixed normalized passed|32/32|32/32|
|Raw failed task IDs|none|24,27|

The two Triton raw failures are single-fence syntax errors. No indentation
repair, scoring change or quality waiver. All32 serialized request bodies
match; raw/normalized text matches21/32 and strict AST24/32 (also24 after
removing only docstrings). Test parity is not general semantic or arithmetic
equivalence. Each actual rank has40 expected CT INT8 MoE layers and canonical
global256/local128 maps. Startup/final fresh route artifacts are bound by their
actual model-cohort SHA, not asserted equal to the component binary.

Separate warmed work uses four actual3603-token prompts and64 fixed output
tokens, concurrency1/4, two warmup plus five measured rounds,35 raw responses
per arm. Exact requests, per-response token counts and measured prefix hits
match. Client-wall medians are:

| Concurrency | AITER token/s | Triton token/s | AITER delta |
| --- | --- | --- | --- |
|1|62.48815|62.42071|+0.108%|
|4|183.71948|188.49377|−2.533%|

Performance remains open: c1 is effectively tied and c4 is slower. This is a
fixed-order exploratory five-sample comparison including client overhead,
with shared host resources and an untuned official BW config. Output variants
and their round positions are retained; prefix hits do not establish semantic
cache correctness or batch invariance. The earlier P4 pair used GPUs3,7;
cross-pair changes cannot establish a unique route-fusion performance effect.

All4610 frozen inputs (including53 private native seeds) are independently
rehashed unchanged. Four full parent checkpoint manifests match26 assets/14
shards; independent review checks26 stats and12 small-asset hashes and compares
the full manifests, without independently rereading37GB of shards.

Both API/validator parents exit0 and groups/cards/KFD release. Both logs have
zero ERROR/EngineDeadError lines, but retain worker-grace SIGTERMcount2 and an
explicit force kill of one remaining EngineCore; individual internal waits
remain unknown. **Lifecycle HOLD** remains. Outer resource release is not
clean framework teardown.

Independent pair audit SHA256:
`1d869dc63438e1166e5741ea18639cc70072eda70c13458d3ef336754111fa74`;
status `PASS_SCOPED_NORMALIZED32_PAIR_MIXED_WARMED_RESULT_LIFECYCLE_HOLD`.

The first planned GPU3,7 model launch was refused by final admission because
GPU7 had an external allocation; its driver never started. That rejected
cohort and all earlier failures remain sealed. This new pair uses2,3 for both
arms and never stops unrelated processes.

## Integration limits

Fresh three-repository API plus actual-fetch checks preserve concurrent work.
Adapter main advanced2c06→cd1a during follow-up: actual diff adds SGLang startup
and metadata/JIT adaptations, with no vLLM bridge/public ABI, packaging or CI
change. This source-impact decision does not turn the old paired installation
into a latest-main runtime validation.

Existing PR373 is updated instead of creating a duplicate. Required remote
device-image CI and normal merge approvals remain mandatory; no image fallback,
policy bypass, general acceleration, wider topology, clean lifecycle, complete
148-feature acceptance or last-good claim is made. AI assistance was used.
