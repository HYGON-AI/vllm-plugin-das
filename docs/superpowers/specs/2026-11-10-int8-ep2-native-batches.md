# Qualified native batch profiles for public INT8 EP2

The published PR373 runtime at33847792 qualifies global-profile compatibility
only for M1/17/128. New independent native stage a6c60be8 passes156cases,
628states/471changedGraph observations on one realgfx936 card with simulated
canonical map ranks and actual globalE256 API plans/localE128 misses. This is
prerequisite evidence, not new public EP/model/performance acceptance.

Extend only the exact tested M/plan tuples recorded in
`tests/vllm033/data/int8_ep2_gfx936_qualified_profiles.json`. M2/3/4/8/16 use
10013/20000/B16; M64 uses10008/20000/B16; M32/128/256 use10011/20000/B16;
M17 uses10009/20000/B16; M1 uses10010/20000/B16. M512 uses11005/21000/B32,
M1024 uses12001/22000/B64. Guard SOL_ID1/SOL_ID2/BLOCK_SIZE_M together.
Keep actual local vendor profile precedence. Unknown M, mismatched plan,
shuffled/padded/unsupported dtype/geometry, or missing opt-in keeps complete
Core fallback before native launch. Existing topology/health/loader/ABI guards
and native-error propagation remain. No vendor CSV/cache edit, private seam,
platform spoof or exact-wheel runtime pin. This is a finite extension, not all-M
native tuning; ordinaryTP/CK and officialAUTO/Triton stay unchanged.

New source/wheel/env must be separate from all frozen launched sources and
installed338runtime. CPU behavioral RED→GREEN, source/package+installed identity
and independent code review precede device launch. Public2rank factory/CTload/
sharedRowParallel/reload2generations must use all13M, actualCoreNoDPEP andfinal
collective, not manual composition.104distinctdistributedstates/208rankmetrics,
91distributedGraphwaves/182rankGraphobservations incl26unchanged-inputreload
observations. Same relL2.005/maxabs.02/cos.9999. Full runner warmups allshapes,
max scratch1024*2048 BF16=4194304bytes andpublicworkspace lock beforecapture;
stableptr/no retainedviewpin andinput/priorhost attribution guards retained.

Then rerun full realQwen35INT8 TP2EP2 originalHE0–31 AITER/officialTriton on new
artifact; preserve raw/normalizedscores/failedsets/text+AST differences/finish/
tokens/checkpoint+sourcefullhashes and allinternalteardown warnings. Include
read-only primitiveJSON cachegroup/spec/tensor inspection in a new observer;
managerblock1088 versusrequested64/prefixunit64 must remain distinct. Public
metadata must not leak TorchVersion/Tensor objects to Core msgpack.

Warmed c1/c4 comparison must use actual tokenized eligible long prefixes,
identical fixed64token request bodies,2warmup+5measuredrounds/cell, raw samples
and counter deltas, separate HE32 metrics. Both stages keep private ordinary/
AOT caches and exact artifact provenance. No cold/JIT latency as warmed speedup,
no broad/order-independent/p95 claim from fixed-order singlepair. Lifecycle
HOLD does not disappear because API0/resourcesrelease; no timeout/GC change
without causal evidence. Existing Qwen330B28 qualityHOLD remains separate.

Before publication repeat latest3repo actual code/openMR support checks. Update
existing373 ifstillopen, otherwise use a newly justified forkbranch/MR. No CI or
humanapproval bypass. Missing verifiedCIimage/config question remains pending;
full148featurematrix/lastgoodfalse/wholegoal intact.
