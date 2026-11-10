# Native INT8 EP2 batch-profile candidate

This follow-up starts from published PR373 head33847792. It changes only the
global-profile compatibility policy after an actual localE128 miss. Actual local
vendor profiles keep priority; geometry/topology/dtype/layout/opt-in guards,
official whole-call fallback, canonical maps and native-error propagation remain.

The real installed native stage used global256/local128, BF16/H2048/N512/top8,
Core unique fused_topk and canonical linear/round-robin maps for simulated ranks
0/1 on one actual gfx936 card. It tested156 cases,628 numeric states and471
changed-input/map Graph observations, including212 exact-zero remote/padding
states. The unchanged thresholds were relL2<=.005, maxabs<=.02, cosine>=.9999.
Observed extrema: .0005130976787768304/.001708984375/.9999997615814209.

| M | SOL_ID1 | SOL_ID2 | BLOCK_SIZE_M |
| --- | --- | --- | --- |
|1|10010|20000|16|
|2,3,4,8,16|10013|20000|16|
|17|10009|20000|16|
|32,128,256|10011|20000|16|
|64|10008|20000|16|
|512|11005|21000|32|
|1024|12001|22000|64|

These are actual public API lookup results, not inferred BF16 CSV rows. All13
localE128 lookups missed. Native loading records identify10 frozen code objects;
metadata, source/native/config files and53 private seed copies are unchanged.
The worker/controller exit0, group empties and GPU7 releases to4MiB/0% with no
KFD allocations. Independent report status PASS_SCOPED_BROADER_NATIVE_PROFILE_STAGE,
SHA a6c60be8ea33be0bdf533174795fcac0f16ba9f8de0aede79918850f78041c5f;
cohort `v033-int8-native-profile-expansion-r1-20261110` in the upgrade workspace.
The test fixture retains the observed plans and report identity.

M512/1024 change both stages and block size. Admission must match the complete
tuple; a known first-stage ID alone does not qualify the other values. Unknown
M, mismatched tuple, missing opt-in, wrong dtype, padded/shuffled plans and other
unsupported conditions retain complete Core fallback before native launch.

This new source candidate still requires a separately built/installed artifact,
actual two-rank public factory/CT loading/shared runner/reload/Graph validation
and a new full checkpoint originalHE32 pair plus warmed serving comparison.
The earlier EP2 HE32 at runtime7e088e9b/33847792 does not certify this extension.
No broader all-M profile tuning, FP16 EP2, model count, performance, clean model
lifecycle, CI or whole migration acceptance is transferred. Earlier R7FAIL,
model lifecycle HOLD and Qwen3-30B normalized28 quality HOLD stay unchanged.
