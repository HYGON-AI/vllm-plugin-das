# Qwen3.5 INT8 public EP2 HumanEval32

The actual `/model/Qwen3.5-35B-A3B-Channel-INT8-w8a8` checkpoint completed a
sequential AITER/official Triton pair on two gfx936 cards. Each arm used TP2+EP2,
BF16 activations, MTP0, TRITON_ATTN, default FULL_AND_PIECEWISE, prefix caching,
seed0, thinking off and the original HumanEval0–31 protocol with2048 maximum
completion tokens,120s/request and isolated3s/test scoring. Core owns model/CT
loading, shared experts, prepare/finalize and final PYNCCL reduction; custom
all-reduce was disabled identically in both arms.

The frozen Core is `0.33.0+das.0b7b7bd23.dtk2610.epbase2`, including merged
MR82/83/76/75. The20 installed Plugin runtime Python files match the candidate
production files in `7e088e9bb21fff136f43956123f00f95544d98ec`; subsequent
documentation commits do not change those files. The recorded paired Adapter
and DTK/vendor/kernel identities are experiment inputs, not a full standalone
image or complete Core dependency-resolution result.

| Result | AITER | Official Triton |
| --- | ---: | ---: |
| HTTP200 with stop |32/32|32/32|
| Raw tests passed |30/32|30/32|
| Normalized tests passed |32/32|32/32|
| Raw failed task IDs |24,27|2,24|
| Prompt tokens |4203|4203|
| Completion tokens |4858|4901|

All raw failures above are single-code-fence syntax errors. The existing v2
normalizer extracts that single fence; no indentation repair, new normalization
rule or quality waiver was added. Request bodies and their serialized wire
bytes match for all32 tasks. Protocols differ only by endpoint and the derived
request-file digest. Raw and normalized texts each match20/32; ASTs match24/32
even after removing only docstrings. The eight remaining AST differences are
tasks1/9/10/12/18/19/20/21. Both pass the fixed benchmark tests; this is sample
quality parity, not identical arithmetic, text or general semantic equivalence.

Each rank exposes40 real CT INT8 expert layers, global256/local128 weights and
canonical maps, hidden2048/intermediate512/top8. MoE resolves to TP1/EP2.
The AITER arm records actual M1 and M128 ASM compatibility selections and six
successful module loads of three distinct frozen code objects. Other token
shapes retain complete official fallback. The Triton arm records TritonExperts
and no ASM selection/module loads. Both show actual FULL decode events and
separate per-rank ordinary/AOT artifacts; backend cache keys differ at both
levels. Selection/module-load records corroborate this bounded path without
being an all-kernel trace or a new tuned local-E128 profile.

Core changes the requested attention block64 to1088 to align Mamba pages
(3.82% padding), with LBNHC layout. Requested prefix-match unit64 is distinct;
observed prefix hits are0. This run does not qualify effective prefix reuse,
the complete cache-spec byte layout, concurrent serving or warmed performance.

All26 regular checkpoint assets, including14 shards, have matching full parent
pre/post manifests totaling37,602,508,946 bytes. All4550 frozen non-checkpoint
inputs, including53 private native seed copies, are unchanged. The independent
review rehashes those frozen inputs, compares all four full checkpoint manifests
and checks26 asset stats/12 small asset hashes; it does not independently reread
the14 weight shards. API/validator exits are0, both owned groups empty, and both
cards release to4MiB/0% with empty KFD allocations.

Clean lifecycle remains **HOLD**: both services log worker-grace SIGTERMcount2
and internal EngineCore forced termination after30s. No individual worker or
Engine wait status is invented from outer success. The model result does not
cover full-model reload or relax these warnings. R7 remains a separate failed
component cohort; R8 only corrected its producer workspace-lifetime contract.

Independent terminal audit:
`PASS_SCOPED_EP2_HE32_PAIR_NORMALIZED32_LIFECYCLE_HOLD`, SHA256
`4ee97e9921c1f15df8b983148fca53253e871a2740cfc787f35409655ed30f4d`.
Evidence cohort: `v033-int8-ep2-he32-pair-r1-20261110` in the upgrade workspace.
Remote required CI, broader model/topology/native-profile/performance coverage
and the complete migration matrix remain unaccepted. This is the same Qwen3.5
checkpoint as the earlier TP1 evaluation, not an additional unique model.
