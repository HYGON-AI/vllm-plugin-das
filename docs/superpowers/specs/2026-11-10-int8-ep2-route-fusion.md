# Integrate exact GPU EP route preparation

Use current PR373 source2a80b848, retaining the gfx938 validation update, plus
the unchanged13qualified-profile increment. Frozen P4 fullpair normalized32
both/raw29vs31; AITER warmed c1/c4 is7.87%/5.96% slower, lifecycleHOLD. Do not
transfer those model results or claim this candidate is faster before rerunning.

Independent primitive f05463de passes264cases/1044states/741changed-inputGraph
on realgfx936 GPU3. Exact local-slot permutations, signed64 bounds before
narrowing, remote-only constant-1, identity/no-map and primitiveempty covered.
Graph ratio-of-medians~9–13x applies only to the helper; not a model root cause
or end-to-end claim. Actual environment Torch2.11.0+DTK26.10/HIP7.2.26365,
Triton3.6.0 (distribution suffixes recorded in borrowed-base receipt).

Add one optional `moe/int8_routes.py` containing the proven integer kernel and
wrapper. Dispatch there only for GPU tensors with a non-None canonical map,
through the existing gfx936-only EP2 public admission. Keep CPU reference,
ordinary TP/no-map including int64, CK and unknown-profile fallback unchanged.
No float arithmetic, sorting/GEMM/quantization/reduction changes, Core or Adapter
patch, native CSV tuning, new backend/default or widened topology/architecture.

Include the new module in pre-publication FrozenFiles identity so ordinary/AOT
factors invalidate when its source changes. Existing late/inherited consumer
refusal, getter/module/architecture health and fatal publication remain intact.
Do not initialize GPU/JIT at extension discovery. Core's scalar DeepGEMM
apply_expert_map is not a tensor-launch ABI with these signed bounds; no Core
MoE/router/collective is copied. Public Core remains owner of those stages.

Regression: real identity capture must reject late edits to the new module;
CPU metadata/source/package,21source/wheel/installed files, real13M two-rank
loader/shared/NoDPEP/reload/Graph and native/fused dispatch, then original
HE0–31 and matched warmed3603→64 c1/c4 on a new frozen artifact. Keep all raw
scores/text variation, actual cache allocations, timing samples and lifecycle
warnings. No quality/tolerance/CI waiver. Fresh three refs and actual MR source
feature checks at build/validation/publication; normal required CI/approval.
