# INT8 ASM public EP2 component

This local candidate preserves Core CT loading, prepare/finalize, shared expert
composition and collective ownership. Its qualified geometry is BF16 on gfx936,
global256/local128 experts, hidden2048, intermediate512 and top8, with static
TP2+EP2 and DP/PCP/SP1. MoE itself resolves to TP1/EP2.

Canonical global IDs become actual local slots before the ASM sorter. Local
vendor profiles take priority. Only after a local E128 miss may the qualified
E256 profiles serve M1/17/128; other misses use complete official fallback.
Passthrough is rejected because its pre-reduced flag would bypass the runner's
required final collective. Resolved static elastic capacity1 is accepted while
EPLB, expansion and unqualified topology remain rejected.

The installed R8 component used Core `0b7b7bd231d687209ba3144f7006641acde9c14a`,
the public factory and CT method, two generations of controlled INT8 weights and
FP32 channel scales loaded through the real module loader, and an actual shared
BF16 RowParallelLinear. The official runner output matched the full global256
official expert reference plus shared output for M1/2/17/128 across64 rank states
and56 changed-input/reload Graph observations. Raw local weight values and
pointers, input immutability, cross-rank output hashes and shared invocation
counts were checked. The public workspace was warmed at every shape and locked
before capture; its base stayed stable through reload and all replays.

Independent terminal audit: `PASS_SCOPED_PUBLIC_CT_EP2_RUNNER_RELOAD_COMPONENT`,
SHA256 `bd0385749779e57ee1896fc00ad45d034a94ef4e2639cc43d51f69714dccb41b`.
Both ranks/coordinator exited0,4411 frozen inputs and106 native seed copies stayed
unchanged, and GPUs3,7 released to4MiB with no KFD allocations or outer cleanup.
The evidence cohort is `v033-int8-ep2-public-r8-20261110` in the upgrade workspace.

R7 remains failed: an input/device-snapshot comparison differed after reload.
Its producer captured before workspace high-water warmup/locking. That source
lifetime defect is established, but no saved failing pair proves which side
changed or identifies a unique retrospective cause. R8 corrects that fixture
contract without changing the production algorithms or numerical thresholds.

This is synthetic component loading and manual Graph capture. The later
[real-checkpoint HumanEval32 pair](vllm033-int8-ep2-he32.md) separately covers
checkpoint loading and default engine Graph on the stated model. Full-model
reload, clean serving lifecycle, performance, broader profiles/topologies and
remote device CI remain separate gates.
