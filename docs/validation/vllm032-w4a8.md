# vLLM 0.32 SlimQuant W4A8 review scope

Immutable candidate base: `153e4d9eb88a82f3c59a6fd5e87d790c59675104`.
This is an explicit base, not a claim that MR321 is the latest upstream head.
The read-only upstream v0.32 ref query failed because the configured proxy was
unavailable; its failure is retained in external review evidence.
CPU target: official-derived `27c12321d2c51fce83043806ef0f1519dd52636b`.
The target's AR registration seam is not a W4 dependency.

The new owner is limited to four `hcu_vllm_das/w4a8` modules and its Adapter
entry point. It subclasses the official compressed-tensors config, reuses
parameter loading/mixed ignored layers/clamping/quant tuples and retains the
matching official INT8 math path. Native AITER selection is opt-in, with
checkpoint nibble/byte/scale-times-16 and shape/prepack admission contracts.

Native hidden-width alignment is local to the method. Official routing still
receives logical H2560; the kernel receives zero-padded H3072 and I192; the
result returns to H2560. Fake output agrees through the two public method
properties. The fallback keeps its original properties and delegation. CPU
coverage includes the real official runner and GateLinear, shared expert path,
logical/fake outputs, loaded weak ownership, mutation refusal, fork poisoning,
ordinary/AOT implementation identity, and actual spawn/config unpickling before
EngineCore plugin activation. The INT8 consumer import is deferred until its
fallback constructor; unrelated early consumers still make the oracle refuse.

The standalone manual HCU probe, including a CPU FP32 reference, is
`tests/tools/_probe_w4a8.py`. Existing
legacy HCU CI registry jobs do not select the v0.32 tests. This proposal does not
silently claim that CI runs these new contracts; a subsequent CI integration
must choose a supported v0.32 owner and explicit native opt-in. The portable
pytest files contain CPU guards and never initialize a GPU.

With official CPU platform selected, the real Triton INT8 fallback constructor
rejects `current device cpu`. The spawn contract records that rejection after
checking delayed consumer import and real oracle binding; it does not bypass
the device predicate. Separate CPU-provider doubles cover constructor wiring.
Hidden GPU visibility and no kernel calls are not proof of no KFD metadata
clients: importing this installed Torch stack can briefly open metadata FDs.
Do not run even these CPU tests during an exclusive resource-admission window.

## Evidence boundaries

The historical frozen `0a1086f` Plugin / `3aa22ee` Adapter run is separate from
this source proposal. Its real HCU M1/M17/M65/M512 changed-input Graph and named
native UP/DOWN traces passed; its original TP4 checkpoint with BF16 pinned PLE,
FLA, Graph and prefix passed normalized HumanEval32, 24 real worker refusals,
prefix36 and actual four-rank native events. Raw scoring was 0/32 for fenced
answers; normalized scoring was 32/32. These are distinct results, not a relaxed
quality threshold.

That run required framework SIGKILL of workers at about 9 seconds and EngineCore
at 15 seconds. API/monitor exit zero is not clean model teardown. The offline
source audit establishes existing 5s/4s/15s escalation, not the time spent in
profiler, GPU synchronization, pinned-memory release or interpreter cleanup.
No measured deadlock or pinned-PLE release diagnosis is available.

This proposal uses current fatal semantics with a locally validated public ABI,
so its source identity differs from the historical hardware-tested pair.
Source CPU tests alone do not establish wheel provenance, installed behavior,
new native/Graph numerical equivalence, real model quality or shutdown health.
Those gates must run on a new immutable build after dependency review.

## Dependencies and remaining scope

Minimum dependencies are Adapter PR17, PR19's public
`bridges.vllm_model_load` facade (`2be0047`) and independent public-fatal PR23
(`09bd79a`). Source CPU contracts use a test-only composition of exactly these
published sources, not the broader integration tree. PR20 supplies optional
environment-required startup policy; public ExtensionContext required tests
do not establish that env-policy layer. No exact wheel/SHA is a runtime gate.
Adapter PR18's FP8 fallback is unrelated to W4 INT8. Plugin PR322–326 do not
provide W4; RMS AOT, E5M2, FLA and Flash owners remain separate review units.
No private Adapter registry is read or written by production W4 code.

Official W4A8 already exists: dense CT and AutoAWQ stay official. The exact
target also has `CompressedTensorsW4A8Int8MoEMethod`, whose oracle only exposes
CPU_INT4 and whose expert implementation requires Arm CPU. This is not an
available HCU takeover. This Plugin adds the explicit SlimQuant signed-nibble
checkpoint protocol and HCU AITER expert provider, not a duplicate of the
official CPU class, loader, MoE runner or shutdown body.

Current CPU regression scope includes actual public Adapter atomic hooks and
official alias registry, bad diagnostic strings raising Exception/BaseException,
fork during formatting, preparation/publication inflight fork, fatal dependency
chains (including suppressed context), and clean atomic failure retry. Only
fault providers/device-side model state are CPU doubles. Counts and exact
commands belong to the candidate evidence; no historical180 count is promoted.

The W4 lifecycle validates dependencies locally, before publication. It writes
a safe fatal sentinel before error rendering and refuses inherited incomplete
installs. Pair rollback covers only the two Adapter hooks. Later alias/factor
failure is fatal, not a generic rollback or recoverable optional mount.

MTP, INT8 PLE, EP and `deepep_auto` remain pending independent implementation and
hardware/model gates. Checkpoint MTP tensor inventory is not a working draft
path. This proposal makes no new topology, shutdown-cleanliness or performance
claim and does not merge unrelated optional features.
