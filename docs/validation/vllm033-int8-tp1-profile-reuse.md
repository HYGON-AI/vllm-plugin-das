# Qualified raw INT8 TP1 plan reuse

The explicit public AITER INT8 expert backend can reuse measured ASM plans for
E128, hidden2048, intermediate768, top8 when its real local vendor lookup misses.
Core retains CT loading, routing, quantization, activation/reduction and runner
ownership. Adapter supplies real device detection and extension health.

The additional path requires gfx936, ordinary TP1/EP1rank0/DP1/PCP1 with no EP,
sequence parallelism, EPLB or passthrough; matching BF16/FP16 input/output; dynamic
token symmetric INT8 without zero points/static activation scales; raw canonical
local experts and no expert map. It is entered only by `asm_only=True`.
Adapter admission checks visible device zero. For a nonzero tensor ordinal,
the actual tensor device must also report gfx936; mixed or unresolved devices
retain fallback. This covers TP1 workers placed on a different visible ordinal.

Only M=1,2,3,4,8,16,17,32,64,128,256,512,1024 is qualified. The provider queries
the real E256/H2048/N512/top8 donor at the same M and checks both stage IDs and
the sorting block against the measured tuple. Missing or changed plans, padding,
shuffle, other geometries/dtypes/topologies/architectures and unqualified M keep
the whole official fallback. A real local vendor profile takes precedence.
The log identifies TP1 reuse and the donor origin; it is not a locally tuned plan.

The existing normal/AOT identity covers the changed provider source plus actual
vendor configuration and code objects. The policy does not replace vendor tables,
change global architecture/AITER flags, add Core/Adapter model implementations or
weaken the existing EP gate.

## Evidence and limits

On Core0914dbb6/Adapter2536f546/Plugind532b182, gfx936 physical7, a separate direct
native experiment ran26 new-geometry and6 old-geometry controls across BF16/FP16.
All192 numeric triplets and96 input-changing, NaN-poisoned-output Graph replays
passed unchanged relL2 .005(BF16)/.001(FP16), maxAbs .02, cosine .9999. New geometry
maxrelL2 was .00042007493902929127, maxAbs .0009765625, mincos .9999998807907104.
Native output destination reuse and allocation equality passed32 cases. In the
original d532 public path, all26 new-geometry cases still fell back and left the
supplied hint unchanged. The experiment called unchanged native composition
explicitly with genuine donor configurations; it did not deliver this policy.

Independent terminal report SHA256:
`e190fabfe62667f87f9318919dc348aaca41ad8e30bc0b65c0fdd824347f2edb`.
The test data fixture retains the actual13 donor tuples and build identities.

This candidate additionally has source selection/rejection tests. Genuine
installed public dispatch/numerics/Graph, full checkpoint original HumanEval32,
prefix/concurrency, throughput and model lifecycle still require their own runs.
Direct-call results and CPU mocks do not establish those outcomes or transfer
qualification to gfx938/EP2/reload. No global last-good claim follows.
