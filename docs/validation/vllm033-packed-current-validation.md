# Native packed integration validation, 2026-11-14

The native packed candidate was merged normally with current v0.33.0
`2cdc04242cee17fb24f7e758e62cb5c5ea0a3222`, producing runtime source
`01dee6db8abfe689d0aa796de2edfd2361bdef07`. The integration preserves the
0.33 package version, current Adapter minimum, AITER entry points, root-build
guard and concurrent upstream additions. It does not reintroduce the legacy
platform plugin. Unrelated DeepSeek V4.1 testing was deferred by the user.

The normally resolved Plugin wheel has SHA256
`2ae2e1a21b984502921743e235106b8dad4290bdbaeb49d6ace70c71cec935ee`.
Core `ae5ebff332de0bd467f285e1df0127b5e7d6b3ac` HIP modules were rebuilt;
the installed wheel, Python sources and parent/child origins were checked.
Portable target-source contracts passed 622 tests with 6 HCU cases deselected;
the final installed packed CPU contracts passed 64 tests. The source-overlay
and fully installed checks are distinct evidence scopes.

On gfx936, the four existing TP1 native packed tests passed for BF16/FP16 and
padding/no-padding, with two official raw reload generations and two changed
input/routing states each: 16 replay states. The tests use four experts,
hidden size 256, independent signed INT4 CPU references, stable raw/scale/output
addresses and wrong-nibble/signedness negative controls. These are component
results, not TP2/TP4 or full-checkpoint/model quality/performance acceptance.

The retained `resume-20261114/packed-device-summary.json` indexes the native
receipts. Interpreter teardown still reports the installed BoltOps
duplicate-qualified schema registration/finalizer error. An isolated vendor
source reproduction and candidate correction passed on CPU; the installed
vendor package was left unchanged. Therefore clean native lifecycle is not
accepted, despite the numerical assertions passing and VRAM returning to its
baseline. This vendor issue must not be hidden by ignoring teardown errors.

Current project image/device CI remains a separate gate: the configured
`HARBOR_SITE` and approved immutable `HCU_CI_IMAGE_V033` must resolve before
device tests can run. The local component results do not waive that gate.
Full packed checkpoint HumanEval 0-31, required TP2/TP4 coverage and clean
shutdown remain open for this combined artifact.
