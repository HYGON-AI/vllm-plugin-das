---
name: vllm032-w4a8
description: Maintain the opt-in SlimQuant signed-nibble W4A8 provider through public vLLM and Adapter seams.
---

# SlimQuant W4A8 maintenance

Read [the lifecycle ABI](../../adapter_extension/W4A8_LIFECYCLE_ABI.md) and
[validation scope](../../docs/validation/vllm032-w4a8.md) before changes.

Keep official CT/AutoAWQ, loader, model, router, MoE runner, Graph scheduling
and shutdown ownership. This Plugin owns the explicit SlimQuant checkpoint
format and additional HCU AITER expert path. Do not copy an official class body,
pretend CDNA/CPU identity, or add a Plugin owner for already usable behavior.

Use public `hcu_adapter.bridges.vllm_model_load` for vLLM resolved-worker,
executor and pre-env-cache admission. Shared declarations/contexts remain in
framework-free `hcu_adapter.model_load`. Minimum Adapter sources are PR17,
public-facade PR19 and public-fatal PR23; env-required deployment additionally
uses PR20. Validate compatible synchronous callables, not an exact wheel/hash.

Keep a local public fatal ABI validator; do not import a private bundle helper.
Before provider import or mutation, preflight both named hooks, alias/cache
conflicts and callable ABI. Atomically publish the hook pair before alias/factor.
Clean second-hook failure remains optional and retryable. Successful pair then
alias/factor failure must latch public ExtensionCommitError with its cause.
Store a fixed refusal before formatting: `__str__` can throw BaseException,
reenter or fork. Fatal dependency chains cannot become clean optional errors.
An inflight/failed/frozen parent cannot become a fresh healthy child by resetting
locks. Host health queries occur before owner locks; never implement recovery.

Preserve explicit `-q slimquant_w4a8`, native default OFF via
`VLLM_HCU_DAS_W4A8_NATIVE=0`, and `1` as opt-in. Global AITER stays OFF unless
independently requested; explicit backend wins. OFF selects the matching
official fallback, not a claim of a specific GPU kernel. Policy changes after
installation/freeze are refused or stay visible in both ordinary/AOT factors.

Keep signed high/low nibble, scale-times16, all-M admission before packing,
logical router width, native-only internal zero padding/output trimming, and
input alias contracts. Preserve lazy official INT8 consumer import across
spawn/config unpickling and the priority100 bundle / priority90 W4 order;
unrelated already-imported consumers must still be rejected.

CPU gate: the four `tests/vllm032/test_w4a8_*.py` files plus package-boundary
tests, with all device visibility empty, HOSTnone, plugins disabled, explicit
target/Adapter sources and TMP/cache in an independent SHM root. Tests must be
pytest compatible and bound their own fork children. Real registry/hooks/Worker
contracts and CPU doubles must be identified separately. Test pure import,
callable signature failures, clean versus fatal publication, broken formatting,
fork before/during publication, target/draft guards, config overrides and hashes.

Source tests do not prove installation, HCU numerics, changed-input Graph,
engine AOT persistence, model32 accuracy or clean exit. Rebuild immutable
artifacts and bind RECORD/source/parent-child origins before claiming new
installed results. Real numerical gates need an independent FP32 reference,
multiple shapes and changed-input Graph; full models keep original32 raw and
normalized scores/prefix criteria. Historical forced shutdown stays failed:
API0 or all-owned-absent after framework SIGKILL is not clean teardown.
MTP/INT8 PLE/EP/deepep_auto remain separately gated.
