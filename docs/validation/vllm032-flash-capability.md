# Explicit Flash capability ownership

The independent `das_hcu_flash` entry point declares `hcu-flash-attention`.
Its factory imports no framework/vendor runtime. Preparation checks the
vLLM 0.32 release line, actual Adapter HCU detection, the existing official
attention seams, and the public vendor varlen callable. It installs no RMS,
AITER, quantization aliases or model lifecycle hooks. The legacy `das_hcu`
bundle retains its existing semantics and capability names.

The paired Adapter gate requires exactly one process-local ACTIVE owner before
it imports or returns the explicit HCU FLASH_ATTN class. A missing, FAILED or
ambiguous owner refuses selection. Forked children must activate independently;
the parent's report is not accepted. Official configuration validation still
runs after the owner gate. Default TRITON, MLA and sparse MLA selection retain
their official paths without requiring Flash. Direct embedding API users must
complete public runtime activation first; the selector performs no activation.

Preparation and ACTIVE reports do not prove that a configuration is supported,
that a native kernel ran, or that model accuracy passed. Missing vendor support
is an optional FAILED report unless this capability/extension is required.
A clean legacy preparation failure does not invalidate this independent owner;
this change makes no rollback claim for dirty legacy activation failures.

Review branch baseline: merged Plugin MR321 `153e4d9` and Adapter PR17
`35f8f05`. The Plugin review adds only this independent entry point/factory,
tests and this note; it contains no W4, E5M2 scale, RMS, FLA or SlimQuant changes.
The paired Adapter review adds only its own-capability selector gate/tests.

Exact clean-review source validation on target `a198423b7`: Plugin55 CPU tests
passed, Adapter49 bridge/Flash CPU tests passed. Existing Flash tests differ
from the larger integration candidate; its58/78 counts and installed GPU36
result are not assigned to this review artifact. Installed and actual CLI
checks for this exact clean pair remain pending. No full model/quality,
Graph device-trace completeness or dirty-host recovery claim is made.
