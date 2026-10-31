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

Source validation on frozen target a198423b7: Adapter 78 CPU tests passed,
including missing/failed/duplicate owners and actual fork. The unchanged
baseline fails six new owner/fork assertions (six others pass). Plugin 58 CPU
tests passed, including blocked-runtime discovery, real public host/entry point
contracts and existing Flash tests. First discovery attempt exposed the existing
attention package's eager backend import; the factory was moved to a top-level
module, preserving the package's existing exports. Both failure logs remain in
the task evidence directory. Installed/native/model evidence is pending.
