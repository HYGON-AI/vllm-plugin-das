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
checks for this exact clean pair are recorded below. No full model/quality,
Graph device-trace completeness or dirty-host recovery claim is made.


Exact clean pair installed validation completed: Plugin55 passed;
Adapter48 passed/one source-layout test deselected. The initial absolute
node-id deselection did not match pytest's relative node-id and retained that
source-only failure; the final runner used the exact test name selection,
without production changes. Source49 includes this assertion.

Independent wheels bind12 Plugin/194 Adapter Python files and every hashed
RECORD row;2649 target Python/native files plus parent/child native module
origins were checked. Plugin wheel SHA256
`7e6052468f71d543724309fb887510aa68fb98d046de4704b3dfa7fc295ba5e9`;
Adapter wheel SHA256
`d5c3abd89bd11751a9079ecf5a5a52e33f9bdfb846b70ebde5b0576c110c77d6`.

The actual installed, unmodified official CLI with real HCU/vendor imports
passed. Its process-local reports and selector accepted explicit auto/E5M2
Flash, refused invalid block128, and returned official Triton by default.
No provider/platform/discovery doubles or GPU/model claims are involved.
The PR17 baseline does not include the separate required-startup env gate;
this evidence checks actual reports and selector behavior directly.

With the target and paired wheels installed in an isolated environment,
run `python tests/tools/check_flash_owner_cli.py /tmp/flash-owner-cli.json`
from outside production source paths, with empty PYTHONPATH,
`HCU_ADAPTER_HOST=none`, `VLLM_PLUGINS=hcu_adapter`, actual DTK/HCU library
settings and one visible HCU. The output path must not already exist.
The driver observes unmodified official `--help` and the public selector.
