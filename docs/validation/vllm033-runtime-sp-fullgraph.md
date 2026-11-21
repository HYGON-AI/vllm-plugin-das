# Runtime-SP real projection fullgraph candidate

This follow-up to #479 fixes a CPU-reproduced compiler boundary without changing
Core, Adapter, vendor sources, native binaries or GEMM algorithms. It remains a
candidate. The original TP16/24 and full-GEMM-reference4/24 numerical failures,
unrun checkpoint HumanEval and device/default-serving gates remain open.

## Implementation

Real QKV/Row/MergedColumn projections on Core0.33 `98a09850` reproduce the
`posix.getpid` fullgraph failure in all three original paths. Keep birth PID,
group identity and geometry checks inside narrow tensor-producing preparation
and collective ops registered through Core's `direct_register_custom_op`.
The preparation result is consumed before official quantization/GEMM; group
collectives retain their existing implementation. No PID is marked constant,
no compiler/Graph policy is disabled, and no numerical tolerance is changed.

Preparation applies the original local/global device mask and passes it as
an explicit tensor input when present. Reading it only
inside the opaque op would hide its capture-time address from compiled Graph
input management. Four regressions reject that earlier hidden-mask candidate;
the final tests execute captured FX graphs without Python SP batch context and
replace the batch mask storage while retaining one compiled graph.

A standalone call without a batch clones the input to preserve the functional op's non-aliasing
schema; a real model batch already needs the masked output allocation. This
extra standalone copy and new boundaries are not performance-qualified.

An enabled outer model registers a weak health owner with the required
communication capability after construction. Health checks the model TP group,
exact named projection bindings and frozen geometry, plus existing attention
and MoE publication owners. Publication uses the existing Adapter lifecycle
lock and immutable snapshots. An expired model refuses health rather than
silently allowing an old captured graph. No model weights are retained by the
registry and no forward path repairs an owner.

Python op bodies do not execute on cached CUDA Graph replay. Adapter's actual
required-owner Worker boundary is therefore a separate prerequisite alongside
the communicator selector. Activation rejects a missing
`require_forward_healthy` public seam before selector publication. Proposed
Adapter #97 and the separate selector candidate are not claimed merged or
automatically delivered together; a compatible combined Adapter is still needed.

## Validation scope

The device-hidden CPU process uses actual Core0.33/Torch2.13 DTK2610 constructors
and loaders with explicit CPU GEMM/collective doubles. An unused FlashAttention
native import aborted the first orchestration attempt. The final CPU runner
models that unused provider as absent; it supplies no FlashAttention kernel.
Architecture metadata is doubled only during Core's ROCm module import and
restored before assertions. Device visibility is empty, device nodes are absent,
and GPU initialization is denied. Earlier orchestration failures are retained.

- Baseline: three real projection fullgraph regressions fail at `posix.getpid`.
- Candidate: **305 source cases** pass, with zero failures/errors/skips. The
  independent noneditable installed phase passes the same305 cases plus the
  four external Worker/CoreGraph composition cases below (**309 total**).
  Common source/installed JUnit case identities match. The default CPU Inductor
  is exercised independently of eager/AOT compiler backends.
- Static selection: **154 cases** pass (source8, CI helpers98, skill48), with
  zero failures/errors/skips. The426 reported subtests are not added as unique
  cases. Skill schema validation passes without importing Torch. An initial
  151/3 result caused by inherited PYTHONPATH is retained separately; clearing
  the management environment makes the unchanged tests pass.
- The installed wheel's115 package payloads match source/wheel/install bytes;
  installed RECORD hashes are checked. This does not qualify a native build.
- Real fork tests cover inherited compiled projection rejection and required
  model health while a separate parent thread holds the lifecycle lock.
- Four CPU composition cases use actual installed Adapter #97 Worker and
  Core `CudaGraphManager.run_fullgraph`, with only native built-in startup,
  platform selector registration, GPU graph and transport replaced by doubles.
  The healthy arm reaches replay; model-group, projection-binding and
  projection-group drift stop before PP wait, offload and replay. This is not
  actual device Graph execution or a released combined Adapter verdict. These
  four cases use an external evidence script and are not included in the
  repository's305-case selection. Restoring the owner does not clear the
  actual Adapter host's latched refusal.

During implementation the original dense-down owner regressions caught a
changed validation order (285 passed,3 failed). Preparation was moved before
shape validation; no test or assertion was weakened. Subsequent complete
selected matrices pass. Keep the original failure as diagnostic evidence.

The new regressions live in `tests/vllm033/test_runtime_sp_compile.py` and
`test_runtime_sp_replay_health.py`; original transport, loading and padding
tests remain in the selected matrix. The one added fixture field represents
the outer model's new frozen projection-owner snapshot; it does not disable
the original assertions.

The305-case selection comprises these existing/new test files:

```text
tests/vllm033/test_runtime_sp_compile.py
tests/vllm033/test_runtime_sp_replay_health.py
tests/vllm033/test_runtime_sp_attention.py
tests/vllm033/test_runtime_sp_model.py
tests/vllm033/test_runtime_sp_boundaries.py
tests/vllm033/test_runtime_sp_communication.py
tests/vllm033/test_runtime_sp_trace.py
```

The final noneditable Plugin wheel has SHA256
`fb41d3babef5a68e3035b3321ae947f3541cfe5e151c1bf6a0495ab64b1a8984`.
Local source/installed/static JUnit, runner commands, module-origin observations
and failed negative controls are retained separately with their hashes. No
terminal CI result from an older production head qualifies this follow-up.

Device collectives/changed-input Graph, the real Core serving compiler pipeline,
trained checkpoint/default runner, all required topology/features, HumanEval32,
original numerical gates, performance, normal shutdown and gfx836 acceptance
must still be completed. CPU default Inductor is not a GPU or model verdict.
Keep #479 Draft. AI-assisted implementation/review: OpenAI Codex.

## Native projection gate

`test_runtime_sp_projection_hcu.py` adds 24 installed-device cases: six real
QKV/attention-row/dense/shared MLP projection paths, BF16/FP16, with and without
an explicit padding mask. The tests create an actual Core TP1 NCCL group and
retain Core's loader, post-load processing and selected GEMM implementation.
They use default Inductor with `fullgraph=True`, then capture that compiled path
in a HIP Graph. Three replays change inputs and masks at persistent addresses;
masked NaNs must be removed before GEMM. An independent CPU FP32 dyadic-input
oracle followed by the output dtype cast requires exact equality.

The existing device tier selects the file through its `hcu` marker and
`VLLM033_HCU_TEST=1`; portable selection excludes its `_hcu.py` filename. No
workflow, gate, provider, compiler backend or native source is relaxed.
Adding the gate is not a passing result. Pending device execution must stay
pending; the prior green CI does not qualify these added cases. TP1 does not
exercise multi-rank exchange/gather/scatter, model activation, Worker replay
health, a trained checkpoint or HumanEval. Those original gates remain open.
