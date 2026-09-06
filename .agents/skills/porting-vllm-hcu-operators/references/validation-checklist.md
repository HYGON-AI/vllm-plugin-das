# vLLM-HCU operator porting checklist

Use this checklist as evidence collection, not as permission to change external
systems or broaden the requested model/operator scope.

## 1. Pin and inventory

1. Confirm the worktree, branch, dirty files, base SHA, vLLM SHA, and SGLang
   SHA. Preserve unrelated changes.
2. Resolve the vLLM source root by checking `<root>/vllm/__init__.py`. On the
   common `/models/zb/vllm_025` layout the repository may actually be
   `/models/zb/vllm_025/vllm`; passing the outer directory can silently make a
   test runner use the installed wheel.
3. Record Python, Torch/HIP, device architecture, DTK, LightOp, AITER, and vLLM
   versions. Use installed distribution metadata when a module reports
   `unknown` or a source checkout reports `dev`; record both source and
   distribution identities when they differ. Inspect callable signatures from
   the active interpreter.
4. Search both repositories with `rg` for import sites and for the downstream
   tensor ownership, not only matching function names.
5. Recheck prior `dependency-unavailable` rows. Package upgrades can move a
   symbol without adding device configs or a compatible ABI.

For each candidate, answer:

| Question | Required evidence |
|---|---|
| Is it installed? | Exact public import succeeds and required exports are callable. |
| Is it supported here? | Device config/solution exists or an explicit validated config works. |
| Is there a seam? | Exact vLLM/plugin expression owns all mutated tensors and lifecycle. |
| Is it correct? | Live-HCU comparison with an independent reference. |
| Is it useful? | Stable full-path speedup for every admitted shape. |

## 2. Choose the disposition

- `accepted`: all five questions pass; add a narrowly eligible production route.
- `already-covered`: existing plugin path provides the same contract; test or
  document it instead of duplicating dispatch.
- `performance-rejected`: accuracy passes but the complete route misses the
  performance gate; keep benchmark support, not a dormant production switch.
- `dependency-unavailable`: the required installed public ABI cannot execute.
- `no-vllm-seam`: SGLang owns a tensor/cache lifecycle vLLM does not expose at
  an equivalent boundary. Do not transplant model-local mutations.

## 3. Plugin implementation contract

- Put runtime helpers under `vllm_hcu`; never edit the canonical vLLM checkout.
- Use existing patch helpers such as exact-module loading, callable/class
  validation, and signature validation. Declare `PATCH_ID`, `TARGET_MODULE`,
  and exact `TARGETS`; register the callback and a direct test reference.
- Validate every target before the first mutation. On partial/stale markers,
  raise compatibility errors instead of layering another wrapper.
- Audit `from x import Symbol` consumers. Replacing `x.Symbol` later does not
  update their captured binding or constructors that captured selectors.
- Preserve output shape, dtype, device, strides when required, mutation
  semantics, stream, distributed topology, compile/fake registration, and
  CUDA-graph behavior.
- Optional dependency absent or non-callable: reject eligibility and use the
  unchanged official path. Do not catch broad exceptions around an eligible
  kernel merely to make tests pass.
- For backend-specific weight packing, select before packing, mark the layout,
  make installation idempotent, validate marker generations/shapes, and fail
  closed after mutation.
- Add a one-time route marker early enough to appear in worker logs. Logging is
  evidence only when validation reads bytes written by the current invocation.

## 4. Controls

Every new route is subordinate to:

```text
VLLM_HCU_USE_CUSTOM_OPS && VLLM_HCU_USE_<DESCRIPTIVE_LEAF>
```

Test lazy parsing of `1/true/0/false`, the default, master-off, and leaf-off.
Document the eligibility matrix and fallback. Avoid a switch for rejected or
unavailable code.

## 5. Numerical and performance validation

Portable RED/GREEN tests should cover dependency absence, non-callable
exports, bad rank/dtype/device/shape/stride/topology, happy-path routing,
official fallback, idempotence, and ABI failure behavior.

Live-HCU tests should use production dimensions plus small edge cases, fixed
seeds, zeros/extremes where meaningful, independent FP32 math, predetermined
tolerances, NaN/Inf checks, and input/weight mutation checks. A comparison only
against another optimized kernel is not independent evidence.

Benchmark the exact call path. Include transposes, `contiguous`, quantization,
packing amortization policy, allocations, output conversions, and required
synchronization. Warm up compilation, use repeated medians, keep all raw shape
records, and compare with the backend that production actually uses. An
operator that wins only when setup costs are excluded is rejected or gets a
predicate that excludes those shapes.

Typical commands (adjust paths after verifying them):

```bash
python3 tools/run_patch_tests.py --suite contract \
  --vllm-source /models/zb/vllm_025/vllm

HIP_VISIBLE_DEVICES=4 python3 tools/run_patch_tests.py \
  --suite accuracy-hcu --vllm-source /models/zb/vllm_025/vllm \
  --target tests/accuracy/test_<operator>.py -- -s

python3 tools/check_patch_test_coverage.py --json
python3 tools/check_production_boundary.py
python3 -m compileall -q vllm_hcu tests tools
git diff --check
```

Store reproducible benchmark reports under a task-specific `/tmp` path and
summarize exact command, environment, shape count, pass count, min/median/max,
and error bounds in the repository validation document.

## 6. Models and shared accelerators

Inspect `/models/*/config.json` for architectures and dimensions; do not assume
every checkpoint exercises the new route. Run operator accuracy before loading
large checkpoints. Select idle devices with `rocm-smi`, record external load,
and terminate only process groups created by the current test.

Feature-off and feature-on runs must use identical model/eval settings and
separate clean artifact directories. Capture the server-log byte offset before
each start, require the enabled marker only in new bytes, and reject stale log
matches. HumanEval-32 acceptance requires report sample count 32 plus exactly
32 prediction and review records; compare Pass@1, not just server health.

Do not turn noisy end-to-end TPS into an operator rejection. If isolation is a
user requirement, rerun on an idle host; otherwise report TPS as observation.
When the user explicitly permits operator-only evidence, a model that cannot
cover the route or finish under shared load is not a blocker, but the report
must say the model result is incomplete.

## 7. CI, review, and MR

- Add live tests to `tests/hcu_ci_registry.py` and the appropriate selector
  patterns; verify collection is nonzero and the job architecture matches.
- Run inventory, portable contract, affected HCU accuracy, integration logic,
  patch coverage, production-boundary, compile, and whitespace checks.
- Review `base..HEAD` for generated binaries, benchmark artifacts, unrelated
  changes, credentials, and edits outside plugin ownership.
- Keep one focused branch/PR when requested. Push with existing credential
  helpers; never interpolate a user token into a shell command or remote URL.
- Report rejected candidates and incomplete model runs as explicitly as
  accepted routes. Importability and server startup are not acceptance proof.

## Failure patterns worth remembering

- A direct package export can be patched while the consumer still holds an old
  imported class or selector.
- Default AITER config lookup can fail on a new HCU even when the function is
  callable; test the actual device and include explicit-config costs.
- Packed weights make cross-backend fallback unsafe after conversion.
- Early worker INFO messages may not reach model validation logs; use an
  appropriate one-time visible marker without logging per token.
- Appended log files and old EvalScope artifacts can create false route or
  HumanEval passes unless the invocation owns and resets them.
