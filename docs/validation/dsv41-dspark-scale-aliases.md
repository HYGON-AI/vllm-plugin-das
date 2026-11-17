# DSV4.1 DSpark Channel-FP8 scale alias loading

## Scope and anchors

Start: Plugin `0fd8511`, Core read-only `368701db06`, Adapter read-only
`3257a934`. Historical references: `17c91bab` and `cc1b9a7d`.

The actual Core V41 AMD draft loader is
`vllm.models.deepseek_v41.amd.dspark.DSparkDeepseekV4ForCausalLM.load_weights`
(`dspark.py:405-517`). It enumerates parameters at 439, translates `.scale`
at 451-458, and indexes exact names after expert/stacked/direct mapping at
477/499/508. Unlike target loading, this loop has no inverse-scale resolver.
The V41 draft architecture is `DSparkV41DraftModel` (`registry.py:677-680`);
Core normalizes to this name in `speculative.py:1552-1567`. Public
`ModelRegistry.register_model` supports replacing its lazy binding. No Core
class method or private oracle is to be rebound.

The Plugin subclass inherits native construction, forward, mapping and loading.
For the distinct HCU Channel-FP8 config only, its public parameter enumeration
provides missing scale spellings during the native load; synthesized returned
names are canonicalized. Real parameter names win; no alias parameter objects
are registered. Ordinary and fused Linear scales must belong to a CT W8A8 FP8
scheme; expert scales must belong to the CT W8A8 FP8 MoE method. Other scales
and quantization configurations keep native behavior. Native target embedding/
head sharing and absent-confidence-head handling remain native.

HCU activation preflights the stock or same-owned V41 draft entry, publishes
through the public registry, and includes its identity in existing health/fatal
activation semantics. The entry name is shared with native V41 drafts, so other
quantization configurations must delegate unchanged. Non-HCU routing activation
does not replace the model entry. Registration is lazy; discovery imports no
framework/vendor/model implementation. No new README or packaging entry point
is needed.

## Failure modes before implementation

| ID | Failure | Verification boundary |
| --- | --- | --- |
| D01 | A correct helper never becomes the selected draft class. | Use actual public model resolution after activation; verify only V41 draft binding changes. |
| D02 | Ordinary/fused `.scale`, direct or inverse spellings miss CT parameters. | Native loop and real CT Linear parameter/shard loaders; compare exact loaded bytes to native canonical loads. |
| D03 | Expert aliases lose expert IDs, shard IDs or success behavior. | Real native expert mapping/loop, CPU expert destination callback, exact fused scale slices and canonical result set. |
| D04 | Alias registration creates duplicate parameters or stale tensors. | Compare real enumeration/state keys and object identities before/after load; aliases are lookup-only. |
| D05 | A real direct/inverse pair is overwritten by a synthetic alias. | Both native names win and load independently. |
| D06 | Returned synthesized names fail canonical accounting. | Returned loaded set contains actual loaded parameter names; shared/unloaded placeholders remain absent. |
| D07 | Exceptions or repeated calls leak alias state. | Raise from a weight iterator, re-enumerate canonical names, then retry a valid native load. |
| D08 | A native/non-CT scale is accidentally remapped. | Noncustom config delegates; non-CT and unknown names retain native missing-key behavior. |
| D09 | Alias state leaks to other threads or concurrent load mutates the same model. | Other-thread enumeration remains canonical; reentrant/concurrent same-model load is rejected without changing parameters. |
| D10 | Confidence/head/markov/non-mtp logic is copied incorrectly. | Inherit native loop; exercise confidence presence/absence, Markov names and ignored target weights. |
| D11 | Repeated activation changes ownership or loses failure semantics. | Public activation idempotence, foreign owner refusal before quant publication, lost owner/partial publication fatal and sticky. |
| D12 | Preparation imports draft/GPU/vendor code. | Registration retains lazy module/class descriptors; factory remains framework-free in existing contracts. |
| D13 | Unreviewed native loader signature silently runs with a stale view. | Reject mismatched inherited load_weights/named_parameters signatures before model use. |
| D14 | CPU replay is claimed as GPU/complete DSpark acceptance. | Record paths, SHAs and no CUDA initialization; distinguish TP/expert/kernel doubles and unrun runtime. |

The replay is written before production implementation and will be fixed before
the first production edit. Its alias differential uses 200 deterministic seeds
with native canonical loads as reference. It must import the candidate from the
production module after implementation. This is CPU source/contract integration,
not full-model or device E2E.

## Results

The complete pre-implementation replay recorded **7/17 passed, 10/17 failed**
(`red-final/`). After importing the production subclass through actual public
model resolution, the unchanged replay recorded **17/17 passed** (`green/`).
Its 200 deterministic seeds each compare three scale spellings against a native
canonical load: 600 candidate tiny-draft loads and 200 native reference loads.
All parameter tensors match byte-for-byte, with matching dtype/shape/device and
canonical returned sets. No cases were added after production implementation.

The existing target/zero-row replay passed **25/25** (`target-regression/`). The
unchanged portable suite passed **312 tests, 66 skipped, 14 warnings in 191.15
seconds** (`regression.log`, `regression.xml`). AST, shell syntax and diff checks
passed. Existing unit files/assertions were not modified.

Evidence root:
`tests/vllm033/.cache/dsv41-dspark-scale-aliases/`. The wrapper below records actual
source paths, three SHAs, source hashes, installed metadata and each result, and
saves `report.json` plus `replay.log`:

```bash
VLLM_TEST_PYTHON=/path/to/python3.12 \
  bash tests/vllm033/run_dsv41_dspark_scale_aliases_cpu.sh /path/to/output
```

Run as the checkout owner, with Core and Adapter as sibling checkouts (or set
`VLLM_TARGET_ROOT` and `HCU_ADAPTER_SOURCE`). The wrapper empties all GPU visibility
variables and disables model downloads. The existing bw47 container
`vllm_032_1030_laibao` was reused; no container or package was created/installed.
Observed installed vLLM metadata in this run is
`0.32.0rc1.dev944+g368701db0.rocm72`; do not infer an installed-v0.33 deployment
from its source SHA. The `_C_AVX2` import warning is preserved. CUDA remained
uninitialized in all replay reports.

## Delivery and remaining limits

- Production: existing `dsv41_activation.py` plus a lazy registration helper and
  `models/dsv41_dspark.py` subclass (with a framework-free models namespace).
  Public model handoff is only for `DSparkV41DraftModel` on HCU. Adapter itself
  verifies its HCU platform before extension activation at
  `hcu_adapter/bridges/vllm.py:365-378`.
- The native class's constructor/forward/load methods and Torch enumeration
  function identities remain unchanged. The Plugin overrides its own load and
  enumeration methods; there is no legacy PATCH_ID, dispatcher callback,
  monkeypatch or mirrored constructor/loader loop.
- CT ownership checks limit aliases to the selected FP8 Linear and FP8 MoE
  methods. Same-model overlapping/reentrant loads and foreign instance
  enumeration overrides are rejected. Other threads/processes enumerate real
  names; ordinary loads have no persisted alias state.
- Registration assumes serialized plugin startup. The public registry has no
  cross-owner atomic reservation; external concurrent writers are not covered.
  Foreign owners are refused, and later ownership loss/partial publication is
  fatal and sticky. No automatic repair/rollback is claimed.
- CPU fixtures use actual native draft mapping/loading, CT Linear parameters
  and merged loaders. Hardware/TP/kernel selection and expert destination
  callbacks are doubles; the native model constructor/forward and real expert
  backend are not executed. No model accuracy/performance, GPU kernels, Graph,
  distributed progress, full checkpoint or gfx938 acceptance is established.
- Canonical results cover actually loaded draft parameters. Target embedding/
  lm_head placeholders are not falsely marked loaded. Full strict draft weight
  tracking still has the native timing of target-module sharing after get_model;
  this alias adapter does not certify or change that broader lifecycle.
- Core and Adapter sources stay read-only. This increment does not establish
  complete DSpark inference support or migrate its DP/capacity/communication work.
