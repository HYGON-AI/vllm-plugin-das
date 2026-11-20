# Validate a native provider built from source

Resolve the provider's full commit in its actual repository and freeze the build
profile, dependencies and wheel checksum. Matching installed Python files can
identify a source candidate but does not establish the original native build.
Inspect the selected packaging entry point: unified and legacy projects can
declare different versions at the same commit. Preserve the resulting metadata;
do not rename the new artifact to impersonate the installed vendor release.

Check native dependencies after installation outside the source and build trees.
Read the actual worker's architecture, UUID and PCI identity after visibility
filtering before choosing a hardware gate. A host's old architecture label is
not evidence for its current allocation. Preserve an architecture-preflight
refusal; a later run explicitly targeting the observed supported architecture
qualifies only that device, without broadening the original gate or result.
A wheel repair can bundle ordinary libraries while leaving DTK/HSA or network
libraries external. Record those runtime requirements separately from the wheel's
platform tag. Verify the full installed RECORD, wheel payload and actual parent
and child module/DSO origins; build success alone does not qualify the artifact.

Preserve the original packaging regression when it fails. If a facade assertion
conflicts with an intentional private helper, inspect the actual callers and
dedicated facade tests before changing runtime behavior or the assertion. Check
attribute access, public exports and backend-specific rejection separately. A
passing diagnostic does not make the unchanged packaging regression pass.

Run source fixtures that require a missing native extension in an environment
without the installed provider. A namespace package using `extend_path` can find
the installed extension even when PYTHONPATH points at source, invalidating that
negative control. Use a separate installed-wheel environment for native tests;
retain initial environment failures and bind each result to its actual source.

Use existing transfer tests where possible. Native TCP read/write, async/scatter
and GIL tests establish host transport behavior only. HIP IPC, device registration,
RDMA, P/D model output, Graph, accuracy, lifecycle and performance retain their
own applicable gates. Record owned-process exits and listener cleanup separately
from numerical assertions; retain an initial bind refusal and any later recovery.


For a default-OFF native optimization, trace real activation and the ordinary
model path as well as the enable switch: registering an OOT class or fake/custom
op need not import its vendor DSO. Preserve official fallback, compile-cache
factors and explicit required-capability admission while deferring an unused
provider. Required extensions use published entry-point names, which can differ
from declared spec names; exercise the real host activation with that metadata.
Reuse Adapter's process-local lifecycle for serialized first-use ABI checks
and fork refusal; a health checker reports failures without initializing
or repairing the provider. Verify enabled ordinary/fused execution and changing
input Graph replay after an installed-wheel binding check. Count shutdown leaks
and the original cleanup deadline separately: fewer imports do not establish a
clean exit or a speedup. The Plugin RMS example is documented in
`docs/validation/vllm033-rms-lazy-provider.md`.

Before carrying a constructor patch that forces a native operator, inspect the
target's dispatch and precision guards. For clamp SwiGLU, current v0.33 Core
already selects its ROCm kernel for default alpha/beta and retains its Torch
path for nondefault parameters. Preserve that policy and official custom-op
enablement rather than forcing every configuration through the kernel. Reuse
upstream eager tests and add changing-input Graph coverage on the real class;
see `docs/validation/vllm033-official-clamp-swiglu.md`. Component ownership does
not qualify a checkpoint, HumanEval, performance or whole-model shutdown.

## Cross-framework GDN verify contracts

For Adapter's GDN target-verify bridge, describe the upstream and vendor
signatures independently. An upstream-only `lower_bound=None` may be omitted
from the vendor call after reviewing its meaning. Every non-None request,
including zero and `False`, must retain the original upstream semantics;
do not silently discard it. Unknown added parameters still reject publication.
Preserve the original callable object, signature and previously cached aliases.

Inspect the actual native gate strides. In the observed AITER verify kernel,
`a` has no token/head stride and `b` has no head stride. Materialize interleaved
gate views without changing their values; retain the existing Q/K/V stride
contract. Check independent FP32 recurrence, snapshot slots, unchanged initial
state and changing-input Graph replay. Successful binding alone does not
establish numerical compatibility.

Keep explicit-provider numerical results separate from automatic admission.
A passing explicit suite on a new stack does not qualify that stack's default
selection or skipped automatic-admission tests. Extending admission still needs
its actual eager/Graph/model evidence; do not loosen the table to make skips pass.

## Native JIT and interrupted cold builds

An isolated child can import Torch and native DSOs while its compiler cannot
find device bitcode because the launcher dropped the toolchain roots. Preserve
and record the actual validated `ROCM_PATH` and `HIP_PATH` in compiler children;
derive paths from the current DTK installation rather than an old example.
For clang's missing-ROCm-device-library error, inspect those paths and the
resolved bitcode directory before changing attention math or providers. Retain
the failed run, then rerun its unchanged native matrix after repairing the
launch environment; CPU import success does not replace that execution.

When a vendor builds after a native-module miss, compatibility stubs must not
report a missing native module as successfully imported. Verify real discovery,
the missing-module exception, the vendor build and native execution separately.
Use privately owned package and cache directories for default JIT writes;
checking wheel RECORDs does not identify newly generated native artifacts.

After a cold-build timeout, a live retry can be waiting on an abandoned
FileBaton lock. Inspect its actual stack, compiler descendants and artifact
progress rather than inferring compilation from a live parent or more jobs.
Preserve the timeout and waiting evidence. Before recovery, confirm all users
of that private cache have exited, bind the lock path/inode/time to the original
run, and preserve the old lock in a new run's evidence. Do not clear shared or
another client's locks. An interrupted run retains its failed lifecycle result;
cache recovery does not retroactively qualify its tests.

KFD sysfs can expose processes from another PID namespace. A nonempty global
KFD list alone does not identify this run's surviving owners. Compare the saved
PID/startticks ledger with KFD `vpid`, per-GPU queue IDs and VRAM counters before
attributing a resource refusal to model teardown. A zero `vpid` identifies a PID
invisible in the reader's namespace. Keep the original refusal and completed
answers; score those answers separately when device progression has stopped.

## Varlen FlashAttention and allocation-sensitive failures

When identical `cu_seqlens_q` and `cu_seqlens_k` values produce different native
results depending on pointer identity, compare shared and separate buffers in
fresh processes while always allocating both buffers. This controls allocation
count as well as metadata values; retaining or synchronizing a temporary tensor
alone does not establish a lifetime bug. Preserve the original separate-buffer
case rather than forcing aliasing to pass.

To probe a suspected sequence-boundary read, preserve logical Q/K/V and metadata
while varying only allocated storage after the logical K/V end (zero versus NaN,
K and V independently). Compare with an independent FP32 attention reference in
eager and changing-input Graph replay. A result that depends on this tail
localizes a native memory-boundary problem; it does not prove which source line
is defective, or establish FP8 KV support.

Resolve the installed provider's real source revision before attributing the
failure to a current vendor branch. A masking-loop bound found in different
source remains a candidate until that source is built and the original failing
cases pass on the device. Avoid an Adapter/Plugin padding or metadata-alias
workaround that conceals a native out-of-bounds read; keep provider ownership and
unverified capability explicit.

## CPU import probes and device namespaces

Separate framework management queries from provider computation when isolating
CPU contracts. In the inspected v0.33 Core, importing `RocmPlatform` queries
architecture through AMD SMI and falls back to Torch device properties if that
query fails. Hiding every device node can therefore fail collection before a
selector or provider is tested. Check the actual management query and Torch
initialization state before choosing the namespace; preserve a real architecture
query rather than inventing architecture or availability results.

Vendor imports have separate requirements. The observed DeepGEMM import runs
`rocminfo` for a CU count and fails on `None` when its device namespace prevents
that query. A parsed count alone does not identify a GPU or prove usable kernels.
Keep that failure and its original JUnit. If matching production CI's namespace
repairs the environment, rerun the unchanged affected contract with real provider
imports, empty compute masks and a Torch-initialization guard where applicable.
Reuse unaffected completed cases only with unchanged source, dependencies and
configuration evidence. If combining JUnit reports, label the derived report,
retain each original report and verify the exact case multiset. CPU import/ABI
success does not qualify native numerics, Graph, model accuracy or lifecycle.

If an installed contract forwards the parent's `sysconfig.get_path("purelib")`
to a child, use an environment whose real prefix and purelib contain the
candidate distribution. A `--target` install plus a `sys.path`/executable wrapper
does not change sysconfig; the observed Step5 probes therefore imported an older
global Plugin. Preserve the original child interpreter flags and verify its
actual module origins after repairing the environment. On Python 3.12, `-S`
can retain the base prefix even for a venv executable, so prefix alone is not
proof of which payload ran.

Check conditional vendor exports before attributing a CPU activation failure to
the Plugin. The inspected FlashAttention package exports its native callables
only when `torch.cuda.is_available()` is true at import. Empty compute masks
leave the package discoverable without `flash_attn_varlen_func`. Preserve that
failure and qualify the unchanged activation on a real supported device; do not
fake availability or inject the missing callable to claim host activation.


## Derived audit snapshots and model attribution

A derived review may change when another arm finishes, even when raw inputs
do not. Retain the exact derived bytes in an immutable content-addressed snapshot
before updating a latest-view pointer; publish that snapshot path and digest.
If an earlier published view was overwritten without retention, disclose that
missing derived artifact and re-audit the preserved raw inputs. Do not fabricate
historical bytes or claim the earlier digest identifies the new review.

Compare a numerical regression within the actual changed execution branch.
A no-Plugin official baseline versus vendor attention can identify a lost answer,
but does not isolate a softcap/window-only change on a full-window, zero-softcap
checkpoint. Preserve that broader failure and use matched implementations or a
checkpoint exercising the changed branch before assigning causality.

For differing greedy outputs, preserve the failing cohort and repeat the
unchanged merged baseline in a fresh service with the same physical device,
requests, concurrency, default Graph/prefix and private cold caches. Compare
per-task pass sets and messages, not only total scores. Distinguish a repeated
failure from reproduction of the exact candidate continuation: equal per-task
pass flags can accompany different texts and failure mechanisms. A repeated
baseline failure narrows attribution; it does not explain every change or
waive the original loss, lifecycle or performance gates. Do not treat two
absent generated-token fields as token parity; obtain actual integer token
lists in a separately labeled diagnostic when the target API supports them.

Before a public backend-selection probe, check the actual platform and extension
publication. HCU admission requires real available HIP device evidence; hiding
every device can legitimately leave the official ROCm platform selected even
when the Plugin is installed and required names are set. Preserve that failed
probe. A separately labeled real-device configuration probe may initialize HIP
and check the genuine public selector; do not fake detection, replace the
platform or count that probe as CPU-only or pretrained-model qualification.

Retain each run's live Inductor/Triton tuning artifacts when compiled reductions
can differ. Compare selected launch/reduction parameters separately from tuning
time, source/debug paths and build hashes. Different cold-cache choices are a
lead, not proof of numerical causality; equal choices do not prove runtime
equivalence. Any experiment pinning or transferring choices is diagnostic and
does not qualify the requested default tuning policy.


## Expected outputs in paired probes

When a frozen probe artifact defines an expected value, check each arm against
that value as well as comparing the arms. Two implementations can return the
same wrong answer. Preserve literal text equality separately from any explicitly
labeled whitespace-only comparison; a correct extracted key does not pass a
failed strict-format probe. Keep expected-value checks in the derived audit
when the original generation client is frozen, without changing its requests
or regenerating answers.

For a real windowed checkpoint, map every actual worker attention layer back
to its original configured layer type before and after generation. Test input
lengths on both sides of the window boundary, changed batches and prefix reuse;
retain literal expected answers and actual integer token lists independently
from whitespace comparisons and aggregate scores. A pipeline text encoder may
need its bundled tokenizer and the target's official text-only modality limit;
retain the original architecture and checkpoint configuration.

Audit worker GPU traces as well as capture counters: bind HIP graph launches
to the actual worker PID and correlate them with GPU kernel events. Reused
capture counts, backend selection and HTTP success alone do not establish
replay execution. Profiling changes timing, so these traces are execution
evidence rather than serving performance acceptance. A BF16 window-model pair
does not qualify FP8 or softcap; a BF16/FP8 comparison changes KV precision and
must be labeled accordingly. Keep raw fenced-code failures separate from
isolated format-normalized scores even when both backends produce identical
messages, tokens and pass flags.

Compare per-task pass sets even when the aggregate raw score improves: newly
passed tasks can conceal a lost reference task. If fenced output fails raw
scoring, compare the normalized programs too. A changed AST or added branch
prevents attributing the whole difference to formatting; equal normalized
pass flags do not prove equivalent programs or waive the original raw gate.


## Softcap checkpoints that expose embeddings

A pipeline text encoder can exercise real attention softcap and alternating
windows through the official pooling conversion without providing generation
weights. Preserve its original architecture, tokenizer and per-layer policy;
embedding coverage does not replace the required generation/HumanEval gates.
For vector comparisons, retain every coordinate, repeat identical requests
across the selected batches and prefix reuse, and compare repeat stability in
each arm separately from cross-backend equality. Freeze the metric and criterion
before running. If an official reference also fails exact repeat equality,
retain that failure and quantify it; do not relax the criterion afterward or
attribute the entire difference to the Plugin. Verify actual Graph kernels and
source/payload identity independently. Clean process exit can coexist with
binding leaks or engine errors and does not qualify the complete lifecycle.

## Inverse-RoPE rounding boundaries

Compare the target Core's emitted instructions, not only its Python arithmetic.
One DTK 26.10/gfx938 Triton build rounds the sine product before fusing the
cosine product; another supported installed build emits separate products
and sum even with `enable_fp_fusion=True`. Do not transfer either contract
across installations or infer it from that compiler flag or a wheel string.
At provider preparation, compare both native modes with the actual Core helper
using distinguishing cancellation inputs and both position dtypes. Admit only
one consistent match; reject unknown or ambiguous behavior before execution.
Bind the selected mode once, before Graph capture; forward and health checks
must not rerun calibration, initialize vendors or change the selection.
Keep the original failed gates and the old installation's results separately.
Freeze cancellation witnesses with independent FP64 oracles for both modes,
retain exact eager and changed-input Graph assertions, and cover both position
index dtypes. This component result does not qualify a model or other builds.
PCP forward Q shares the native layout but requires separate FP32 products;
validate both users together rather than changing the shared rounding globally.

## Shared dependency ownership

When a prepared prefix intentionally shares framework dependencies through
symlinks, resolve RECORD paths from the actual distribution metadata directory.
Relative console paths belong to that owner, not automatically to the new
prefix. Keep physical owner seals and shared dependency checks separate; do
not relocate or repair a shared installation to satisfy the audit. Comparing
current shared bytes cannot retroactively establish complete dependency seals
for historical cohorts. Record that limit alongside each pre/post owner seal.
