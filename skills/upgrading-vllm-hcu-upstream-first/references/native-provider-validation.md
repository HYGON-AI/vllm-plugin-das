# Validate a native provider built from source

Resolve the provider's full commit in its actual repository and freeze the build
profile, dependencies and wheel checksum. Matching installed Python files can
identify a source candidate but does not establish the original native build.
Inspect the selected packaging entry point: unified and legacy projects can
declare different versions at the same commit. Preserve the resulting metadata;
do not rename the new artifact to impersonate the installed vendor release.

Check native dependencies after installation outside the source and build trees.
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

## Packed MoE serving performance

Compare expanded INT8 and packed SlimQuant on the same checkpoint, devices,
TP geometry, Graph policy, PLE placement and KV allocation. A small native
reload/Graph test does not establish that the full expanded control fits.
Record all-rank method/storage ownership before and after measurement.

When using the target's official random serving benchmark, inspect its actual
length-range contract and retain the final requests after server-tokenizer
alignment. Match all model-input fields across arms; preserve the benchmark's
unique per-run request IDs separately. A common seed alone is not that evidence.
Warm each arm, retain prefix-reset/cache-hit state and every
measured repetition, and collect kernel attribution in a separate profiler
session. Forced output length is a synthetic performance workload, not model
accuracy evidence. Keep the original full-model lifecycle gate independent
of a clean component exit or successful benchmark.

Keep a measured packed-provider regression as a failed performance gate even
when native math and every request pass. Separate padding preparation from
vendor GEMM attribution; a missing tuned configuration warning is not proof
of the cause. Kernel timings from a separate profiler session do not replace
serving measurements. Record executor-internal worker termination as well as
controller signals and exit codes. Profiler teardown can affect lifecycle;
retain that scope instead of claiming a packed-only shutdown root cause.

If fusing padding preparation, preserve active IDs/weights and assign distinct
unused experts to padding; clamping every negative ID to zero can collide in
the vendor sorter. Select zero for wholly padded NaN/Inf hidden rows rather
than multiplying by zero. Bound any K-by-K candidate tile and whole-row hidden
tile, preserving the original path outside that geometry. Validate non-power-
of-two top-k, strided route metadata, empty batches, fallback shapes and Graph
replay that changes a padded row into an active row and back. A preparation-only
speedup does not qualify full serving performance.

Do not assume a vendor is invariant to different zero-weight padding-ID
permutations. Compare the original and fused payloads separately, retaining
which fields differ, and qualify the real provider at the relevant geometry.
Audit existing benchmark outputs as a separate functional diagnostic: preserve
every differing round even if previous rounds match. Keep a failed exact-output
gate failed while completing independent provenance/performance audits; neither
algebraic zero contribution nor successful HTTP responses resolve its cause.

When investigating that permutation, use the actual expert count, hidden and
intermediate widths and top-k with the same packed weights/scales in both arms.
Include repeated eager execution, changing-input Graph replay, mixed/all-padding
rows and an independent signed-INT4 reference for nonzero valid outputs. A
matching vendor component matrix narrows the hypothesis; it does not explain
a prior model divergence. Do not reproduce an unspecified Torch sort tie order
as a production fix without causal evidence. For the next model diagnostic,
save the actual HTTP bytes and token IDs/top logprobs, and distinguish single
from explicitly grouped requests. Label short diagnostic requests separately
from the original performance or accuracy cohort and preserve its failed gate.

Persist each child exit code and identity-bound descendant state before port
or attestation checks that can raise. If the controller loses an unsaved exit
code, record it as unknown. A later successful bind does not establish immediate
port release or exit0; retain the first refusal and resume only missing work.

## Native JIT and interrupted cold builds

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
