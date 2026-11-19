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
