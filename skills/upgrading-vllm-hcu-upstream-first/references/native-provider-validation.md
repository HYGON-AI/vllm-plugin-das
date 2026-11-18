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
