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

For quantized prefill performance, inspect the public GEMM's implementation
before adding a new size fallback: it may already dequantize and call ordinary
matmul. Compare the actual compiled serving IR, since eager official dispatch
can select a different algorithm. If an optional provider stores different
packed weights, official dequantization cannot read them directly. A narrow
Plugin layout conversion can retain official matmul, but admit it from actual
repacked values, preserve unknown-layout fallback and freeze the selected
mode across reload. Validate the final installed implementation independently
of a prototype, including both packing branches and changed-input Graph replay.
Measure throughput and TTFT separately; a throughput gain can accompany a
substantial first-token regression. See
`docs/validation/vllm033-autoawq-prefill.md` for the scoped AutoAWQ example.

Compare temporary layouts using conversion plus the downstream GEMM: returning
a transposed view can accelerate conversion but select a slower BLAS algorithm.
Bind trace comparisons to matching prefill geometry; mixed prefill/decode
totals do not isolate conversion cost. Check launch traversal against physical
packing before changing the format, and revalidate both physical packing tiles.
Keep physical packing separate from launch tuning. A wider program that wins
on the largest weight can regress smaller shards; compare conversion plus GEMM
across the admitted shapes before choosing a size policy, then validate the
final installed kernel and changed-input Graph independently of the prototype.
