# Mooncake stage correlation

Read the legacy output wrapper before treating TTFT as aggregate metrics.
Cross-stage correlation needs wall-clock timestamps, request IDs and a shared
transfer ID for producer-ready, send-start, send-done, decoder-KV-ready and
first-output events. Preserve explicit router transfer IDs before using a
completion-UUID fallback. First frontend output is distinct from client receipt.

Use a router transfer ID different from the completion UUID when validating
all five stages. A decode output may contain no transfer response metadata;
retain the original request's explicit ID in the frontend request state rather
than relying on the UUID fallback. Verify first-token correlation and unchanged
weight-version spans through the real OutputProcessor, then a public P/D run.

If P/D output differs from a cold standalone request, check the actual registered
KV bytes and compare with a local warm-prefix request using the same prompt,
backend and Graph policy. Recomputing one suffix token after a cache hit can
select different BF16 arithmetic than cold prefill. Compare returned logprobs
only at matching input histories; later divergent histories are separate inputs.
A matching local-prefix control localizes that case without waiving original
output differences, model accuracy or lifecycle gates.

Inspect the real OutputProcessor callers as well as its definition. A wrapper
can reject a new keyword such as `weight_version` even when tracing is off.
Keep first-output observation in the official owner loop or a suitable public
observer; do not add another batch traversal or copy the connector. A failed
receive must not emit successful KV-ready. Check pending peers, duplicate
responses, release-only requests and transfer failures through real constructors.

Translate a legacy tracing flag at the Adapter extension boundary before Core
caches environment values. An explicit Core setting, including off, takes
precedence. Keep the Core getter/cache and OutputProcessor methods unchanged;
late enablement should report the ordering failure instead of clearing caches.
Validate installed entry points and required-owner failures separately from
provider computation. Discovery can stay framework-free even if the actual
Core import itself loads Torch; measure new imports against the correct phase.
Exercise Core's real general-plugin loader as well as direct extension calls.
An unusable Core should produce an optional failure report or fail startup when
`HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS` requires the alias owner.

For an offline public-server regression, resolve an official model redirect
before constructing `RemoteOpenAIServer`: some targets resolve repository paths
in `EngineArgs` before reaching ModelConfig's redirect. Preserve the original
offline-resolution failure, and compare output token IDs with tracing off/on in
fresh services. Component tests and successful activation do not replace that
real-weight output regression or the separate P/D and lifecycle matrix.

Mock the HTTP client actually imported by the tested owner. A target may use
`huggingface_hub.utils.httpx` while an old fixture patches top-level `httpx`.
Retain accidental network attempts and timeout evidence before repairing the
external-service fixture. Explicit test listener/socket cleanup does not prove
normal runtime teardown. Likewise, host byte transfers and a CUDA-uninitialized
process do not qualify GPU KV or absence of HIP driver access. Keep native,
P/D, model accuracy, performance and service-lifecycle gates separate.

Include the native transport distribution in installation checks. A shared
package link can import successfully while recorded console scripts outside
site-packages are missing. Prefer installing its frozen wheel in the isolated
prefix. If only an existing installation is available, preserve its complete
original RECORD and payload hashes, copy all recorded files, regenerate declared
launchers for the new interpreter, and verify parent/child RECORDs. Describe that
as a copied installed-provider trial; regenerated hashes do not establish its
source/build provenance or qualify a release wheel.

If a native failure reports only Go `runtime.sigfwd` or a UCX signal handler,
capture the first SIGSEGV under a debugger before attributing the fault to that
handler. Reproduce the same GPU-buffer API without Core or Plugin imports and
record the actual HIP/HSA libraries. A debugger's zero exit code does not mean
the inferior completed its transfer.

For same-host HIP IPC, inspect both ROCR and HIP visibility. On the tested
DTK2610/gfx938 stack, disjoint single-device ROCR masks reproduce a first-copy
SIGSEGV in a standalone HIP program without Torch or Mooncake. A shared ROCR
list containing the two allocated peers, with per-process HIP/CUDA masks,
passes the same changed-input descriptors in direct HIP and Mooncake trials.
HIP indexes are relative to that ROCR list; verify each process's PCI bus and
device count. Treat this as a tested deployment condition for that runtime,
not a general requirement or a native source fix. Preserve the failed masks;
do not disable SDMA, change transport or transfer native-buffer results to P/D,
model accuracy, Graph, performance or service-lifecycle acceptance.
