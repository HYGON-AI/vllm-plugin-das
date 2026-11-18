# Ordinary FlashAttention E4M3 cache reader

Keep explicit `fp8_e4m3` storage separate from all-FP8 query execution and
from E4M3FNUZ. Validate native writer bytes against a saturated independent
OCP E4M3 cast before advertising the dtype. On the observed v0.33/gfx936
vendor build, single-token paged attention accepts BF16 Q with E4M3 KV,
while paged prefill casts Q to E4M3 and rejects head size 64; its mixed-dtype
gather also failed an independent E4M3-to-BF16 equality check. These are
provider-specific observations, not a universal E4M3 hardware prohibition.

A narrow Plugin composition may keep native cache writes and paged decode,
use official `cp_gather_cache` to preserve cache bytes in both layouts, and
explicitly convert gathered E4M3 to the query dtype before vendor varlen
attention. Keep lengths/cumulative offsets on the device and bound allocation
from static metadata for Graph. Preserve key/value descales, the caller's
output buffer and any requested LSE; do not silently quantize Q, reinterpret
E4M3 as E5M2/FNUZ, or change the selected backend. Prefer deleting the
composition when the vendor provides a verified mixed-dtype prefill path.

Require non-unit scales, exact cache bytes, declared strides, prefill/decode,
changed-input Graph and applicable head sizes. A component pass still needs
the same-checkpoint BF16-KV model comparison and original HumanEval protocol;
retain raw scores and shutdown failures independently.

A KV-dtype pair on the same Plugin measures a different change from an
official-versus-Plugin pair at one fixed dtype. Retain both scopes; neither
substitutes for the other. Compare per-task pass sets as well as scores:
equal totals can hide a lost task offset by a gained task. Inspect the saved
program and isolated checker result before attributing a loss to formatting
or a kernel. Repeat the original full baseline cohort when characterizing
default-serving variation, preserving the first candidate failure. An exit
within the existing grace can still emit native leak diagnostics; record
those separately from forced cleanup and normal lifecycle qualification.

For a token-choice regression, compare both saved continuation histories with
teacher-forced prompt logprobs before assigning the cause to decode. Verify
tokenization boundaries; retokenized text is not the original emitted token ID
stream. Warmed-prefix probes need actual cache-hit counts to establish a
single-query call. Isolate a reader substitution over the same writer, typed
cache and scales, and trace its execution. A restored choice establishes reader
sensitivity, not a kernel defect, full-model recovery or accuracy acceptance;
use the same captured inputs and an independent numerical reference next.

Validate positive softcap separately from ordinary FP8 attention. A vendor
paged path can accept the argument while ignoring it. When routing softcap to
Core's official attention, cover every advertised cache dtype: preserve E4M3
and E5M2 typed views and apply their descales exactly once. The shared scale
adapter rounds query/output products to the query dtype; keep an independent
FP32 reference with unchanged tolerances. Exercise public forward dispatch,
prefill-gather on/off, non-unit scales and changed-input Graph replay in both
layouts. Core owns the softcap algorithm; a provider fallback does not prove
the vendor implements it or qualify model accuracy.

Bound temporary bytes as well as token counts. A batch-times-maximum-context
gather can reserve every short decode row at one long prefill's length; count
both FP8 gathers and both converted tensors while they coexist. Group sequences
to a byte target with at most one context as the minimum group, retain a stated
single-context bound, and account separately for query compaction/scatter.
Keep offsets on device and rebase each group's cumulative lengths. Test the
large metadata shape without allocating its old huge buffers, then exercise
forced groups, empty query rows, padding, LSE and changed query/KV lengths in
native Graph replay. A small numerical matrix alone misses this allocation bug.

Budget initialized vendor tile backing as well as logical tokens. A masked V
lane containing NaN can taint PV even when its probability is zero. Independent
canary probes on the observed provider reproduce this in changed-length Graph
replay when KV backing ends mid-tile; keep the final 256-token tile initialized
and include it in the byte bound. Poison uninitialized allocations in native
regressions so allocator reuse cannot hide the failure. A real large mixed batch
should assert both its output and a practical peak extra-device-memory bound.

Exercise both requested and unrequested LSE for single-token decode. The
observed paged decode fast path returns only output even when LSE is requested;
a generic promise to preserve LSE is not enough. For E4M3, a finite-window
fallback also re-enters the vendor path that casts Q to cache dtype. Route
explicit decode-LSE through the verified bounded byte-gather/ordinary-varlen
composition and independently check output/LSE in eager and changed-input
Graph tests; retain paged decode when no normalizer is requested.

Keep the final-source model qualification separate from component acceptance.
The [Qwen3-8B final-payload comparison](../../../docs/validation/vllm033-flash-e4m3-model-boundary.md)
records an E4M3 raw loss despite equal normalized totals and passing Graph/prefix
probes. Its unchanged full-cohort BF16 repeat preserves the failing-task boundary;
neither that repeat nor normal process exits waive accuracy or native-owner leaks.
