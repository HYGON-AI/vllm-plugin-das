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

When public responses preserve prompt and generated IDs, replay those integer
arrays directly and bind source response hashes; verify current chat rendering
still matches saved prompt IDs. Re-tokenizing answer text loses that exact
history guarantee. Keep teacher-forced and warmed-prefix geometry distinct
from the original concurrent free-running cohort.

A whole Plugin installation can also register non-attention operators. When
isolating an attention regression, an additional Plugin/official-backend
control can distinguish registration effects from the requested backend path.
Observe actual layer implementations, operator forwards, resolved layouts and
kernel traces. Fresh services may autotune identical generated reduction
source to different tiles and round differently. Retain selected tuning
configurations and compare a fresh unchanged-path repeat before assigning
a probability change to an OOT class. Narrowing a probability difference
to one service path does not
establish a defective native kernel or waive the original model loss. See the
[BF16 code-pair and owner-isolation record](../../../docs/validation/vllm033-flash-bf16-owner-isolation.md).

Validate positive softcap separately from ordinary FP8 attention. A vendor
paged path can accept the argument while ignoring it. When routing softcap to
Core's official attention, cover every advertised cache dtype: preserve E4M3
and E5M2 typed views and apply their descales exactly once. The shared scale
adapter rounds query/output products to the query dtype; keep an independent
FP32 reference with unchanged tolerances. Exercise public forward dispatch,
prefill-gather on/off, non-unit scales and changed-input Graph replay in both
layouts. Core owns the softcap algorithm; a provider fallback does not prove
the vendor implements it or qualify model accuracy.

For pretrained softcap checkpoints, inspect per-layer attention policy too.
Gemma2 alternates full and sliding layers; a full-attention synthetic fixture
does not establish that combination. Preserve the requested window and Core's
cache manager rather than removing the model's sliding configuration. Translate
a causal window W to `(W - 1, 0)` and test W=1, a boundary-crossing small window
and the checkpoint window. Cover finite windows with both zero and positive
softcap, native cache writes, FP8 scales, ragged/empty rows and changed-input
Graph against an independently masked reference. Mask support in a contiguous
vendor path does not establish paged-window correctness. Reusing official Core
attention through the Plugin boundary keeps the model and scheduler upstream.

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


For a sliding-window model, include sequence positions on both sides of the
window boundary and changed concurrent prefix repeats. Preserve exact integer
prompts and generated tokens, inspect each layer's actual window/softcap/cache
policy, and record prefix-hit counters plus actual Graph/kernel traces. A long
prompt can still reuse cached blocks; distinguish its total sequence length
from the uncached prefill length instead of calling every first-wave request
cold. State which request waves the profile captures. Short key-retrieval probes
supplement the original model evaluation and do not establish long uncached
prefill, cache reclamation or accuracy/lifecycle acceptance.

When replaying observed tuning choices as a diagnostic, retain the official
cache key/config-hash validation and verify cache hits plus actual selected
worker launchers before and after requests. Frozen objects may be absent
from GC enumeration; inspect the compiler-owned module references without
unfreezing GC or reloading kernels. For a single-choice control, verify all
other effective configurations and generated-source identities remain fixed.
Such sequential controls do not replace the original concurrent model cohort.
For a pretrained startup launch failure after successful weight loading and partial
Graph capture, retain that original failed cohort. A small probe with matching
head/GQA geometry may still pass: compare the actual failing tensor shapes,
strides, storage offsets, layout and static sequence bounds before changing the
Plugin route. A failure-only external metadata observer can preserve the original
call and exception without replacing the installed package or native library;
its run remains diagnostic, including if observation changes whether it fails.
Do not infer healthy serving, HumanEval or normal shutdown from completed model
loading or capture progress. Record controller cleanup signals separately from
framework-internal signals and the final surviving-process check.
