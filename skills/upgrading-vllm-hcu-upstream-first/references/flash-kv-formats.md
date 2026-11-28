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

## Strict native PA contiguity

A newer provider may require contiguous Q, output, page tables, lengths and
K/V even though each Plugin cache page is dense. Reproduce the original model
with the requested backend, dtype, Graph and prefix before changing ownership.
Keep the official allocator and native writer. For whole-page block gaps,
read-only dense views can span intervening storage and multiply page IDs by
`block_stride / dense_page_elements`; bound each view to
`(real_blocks - 1) * ratio + 1` pages and retain its storage offset. Never
compact the full cache each decode or split the entire allocation into K/V
halves. A subpage padding gap needs a separate bounded active-context gather
path, with its scratch/copy cost disclosed.

Test actual native writer/readback, Q/output/metadata strides, both declared
layouts, multi-page indices and changed-input Graph replay against independent
numerics. Preserve output aliases and page-table inputs. Use the allocator's
real alignment: an artificial unaligned FP8 offset can fault native PA after a
successful writer sync. Retain such a failed fixture separately from the
original model failure; CPU address arithmetic does not qualify device loads.
Remove the adapter only after a replacement passes the same capability tests,
not because its full wheel version matches an allowlist.

When a remapped PA result exceeds an independent numerical gate, verify native
writer bytes and compare the same inputs with contiguous native PA. Bitwise
equality establishes layout equivalence; it does not qualify native precision.
Keep the original failure and tolerance. A bounded gather/unquantized varlen
composition can retain quantized storage for the affected geometry, with its
scratch/copy costs stated and Graph/model gates rerun separately.
