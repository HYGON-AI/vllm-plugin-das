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
