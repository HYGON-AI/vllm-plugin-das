# Channel-FP8 draft Q/KV scheduling

The target's FP4 attention already splits the verified native Q RoPE and KV
insertion. Native draft attention still used the horizontally fused kernel;
the v1 serving trace measured three such draft calls at 320.956 us in total.

Reuse that existing Q/KV schedule in the owned Channel-FP8 draft constructor.
Bind only exact native attention instances whose method has not been replaced.
The common native class and all other instances stay unchanged. The runtime
eligibility, installed-provider qualification and original-method fallback are
shared with the target. Unsupported layouts, topology, metadata and token
counts retain the native method. Context-KV precomputation stays native: its
Q-normalizing kernel has a distinct rounding contract.

The shared helper is extracted from the target method without changing its
eligibility or arithmetic. There are no new knobs or assumptions about the
number of draft layers, request length or batch size.

On nmz30/gfx938, 33 device checks passed against the actual native draft method:
M=1/7/17/35/42/64/129/257, pages32/64, full/partial slot maps, masked slots,
all untouched cache bytes, input immutability, profile metadata, idempotence
and foreign-method preservation. Three graph replays per case change Q/KV,
positions, slot maps and sin/cos values. Both Q and complete KV cache bytes
match exactly, including the native fallback beyond the split range.
Six shared-consumer constructor checks also pass after the additional hook.

The first test attempt assigned a read-only native property in its fixture;
the corrected fixture uses the actual property. No numeric tolerance changed.
Whole HumanEval164, actual dispatch traces and paired serving performance are
still required before claiming a model-level benefit.
