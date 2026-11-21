# DBO request-phase metadata and workspace

Inspect the selected Core implementation of `_make_metadata_with_slice`.
When `CommonAttentionMetadata.is_prefilling` is present, preserve its
`request_slice`, including requests spanning microbatches. Preserve `None`
when the caller omits the field. A one-token query can still be prefill;
do not replace explicit phase flags with a query-length heuristic or relax
the hybrid backend's required-flag check.

Trace the actual consumer: Mamba's builder requires the field, while Qwen3.5's
GDN builder tolerates `None` and clears its first-chunk-prefill flags in that
case. Observing missing flags in a Qwen run does not reproduce Mamba's
required-field assertion. Cover the consuming backend's actual behavior.

This is Core's slicing algorithm. Adapter owns hardware admission and Plugin
owns optional consumers; neither should copy the whole algorithm to repair
one dropped field. At OpenDAS v0.33 `c440c08`, MRV1 omits this field. The public
upstream [PR #54344](https://github.com/vllm-project/vllm/pull/54344) includes an
equivalent fix. Check its current status before opening a duplicate upstream
request; a private release backport still needs target-bound validation.

Use real metadata constructors, `split_attn_metadata`, and the downstream
`split_decodes_and_prefills(..., treat_short_extends_as_decodes=False)`.
Cover mixed prefill/decode requests, a request spanning two slices, explicit
flags and the optional `None` contract. Run the regression on the unmodified
base and candidate; existing bounds/position tests can pass while flags are
lost. Verify the installed wheel as well as any explicitly identified source
overlay.

At this target, DP1 skips microbatching. MRV2 slices its request flags separately
and rejects hybrid+DBO; a hybrid MRV1 reproducer needs an admitted DP2 topology,
normally with EP2 for MoE. A configuration constructor does not prove model
admission: on the observed DTK26.10/gfx938 device, the default ROCm communication
reservation is 64 CUs and the device has 64 CUs, so the SM-control constructor
asserts before metadata slicing. An explicit `VLLM_DBO_COMM_SMS=0` experiment
does not qualify the unchanged default or establish a performance policy.

## Bound-device communication budget

Read the installed Core's effective default and the already-bound worker's
actual CU count. A default that exhausts that device can be an admission
failure even when the selected communication manager's SM setter is a no-op.
Do not infer the budget from a frontend's first visible GPU or copy a release
constant. Adapter owns a narrow configuration adaptation; Plugin should not
copy `SMControlContextManager` or the model runner to solve this failure.

For an affected implicit default, Core's supported zero-reservation path can
restore admission without choosing a speculative performance budget. Preserve
explicit official inputs, including their normal parser failures. Check
backend-specific caps and ROCm DeepEP HT's existing zero-reservation override
before adapting another backend. Public upstream
[PR #68535](https://github.com/vllm-project/vllm/pull/68535) proposes a CUDA AgRs
rule; verify platform guards and the selected target before treating it as an
HCU fix. Zero reservation does not disable DBO or establish overlap performance.

If Adapter scopes the official environment input, cover the actual controller
construction points: this release uses V1 model loading and V2 KV
initialization. Restore the previous environment on exceptions and between
models. A cached stale Core value must fail clearly; clearing Core's cache is
not validation. Test real configuration constructors, the bound-device
controller and the installed model separately. Compare stock and automatic
policy with the official variable absent; record an explicit-zero control as
a different configuration. Retain independent accuracy and normal-exit gates.

For serving validation, bind the actual runner, source/wheel origins, each
rank's microbatch execution, model bytes and requested attention/Graph/prefix
configuration. Keep output, HumanEval32, lifecycle and performance results
separate. Shared-device runs can verify correctness within available memory;
preserve resource-preflight failures and do not use shared-device timings as
performance acceptance. Do not claim that CPU regression success or a startup
failure in an earlier component validates the metadata fix end to end.

## Metadata reuse within each microbatch

Inspect the runner's metadata cache as well as the slicing helper. A backend
that supports `update_block_table` may preserve query offsets, sequence lengths
and scheduler metadata while replacing only block tables and slot mappings.
Reusing that object across microbatches mixes one batch's sequence metadata
with another's KV addresses. Scope compatible-group reuse by microbatch ID;
preserve the unsplit `None` identity. This generic runner contract belongs in
Core, without copying the runner or changing the attention backend in
Adapter/Plugin.

Public [PR #48659](https://github.com/vllm-project/vllm/pull/48659) proposes this
cache-key fix. Check its current status and the selected release's actual
implementation before backporting; an existing upstream request is not a
reason to create another public duplicate. A hybrid model's GDN builder may
disable block-table reuse while its full-attention builder enables it, so
trace each actual builder rather than attributing the failure by model family.

Observe expected and returned request counts, token counts, query offsets and
sequence lengths for each microbatch. Record pointer identity without a device
copy where the backend contract preserves those tensors. Phase flags and
workspace capacity can pass while this separate cache contract corrupts
outputs. Concurrent HTTP calls alone do not prove microbatch execution:
client construction and request arrival can serialize a small test. Confirm
the failing path in the unmodified base and correct output after the fix.
Preserve unsuccessful reproducers rather than treating a passing base as RED.

Keep protocol, request and problem bytes equal in accuracy comparisons. An
output containing only punctuation until the fixed token limit is corruption
evidence, not a reason to increase the budget. Saved text does not establish
NaNs, generated token IDs or logits; collect those separately if needed.

## Unsplit MoE workspace after microbatched warmup

Trace DP coordination and the selected dispatch backend separately from phase
metadata. A single long prefill can leave a peer rank running a one-token
dummy batch. If coordination rejects microbatching, all-gather dispatch sees
the sum of those unequal token counts. That unsplit expert input can exceed
the high-water mark from a split maximum-token warmup, even though each rank
stays within its own scheduler budget.

For the affected MRV1 path, Core's `GPUModelRunner.profile_run` must cover the
unsplit maximum-token geometry before retaining its normal microbatched pass.
Both belong inside memory profiling, before Graph capture and workspace
locking, so automatic KV sizing accounts for the reservation. Do not
unlock captured storage, grow locked buffers, add an arbitrary margin or copy
the runner into Adapter/Plugin. Check workspace slots separately; increasing
slot zero does not prove the other microbatch slots are reserved.

Public [PR #66633](https://github.com/vllm-project/vllm/pull/66633) proposed this
profiling order but closed unmerged after newer main removed MRV1 and rejected
hybrid DBO. The private v0.33 `c440c08` retains the affected runner. Recheck the
actual release implementation and current upstream status before deciding to
backport. Public [PR #61584](https://github.com/vllm-project/vllm/pull/61584)
addresses peer attention scratch; it does not establish the unsplit MoE bound.

Use a real server regression with a prefill longer than the token budget and
an idle DP peer, preserving the requested backend, Graph and prefix settings.
Confirm the failure before the fix, correct output and cache hits afterward,
then run the fixed accuracy and lifecycle protocols. Real constructor CPU
tests cover metadata and coordination; mocked runner-call counts do not prove
workspace capacity. Check CI runner selection as well as file selection:
this target normally falls back to MRV1 for DBO unless MRV2 is explicitly
enabled.

When observing the allocator, record actual gathered rows, dtype, shapes,
alignment, required bytes, slot capacity and locked state. Track warmup and
locked serving observations separately: a global maximum-only observer can
hide all serving requests after a larger warmup. An HTTP success or absence
of an allocation exception alone does not qualify full DBO, overlap
performance, default communication-CU policy or normal service exit.
