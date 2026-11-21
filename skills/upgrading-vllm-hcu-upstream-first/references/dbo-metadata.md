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

For serving validation, bind the actual runner, source/wheel origins, each
rank's microbatch execution, model bytes and requested attention/Graph/prefix
configuration. Keep output, HumanEval32, lifecycle and performance results
separate. Shared-device runs can verify correctness within available memory;
preserve resource-preflight failures and do not use shared-device timings as
performance acceptance. Do not claim that CPU regression success or a startup
failure in an earlier component validates the metadata fix end to end.

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
