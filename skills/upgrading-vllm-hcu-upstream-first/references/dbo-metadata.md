# DBO request-phase metadata

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
