# Compressed-tensors W4A8 scheme admission

PR #299 already provides the HCU compressed-tensors channel INT4/token INT8
owner, native checkpoint loading, AITER/Triton representation conversion,
reload/cache handling and the MTP prefix adapter. This follow-up reuses that
implementation. The older duplicate owner and loader wrappers from this branch
are removed from the final diff; their historical artifacts remain in Git.

The remaining gap is admission: native v0.28.1-dev `get_moe_method` compares the
three projections of expert 0, with an explicit TODO for other experts. The HCU
fallback previously checked only `.0.down_proj` and ignored output quantization.
The execution owner requires one supported scheme across every expert.

Keep native selection first. Only when the existing native CPU-only W4A8 error
selects the HCU fallback, check every logical expert and gate/up/down projection
before calling the existing storage constructor. Reject partly ignored experts,
unsupported formats/bit widths and output quantization. Both supported channel
group-size spellings, None and -1, remain equivalent. Future native support and
unrelated native failures retain their original behavior.

## Validation and scope

- 22 new component cases fail on the current base; all 26 pass after the guard.
- 25 existing dispatch/packing contracts pass with real CPU Torch. The broader
  fixture suite is explicitly outside this run (84 cases deselected).
- No skips in these runs. Existing Torch deprecations and the isolated harness's
  unregistered HCU-marker warning are retained.
- Actual Core source inspected at v0.28.1-dev commit
  `b95ba2d1df6d8046aa58cf83e813805312ef2884`; target-only source inspection is
  distinct from qualifying the prepared CI image's installed runtime.
- The harness uses empty Plugin package namespaces to exercise exact callback,
  registry helper and runtime functions. It does not execute Plugin discovery,
  a native kernel, Graph or a model. Current-head CI remains mandatory.

The original two-node P TP8 / D TP8 Mooncake KV transfer and accuracy acceptance
is still outstanding and tracked in the upgrade task. Prior model/Graph evidence
for #299 belongs to its recorded artifacts; this follow-up does not create a new
model score. Keep Draft until the required current-artifact gates are closed.
