# v0.33 GDN/FLA leaf migration candidate

The production `adapter_extension` previously provided no replacement for the
three legacy FLA leaves or two Qwen GDN decode functions. This candidate adds
an independent, explicitly requested `hcu-gdn-fla` extension. It keeps Core's
complete algorithm and its constructor's final backend dispatch.

## Ownership and temporary seam

Core v0.33 exposes whole-op/layer OOT registration but no FLA leaf registry.
The Plugin therefore replaces only three aliases in
`vllm.third_party.flash_linear_attention.ops.chunk` and two aliases in
`vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn`. Existing pipeline
functions and model methods resolve those same module globals. Canonical leaf
modules and unrelated consumers remain official.

This is a temporary consumer alias seam. Migrate to Core's public leaf-provider
registry when an interface covers the required metadata, state and output
contracts. Do not copy the complete pipeline or reactivate the legacy root.
The Plugin neither registers extra official backend names nor modifies the
official prefill selector, convolution, recurrent state allocator, weight
loader, RMSNormGated or fused post-convolution branches.

## Selection and contracts

Request capability `hcu-gdn-fla`. The effective policy is the conjunction of
`VLLM_HCU_USE_CUSTOM_OPS` (default OFF) and the two stage selections:

| Setting | Values | Default |
| --- | --- | --- |
| `VLLM_HCU_GDN_PREFILL_BACKEND` | `triton`, `hcu-boltops`, `hcu-aiter` | `triton` |
| `VLLM_HCU_GDN_DECODE_BACKEND` | `triton`, `hcu-boltops`, `hcu-aiter` | `triton` |

The official `gdn_prefill_backend` remains authoritative. A whole-op backend
such as `aiter_flydsl` can bypass the prefill aliases. Optional fused decode
branches can bypass the two decode aliases. Activation is not execution proof;
qualification must establish the actual route for the selected model.

Before publication, all five Core signatures and selected provider signatures
are checked, including parameter defaults and synchronous execution kind.
Missing packages or symbols follow AITER → BoltOPs → official, or
BoltOPs → official. Missing internal dependencies and incompatible installed
ABIs are errors. Native execution errors propagate after selection.

Adapters retain the official metadata, `[N,H,V,K]` state layout, initial/final
state, `save_new_value` and output backing. BoltOPs receives explicit
`null_state_index=0`; AITER decode converts indices ≤0 to -1 without mutating
the original index tensor. AITER chunk-H receives int64 offsets. AITER has no
standalone recompute entry, so that leaf uses BoltOPs or official.

Gated AITER chunk-O remains excluded because earlier device evidence showed
numerical divergence. AITER-incompatible inputs use the next prepared route.
The recorded exact BF16 chunk-H `(Hk,Hv,K,V)=(4,12,128,128)` retains official
execution on the BoltOPs route. These constraints do not qualify other shapes
or promote a provider default.

All prechecks finish before any alias changes. Partial publication, reentry,
or late policy/owner drift permanently refuses the process. A complete
inherited publication requires fresh child runtime activation; a fork during
preparation/publication refuses. Adapter owns runtime admission and the serving
health boundary, including checks before cached/Graph execution. Python alias
guards alone do not run inside an already captured device Graph.

Effective routes and adapter/provider source identity enter official normal
and AOT compile factors. This is a source identity, not an exact native wheel
pin or a replacement for the providers' native/JIT cache ownership.

## Validation and remaining gates

Source CPU qualification with real Core `98a0985` and Adapter `719e203`:
**36 passed / 0 skipped** across `test_gdn_leaf_providers.py` and
`test_package_boundary.py`. This includes real Core aliases and constructor
dispatch, strict provider call contracts, NULL and speculative metadata,
output backing/padding, native-error propagation, publication failure,
owner drift, reentry and actual fork tests. Provider doubles do not prove
native kernel numerics. Earlier failed test-fixture runs remain in the task
evidence rather than being relabeled as device failures.

`test_gdn_leaf_hcu.py` provides an opt-in complete official FLA dataflow test:
legal WY intermediates, real gates, head ratios 4/8 and 4/12, nonaligned varlen,
initial/final state, padded output backing and three changed-input Graph
replays. Run each provider selection in a fresh process with
`HCU_V033_TEST_GDN=1`. It requires at least 1 GiB actually free on the selected
device and has not yet been executed for this candidate.

Installed wheel qualification, native prefill/decode, provider performance,
applicable hybrid model accuracy/HumanEval32, MTP and serving Graph reachability
remain gates. Existing legacy-provider and BF16 Qwen3 results do not certify
this candidate. Do not mark this migration complete or merge it on CPU
contracts alone. DeepSeek V4.1 remains user-deferred.
