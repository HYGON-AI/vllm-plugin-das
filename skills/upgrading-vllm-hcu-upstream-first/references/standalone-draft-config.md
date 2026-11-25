# Standalone draft configuration through a public registry

Inspect both the target parser and Transformers registry before carrying a
Core config patch. A missing standalone `model_type` can be registered with
`AutoConfig.register()` during Adapter extension activation, before ordinary
model-config construction. Keep Adapter as the platform/general-plugin owner.
Do not replace the `hf` parser or its private registry when the public registry
already reaches the real consumer.

Use a distinct subclass of the official target config for the draft type.
Registering the same class under target and draft keys can mutate class-level
`model_type` and break later target loads. Reuse upstream validation, nested
configuration, MTP-inclusive layer arrays and quantization metadata. Preserve
explicit checkpoint values; document any standalone defaults.

Check for an existing owner explicitly. In the tested Transformers release,
`exist_ok=False` protects built-in entries but does not reject replacement of
another extra-content registration. Refuse a foreign owner before publication;
after an attempted commit or observed ownership drift, report a fatal extension
failure without pretending to roll back. Health checks must be read-only. Test
idempotence, fresh spawn and explicit adoption of an exact inherited fork
registration, including a parent lock held by another thread.

Validate the installed wheel with real entry-point discovery and Adapter API1
activation, then actual `get_config` with both `hf` and `auto` formats and
`trust_remote_code=False`. Include save/load, pickle, invalid metadata, default
and explicit MTP layers, target text/VL round trips, foreign ownership and a
fault after real registry publication. Keep discovery free of framework and
vendor imports. A configuration test does not prove model registry resolution,
tokenization, weight loading, sparse kernels, Graph or accepted draft tokens.

When the official release adds its own standalone type, reassess ownership and
retire the optional registration after equivalent behavior is verified. Do not
overwrite the new owner or keep a copied config implementation.

## Qualify the selected execution separately

After configuration resolves, inspect the official model and normalization
classes actually selected by the target. Step5's CUDA-named Triton libdevice
imports and CuTe normalization are separate compatibility boundaries. A config
registration or successful Graph capture does not establish either one.

For a shared HIP libdevice adapter, check both package attributes and
`sys.modules`: they can hold different module objects. Preserve cached module
aliases and metadata while binding real HIP exports. Previously cached
individual functions and compiled kernels require earlier activation; repairing
module attributes cannot repair them retrospectively. Validate the real host
startup order, read-only ownership checks and changed-input Graph numerics.
Keep this shared hardware adaptation in Adapter rather than copying it into
each Plugin provider.

If a CUDA-only normalization import prevents the official fallback from being
reached, establish that boundary explicitly. Reuse the target's existing math
where its semantics fit, including residual precision, zero-centered weights,
output dtype and output-buffer behavior. A process-local diagnostic shim can
test that fallback but does not deliver a production provider or establish
native CuTe support. Preserve the original failure and qualify the eventual
installed solution through the public model path before claiming support.

## Draft request buffers and dummy batches

Trace the selected runner's input residency before porting an asynchronous-copy
wrapper. In the inspected v0.33 V2 runner, both normal and dummy draft calls pass
the sampler's GPU temperature/seed buffers. A legacy H2D rationale alone does
not establish a performance gap on that path. Preserve the official preallocated
destinations and the current `dummy_run` argument: a dummy batch invalidates
every request index, while ordinary padding invalidates only inactive rows.
Blindly retaining the old five-argument replacement loses this newer contract.

Exercise the real Core helper with changing request counts, full/empty batches,
large integer seeds, caller-input reuse and stable destination addresses.
`test_draft_input_owner.py` covers those CPU buffer contracts without replacing
the helper. It does not establish async transfers, device Graph replay, accepted
draft tokens or MTP throughput; keep those gates separate before retiring any
remaining performance requirement.
# Model-family delivery

A model port can pass the active package's boundary tests while all changed
model code remains outside its wheel. Pre-merge the current target locally and
inspect the actual wheel payload, installed entry points, real framework
registry and CI selection for the new feature outside the repository. Do not
restore a retired root package or add source fallbacks to hide missing delivery.

Register missing metadata with public AutoConfig where appropriate, using a
capability named for configuration parsing. Preflight all new names before
publication; preserve official configs, refuse foreign owners and latch partial
publication failures. Test actual Core `get_config`, nested/flat multimodal
configs, serialization, geometry validation and fork/spawn discovery from an
installed wheel. Passing those checks does not establish model registration,
tokenizer/processor behavior, weights, attention, device accuracy or lifecycle.
