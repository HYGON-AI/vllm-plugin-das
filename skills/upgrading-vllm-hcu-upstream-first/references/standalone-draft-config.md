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
