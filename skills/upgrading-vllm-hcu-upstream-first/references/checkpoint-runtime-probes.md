# Checkpoint recovery and runtime inspection

Use these checks when a model download stalls or a live worker needs an
execution-path probe. They preserve the evaluation's checkpoint and runtime.

## Recover the same checkpoint

Resolve the requested upstream revision and its required assets before using
another mirror. Match every shard's size and upstream LFS SHA256; verify the
configuration, tokenizer and index against upstream Git blob IDs. A matching
filename or size alone does not establish checkpoint identity.

Check the download controller's PID/startticks and open files before assigning
another writer. Preserve failed partials and receipts. Reuse a verified prefix
through a separately owned partial file; never append from two controllers.
An observation timeout does not prove that the original writer exited.

Before appending an HTTP Range response, require status 206, the exact requested
Content-Range and total size, and the exact body length. A status 200 response
cannot be appended as a range. Refresh an expired signed URL for the same
immutable file revision; retain the failure without logging URLs or credentials.

After transfer, verify all assets against the original manifest, including an
independent full digest read with stable file identity. Reconcile safetensors
offsets, tensor names, dtypes and shapes with the complete index and model
configuration. Partial-shard headers are diagnostic only. Record upstream
identity separately from mirror transport; a complete download does not qualify
the provider, Graph execution, accuracy or another hardware architecture.

## Inspect a live worker without copying its owners

A live CompilationConfig can reference model modules and KV-cache tensors
through static_forward_context. dataclasses.asdict and deepcopy can clone
those owners and exhaust device memory. Read selected primitive configuration
fields explicitly. For tensors, record metadata such as shape, dtype, stride,
device and address without cloning, transferring or materializing values.

Keep an inspection failure as evidence. Recheck the recorded service and worker
identities, health and selected implementation before continuing the same
service. Label that continuation as warmed execution after the failed probe;
it does not establish clean startup or lifecycle acceptance. Prepare a corrected
probe separately rather than replacing modules beneath the live worker.
