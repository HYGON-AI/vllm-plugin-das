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

## Check installation isolation before adding dependencies

A prepared venv may contain top-level package symlinks into shared
site-packages while lacking matching distribution metadata. Installing a
distribution that exports that package can follow the symlink and overwrite
shared files, even with `--no-deps`. Inspect both package destinations and
dist-info ownership before installation; use an independently owned package
directory when adding or replacing a dependency.

If an overwrite occurs, preserve the affected bytes and identify the original
distribution from its existing metadata. Restore only payloads whose digests
and sizes match its pre-existing RECORD, then verify all hashed entries and
the unchanged RECORD. Keep candidate and shared payloads separate. A repair
cannot establish what an already-running process loaded or retroactively
qualify earlier runtime results; recheck the affected environment before using
it for new validation.

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

## Resume scoring without regenerating answers

Before a long generation run, check the evaluator wrapper's input-root policy
against the resolved output path. A symlink beneath an allowed evidence root
can resolve outside that root and fail admission after generation completes.
Use a real output directory within the allowed root when the wrapper requires
one; temporary caches can remain on separate storage.

If scoring alone fails after answers are complete, preserve that failure and
the original requests, responses, samples and problem data. Copy the existing
inputs into an admitted directory if needed, compare every copied file's bytes
or digest, then run the unchanged isolated evaluator on those inputs. Record
the original and recovered paths and score receipt. Do not loosen the path
guard, repeat model generation, or treat recorded functional-probe failures
as passes because a score was recovered.
