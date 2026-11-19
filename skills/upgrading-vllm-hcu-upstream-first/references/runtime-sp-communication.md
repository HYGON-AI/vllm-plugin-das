# Runtime-SP communicator construction

Trace `make_parallel_config`, `ParallelConfig.use_all2all` and the actual EP
communicator together. In the inspected 0.33 target, TP2/DP1/PCP1 with explicit
MoE SP and EP requires all-to-all kernels, but the global parallel config does
not admit a manager. Changing a runner flag after factory construction cannot
repair this mismatch.

Use the official platform communicator-class interface. Adapter owns a single
selector registry and retains the official default; an explicit Plugin
communication provider owns the custom-SP sidecar policy and subclasses the
official communicator. Pass EP admission into its constructor, preserving all
arguments including `tcp_store_group` and stateless rank metadata. Reuse Core's
manager/backend selection and destruction; do not construct a replacement in
dispatch or during capture. Required capability health must bind the exact
current-process report before the selector returns a provider class.

Transport token/head split buffers through public Torch collectives over the
borrowed TP device group. Refuse a missing multi-rank group instead of falling
back to the world group. Check independent layouts/inverse transforms, changed
inputs on Graph replay, stable addresses and natural cleanup. Byte transport
does not qualify FP8 scaling or quantization.

Core's preallocated reduce-scatter may return a new view of the supplied output.
Check storage address, shape, dtype and contents rather than Python object
identity; retain the full numerical and changed-input Graph assertions.

The `runtime-sp-communication` capability is a communication subfeature under
development, not complete GLM4/HY3 custom-SP. It uses the existing
`additional_config['hcu']['enable_custom_sp']` boolean and an explicit Adapter
capability request; no new environment policy is needed. Keep model embeddings,
residual/position layouts, replicated GQA, weight/scale reload, shared experts,
factory routing/configuration, target/draft batching, MTP/PP/DP/PCP, model
HumanEval32, performance and the original shutdown gate independently open.
