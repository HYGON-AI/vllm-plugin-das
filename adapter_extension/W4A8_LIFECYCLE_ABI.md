# Optional W4A8 lifecycle capability

The independent `das_hcu_w4a8` extension requires the following public callable
ABI from `hcu_adapter.bridges.vllm_model_load`, in addition to extension API v1:

```python
register_post_model_load(owner, callback)
require_post_model_load(vllm_config)
register_model_mutation_guard(owner, callback)
require_model_mutation_guard(vllm_config)
preflight_model_lifecycle_hooks(
    *, post_model_load_owner, post_model_load_callback,
    model_mutation_owner, model_mutation_guard,
)
register_model_lifecycle_hooks(
    *, post_model_load_owner, post_model_load_callback,
    model_mutation_owner, model_mutation_guard,
)
```

These must be synchronous callables whose signatures accept the documented
arguments without extra required parameters. Compatible optional arguments are
accepted. This is a minimum capability requirement, not an exact Adapter wheel,
version string or commit check. Adapter PR19's public facade revision
`2be0047` supplies these vLLM admission queries and reexports the shared pair
API. Declarations, contexts and synchronization remain framework-free in
`hcu_adapter.model_load`; worker/executor/env-cache admission belongs to the
facade. Do not add reverse imports or old-path aliases to the shared module.
PR revisions are traceability references, not runtime SHA admission rules.

W4 first checks all six signatures, then calls public read-only preflight before
importing its quantization config/provider or registering anything. Preflight
checks the two hook owners but does not reserve them. Atomic registration repeats
those checks and commits both Adapter hooks, or neither, before W4 publishes its
quantization alias or compile factor. Its scope is the two Adapter hook
registries; it does not transact arbitrary vLLM/Plugin registries. There is no
split-registration compatibility path and no access to private Adapter fields.
Missing or incompatible ABI causes a clear optional-capability activation failure
before these mutations; required-capability policy remains Adapter-owned.

The resolved-config `require_post_model_load` and
`require_model_mutation_guard` calls remain at actual W4 selection. Successful
target/draft load records weak worker/model ownership and freezes the effective
policy before env caching. A protected worker refuses reload and target/draft
update starts before providers run, including after config/model replacement;
unowned official CT workers delegate. Fork preserves declarations and poisons
inherited protected W4 state, while unowned CT workers still delegate.

Implementation/vendor/config identity remains part of ordinary and model-AOT
compile factors. Distribution versions and source/wheel hashes are provenance;
they do not replace callable, worker, native-layout or numerical gates. This
lifecycle change adds no GPU, Graph, full-model, MTP, INT8-PLE or EP acceptance.

## Failure after hook publication

Before importing the quantization/provider or publishing any owner state, W4
also checks this public Adapter ABI:

```python
# hcu_adapter.extensions
ExtensionCommitError(message, reports=())  # RuntimeError subclass
is_fatal_extension_error(error)
# hcu_adapter.bridges.vllm
require_runtime_healthy()
```

Compatible optional parameters are accepted; asynchronous/generator callables
and additional required parameters are refused. The health query runs before
owner locks. W4 validates these public interfaces locally and does not import a
private helper from the shared `das_hcu` bundle.

The Adapter pair transaction covers only its two lifecycle hook registries.
If subsequent vLLM quantization-alias or compile-factor publication fails, W4
latches and raises the public fatal error. Selection, activation, repeated
installation and fork cannot retry that partially published owner as healthy.
This deliberately does not claim rollback of arbitrary framework registries.

The owner stores a fixed fatal sentinel before rendering an error. Rendering
may itself raise `BaseException`, reenter, or fork; none can clear that failure.
The install attempt starts before dependency preparation. An inherited
in-progress attempt is refused in a forked child, while a clean parent remains
usable. An already fatal dependency (including suppressed exception context)
remains fatal even before W4 publishes anything. Clean pair-registration failure
remains optional and retryable after the Adapter rolls back its two hooks.

Minimum source dependencies are Adapter PR17, public-facade PR19 (`2be0047`)
and independent public-fatal PR23 (`09bd79a`). PR23 does not depend on PR19.
Environment-driven required owner/capability deployment additionally uses PR20;
the existing public ExtensionContext required API is not evidence of PR20 env
parsing. FP8 PR18, Flash PR21, runtime admission PR22 and AR compatibility are
separate features, not W4 source requirements. Exact wheel/version pins do not
replace callable, resolved-config or immutable installed provenance checks.

The independent owner does not import a private helper from the legacy bundle.
Its new source identity requires fresh wheel/installed/HCU/model gates;
historical `0a1086f`/`3aa22ee` results apply only to that frozen pair. No new
shutdown callback, timeout relaxation or clean-shutdown claim is introduced.
