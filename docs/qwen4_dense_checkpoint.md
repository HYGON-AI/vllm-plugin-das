# Optional Qwen4 dense checkpoint loading (vLLM 0.33)

The official AMD Qwen4 decoder already constructs `Qwen3NextAttention` when
`indexer_n_heads` is `None`, including the `indexed_attention` and
`qwen_sparse_attention` layer types. Keep that constructor and its cache-scale
mapping. The old wrapper for `full_attention` does not match this interface.

When reusing a QSA checkpoint for this deliberately different dense mode,
unused language-model/draft `self_attn.indexer` weights must be discarded before
the official loader traverses the module tree. The optional
`qwen4-dense-qsa-checkpoint` capability provides only that boundary. It publishes
lazy subclasses through the public model registry, preserving architecture names
and automatic MTP selection. It inherits constructors, forwards, compilation
annotations and the official multimodal processor without replacement.

Select the extension explicitly, using the Adapter's existing configuration:

```bash
export HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS=das_hcu_qwen4_dense
# For a multimodal/root Qwen4 config, add to the normal vllm serve arguments:
# --hf-overrides '{"text_config":{"indexer_n_heads":null}}'
# For a standalone text config instead:
# --hf-overrides '{"indexer_n_heads":null}'
```

Apply nested overrides through the public `ModelConfig`/CLI path. Calling the
lower-level config reader with a nested replacement dict has different semantics
and is not a test of CLI overrides. A root-only override on a multimodal config
does not disable the text indexer. The official model-config hash changes when
the real text configuration changes; no private environment switch is added.

Installation/discovery alone does not publish these subclasses. Activation
requires HCU and the epoch-zero vLLM 0.33 release line. Foreign model owners are
preserved, partial publication is fatal, and later ownership drift is rejected
without registry repair. Process locks/failure handling use Adapter's shared
`ProcessLocalLifecycle`; a healthy fork must reactivate in the child, while an
interrupted publication cannot be resumed there.

With QSA still enabled, the exact input iterator reaches the official loader.
In dense mode only anchored language-model/draft indexer paths are filtered,
lazily; other unknown weights and visual indexers retain official errors.
The checkpoint and installed Core/FlashAttention files are never rewritten.

Validation entry point: `tests/vllm033/test_qwen4_dense_cpu.py`. Its official
constructors use explicit CPU leaf doubles, and its real loaders use synthetic
CPU module trees. It also tests actual public config parsing and installed entry
points. These checks do not establish full checkpoint construction, GPU kernels,
Graph replay, mixed decode/prefill, MTP acceptance, model accuracy or performance.
Those remain required device/end-to-end gates before this capability is Ready.
Disabling QSA changes the model's attention semantics; it is an explicit mode,
not a transparent accuracy-preserving optimization.

The installed CPU qualification for runtime `1fba2bc1` (Core `c440c089`,
Adapter/native `d9bb7fa5`) passes the complete portable selection: 1408 cases,
including the 15 new CPU scenarios, plus 11 fresh public-selector processes.
There are no failures/errors/skips; 46 HCU cases are deliberately deselected.
All 18 owned distribution RECORDs are unchanged before/after the run.
Real checkpoint-index inspection selects 39 indexer names and preserves all
152049 other names. Clean-environment probes also retain the official MTP
architecture mapping for multimodal and standalone text configurations.
Configuration probes use `tokenizer_mode="skip"`; they do not qualify actual
multimodal processing. The device/model gates above remain open.
