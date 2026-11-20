# Qwen4 optional dense-QSA checkpoint boundary (0.33)

Inspect the actual AMD decoder before carrying `VLLM_HCU_DISABLE_QSA` wrappers.
The 0.33 decoder already selects dense `Qwen3NextAttention` when the text config's
`indexer_n_heads` is `None`; current aliases include `indexed_attention` and
`qwen_sparse_attention`, rather than the old wrapper's `full_attention`.
Use public `ModelConfig`/CLI `hf_overrides` (nested `text_config` for a multimodal
root), verify the resulting typed text config and official config hash, and
retain the official `_qsa_layer_ids`/cache-scale mapping. A lower-level config
reader's replacement-dict failure does not establish a CLI override defect.

QSA checkpoints still contain disabled indexer weights. If a Plugin loader
adapter is needed, restrict it to explicit dense-mode language-model/draft
indexer paths and stream into the official loader; preserve QSA passthrough and
errors for unrelated/vision weights. Keep original constructors and multimodal
processor inheritance, and verify automatic MTP architecture selection. The
optional capability and remaining validation gates are documented in
`docs/qwen4_dense_checkpoint.md` in the Plugin repository. CPU constructor/loader
contracts do not establish device, mixed-batch, MTP or model accuracy acceptance.

For native Qwen4 loader probes, keep the miniature configuration consistent
with the target's hybrid model and state manager. An all-attention fixture can
be rejected as having no Mamba layers before the loader is exercised. Inspect
the resolved GDN output gate too: the tested Transformers config defaults
`output_gate_type` to `None`, while the target GDN constructor requires an
explicit supported activation. Use the intended checkpoint's value for a
representative fixture; do not rewrite checkpoint configs or normalize production
semantics to make a synthetic test run. The local trained checkpoints use
`sigmoid`; a synthetic `silu` run only covers that synthetic configuration.

Bind native evidence to the actual worker model class, attention implementation,
cache configuration and changed-input Graph execution. A loader probe should
show a real parameter update through the official loader and restore its original
bytes before generation. Keep constructor failures and lifecycle diagnostics;
synthetic output parity does not establish trained-checkpoint, MTP or HumanEval
acceptance.

Check the actual platform/warmup guard before installing a missing CUDA
dependency on HIP. Installed CUDA CUTLASS can make an optional warmup
appear available even when it cannot execute on HCU. A separate HIP-only
dependency cohort must retain the original failure and reverify all retained
distributions; it cannot qualify the original package combination.
