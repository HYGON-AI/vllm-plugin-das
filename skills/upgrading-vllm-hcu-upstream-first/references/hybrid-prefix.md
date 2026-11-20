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
