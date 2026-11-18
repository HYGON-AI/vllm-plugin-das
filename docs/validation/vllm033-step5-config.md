# v0.33 standalone Step5 draft configuration

Official v0.33 already owns `Step5Config`, `Step5VLConfig` and the in-checkpoint
Step5 MTP conversion. A separately saved `model_type="step5_mtp"` configuration
is not recognized by the ordinary HF parser on Core `c440c089`.

The Plugin uses Adapter API1 activation to register a small subclass of official
`Step5Config` through public Transformers `AutoConfig.register()`. Core files,
private config/parser tables and official target classes remain unchanged.
The default standalone architecture is `Step5MTP` and default prediction-layer
count is one; checkpoint values override these defaults.

`tests/vllm033/test_step5_config_cpu.py` checks the built, installed Plugin:

- Actual metadata discovery and Adapter API1 activation.
- Default/explicit draft metadata through real `get_config`, both `hf` and
  `auto`, with remote code disabled; save/load and pickle preserve layer arrays,
  quantization metadata and the prediction-layer count.
- Ordinary official Step5 text/VL round trips and inherited invalid-input
  rejections.
- Foreign-owner refusal, read-only health, latched publication failure and
  registry drift, repeated activation, fresh spawn and a fork while a parent
  thread holds the extension lock.

The portable CI selector collects this file. Packaging tests check the new
entry point in source metadata and the built wheel. These are configuration and
lifecycle contracts, not native numerical tests.

No Step5 weights or model were executed by these tests. Standalone checkpoint
loading, Step4 tokenizer compatibility, sparse attention/environment behavior,
Graph, MTP drafted/accepted tokens, HumanEval and performance remain separate
gates. Do not retire the old composite Step5 patches wholesale or call this
full Step5 support.
