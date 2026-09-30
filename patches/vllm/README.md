# vLLM source patches

vLLM-HCU does not modify vLLM source at install or runtime. These patches are
applied to the installed vLLM package separately when a model needs vLLM core
changes that are not yet available from the plugin.

| Patch | Purpose | Base |
| --- | --- | --- |
| `0001-vllm0.28-step5.patch` | Step5 model support (`Step5ForCausalLM`, `Step5ForConditionalGeneration`, Step5 configs, Step4 DSA backend, `VLLM_STEP5_*` env vars) | vLLM `0.28.1` package tree, commit `2d1f703` → `00afa28` |

Paths in the patch are relative to the `vllm` package directory. Apply it with:

```bash
VLLM_DIR=$(python -c "import os, vllm; print(os.path.dirname(vllm.__file__))")
patch -p1 -d "$VLLM_DIR" < patches/vllm/0001-vllm0.28-step5.patch
```

The companion HCU-side changes (Step4/Step5 ByteLevel tokenizer loading, mixed
backend block-size realignment, and the 64-token HCU flash-attention page) are
in `vllm_hcu`.
