# HCU vLLM DAS optimization plugin

This subproject is the optimization-only DAS extension for
`hcu-vllm-adapter` and official vLLM 0.32. Build this directory, not the
repository root:

```console
python3 -m pip wheel --no-deps ./adapter_extension
```

The initial package boundary contains no legacy `vllm_hcu` platform, Worker,
model, patch, or native-extension modules. Runtime capabilities are added only
after their vLLM 0.32 contracts and HCU validation pass.

