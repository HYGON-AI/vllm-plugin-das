# Step5 runtime integration

Step5 support now lives in `vllm_hcu`. The former 24 source patches have been
retired; installation and startup must not rewrite the installed vLLM package.

See [Step5 runtime integration](../../docs/step5_runtime_integration.md) for
module ownership, registration order and validation. Historical source patches
remain recoverable from Git history.
