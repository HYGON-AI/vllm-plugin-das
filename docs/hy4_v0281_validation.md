# Hy4 v0.28.1-dev HCU validation record

## Frozen baseline, 2026-09-25 UTC

- Plugin source after proxy-free fetch: origin/v0.25.1 at 6ea7b12f3d77d4564d613aac31c5e6c5e7a8641d.
- Plugin target after proxy-free fetch: origin/v0.28.1-dev at 93021650a5c768121507d5b8241d0e848756e44e.
- Feature worktree: feat/hy4-v0281-alignment, based on the target SHA above.
- OpenDAS vLLM: 0.28.1+dtk2604.torch2110.2609171627.g77acaf, imported from /usr/local/lib/python3.10/dist-packages/vllm. This version reports the 77acaf lineage; the installed wheel artifact and checksum are not available in the inspected /models locations or dist-info direct_url.json, so exact delivery provenance remains unverified.
- Installed plugin distribution: vllm-hcu 0.28.1rc1.dev491+dtk2604.torch2110.2609191510.g06f72c. Tests import plugin source from this worktree, not the installed distribution.
- Python 3.10.12; PyTorch 2.11.0, HIP 6.3.26113; DTK 2604 is encoded in the installed package versions.
- Proprietary providers: aiter 0.1.6+dtk2604.torch2110.2609200940.g0cb699; flash_attn 2.8.4+dtk2604.torch2110.2609241509.g624d7b; flash_mla 1.2.0+dtk2604.torch2110.2609161027.g5b030d; lightop 0.6.0+dtk2604.torch2110.2609201832.g8cd3e1.
- Model: /models/Hy4-preview-Channel-FP8-w8a8-v2; config.json SHA-256 d4a648cb09bb89f4b8778e60629e43618f1abb581ef3aa38bd67b2b2cd998441; architecture HYV4ForCausalLM, model_type hy_v4, 78 hidden layers, one checkpoint MTP layer.
- Eight HCU devices were visible; each reported 147440 MiB total and 2 MiB used before implementation.
- The configured 127.0.0.1:7897 HTTP proxy was unavailable. Fetch succeeded with proxy environment variables unset and git http.proxy cleared; no credentials were put into commands or this record.

## Model Runner V2 gate

Every Hy4 server command must explicitly set:

~~~bash
export VLLM_USE_V2_MODEL_RUNNER=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
~~~

The target HCU worker rejects Model Runner V1. The existing plugin's automatic V2 normalization applies only to its GLM DSA architecture and is not a Hy4 selector. The live TP8, MTP3, and FP8-KV gates must each confirm VllmConfig.use_v2_model_runner=true and actual HcuGPUModelRunnerV2 construction, not just the environment variable.

Baseline characterization:

~~~text
PYTHONNOUSERSITE=1 PYTHONPATH=. pytest -q tests/patch/test_plugin_lifecycle.py -k 'model_runner or worker_rejects'
4 passed, 32 deselected, 14 torch deprecation warnings, 18.65s
~~~

No Hy4 service, GPU inference, CUDA Graph capture, MTP acceptance, FP8 KV accuracy, or HumanEval result has yet been measured on this target branch. Installed-wheel provenance and isolated-artifact bootstrap remain open gates.
