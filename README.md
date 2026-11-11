# vLLM HCU Plugin — v0.33.0

[English](README.md) | [简体中文](README.zh-CN.md)

This release line provides optional HCU optimizations for official vLLM 0.33
through `adapter-das`. Build the independent project in
[`adapter_extension/`](adapter_extension/); its distribution is
`hcu-vllm-das-plugin` and its Python package is `hcu_vllm_das`.

## Compatibility and ownership

The target is the epoch-zero vLLM 0.33 release line, including patch,
prerelease, development, postrelease and local build suffixes. Package and
upstream metadata live in
[`version.py`](adapter_extension/src/hcu_vllm_das/version.py).
Passing release admission does not validate the native ABI or a model.

- vLLM owns models, scheduling, allocation and official execution interfaces.
- Adapter (host `hcu-adapter`, kernel `sgl-kernel`, package `hcu_adapter`) owns
  real HCU identity,
  runtime admission, platform/general entry points and process health.
- This Plugin uses `hcu_adapter.extensions` to register optional RMSNorm and
  explicit FlashAttention providers. Official vLLM retains INT8 model routing;
  the historical AITER INT8 prototype is not an advertised model capability.
  Each active provider has separate prerequisites and acceptance gates.

The Plugin distribution version is `0.33.0`, aligned with the vLLM release
line. The retained repository-root `vllm_hcu` source is the legacy vLLM 0.28.1
development reference. Root `setup.py` rejects builds before importing Torch
and points to `./adapter_extension`; it must not produce a legacy wheel on
this branch. Use the original release branch for historical builds.

## Build and install

Prepare ABI-matched DTK PyTorch, Triton and the required vendor libraries in
an isolated environment. From the repository root:

```bash
python3 -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir ./dist ./adapter_extension
python3 -m pip install --no-deps /wheels/vllm-0.33.0+*.whl
python3 -m pip install --no-deps /wheels/sgl_kernel-*.whl /wheels/hcu_adapter-*.whl
python3 -m pip install --no-deps ./dist/hcu_vllm_das_plugin-*.whl
```

Installation, provider switches, rollback and runtime verification are described
in the [Plugin guide](adapter_extension/README.md).

## Validation and development

Use [`tests/vllm033/`](tests/vllm033/) for this release. Actual test commands,
implementation boundaries and remaining device/model gates are recorded in
[v0.33 validation](docs/validation/vllm033-upgrade.md). CPU contracts, device
numerics, changed-input Graph replay, model accuracy and performance are
separate results; source tests do not certify an installed wheel.

The current task's user-authorized acceptance of precision differences for
models below 7B, preserved original scores and PR330 review scope are recorded
in [the acceptance review](docs/validation/vllm033-small-model-acceptance.md).

Files under `validation/v0.32.0`, `tests/vllm032` and older versioned design or
validation documents are historical evidence. Their scores and artifacts do
not certify this release. The historical compatibility-lock test is retained
with its original inputs.

## License

Apache License 2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
