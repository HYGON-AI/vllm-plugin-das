# HCU vLLM DAS optimization plugin

`hcu-vllm-das-plugin` is the optimization-only extension for official vLLM
0.32 and `hcu-vllm-adapter`. The adapter owns the HCU platform and runtime
lifecycle; this wheel contributes optional, independently removable operator
optimizations through the adapter extension API.

The first validated capability is the LightOp `SiluAndMul` path on `gfx938`.
No attention, MoE, model, topology, or performance support is implied by this
initial slice.

## Compatibility

- vLLM release: exactly `0.32.0`; prerelease, postrelease, development, and
  local version suffixes on that release are accepted.
- Adapter: `hcu-vllm-adapter>=0.1.0.dev0`, extension API v1.
- Runtime: a prepared HCU PyTorch/DTK/LightOp stack. LightOp is intentionally
  not fetched by pip because production deployments must use an ABI-matched
  local build.
- The plugin does not own `vllm.platform_plugins` or `vllm.general_plugins`.

The repository root remains the vLLM 0.28.1 development reference. Build only
this directory for vLLM 0.32:

```bash
python3 -m pip wheel --no-deps --wheel-dir ./dist ./adapter_extension
```

## Production installation

Start with a fresh virtual environment or image containing the matching HCU
PyTorch, DTK, and LightOp builds. Install audited local wheels without dependency
resolution, in this order:

```bash
python3 -m pip install --no-deps /wheels/vllm-0.32.0+rocm*.whl
python3 -m pip install --no-deps /wheels/hcu_vllm_adapter-*.whl
python3 -m pip install --no-deps /wheels/hcu_vllm_das_plugin-*.whl
```

If the shell exports `PYTHONPATH`, clear it or invoke Python with `-E` during
verification so an older system vLLM cannot precede the environment's wheel:

```bash
VLLM_PLUGINS=hcu_adapter python3 -E adapter_extension/tools/verify_installed.py
```

The verifier fails closed unless all three distributions have non-empty wheel
records, their imported modules are owned by those records, the adapter is the
sole expected vLLM entry-point owner, the DAS extension is active, and vLLM's
OOT registry selects `hcu_vllm_das.silu_and_mul.HcuSiluAndMul`.

## Rollback

The optimized path is enabled by default. Either switch below returns
`SiluAndMul` to vLLM's native implementation without uninstalling the plugin:

```bash
# Disable all optional DAS custom operators.
export VLLM_HCU_USE_CUSTOM_OPS=0

# Or disable only the LightOp SiluAndMul path.
export VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=0
```

For complete removal, uninstall `hcu-vllm-das-plugin`; the adapter and vLLM
remain installed and retain platform ownership.

See
[`docs/validation/v032-adapter-extension-silu.md`](../docs/validation/v032-adapter-extension-silu.md)
for the exact wheel hashes and live-HCU evidence for this revision.
