# vLLM 0.28.1 gfx938 validation provenance

Recorded at `2026-10-03T18:29:55+00:00`. This record freezes the wheel pair
used to start the gfx938 model-validation campaign. A final plugin wheel is
rebuilt after any runtime fixes; this pre-validation wheel remains immutable
evidence for the initial harness and compatibility gates.

## Source topology

- Plugin repository branch: `codex/validate-v0281-gfx938-models`
- Plugin wheel source commit:
  `f189798e80b34a94b1577de19a70fab549236661`
- Rebased PR 181 stack head:
  `592b38cf9ce9f6e36ab1c846ecd2483f03e20fd9`
- PR 180 base head:
  `f24e99be620c3a0d3611a7d96079bb7be7e63de8`
- `origin/v0.28.1-dev` baseline:
  `9788dc633fb3d0a0a0a124b5886f62c07b3de765`

The plugin wheel was built from a clean tracked worktree with:

```bash
ROCM_PATH=/opt/dtk MAX_JOBS=16 python setup.py bdist_wheel
```

## Frozen wheels

| Package | Filename | SHA256 |
| --- | --- | --- |
| vLLM | `vllm-0.28.1+dtk2604.torch2110.2609171627.g77acaf-cp310-cp310-linux_x86_64.whl` | `2d6b392dcff0d5064c754e838d9ac5ea3ffd5d6c7cac86ed2b5b1b2811cb9b3b` |
| vLLM HCU plugin | `vllm_hcu-0.28.1rc1.dev491+das.f189798.dtk26041-cp310-cp310-linux_x86_64.whl` | `407b68d3a37a1148c835e3b12a10c7c59cd5d184c2f95ceb0a0484a3a67c8fd5` |

The vLLM wheel is stored at `/models/artifacts/v0281-gfx938/`. The plugin
wheel is the `dist/` artifact from the source commit above. The `dtk26041`
local-version component is the plugin build system's encoding of the installed
DTK `26.04.1` release.

The wheels were installed without dependencies into these immutable targets:

- vLLM: `/models/.installs/vllm-v0281-gfx938-g77acaf`
- plugin: `/models/.installs/vllm-plugin-v0281-gfx938-f189798`

`importlib.metadata` reported the following distributions from those exact
roots:

- `vllm==0.28.1+dtk2604.torch2110.2609171627.g77acaf`
- `vllm-hcu==0.28.1rc1.dev491+das.f189798.dtk26041`

## Parent and child import roots

The import probe ran with `PYTHONNOUSERSITE=1`, the vLLM target and plugin
target at the front of `PYTHONPATH`, and path assertions in both the parent and
a fresh child interpreter. Both processes resolved the same files:

```text
vllm      /models/.installs/vllm-v0281-gfx938-g77acaf/vllm/__init__.py
vllm_hcu  /models/.installs/vllm-plugin-v0281-gfx938-f189798/vllm_hcu/__init__.py
hcu_ops   /models/.installs/vllm-plugin-v0281-gfx938-f189798/vllm_hcu/hcu_ops.cpython-310-x86_64-linux-gnu.so
```

The following vLLM native modules were also resolved below the isolated vLLM
root:

```text
vllm._C                         vllm/_C.abi3.so
vllm._rocm_C                    vllm/_rocm_C.abi3.so
vllm._C_stable_libtorch         vllm/_C_stable_libtorch.abi3.so
vllm._moe_C_stable_libtorch     vllm/_moe_C_stable_libtorch.abi3.so
```

## Runtime environment

- Host kernel: `Linux 6.6.92-34.1.tl4.x86_64`
- Python: `3.10.12` (`/usr/bin/python`)
- DTK release files: `26.04.1`
- HIP compiler version: `6.3.26255`
- Torch: `2.11.0+dtk2604.2609291046.ge12687`
- Hardware: 8 x `BW1100`, `gfx938:sramecc+:xnack-`, 147440 MiB VRAM per
  device
- `/usr/lib/x86_64-linux-gnu/librt.so`: absent
- `/usr/lib/x86_64-linux-gnu/librt.so.1`: present

Installed operator/provider distributions:

| Distribution | Version |
| --- | --- |
| `aiter` | `0.1.6+dtk2604.torch2110.2609300837.gf9b9c8` |
| `boltops` | `0.1.0+dtk2604.torch2110.2609301634.g137f5d` |
| `flash-attn` | `2.8.4+dtk2604.torch2110.2609241509.g624d7b` |
| `flash-mla` | `1.2.0+dtk2604.torch2110.2609291807.gc24e3f` |
| `lightop` | `0.6.0+dtk2604.torch2110.2609301833.g9d6ed8` |
| `deep-ep` | `1.1.0+dtk2604.torch2110.2609231914.g49988b` |
| `deepgemm` | `2.1.0+dtk2604.torch2110.2609301620.g9dc3b6` |

## Artifact verification

The installed-root compatibility, packaging, and lifecycle gates were run from
outside the checkout with pytest importlib mode so the source tree could not
shadow either wheel:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/models/.installs/vllm-v0281-gfx938-g77acaf:/models/.installs/vllm-plugin-v0281-gfx938-f189798 \
VLLM_TARGET_ROOT=/models/.installs/vllm-v0281-gfx938-g77acaf \
VLLM_HCU_TARGET_ROOT=/models/.installs/vllm-plugin-v0281-gfx938-f189798 \
VLLM_PLUGINS=__disabled__ \
python -m pytest --import-mode=importlib -q \
  /models/.worktrees/vllm-plugin-das-v0281-gfx938-validation/tests/patch/test_compatibility_gate.py \
  /models/.worktrees/vllm-plugin-das-v0281-gfx938-validation/tests/patch/test_setup_packaging.py \
  /models/.worktrees/vllm-plugin-das-v0281-gfx938-validation/tests/patch/test_plugin_lifecycle.py
```

Result: `58 passed, 14 warnings in 107.64s`.
