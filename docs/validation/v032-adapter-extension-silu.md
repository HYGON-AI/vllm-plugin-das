# vLLM 0.32 DAS extension: LightOp SiluAndMul validation

Date: 2026-10-26 (UTC)

## Scope and provenance

This record covers the first optimization-only capability in
`hcu-vllm-das-plugin`: LightOp `SiluAndMul`, loaded through
`hcu-vllm-adapter` rather than a vLLM platform or general-plugin entry point.

| Component | Version or revision |
| --- | --- |
| Official vLLM source contract | `v0.32.0`, commit `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| Installed vLLM wheel | `0.32.0+rocm723` |
| Adapter source | `45525d0056e788fd189172d53756c4b53a151e8a` |
| Adapter wheel | `0.1.0.dev0` |
| DAS reference baseline | `v0.28.1-dev`, commit `cd412a0d53b9b517fbc3dcdf96121dc3daa1096c` |
| DAS extension wheel | `0.1.0.dev0` |
| Device | HCU `gfx938` (device 0 selected for the focused test) |

The tested wheel files and SHA256 values are:

```text
0c9e3512dccbe5a4018218c710fb46608b4b6776cb5ef21b52d7f73e0cdfcd5d  hcu_vllm_adapter-0.1.0.dev0-py3-none-any.whl
9741f9a5a51c1aac6b88781eea2f4f461c5e640124d739fe7def31d8aafd0ac7  hcu_vllm_das_plugin-0.1.0.dev0-py3-none-any.whl
da7bda7ce51871b064f957076cfae43b40fee7226711432e88dcf2a5c0c91640  vllm-0.32.0+rocm723-cp312-cp312-linux_x86_64.whl
```

Artifacts were staged under
`/models/artifacts/vllm032-das-extension-2A91Bh`. This path identifies the
validation host's evidence directory; the hashes, not the path, identify the
tested wheels.

## Installed-state attestation

The wheels were installed with `--no-deps` into a fresh virtual environment
created with `--system-site-packages` to reuse the prepared HCU runtime. The
host exports a system `PYTHONPATH`, so `-E` was required to ensure the virtual
environment's vLLM 0.32 wheel preceded the older system distribution.

```bash
VLLM_PLUGINS=hcu_adapter \
HSA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
/models/artifacts/vllm032-das-extension-2A91Bh/venv/bin/python -E \
  adapter_extension/tools/verify_installed.py
```

Result: exit code 0. The attestation selected vLLM `0.32.0+rocm723`, adapter
`0.1.0.dev0`, and plugin `0.1.0.dev0`; extension `das_hcu` was `active`, its
capability list was `lightop-silu-and-mul`, and the OOT owner was
`hcu_vllm_das.silu_and_mul.HcuSiluAndMul`.

The installed `RECORD` content hashes reported by the verifier were:

```text
vllm:                766975ac0bb5d304e435903b782b3ab8295941bdf70ca9929bcc929839c22d40
hcu-vllm-adapter:    80c6c83cd0a3588ea588dc31c51cbc91cc60c949ff2f950c019c50a266242d07
hcu-vllm-das-plugin: 14a12f48ad8748210d30cf2867d3b9236c5de33af180f8cc85ac1e83f156be47
```

## Live HCU numerical check

```bash
HCU_VLLM032_LIVE=1 VLLM_PLUGINS=hcu_adapter \
HSA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
/models/artifacts/vllm032-das-extension-2A91Bh/venv/bin/python -E -m pytest \
  tests/vllm032/test_silu_and_mul_hcu.py -q -rs
```

Result: `1 passed` on `gfx938`. The test used BF16 input of shape
`(257, 8192)`, compared the optimized result with a float32 PyTorch SiLU-times-
gate reference at `rtol=2e-2, atol=2e-2`, and independently checked both the
master and operator-specific rollback paths against the same reference.

## Test-suite status

The vLLM 0.32 release-gate suite completed with `36 passed, 1 skipped`; the
single skipped case is the opt-in live-HCU test above, which passed separately
against the installed final wheel (`1 passed`). The installed-state verifier
also exited 0 against that wheel.

The retained vLLM 0.28 reference suite is not a release gate for this wheel and
did not complete on this checkout. With the adapter source on `PYTHONPATH`, a
focused first-failure run reached `100 passed` before
`tests/accuracy/test_hcu_kernel_accuracy.py::test_concat_and_cache_mla_e5m2_matches_reference_and_graph`
failed because the legacy root native module `vllm_hcu.hcu_ops` had not been
built. A continued run reached 11% and then the Python process aborted inside
`tests/kernels/test_gfx938_fp8_indexer_reader.py::test_layer_split_sized_stride_with_high_page_ids_matches_oracle`.
Neither failing test imports the independent `hcu_vllm_das` package, and no
legacy failure or abort is represented as a pass.

## Claim boundary

This evidence validates package ownership, adapter-mediated activation, the
vLLM 0.32 OOT contract, and focused `SiluAndMul` numerical correctness on the
recorded stack. It is not model accuracy, serving, concurrency, long-duration,
attention/MoE, `gfx92a`, or performance evidence. Those require separate
records before production enablement for the corresponding workload.
