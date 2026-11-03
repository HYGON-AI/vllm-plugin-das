# v0.33 official ownership correction

The upstream-first review found a concrete architecture violation in the
previous INT8 integration: `int8_oracle.install()` replaced the official
`Int8MoeBackend`, `backend_to_kernel_cls`, `map_int8_backend` and
`convert_to_int8_moe_kernel_format`. Passing numerical/model tests did not
make those assignments a public registration API. The user explicitly
reaffirmed that existing official features must remain official owners.

The corrected extension removes that installation and AITER import/ABI
dependency, advertises only `lightop-rms-norm`, and leaves official INT8
selection, conversion, quantization and kernel construction intact. The old
installer now rejects without importing or mutating Core. Its unregistered
expert factory and standalone provider are operator experiments only.
`aiter-int8-moe` is explicitly unavailable, including required-capability
requests. This is not acceptance of the requested native AITER integration.

## Source-based ownership decisions

Official target SHA: `4ba787ec9959e566d3b587a796dff07673324da5`.

| Capability | Official implementation / decision | Remaining HCU work |
| --- | --- | --- |
| INT8 channel-weight/token-activation models | Official CT/online/Quark methods and `fused_moe.oracle.int8`; preserve auto/Triton/Humming/CPU selection | AITER is absent from this INT8 oracle; no public expert registry. No private rebind. |
| BF16/FP16 AITER experts | Official `AiterUnquantizedExperts` and unquantized oracle | Adapter narrow HCU admission/neutral vendor ABI, then real kernel/model validation; no duplicate Plugin math |
| INT8 activation/quantization/reduction | Official utilities already reused by standalone ASM experiments | Their tests do not grant a model registration seam |
| SiluAndMul | Official CustomOp implementation | No Plugin override |
| GDN/recurrent attention and causal conv | Official `vllm.third_party.flash_linear_attention` / packed recurrent kernels / causal-conv Triton | Validate actual required model/topology; do not blindly port historical provider replacements |
| MTP/draft buffers, runner, scheduling, Graph | Official v0.33 implementations | Validation and narrow platform compatibility; no Plugin runner/speculator copies |
| Hybrid prefix/checkpoint/resume/allocation | Official manager and virtual page splitting | Plugin Flash owns only the explicit vendor page/cache-view contract |
| BF16/FP16 cache writing and public dtype names | Official native writer / dtype resolver | Plugin forwards names; E5M2 byte writer/scales remain a separate optional increment |
| Ordinary/specialized attention selection | Official selector; official Triton/MLA paths remain | Explicit vendor Flash/E5M2 provider only through public backend API; current quality gate remains open |
| RMSNorm | Official fallback remains default | Existing LightOp candidate stays explicit opt-in through public OOT; does not justify replacing other official ops |

The public quantization-config registry can register a *distinct* new method.
It does not establish equivalent CT loaders, weight updates, expert layouts,
backend admission or lifecycle. Do not overwrite official `compressed-tensors`
or copy its private implementation to restore the retired AITER selector.
No replacement method is claimed in this correction.

## Verification and evidence boundaries

Two new regressions fail before the correction: ordinary operator activation
requires AITER/private INT8 installation, and discovery advertises the
unsupported capability. Corrected portable tests verify activation without
AITER dependencies, the retired installer leaves loaded consumers/oracle
unchanged, and capability discovery makes no false claim. Standalone expert
admission/provider checks remain separate from actual model routing.

The installed verifier schema4 checks the seven official oracle symbols
before/after activation, validates official module ownership and explicit
Triton expert selection, and rejects a surviving private installer marker
or false AITER capability advertisement. A mutation-in-activation regression
ensures that merely observing an ACTIVE report cannot pass this check.

Final candidate/wheel and installed evidence will be recorded in
`validation/v0.33.0/official-ownership-gfx936-candidate.json`.
Earlier ec01e209/529855a5 wheels, numerical results and all failed model
profiles remain frozen. The corrected candidate must not inherit their
native AITER model claim. No Core runtime or Adapter production edit is
needed for this removal. PR330 remains on hold for its existing quality
gates and incomplete AITER public integration. Goal remains active;
`last_good=false`. AI assistance was used.
