# vLLM 0.32 AITER INT8 MoE Extension Design

## Intent

Make an explicitly requested `--moe-backend aiter` run channel-weight,
dynamic-token INT8 W8A8 MoE checkpoints on HCU with official vLLM 0.32 while
keeping the AITER kernels in the installed vendor `aiter` wheel. The DAS
optimization wheel owns only the vLLM-to-AITER selection and ABI bridge;
`adapter-das` remains the sole platform, version, DTK, and extension-lifecycle
owner.

The first model gate is `/model/Qwen3.5-35B-A3B-W8A8`, whose compressed-tensors
metadata declares channel INT8 weights and dynamic per-token INT8 activations.
The required launch path keeps Graph enabled and explicitly passes
`--moe-backend aiter`.

## Goals

- Accept explicit AITER selection in vLLM 0.32's INT8 MoE oracle.
- Reuse vLLM's official `int8_w8a8_moe_quant_config`; do not implement a
  second W8A8 quantization description or algorithm.
- Map vLLM's INT8 channel/token metadata to the public vendor API
  `aiter.moe.MoeQuantType.W8A8`, `get_aiter_moe_config`, and `aiter_moe`.
- Keep BF16, FP8, MXFP4, auto selection, and non-AITER INT8 routes on their
  official vLLM owners.
- Record the selected AITER solution/config and every per-shape fallback.
- Preserve an official vLLM Triton functional fallback when AITER has no
  compatible native-layout configuration for a runtime shape.
- Prove selector reachability, numerical parity, Graph replay, installed-wheel
  ownership, model startup, generation, and HumanEval64 accuracy.

## Non-goals

- Do not copy or fork AITER kernels into either repository.
- Do not activate the legacy `vllm_hcu` dispatcher or package its platform,
  Worker, model, DeepGEMM, attention, collective, or checkpoint patches.
- Do not add AITER implementation code to `adapter-das`, `sitecustomize.py`, or
  `hcu_adapter.bridges.vllm`.
- Do not change vLLM source files or register another
  `vllm.platform_plugins`/`vllm.general_plugins` owner.
- Do not claim that passing a Triton fallback shape proves AITER provider
  coverage.
- Do not add a new environment variable: the existing CLI backend selection
  is the user-visible on/off control for this capability.

## Constraints

- Target the epoch-zero vLLM 0.32 release line (`>=0.32.0,<0.33.0`) through
  the existing `hcu_adapter.extensions` API v1 lifecycle.
- The extension must install before vLLM's compressed-tensors, online-INT8, or
  Quark INT8 consumers freeze `from ...oracle.int8 import ...` aliases. Late
  installation fails loudly instead of rebinding already imported consumers.
- Explicit AITER selection must never silently become global Triton selection.
  Runtime shape fallback is allowed only with a once-per-shape record, and at
  least one real AITER kernel/config hit is required for the AITER validation
  row.
- Vendor availability and exact callable signatures are capability checks;
  full wheel strings are recorded as provenance, not used as exact runtime
  locks.
- The extension wheel remains optional and contains only `hcu_vllm_das` code.

## Approaches

### Put INT8 AITER support in `adapter-das`

This would make the host adapter own an optional performance provider and
violate the documented boundary that independent optimizations enter through
`hcu_adapter.extensions`. It is rejected.

### Enable the legacy `vllm_hcu` dispatcher

This would quickly expose the old AITER path, but also activates a broad set of
version-coupled platform, Worker, model, DeepGEMM, and runtime patches. It
would duplicate adapter ownership and make v0.32 failures hard to attribute.
It is rejected.

### Add a narrow optimization-only Plugin capability (selected)

Add a v0.32-specific oracle seam and an AITER provider adapter under
`adapter_extension/src/hcu_vllm_das/moe`. The oracle returns a Plugin-owned
subclass of official `AiterExperts` only for the INT8 channel/token pair. The
subclass uses the public vendor API for actual work and delegates unsupported
runtime shapes to official vLLM's INT8 Triton functional implementation. This
keeps official classes and other quantization routes untouched.

## Package structure and ownership

```text
adapter_extension/src/hcu_vllm_das/
  extension.py                 extension capability and activation order
  moe/
    __init__.py                framework-light package marker
    int8_oracle.py             vLLM 0.32 selector and expert seam
    fused_moe_aiter.py         public AITER API adapter and config evidence
```

The actual kernel and tuned CSV remain owned by the installed `aiter` package.
`fused_moe_aiter.py` may select and invoke those assets but contains no Triton,
HIP, or copied AITER kernel implementation.

## Oracle seam

`int8_oracle.install()` validates the exact vLLM 0.32 symbols and parameter
names before changing state. It extends the process-local `Int8MoeBackend`
enum with `AITER`, maps only the explicit string `"aiter"`, returns
`HcuAiterInt8Experts`, and treats AITER weights as already canonical in
`convert_to_int8_moe_kernel_format`.

The official `select_int8_moe_backend`, `make_int8_moe_quant_config`, and
`make_int8_moe_kernel` bodies remain in use. Auto backend priority is not
changed. The install is idempotent and keeps original callables for audit and
tests.

`HcuAiterInt8Experts` subclasses official `AiterExperts`, admits only
`(kInt8StaticChannelSym, kInt8DynamicTokenSym)` beyond the parent support
matrix, and overrides only the INT8 execution leaf. Other quant schemes remain
owned by the parent and are not routed through this extension.

## AITER execution and fallback

For each runtime shape, the provider adapter builds the exact problem from the
actual tensors and MoE configuration and calls `get_aiter_moe_config` with
`MoeQuantType.W8A8`, channel block size zero, and no requested weight shuffle.
When a native-layout config is available, it logs the solution type and stable
config fields once and invokes `aiter_moe` with the canonical weights, official
vLLM scales/zero points, expert map, activation, and output dtype.

If lookup fails or the selected config requires a weight/scale layout this
first v0.32 slice does not own, it logs a once-per-shape fallback and calls the
official vLLM `fused_experts_impl` with `use_int8_w8a8=True` and
`per_channel_quant=True`. This is a correctness fallback, not AITER execution
evidence. Vendor invocation errors after a compatible AITER config is selected
propagate; they do not silently change providers.

The audited Qwen3.5 target shape (`E=256`, `N1=1024`, `N2=2048`, `K=2048`,
`top_k=8`) currently resolves M=1 through M=1024 to AITER ASM entries from
`tuned_fmoe_asm_w8a8_channel.csv` without shuffle. Runtime logs, not this
historical probe, determine acceptance.

## Activation and packaging

The existing `das_hcu` extension adds capability `aiter-int8-moe`. Activation
validates the required public AITER symbols and frozen vLLM interfaces before
importing `hcu_vllm_das.moe.int8_oracle` and calling `install()`. Discovery
continues to import no Torch, vLLM, LightOp, AITER, or legacy `vllm_hcu` code.

The installed verifier checks the new capability, module ownership, oracle
marker, explicit selector result, and absence of Plugin-owned platform/general
entry points. Wheel inspection must contain the new `hcu_vllm_das/moe` files
and no `vllm_hcu` package.

## Verification

Portable tests use fake vLLM/AITER modules to cover ABI drift, late install,
idempotence, explicit selection, unchanged auto priority, official quant-config
reuse, native-layout AITER invocation, config-miss/layout fallback, and error
propagation.

Live HCU tests compare AITER INT8 output with official vLLM Triton on legal
channel/token tensors, then capture and replay with changed input. Installed
wheel validation launches `/model/Qwen3.5-35B-A3B-W8A8` non-eager with
`--moe-backend aiter`, requires actual AITER config/kernel evidence, runs a
deterministic smoke request, and evaluates HumanEval64 in the existing isolated
scoring environment. The Triton control is reported separately and cannot
satisfy the AITER row.

