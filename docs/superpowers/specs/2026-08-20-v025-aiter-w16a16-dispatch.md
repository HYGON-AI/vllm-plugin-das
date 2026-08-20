# vLLM 0.25 HCU AITER W16A16 Dispatch Design

## Goal

Port the vLLM 0.21 HCU `_aiter_ops` W16A16 dispatch behavior to the vLLM
0.25 plugin so a real `UnquantizedMoeBackend.AITER` request prefers
`fused_experts_asm_impl`, including the configured solution id, instead of
preferring `aiter.moe.aiter_moe`.

## Scope

- Base the work on `origin/v0.25.1` in an isolated branch and MR.
- Keep the existing call chain:
  `AiterExperts.apply -> rocm_aiter_fused_experts -> rocm_aiter_ops.fused_moe
  -> _rocm_aiter_fused_moe_impl`.
- Keep `aiter_moe_request_context` as the explicit backend signal.
- For unquantized inputs with no scales, call `fused_experts_asm_impl` first.
- When `VLLM_HCU_USE_AITER_MOE_CONFIG=1`, resolve the configured solution and
  pass its `solution_id` to `fused_experts_asm_impl`.
- Preserve vLLM 0.25 validation for unsupported `gate_mode`, sorting policy,
  padding, biases, and router-weight-on-input arguments.
- Preserve delegation to the original vLLM implementation for quantized or
  scaled inputs.
- Do not modify `/models/zb/vllm_025/vllm`.

## Validation

- Behavior tests must prove the configured path calls
  `fused_experts_asm_impl` and does not call `aiter.moe.aiter_moe`.
- Tests must prove the solution id and 0.25-only arguments are handled
  correctly, and non-W16A16 requests still delegate upstream.
- Run focused and adjacent CPU suites, compile checks, and diff checks.
- If HCU resources are free, launch `/models/Qwen3.5-35B-A3B` with
  `--moe-backend aiter` and verify a deterministic 4096-token request before
  opening the MR.
