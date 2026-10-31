# vLLM 0.32 HCU FlashAttention E5M2 KV design

## Intent

Add model-independent `fp8_e5m2` KV-cache support to the Plugin-owned HCU
`FLASH_ATTN` backend. Qwen is one validation row, not an implementation
condition. Any ordinary decoder model already accepted by this backend should
receive the same capability.

## Scope

- Keep vLLM as owner of cache allocation, scales, and
  `reshape_and_cache_flash` writes.
- Keep Adapter as owner of selecting the Plugin only for the exact
  `FLASH_ATTN` backend.
- Let the Plugin advertise E5M2, bypass the upstream NVIDIA-specific E5M2
  rejection only for this HCU backend, keep BF16/FP16 queries, and present K/V
  bytes to the vendor reader as `torch.float8_e5m2`.
- Preserve both `LBHNC`/HND and `LBNHC`/NHD, 64-token pages, default
  `FULL_AND_PIECEWISE` Graphs, and all existing fail-closed feature limits.
- Do not change vLLM, Adapter, MLA, sparse MLA, multimodal-prefix, sliding
  window, hybrid attention, DCP, PCP, FP8 checkpoint-weight policy, or model
  code.

## Constraints

- No model-name or checkpoint allowlist.
- No global `current_platform.fp8_dtype()` change: E4M3 users must remain
  unchanged.
- No copied upstream `forward`; the vLLM 0.32 seam remains the owner of the
  data flow.
- No alternate writer. The installed HCU vLLM native writer must accept
  `fp8_e5m2` and its output bytes must be consumed by the public vendor
  `flash_attn_varlen_func`.
- A launch flag, successful cache write, or Graph capture alone is not a pass.
  Numerical writer-to-reader, changed-input Graph replay, model output, and
  checkpoint-specific BF16-KV/E5M2-KV A/B evidence are required.

## Approaches

1. **Selected: narrow Plugin capability.** Extend the HCU subclass at the
   existing vLLM seams. Temporarily present `auto` only to the upstream
   constructor's NVIDIA capability guard, restore the requested E5M2 state,
   disable FP8 query quantization, and correct K/V byte views immediately at
   the vendor boundary. This retains the official data flow and confines the
   semantic exception to HCU.
2. **Rejected: checkpoint allowlist.** It would require code changes for every
   model and confuse checkpoint accuracy policy with a backend byte contract.
3. **Rejected: vLLM global FlashAttention change.** Upstream intentionally
   rejects E5M2 for its NVIDIA FlashAttention implementation; globally
   weakening that rule would affect unrelated platforms.

## Validation

- Portable contract tests prove capability advertisement, constructor state,
  BF16 query preservation, exact E5M2 K/V view, scale forwarding, and unchanged
  BF16/E4M3 behavior.
- Live HCU tests cover official writer to vendor reader for both layouts,
  prefill/decode, MHA/GQA, storage padding, non-unit blocks, and changed-input
  Graph replay against a Torch reference built from the actually quantized K/V.
- Build a fresh wheel, install it into an isolated target, run package and
  installed-state gates, then repeat the HCU matrix from that installation.
- Model A/B rows use otherwise identical BF16-KV and E5M2-KV commands with
  default Graph mode. At minimum: Qwen dense, Meta-Llama dense, and one
  non-Qwen MoE checkpoint using the required AITER MoE backend. Keep raw sample
  IDs, outputs, finish reasons, token counts, errors, and HumanEval results.
- Models that select MLA/sparse MLA or require an already unsupported
  multimodal/hybrid feature remain separate backend rows; they are not evidence
  for or against this ordinary FlashAttention capability.

## Completion rule

The backend change may be proposed only after portable and HCU gates pass from
the final wheel and at least Qwen plus Meta-Llama complete BF16/E5M2 A/B. The
non-Qwen MoE row must be `pass` or explicitly `blocked` with a reproduced
model/topology/provider reason; a dense model cannot substitute for it. No
cross-model accuracy regression is accepted without an explicit project/user
tolerance.
