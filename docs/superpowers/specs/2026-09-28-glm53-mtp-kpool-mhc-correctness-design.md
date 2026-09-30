# GLM-5.3 MTP Kpool and mHC Correctness Design

## Goal

Restore correct GLM-5.3-Flash inference on the v0.28.1-dev target for
MTP3 under both `VLLM_HCU_USE_CUSTOM_OPS` states.

## Required behavior

- A kpool tail ring must retain the current pool plus every speculative draft
  until acceptance is known. For `index_kpool=4` and MTP3, the physical ring
  is 8 entries, while pool completion remains four-token granular.
- The tail ring must divide the attention cache block size so prefix-cache
  block alignment is preserved.
- Tail seeding and decode updates must address physical slots by ring width,
  while completion phase and compression remain keyed by pool width.
- With `VLLM_HCU_USE_CUSTOM_OPS=0`, GLM-5.3 decoder construction must bind the
  official vLLM native mHC equations and must not enter BoltOPs or local
  TileLang mHC kernels.
- With `VLLM_HCU_USE_CUSTOM_OPS=1`, the existing BoltOPs mHC path remains
  unchanged.
- Explicit MoE backend selection remains unchanged. The current GLM shape may
  continue to fall back from the AITER selector to vLLM Triton MoE when AITER
  has no tuned solution.

## Upstream alignment

Backport the behavior of official vLLM commits `2617fe9383` (kpool rejected
draft corruption) and `5fcc6e7c73` (ROCm wave64 mHC correctness) only where
the frozen v0.28.1 target lacks it. Master-off uses the official torch-native
mHC implementation instead of carrying another TileLang variant.

## Validation

- CPU/runtime-patch tests cover pool4/MTP3 ring sizing, kernel launch ring
  metadata, rejected-draft redo, and master-off/on production mHC binding.
- HCU validation covers TP4 MTP3 with master off and on, prefix-cache reuse,
  repeated decoding, and HMMT25 accuracy.
