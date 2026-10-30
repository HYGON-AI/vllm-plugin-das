# vLLM 0.32 HCU FlashAttention Design

## Intent

Make explicit `--attention-backend FLASH_ATTN` correct on HCU with Graph
enabled, while keeping official vLLM orchestration upstream-owned, optional
optimization code in `vllm-plugin-das`, and host selection in `adapter-das`.

## Root cause

vLLM 0.32 allocates each FlashAttention page as a logical
`[B, H, N, 2D]` tensor and splits the final dimension into K and V views.
Those views have a `2D` row stride. The installed HCU vendor
`flash_attn_varlen_func` accepts NHD/HND pages but reads dense K and V pages;
it does not honor the packed split-view stride. Live HCU probes showed:

- official and HCU-native cache writers both matched the Torch reference
  exactly for NHD and HND;
- vendor varlen attention on separate dense pages matched the Torch reference
  within BF16 tolerance (`max_abs <= 0.0078125`); and
- the same vendor call on vLLM-style packed split views diverged by
  `max_abs 1.9375-2.1875`.

## Selected approach

Add two protected extension seams to the OpenDAS vLLM 0.32
`FlashAttentionImpl`: one that derives K/V views from the allocated cache and
one that invokes the attention kernel. The default implementations preserve
official behavior exactly.

The Plugin subclasses the official implementation. It reinterprets each flat
LBHNC page as dense `[K/V, H, N, D]` storage, uses the official stride-aware
HIP cache writer, and invokes only the public vendor
`flash_attn_varlen_func` with `layout="bhsd"`. Unsupported FA4-only optional
arguments are accepted only when inactive (`None`); active use fails closed.
The backend advertises a 64-token kernel page and LBHNC layout.

The Adapter maps explicit `FLASH_ATTN` selection to the Plugin backend class.
It contains no FlashAttention numerical logic and does not import the Plugin
at bootstrap time.

## Alternatives considered

1. Copy the complete official FlashAttention forward into the Plugin. Rejected
   because it would hide future upstream fixes and create a large maintenance
   fork.
2. Copy/repack K and V on every forward. Rejected because it adds a full-cache
   copy to the hot path and weakens Graph and performance behavior.
3. Monkeypatch vLLM module globals. Rejected because import order and global
   state would make ownership and rollback ambiguous.

## Scope and constraints

- Target release line: vLLM `>=0.32.0,<0.33.0`; no exact wheel lock.
- Final validation keeps Graph enabled; eager is diagnostic only.
- Initial supported path is BF16/FP16, LBHNC, 64-token pages, single-card
  dense MHA/GQA, and the vendor varlen ABI.
- FP8 KV, DCP, hybrid caches, long-prefix reuse, and performance promotion are
  not implied until their own matrix rows pass.
- The Adapter change must remain a class-path routing seam only.
- The Plugin path is removable by selecting another official backend or
  uninstalling the Plugin.

## Success criteria

1. Portable tests prove default vLLM seams preserve existing packed views.
2. Plugin tests prove dense K/V views do not overlap, preserve block stride,
   and reject active unsupported FA4 features.
3. Live writer-to-reader probes match a Torch reference for prefill and
   decode, including changing-input Graph replay.
4. `/model/Qwen2-0.5B-Instruct` starts with explicit FLASH_ATTN and Graph,
   generates coherent output matching the TRITON control at the protocol
   level, and completes HumanEval16 with raw finish reasons retained.
