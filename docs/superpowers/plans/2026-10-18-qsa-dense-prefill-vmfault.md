# QSA Dense-Prefill VMFault Fix Plan

**Goal:** Prevent the reproducible HCU invalid-address VMFault in batched
prefix/chunked prefill while retaining dense FlashAttention for context-free
prefill.

**Root-cause boundary:** A single 22,001-token history/chunked request succeeds,
but 16 concurrent requests (four sequences per DP rank) fault inside the
plugin's non-public fixed-stride FlashAttention path. That path combines
fixed-stride `cu_seqlens_k` with `seqused_k`; the public vendor wrapper does not
support that non-paged combination.

## Task 1: Lock in safe routing

- Change the routing test so all prefix/chunked prefill retains QSA.
- Run the focused test and confirm it fails against the current implementation.
- Restrict dense prefill to context-free batches and remove the unsafe history
  gather/native-ABI path.
- Run focused runtime-patch and environment-routing tests.

## Task 2: Verify on HCU

- Reinstall the plugin with `--force-reinstall --no-deps`.
- Restart with dense prefill and prefix caching enabled.
- Verify a context-free request still routes to dense FlashAttention.
- Repeat the exact 16-concurrent, 22K-token reproduction and verify all
  requests succeed without VMFault.

