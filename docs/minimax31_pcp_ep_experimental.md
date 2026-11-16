# MiniMax M3.1 PCP / EP experimental path

This opt-in path combines the same-ABI teacher MSA prefill kernels with fixed-count DeepEP routing and the existing Bolt local-expert computation. Defaults retain the legacy path.

## Activation and identity

The tested v5 used HCU_MINIMAX_NATIVE_MSA=1, HCU_MINIMAX_DEEPEP_FIXEDCOUNTS=1 and HCU_MINIMAX_FP8_PCP=1. Set HCU_MINIMAX_MSA_EXTENSION_DIR to the private same-ABI build_lib directory. The tested prefill kernel was v2, with MSA128_FP8_PREFILL_SLOTS=2. The historical Python 3.10 / Torch 2.11 binary is not a compatible replacement for the private Python 3.12 / Torch 2.13 build.

Create a fresh configuration and derive its real source/weight identity before loading this repository revision. Do not reuse a sidecar by rewriting its source key. The historical measured source remains in task_dir/support_minmax_31_pcp8_ep8/wheel_contract/isolated_plugin_sol_reuse_v5. The integration manifest proves equivalent runtime AST for the six integrated files; only comments/docstrings differ.

## Behavior and validation

The MSA bridge validates the actual uint8 cache layout and metadata, binds the cache without copying and preserves the stock FP8 scale semantics. DeepEP receives local expert IDs and exact per-expert counts. Pack/restore pointer products use explicit 64-bit arithmetic. Route counts remain runtime arguments to prevent count-specific JIT compilation. Shared-expert launch/wait follows the stock lifecycle, including empty/short ranks.

Only the private packed local-expert activation buffer enables Bolt inplace output. The original received hidden tensor and producer weights remain unchanged. This avoids an extra full route-capacity output allocation during the 65536-token startup profile; the stock internal GEMM chunk remains 32768.

Independent checks covered address offsets beyond 2^31, changing route counts/cache keys, shared-expert lifecycle, actual cache writer/stride, two synthetic expert inputs and two real checkpoint experts. Real-weight full-width tests against CPU FP64 achieved relative L2 errors 4.984e-7 and 6.692e-8. They do not constitute validation of every model expert.

The v5 full model passed 12 functional cases and startup with the full 65536-token budget. Five unprofiled target measurements had client completion median 14.426914 s, compared with 22.069 s for the frozen baseline. All target requests had prompt 147456, cached 131072 and output 1; all ranks processed 8192 local tokens. An independent eight-rank trace measured a slowest GPU span of 13.819251 s versus baseline 21.331863 s and SGLang 2.475923 s. The target has not been reached.

Token IDs matched the frozen baseline in 21 of 24 benchmark requests. Three differences remain unexplained; functional cases and small expert checks do not resolve full-model output equivalence. The native MSA path also changes BF16-Q arithmetic to FP8-Q/P; measured module differences must not be used to relax model quality requirements.

## Evidence and remaining work

Evidence is preserved under task_dir/support_minmax_31_pcp8_ep8_perf/prototypes/sol and wheel_contract/runs/sol_reuse_v5{,_trace}. The trace audit identifies about 5.236 s of Bolt GEMMs on rank 0 and 5.129 s in cached_notify_combine. Per-layer compute and notify wait are strongly anticorrelated across ranks; these durations are not independent wall-time savings. Route max/average is median 2.107, maximum 4.532.

At the v5 cutoff, the newer DeepGEMM HIPC single-GEMM probe (m_grouped_w4a8_gemm_nt_contiguous_hipc, candidate MODE1006; LightOP supplies quantization/OAI) passed an independent FP64 oracle. The full two-GEMM/OAI chain has not passed: its first run stopped at quantization comparison, before GEMM1. A revised half-integer-aware independent checker has not run on GPU. The earlier Marlin VM fault remains unproven and must not be treated as solved.

GPU resources were released on 2026-11-13. No consumer, producer or GPU test may be restarted until resources are explicitly reassigned. Release evidence is task_dir/support_minmax_31_pcp8_ep8_perf/release_preload_20261113.json.

Historical launcher, ELF symbol and trace evidence strongly identify the reference GEMM as DeepGEMM HIPC with an internal Marlin kernel. The candidate MODE1006 is not established as the historical configuration. Current source places weighted gather after second-GEMM BF16 output; the Bolt path weights before its final BF16 store, so numerical ordering remains a comparison boundary. Historical runtime maps were not captured.

The subsequent executable HIPC backend, real-expert module checks and new weight contract are documented in [minimax31_hipc_backend.md](minimax31_hipc_backend.md).

## Process lifecycle review (2026-11-16)

MiniMax quantization and PCP extension admission checks process identity before
acquiring any inherited locks. Active or failed publication state is refused in
fork children; a clean child gets its own lock. Use spawn for active extensions.
The experimental Kimi preload owner follows the same rule. It requires the
optional companion Adapter Kimi preload bridge and refuses clearly before any
publication when that dependency is absent; the MiniMax Adapter branch does not
provide Kimi preload support. Its registry unit test supplies an explicit bridge
double, and is not evidence of real preload integration.

The PCP producer supervisor watches all child sentinels after READY, rather than
joining the first live rank indefinitely. Any unrequested rank exit is an error.
Shutdown terminates all started workers, waits for a bounded grace period, kills
unresponsive workers and joins them. CPU tests use real spawn children, including
a later-rank failure and a child that ignores SIGTERM.
