# Qwen3.5 TP2 LightOp W16A16 Design

## Goal

Make the opt-in LightOp W16A16 MoE path safe for Qwen3.5-35B-A3B in pure
TP2 on gfx936/80 CU, including the requested scheduler token budget of 8192,
without changing other MoE models or explicit backend choices.

## Root cause

vLLM 0.28.1 allocates `workspace13` and the modular expert output from one
common buffer.  The current adapter also places GEMM2's `cache3` in
`workspace13`, then asks LightOp reduction to read `cache3` while writing the
aliased output.  The isolated operator test used a separately allocated output
and therefore missed this overlap; the real model profile exposes it as an HCU
VMFault.

Qwen3.5 TP2 has `(E,K,N,top-k)=(256,2048,256,8)`, while the current eligibility
gate admits only `N=512`.  The installed LightOp package contains both required
gfx936/80-CU config files, but their M=1 records fail while the M=2 records pass
the same one-token workload.  LightOp chooses the first configured M greater
than or equal to the request, so requesting M=2 for this exact M=1/device/shape
case is a bounded compatibility correction.

## Design

Keep the two vLLM-managed allocations and change only their lifetimes:

- `workspace2` stores `cache1`, then `cache3` after activation;
- `workspace13` stores `cache2`, then the aliased final output after GEMM2;
- no simultaneously live input/output views overlap and total scratch size is
  unchanged for the Qwen3.5 TP2 dimensions.

Admit only the existing validated shapes: the prior N=512 shape and Qwen3.5
TP2 N=256.  Keep BF16, SiLU, no-bias, pure-TP and no-EP constraints.  Allow the
Qwen3.5 shape a maximum scheduler budget of 8192 while preserving the prior
N=512 limit of 16.  Backend selection probes M=1 and the maximum budget rather
than every integer because LightOp's config status is file/shape based and its
API maps intermediate M values to the next configured bucket.

For gfx936/80 CU and the exact Qwen3.5 TP2 problem, map only `expected_m=1` to
the package's M=2 config.  Other architectures and shapes retain their native
M=1 lookup.

## Validation boundary

CPU contract tests must prove view non-overlap, shape/range gating, and bounded
config lookup.  HCU tests must cover aliased-output execution, changed-input
graph replay, model profile/startup, one real request, and a short performance
comparison.  If model startup still faults, the new shape remains unmerged and
the branch records the failing evidence.
