# v0.25.1 → v0.28.1-dev source-only commit disposition

Frozen comparison: `origin/v0.25.1@6ea7b12` versus
`origin/v0.28.1-dev@9302165`, refreshed 2026-09-25. The 28 rows below are
the complete non-merge `target..source` set. “Present” means a target owner
exists; it is not an accuracy or performance claim. Device results and final
equivalence rulings belong in `hy4_v0281_validation.md`.

| Commit | Functional owner / target gap | Disposition and selected files / gate |
| --- | --- | --- |
| `6ea7b12` | DTK container version label | Release-only; exclude container edit. Verify installed DTK/provenance in validation record. |
| `3f30e7c` | Hy4 PCP+EP DeepEP wait | Optional PCP/EP excluded; TP8 Hy4 uses `models/hy_v4/{moe,model}.py`, PCP/DCP rejected. |
| `df99a09` | Hy4 LightOp mask/top-k experiments | Use final sparse indexer behavior through `models/hy_v4/attention.py`; removed regressive/optional LightOp branches excluded. Sparse tests and TP8 live gate. |
| `4f1275d` | DTK 26.04 image build | Release-only; no image change. |
| `534162d` | CI quality gate workflow | CI-only; exclude. |
| `2b7de5f` | Hy4 PCP+EP and profile memory | TP8-only model adaptation selected; PCP/EP excluded pending Graph/accuracy gate. |
| `4f1f266` | V1 steady-decode scheduler and M-RoPE | Scheduler registry owner differs in MRV2; do not port V1 steady-decode or change default without same-workload evidence. M-RoPE candidate remains unverified for this model; target scheduler review in common Task 4. |
| `904f4e2` | Kimi-K3 boltops pin | Independent model; follow-up MR. |
| `7cdd654` | Hy4 PCP-sharded linear gate | TP-only gate runs `models/hy_v4/attention.py`; PCP path excluded. |
| `e9f4e2d` | Kimi-K3 INT4 W4A8 / TP16 | Independent model; follow-up MR. |
| `ce2b65f` | Scheduled CI trigger | CI-only; exclude. |
| `2c73085` | Docker-image scheduler | Release-only; exclude. |
| `5e68a65` | Hy4 config/model/MTP/iHC/MoE | Adapt final model behavior in `models/hy_v4/{attention,hc,moe,model,mtp}.py`; target vLLM owns config and speculative conversion. Hy4 unit and live gates. |
| `bc93290` | EvalScope CI harness | CI-only; retain a narrow reproducible HumanEval/0-7 harness in `tests/integration/server`. |
| `9218102` | DeepSeek-V4 INT8 and generic sparse-indexer fallback memory | DeepSeek-only changes excluded; bounded generic fallback selected for `v1/attention/ops/rocm_aiter_mla_sparse.py` and dedicated parity/Graph tests in common Task 3. |
| `8aebc86` | CI Torch 2.11 lock | CI-only; installed Torch 2.11 recorded, no workflow edit. |
| `1ea04f2` | Qwen3 DSpark NN Graph | Independent model; follow-up MR. |
| `bf8633d` | PyPI URL release fix | Release-only; exclude. |
| `11029ca` | Python 3.12 packaging | Release-only; no packaging expansion. |
| `1993931` | CI model timeout | CI-only; exclude. |
| `b701be3` | V1 PCP+EP / DeepEP-LL bubble and communicator hooks | Target `ParallelConfig`/MRV2 communicator owners differ; do not port V1 `DeviceCommunicatorBase.__init__` override. Existing target PCP+EP tests and fail-closed checks in common Task 4. |
| `15b7cd8` | Shared CI datasets/artifacts | CI-only; exclude. |
| `39bc3a3` | CI node and AICC install | CI-only; exclude. |
| `8565e54` | LightOp/SGLang operator routes | Target already has `sqrtsoftplus_routing`, MLA concat, gated RMSNorm, W16A16 routing; compare signatures/guards with target runtime tests in common Task 4 before declaring equivalence. |
| `325dee8` | GDN/FLA kernel routes | Target has GDN/FLA patch owners; run `test_attention_mla_fla_mamba.py` and inspect current dispatch in common Task 4. No duplicate port without a failing contract. |
| `88f8a07` | CI notification/database and image | CI/release-only; exclude. |
| `021625d` | SlimQuant, MoE routing, native FP8 KV, scale guards | Target has `valid_token_counts`, native KV writer, SlimQuant and MoE owners. Run `test_hcu_cache_kernel_source.py`, `test_moe_deepep.py`, quant/runtime tests in common Task 4; port only an evidenced missing call site. |
| `50c8f55` | CI Qwen max length | CI-only; exclude. |

## Common-runtime review result

The selected generic part of `9218102` is adapted in
`rocm_aiter_mla_sparse.py`: decode gathers bounded page chunks without host
`.item()` reads, and prefill tiles both query and key axes. Its 18 numerical,
workspace and HIP Graph tests plus 19 adjacent loading tests passed on this
target (`37 passed`). The DeepSeek-V4-specific branch of that commit is not
ported.

For `021625d`, the target already has separate SlimQuant, valid-token MoE,
native FP8 KV writer and scale-guard owners. For `8565e54`, it already has
LightOp MLA concat, gated RMSNorm, sqrtsoftplus routing and W16A16 MoE
owners. For `325dee8`, it already has Qwen GDN and FLA adapters. A focused
target suite covering those owners, DeepEP and PCP configuration passed
`212/212` after a current-target fake-config fixture was repaired. That is
contract evidence, **not** a claim of identical performance or numerical
parity to every v0.25.1 kernel. No untested duplicate kernel is selected for
this MR.

`b701be3` is V1 communicator/bubble code and is intentionally excluded from
the MRV2 path; the target's `ParallelConfig` and HCU MRV2 runner own the
equivalent integration points. `4f1f266`'s steady-decode scheduler is also
V1-specific (TP1, no prefix caching, no MTP), so it is excluded from the TP8
Hy4 run; its independent M-RoPE component remains unverified rather than
declared equivalent. Neither change is an implicit scheduler default.

No source commit is cherry-picked wholesale. The binary HCU extension is not
part of this source diff; the live validation record distinguishes an
installed-binary overlay from a newly built artifact.
