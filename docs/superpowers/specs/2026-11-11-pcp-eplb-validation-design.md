# PCP EPLB validation compatibility

## Intent

Allow the HCU plugin's validated HYV4 static EPLB producer shape
`TP1/DP1/PCP16` to pass vLLM 0.25.1's generic `ParallelConfig` validation.
Today the plugin accepts this shape later in its HCU contract checks, but the
upstream Pydantic validator rejects it first because it counts only TP and DP.

## Constraints

- Keep the change inside `vllm-plugin-das`; do not modify installed vLLM files.
- Preserve upstream behavior for non-PCP inputs and for unsupported platforms.
- Preserve rejection when expert parallelism is disabled or TP/DP/PCP are all 1.
- Preserve `num_redundant_experts`, including the HY4 R16 value of 16.
- Fail closed if the pinned vLLM/Pydantic validator shape changes.

## Approach

Wrap the `ParallelConfig._validate_parallel_config` after-validator with a
dedicated `patch_parallel_config` adapter in the plugin's platform core-fix
lifecycle. Update its Pydantic decorator metadata and rebuild the dataclass so
both direct construction and nested validation use the wrapper. Fields and
`__post_init__` run unchanged, including nested EPLB constraints and communicator
selection. For `TP1/DP1/PCP>1 + EPLB + EP` on a CUDA-like HCU platform, only the
after-validator sees EPLB temporarily disabled and a copied EPLB config with
zero redundant experts. Restore both fields in `finally`. Other inputs delegate
unchanged. Restore the original compiled state if rebuilding fails.

Changing upstream vLLM was rejected because the fix must ship in the plugin.
Rewriting vLLM files at install time was rejected because this repository's
post-install contract is read-only. Replacing only the method is insufficient:
the Pydantic decorator metadata must also point at the new function.

## Validation

Use the existing HYV4 PCP static EPLB construction tests as the primary
regression. Add direct tests for R16 preservation, PCP1 rejection, EP-disabled
rejection, nested validation, communicator initialization, remaining DCP checks,
invalid field rejection, and idempotent patch application. Finally launch the
real P command without `PYTHONPATH` and confirm
it advances past `ParallelConfig` construction with the offline R16 map.
