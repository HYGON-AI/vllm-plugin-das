# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared stamp constants for the HCU triton MLA non-causal contract.

Dependency-free by design: the worker patch framework imports every
framework_opt adapter at registration time (lazy-arming contract -- no
patch target module may be imported there), so the producer
(``patch_dspark_non_causal_stamp``) and the consumer
(``vllm_hcu.v1.attention.backends.mla.triton_mla``, which re-exports the
constant) must both be able to resolve the stamp without pulling torch or
vLLM internals.
"""

NON_CAUSAL_DRAFT_STAMP = "_vllm_hcu_non_causal_draft"

__all__ = ["NON_CAUSAL_DRAFT_STAMP"]
