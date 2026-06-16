# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# set_forward_context: skip dp_metadata when deepep low-latency all2all.
# 注意：import 必须顶格，与 forward_context.py 一致，不能写成 8 个空格开头。

PATCHES = [
    (
        """    if (
        vllm_config.parallel_config.data_parallel_size > 1
        and vllm_config.parallel_config.is_moe_model is not False
        and (attn_metadata is not None or num_tokens is not None)
    ):""",
        """    if (
        vllm_config.parallel_config.data_parallel_size > 1
        and vllm_config.parallel_config.is_moe_model is not False
        and (attn_metadata is not None or num_tokens is not None)
        and vllm_config.parallel_config.all2all_backend != "deepep_low_latency"
    ):""",
    ),

################ lightly cp###########################
(
'''
    additional_kwargs: dict[str, Any] = field(default_factory=dict)
''',
'''
    additional_kwargs: dict[str, Any] = field(default_factory=dict)

    scatter_indexes_tensor: torch.Tensor | None = None
    gather_indexes_tensor: torch.Tensor | None = None
    enable_lightly_cp: bool = False
    enable_lightly_cplb : bool = False
'''
),

(
'''
    additional_kwargs: dict[str, Any] | None = None,
    skip_compiled: bool = False,
''',
'''
    additional_kwargs: dict[str, Any] | None = None,
    skip_compiled: bool = False,
    scatter_indexes_tensor: torch.Tensor | None = None,
    gather_indexes_tensor: torch.Tensor | None = None,
    enable_lightly_cp: bool = False,
    enable_lightly_cplb: bool = False
'''
),

(
'''
        ubatch_slices=ubatch_slices,
        skip_compiled=skip_compiled,
''',
'''
        ubatch_slices=ubatch_slices,
        skip_compiled=skip_compiled,
        scatter_indexes_tensor=scatter_indexes_tensor,
        gather_indexes_tensor=gather_indexes_tensor,
        enable_lightly_cp=enable_lightly_cp,
        enable_lightly_cplb=enable_lightly_cplb,
'''
),

(
'''
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    skip_compiled: bool = False,
''',
'''
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    skip_compiled: bool = False,
    scatter_indexes_tensor: torch.Tensor | None = None,
    gather_indexes_tensor: torch.Tensor | None = None,
    enable_lightly_cp: bool = False,
    enable_lightly_cplb: bool = False,
'''
),

(
'''
        additional_kwargs,
        skip_compiled,
''',
'''
        additional_kwargs,
        skip_compiled,
        scatter_indexes_tensor,
        gather_indexes_tensor,
        enable_lightly_cp,
        enable_lightly_cplb
'''
),
################ lightly cp###########################
]
