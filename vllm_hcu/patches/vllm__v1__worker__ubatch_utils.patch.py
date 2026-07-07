# SPDX-License-Identifier: Apache-2.0

"""
Patch for vllm.v1.worker.ubatch_utils.

Preserve DBO attention metadata while slicing ubatches and normalize slice
boundaries to Python ints so downstream Triton specialization does not see
numpy scalar values.
"""

PATCHES = [
(
"""
    padded_last_request_slice = slice(last_slice.request_slice.start, num_reqs_padded)
    padded_last_token_slice = slice(last_slice.token_slice.start, num_total_tokens)
""",
"""
    padded_last_request_slice = slice(
        int(last_slice.request_slice.start), int(num_reqs_padded)
    )
    padded_last_token_slice = slice(
        int(last_slice.token_slice.start), int(num_total_tokens)
    )
""",
),
(
"""
    if split_point is None:
        split_point = int(num_tokens_padded) // num_ubatches

    token_split_points = [split_point * i for i in range(1, num_ubatches)]
""",
"""
    num_tokens_padded = int(num_tokens_padded)
    num_reqs_padded = int(num_reqs_padded)
    num_ubatches = int(num_ubatches)

    if split_point is None:
        split_point = num_tokens_padded // num_ubatches

    if isinstance(split_point, list):
        token_split_points = [int(point) for point in split_point]
    else:
        split_point = int(split_point)
        token_split_points = [split_point * i for i in range(1, num_ubatches)]
""",
),
(
"""
    all_points = token_split_points + [cu_num_tokens[-1]]

    for end_token in all_points:
        token_slice = slice(start_token, end_token)
""",
"""
    all_points = token_split_points + [int(cu_num_tokens[-1])]

    for end_token in all_points:
        end_token = int(end_token)
        token_slice = slice(start_token, end_token)
""",
),
(
"""
    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    return CommonAttentionMetadata(
""",
"""
    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]
    positions = (
        attn_metadata.positions[token_slice]
        if attn_metadata.positions is not None
        else None
    )
    is_prefilling = (
        attn_metadata.is_prefilling[request_slice]
        if attn_metadata.is_prefilling is not None
        else None
    )

    return CommonAttentionMetadata(
""",
),
(
"""
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
""",
"""
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        positions=positions,
        is_prefilling=is_prefilling,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
""",
),
]
