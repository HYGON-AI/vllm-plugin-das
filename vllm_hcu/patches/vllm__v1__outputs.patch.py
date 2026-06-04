# SPDX-License-Identifier: Apache-2.0

"""
Patch vllm.v1.outputs with qwen3.5 PP+MTP fixes from vLLM commit
fc551867.
"""

PATCHES = [
(
"""
    sampled_token_ids: list[list[int]] = field(default_factory=list)
""",
"""
    sampled_token_ids: list[list[int]] = field(default_factory=list)
    # num_reqs x num_spec_tokens
    spec_token_ids: list[list[int]] | None = None
""",
),
(
"""
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        pooler_output=pooler_output,
    )
""",
"""
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        spec_token_ids=None,
        pooler_output=pooler_output,
    )
""",
),
(
"""
EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={})
""",
"""
EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
    req_ids=[], req_id_to_index={}, spec_token_ids=None
)
""",
),
]
