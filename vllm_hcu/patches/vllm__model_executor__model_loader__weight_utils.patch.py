# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.model_loader.weight_utils safetensors_weights_iterator
"""

PATCHES = [
    (
        'loading_desc = "Loading safetensors checkpoint shards"',
        'loading_desc = "Loading safetensors checkpoint shards"\n    print("start load model")',
    ),
    (
        'sorted_files = sorted(hf_weights_files, key=_natural_sort_key)',
        'sorted_files = sorted(hf_weights_files, key=_natural_sort_key)\n    print("sorted model")',
    ),
]