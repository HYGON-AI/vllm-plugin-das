# SPDX-License-Identifier: Apache-2.0
"""Source injection shared by CPU contracts and compiled hipBLASLt tests."""

BLASLT_FAILURE_SITES = [
    ("hipblasLtMatmul", 0, 1),
    ("hipblasLtMatmulDescCreate", 0, 1),
    ("hipblasLtMatrixLayoutCreate", 0, 3),  # A: no layouts created yet
    ("hipblasLtMatrixLayoutCreate", 1, 3),  # B: A must be cleaned up
    ("hipblasLtMatrixLayoutCreate", 2, 3),  # C: A and B must be cleaned up
]
BLASLT_FAILURE_IDS = ["matmul", "descriptor", "layout-a", "layout-b", "layout-c"]
BLASLT_FAILURE_STATUSES = [
    "HIPBLAS_STATUS_NOT_SUPPORTED", "HIPBLAS_STATUS_INVALID_VALUE",
]


def failing_blaslt_loader(load_inline, api, status, occurrence, expected_calls):
    def injected_extension(**kwargs):
        source = kwargs["cuda_sources"]
        call = api + "("
        assert source.count(call) == expected_calls
        assert 0 <= occurrence < expected_calls
        parts = source.split(call)
        prefix = call.join(parts[:occurrence + 1])
        suffix = call.join(parts[occurrence + 1:])
        kwargs["cuda_sources"] = (
            prefix + f"([](auto... args) {{ return {status}; }})(" + suffix
        )
        kwargs["name"] += f"_{api}_{occurrence}_{status}"
        return load_inline(**kwargs)

    return injected_extension
