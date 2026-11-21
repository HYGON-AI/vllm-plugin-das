# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Reject the retired root build on the vLLM 0.33 release line."""

raise SystemExit(
    "The repository-root legacy vllm_hcu build is disabled on v0.33.0. "
    "Build hcu-vllm-das-plugin 0.33.0 from the active project instead:\n"
    "  python -m pip wheel --no-deps --no-build-isolation "
    "--wheel-dir ./dist ./adapter_extension\n"
    "For historical vllm_hcu builds, use their original release branch."
)
