# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU integration boundary for vLLM's Model Runner V2."""

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class HcuGPUModelRunnerV2(GPUModelRunner):
    """Thin HCU adapter around the upstream v0.25.1 Model Runner V2."""

    pass
