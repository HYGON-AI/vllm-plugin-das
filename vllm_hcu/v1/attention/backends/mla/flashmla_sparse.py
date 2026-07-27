# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend

class HcuFlashMLASparseBackend(FlashMLASparseBackend):
    
    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE"