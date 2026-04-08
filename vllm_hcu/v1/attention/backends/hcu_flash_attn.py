# SPDX-License-Identifier: Apache-2.0
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

class HcuFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"