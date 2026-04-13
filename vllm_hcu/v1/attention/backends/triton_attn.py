# SPDX-License-Identifier: Apache-2.0
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

class HcuTritonAttentionBackend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"
