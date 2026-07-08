from typing import ClassVar

from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend

class HcuTritonMLABackend(TritonMLABackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]
    
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"
