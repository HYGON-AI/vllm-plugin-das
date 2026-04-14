from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend

class HcuTritonMLABackend(TritonMLABackend):
    
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"