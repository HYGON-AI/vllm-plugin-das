from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend

class HcuFlashMLASparseBackend(FlashMLASparseBackend):
    
    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE"