"""HCU attention providers for the vLLM adapter extension."""

from .flash_attn import HcuFlashAttentionBackend

__all__ = ["HcuFlashAttentionBackend"]
