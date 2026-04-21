from vllm.logger import init_logger

logger = init_logger(__name__)


def patch_unified_kv_cache_update() -> None:
    """Patch unified_kv_cache_update in-place for FLASH_ATTN tuple kv_cache."""
    try:
        import torch
        import vllm.model_executor.layers.attention.attention as attn_mod

        old_fn = attn_mod.unified_kv_cache_update

        def _patched_unified_kv_cache_update(
            key: torch.Tensor,
            value: torch.Tensor,
            layer_name: str,
        ) -> torch.Tensor:
            import torch
            import vllm_hcu.platforms.envs as henvs
            import vllm.model_executor.layers.attention.attention as patched_attn_mod

            _, attn_layer, kv_cache, layer_slot_mapping = (
                patched_attn_mod.get_attention_context(layer_name)
            )
            if layer_slot_mapping is not None:
                assert hasattr(attn_layer.impl, "do_kv_cache_update"), (
                    f"{attn_layer.impl.__class__.__name__} "
                    "does not support kv cache update"
                )
                attn_layer.impl.do_kv_cache_update(
                    attn_layer,
                    key,
                    value,
                    kv_cache,
                    layer_slot_mapping,
                )

            if henvs.VLLM_HCU_USE_FLASH_ATTN and isinstance(kv_cache, tuple):
                return torch.empty(0, device=key.device, dtype=key.dtype)
            return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)

        old_fn.__code__ = _patched_unified_kv_cache_update.__code__
        old_fn.__defaults__ = _patched_unified_kv_cache_update.__defaults__
        old_fn.__kwdefaults__ = _patched_unified_kv_cache_update.__kwdefaults__
    except Exception as e:
        logger.debug(f"Skip runtime unified_kv_cache_update patch: {e}")
