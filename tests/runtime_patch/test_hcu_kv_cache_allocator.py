from types import SimpleNamespace

import torch

import vllm_hcu.v1.kv_cache as kv_cache


class _AttentionSpec:
    block_size = 576
    storage_block_size = 576
    num_kv_heads = 2
    head_size = 256
    dtype = torch.bfloat16
    kv_quant_mode = SimpleNamespace(name="NONE")


class _MambaSpec:
    page_size_bytes = 4096
    shapes = ((128,), (256,))
    dtypes = (torch.float32, torch.float32)


class _FlashBackend:
    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_heads, head_size, **_kwargs):
        return (num_blocks, 2, block_size, num_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order():
        return (0, 1, 2, 3, 4)


_FlashBackend.__module__ = "vllm_hcu.v1.attention.backends.flash_attn"


def test_official_flash_cache_is_reinterpreted_without_new_storage(monkeypatch):
    physical_cache = torch.zeros((27, 64, 2, 512), dtype=torch.bfloat16)
    official_cache = physical_cache.permute(0, 2, 1, 3)
    monkeypatch.setattr(
        torch,
        "empty_strided",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stride calculation must not allocate a tensor")
        ),
    )

    flash_cache = kv_cache._reshape_hcu_flash_cache(
        official_cache,
        block_size=64,
        num_kv_heads=2,
        head_size=256,
    )

    assert flash_cache.shape == (27, 2, 64, 2, 256)
    assert flash_cache.untyped_storage().data_ptr() == official_cache.untyped_storage().data_ptr()
    key_cache, value_cache = flash_cache.unbind(1)
    assert key_cache.stride() == (65536, 512, 256, 1)
    assert value_cache.stride() == key_cache.stride()
