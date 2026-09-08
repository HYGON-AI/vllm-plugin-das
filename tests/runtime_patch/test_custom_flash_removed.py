from vllm_hcu.patch.platform.core_fix import patch_engine_args


def test_custom_flash_backend_alias_is_removed():
    assert "FLASH_ATTN_CUSTOM" not in patch_engine_args._HCU_FLASH_ATTN_ALIASES
