from types import ModuleType

import pytest

from vllm.platforms import current_platform
from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_c128_splits as patch
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


def target():
    module = ModuleType(patch.TARGET_MODULE)
    def native(num_actual, compress_ratio, head_dim):
        return 2 if num_actual == 4096 else 1
    module._pick_compress_num_splits = native
    return module


@pytest.mark.parametrize("rows,ratio,head,rocm,master,splits,expected", [
    (32768, 128, 512, True, True, 4, 4),
    (4096, 128, 512, True, True, 4, 4),
    (4096, 128, 512, True, True, 0, 2),
    (32768, 128, 512, True, False, 4, 1),
    (32768, 128, 512, False, True, 4, 1),
    (32768, 4, 512, True, True, 4, 1),
    (32768, 128, 128, True, True, 4, 1),
    (4095, 128, 512, True, True, 4, 1),
    (0, 128, 512, True, True, 4, 1),
])
def test_dispatch_guards_and_native_rollback(monkeypatch, rows, ratio, head, rocm, master, splits, expected):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: rocm)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1" if master else "0")
    monkeypatch.setenv("VLLM_HCU_C128_HEAD_SPLITS", str(splits))
    module = target()
    assert patch.apply_to_module(module)
    assert module._pick_compress_num_splits(rows, ratio, head) == expected
    assert not patch.apply_to_module(module)


def test_stale_marker_and_wrong_signature_rejected():
    module = target()
    patch.apply_to_module(module)
    module._pick_compress_num_splits = lambda *args: 1
    with pytest.raises(PatchCompatibilityError):
        patch.apply_to_module(module)
    module = target()
    module._pick_compress_num_splits = lambda rows: 1
    with pytest.raises(PatchCompatibilityError):
        patch.apply_to_module(module)


@pytest.mark.parametrize("splits", [-1, 3, 32])
def test_invalid_tiles_rejected(monkeypatch, splits):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_C128_HEAD_SPLITS", str(splits))
    module = target()
    patch.apply_to_module(module)
    with pytest.raises(ValueError, match="HEAD_SPLITS"):
        module._pick_compress_num_splits(32768, 128, 512)
