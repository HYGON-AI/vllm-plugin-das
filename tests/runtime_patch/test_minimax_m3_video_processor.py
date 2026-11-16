# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from vllm_hcu.patch.import_coordinator import ExactImportCoordinator
from vllm_hcu.patch.platform.core_fix import (
    patch_minimax_m3_video_processor as patch,
    platform_core_callback_names,
)
from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError
from vllm_hcu.patch.runtime_state import PatchRegistry


def _module(processor: type) -> ModuleType:
    module = ModuleType(patch.TARGET_MODULE)
    module.MiniMaxM3VLVideoProcessor = processor
    return module


def _legacy_processor() -> type:
    class VideoProcessor:
        do_convert_rgb = True

        def _preprocess(self, videos, do_convert_rgb, scale):
            # Like MiniMax-M3, do_convert_rgb is unused in this implementation.
            return [pixel * scale for pixel in videos]

        def preprocess(self, videos, **kwargs):
            # Reproduce Transformers consuming RGB before the subclass call.
            rgb = kwargs.pop("do_convert_rgb", self.do_convert_rgb)
            prepared = [pixel + 1 if rgb else pixel for pixel in videos]
            return self._preprocess(videos=prepared, **kwargs)

    return VideoProcessor


@pytest.mark.parametrize("convert_rgb", (True, False))
def test_base_preprocess_consumes_rgb_without_missing_argument_or_double_conversion(
    convert_rgb,
):
    processor = _legacy_processor()
    instance = processor()
    with pytest.raises(TypeError, match="do_convert_rgb"):
        instance.preprocess([1, 2], scale=3, do_convert_rgb=convert_rgb)

    module = _module(processor)
    original = processor._preprocess
    assert patch.apply_to_module(module) is True
    expected = [6, 9] if convert_rgb else [3, 6]
    assert instance.preprocess([1, 2], scale=3, do_convert_rgb=convert_rgb) == expected
    assert processor._preprocess.__wrapped__ is original
    assert patch.apply_to_module(module) is False


def test_explicit_legacy_argument_is_preserved_for_positional_and_keyword_calls():
    class VideoProcessor:
        do_convert_rgb = True

        def _preprocess(self, videos, do_convert_rgb, scale):
            return videos, do_convert_rgb, scale

    patch.apply_to_module(_module(VideoProcessor))
    instance = VideoProcessor()
    assert instance._preprocess([1], False, 2) == ([1], False, 2)
    assert instance._preprocess([1], do_convert_rgb=False, scale=2) == ([1], False, 2)
    assert instance._preprocess([1], scale=2) == ([1], True, 2)
    with pytest.raises(TypeError, match="scale"):
        instance._preprocess(videos=[1])


@pytest.mark.parametrize("removed", (True, False))
def test_already_fixed_processor_is_not_wrapped(removed):
    class VideoProcessor:
        do_convert_rgb = True

        if removed:
            def _preprocess(self, videos, scale):
                return [pixel * scale for pixel in videos]
        else:
            def _preprocess(self, videos, scale, do_convert_rgb=True):
                return [pixel * scale for pixel in videos]

    original = VideoProcessor._preprocess
    module = _module(VideoProcessor)
    assert patch.apply_to_module(module) is False
    assert patch.apply_to_module(module) is False
    assert VideoProcessor._preprocess is original
    assert VideoProcessor()._preprocess([2], scale=3) == [6]


def test_patch_is_registered_in_platform_phase_without_importing_target():
    assert (patch.PATCH_ID, patch.TARGET_MODULE) in platform_core_callback_names()
    assert patch.TARGET_MODULE not in sys.modules


def test_exact_import_callback_repairs_an_already_loaded_processor(monkeypatch):
    processor = _legacy_processor()
    module = _module(processor)
    monkeypatch.setitem(sys.modules, patch.TARGET_MODULE, module)
    coordinator = ExactImportCoordinator(registry=PatchRegistry())
    coordinator.register_callback(
        patch.PATCH_ID, patch.TARGET_MODULE, patch.apply_to_module, targets=patch.TARGETS
    )
    assert processor().preprocess([1], scale=3) == [6]


def test_wrong_module_and_missing_processor_fail_visibly():
    with pytest.raises(PatchCompatibilityError, match="expected module"):
        patch.apply_to_module(ModuleType("unrelated"))
    with pytest.raises(PatchCompatibilityError, match="missing"):
        patch.apply_to_module(ModuleType(patch.TARGET_MODULE))
