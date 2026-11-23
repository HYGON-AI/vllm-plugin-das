# SPDX-License-Identifier: Apache-2.0
"""Video config regressions; local files only, no model/tokenizer downloads.

RealHFLoaderTests calls public HF get_dict and standardization methods without Torch.
FactoryAstTests isolates the production factory/constructor with test doubles;
it does NOT test HF file resolution or tensor processing. RealHFLocalFilesTests
uses the actual plugin and HF loaders/constructors, and explicitly skips when
Torch/torchvision/Transformers/vLLM are not installed. Neither suite runs a model.
"""

import ast
import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4/processor.py"
TOKENS = {
    "image_start_token": "<image_start>",
    "image_end_token": "<image_end>",
    "image_token": "<image>",
    "image_newline_token": "<image_newline>",
    "video_start_token": "<video_start>",
    "video_end_token": "<video_end>",
    "video_token": "<video>",
}
VISION = {
    "patch_size": 16,
    "spatial_merge_size": 2,
    "spatial_patch_size": 1,
    "temporal_patch_size": 1,
}
IMAGE = {
    "size": {"shortest_edge": 4096, "longest_edge": 4194304},
    "image_mean": [0.48, 0.45, 0.40],
    "image_std": [0.26, 0.27, 0.28],
}
VIDEO = {
    "size": {"shortest_edge": 4096, "longest_edge": 33554432},
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "fps": 3,
    "max_frames": 80,
    "timestamp_token_reserve": 1000,
    "video_max_image_size": 768,
}


class ProcessorSettingsAssertions:
    def assert_settings(self, component, settings):
        for name, value in settings.items():
            actual = getattr(component, name)
            if name in ("image_mean", "image_std"):
                # HF canonicalizes JSON lists to tuples; contents must still match.
                self.assertSequenceEqual(actual, value, name)
            else:
                self.assertEqual(actual, value, name)


class SettingsAssertionsTests(ProcessorSettingsAssertions, unittest.TestCase):
    def test_mean_std_accept_lists_and_tuples(self):
        for actual_type in (list, tuple):
            for expected_type in (list, tuple):
                with self.subTest(actual=actual_type, expected=expected_type):
                    self.assert_settings(
                        SimpleNamespace(
                            **{
                                name: actual_type(IMAGE[name])
                                for name in ("image_mean", "image_std")
                            }
                        ),
                        {
                            name: expected_type(IMAGE[name])
                            for name in ("image_mean", "image_std")
                        },
                    )

    def test_mean_std_reject_wrong_values_order_length_and_scalars(self):
        for name in ("image_mean", "image_std"):
            expected = IMAGE[name]
            for actual in ([0.0] * 3, expected[::-1], expected[:-1], 0.5):
                with self.subTest(name=name, actual=actual):
                    with self.assertRaises(AssertionError):
                        self.assert_settings(
                            SimpleNamespace(**{name: actual}), {name: expected}
                        )

    def test_other_settings_still_require_exact_values(self):
        for name, expected, actual in (
            ("size", VIDEO["size"], IMAGE["size"]),
            ("fps", 3, 7),
            ("timestamp_token_reserve", 0, 1000),
            ("do_sample_frames", False, True),
        ):
            with self.subTest(name=name):
                with self.assertRaises(AssertionError):
                    self.assert_settings(
                        SimpleNamespace(**{name: actual}), {name: expected}
                    )


class FactoryAstTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        factory = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "HYV4VLProcessor"
        )
        methods = [
            n
            for n in factory.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("__init__", "from_pretrained")
        ]

        class Mixin:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class Component:
            def __init__(self, **kwargs):
                self.patch_size = 16
                self.merge_size = 2
                self.spatial_patch_size = self.temporal_patch_size = 1
                self.__dict__.update(kwargs)

        self.image_class = type("Image", (Component,), {})
        self.video_class = type("Video", (Component,), {})
        self.image_class.get_image_processor_dict = Mock(
            return_value=(deepcopy(IMAGE), {})
        )
        self.video_class.get_video_processor_dict = Mock(
            return_value=(deepcopy(VIDEO), {})
        )
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(**VISION),
            **{f"{name}_id": i for i, name in enumerate(TOKENS, 1)},
        )
        self.tokenizer = SimpleNamespace(
            convert_ids_to_tokens=lambda i: list(TOKENS.values())[i - 1],
            convert_tokens_to_ids=lambda token: list(TOKENS.values()).index(token) + 1,
            chat_template="local template",
        )
        # The factory only consults these declared option names. These are
        # test doubles, not an emulation of the HF loader or dispatcher.
        annotations = {key: object for key in (*VIDEO, *VISION, "merge_size")}
        options = SimpleNamespace(__annotations__=annotations)
        self.config_loader = Mock(return_value=self.config)
        self.tokenizer_loader = Mock(return_value=self.tokenizer)
        namespace = {
            "ProcessorMixin": Mixin,
            "HYV4VLImageProcessor": self.image_class,
            "HYV4VLVideoProcessor": self.video_class,
            "HYV4ImageProcessorKwargs": options,
            "HYV4VideoProcessorKwargs": options,
            "HYV4VLConfig": SimpleNamespace(from_pretrained=self.config_loader),
            "AutoTokenizer": SimpleNamespace(from_pretrained=self.tokenizer_loader),
        }
        isolated = ast.ClassDef(
            name="HYV4VLProcessor",
            bases=[ast.Name(id="ProcessorMixin", ctx=ast.Load())],
            keywords=[],
            body=methods,
            decorator_list=[],
        )
        module = ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[]))
        exec(compile(module, str(SOURCE), "exec"), namespace)
        self.factory = namespace["HYV4VLProcessor"].from_pretrained

    def load(self, **kwargs):
        return self.factory(
            "local-checkpoint", config=self.config, tokenizer=self.tokenizer, **kwargs
        )

    def test_independent_loader_and_common_hub_arguments(self):
        options = dict(
            revision="test-revision",
            cache_dir="local-cache",
            local_files_only=True,
            token="test-token",
        )
        result = self.factory("local-checkpoint", **options)
        self.image_class.get_image_processor_dict.assert_called_once_with(
            "local-checkpoint", **options
        )
        self.video_class.get_video_processor_dict.assert_called_once_with(
            "local-checkpoint", **options
        )
        self.config_loader.assert_called_once_with("local-checkpoint", **options)
        self.tokenizer_loader.assert_called_once_with(
            "local-checkpoint", trust_remote_code=False, **options
        )
        self.assertEqual(result.image_processor.image_mean, IMAGE["image_mean"])
        self.assertEqual(result.video_processor.image_mean, VIDEO["image_mean"])
        self.assertEqual(result.video_processor.size, VIDEO["size"])
        self.assertEqual(result.video_processor.fps, 3)
        self.assertEqual(result.chat_template, "local template")

    def test_explicit_video_options_win_without_mutating_inputs(self):
        overrides = {
            "fps": 7,
            "max_frames": 40,
            "image_mean": [0.25] * 3,
            "timestamp_token_reserve": 0,
            "do_sample_frames": False,
        }
        before = deepcopy(overrides)
        result = self.load(fps=5, max_frames=60, videos_kwargs=overrides)
        for key, value in overrides.items():
            self.assertEqual(getattr(result.video_processor, key), value)
        self.assertEqual(result.image_processor.image_mean, IMAGE["image_mean"])
        self.assertEqual(overrides, before)
        self.assertEqual(
            self.video_class.get_video_processor_dict.return_value[0], VIDEO
        )

    def test_shared_loader_result_and_empty_video_config_are_not_merged(self):
        for video_dict in (deepcopy(IMAGE), {}):
            with self.subTest(video_dict=video_dict):
                self.video_class.get_video_processor_dict.return_value = (
                    video_dict,
                    {},
                )
                result = self.load()
                self.assertEqual(
                    hasattr(result.video_processor, "image_mean"),
                    "image_mean" in video_dict,
                )
                if video_dict:
                    self.assertEqual(
                        result.video_processor.image_mean, IMAGE["image_mean"]
                    )

    def test_loader_error_is_not_silently_replaced_by_image_config(self):
        self.video_class.get_video_processor_dict.side_effect = OSError(
            "bad video config"
        )
        with self.assertRaisesRegex(OSError, "bad video config"):
            self.load()

    def test_all_patch_config_checks_remain_active(self):
        for component, option in (
            ("images_kwargs", "patch_size"),
            ("videos_kwargs", "patch_size"),
            ("videos_kwargs", "merge_size"),
            ("videos_kwargs", "spatial_patch_size"),
            ("videos_kwargs", "temporal_patch_size"),
        ):
            with self.subTest(component=component, option=option):
                with self.assertRaisesRegex(ValueError, "disagrees with vision_config"):
                    self.load(**{component: {option: 99}})


@unittest.skipUnless(
    importlib.util.find_spec("transformers") is not None,
    "real HF get_dict tests require Transformers (Torch is not required)",
)
class RealHFLoaderTests(ProcessorSettingsAssertions, unittest.TestCase):
    """Actual HF config resolution and kwargs standardization; no tensors or Torch."""

    @classmethod
    def setUpClass(cls):
        from transformers.image_processing_base import ImageProcessingMixin
        from transformers.video_processing_utils import BaseVideoProcessor

        cls.image_loader = ImageProcessingMixin.get_image_processor_dict
        cls.video_loader = BaseVideoProcessor.get_video_processor_dict

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value), encoding="utf-8")

    def check(self, expected_image, expected_video):
        for loader, expected in (
            (self.image_loader, expected_image),
            (self.video_loader, expected_video),
        ):
            actual, unused = loader(
                self.directory, local_files_only=True, probe_option="returned-unused"
            )
            self.assertEqual(actual, expected)
            self.assertEqual(unused, {"probe_option": "returned-unused"})

    def test_real_hf_standardized_mean_std_match_config_sequences(self):
        from transformers.image_processing_utils import BaseImageProcessor

        processor = BaseImageProcessor()
        for settings in (
            IMAGE,
            VIDEO,
            {"image_mean": [0.25] * 3, "image_std": [0.35] * 3},
        ):
            for sequence_type in (list, tuple):
                expected = {
                    name: sequence_type(settings[name])
                    for name in ("image_mean", "image_std")
                }
                with self.subTest(settings=settings, sequence_type=sequence_type):
                    actual = processor._standardize_kwargs(**expected)
                    for name in expected:
                        self.assertIsInstance(actual[name], tuple)
                    self.assert_settings(SimpleNamespace(**actual), expected)

    def test_separate_video_file(self):
        self.write("preprocessor_config.json", IMAGE)
        self.write("video_preprocessor_config.json", VIDEO)
        self.check(IMAGE, VIDEO)

    def test_nested_configs(self):
        self.write(
            "processor_config.json",
            {"image_processor": IMAGE, "video_processor": VIDEO},
        )
        self.check(IMAGE, VIDEO)

    def test_nested_takes_precedence_over_separate_files(self):
        self.write("preprocessor_config.json", {"image_mean": [0.1] * 3})
        self.write("video_preprocessor_config.json", {"fps": 99})
        self.test_nested_configs()

    def test_shared_image_file_fallback(self):
        self.write("preprocessor_config.json", IMAGE)
        self.check(IMAGE, IMAGE)

    def test_legacy_processor_metadata_keeps_shared_fallback(self):
        self.write("processor_config.json", {"processor_class": "HYV4VLProcessor"})
        self.test_shared_image_file_fallback()

    def test_nested_image_with_separate_video(self):
        self.write("processor_config.json", {"image_processor": IMAGE})
        self.write("video_preprocessor_config.json", VIDEO)
        self.check(IMAGE, VIDEO)

    def test_empty_video_dict_does_not_fall_back_to_image(self):
        self.write("preprocessor_config.json", IMAGE)
        self.write("video_preprocessor_config.json", {})
        self.check(IMAGE, {})
        self.write(
            "processor_config.json", {"image_processor": IMAGE, "video_processor": {}}
        )
        self.check(IMAGE, {})

    def test_malformed_video_file_is_not_ignored(self):
        self.write("preprocessor_config.json", IMAGE)
        (self.directory / "video_preprocessor_config.json").write_text(
            "{bad", encoding="utf-8"
        )
        with self.assertRaises((OSError, ValueError)):
            self.video_loader(self.directory, local_files_only=True)


HAS_REAL_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "torchvision", "transformers", "tokenizers", "vllm")
)


@unittest.skipUnless(
    HAS_REAL_DEPS,
    "real local-files HF/plugin tests require Torch, torchvision, Transformers, tokenizers and vLLM",
)
class RealHFLocalFilesTests(ProcessorSettingsAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        from vllm_hcu.models.hy_v4 import processor

        cls.processor = processor
        vocab = {"[UNK]": 0, **{token: i for i, token in enumerate(TOKENS.values(), 1)}}
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]")),
            unk_token="[UNK]",
            additional_special_tokens=list(TOKENS.values()),
        )
        cls.config = processor.HYV4VLConfig(
            vision_config=VISION,
            **{f"{name}_id": vocab[token] for name, token in TOKENS.items()},
        )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value), encoding="utf-8")

    def load(self, **kwargs):
        return self.processor.HYV4VLProcessor.from_pretrained(
            self.directory,
            config=self.config,
            tokenizer=self.tokenizer,
            local_files_only=True,
            **kwargs,
        )

    def test_separate_video_file(self):
        self.write("preprocessor_config.json", IMAGE)
        self.write("video_preprocessor_config.json", VIDEO)
        result = self.load()
        self.assert_settings(result.image_processor, IMAGE)
        self.assert_settings(result.video_processor, VIDEO)

    def test_nested_configs(self):
        self.write(
            "processor_config.json",
            {"image_processor": IMAGE, "video_processor": VIDEO},
        )
        result = self.load()
        self.assert_settings(result.image_processor, IMAGE)
        self.assert_settings(result.video_processor, VIDEO)

    def test_nested_settings_take_hf_precedence_over_separate_files(self):
        self.write("preprocessor_config.json", {**IMAGE, "image_mean": [0.1] * 3})
        self.write("video_preprocessor_config.json", {**VIDEO, "fps": 12})
        self.test_nested_configs()

    def test_legacy_shared_file_fallback(self):
        self.write("preprocessor_config.json", IMAGE)
        result = self.load()
        self.assert_settings(result.image_processor, IMAGE)
        self.assert_settings(result.video_processor, IMAGE)

    def test_legacy_flat_processor_metadata_does_not_mask_shared_file(self):
        self.write("processor_config.json", {"processor_class": "HYV4VLProcessor"})
        self.test_legacy_shared_file_fallback()

    def test_nested_image_with_separate_video_file(self):
        self.write("processor_config.json", {"image_processor": IMAGE})
        self.write("video_preprocessor_config.json", VIDEO)
        result = self.load()
        self.assert_settings(result.image_processor, IMAGE)
        self.assert_settings(result.video_processor, VIDEO)

    def test_empty_video_dict_uses_video_defaults_not_image_settings(self):
        self.write("preprocessor_config.json", IMAGE)
        for filename, contents in (
            ("video_preprocessor_config.json", {}),
            (
                "processor_config.json",
                {"image_processor": IMAGE, "video_processor": {}},
            ),
        ):
            with self.subTest(filename=filename):
                self.write(filename, contents)
                result = self.load()
                self.assertEqual(result.video_processor.size["longest_edge"], 33554432)
                self.assert_settings(
                    result.video_processor,
                    {"image_mean": [0.5] * 3, "image_std": [0.5] * 3, "fps": 2},
                )

    def test_explicit_video_overrides_win_over_files_and_flat_options(self):
        self.write(
            "processor_config.json",
            {"image_processor": IMAGE, "video_processor": VIDEO},
        )
        overrides = {
            "fps": 7,
            "max_frames": 40,
            "image_mean": [0.25] * 3,
            "timestamp_token_reserve": 0,
            "do_sample_frames": False,
        }
        result = self.load(fps=5, max_frames=60, videos_kwargs=overrides)
        self.assert_settings(result.video_processor, overrides)
        self.assert_settings(result.image_processor, IMAGE)

    def test_flat_options_still_override_video_file(self):
        self.write("preprocessor_config.json", IMAGE)
        self.write("video_preprocessor_config.json", VIDEO)
        result = self.load(fps=5, max_frames=60)
        self.assertEqual(result.video_processor.fps, 5)
        self.assertEqual(result.video_processor.max_frames, 60)

    def test_malformed_video_file_does_not_fall_back_to_image(self):
        self.write("preprocessor_config.json", IMAGE)
        (self.directory / "video_preprocessor_config.json").write_text(
            "{bad", encoding="utf-8"
        )
        with self.assertRaises((OSError, ValueError)):
            self.load()

    def test_video_patch_mismatch_rejected_and_explicit_override_can_repair(self):
        self.write("preprocessor_config.json", IMAGE)
        for name in (
            "patch_size",
            "merge_size",
            "spatial_patch_size",
            "temporal_patch_size",
        ):
            with self.subTest(name=name):
                self.write("video_preprocessor_config.json", {**VIDEO, name: 99})
                with self.assertRaisesRegex(ValueError, "disagrees with vision_config"):
                    self.load()
                expected = VISION[
                    "spatial_merge_size" if name == "merge_size" else name
                ]
                result = self.load(videos_kwargs={name: expected})
                self.assertEqual(getattr(result.video_processor, name), expected)

    def test_real_video_loader_receives_common_load_options(self):
        self.write("preprocessor_config.json", IMAGE)
        self.write("video_preprocessor_config.json", VIDEO)
        component = self.processor.HYV4VLVideoProcessor
        with patch.object(
            component,
            "get_video_processor_dict",
            wraps=component.get_video_processor_dict,
        ) as loader:
            self.load(
                revision="local-revision", cache_dir=str(self.directory / "cache")
            )
        loader.assert_called_once_with(
            self.directory,
            revision="local-revision",
            cache_dir=str(self.directory / "cache"),
            local_files_only=True,
        )


if __name__ == "__main__":
    unittest.main()
