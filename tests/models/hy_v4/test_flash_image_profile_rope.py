# SPDX-License-Identifier: Apache-2.0
"""Image profile/cache arithmetic; native public preprocessing lives alongside it.

The AST lane has no Torch/Core dependency and is not encoder/GPU validation.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4"


def image_functions():
    namespace = {"math": math, "ImageSize": NS}
    nodes = []
    for path, name in (
        (ROOT / "processor.py", "smart_resize"),
        (ROOT / "mm_preprocess.py", "get_image_size_with_most_features"),
    ):
        nodes.append(
            next(
                n
                for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
        )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT), "exec"), namespace)
    return namespace


class ImageProfileRopeTests(unittest.TestCase):
    def test_default_image_only_profile_respects_real_raw_patch_cache(self):
        ns = image_functions()
        for newline in (False, True):
            config = NS(
                use_2d_rope=True,
                max_image_size=2048,
                patch_size=16,
                use_image_newline_token=newline,
            )
            processor = NS(
                patch_size=16,
                merge_size=2,
                size={"longest_edge": 4194304, "shortest_edge": 262144},
            )
            info = NS(
                get_hf_config=lambda: NS(vision_config=config),
                get_hf_processor=lambda: NS(image_processor=processor),
                _merge_and_resolve_mm_processor_kwargs=lambda _: {},
            )
            size = ns["get_image_size_with_most_features"](info)
            self.assertEqual((size.height, size.width), (2048, 2048))
            edge = config.max_image_size // config.patch_size + 1
            self.assertLessEqual(max(size.height, size.width) // 16, edge)
            # The reported old image-only witness really is unsafe for this cache.
            self.assertGreater(26208 // 16, edge)
            self.assertEqual(
                ns["smart_resize"](size.height, size.width, 32, 262144, 4194304),
                (size.height, size.width),
            )

    def test_disabling_rope_preserves_unrestricted_image_profile(self):
        ns = image_functions()
        config = NS(use_2d_rope=False)
        processor = NS(
            patch_size=16,
            merge_size=2,
            size={"longest_edge": 4194304, "shortest_edge": 262144},
        )
        info = NS(
            get_hf_config=lambda: NS(vision_config=config),
            get_hf_processor=lambda: NS(image_processor=processor),
            _merge_and_resolve_mm_processor_kwargs=lambda _: {},
        )
        size = ns["get_image_size_with_most_features"](info)
        self.assertEqual((size.height, size.width), (26208, 160))

    def test_overrides_and_minimum_upscale_cannot_escape_cache(self):
        ns = image_functions()
        config = NS(use_2d_rope=True, max_image_size=128, patch_size=16)
        processor = NS(
            patch_size=16,
            merge_size=2,
            size={"longest_edge": 4194304, "shortest_edge": 262144},
        )
        options = {"min_pixels": 1024, "max_pixels": 8192}
        info = NS(
            get_hf_config=lambda: NS(vision_config=config),
            get_hf_processor=lambda: NS(image_processor=processor),
            _merge_and_resolve_mm_processor_kwargs=lambda _: {"images_kwargs": options},
        )
        size = ns["get_image_size_with_most_features"](info)
        self.assertLessEqual(max(size.height, size.width) // 16, 9)
        self.assertLessEqual(size.height * size.width, 8192)
        options["min_pixels"] = 65536
        options["max_pixels"] = 65536
        with self.assertRaisesRegex(ValueError, "2D RoPE cache"):
            ns["get_image_size_with_most_features"](info)


if __name__ == "__main__":
    unittest.main()
