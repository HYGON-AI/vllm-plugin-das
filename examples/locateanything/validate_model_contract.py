#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contract checks against checkpoint source, not a second adapter implementation.

Run in the existing vendor environment; this script installs nothing and never
constructs an inference engine or moves tensors to a device. The original image
processor and placeholder replacement method are extracted unchanged from the
checkpoint source to avoid its obsolete ProcessorMixin constructor and unrelated
video dependencies. Actual vLLM preprocessing is exercised after registration.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import types
import unittest


def source_definition(path, name, namespace, class_name=None):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(code)
    scope = dict(namespace)
    exec(compile(code, str(path), "exec"), scope)
    return scope[name]


def make_image(width, height, offset=0):
    yy, xx = np.indices((height, width), dtype=np.int32)
    rgb = np.stack(((xx * 17 + offset) % 256, (yy * 29 + 11) % 256,
                    (xx * 3 + yy * 7 + offset) % 256), axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CpuContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        image_path = ARGS.model_dir / "image_processing_locateanything.py"
        from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
        from transformers.image_utils import make_list_of_images, valid_images
        from torchvision.transforms import functional as TF
        ns = dict(torch=torch, np=np, math=math, Image=Image, TF=TF,
                  BaseImageProcessor=BaseImageProcessor, BatchFeature=BatchFeature,
                  make_list_of_images=make_list_of_images, valid_images=valid_images,
                  MEAN=(0.5, 0.5, 0.5), STD=(0.5, 0.5, 0.5))
        ref_cls = source_definition(image_path, "LocateAnythingImageProcessor", ns)
        image_config = json.loads((ARGS.model_dir / "preprocessor_config.json").read_text())
        cls.reference = ref_cls(**image_config)
        cls.native = native_processing.LocateAnythingImageProcessor.from_pretrained(ARGS.model_dir)
        cls.images = [make_image(55, 83), make_image(641, 479, 91), make_image(840, 1260, 37)]
        cls.replace_media = staticmethod(source_definition(
            ARGS.model_dir / "processing_locateanything.py", "replace_media_placeholder",
            {"torch": torch, "np": np, "re": re}, class_name="LocateAnythingProcessor"))
        cls.original_merge = staticmethod(source_definition(
            ARGS.model_dir / "modeling_vit.py", "patch_merger", {"torch": torch}))
        cls.vllm_merge = staticmethod(source_definition(
            ARGS.vllm_source / "model_executor/models/moonvit.py", "patch_merger", {"torch": torch}))
        cls.tokenizer = AutoTokenizer.from_pretrained(str(ARGS.model_dir), trust_remote_code=True,
                                                      local_files_only=True)
        cls.chat_template = json.loads((ARGS.model_dir / "chat_template.json").read_text())["chat_template"]

    def test_pixels_exact_after_both_resize_stages(self):
        # Includes tiny inputs, rounding boundaries, an asymmetric image, and
        # a >25,600-patch input that needs both original PIL resize operations.
        for width, height in [(1, 1), (28, 28), (55, 83), (641, 479), (840, 1260), (3001, 2111)]:
            with self.subTest(width=width, height=height):
                image = make_image(width, height)
                ref = self.reference(images=[image], return_tensors="pt")
                got = self.native(images=[image], return_tensors="pt")
                torch.testing.assert_close(got["pixel_values"], ref["pixel_values"], rtol=0, atol=0)
                torch.testing.assert_close(got["image_grid_hws"], ref["image_grid_hws"], rtol=0, atol=0)
                self.assertEqual(native_processing.rescale_size(width, height), self.reference.rescale(image).size)

    def test_batch_patch_layout_matches_independent_images(self):
        packed = self.native(self.images)
        ref = self.reference(self.images, return_tensors="pt")
        torch.testing.assert_close(packed["pixel_values"], ref["pixel_values"], rtol=0, atol=0)
        torch.testing.assert_close(packed["image_grid_hws"], ref["image_grid_hws"], rtol=0, atol=0)
        start = 0
        for image, grid in zip(self.images, packed["image_grid_hws"]):
            length = int(grid.prod())
            single = self.native([image])
            torch.testing.assert_close(packed["pixel_values"][start:start + length],
                                       single["pixel_values"], rtol=0, atol=0)
            start += length
        self.assertEqual(start, packed["pixel_values"].shape[0])

    def test_patch_merge_preserves_spatial_and_image_order(self):
        grids = torch.tensor([[2, 4], [4, 2], [4, 6]], dtype=torch.long)
        x = torch.arange(int(grids.prod(-1).sum()) * 3, dtype=torch.float32).reshape(-1, 3)
        original = self.original_merge(x, grids)
        vllm = self.vllm_merge(x, grids)
        for ref, got in zip(original, vllm):
            torch.testing.assert_close(got.flatten(1), ref, rtol=0, atol=0)
        # Independent spatial fixture: first 2x2 tile of a 2x4 raster is 0,1,4,5.
        expected = x[torch.tensor([0, 1, 4, 5])].reshape(-1)
        torch.testing.assert_close(original[0][0], expected, rtol=0, atol=0)

    def test_worst_case_dummy_is_realizable_and_covers_resize(self):
        gh, gw, source_h, source_w = native_processing.max_image_grid()
        image = Image.new("RGB", (source_w, source_h))
        got = self.reference(images=[image], return_tensors="pt")["image_grid_hws"][0]
        self.assertEqual(got.tolist(), [gh, gw])
        max_tokens = gh * gw // 4
        for w, h in [(840, 1260), (1260, 840), (3001, 2111), (7110, 31), (31, 7110),
                     (906, 7455), (7324, 766)]:
            grid = self.reference([Image.new("RGB", (w, h))], return_tensors="pt")["image_grid_hws"][0]
            self.assertLessEqual(int(grid.prod()) // 4, max_tokens)

    def oracle_inputs(self, images, categories):
        question = "Locate all the instances that matches the following description: " + "</c>".join(categories) + "."
        messages = [{"role": "user", "content": [
            *[{"type": "image", "image": image} for image in images],
            {"type": "text", "text": question},
        ]}]
        prompt = self.tokenizer.apply_chat_template(messages, chat_template=self.chat_template,
                                                   tokenize=False, add_generation_prompt=True)
        holder = types.SimpleNamespace(image_placeholder="image", video_placeholder="video",
                                       image_processor=self.reference, image_start_token="<img>",
                                       image_end_token="</img>", image_token="<IMG_CONTEXT>")
        expanded, pixels, grids, _, _ = self.replace_media(
            holder, prompt, images, [], None, None, images_kwargs={"return_tensors": "pt"})
        return self.tokenizer(expanded)["input_ids"], pixels, torch.as_tensor(grids)

    def test_actual_vllm_placeholder_expansion_matches_official_tokens(self):
        from vllm_hcu.models.locateanything import register
        from vllm.config import ModelConfig
        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm.multimodal.processing import ProcessorInputs, TimingContext
        register()
        config = ModelConfig(model=str(ARGS.model_dir), trust_remote_code=True, dtype="bfloat16",
                             max_model_len=32768, enforce_eager=False, limit_mm_per_prompt={"image": 3},
                             mm_processor_cache_gb=0)
        self.assertFalse(config.enforce_eager)
        processor = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=self.tokenizer)
        # Reuse one processor across different requests, returning to request A
        # after request B; checks request-local placeholder lengths and payloads.
        cases = [(self.images[:1], ["person", "dog"]),
                 (self.images[1:], ["dog", "person", "dog"]),
                 (self.images[:1], ["person", "dog"])]
        for images, categories in cases:
            with self.subTest(images=len(images), categories=categories):
                expected_ids, expected_pixels, expected_grids = self.oracle_inputs(images, categories)
                prompt = native_processing.build_prompt(categories, num_images=len(images))
                inputs = ProcessorInputs(prompt=self.tokenizer(prompt)["input_ids"],
                                         mm_data_items=processor.info.parse_mm_data({"image": images}))
                output = processor.apply(inputs, TimingContext(enabled=False))
                self.assertEqual(output["prompt_token_ids"], expected_ids)
                data = output["mm_kwargs"].get_data(device="cpu", pin_memory=False)
                # vLLM InputProcessingContext intentionally casts floating
                # processor outputs to model dtype; compare the same cast.
                self.assertEqual(data["pixel_values"].dtype, config.dtype)
                torch.testing.assert_close(data["pixel_values"], expected_pixels.to(config.dtype), rtol=0, atol=0)
                torch.testing.assert_close(data["image_grid_hws"], expected_grids, rtol=0, atol=0)
                ranges = output["mm_placeholders"]["image"]
                self.assertEqual(len(ranges), len(images))
                for grid, item in zip(expected_grids, ranges):
                    self.assertEqual(item.length, int(grid.prod()) // 4)

    def test_checkpoint_projector_numerics_and_batch_separation(self):
        from safetensors import safe_open
        from vllm_hcu.models.locateanything.configuration import LocateAnythingConfig
        from vllm_hcu.models.locateanything.model import LocateAnythingProjector
        config = LocateAnythingConfig.from_pretrained(str(ARGS.model_dir))
        native = LocateAnythingProjector(config).float().eval()
        original = torch.nn.Sequential(torch.nn.LayerNorm(4608), torch.nn.Linear(4608, 2048),
                                       torch.nn.GELU(), torch.nn.Linear(2048, 2048)).float().eval()
        index = json.loads((ARGS.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        weights = {}
        for key, shard in index.items():
            if key.startswith("mlp1."):
                with safe_open(str(ARGS.model_dir / shard), framework="pt", device="cpu") as reader:
                    weights[key[len("mlp1."):]] = reader.get_tensor(key).float()
        self.assertEqual(len(weights), 6)
        native.load_state_dict(weights, strict=True)
        original.load_state_dict(weights, strict=True)
        generator = torch.Generator(device="cpu").manual_seed(77)
        features = torch.randn(9, 4, 1152, generator=generator)
        with torch.no_grad():
            expected = original(features.flatten(1))
            got = native(features)
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
            pieces = torch.cat([native(features[:2]), native(features[2:5]), native(features[5:])])
            torch.testing.assert_close(pieces, got, rtol=1e-5, atol=1e-5)

    def test_tied_embeddings_cannot_discard_a_distinct_checkpoint_head(self):
        from vllm_hcu.models.locateanything.configuration import LocateAnythingConfig
        config = LocateAnythingConfig.from_pretrained(str(ARGS.model_dir))
        index = json.loads((ARGS.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        names = ("language_model.model.embed_tokens.weight", "language_model.lm_head.weight")
        if not config.text_config.tie_word_embeddings:
            self.assertTrue(all(name in index for name in names), "Untied model needs both source tensors")
            return
        streams = []
        try:
            entries = []
            for name in names:
                stream = (ARGS.model_dir / index[name]).open("rb")
                streams.append(stream)
                header_size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_size))
                entry = header[name]
                entries.append(entry)
                stream.seek(8 + header_size + entry["data_offsets"][0])
            self.assertEqual(entries[0]["shape"], entries[1]["shape"])
            self.assertEqual(entries[0]["dtype"], entries[1]["dtype"])
            remaining = entries[0]["data_offsets"][1] - entries[0]["data_offsets"][0]
            chunk_index = 0
            while remaining:
                size = min(4 * 1024 * 1024, remaining)
                left, right = (stream.read(size) for stream in streams)
                self.assertEqual(len(left), size)
                self.assertEqual(len(right), size)
                self.assertTrue(left == right, "Distinct source head/embedding bytes: tying loses a tensor "
                                f"(first mismatch within 4 MiB chunk {chunk_index})")
                remaining -= size
                chunk_index += 1
        finally:
            for stream in streams:
                stream.close()

    def test_hf_constructor_backend_metadata_preserves_checkpoint_contract(self):
        from vllm_hcu.models.locateanything.configuration import (
            LocateAnythingConfig, normalize_hf_attention_metadata,
        )
        config = LocateAnythingConfig.from_pretrained(str(ARGS.model_dir))
        raw = json.loads((ARGS.model_dir / "config.json").read_text())
        for component in (config, config.vision_config, config.text_config):
            self.assertNotEqual(getattr(component, "_attn_implementation", None), "magi")
            self.assertNotEqual(getattr(component, "_attn_implementation", None), "eager")
        for section, keys in {
            "vision_config": ("hidden_size", "num_attention_heads", "num_hidden_layers",
                              "patch_size", "merge_kernel_size", "intermediate_size"),
            "text_config": ("hidden_size", "num_attention_heads", "num_key_value_heads",
                            "num_hidden_layers", "vocab_size", "rope_theta", "block_size",
                            "causal_attn", "use_cache", "tie_word_embeddings"),
        }.items():
            for key in keys:
                component = getattr(config, section)
                if key == "rope_theta" and not hasattr(component, key):
                    # Transformers 5 normalizes legacy rope_theta into the
                    # rope_parameters mapping. Verify its value, not merely
                    # the presence of an old attribute which no longer exists.
                    parameters = getattr(component, "rope_parameters", None)
                    self.assertIsInstance(parameters, dict, "Normalized RoPE parameters are required")
                    self.assertIn("rope_theta", parameters)
                    value = parameters["rope_theta"]
                else:
                    value = getattr(component, key)
                self.assertEqual(value, raw[section][key], (section, key))
        # A top-level repair must not silently rewrite an already valid explicit
        # sub-config choice through Transformers' composite config setter.
        config._attn_implementation = "magi"
        config.vision_config._attn_implementation = "flash_attention_2"
        config.text_config._attn_implementation = "sdpa"
        normalize_hf_attention_metadata(config)
        self.assertNotEqual(config._attn_implementation, "magi")
        self.assertEqual(config.vision_config._attn_implementation, "flash_attention_2")
        self.assertEqual(config.text_config._attn_implementation, "sdpa")

    def test_encoder_graph_metadata_bound_and_request_selection(self):
        from vllm_hcu.models.locateanything.configuration import LocateAnythingConfig
        from vllm_hcu.models.locateanything.model import LocateAnythingForConditionalGeneration

        class MetadataOnlyTower(torch.nn.Module):
            # No encoder model or attention is executed. Exercise the actual
            # adapter/base graph input builder and its public shape contracts.
            def __init__(self):
                super().__init__()
                self.encoder = types.SimpleNamespace(rope_2d=types.SimpleNamespace(max_height=512, max_width=512))

            def prepare_encoder_metadata(self, grids, max_batch_size=None,
                                         max_seqlen_override=None, device=None):
                lengths = torch.tensor([h * w for h, w in grids], dtype=torch.int32)
                cumulative = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0))).to(torch.int32)
                return {"cu_seqlens": cumulative,
                        "max_seqlen": torch.tensor(max_seqlen_override or int(lengths.max()), dtype=torch.int32)}

        model = LocateAnythingForConditionalGeneration.__new__(LocateAnythingForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.config = LocateAnythingConfig.from_pretrained(str(ARGS.model_dir))
        model.vision_tower = MetadataOnlyTower()
        for token_budget, batch_size in [(64, 1), (65, 2), (128, 3)]:
            buffers = model.prepare_encoder_cudagraph_capture_inputs(
                token_budget, batch_size, 1, torch.device("cpu"), torch.bfloat16).values
            actual_max = int(torch.diff(buffers["cu_seqlens"]).max())
            self.assertGreaterEqual(int(buffers["max_seqlen"]), actual_max)
            self.assertEqual(buffers["max_seqlen"].device.type, "cpu")
        grids = torch.tensor([[2, 2], [2, 4], [4, 4]], dtype=torch.long)
        pixels = torch.cat([torch.full((int(g.prod()), 3, 14, 14), float(i)) for i, g in enumerate(grids)])
        chosen = model.select_encoder_cudagraph_items({"pixel_values": pixels, "image_grid_hws": grids}, [2, 0])
        torch.testing.assert_close(chosen["image_grid_hws"], grids[[2, 0]], rtol=0, atol=0)
        torch.testing.assert_close(chosen["pixel_values"][:16], torch.full((16, 3, 14, 14), 2.0), rtol=0, atol=0)
        torch.testing.assert_close(chosen["pixel_values"][16:], torch.zeros(4, 3, 14, 14), rtol=0, atol=0)


def main():
    global ARGS, torch, np, Image, AutoTokenizer, native_processing
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "remote-sources/vllm")
    parser.add_argument("--output", type=Path)
    ARGS = parser.parse_args()
    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoTokenizer
    from vllm_hcu.models.locateanything import processing as native_processing
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CpuContract)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "success": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": [(str(t), e) for t, e in result.failures],
        "errors": [(str(t), e) for t, e in result.errors],
        "skips": [(str(t), why) for t, why in result.skipped],
        "scope": "CPU preprocessing, original source merge, actual vLLM processor, checkpoint MLP; no GPU engine",
        "torch_version": torch.__version__,
        "sources": {name: source_hash(ARGS.model_dir / name) for name in
                    ("image_processing_locateanything.py", "processing_locateanything.py",
                     "modeling_vit.py", "chat_template.json", "preprocessor_config.json")},
    }
    if ARGS.output:
        ARGS.output.parent.mkdir(parents=True, exist_ok=True)
        ARGS.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("success", "tests_run", "scope")}, ensure_ascii=False))
    return 0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
