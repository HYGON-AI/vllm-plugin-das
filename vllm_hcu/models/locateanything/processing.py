# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Image preprocessing and prompt helpers matching the original checkpoint.

This deliberately avoids the checkpoint's old ProcessorMixin construction API.
It does not change the image resize/interpolation/normalization operations.
vLLM owns expansion of the one-token image placeholders and multimodal caching.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import BatchFeature

IMAGE_TOKEN = "<IMG_CONTEXT>"


def rescale_size(width, height, patch_size=14, in_token_limit=25600,
                 merge_kernel_size=(2, 2)):
    """Return the exact final pixel size used by the original image processor."""
    width, height = int(width), int(height)
    if width < 1 or height < 1 or patch_size < 1 or in_token_limit < 1:
        raise ValueError("Image dimensions, patch size and token limit must be positive")
    if (width // patch_size) * (height // patch_size) > in_token_limit:
        scale = math.sqrt(in_token_limit / ((width // patch_size) * (height // patch_size)))
        width, height = int(width * scale), int(height * scale)
    kh, kw = merge_kernel_size
    width = math.ceil(width / (kw * patch_size)) * kw * patch_size
    height = math.ceil(height / (kh * patch_size)) * kh * patch_size
    if width // patch_size >= 512 or height // patch_size >= 512:
        raise ValueError("Exceed pos emb: each MoonViT grid dimension must be <512")
    return width, height


def max_image_grid(in_token_limit=25600, merge_kernel_size=(2, 2), patch_size=14):
    """Return (grid_h, grid_w, source_height_pixels, source_width_pixels).

    The dummy image is realizable by the unmodified original processor. Both
    the no-resize branch and the floor-based downscale branch are considered;
    the latter can exceed in_token_limit even BEFORE merge-aligned rounding.
    """
    return _max_image_geometry(int(in_token_limit), tuple(merge_kernel_size), int(patch_size))


@lru_cache(maxsize=16)
def _max_image_geometry(limit, merge, patch):
    kh, kw = merge
    cap_h, cap_w = (511 // kh) * kh, (511 // kw) * kw
    best = (0, 0, 0, 0)
    # No downscale: the smallest raw patch floors yielding grid g are g-k.
    for gh in range(kh, 512, kh):
        for gw in range(kw, 512, kw):
            a, b = gw - kw, gh - kh
            if a * b <= limit and gh * gw > best[0] * best[1]:
                best = (gh, gw, b * patch + 1, a * patch + 1)
    if best[0] * best[1] == cap_h * cap_w:
        return best

    def search(max_a, max_b, already_searched=0):
        nonlocal best
        for a in range(1, max_a + 1):
            first_b = limit // a + 1  # Only the downscale branch.
            if a <= already_searched:
                first_b = max(first_b, already_searched + 1)
            for b in range(first_b, max_b + 1):
                scale = math.sqrt(limit / (a * b))
                # For these raw patch floors, choose the largest source pixels
                # whose resized dimensions still fit the original RoPE range.
                # scale <= 1 means every integer resized dimension is reachable.
                source_w, source_h = (a + 1) * patch - 1, (b + 1) * patch - 1
                if int(source_w * scale) > cap_w * patch:
                    source_w = min(source_w, math.ceil((cap_w * patch + 1) / scale) - 1)
                    while int(source_w * scale) > cap_w * patch:
                        source_w -= 1
                if int(source_h * scale) > cap_h * patch:
                    source_h = min(source_h, math.ceil((cap_h * patch + 1) / scale) - 1)
                    while int(source_h * scale) > cap_h * patch:
                        source_h -= 1
                if source_w < a * patch or source_h < b * patch:
                    continue
                width, height = int(source_w * scale), int(source_h * scale)
                if width < 1 or height < 1:
                    continue
                gw = math.ceil(width / (kw * patch)) * kw
                gh = math.ceil(height / (kh * patch)) * kh
                if gh * gw > best[0] * best[1]:
                    best = (gh, gw, source_h, source_w)

    # A bounded initial scan tightens the proof bound; its extent is not used
    # as an assumption about supported input-image dimensions.
    seed = 1024
    search(seed, seed)
    if best[0] * best[1] == cap_h * cap_w:
        return best

    # Exhaustive finite bound for ANY improving downscaled image:
    # let a,b be original floor(pixel/patch), c=1-1/patch and s=sqrt(L/ab).
    # A resulting grid g needs s*(a+c) >= g-k+1/patch in each dimension.
    # Consequently delta <= c/a+c/b+c*c/(ab), where delta is the smallest
    # product excess over L among all grids larger than the current best.
    # Valid output width also gives a/b < R=(cap_w+1/patch)^2/L, hence
    # delta*a^2 <= c*(1+R)*a+c*c*R. Solve this quadratic (and its height twin).
    # All raw floor dimensions beyond those bounds are proven unable to improve.
    delta = min(
        ((gh - kh + 1 / patch) * (gw - kw + 1 / patch) / limit - 1)
        for gh in range(kh, 512, kh) for gw in range(kw, 512, kw)
        if gh * gw > best[0] * best[1]
    )
    if delta <= 0:
        raise AssertionError("No-resize search must cover grids with nonpositive excess")
    delta = math.nextafter(delta, 0.0)
    c = 1 - 1 / patch

    def bound(cap):
        ratio = (cap + 1 / patch) ** 2 / limit
        linear = c * (1 + ratio)
        return math.ceil((linear + math.sqrt(linear * linear + 4 * delta * c * c * ratio)) / (2 * delta)) + 2

    max_a, max_b = bound(cap_w), bound(cap_h)
    if max_a > seed or max_b > seed:
        search(max_a, max_b, already_searched=seed)
    return best


def image_placeholder(index=1):
    return f"<image {index}><img>{IMAGE_TOKEN}</img>"


def build_prompt(categories, num_images=1, *, legacy_system_newline=False):
    """Canonical detection prompt BEFORE vLLM expands image token slots.

    The default follows the checkpoint's chat_template.json. Its Python
    fallback has an extra system newline; that explicit compatibility option
    is available for reproducing a run which actually used the fallback.
    """
    if isinstance(categories, str):
        raise TypeError("categories must be an ordered sequence of category names")
    question = ("Locate all the instances that matches the following description: "
                + "</c>".join(categories) + ".")
    system = "You are a helpful assistant." + ("\n" if legacy_system_newline else "")
    images = "".join(image_placeholder(i + 1) for i in range(num_images))
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{images}{question}<|im_end|>\n"
            "<|im_start|>assistant\n")


class LocateAnythingImageProcessor:
    model_input_names = ["pixel_values", "image_grid_hws"]

    def __init__(self, patch_size=14, in_token_limit=25600,
                 image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
                 merge_kernel_size=(2, 2), **kwargs):
        self.patch_size = int(patch_size)
        self.in_token_limit = int(in_token_limit)
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.merge_kernel_size = tuple(merge_kernel_size)
        if self.patch_size <= 0 or self.in_token_limit <= 0:
            raise ValueError("patch_size and in_token_limit must be positive")
        if len(self.merge_kernel_size) != 2 or min(self.merge_kernel_size) <= 0:
            raise ValueError("merge_kernel_size must contain two positive integers")

    @classmethod
    def from_pretrained(cls, model_path, **overrides):
        config_path = Path(model_path) / "preprocessor_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Original checkpoint preprocessor config required: {config_path}")
        settings = json.loads(config_path.read_text(encoding="utf-8"))
        names = ("patch_size", "in_token_limit", "image_mean", "image_std", "merge_kernel_size")
        settings.update({k: v for k, v in overrides.items() if k in names})
        return cls(**settings)

    def rescale(self, image, merge_kernel_size=None):
        if not isinstance(image, Image.Image):
            raise TypeError("Use decoded PIL images (vLLM's image modality supplies PIL images)")
        width, height = image.size
        patch = self.patch_size
        count = (width // patch) * (height // patch)
        # Preserve TWO PIL resize operations when both are needed. Combining
        # them changes pixels and may change final detection accuracy.
        if count > self.in_token_limit:
            scale = math.sqrt(self.in_token_limit / count)
            image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.BICUBIC)
        kh, kw = merge_kernel_size or self.merge_kernel_size
        width, height = image.size
        target = (math.ceil(width / (kw * patch)) * kw * patch,
                  math.ceil(height / (kh * patch)) * kh * patch)
        if target != image.size:
            image = image.resize(target, Image.Resampling.BICUBIC)
        if image.width // patch >= 512 or image.height // patch >= 512:
            raise ValueError("Exceed pos emb")
        return image

    def patchify(self, image):
        channels, height, width = image.shape
        patch = self.patch_size
        patches = image.reshape(channels, height // patch, patch, width // patch, patch)
        patches = patches.permute(1, 3, 0, 2, 4).contiguous()
        return patches.view(-1, channels, patch, patch), (height // patch, width // patch)

    def _preprocess(self, image):
        image = self.rescale(image)
        tensor = TF.to_tensor(image.convert("RGB"))
        tensor = TF.normalize(tensor, self.image_mean, self.image_std)
        return self.patchify(tensor)

    def preprocess(self, images, return_tensors="pt", **kwargs):
        if return_tensors not in (None, "pt"):
            raise ValueError("Native vLLM processor returns PyTorch tensors")
        if isinstance(images, Image.Image):
            images = [images]
        pixel_values, grids = [], []
        for image in images:
            patches, grid = self._preprocess(image)
            pixel_values.append(patches)
            grids.append(grid)
        if not pixel_values:
            return BatchFeature({
                "pixel_values": torch.empty((0, 3, self.patch_size, self.patch_size)),
                "image_grid_hws": torch.empty((0, 2), dtype=torch.long),
            })
        return BatchFeature({
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_hws": torch.tensor(grids, dtype=torch.long),
        })

    __call__ = preprocess


class LocateAnythingProcessor:
    """Minimal callable accepted by vLLM's InputProcessingContext."""
    image_token = IMAGE_TOKEN
    model_input_names = ["pixel_values", "image_grid_hws"]

    def __init__(self, tokenizer=None, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor or LocateAnythingImageProcessor()

    def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
        # text here is vLLM-generated dummy text. The actual user prompt is
        # tokenized by its renderer, then expanded by PromptReplacement.
        return self.image_processor(images or [], return_tensors=return_tensors)
