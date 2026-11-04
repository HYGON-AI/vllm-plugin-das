# SPDX-License-Identifier: Apache-2.0
"""Reproducible HYV4VL image checks against a running OpenAI API server."""

import argparse
import base64
import json
import re
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


def check_answer(case, answer):
    """Check constrained synthetic fixtures; photos still need human review."""
    answer = answer.lower().strip()
    red = re.search(r"红|red", answer)
    blue = re.search(r"蓝|blue", answer)
    if case in ("red", "red_repeat"):
        return bool(red and not blue)
    if case == "blue":
        return bool(blue and not red)
    if case == "multi":
        return bool(red and blue and red.start() < blue.start())
    if case == "ocr":
        return re.sub(r"\s+", "", answer) == "hy47285"
    if case in ("text", "count"):
        number = "2" if case == "text" else "3"
        chinese = "二" if case == "text" else "三"
        digits = re.findall(r"\d+", answer)
        return digits == [number] or (not digits and chinese in answer)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default="hy4-vl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--photo", type=Path)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    for color in ("red", "blue"):
        im = Image.new("RGB", (384, 256), "white")
        ImageDraw.Draw(im).rectangle((80, 40, 304, 216), fill=color)
        im.save(root / f"{color}.png")
    im = Image.new("RGB", (512, 256), "white")
    draw = ImageDraw.Draw(im)
    for x in (80, 230, 380):
        draw.ellipse((x - 40, 90, x + 40, 170), fill="red")
    im.save(root / "three_circles.png")
    im = Image.new("RGB", (512, 256), "white")
    ImageDraw.Draw(im).text(
        (65, 92), "HY4 7285", font=ImageFont.load_default(size=52), fill="black"
    )
    im.save(root / "ocr.png")
    cases = [
        ("text", [], "1+1等于几？只回答数字。"),
        ("red", [root / "red.png"], "图中矩形是什么颜色？只回答颜色。"),
        ("blue", [root / "blue.png"], "图中矩形是什么颜色？只回答颜色。"),
        ("count", [root / "three_circles.png"], "图中有几个红色圆形？只回答数量。"),
        ("ocr", [root / "ocr.png"], "请逐字读出图中的文字。"),
        (
            "multi",
            [root / "red.png", root / "blue.png"],
            "分别回答第一张和第二张图片中矩形的颜色。",
        ),
        ("red_repeat", [root / "red.png"], "图中矩形是什么颜色？只回答颜色。"),
    ]
    if args.photo:
        cases.append(("photo", [args.photo], "请描述这张图片。"))
    results = []
    for name, paths, prompt in cases:
        content = []
        for path in paths:
            encoded = base64.b64encode(path.read_bytes()).decode()
            mime = "jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "png"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{encoded}"},
                }
            )
        content.append({"type": "text", "text": prompt})
        body = {
            "model": args.model,
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"reasoning_effort": "no_think"},
            "messages": [{"role": "user", "content": content}],
        }
        start = time.monotonic()
        response = requests.post(
            args.base_url.rstrip("/") + "/v1/chat/completions", json=body, timeout=300
        )
        result = {
            "case": name,
            "status": response.status_code,
            "seconds": time.monotonic() - start,
            "response": response.json(),
        }
        message = result["response"].get("choices", [{}])[0].get("message", {})
        result["correct"] = (
            check_answer(name, message.get("content") or "")
            if response.status_code == 200
            else False
        )
        (root / f"{name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)
    (root / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    if any(item["status"] != 200 or item["correct"] is False for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
