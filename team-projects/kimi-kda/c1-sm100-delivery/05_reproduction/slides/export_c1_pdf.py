#!/usr/bin/env python3
"""Render a C1 PPTX to a compact, visually faithful PDF with the archive link."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


DELIVERY_URL = (
    "https://github.com/newmancai/assignment02-tensor-core-pipeline/tree/main/"
    "team-projects/kimi-kda/c1-sm100-delivery"
)
PAGE_SIZE = (960.0, 540.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--author", default="奶龙必胜")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def render_tool() -> Path:
    configured = os.environ.get("C1_PRESENTATIONS_SKILL_DIR")
    if configured:
        root = Path(configured)
    else:
        root = Path.home() / (
            ".codex/plugins/cache/openai-primary-runtime/presentations/"
            "26.909.12148/skills/presentations"
        )
    return root / "container_tools/render_slides.py"


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="c1-pdf-render-") as scratch:
        scratch_path = Path(scratch)
        rendered = scratch_path / "slides"
        subprocess.run(
            [sys.executable, str(render_tool()), str(input_path), "--output_dir", str(rendered)],
            check=True,
            env=os.environ.copy(),
        )

        slide_paths = sorted(
            rendered.glob("slide-*.png"),
            key=lambda item: int(item.stem.rsplit("-", 1)[1]),
        )
        if len(slide_paths) != 15:
            raise RuntimeError(f"expected 15 rendered slides, found {len(slide_paths)}")

        pdf = canvas.Canvas(str(output_path), pagesize=PAGE_SIZE, pageCompression=1)
        pdf.setTitle("FlashKDA SM100 迁移分析")
        pdf.setAuthor(args.author)
        for index, slide_path in enumerate(slide_paths, start=1):
            jpeg_path = scratch_path / f"slide-{index:02d}.jpg"
            with Image.open(slide_path) as image:
                image.convert("RGB").save(
                    jpeg_path,
                    "JPEG",
                    quality=args.jpeg_quality,
                    optimize=True,
                    subsampling=0,
                )
            pdf.drawImage(ImageReader(str(jpeg_path)), 0, 0, *PAGE_SIZE)
            if index == 15:
                pdf.linkURL(DELIVERY_URL, (55.5, 22.5, 520.5, 39.0), relative=0, thickness=0)
            pdf.showPage()
        pdf.save()


if __name__ == "__main__":
    main()
