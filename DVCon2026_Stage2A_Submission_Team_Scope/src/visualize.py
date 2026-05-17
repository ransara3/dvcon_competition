"""
visualize.py
------------
Draw detection results on images and save annotated outputs.
"""

from __future__ import annotations
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Colour palette (one per query index, cycles if > 20 queries)
_PALETTE = [
    "#FF3333", "#FF8800", "#FFDD00", "#33CC33", "#00BBFF",
    "#AA44FF", "#FF44AA", "#00FFAA", "#FF6688", "#44BBFF",
    "#FF9944", "#55FF55", "#FF55FF", "#FFFF44", "#44FFFF",
    "#CC4444", "#4444CC", "#44CC44", "#CC44CC", "#CCCC44",
]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def draw_detections(
    image: Image.Image,
    detections: list[dict],
    query_list: list[str],
    line_width: int = 3,
    font_size: int = 14,
) -> Image.Image:
    """
    Overlay bounding boxes and labels on *image*.

    Parameters
    ----------
    image      : original PIL image
    detections : list of dicts from ObjectDetector.detect()
    query_list : ordered list of all query strings (for consistent colour assignment)
    """
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        q = det["query"]
        idx = query_list.index(q) if q in query_list else 0
        colour = _PALETTE[idx % len(_PALETTE)]
        r, g, b = _hex_to_rgb(colour)

        x0, y0, x1, y1 = det["box"]
        # Clamp to image bounds so rectangles are always valid
        W, H = img.size
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue  # degenerate box, skip

        # Semi-transparent fill
        draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, 40))
        # Solid border
        draw.rectangle([x0, y0, x1, y1], outline=(r, g, b, 220), width=line_width)

        # Label background
        label = f"{q}  {det['score']:.2f}"
        bbox_text = draw.textbbox((x0, y0), label, font=font)
        text_h = bbox_text[3] - bbox_text[1] + 4
        text_w = bbox_text[2] - bbox_text[0] + 4
        draw.rectangle(
            [x0, max(0, y0 - text_h), x0 + text_w, y0],
            fill=(r, g, b, 200),
        )
        draw.text((x0 + 2, max(0, y0 - text_h + 2)), label, fill="white", font=font)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result


def save_result(
    image: Image.Image,
    detections: list[dict],
    query_list: list[str],
    output_path: str | Path,
) -> None:
    """Draw detections and write the image to *output_path*."""
    annotated = draw_detections(image, detections, query_list)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Normalise extension — always save as JPEG to avoid typo issues
    if output_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        output_path = output_path.with_suffix(".jpg")
    fmt = "PNG" if output_path.suffix.lower() == ".png" else "JPEG"
    annotated.save(str(output_path), format=fmt)
