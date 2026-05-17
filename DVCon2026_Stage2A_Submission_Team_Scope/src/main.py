"""
main.py
-------
Command-line interface for the DVCon India 2026 Stage 2A
Open-Vocabulary Object Detector.

Single-image mode:
    python main.py --image dataset/images/img001.jpg \
                   --queries "a person wearing a helmet" "a red car" \
                   --output results/img001_result.jpg

Batch mode (folder of images, all queries from data/queries.json):
    python main.py --folder dataset/images/ \
                   --queries-file data/queries.json \
                   --output-dir results/
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from PIL import Image

# ── ensure src/ is on the path when run from project root ────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from detector import ObjectDetector, DEFAULT_THRESHOLD
from visualize import save_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_queries(queries_arg: list[str] | None, queries_file: str | None) -> list[str]:
    if queries_file:
        with open(queries_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            # Either ["query1", ...] or [{"id":..,"task_name":..,"query":..}, ...]
            if data and isinstance(data[0], dict):
                return [d["query"] for d in data]
            return data
        raise ValueError("queries JSON must be a list")
    if queries_arg:
        return queries_arg
    raise ValueError("Provide --queries or --queries-file")


def load_query_map(queries_file: str) -> dict[str, str]:
    """Return {query_text: task_name} for display purposes."""
    with open(queries_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data and isinstance(data[0], dict) and "task_name" in data[0]:
        return {d["query"]: d["task_name"] for d in data}
    return {}


def collect_images(folder: str) -> list[Path]:
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def run_single(
    detector: ObjectDetector,
    image_path: Path,
    queries: list[str],
    output_path: Path,
    save_json: bool = True,
    query_map: dict[str, str] | None = None,
) -> list[dict]:
    logger.info(f"Processing: {image_path.name}")
    image = Image.open(image_path).convert("RGB")
    detections = detector.detect(image, queries)

    logger.info(f"  {len(detections)} detection(s) found")
    for d in detections:
        task_label = (query_map or {}).get(d["query"], d["query"])
        logger.info(f"    [{d['score']:.3f}] Task: {task_label}  box={d['box']}")

    # Save annotated image
    save_result(image, detections, queries, output_path)
    logger.info(f"  Saved: {output_path}")

    # Optionally save JSON sidecar
    if save_json:
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"image": str(image_path), "queries": queries, "detections": detections},
                f,
                indent=2,
            )
    return detections


# ── main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DVCon 2026 Stage 2A — Open-Vocabulary Object Detector"
    )
    # Input
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image")
    group.add_argument("--folder", help="Path to a folder of images (batch mode)")

    # Queries
    q_group = p.add_mutually_exclusive_group(required=True)
    q_group.add_argument("--queries", nargs="+", help='Text queries, e.g. "a dog"')
    q_group.add_argument("--queries-file", help="Path to queries JSON file")

    # Output
    p.add_argument("--output", help="Output image path (single-image mode)")
    p.add_argument("--output-dir", default="results", help="Output directory (batch mode)")

    # Model config
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Detection confidence threshold")
    p.add_argument("--model", default="google/owlv2-base-patch16-ensemble",
                   help="HuggingFace model ID")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Inference device")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    queries = load_queries(args.queries, args.queries_file)
    logger.info(f"Queries ({len(queries)}): {queries}")

    detector = ObjectDetector(
        model_name=args.model,
        threshold=args.threshold,
        device=args.device,
    )

    # ── Single image ──
    if args.image:
        out = Path(args.output) if args.output else Path("results") / (
            Path(args.image).stem + "_result.jpg"
        )
        run_single(detector, Path(args.image), queries, out)

    # ── Batch mode ──
    else:
        images = collect_images(args.folder)
        if not images:
            logger.error(f"No images found in {args.folder}")
            sys.exit(1)
        logger.info(f"Found {len(images)} image(s) in {args.folder}")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}
        for img_path in images:
            out_path = out_dir / (img_path.stem + "_result.jpg")
            dets = run_single(detector, img_path, queries, out_path)
            all_results[img_path.name] = dets

        # Save a combined summary
        summary_path = out_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"queries": queries, "results": all_results}, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
