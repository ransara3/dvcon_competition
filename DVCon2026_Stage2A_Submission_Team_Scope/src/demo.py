"""
demo.py
-------
Interactive / notebook-friendly demo.
Loads one image, runs all 14 queries, and displays the annotated result.

Run:
    python demo.py --image dataset/images/sample.jpg
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).parent))
from detector import ObjectDetector, DEFAULT_THRESHOLD
from visualize import draw_detections

QUERIES_FILE = Path(__file__).parent.parent / "data" / "queries.json"


def load_queries(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data[0], str):
        return data
    return [d["query"] for d in data]


def show_results(image: Image.Image, detections: list[dict], queries: list[str]):
    annotated = draw_detections(image, detections, queries)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(annotated)
    axes[1].set_title(f"Detections ({len(detections)} found)", fontsize=13)
    axes[1].axis("off")

    # Print legend
    legend_text = "\n".join(
        f"{i+1:>2}. [{d['score']:.3f}]  {d['query']}"
        for i, d in enumerate(detections)
    )
    fig.text(0.01, 0.01, legend_text or "No detections above threshold",
             fontsize=8, family="monospace", verticalalignment="bottom")

    plt.tight_layout()
    plt.savefig("results/demo_output.jpg", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/demo_output.jpg")


def main():
    parser = argparse.ArgumentParser(description="DVCon 2026 Interactive Demo")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--queries-file", default=str(QUERIES_FILE))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    queries = load_queries(Path(args.queries_file))
    print(f"Running {len(queries)} queries on: {args.image}")

    detector = ObjectDetector(threshold=args.threshold, device=args.device)
    image = Image.open(args.image).convert("RGB")
    detections = detector.detect(image, queries)

    print(f"\nFound {len(detections)} detection(s):")
    for d in detections:
        print(f"  [{d['score']:.3f}]  {d['query']}   box={d['box']}")

    Path("results").mkdir(exist_ok=True)
    show_results(image, detections, queries)


if __name__ == "__main__":
    main()
