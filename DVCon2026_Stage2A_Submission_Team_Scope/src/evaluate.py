"""
evaluate.py
-----------
Runs the detector against ALL 14 contest queries on the provided dataset,
computes per-query detection statistics, and generates a results report.

Usage:
    python evaluate.py --dataset dataset/images/ \
                       --queries data/queries.json \
                       --output-dir results/evaluation/
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

from PIL import Image
from tqdm import tqdm

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


def load_queries(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data[0], str):
        return [{"id": i + 1, "task_name": f"task_{i+1}", "query": q} for i, q in enumerate(data)]
    # ensure task_name is present
    for i, q in enumerate(data):
        if "task_name" not in q:
            q["task_name"] = f"task_{q.get('id', i+1)}"
    return data


def collect_images(folder: str) -> list[Path]:
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def evaluate(
    detector: ObjectDetector,
    image_paths: list[Path],
    queries: list[dict],
    output_dir: Path,
) -> dict:
    """
    For each (image, query) pair, run detection and collect statistics.

    Returns a results dict with per-query and per-image breakdowns.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    query_strings = [q["query"] for q in queries]

    per_query_stats: dict[str, dict] = {
        q["query"]: {"id": q["id"], "task_name": q.get("task_name", f"task_{q['id']}"), "total_detections": 0, "images_with_detections": 0, "top_scores": []}
        for q in queries
    }

    all_results = []

    for img_path in tqdm(image_paths, desc="Images", unit="img"):
        try:
            image = Image.open(img_path).convert("RGB")
            detections = detector.detect(image, query_strings)

            # Save annotated image immediately (per-image, so partial results survive a crash)
            out_img = output_dir / "annotated" / (img_path.stem + "_annotated.jpg")
            out_img.parent.mkdir(parents=True, exist_ok=True)
            save_result(image, detections, query_strings, out_img)

            # Save per-image JSON sidecar immediately
            sidecar = output_dir / "annotated" / (img_path.stem + ".json")
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump({"image": img_path.name, "detections": detections}, f, indent=2)

        except Exception as e:
            logger.warning(f"Error on {img_path.name}: {e}")
            detections = []

        image_record = {"image": img_path.name, "detections": []}
        detected_queries: set[str] = set()

        for det in detections:
            q = det["query"]
            per_query_stats[q]["total_detections"] += 1
            per_query_stats[q]["top_scores"].append(det["score"])
            detected_queries.add(q)
            image_record["detections"].append(det)

        for q in detected_queries:
            per_query_stats[q]["images_with_detections"] += 1

        all_results.append(image_record)

    # Compute averages
    for q_str, stats in per_query_stats.items():
        scores = stats.pop("top_scores")
        stats["avg_score"] = round(sum(scores) / len(scores), 4) if scores else 0.0
        stats["max_score"] = round(max(scores), 4) if scores else 0.0

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_images": len(image_paths),
        "num_queries": len(queries),
        "per_query": per_query_stats,
        "per_image": all_results,
    }

    # Save full report JSON
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report saved to {report_path}")

    # Save human-readable summary
    _write_text_summary(report, output_dir / "summary.txt")

    return report


def _write_text_summary(report: dict, path: Path):
    lines = [
        "=" * 60,
        "DVCon India 2026 Stage 2A – Evaluation Summary",
        "=" * 60,
        f"Generated : {report['generated_at']}",
        f"Images    : {report['num_images']}",
        f"Queries   : {report['num_queries']}",
        "",
        f"{'#':<4} {'Task Name':<26} {'Dets':>6} {'Imgs':>6} {'AvgScore':>9} {'MaxScore':>9}",
        "-" * 64,
    ]
    for q_str, stats in sorted(report["per_query"].items(), key=lambda x: x[1]["id"]):
        lines.append(
            f"{stats['id']:<4} {stats.get('task_name', q_str):<26} "
            f"{stats['total_detections']:>6} "
            f"{stats['images_with_detections']:>6} "
            f"{stats['avg_score']:>9.4f} "
            f"{stats['max_score']:>9.4f}"
        )
        # also show the full query text
        lines.append(f"     Query: {q_str}")
        lines.append("")
    # remove the trailing blank if any
    lines += ["", "=" * 60]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Text summary saved to {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DVCon 2026 Stage 2A Evaluator")
    parser.add_argument("--dataset", default="dataset/images", help="Folder of input images")
    parser.add_argument("--queries", default="data/queries.json", help="Queries JSON file")
    parser.add_argument("--output-dir", default="results/evaluation", help="Output directory")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    queries = load_queries(args.queries)
    logger.info(f"Loaded {len(queries)} queries")

    images = collect_images(args.dataset)
    if not images:
        logger.error(f"No images found in {args.dataset}")
        sys.exit(1)
    logger.info(f"Found {len(images)} images")

    detector = ObjectDetector(
        model_name=args.model,
        threshold=args.threshold,
        device=args.device,
    )

    evaluate(detector, images, queries, Path(args.output_dir))


if __name__ == "__main__":
    main()
