"""
run_all.py
----------
One-click script to run the full Stage 2A evaluation.

Usage:
    python run_all.py

Steps performed:
  1. Validates that dataset/images/ contains images
  2. Runs evaluate.py for all 14 queries × all images
  3. Prints the summary table to stdout
  4. Results are saved to  results/evaluation/
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
DATASET  = BASE / "dataset" / "images"
QUERIES  = BASE / "data"    / "queries.json"
OUT_DIR  = BASE / "results" / "evaluation"
SRC      = BASE / "src"     / "evaluate.py"

def check_dataset():
    imgs = list(DATASET.glob("*.jpg")) + list(DATASET.glob("*.png"))
    if not imgs:
        print(
            f"\n[ERROR] No images found in  {DATASET}\n"
            "Please copy the contest dataset images into  dataset/images/\n"
        )
        sys.exit(1)
    print(f"[OK] Found {len(imgs)} image(s) in dataset/images/")
    return imgs

def main():
    print("=" * 60)
    print("DVCon India 2026 – Stage 2A: Full Evaluation")
    print("=" * 60)

    check_dataset()

    cmd = [
        sys.executable, str(SRC),
        "--dataset",   str(DATASET),
        "--queries",   str(QUERIES),
        "--output-dir", str(OUT_DIR),
        "--device",    "cpu",
        "--threshold", "0.10",
    ]

    print("\nRunning evaluation …  (this may take several minutes on CPU)\n")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print("\n[ERROR] Evaluation failed. Check the log above.")
        sys.exit(result.returncode)

    # Print text summary if it was generated
    summary_path = OUT_DIR / "summary.txt"
    if summary_path.exists():
        print("\n" + summary_path.read_text(encoding="utf-8"))

    print(f"\nAll outputs saved to:  {OUT_DIR}")
    print("Annotated images  :  ", OUT_DIR / "annotated")
    print("JSON report       :  ", OUT_DIR / "evaluation_report.json")

if __name__ == "__main__":
    main()
