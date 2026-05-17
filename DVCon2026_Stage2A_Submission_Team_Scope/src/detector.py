"""
detector.py
-----------
Open-vocabulary object detector using OWL-ViT v2 (Google, via HuggingFace).
Accepts a list of natural-language text queries and returns bounding boxes
with confidence scores for each query on a given image.

Usage (standalone test):
    python detector.py --image path/to/image.jpg --queries "a red car" "a dog"
"""

import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Model config ─────────────────────────────────────────────────────────────
MODEL_NAME = "google/owlv2-base-patch16-ensemble"
DEFAULT_THRESHOLD = 0.05   # lower = more detections (tune per use case)


class ObjectDetector:
    """
    Wraps OWL-ViT v2 for zero-shot, text-prompted object detection.

    Parameters
    ----------
    model_name   : HuggingFace model ID
    threshold    : minimum confidence score to keep a detection
    device       : 'cpu' (default) or 'cuda'
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cpu",
    ):
        self.threshold = threshold
        self.device = device
        logger.info(f"Loading model: {model_name}  (device={device})")
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        logger.info("Model ready.")

    # ------------------------------------------------------------------
    def detect(self, image: Image.Image, queries: list[str]) -> list[dict]:
        """
        Run detection for one image and a list of text queries.

        Parameters
        ----------
        image   : PIL.Image (RGB)
        queries : list of text strings, e.g. ["a cat", "a traffic light"]

        Returns
        -------
        List of dicts, one per detection above threshold:
            {
              "query"  : str   – the query text that matched
              "score"  : float – confidence score
              "box"    : [x_min, y_min, x_max, y_max]  (pixel coords)
              "box_norm": [cx, cy, w, h]                (normalised, 0-1)
            }
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Wrap queries as a single nested list (batch of 1 image)
        text_queries = [queries]

        inputs = self.processor(
            text=text_queries,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to (xyxy pixel) boxes.
        # Owlv2Processor exposes post_process_grounded_object_detection for
        # both text-queried and image-guided detection.
        target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )[0]

        detections = []
        for score, label_idx, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            s = float(score)
            if s < self.threshold:
                continue
            box_px = [round(float(v), 2) for v in box.tolist()]
            W, H = image.size
            box_norm = [
                round((box_px[0] + box_px[2]) / 2 / W, 4),  # cx
                round((box_px[1] + box_px[3]) / 2 / H, 4),  # cy
                round((box_px[2] - box_px[0]) / W, 4),       # w
                round((box_px[3] - box_px[1]) / H, 4),       # h
            ]
            detections.append(
                {
                    "query": queries[int(label_idx)],
                    "score": round(s, 4),
                    "box": box_px,
                    "box_norm": box_norm,
                }
            )

        # Sort by confidence descending
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections


# ── CLI quick-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Quick detector test")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--queries", nargs="+", required=True, help="Text queries")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    det = ObjectDetector(threshold=args.threshold)
    img = Image.open(args.image).convert("RGB")
    results = det.detect(img, args.queries)
    print(json.dumps(results, indent=2))
