# src/__init__.py
# Makes src/ a proper Python package for cleaner imports.

from .detector   import ObjectDetector, DEFAULT_THRESHOLD
from .visualize  import draw_detections, save_result

__all__ = ["ObjectDetector", "DEFAULT_THRESHOLD", "draw_detections", "save_result"]
