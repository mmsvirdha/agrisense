"""
AGRISENSE AI — YOLO Detection Engine (drop-in replacement for classical CV)
============================================================================

This is the "swap-in point" the README always pointed at: same input (a
BGR numpy image), same output (a list of `detection.Detection` objects),
so `main.py`, `tracker.py`, `annotate_image()`, and the entire frontend
need ZERO changes to use it. The only thing that changes is which engine
`main.py` calls — controlled by the AGRISENSE_ENGINE env var.

WHAT THIS DOES DIFFERENTLY FROM detection.py
----------------------------------------------
The classical engine reads peel *color* and guesses a maturity label from
it (red -> "ripe"). That's a heuristic, not a diagnosis, and it doesn't
know what species it's looking at.

A YOLO model trained on a labeled ripeness dataset instead learns to
directly recognize a class like "strawberry-ripe" or "strawberry-unripe"
from shape + texture + color together — closer to how a person actually
judges ripeness, and it can generalize to lighting/background conditions
the color-threshold rules can't.

CLASS NAME MAPPING
-------------------
Whatever class names your trained model uses (they come straight from
your dataset's data.yaml), map them here to a maturity label/score so the
existing `annotate_image()` and `summarize()` functions keep working
unmodified. Add rows to CLASS_MATURITY_MAP for any class your dataset
uses that isn't already listed — unmapped classes fall back to a neutral
"Detected (unmapped class)" reading rather than crashing.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from detection import Detection, _blob_health

# ---------------------------------------------------------------------------
# Map trained-model class names -> the maturity label/score AGRISENSE reports.
# Edit this once you know your dataset's exact class names (check data.yaml
# after downloading, or model.names after loading — printed at startup).
# `color_profile` here is just a display bucket (drives the on-screen box
# color + the frontend's color chips) — it doesn't have to be a real color.
# ---------------------------------------------------------------------------
CLASS_MATURITY_MAP: dict[str, dict] = {
    # --- common Roboflow strawberry-ripeness dataset class names ---
    "strawberry-ripe":    {"color_profile": "red",    "maturity_label": "Ripe (model-detected)",          "maturity_score": 92},
    "ripe":                {"color_profile": "red",    "maturity_label": "Ripe (model-detected)",          "maturity_score": 92},
    "strawberry-unripe":  {"color_profile": "green",  "maturity_label": "Unripe (model-detected)",        "maturity_score": 20},
    "unripe":              {"color_profile": "green",  "maturity_label": "Unripe (model-detected)",        "maturity_score": 20},
    "half-ripe strawberry": {"color_profile": "yellow", "maturity_label": "Developing / half-ripe (model-detected)", "maturity_score": 60},
    "immature strawberry": {"color_profile": "green",  "maturity_label": "Immature (model-detected)",      "maturity_score": 15},
    "rotten":               {"color_profile": "orange", "maturity_label": "Overripe / rotten (model-detected)", "maturity_score": 5, "health_penalty": True},
    "moldy strawberry":    {"color_profile": "orange", "maturity_label": "Moldy — do not harvest",         "maturity_score": 0, "health_penalty": True},
}

DEFAULT_MATURITY = {"color_profile": "yellow", "maturity_label": "Detected (unmapped class)", "maturity_score": 50}

_model_cache: dict[str, "object"] = {}


def load_model(weights_path: str):
    """Loads (and caches) a trained ultralytics YOLO model. Call this once
    at app startup, not per-request — loading weights is the slow part."""
    from ultralytics import YOLO  # imported lazily so classical-only
                                    # deployments don't need ultralytics installed

    if weights_path not in _model_cache:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"YOLO weights not found at '{weights_path}'. Train a model first "
                f"(see yolo/README_YOLO.md) or point AGRISENSE_YOLO_WEIGHTS at an "
                f"existing .pt file."
            )
        _model_cache[weights_path] = YOLO(weights_path)
    return _model_cache[weights_path]


def detect_fruit_blobs_yolo(
    image_bgr: np.ndarray,
    model,
    conf: float = 0.35,
    iou: float = 0.45,
    imgsz: int = 640,
) -> list[Detection]:
    """Same contract as detection.detect_fruit_blobs(): BGR image in,
    list[Detection] out. `model` is a loaded ultralytics YOLO instance
    (see load_model()). `conf`/`iou` are exposed so main.py can pass them
    through as tunable request params, same pattern as min_area/min_circularity
    for the classical engine."""
    results = model.predict(image_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    r = results[0]

    detections: list[Detection] = []
    names = r.names  # {class_id: "class_name"}, comes from the model's data.yaml

    if r.boxes is None or len(r.boxes) == 0:
        return detections

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)

    for i, (box, confidence, cls_id) in enumerate(zip(boxes_xyxy, confs, cls_ids), start=1):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_bgr.shape[1], x2), min(image_bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue

        class_name = names.get(cls_id, str(cls_id))
        meta = CLASS_MATURITY_MAP.get(class_name.lower(), DEFAULT_MATURITY)

        crop = image_bgr[y1:y2, x1:x2]
        dark_ratio, health_flag, health_score = _blob_health(crop)
        if meta.get("health_penalty"):
            # rotten/moldy classes should read as unhealthy regardless of
            # the generic dark-patch heuristic
            health_score = min(health_score, 25)
            health_flag = "model flagged this detection as rotten/moldy — do not harvest"

        detections.append(Detection(
            id=i,
            x1=x1, y1=y1, x2=x2, y2=y2,
            color_profile=meta["color_profile"],
            confidence=float(confidence),
            maturity_label=meta["maturity_label"],
            maturity_score=meta["maturity_score"],
            dark_spot_ratio=dark_ratio,
            health_flag=health_flag,
            health_score=health_score,
        ))

    return detections
