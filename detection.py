"""
AGRISENSE AI — Prototype Detection Engine
==========================================

Color + shape based fruit detection. See module docstring in the project
README for the honest scope of what this does and does not do.
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass


COLOR_PROFILES = {
    "red":    {"ranges": [((0, 90, 60), (8, 255, 255)), ((170, 90, 60), (179, 255, 255))]},
    "orange": {"ranges": [((9, 90, 60), (20, 255, 255))]},
    "yellow": {"ranges": [((21, 70, 80), (34, 255, 255))]},
    "green":  {"ranges": [((35, 50, 40), (85, 255, 255))]},
}

MATURITY_BY_COLOR = {
    "green":  {"label": "Unripe (green)", "score": 25},
    "yellow": {"label": "Developing / ripening (yellow)", "score": 60},
    "orange": {"label": "Near-ripe to ripe (orange)", "score": 80},
    "red":    {"label": "Ripe (red)", "score": 90},
}

MIN_BLOB_AREA = 900
MAX_BLOB_AREA_RATIO = 0.6
MIN_CIRCULARITY = 0.45
MIN_COLOR_PURITY = 0.55

# Per-profile minimum circularity. Ripe fruit (red/orange/yellow) is
# naturally round. Green leaves are jagged — so green blobs must be
# dramatically rounder to count as fruit.
PER_PROFILE_MIN_CIRC = {
    "red": 0.45,
    "orange": 0.45,
    "yellow": 0.45,
    "green": 0.70,   # leaves rarely reach 0.70 circularity
}


@dataclass
class Detection:
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    color_profile: str
    confidence: float
    maturity_label: str
    maturity_score: int
    dark_spot_ratio: float
    health_flag: str
    health_score: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "box": [self.x1, self.y1, self.x2, self.y2],
            "color_profile": self.color_profile,
            "confidence": round(self.confidence, 2),
            "maturity_label": self.maturity_label,
            "maturity_score": self.maturity_score,
            "dark_spot_ratio": round(self.dark_spot_ratio, 3),
            "health_flag": self.health_flag,
            "health_score": self.health_score,
        }


def _mask_for_profile(hsv: np.ndarray, profile: str) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_PROFILES[profile]["ranges"]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _blob_health(crop_bgr: np.ndarray) -> tuple[float, str, int]:
    if crop_bgr.size == 0:
        return 0.0, "not enough data", 100
    v = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    dark_ratio = float((v < 55).mean())
    score = int(max(0, min(100, round(100 - dark_ratio * 220))))
    if dark_ratio > 0.18:
        flag = "possible blemish / dark patch — inspect visually"
    elif dark_ratio > 0.08:
        flag = "minor surface variation"
    else:
        flag = "no visible blemish"
    return dark_ratio, flag, score


def detect_fruit_blobs(
    image_bgr: np.ndarray,
    min_area: int = MIN_BLOB_AREA,
    min_circularity: float = MIN_CIRCULARITY,
    max_area_ratio: float = MAX_BLOB_AREA_RATIO,
    min_aspect: float = 0.5,
    max_aspect: float = 2.0,
    min_color_purity: float = MIN_COLOR_PURITY,
) -> list[Detection]:
    h_img, w_img = image_bgr.shape[:2]
    frame_area = h_img * w_img

    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    candidates: list[Detection] = []
    next_id = 1

    for profile in ("red", "orange", "yellow", "green"):
        # Effective circularity threshold for this profile:
        # the stricter of the user's setting and the profile minimum.
        effective_min_circ = max(min_circularity, PER_PROFILE_MIN_CIRC[profile])

        mask = _mask_for_profile(hsv, profile)
        mask = _clean_mask(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > frame_area * max_area_ratio:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < effective_min_circ:
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h if h else 0
            if aspect < min_aspect or aspect > max_aspect:
                continue

            # Color purity check: the blob's own pixels must be dominated
            # by the target color, not just touch it at the edges. This
            # rejects leaf clumps with a few sunlit pixels in a fruit-color
            # range.
            blob_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, thickness=-1)
            blob_pixel_count = int(blob_mask.sum() / 255)
            if blob_pixel_count == 0:
                continue
            matching = cv2.bitwise_and(mask, blob_mask)
            matching_count = int(matching.sum() / 255)
            purity = matching_count / blob_pixel_count
            if purity < min_color_purity:
                continue

            crop = image_bgr[y:y + h, x:x + w]
            dark_ratio, health_flag, health_score = _blob_health(crop)
            maturity = MATURITY_BY_COLOR[profile]

            # Confidence weighs both shape roundness and color purity.
            confidence = float(min(0.95, 0.35 + circularity * 0.3 + purity * 0.3))

            candidates.append(Detection(
                id=next_id,
                x1=x, y1=y, x2=x + w, y2=y + h,
                color_profile=profile,
                confidence=confidence,
                maturity_label=maturity["label"],
                maturity_score=maturity["score"],
                dark_spot_ratio=dark_ratio,
                health_flag=health_flag,
                health_score=health_score,
            ))
            next_id += 1

    return _suppress_overlaps(candidates)


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.x1, a.y1, a.x2, a.y2
    bx1, by1, bx2, by2 = b.x1, b.y1, b.x2, b.y2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def _suppress_overlaps(dets: list[Detection], iou_thresh: float = 0.35) -> list[Detection]:
    dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for d in dets:
        if all(_iou(d, k) < iou_thresh for k in kept):
            kept.append(d)
    for i, d in enumerate(kept, start=1):
        d.id = i
    return kept


def annotate_image(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = image_bgr.copy()
    palette = {
        "red": (60, 60, 220), "orange": (30, 140, 245),
        "yellow": (40, 210, 235), "green": (90, 170, 70),
    }
    for d in detections:
        color = palette.get(d.color_profile, (255, 255, 255))
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        label = f"#{d.id} {d.color_profile} {int(d.confidence * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (d.x1, max(0, d.y1 - th - 8)), (d.x1 + tw + 6, d.y1), color, -1)
        cv2.putText(out, label, (d.x1 + 3, d.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def summarize(detections: list[Detection]) -> dict:
    if not detections:
        return {
            "total_detected": 0,
            "by_color": {},
            "avg_maturity_score": 0,
            "avg_health_score": 0,
            "attention_needed": 0,
        }
    by_color: dict[str, int] = {}
    for d in detections:
        by_color[d.color_profile] = by_color.get(d.color_profile, 0) + 1
    avg_maturity = round(sum(d.maturity_score for d in detections) / len(detections))
    avg_health = round(sum(d.health_score for d in detections) / len(detections))
    attention = sum(1 for d in detections if d.health_score < 70)
    return {
        "total_detected": len(detections),
        "by_color": by_color,
        "avg_maturity_score": avg_maturity,
        "avg_health_score": avg_health,
        "attention_needed": attention,
    }