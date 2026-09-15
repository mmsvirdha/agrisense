"""
AGRISENSE AI — Crop Intelligence Layer
========================================

Everything in this module is rule-based post-processing on top of the
Detection objects the vision engine (classical CV or YOLO) already
produces. No dataset, no training, no new model — just turning "7 fruit
detected, avg maturity 50" into something closer to what a person
actually wants to know: is this ready, what's the risk, why.

This is deliberately transparent. Every score below is a documented
weighted formula over numbers already in each Detection
(maturity_score, health_score, color_profile, dark_spot_ratio). Nothing
here is learned or calibrated against ground truth — treat the day
estimates and risk percentages as a structured, explainable *prototype*
opinion, not a validated agronomic prediction. That distinction matters
and should stay visible in the UI, not just in this docstring.
"""

from __future__ import annotations

import statistics
from typing import Optional

from detection import Detection


# ---------------------------------------------------------------------------
# Fruit-level priority
# ---------------------------------------------------------------------------

def classify_fruit_priority(d: Detection) -> dict:
    """Per-fruit triage label. Pure thresholding on maturity_score /
    health_score — no history needed.

    IMPORTANT: thresholds are set so that 'harvest' is only reachable by
    fruit whose color profile reads as ripe-orange or ripe-red. Green and
    yellow fruit can never be labeled 'harvest' regardless of confidence,
    because maturity_score for those colors (25 and 60) sits below the
    harvest threshold (80). This is intentional — a green strawberry is
    not ready to harvest just because the detector is confident it's a
    strawberry.
    """
    if d.health_score < 55:
        return {"priority": "attention", "priority_label": "Needs attention",
                "reason": "Low surface-health score"}
    if d.maturity_score >= 80 and d.health_score >= 60:
        return {"priority": "harvest", "priority_label": "Harvest",
                "reason": "High maturity color, healthy surface"}
    if d.maturity_score >= 45:
        return {"priority": "monitor", "priority_label": "Monitor",
                "reason": "Mid-range maturity — check again soon"}
    return {"priority": "developing", "priority_label": "Developing",
            "reason": "Early-stage color, not close to ready"}


# ---------------------------------------------------------------------------
# Explainable health score
# ---------------------------------------------------------------------------

def explainable_health(detections: list[Detection]) -> dict:
    if not detections:
        return {
            "score": 0, "assessment": "No data",
            "positive_factors": [], "risk_factors": ["No fruit detected in this scan."],
        }

    health_scores = [d.health_score for d in detections]
    maturity_scores = [d.maturity_score for d in detections]
    avg_health = sum(health_scores) / len(health_scores)
    avg_maturity = sum(maturity_scores) / len(maturity_scores)

    spread = statistics.pstdev(maturity_scores) if len(maturity_scores) > 1 else 0
    uniformity = max(0, 100 - spread * 1.5)

    blemished = sum(1 for d in detections if d.health_score < 70)
    blemish_fraction = blemished / len(detections)

    score = round(avg_health * 0.5 + avg_maturity * 0.3 + uniformity * 0.2)
    score = max(0, min(100, score))

    positive: list[str] = []
    risks: list[str] = []

    if avg_health >= 80:
        positive.append("Most fruit shows no visible surface blemish")
    if blemish_fraction <= 0.1 and len(detections) >= 3:
        positive.append("Very few fruit flagged for surface concerns")
    if spread < 20 and len(detections) >= 3:
        positive.append("Maturity is progressing fairly evenly across the crop")
    if avg_maturity >= 55:
        positive.append("Crop is trending toward the ripe end of the color range")

    if blemish_fraction > 0.25:
        risks.append(f"{blemished} of {len(detections)} detections show a dark or uneven patch")
    if spread >= 30 and len(detections) >= 3:
        risks.append("Wide spread between least and most mature fruit — uneven progression")
    if avg_maturity < 35:
        risks.append("Most fruit is still early-stage by color")
    if not positive and not risks:
        risks.append("Not enough detections in this scan for a confident read")

    if score >= 85:
        assessment = "Excellent"
    elif score >= 70:
        assessment = "Good"
    elif score >= 50:
        assessment = "Moderate"
    else:
        assessment = "Needs attention"

    return {
        "score": score,
        "assessment": assessment,
        "positive_factors": positive,
        "risk_factors": risks,
        "components": {
            "avg_health_score": round(avg_health),
            "avg_maturity_score": round(avg_maturity),
            "uniformity_score": round(uniformity),
        },
    }


# ---------------------------------------------------------------------------
# Crop risk engine
# ---------------------------------------------------------------------------

def risk_engine(detections: list[Detection]) -> dict:
    if not detections:
        return {
            "overall_risk": "unknown", "overall_risk_label": "No data",
            "monitoring_priority": "low",
            "disease_risk_pct": 0, "over_ripening_risk_pct": 0,
            "under_ripening_risk_pct": 0, "damage_risk_pct": 0,
        }

    n = len(detections)
    dark_ratios = [d.dark_spot_ratio for d in detections]
    avg_dark_ratio = sum(dark_ratios) / n

    disease_risk = min(100, round(avg_dark_ratio * 300))

    red_and_blemished = sum(
        1 for d in detections if d.color_profile == "red" and d.health_score < 70
    )
    over_ripening_risk = round(100 * red_and_blemished / n)

    immature = sum(1 for d in detections if d.color_profile in ("green", "yellow"))
    under_ripening_risk = round(100 * immature / n)

    blemished = sum(1 for d in detections if d.health_score < 70)
    damage_risk = round(100 * blemished / n)

    overall_score = max(disease_risk, over_ripening_risk * 0.6, damage_risk * 0.8)
    if overall_score >= 60:
        overall_risk, overall_label, monitor = "high", "High", "high"
    elif overall_score >= 30:
        overall_risk, overall_label, monitor = "moderate", "Moderate", "medium"
    else:
        overall_risk, overall_label, monitor = "low", "Low", "low"

    return {
        "overall_risk": overall_risk,
        "overall_risk_label": overall_label,
        "monitoring_priority": monitor,
        "disease_risk_pct": disease_risk,
        "over_ripening_risk_pct": over_ripening_risk,
        "under_ripening_risk_pct": under_ripening_risk,
        "damage_risk_pct": damage_risk,
    }


# ---------------------------------------------------------------------------
# Harvest readiness
# ---------------------------------------------------------------------------

def harvest_readiness(detections: list[Detection]) -> dict:
    if not detections:
        return {
            "score": 0, "estimated_days_label": "Not enough data",
            "recommendation": "No fruit detected in this scan — try adjusting detection sensitivity or a different angle.",
            "counts": {"ready": 0, "developing": 0, "unripe": 0, "attention": 0},
        }

    n = len(detections)
    priorities = [classify_fruit_priority(d)["priority"] for d in detections]
    ready = priorities.count("harvest")
    developing = priorities.count("monitor")
    unripe = priorities.count("developing")
    attention = priorities.count("attention")

    avg_maturity = sum(d.maturity_score for d in detections) / n
    avg_health = sum(d.health_score for d in detections) / n
    ready_fraction = ready / n

    score = round(avg_maturity * 0.55 + avg_health * 0.2 + ready_fraction * 100 * 0.25)
    score = max(0, min(100, score))

    if score >= 85:
        days_label = "0–2 days"
    elif score >= 65:
        days_label = "3–6 days"
    elif score >= 40:
        days_label = "7–12 days"
    else:
        days_label = "13+ days"

    if ready > 0:
        rec = f"{ready} fruit look ready now. Prioritize those first, then re-scan in a few days for the rest."
    elif score >= 50:
        rec = "Crop is progressing but nothing is fully ready yet. Re-scan in a few days."
    else:
        rec = "Crop is still early-stage. Continue monitoring; no action needed yet."

    if attention > 0:
        rec += f" {attention} fruit flagged for a closer look before harvesting."

    return {
        "score": score,
        "estimated_days_label": days_label,
        "recommendation": rec,
        "counts": {"ready": ready, "developing": developing, "unripe": unripe, "attention": attention},
    }


# ---------------------------------------------------------------------------
# Priority merge into serialized detections
# ---------------------------------------------------------------------------

def annotate_detections_with_priority(
    detection_dicts: list[dict],
    detections: list[Detection],
) -> list[dict]:
    """Merge classify_fruit_priority() into the already-serialized detection
    dicts from Detection.to_dict().

    Uses list POSITION (zip), NOT Detection.id, because video tracking can
    produce multiple Detections that share the same .id (the ID from the
    frame they were first seen in). Keying by id causes priorities to be
    applied to the wrong rows in the video endpoint, which was observed in
    testing as ripe red and unripe green fruit both landing in the same
    'Harvest' bucket. Caller must pass detection_dicts and detections in
    the SAME ORDER — main.py does, because both are built by iterating the
    same list.
    """
    for dd, d in zip(detection_dicts, detections):
        dd.update(classify_fruit_priority(d))
    return detection_dicts


def build_intelligence_report(detections: list[Detection]) -> dict:
    """Single entry point main.py calls once per analyze request."""
    return {
        "health": explainable_health(detections),
        "risk": risk_engine(detections),
        "harvest_readiness": harvest_readiness(detections),
    }