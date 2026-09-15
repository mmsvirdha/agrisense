"""
AGRISENSE AI — Scan History Store
===================================

A deliberately simple JSON-file store, one file per "field" (a free-text
label the user types in, e.g. "Strawberry Bed 01"). This is enough to
power scan history + trend charts + "what changed since last scan"
without standing up a database for a prototype.

Not designed for concurrent writers — fine for a single local FastAPI
process serving one browser tab, which is this project's actual use case.
If this grows into the multi-user Next.js dashboard from the original
roadmap, swap this module for a real Postgres table; nothing in main.py
beyond the import line would need to change if you keep the same function
signatures.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

HISTORY_DIR = os.path.join("/tmp/agrisense_outputs", "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

MAX_SCANS_PER_FIELD = 200


def _safe_field_filename(field: str) -> str:
    field = (field or "default").strip() or "default"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", field).strip("_") or "default"
    return slug[:80] + ".json"


def _path_for_field(field: str) -> str:
    return os.path.join(HISTORY_DIR, _safe_field_filename(field))


def load_history(field: str) -> list[dict]:
    path = _path_for_field(field)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def append_scan(field: str, record: dict) -> dict:
    record = dict(record)
    record.setdefault("timestamp", time.time())
    record.setdefault("field_label", field or "default")

    history = load_history(field)
    history.append(record)
    if len(history) > MAX_SCANS_PER_FIELD:
        history = history[-MAX_SCANS_PER_FIELD:]

    path = _path_for_field(field)
    with open(path, "w") as f:
        json.dump(history, f)

    return record


def list_fields() -> list[str]:
    fields = []
    for fname in os.listdir(HISTORY_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(HISTORY_DIR, fname)
        try:
            with open(path, "r") as f:
                records = json.load(f)
            if records:
                fields.append(records[-1].get("field_label", fname[:-5]))
            else:
                fields.append(fname[:-5])
        except (json.JSONDecodeError, OSError):
            fields.append(fname[:-5])
    return sorted(fields)


def _num(d: dict, *path, default=0):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if isinstance(cur, (int, float)) else default


def compare_scans(prev: dict, curr: dict) -> dict:
    def delta(path):
        p = _num(prev, *path)
        c = _num(curr, *path)
        return {"previous": p, "current": c, "delta": round(c - p, 1)}

    return {
        "previous_timestamp": prev.get("timestamp"),
        "current_timestamp": curr.get("timestamp"),
        "fruit_count": delta(("summary", "total_detected")),
        "avg_maturity_score": delta(("summary", "avg_maturity_score")),
        "avg_health_score": delta(("summary", "avg_health_score")),
        "harvest_readiness_score": delta(("intelligence", "harvest_readiness", "score")),
        "ready_to_harvest": delta(("intelligence", "harvest_readiness", "counts", "ready")),
        "attention_needed": delta(("summary", "attention_needed")),
    }


def compare_last_two(field: str) -> Optional[dict]:
    """Compare the two most recent scans of a field.

    If every delta is zero, this almost certainly means the same file was
    uploaded twice with the same settings — there's nothing meaningful to
    show. In that case return None so the frontend hides the panel rather
    than displaying an all-zeros comparison, which would just be noise.
    """
    history = load_history(field)
    if len(history) < 2:
        return None
    diff = compare_scans(history[-2], history[-1])
    any_change = any(
        v["delta"] != 0
        for v in diff.values()
        if isinstance(v, dict) and "delta" in v
    )
    if not any_change:
        return None
    return diff