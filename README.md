
# AGRISENSE AI — Crop Intelligence Prototype

A working, testable slice of the AGRISENSE AI concept: feed it a photo or a
short video of fruit, and it detects fruit-like blobs, tracks them across
frames, gives each one a color-based maturity read and a rough surface-health
read, and layers a rule-based crop-intelligence pass on top (per-fruit
priority, health score, crop risk, harvest readiness, scan history).

It runs as both a FastAPI web app and a standalone CLI tool. It's a real
end-to-end pipeline (detection → tracking → annotation → intelligence →
downloadable annotated video), not a mockup.

**Read this before you judge the results.** The default engine is classical
computer vision (HSV color segmentation + contour shape analysis), not a
trained neural network. See "Why classical CV, and what's next" below.
**A trained YOLOv8 model is available as a second engine** — see
`yolo/README_YOLO.md` to train one and switch to it.

---

## What's actually in this build

- `main.py` — FastAPI backend. Endpoints for image/video analysis, field
  listing, scan history, comparison, and video download. Dispatches to
  whichever detection engine is configured via `AGRISENSE_ENGINE`.
- `detection.py` — the classical CV engine. HSV color segmentation,
  contour/circularity filtering, per-profile circularity thresholds,
  color-purity check, dark-spot health flag. Default engine.
- `intelligence.py` — rule-based crop-intelligence layer. Per-fruit priority,
  explainable health score, crop risk engine, harvest readiness. Pure
  post-processing over the detections the vision engine already produces.
- `history_store.py` — JSON-file scan history, one file per field label.
  Enables "what changed since your last scan" without a database.
- `tracker.py` — centroid tracker with a minimum-hits gate. Keeps persistent
  IDs across frames so the same fruit isn't counted multiple times, and
  suppresses single-frame flicker from inflating the unique-fruit count.
- `yolo_detector.py` + `yolo/` — the trained-model engine and everything
  needed to train it (dataset guide, training/evaluation scripts). Not used
  unless you set `AGRISENSE_ENGINE=yolo`. See `yolo/README_YOLO.md`.
- `annotate.py` — standalone CLI. `python annotate.py your_video.mp4`
  renders an annotated copy (with HUD) without running the web server.
  Re-encodes to H.264 via ffmpeg if available.
- `static/` — the browser UI (plain HTML/CSS/JS, no build step).
- `sample_media/` — a synthetic test photo and clip so you can try it
  immediately with no real footage.

---

## How to run it

Requires Python 3.10+.

```bash
cd agrisense
pip install -r requirements.txt
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

- Click **Photo**, choose a still image, click **Analyze**.
- Click **Video**, choose a short clip, click **Analyze**.

Or use the CLI on a video directly:

```bash
python annotate.py your_video.mp4
```

Writes `<name>_annotated.mp4` next to the input, with detection boxes and a
HUD on every frame.

To run with the trained YOLO model instead of the classical engine (after
following `yolo/README_YOLO.md`):

```bash
export AGRISENSE_ENGINE=yolo
export AGRISENSE_YOLO_WEIGHTS=yolo/runs/train/weights/best.pt
uvicorn main:app --reload
```

If the YOLO weights file is missing, the app logs a warning and falls back
to the classical engine automatically — you can't accidentally serve a
broken deployment.

---

## What it does well

- Clear, close-up photos of round, colorful fruit (apples, oranges, mangoes,
  strawberries, tomatoes, etc.) against a background that isn't the same
  color as the fruit.
- Counting fruit in a slow, steady video pan without double-counting.
- Flagging dark or blotchy patches on a fruit's visible surface.
- Per-fruit triage: harvest / monitor / developing / attention.
- Field-level intelligence: health score, crop risk, harvest readiness,
  scan history and deltas.

## What it does *not* do (be upfront about this)

- **It doesn't identify species.** It reports a color read ("red", "green"),
  not "this is a strawberry." Two fruits with similar peel color get the
  same label.
- **It struggles when fruit color matches background color.** Green fruit
  against green leaves is genuinely hard for a color-based method. This is
  a real limitation, visible in the output, not a bug to hide.
- **"Maturity" is a peel-color estimate, not ripeness.** Some varieties are
  naturally green when ripe.
- **"Health" only means "has a dark or uneven patch on the visible surface."**
  It is not disease detection and should never be presented as one.
- **The crop-intelligence scores are transparent weighted formulas**, not
  calibrated predictions. See `intelligence.py`'s module docstring for the
  exact weights. The "3–6 days" harvest estimate is a structured prototype
  opinion, not a validated agronomic forecast.
- **Tracking is centroid-based**, not ByteTrack/SORT. It handles slow,
  steady footage well but can fragment tracks under fast camera motion or
  extended occlusion.
- **Tightly bunched fruit** (e.g. a bunch of bananas) may be merged into one
  blob or split unpredictably.

Tested on strawberry planter footage: peak 13 fruit detected in a single
frame, ~68 confirmed unique tracks over a 13-second clip, ripe red fruit
consistently detected at 85–90% heuristic confidence, green fruit against
green leaves genuinely missed. That last one is the honest limitation this
project documents rather than hides.

---

## Using it with your own footage

Just upload it — no setup step. The default color profiles (green → yellow
→ orange → red) match the ripening progression of most common fruit, so
real footage should produce some detections out of the box.

What usually needs adjusting is **size and shape**, not color. That's what
the **Detection sensitivity** panel is for:

- **Minimum fruit size** — lower this if small or distant fruit are being
  missed; raise it if background clutter is getting counted as fruit.
- **Roundness required** — lower this if partially hidden fruit are being
  missed; raise it if non-fruit round-ish shapes are triggering false
  positives.

Both params (`min_area` in px², `min_circularity` 0–1) are accepted by both
API endpoints, so you can bake them into a batch script.

For video specifically, `max_frames` (default 1800, ~60s at 30fps) caps
processing time on long clips.

---

## Crop-intelligence layer

On top of the vision engine, every scan gets a rule-based intelligence pass.
No dataset, no training — pure post-processing over the detections that are
already there.

- **Per-fruit priority** — every detection is tagged `harvest` / `monitor` /
  `developing` / `attention`. Only fruit whose color profile reads as
  orange or red can reach `harvest`; green and yellow can't.
- **Explainable health score** — 0–100 plus the actual reasons behind it
  (positive factors and risk factors as text, not just a number).
- **Crop risk engine** — disease/anomaly, over-ripening, under-ripening,
  and damage risk as percentages, with an overall Low/Moderate/High label.
- **Harvest readiness** — 0–100 score, an estimated day-range bucket, a
  plain-language recommendation, and a breakdown of how many fruit are
  ready / developing / unripe / flagged for attention.
- **Scan history + "what changed since last scan"** — type a field label
  (e.g. "Strawberry Bed 01") before analyzing, and each scan is saved under
  that label. The next scan of the same field returns `compare_to_previous`
  with deltas on fruit count, maturity, health, readiness, and how many are
  newly ready to harvest. If the deltas are all zero (same file uploaded
  twice), the comparison is suppressed rather than showing an empty panel.

**Be honest about what this is.** Every score is a documented weighted
formula over the same maturity/health numbers the vision engine already
produces. None of it is trained or calibrated against real outcomes. That's
the same honesty standard the rest of this README holds the vision engine
to.

History is stored as one JSON file per field under
`/tmp/agrisense_outputs/history/` — fine for a local single-user prototype,
not meant to survive a server restart in production. See
`history_store.py`'s docstring for how to swap in a real database later
without touching `main.py` beyond the import line.

---

## API reference

### `POST /api/analyze/image`

Form fields:

- `file` (image) — required
- `min_area` (int, default 900) — classical engine only
- `min_circularity` (float, default 0.45) — classical engine only
- `field_label` (str, default `"default"`) — groups scans for history
- `save_scan` (bool, default `true`) — set to `false` to analyze without
  writing to history

Returns detection list (each with priority), summary stats, a base64
annotated JPEG, an `intelligence` object, `compare_to_previous` (or `null`),
and the active `engine`.

### `POST /api/analyze/video`

Form fields:

- `file` (video, e.g. MP4) — required
- `max_frames` (int, default 1800) — caps processing time
- `min_area` / `min_circularity` — classical engine only
- `field_label`, `save_scan` — same as image endpoint

Processes every frame, draws boxes on each one, and returns unique
confirmed-track detections, a per-frame timeline, a `tracker` object
reporting both raw and confirmed track counts, and `annotated_video_url`
— a downloadable MP4 with boxes on every frame, ready to post.

### `GET /api/fields`

Returns field labels that have at least one saved scan.

### `GET /api/history?field=...`

Returns the full scan history for a field.

### `GET /api/compare?field=...`

Returns the diff between the last two scans of a field. Returns
`available: false` if there are fewer than two scans, or if the last two
scans produced identical results.

### `GET /api/download/{filename}`

Downloads a previously rendered annotated video.

### `GET /api/health`

Liveness check. Also reports which detection engine is active
(`classical-cv-prototype` or `yolo-trained`) and, for the YOLO engine, the
loaded model's class names.

---

## Why classical CV, and what's next

The full AGRISENSE plan calls for a YOLO model trained on a labeled fruit
dataset. This prototype started with color + shape rules instead — honest
about what it is, and it gave something clickable with no dataset or GPU
time required.

**The Phase 2 upgrade already exists in `yolo/`.** It adds a trained YOLOv8
ripeness model as a second engine, toggled with `AGRISENSE_ENGINE`
(`classical` default, or `yolo`). Both engines return the same `Detection`
shape, so `main.py`'s `run_detection()` is the only place that knows which
one is active — `tracker.py`, `annotate_image()`, `summarize()`, the
intelligence layer, and the entire frontend are unchanged either way. You
can demo both side by side: "here's the honest heuristic, here's the trained
model" is a genuinely good thing to show in an interview.

Suggested next milestones, in order:

1. ~~Collect/label a small strawberry (or chosen crop) image dataset~~
2. ~~Train a YOLOv8 detection model, evaluate with precision/recall/mAP~~
3. ~~Swap it into the detection pipeline~~ — done, see `yolo/README_YOLO.md`
4. Replace the color-based maturity heuristic with a trained **ripeness
   classifier** (crop each detection, classify, return a 0–100 score).
   Right now the YOLO model's classes already encode ripeness directly,
   which is a step in this direction; a dedicated classifier head is the
   next refinement.
5. Add a real **disease classifier** on leaf detections, with the
   "possible disease detected — confidence X%, please inspect" language
   style already used in this project.
6. Upgrade the tracker from centroid to **ByteTrack** or **SORT** for
   accurate counts under motion.
7. Move from a single-file app to the fuller architecture (FastAPI +
   Postgres + React Native mobile + Next.js dashboard) once the core
   detection pipeline is validated.

---

## Stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn
- **Computer vision:** OpenCV (headless), NumPy
- **Frontend:** plain HTML / CSS / JS (no build step)
- **Video pipeline:** OpenCV VideoWriter (MP4V codec), optionally re-encoded
  to H.264 via ffmpeg if installed



## License

MIT — see [LICENSE](LICENSE) for details.