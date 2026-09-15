# AGRISENSE AI — Computer Vision Prototype

This is a **working, testable first slice** of the AGRISENSE AI concept: upload
a photo or short video of fruit, and the app detects fruit-like blobs, counts
them, and gives each one a color-based maturity read and a rough surface-health
read. It's a real end-to-end app (backend + browser UI), not a mockup.

**Read this before you judge the results:** the default engine here is
classical computer vision (color segmentation + shape checks), not a
trained neural network — see "Why classical CV, and what's next" below.
**A trained YOLOv8 model is also available as a second engine** —
see `yolo/README_YOLO.md` to train one and switch to it.

## What's actually in this build

- `main.py` — FastAPI backend with `/api/analyze/image` and
  `/api/analyze/video`, dispatching to whichever detection engine is
  configured (`AGRISENSE_ENGINE`, see below)
- `detection.py` — the classical CV engine (HSV color segmentation,
  contour/circularity filtering, maturity color mapping, dark-spot health
  flag) — the default
- `yolo_detector.py` + `yolo/` — the trained-model engine and everything
  needed to train it (dataset guide, training/evaluation scripts). Not
  used unless you set `AGRISENSE_ENGINE=yolo`; see `yolo/README_YOLO.md`
- `tracker.py` — a minimal centroid tracker so a video doesn't count the
  same fruit multiple times as it moves across frames
- `annotate.py` — a standalone CLI: `python annotate.py your_video.mp4`
  renders an annotated copy without running the web server
- `static/` — the browser UI (plain HTML/CSS/JS, no build step)
- `sample_media/` — a synthetic test photo and clip so you can try it
  immediately, even with no real footage yet

## How to run it

Requires Python 3.10+.

```bash
cd agrisense
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

- Click **Photo**, choose `sample_media/sample_fruit.jpg` (or your own image),
  click **Analyze**.
- Click **Video**, choose `sample_media/sample_clip.mp4` (or your own clip),
  click **Analyze**.

Once you have real footage, just upload it the same way — no code changes
needed.

To run with the trained YOLO model instead of the classical engine (after
following `yolo/README_YOLO.md`):

```bash
export AGRISENSE_ENGINE=yolo
export AGRISENSE_YOLO_WEIGHTS=yolo/runs/train/weights/best.pt
uvicorn main:app --reload
```

## Using it with your own real harvest footage

Just upload it — there's no setup step. Open the app, click **Video** (or
**Photo**), choose your file, click **Analyze**. It runs the same
detect → track → summarize pipeline regardless of whether the file is the
included sample or your own footage.

The default color profiles (green → yellow → orange → red) already match
the ripening progression of mango and most common fruit, so real footage
should produce *some* detections out of the box. What usually needs
adjusting for real footage is **size and shape**, not color — a fruit
30px across in a wide shot behaves differently than one filling the frame
in a close-up. That's what the **Detection sensitivity** panel (under the
upload button) is for:

- **Minimum fruit size** — lower this if small or far-away fruit aren't
  being picked up; raise it if background clutter (leaves, shadows,
  soil clumps) is getting counted as fruit
- **Roundness required** — lower this if partially hidden fruit (half
  behind a leaf, cut off at the frame edge) are being missed; raise it if
  non-fruit round-ish shapes are triggering false positives

Adjust the sliders and re-run **Analyze** on the same file — no restart,
no code changes. If you find a good pair of values for your typical
footage, the params are `min_area` (px²) and `min_circularity` (0–1),
both accepted by both API endpoints, so you can also bake them into a
script if you're batch-processing many clips.

For video specifically, `sample_every` (how many frames to skip between
samples) and `max_frames` (cap on frames analyzed) are also adjustable via
the API — not yet exposed in the UI — if you need to trade off speed vs.
thoroughness on long clips.

## What it does well

- Clear, close-up photos of round, colorful fruit (apples, oranges, mangoes,
  limes, tomatoes, etc.) against a background that isn't the same color as
  the fruit
- Counting fruit in a slow, steady video pan without double-counting
- Flagging dark/blotchy patches on a fruit's visible surface

## What it does *not* do (be upfront about this)

- **It doesn't identify species.** It reports a color read ("orange",
  "red"), not "this is a mango." Two different fruits with similar peel
  color will get the same label.
- **It struggles when fruit color is close to the background color** — a
  green fruit against green leaves is genuinely hard for a color-based
  method. This showed up in testing (see below) and is a real limitation,
  not a bug to hide.
- **"Maturity" is a peel-color estimate, not ripeness.** Some fruit
  varieties are naturally green when ripe; this heuristic doesn't know
  that.
- **"Health" only means "has a dark/uneven patch on the visible surface."**
  It is not disease detection and should never be presented as one.
- **Tightly bunched fruit** (e.g. a bunch of bananas touching) may be
  merged into one blob or split unpredictably.

Tested this myself with a synthetic image containing six fruit at four
different colors, one with simulated blemishes: it correctly found and
labeled five of six (right colors, right maturity labels, blemish flagged
at the right severity), and missed the green one against a green
background — which is exactly the kind of limitation described above,
confirmed rather than assumed.

## Why classical CV, and what's next

The full AGRISENSE plan calls for a YOLO model trained on a labeled
mango/fruit dataset. This prototype started with color + shape rules
instead, which is honest about what it is and gave something to actually
click through immediately, with no dataset or GPU time required.

**That Phase 2 upgrade now exists — see `yolo/README_YOLO.md`.** It adds
a trained YOLOv8 ripeness model as a second engine, toggled with the
`AGRISENSE_ENGINE` environment variable (`classical`, the default, or
`yolo`). Both engines return the exact same `Detection` shape, so
`main.py`'s `run_detection()` is the only place that knows which one is
active — `tracker.py`, `annotate_image()`, `summarize()`, and the entire
frontend are unchanged either way. This also means you can demo both
side by side: "here's the honest heuristic version, here's the trained
model" is a genuinely good thing to show in an interview.

Suggested next milestones after that, in order (matches the original
roadmap):

1. ~~Collect/label a small mango (or your chosen crop) image dataset~~
2. ~~Train a YOLOv8 detection model on it, evaluate with precision/recall/mAP~~
3. ~~Swap it into the detection pipeline~~ — done, see `yolo/README_YOLO.md`
4. Replace the color-based maturity heuristic with a trained ripeness
   *classifier* specifically (right now the YOLO model's classes already
   encode ripeness directly — e.g. `strawberry-ripe` — which is a step in
   this direction; a dedicated classifier head is the next refinement)
5. Add a real disease-classification model (with the "possible disease
   detected — confidence X%, please inspect" language style already used
   here)
6. Move from a single-file web app to the fuller architecture (FastAPI +
   Postgres + React Native mobile + Next.js dashboard) once the core AI is
   validated

## Crop intelligence layer (new — no dataset required)

On top of the vision engine (classical CV or YOLO), every scan now also
gets a rule-based intelligence pass — pure post-processing on the
detections that are already there, no training or new data needed:

- **Explainable health score** (`intelligence.health`) — a 0–100 score
  plus the *actual reasons* behind it (e.g. "Very few fruit flagged for
  surface concerns" / "Wide spread between least and most mature fruit"),
  not just a number.
- **Crop risk engine** (`intelligence.risk`) — disease/anomaly, over-
  ripening, under-ripening, and damage risk as percentages, plus an
  overall Low/Moderate/High label.
- **Harvest readiness score** (`intelligence.harvest_readiness`) — a
  0–100 score, an estimated day-range bucket, a plain-language
  recommendation, and a breakdown of how many fruit are ready /
  developing / unripe / flagged for attention.
- **Fruit-level priority** — every individual detection now also carries
  `priority` (`harvest` / `monitor` / `developing` / `attention`) so you
  can see which specific fruit to act on, not just an aggregate.
- **Scan history + "what changed since last scan"** — type a field label
  (e.g. "Strawberry Bed 01") before analyzing, and each scan is saved
  under that label. The next scan of the same field automatically
  returns `compare_to_previous` — deltas on fruit count, maturity,
  health, readiness, and how many are newly ready to harvest. The
  **View scan history for this field** button in the UI lists every past
  scan for that label.

**Be honest about what this is:** every score above is a documented
weighted formula over the same maturity/health numbers the vision engine
already produces (see `intelligence.py`'s module docstring for the exact
weights). None of it is trained or calibrated against real outcomes — the
"3–6 days" estimate is a structured prototype opinion, not a validated
agronomic prediction. That's exactly the same honesty standard the rest
of this README already holds the vision engine to.

History is stored as one JSON file per field under
`/tmp/agrisense_outputs/history/` — fine for a local single-user
prototype, not meant to survive a server restart in production. See
`history_store.py`'s docstring for how to swap in a real database later
without touching `main.py` beyond the import line.

### New API surface

- `GET /api/fields` — field labels that have at least one saved scan
- `GET /api/history?field=...` — full scan history for a field
- `GET /api/compare?field=...` — diff between the last two scans for a
  field (`available: false` if there's fewer than two)
- `POST /api/analyze/image` and `/api/analyze/video` now also accept
  `field_label` (default `"default"`) and `save_scan` (default `true`,
  set to `false` to analyze without writing to history) form fields, and
  return an `intelligence` object and `compare_to_previous` in the
  response.

## API reference

### `POST /api/analyze/image`
Form field: `file` (image). Optional `min_area`/`min_circularity` (only
affect the classical engine — ignored when `AGRISENSE_ENGINE=yolo`).
Returns detection list, summary stats, a base64 annotated JPEG, and which
`engine` produced the result.

### `POST /api/analyze/video`
Form field: `file` (video, e.g. MP4). Optional `max_frames` (default
1800, ~60s at 30fps — caps processing time on long clips) and
`min_area`/`min_circularity` (classical engine only). Processes every
frame (not sampled), draws boxes on each one, and returns unique
tracked-fruit detections, a per-frame timeline, and
`annotated_video_url` — a downloadable MP4 with boxes on every frame,
ready to post.

### `GET /api/download/{filename}`
Downloads a previously rendered annotated video.

### `GET /api/health`
Liveness check. Also reports which detection engine is active
(`classical-cv-prototype` or `yolo-trained`) and, for the YOLO engine,
the loaded model's class names.
