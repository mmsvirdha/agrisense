# AGRISENSE AI — YOLO Ripeness Model (Phase 2)

This is the swap-in the main README always pointed at: a real trained
model instead of the color/shape heuristic in `detection.py`. Nothing
downstream changes — `main.py`, `tracker.py`, `annotate_image()`, and the
whole frontend keep working exactly as they do today. You're only
changing *how a box gets found*, not what happens with it afterward.

Do these four steps in order. Total time: maybe 2-3 hours the first time,
almost entirely GPU-training wait time, not your own effort.

---

## 1. Get a labeled dataset

You don't have to label a strawberry dataset from scratch — a few good
public ones already exist on Roboflow Universe with `ripe`/`unripe`
classes, licensed CC BY 4.0 (free to use with attribution):

- **Strawberry-Detection** by Harvesting Robot Datasets — 1,060 images,
  classes `strawberry-ripe` / `strawberry-unripe`
  https://universe.roboflow.com/harvesting-robot-datasets/strawberry-detection-msf0m
- **Strawberry-detection** by Capstone Project — 1,753 images
  https://universe.roboflow.com/capstone-project-3jugn/strawberry-detection-yrdsj
- Search https://universe.roboflow.com/search?q=class:ripe+strawberry for
  more — filter to "Object Detection" + a permissive license.

**To download one:**

1. Create a free Roboflow account.
2. Open the dataset link above → **Download Dataset** → format **YOLOv8**.
3. Either download the zip directly and unzip it into `yolo/dataset/`, or
   use the code Roboflow shows you (needs `pip install roboflow`):

   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")   # find this under your Roboflow account settings
   project = rf.workspace("harvesting-robot-datasets").project("strawberry-detection-msf0m")
   dataset = project.version(1).download("yolov8", location="dataset")
   ```

Either way you should end up with:

```
yolo/dataset/
  train/images/*.jpg   train/labels/*.txt
  valid/images/*.jpg   valid/labels/*.txt
  test/images/*.jpg    test/labels/*.txt      (optional)
  data.yaml
```

`data.yaml` tells YOLO the class names and where the splits live — it's
generated automatically by the Roboflow export, you shouldn't need to
edit it unless you're merging datasets.

### Optional but worth it: add your own footage

Run `extract_frames.py` on your strawberry garden video, hand-label
~50-100 of the extracted frames in Roboflow (free, draw a box + pick a
class per berry), and merge those into `dataset/train/`. This is what
actually closes the gap between "works on the public dataset's photos"
and "works on your camera, your lighting, your plants." See the
docstring in `extract_frames.py` for the exact workflow.

---

## 2. Install training dependencies

```bash
cd yolo
pip install -r requirements-yolo.txt
```

This pulls in `ultralytics` (the YOLOv8 training/inference library) —
kept separate from the main app's `requirements.txt` so a
classical-CV-only deployment doesn't need to install PyTorch.

---

## 3. Train

**Recommended for a first run: Google Colab** (free T4 GPU, ~15-25
minutes for 60 epochs on ~1000 images with `yolov8n`). Upload
`yolo/dataset/` and `train_yolo.py` to a Colab notebook, `!pip install
ultralytics`, then:

```bash
python train_yolo.py --data dataset/data.yaml --epochs 60
```

Running locally with a GPU works the same way. CPU-only works too, just
count on it being much slower (maybe hours instead of minutes) — if
that's your situation, drop `--epochs` to something like 25 for a first
pass just to confirm the pipeline works, then run a longer job overnight.

When it finishes:

```
runs/train/weights/best.pt   <- use this one
runs/train/weights/last.pt
```

---

## 4. Evaluate honestly, then swap it into the app

```bash
python evaluate_yolo.py --weights runs/train/weights/best.pt --data dataset/data.yaml
```

This prints real precision / recall / mAP@50 / mAP@50-95 — put these
numbers in your README and CV, not a guessed accuracy figure. That's the
same "report what you measured" discipline the classical-CV README
already followed with its 5/6 test result.

**To make the app use it:**

```bash
# from the project root (one level up from yolo/)
export AGRISENSE_ENGINE=yolo
export AGRISENSE_YOLO_WEIGHTS=yolo/runs/train/weights/best.pt
uvicorn main:app --reload
```

Check `GET /api/health` — it now reports `"engine": "yolo-trained"` and
lists your model's class names. Upload a photo/video the same way as
before; nothing else about using the app changes.

If `AGRISENSE_YOLO_WEIGHTS` doesn't point at a real file, `main.py` logs a
warning and falls back to the classical engine automatically — you can't
accidentally serve a broken app by forgetting this step.

### If your dataset's class names don't match the defaults

Open `yolo_detector.py` and look at `CLASS_MATURITY_MAP` near the top.
Whatever class names your `data.yaml` uses (printed at startup, or check
the file directly), make sure each one has a row mapping it to a
`color_profile` / `maturity_label` / `maturity_score`. Classes not listed
fall back to a generic "Detected (unmapped class)" reading rather than
crashing — but you'll want every real class mapped for a clean demo.

---

## What to say about this honestly

- The classical CV engine is still in the repo and still works
  (`AGRISENSE_ENGINE=classical`, the default) — keeping both lets you
  show a side-by-side "heuristic vs. trained model" comparison, which is
  a genuinely good interview talking point.
- A model trained on ~1,000 public images plus a small batch of your own
  footage is a real, defensible "Phase 2" result — don't inflate it into
  "production-grade" without more data and testing across more
  conditions (different lighting, camera angles, growth stages).
- Report the actual mAP/precision/recall numbers from `evaluate_yolo.py`.
  If they're mediocre on your own footage specifically, that's useful
  information (probably means: label more of your own frames), not
  something to hide.
