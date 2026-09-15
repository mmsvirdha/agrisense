from __future__ import annotations

import base64
import os
import shutil
import subprocess
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import detection
import intelligence
import history_store
from tracker import CentroidTracker

app = FastAPI(title="AGRISENSE AI Prototype", version="0.4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGE_MB = 15
MAX_VIDEO_MB = 200

VIDEO_TIMELINE_STRIDE = 5
VIDEO_MAX_PROCESS_FRAMES = 1800

OUTPUT_DIR = "/tmp/agrisense_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AGRISENSE_ENGINE = os.environ.get("AGRISENSE_ENGINE", "classical").strip().lower()
AGRISENSE_YOLO_WEIGHTS = os.environ.get("AGRISENSE_YOLO_WEIGHTS", "yolo/runs/train/weights/best.pt")
AGRISENSE_YOLO_CONF = float(os.environ.get("AGRISENSE_YOLO_CONF", "0.35"))

_yolo_model = None
if AGRISENSE_ENGINE == "yolo":
    import yolo_detector
    try:
        _yolo_model = yolo_detector.load_model(AGRISENSE_YOLO_WEIGHTS)
        print(f"[agrisense] YOLO engine loaded: {AGRISENSE_YOLO_WEIGHTS}")
        print(f"[agrisense] model classes: {_yolo_model.names}")
    except FileNotFoundError as e:
        print(f"[agrisense] WARNING: {e}")
        print("[agrisense] Falling back to the classical CV engine for this run.")
        AGRISENSE_ENGINE = "classical"


def run_detection(image_bgr: np.ndarray, min_area: int, min_circularity: float) -> list[detection.Detection]:
    if AGRISENSE_ENGINE == "yolo" and _yolo_model is not None:
        return yolo_detector.detect_fruit_blobs_yolo(image_bgr, _yolo_model, conf=AGRISENSE_YOLO_CONF)
    return detection.detect_fruit_blobs(image_bgr, min_area=min_area, min_circularity=min_circularity)


def _bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image. Please upload a JPG or PNG.")
    return img


def _bgr_to_data_url(img_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(500, "Failed to encode result image.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _reencode_for_web(src_path: str, dst_path: str) -> bool:
    if not _ffmpeg_available():
        return False
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", src_path,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-an",
                dst_path,
            ],
            capture_output=True, timeout=900,
        )
        return result.returncode == 0 and os.path.exists(dst_path)
    except Exception:
        return False


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "engine": "yolo-trained" if AGRISENSE_ENGINE == "yolo" else "classical-cv-prototype",
        "yolo_weights": AGRISENSE_YOLO_WEIGHTS if AGRISENSE_ENGINE == "yolo" else None,
        "yolo_classes": list(_yolo_model.names.values()) if _yolo_model is not None else None,
        "version": "0.4.1",
    }


@app.get("/api/fields")
def list_fields():
    return {"fields": history_store.list_fields()}


@app.get("/api/history")
def get_history(field: str = "default"):
    history = history_store.load_history(field)
    return {"field_label": field, "scan_count": len(history), "scans": history}


@app.get("/api/compare")
def get_compare(field: str = "default"):
    diff = history_store.compare_last_two(field)
    if diff is None:
        return {"field_label": field, "available": False,
                "message": "Need at least two saved scans with different results for this field to compare."}
    return {"field_label": field, "available": True, "diff": diff}


@app.post("/api/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    min_area: int = Form(detection.MIN_BLOB_AREA),
    min_circularity: float = Form(detection.MIN_CIRCULARITY),
    field_label: str = Form("default"),
    save_scan: bool = Form(True),
):
    data = await file.read()
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {MAX_IMAGE_MB}MB limit.")

    min_area = max(50, min(min_area, 200000))
    min_circularity = max(0.1, min(min_circularity, 0.95))

    img = _bytes_to_bgr(data)
    t0 = time.time()
    detections = run_detection(img, min_area=min_area, min_circularity=min_circularity)
    elapsed_ms = round((time.time() - t0) * 1000)

    annotated = detection.annotate_image(img, detections)
    summary = detection.summarize(detections)
    report = intelligence.build_intelligence_report(detections)
    detection_dicts = intelligence.annotate_detections_with_priority(
        [d.to_dict() for d in detections], detections
    )

    compare = None
    if save_scan:
        record = {
            "timestamp": time.time(),
            "source_type": "image",
            "summary": summary,
            "intelligence": report,
        }
        history_store.append_scan(field_label, record)
        compare = history_store.compare_last_two(field_label)

    return {
        "summary": summary,
        "detections": detection_dicts,
        "annotated_image": _bgr_to_data_url(annotated),
        "processing_ms": elapsed_ms,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "engine": AGRISENSE_ENGINE,
        "intelligence": report,
        "field_label": field_label,
        "compare_to_previous": compare,
    }


@app.post("/api/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    max_frames: int = Form(VIDEO_MAX_PROCESS_FRAMES),
    min_area: int = Form(detection.MIN_BLOB_AREA),
    min_circularity: float = Form(detection.MIN_CIRCULARITY),
    field_label: str = Form("default"),
    save_scan: bool = Form(True),
):
    data = await file.read()
    if len(data) > MAX_VIDEO_MB * 1024 * 1024:
        raise HTTPException(413, f"Video exceeds {MAX_VIDEO_MB}MB limit.")

    max_frames = max(1, min(max_frames, VIDEO_MAX_PROCESS_FRAMES))
    min_area = max(50, min(min_area, 200000))
    min_circularity = max(0.1, min(min_circularity, 0.95))

    tmp_path = f"/tmp/agrisense_upload_{int(time.time() * 1000)}.mp4"
    with open(tmp_path, "wb") as f:
        f.write(data)

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.remove(tmp_path)
        raise HTTPException(400, "Could not open video. Try an MP4 (H.264) file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_filename = f"agrisense_annotated_{uuid.uuid4().hex}.mp4"
    raw_path = os.path.join(OUTPUT_DIR, f"raw_{output_filename}")
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (frame_w, frame_h))

    if not writer.isOpened():
        cap.release()
        os.remove(tmp_path)
        raise HTTPException(500, "Could not initialize video writer on this server.")

    tracker = CentroidTracker()
    timeline: list[dict] = []
    # Map track_id -> best Detection observed for that track.
    seen_track_ids: dict[int, detection.Detection] = {}

    frame_idx = 0
    t0 = time.time()
    truncated = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx >= max_frames:
            truncated = True
            break

        dets = run_detection(frame, min_area=min_area, min_circularity=min_circularity)
        assignments = tracker.update(dets)
        for d in dets:
            track_id = assignments[d.id]
            prev = seen_track_ids.get(track_id)
            if prev is None or d.confidence > prev.confidence:
                seen_track_ids[track_id] = d

        annotated_frame = detection.annotate_image(frame, dets)
        writer.write(annotated_frame)

        if frame_idx % VIDEO_TIMELINE_STRIDE == 0:
            timeline.append({
                "frame_index": frame_idx,
                "timestamp_s": round(frame_idx / fps, 2) if fps else None,
                "detections_in_frame": len(dets),
                "unique_tracks_so_far": tracker.max_track_id_seen,
            })

        frame_idx += 1

    cap.release()
    writer.release()
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    reencoded = _reencode_for_web(raw_path, output_path)
    if reencoded:
        try:
            os.remove(raw_path)
        except OSError:
            pass
    else:
        os.replace(raw_path, output_path)

    elapsed_ms = round((time.time() - t0) * 1000)

    # Only confirmed tracks (seen in >= min_hits frames) count as "unique
    # fruit" — flicker phantoms are filtered out here so the reported
    # number is meaningful, not inflated by frame-to-frame noise.
    unique_detections = [
        d for tid, d in seen_track_ids.items()
        if tid in tracker._confirmed
    ]
    summary = detection.summarize(unique_detections)
    report = intelligence.build_intelligence_report(unique_detections)
    detection_dicts = intelligence.annotate_detections_with_priority(
        [d.to_dict() for d in unique_detections], unique_detections
    )

    compare = None
    if save_scan:
        record = {
            "timestamp": time.time(),
            "source_type": "video",
            "summary": summary,
            "intelligence": report,
        }
        history_store.append_scan(field_label, record)
        compare = history_store.compare_last_two(field_label)

    return {
        "summary": summary,
        "unique_fruit_detections": detection_dicts,
        "timeline": timeline,
        "annotated_video_url": f"/api/download/{output_filename}",
        "video_meta": {
            "fps": round(fps, 2) if fps else None,
            "total_frames": total_frames or None,
            "frames_processed": frame_idx,
            "truncated": truncated,
            "width": frame_w,
            "height": frame_h,
        },
        "processing_ms": elapsed_ms,
        "engine": AGRISENSE_ENGINE,
        "tracker": {
            "confirmed_tracks": tracker.confirmed_count,
            "raw_tracks_seen": tracker.max_track_id_seen,
            "note": (
                "unique_fruit_detections contains only confirmed tracks "
                "(observed in >= 3 frames). raw_tracks_seen includes "
                "single-frame flicker and is not a meaningful count."
            ),
        },
        "video_encoding": {
            "h264_reencoded": reencoded,
            "note": (
                "Re-encoded to H.264 for broad compatibility."
                if reencoded else
                "ffmpeg wasn't found, so this file uses OpenCV's raw MPEG-4 "
                "output — install ffmpeg for guaranteed-compatible H.264."
            ),
        },
        "note": (
            "Counts reflect confirmed tracks only (>= 3 frames). "
            "The downloadable video has detection boxes on every processed frame."
        ),
        "intelligence": report,
        "field_label": field_label,
        "compare_to_previous": compare,
    }


@app.get("/api/download/{filename}")
def download_annotated_video(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.isfile(path):
        raise HTTPException(404, "That annotated video is no longer available. Please re-analyze.")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename="agrisense_annotated.mp4",
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")