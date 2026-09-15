"""
Standalone CLI: annotate a video file without running the web server.

    python annotate.py your_video.mp4

Draws detection boxes on every frame plus an on-screen HUD showing
per-frame detection count, running peak count, unique tracked IDs,
frame/time progress, and which engine produced the result.

Writes <name>_annotated.mp4 next to the input, re-encoded to real H.264
(via ffmpeg, if installed) so it's ready to upload directly.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2

import detection
from tracker import CentroidTracker


# ---- HUD styling ---------------------------------------------------------
HUD_BG = (30, 40, 30)         # BGR dark panel
HUD_TEXT = (235, 240, 230)    # off-white
HUD_ACCENT = (60, 200, 245)   # amber-ish
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE = 0.55
HUD_THICK = 1


def _draw_hud(frame, frame_idx, total_frames, fps, dets_count,
              peak_count, unique_tracks, engine_name):
    """Draw a translucent HUD panel on top of the frame with live stats."""
    h, w = frame.shape[:2]

    # --- top bar ---
    top_bar_h = 34
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_bar_h), HUD_BG, -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # progress fraction
    if total_frames > 0:
        progress = min(1.0, frame_idx / total_frames)
    else:
        progress = 0.0

    # frame counter (left)
    frame_txt = f"Frame {frame_idx}/{total_frames if total_frames else '?'}"
    cv2.putText(frame, frame_txt, (14, 23), HUD_FONT, HUD_SCALE, HUD_TEXT, HUD_THICK, cv2.LINE_AA)

    # timestamp (center-left)
    if fps > 0:
        t_sec = frame_idx / fps
        mins = int(t_sec // 60)
        secs = t_sec - mins * 60
        time_txt = f"{mins:02d}:{secs:05.2f}"
    else:
        time_txt = "--:--"
    cv2.putText(frame, time_txt, (200, 23), HUD_FONT, HUD_SCALE, HUD_TEXT, HUD_THICK, cv2.LINE_AA)

    # per-frame detection count (right, colored by count)
    det_color = HUD_TEXT if dets_count == 0 else HUD_ACCENT
    det_txt = f"{dets_count} detections"
    (tw, _), _ = cv2.getTextSize(det_txt, HUD_FONT, HUD_SCALE, HUD_THICK)
    cv2.putText(frame, det_txt, (w - tw - 14, 23), HUD_FONT, HUD_SCALE,
                det_color, HUD_THICK, cv2.LINE_AA)

    # --- thin progress bar under top bar ---
    bar_y = top_bar_h
    cv2.line(frame, (0, bar_y), (w, bar_y), HUD_BG, 2)
    cv2.line(frame, (0, bar_y), (int(w * progress), bar_y), HUD_ACCENT, 2)

    # --- bottom bar ---
    bot_bar_h = 34
    bot_y0 = h - bot_bar_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bot_y0), (w, h), HUD_BG, -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # peak + unique (left)
    left_txt = f"Peak: {peak_count}  |  Unique tracked: {unique_tracks}"
    cv2.putText(frame, left_txt, (14, h - 12), HUD_FONT, HUD_SCALE,
                HUD_TEXT, HUD_THICK, cv2.LINE_AA)

    # engine badge (right)
    eng_txt = f"Engine: {engine_name}"
    (tw, _), _ = cv2.getTextSize(eng_txt, HUD_FONT, HUD_SCALE, HUD_THICK)
    cv2.putText(frame, eng_txt, (w - tw - 14, h - 12), HUD_FONT, HUD_SCALE,
                HUD_ACCENT, HUD_THICK, cv2.LINE_AA)

    return frame


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


def main():
    if len(sys.argv) < 2:
        print("Usage: python annotate.py your_video.mp4")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.rsplit(".", 1)[0] + "_annotated.mp4"

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Could not open", input_file)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Reading {input_file}  ({w}x{h}, {fps:.1f}fps, {total} frames)")

    tracker = CentroidTracker(max_distance=150.0)
    peak_count = 0
    seen_track_ids: dict[int, detection.Detection] = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_path = os.path.join(tmp_dir, "raw.mp4")
        writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        i = 0
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            dets = detection.detect_fruit_blobs(frame)
            assignments = tracker.update(dets)
            for d in dets:
                tid = assignments[d.id]
                prev = seen_track_ids.get(tid)
                if prev is None or d.confidence > prev.confidence:
                    seen_track_ids[tid] = d

            if len(dets) > peak_count:
                peak_count = len(dets)

            frame = detection.annotate_image(frame, dets)
            frame = _draw_hud(
                frame,
                frame_idx=i,
                total_frames=total,
                fps=fps,
                dets_count=len(dets),
                peak_count=peak_count,
                unique_tracks=tracker.max_track_id_seen,
                engine_name="Classical CV",
            )
            writer.write(frame)

            i += 1
            if i % 30 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"\r  frame {i}/{total}  ({rate:.1f} fps)", end="", flush=True)

        cap.release()
        writer.release()
        print()

        if _reencode_for_web(raw_path, output_file):
            print("Saved (H.264):", output_file)
        else:
            shutil.copyfile(raw_path, output_file)
            print("Saved (MPEG-4 — install ffmpeg for guaranteed H.264 compatibility):", output_file)

    # final summary printed to terminal
    print()
    print("=== Summary ===")
    print(f"  Frames processed   : {i}")
    print(f"  Peak detections    : {peak_count} (in a single frame)")
    print(f"  Unique tracks      : {tracker.max_track_id_seen}")
    print(f"  Peak per frame     : {peak_count}")
    print()
    print("Note: 'Unique tracks' can exceed the true number of physical fruit if")
    print("the tracker creates new IDs for fruit that appears, disappears, or")
    print("moves more than ~150px between frames. Treat it as an upper bound.")


if __name__ == "__main__":
    main()