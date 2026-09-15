"""
AGRISENSE AI — extract frames from a video for labeling

The public Roboflow dataset gets you a working model fast, but it won't
perfectly match your own strawberry video's camera angle, lighting, or
plant variety. Mixing in ~30-100 frames from your own footage (labeled by
hand in Roboflow or CVAT and added to the same dataset before training)
usually improves real-world accuracy more than anything else you can do
in an afternoon — this is what "domain adaptation" means in practice.

USAGE
-----
    python extract_frames.py my_strawberry_video.mp4 --every 15 --out frames/

Then upload the images in frames/ to your Roboflow project (or a new one)
and draw boxes around ripe/unripe berries. Roboflow's annotate tool is
free and drawing ~50-100 boxes takes well under an hour.
"""

from __future__ import annotations

import argparse
import os

import cv2


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video for dataset labeling")
    parser.add_argument("video", help="Path to the source video")
    parser.add_argument("--every", type=int, default=15,
                         help="Save 1 out of every N frames (default 15 — for a 30fps clip, "
                              "that's 2 frames/second, plenty for labeling without huge overlap)")
    parser.add_argument("--out", default="frames", help="Output directory")
    parser.add_argument("--max", type=int, default=150,
                         help="Stop after saving this many frames (default 150 — more than "
                              "enough to hand-label a useful supplement to the public dataset)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")

    frame_idx = 0
    saved = 0
    while saved < args.max:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.every == 0:
            out_path = os.path.join(args.out, f"{base}_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved += 1
            print(f"\rSaved {saved}/{args.max} frames", end="", flush=True)
        frame_idx += 1

    cap.release()
    print()
    print(f"Done. {saved} frames written to {args.out}/")
    print("Next: upload these to Roboflow (or your labeling tool of choice), draw boxes,")
    print("export in YOLOv8 format, and merge into dataset/train (and a few into dataset/valid).")


if __name__ == "__main__":
    main()
