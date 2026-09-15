"""
AGRISENSE AI — YOLO evaluation script

Reports real precision / recall / mAP@50 / mAP@50-95 on the validation (or
test) split — the numbers to put in your README/CV instead of a made-up
accuracy figure. See the "Very important — evaluation" section of the
original AGRISENSE plan: report what you actually measured, not a guess.

USAGE
-----
    python evaluate_yolo.py --weights runs/train/weights/best.pt --data dataset/data.yaml
    python evaluate_yolo.py --weights runs/train/weights/best.pt --data dataset/data.yaml --split test
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained AGRISENSE ripeness model")
    parser.add_argument("--weights", default="runs/train/weights/best.pt")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split)

    print()
    print("=" * 50)
    print(f"AGRISENSE — evaluation on '{args.split}' split")
    print("=" * 50)
    print(f"  mAP@50:      {metrics.box.map50:.3f}")
    print(f"  mAP@50-95:   {metrics.box.map:.3f}")
    print(f"  Precision:   {metrics.box.mp:.3f}")
    print(f"  Recall:      {metrics.box.mr:.3f}")
    print()
    print("Per-class mAP@50:")
    for i, class_name in model.names.items():
        try:
            print(f"  {class_name:25s} {metrics.box.ap50[i]:.3f}")
        except (IndexError, KeyError):
            pass
    print()
    print("Copy these numbers into your README/CV — that's the honest,")
    print("measured version of 'X% accurate', not a guess.")


if __name__ == "__main__":
    main()
