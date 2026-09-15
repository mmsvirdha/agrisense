"""
AGRISENSE AI — YOLOv8 training script

Run this in Google Colab (free T4 GPU, recommended for a first run) or on
your own GPU machine. CPU-only training works too, just slower.

USAGE
-----
    python train_yolo.py --data dataset/data.yaml --epochs 60

After training finishes, your weights are at:
    runs/train/weights/best.pt   (best validation performance)
    runs/train/weights/last.pt   (final epoch)

Copy best.pt into the app as:
    yolo/runs/train/weights/best.pt
(that's the default path main.py looks for — see AGRISENSE_YOLO_WEIGHTS)

WHERE dataset/data.yaml COMES FROM
------------------------------------
See yolo/README_YOLO.md for how to download a labeled strawberry-ripeness
dataset from Roboflow Universe (or export your own CVAT/LabelImg labels in
YOLO format). Either way you end up with:

    dataset/
      train/images/*.jpg   train/labels/*.txt
      valid/images/*.jpg   valid/labels/*.txt
      test/images/*.jpg    test/labels/*.txt   (optional)
      data.yaml
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train the AGRISENSE ripeness detector")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="Base checkpoint to fine-tune. yolov8n.pt = fastest/smallest "
                              "(good for a laptop or Colab free tier). yolov8s.pt is a step up "
                              "in accuracy if you have more GPU time.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs", help="Where to write run outputs")
    parser.add_argument("--name", default="train", help="Run name (output goes to <project>/<name>)")
    parser.add_argument("--patience", type=int, default=15,
                         help="Stop early if val mAP doesn't improve for this many epochs")
    args = parser.parse_args()

    print(f"Loading base checkpoint: {args.model}")
    model = YOLO(args.model)

    print(f"Training on {args.data} for {args.epochs} epochs (imgsz={args.imgsz}, batch={args.batch})")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=args.patience,
        # Light augmentation defaults from ultralytics are already sensible
        # for a fruit dataset (hue/saturation jitter, flips, mosaic). Only
        # touch these if you've actually looked at your training curves and
        # have a reason to.
    )

    print()
    print("Training complete.")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
    print("Run evaluate_yolo.py against that path for a clean metrics report,")
    print("then copy it to yolo/runs/train/weights/best.pt for the app to pick up.")


if __name__ == "__main__":
    main()
