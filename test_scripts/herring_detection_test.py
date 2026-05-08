"""
Run YOLO validation on Herring EASY and HARD test datasets.
Produces: Precision, Recall, F1, mAP@0.5, mAP@0.75, mAP@0.5:0.95,
          confusion matrix, PR curve, F1 curve, and per-class mAP.

Prerequisites: run convert_coco_to_yolo.py first (once).

Run from the project root:
  python test_scripts/herring_detection_test.py
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path

# -------- CONFIG --------
WEIGHTS_LIST = [
    "model/train49.pt",
    "model/train95.pt",
    "model/train110.pt",
    "model/train111.pt",
    "model/train113.pt",
]

DATASETS = {
    "herring_hard": "herring_yolo_test/hard/data.yaml",
    "herring_easy": "herring_yolo_test/easy/data.yaml",
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMGSZ = 640
CONF = 0.25
MAX_DET = 300
PROJECT = "runs/herring_validate"
# ------------------------

# Run from project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)

print(f"Using device: {DEVICE}")

for dataset_name, data_yaml in DATASETS.items():
    print(f"\n{'#'*60}")
    print(f"  DATASET: {dataset_name.upper()}")
    print(f"{'#'*60}")

    for wt in WEIGHTS_LIST:
        run_name = f"{os.path.basename(wt).replace('.pt', '')}_{dataset_name}"
        print(f"\n{'='*55}")
        print(f"Validating: {wt}  ->  {PROJECT}/{run_name}")
        print(f"{'='*55}")

        model = YOLO(wt)

        metrics = model.val(
            data=data_yaml,
            split="test",
            device=DEVICE,
            imgsz=IMGSZ,
            conf=CONF,
            max_det=MAX_DET,
            plots=True,       # confusion matrix, PR curve, F1 curve, etc.
            save_json=True,
            save_txt=True,
            save_conf=True,
            project=PROJECT,
            name=run_name,
        )

        # F1 = 2 * P * R / (P + R)
        p = float(metrics.box.mp)
        r = float(metrics.box.mr)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        print(f"\n--- {run_name} Results ---")
        print(f"  Precision:      {p:.4f}")
        print(f"  Recall:         {r:.4f}")
        print(f"  F1 Score:       {f1:.4f}")
        print(f"  mAP@0.5:        {float(metrics.box.map50):.4f}")
        print(f"  mAP@0.75:       {float(metrics.box.map75):.4f}")
        print(f"  mAP@0.5:0.95:   {float(metrics.box.map):.4f}")
        print(f"  Per-class mAP:  {list(map(lambda x: round(float(x), 4), metrics.box.maps))}")
        print(f"  Plots saved to: {PROJECT}/{run_name}/")

print("\n\nAll done! Check plots in:")
for dataset_name in DATASETS:
    for wt in WEIGHTS_LIST:
        name = f"{os.path.basename(wt).replace('.pt', '')}_{dataset_name}"
        print(f"  {PROJECT}/{name}/")
