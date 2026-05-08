from ultralytics import YOLO
import torch
import os

# -------- CONFIG --------
WEIGHTS_LIST = [
    "model/train49.pt",
    "model/train95.pt",
    "model/train110.pt",
    "model/train111.pt",
    "model/train113.pt",
]

DATA_YAML = "roboflow_test/roboflow_yolo_test/data.yaml"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMGSZ = 640
CONF = 0.25
MAX_DET = 300
PROJECT = "runs/roboflow_validate"
# ------------------------

print(f"Using device: {DEVICE}")
print(f"Dataset: {DATA_YAML}\n")

for wt in WEIGHTS_LIST:
    run_name = os.path.basename(wt).replace(".pt", "") + "_plots"
    print(f"\n{'='*50}")
    print(f"Validating: {wt}  ->  {PROJECT}/{run_name}")
    print(f"{'='*50}")

    model = YOLO(wt)

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        device=DEVICE,
        imgsz=IMGSZ,
        conf=CONF,
        max_det=MAX_DET,
        plots=True,          # generates confusion matrix, PR curve, F1 curve, etc.
        save_json=True,
        save_txt=True,
        save_conf=True,
        project=PROJECT,
        name=run_name,
    )

    print(f"\n--- {run_name} Results ---")
    print(f"  Precision:      {float(metrics.box.mp):.4f}")
    print(f"  Recall:         {float(metrics.box.mr):.4f}")
    print(f"  mAP@0.5:       {float(metrics.box.map50):.4f}")
    print(f"  mAP@0.75:      {float(metrics.box.map75):.4f}")
    print(f"  mAP@0.5:0.95:  {float(metrics.box.map):.4f}")
    print(f"  Per-class mAP:  {list(map(lambda x: round(float(x), 4), metrics.box.maps))}")

print("\n\nAll done! Check plots in:")
for wt in WEIGHTS_LIST:
    name = os.path.basename(wt).replace(".pt", "") + "_plots"
    print(f"  {PROJECT}/{name}/")