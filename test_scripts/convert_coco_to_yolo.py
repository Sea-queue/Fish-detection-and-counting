"""
Convert COCO-format datasets (Herring EASY / HARD) to YOLO format
and write data.yaml files so they can be used with model.val().

Category mapping:
  COCO category_id=1 (River-Herring)  ->  YOLO class 0 (Herring)

Run from the project root:
  python test_scripts/convert_coco_to_yolo.py
"""

import json
import os
import shutil
from pathlib import Path

# ---- CONFIG ----
DATASETS = [
    {
        "coco_dir": "Box Labeling-Video- HERRING-HARD- Job 3.coco/train",
        "out_dir":  "herring_yolo_test/hard",
        "name":     "herring_hard",
    },
    {
        "coco_dir": "Box Labeling-Video-HERRING-EASY- Job 10.coco/train",
        "out_dir":  "herring_yolo_test/easy",
        "name":     "herring_easy",
    },
]

# COCO category_id -> YOLO class index
# Only category_id=1 (River-Herring) appears in annotations
CATEGORY_MAP = {1: 0}
CLASS_NAMES = {0: "Herring"}
# ----------------


def convert_dataset(coco_dir: str, out_dir: str, name: str):
    coco_dir = Path(coco_dir)
    out_dir = Path(out_dir)
    ann_file = coco_dir / "_annotations.coco.json"

    with open(ann_file) as f:
        coco = json.load(f)

    images_out = out_dir / "images" / "test"
    labels_out = out_dir / "labels" / "test"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Build image id -> filename + dimensions map
    id_to_img = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    from collections import defaultdict
    ann_by_image = defaultdict(list)
    skipped = 0
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in CATEGORY_MAP:
            skipped += 1
            continue
        ann_by_image[ann["image_id"]].append(ann)

    converted = 0
    for img_info in coco["images"]:
        img_id = img_info["id"]
        filename = img_info["file_name"]
        w = img_info["width"]
        h = img_info["height"]

        # Copy image
        src = coco_dir / filename
        dst = images_out / filename
        if src.exists():
            shutil.copy2(src, dst)

        # Write YOLO label file
        label_file = labels_out / (Path(filename).stem + ".txt")
        lines = []
        for ann in ann_by_image[img_id]:
            cls = CATEGORY_MAP[ann["category_id"]]
            x, y, bw, bh = [float(v) for v in ann["bbox"]]  # COCO: x_min, y_min, w, h
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            bw_n = bw / w
            bh_n = bh / h
            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw_n = max(0.0, min(1.0, bw_n))
            bh_n = max(0.0, min(1.0, bh_n))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")

        with open(label_file, "w") as f:
            f.write("\n".join(lines))
        converted += 1

    # Write data.yaml
    abs_out = out_dir.resolve()
    yaml_content = f"""path: {abs_out}
train: images/test
val: images/test
test: images/test
names:
  0: Herring
  1: Non-Herring
"""
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[{name}] Converted {converted} images, skipped {skipped} annotations with unknown categories")
    print(f"[{name}] Output: {out_dir.resolve()}")
    print(f"[{name}] data.yaml: {yaml_path.resolve()}")
    return str(yaml_path.resolve())


if __name__ == "__main__":
    # Run from project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root.resolve()}\n")

    yaml_paths = {}
    for ds in DATASETS:
        yaml_path = convert_dataset(ds["coco_dir"], ds["out_dir"], ds["name"])
        yaml_paths[ds["name"]] = yaml_path
        print()

    print("Conversion complete. data.yaml paths:")
    for name, path in yaml_paths.items():
        print(f"  {name}: {path}")
