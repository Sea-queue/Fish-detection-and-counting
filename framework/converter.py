"""
CocoToYoloConverter — converts a COCO-format image dataset to YOLO format.

The conversion is cached: if cache_dir/data.yaml already exists, it is
returned immediately without re-doing the work.  To force re-conversion,
delete the cache_dir manually.

Handles the quirk in the herring datasets where bbox values may be stored
as strings (e.g. '95.86') instead of floats.
"""

from __future__ import annotations
import json
import shutil
from collections import defaultdict
from pathlib import Path


class CocoToYoloConverter:

    def convert(self, entry: dict, project_root: Path | None = None) -> str:
        """
        Convert a COCO dataset described by a datasets.yaml entry.

        Parameters
        ----------
        entry : dict
            The raw datasets.yaml dict for one dataset (type=detection, format=coco).
        project_root : Path, optional
            Resolve relative paths against this root.  Defaults to cwd.

        Returns
        -------
        str
            Absolute path to the generated data.yaml file.
        """
        root = Path(project_root) if project_root else Path.cwd()

        coco_dir    = root / entry["coco_dir"]
        cache_dir   = root / entry["cache_dir"]
        ann_filename = entry.get("coco_annotation_file", "_annotations.coco.json")
        ann_file    = coco_dir / ann_filename
        category_map = {int(k): int(v) for k, v in entry["category_map"].items()}
        class_names  = {int(k): v for k, v in entry["class_names"].items()}

        # --- cache hit ---
        yaml_path = cache_dir / "data.yaml"
        if yaml_path.exists():
            print(f"  [converter] Cache hit — reusing {yaml_path}")
            return str(yaml_path.resolve())

        print(f"  [converter] Converting COCO → YOLO: {coco_dir}")

        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        with open(ann_file) as f:
            coco = json.load(f)

        images_out = cache_dir / "images" / "test"
        labels_out = cache_dir / "labels" / "test"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        # Build image id → metadata map
        id_to_img = {img["id"]: img for img in coco["images"]}

        # Group annotations by image_id, skipping unknown categories
        ann_by_image: dict[int, list] = defaultdict(list)
        skipped = 0
        for ann in coco["annotations"]:
            cat_id = int(ann["category_id"])
            if cat_id not in category_map:
                skipped += 1
                continue
            ann_by_image[ann["image_id"]].append(ann)

        converted = 0
        missing_images = 0

        for img_info in coco["images"]:
            img_id   = img_info["id"]
            filename = img_info["file_name"]
            w        = img_info["width"]
            h        = img_info["height"]

            # Copy image
            src = coco_dir / filename
            if src.exists():
                shutil.copy2(src, images_out / filename)
            else:
                missing_images += 1

            # Write YOLO label file
            label_file = labels_out / (Path(filename).stem + ".txt")
            lines = []
            for ann in ann_by_image.get(img_id, []):
                cls  = category_map[int(ann["category_id"])]
                x, y, bw, bh = [float(v) for v in ann["bbox"]]
                cx, cy, bwn, bhn = self._to_yolo(x, y, bw, bh, w, h)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bwn:.6f} {bhn:.6f}")

            label_file.write_text("\n".join(lines))
            converted += 1

        # Build class_names list in YOLO index order
        max_idx  = max(class_names.keys())
        name_list = [class_names.get(i, f"class_{i}") for i in range(max_idx + 1)]
        names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(name_list))

        yaml_content = (
            f"path: {cache_dir.resolve()}\n"
            f"train: images/test\n"
            f"val:   images/test\n"
            f"test:  images/test\n"
            f"names:\n{names_yaml}\n"
        )
        yaml_path.write_text(yaml_content)

        if skipped:
            print(f"  [converter] Skipped {skipped} annotations with unmapped category IDs")
        if missing_images:
            print(f"  [converter] Warning: {missing_images} images referenced in JSON were not found")
        print(f"  [converter] Done — {converted} images, data.yaml → {yaml_path}")

        return str(yaml_path.resolve())

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_yolo(
        x: float, y: float, bw: float, bh: float,
        img_w: int, img_h: int,
    ) -> tuple[float, float, float, float]:
        """Convert COCO bbox (x_min, y_min, w, h) to YOLO (cx, cy, w, h) normalised."""
        cx  = (x + bw / 2) / img_w
        cy  = (y + bh / 2) / img_h
        bwn = bw / img_w
        bhn = bh / img_h
        # clamp to [0, 1] to handle any floating-point overshoot
        clamp = lambda v: max(0.0, min(1.0, v))
        return clamp(cx), clamp(cy), clamp(bwn), clamp(bhn)
