"""
DetectionEvaluator — runs model.val() on image datasets.

Supports:
  - Datasets already in YOLO format (format: yolo)
  - Datasets in COCO format (format: coco) — converted automatically on first use

Outputs per (dataset, model) pair:
  - Confusion matrix, PR curve, F1 curve, P curve, R curve (from plots=True)
  - predictions.json and label .txt files
  - Entry in the run's _summary/ JSON + CSV
"""

from __future__ import annotations
from pathlib import Path

from ultralytics import YOLO

from .base import Evaluator
from ..config import RunConfig
from ..converter import CocoToYoloConverter
from ..reporting.summary import MetricsSummary


class DetectionEvaluator(Evaluator):

    def setup(self) -> None:
        """Resolve the data.yaml path, converting COCO → YOLO if needed."""
        entry = self.registry.get(self.config.dataset_name)

        if entry.get("type") != "detection":
            raise ValueError(
                f"Dataset '{self.config.dataset_name}' has type='{entry.get('type')}'. "
                "Use 'eval.py detect' only with detection datasets."
            )

        fmt = entry.get("format")
        if fmt == "coco":
            converter = CocoToYoloConverter()
            self._data_yaml = converter.convert(entry, project_root=Path.cwd())
        elif fmt == "yolo":
            yaml_path = self._require_file(entry["data_yaml"], label="data.yaml")
            self._data_yaml = str(yaml_path.resolve())
        else:
            raise ValueError(
                f"Dataset '{self.config.dataset_name}': unsupported format '{fmt}'. "
                "Expected 'yolo' or 'coco'."
            )

        print(f"  [detect] data.yaml → {self._data_yaml}")

    def run(self, weights_path: str) -> dict:
        """Run model.val() for one set of weights. Returns a flat metric dict."""
        self._require_file(weights_path, label="Weights file")
        run_dir = self._make_run_dir(weights_path)

        model = YOLO(weights_path)

        # ultralytics writes outputs to project/name/.
        # Pass absolute paths so ultralytics doesn't prepend its own "runs/detect/".
        metrics = model.val(
            data=self._data_yaml,
            split=self.config.split,
            device=self.config.device,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            max_det=self.config.max_det,
            plots=True,       # confusion matrix, PR/F1/P/R curves — always on
            save_json=True,
            save_txt=True,
            save_conf=True,
            project=str(run_dir.parent.resolve()),
            name=run_dir.name,
        )

        p   = float(metrics.box.mp)
        r   = float(metrics.box.mr)
        f1  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        m50 = float(metrics.box.map50)
        m75 = float(metrics.box.map75)
        m   = float(metrics.box.map)

        result = {
            "dataset":       self.config.dataset_name,
            "weights":       weights_path,
            "precision":     round(p,   4),
            "recall":        round(r,   4),
            "f1":            round(f1,  4),
            "map50":         round(m50, 4),
            "map75":         round(m75, 4),
            "map50_95":      round(m,   4),
            "per_class_map": [round(float(x), 4) for x in metrics.box.maps],
            "output_dir":    str(run_dir),
        }

        # Print inline summary immediately after this model completes
        print(f"\n  --- Results: {Path(weights_path).stem} on {self.config.dataset_name} ---")
        print(f"  Precision    : {p:.4f}")
        print(f"  Recall       : {r:.4f}")
        print(f"  F1           : {f1:.4f}")
        print(f"  mAP@0.5      : {m50:.4f}")
        print(f"  mAP@0.75     : {m75:.4f}")
        print(f"  mAP@0.5:0.95 : {m:.4f}")
        print(f"  Per-class mAP: {result['per_class_map']}")
        print(f"  Plots saved  : {run_dir}/")

        return result

    def summarize(self) -> None:
        """Write JSON + CSV summary for this dataset's entire run."""
        writer = MetricsSummary(
            output_root=self.config.output_root,
            dataset_name=self.config.dataset_name,
            mode="detect",
        )
        writer.write(self.results)
        writer.print_table(self.results)
