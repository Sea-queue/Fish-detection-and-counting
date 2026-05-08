"""
MetricsSummary — writes JSON + CSV result files and prints an ASCII table.

Files are timestamped so repeated runs never overwrite each other.
Both files land in:  {output_root}/{dataset_name}/_summary/
"""

from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List


class MetricsSummary:

    def __init__(self, output_root: str, dataset_name: str, mode: str):
        """
        Parameters
        ----------
        output_root  : top-level output directory (e.g. "runs/eval")
        dataset_name : name of the dataset being evaluated
        mode         : "detect" or "count"
        """
        self.mode = mode
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_dir = Path(output_root) / dataset_name / "_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = summary_dir / f"{mode}_{ts}.json"
        self.csv_path  = summary_dir / f"{mode}_{ts}.csv"

    # ------------------------------------------------------------------ #
    def write(self, results: List[dict]) -> None:
        """Write JSON and CSV files from the list of result dicts."""
        self._write_json(results)
        self._write_csv(results)
        print(f"\n  [summary] JSON → {self.json_path}")
        print(f"  [summary] CSV  → {self.csv_path}")

    # ------------------------------------------------------------------ #
    def _write_json(self, results: List[dict]) -> None:
        payload = {
            "generated_at": datetime.now().isoformat(),
            "mode":         self.mode,
            "results":      results,
        }
        self.json_path.write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------ #
    def _write_csv(self, results: List[dict]) -> None:
        if not results:
            return

        if self.mode == "detect":
            fieldnames = [
                "dataset", "weights",
                "precision", "recall", "f1",
                "map50", "map75", "map50_95",
                "output_dir",
            ]
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)

        elif self.mode == "count":
            # Discover class names from the first result that has counts
            class_names: list[str] = []
            for r in results:
                if "counts" in r and r["counts"]:
                    class_names = sorted(r["counts"].keys())
                    break

            base_fields = ["dataset", "weights", "video", "output_dir"]
            fieldnames  = base_fields + class_names

            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in results:
                    row = {k: r.get(k, "") for k in base_fields}
                    for cls in class_names:
                        row[cls] = r.get("counts", {}).get(cls, 0)
                    writer.writerow(row)

    # ------------------------------------------------------------------ #
    def print_table(self, results: List[dict]) -> None:
        """Print a formatted ASCII results table to stdout."""
        if not results:
            return

        print(f"\n{'='*70}")
        print(f"  SUMMARY — {self.mode.upper()} — {results[0].get('dataset', '')}")
        print(f"{'='*70}")

        if self.mode == "detect":
            header = f"{'Model':<18} {'P':>6} {'R':>6} {'F1':>6} {'mAP50':>7} {'mAP75':>7} {'mAP5095':>8}"
            print(header)
            print("-" * len(header))
            for r in results:
                model = Path(r["weights"]).stem
                print(
                    f"{model:<18} "
                    f"{r['precision']:>6.4f} "
                    f"{r['recall']:>6.4f} "
                    f"{r['f1']:>6.4f} "
                    f"{r['map50']:>7.4f} "
                    f"{r['map75']:>7.4f} "
                    f"{r['map50_95']:>8.4f}"
                )

        elif self.mode == "count":
            # Collect all class names
            class_names: list[str] = []
            for r in results:
                for cls in r.get("counts", {}).keys():
                    if cls not in class_names:
                        class_names.append(cls)
            class_names = sorted(class_names)

            col_w = 14
            header = f"{'Model':<18}" + "".join(f"{c:>{col_w}}" for c in class_names)
            print(header)
            print("-" * len(header))
            for r in results:
                model = Path(r["weights"]).stem
                row = f"{model:<18}"
                for cls in class_names:
                    row += f"{r.get('counts', {}).get(cls, 0):>{col_w}}"
                print(row)

        print(f"{'='*70}\n")
