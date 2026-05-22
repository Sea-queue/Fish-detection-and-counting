#!/usr/bin/env python3
"""
eval.py — Fish Detection & Counting Evaluation CLI
===================================================

Run from the project root (same directory as datasets.yaml).

Usage
-----
  # List all registered datasets
  python eval.py --list-datasets

  # Detection — one dataset, one model
  python eval.py detect --dataset herring_hard --weights model/train113.pt

  # Detection — multiple datasets and models in one command
  python eval.py detect \
      --dataset herring_hard herring_easy roboflow \
      --weights model/train110.pt model/train113.pt

  # Counting — video-based fish counting
  python eval.py count --dataset nemasket_normal --weights model/train113.pt

  # Counting — suppress the preview window (headless / HPC)
  python eval.py count --dataset count5 --weights model/train113.pt --no-display

Notes
-----
  • Adding a new dataset requires only an entry in datasets.yaml — no code changes.
  • COCO-format detection datasets are converted to YOLO format automatically on
    first use and cached; subsequent runs skip conversion.
  • Results (metrics JSON, CSV, confusion matrix, PR/F1 curves) are written to
    runs/eval/{dataset_name}/{weight_stem}/ and runs/eval/{dataset_name}/_summary/.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `framework` is importable
sys.path.insert(0, str(Path(__file__).parent))

from framework.registry import DatasetRegistry
from framework.config import RunConfig
from framework.evaluators import DetectionEvaluator, CountingEvaluator

DATASETS_YAML = Path(__file__).parent / "datasets.yaml"
DEFAULT_OUTPUT = "runs/eval"


# ============================================================================ #
# Argument parsing
# ============================================================================ #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval.py",
        description="Fish detection & counting evaluation framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        default=str(DATASETS_YAML),
        metavar="PATH",
        help=f"Path to datasets.yaml (default: {DATASETS_YAML})",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all registered datasets and exit",
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # ------------------------------------------------------------------ #
    # Common arguments shared by both subcommands
    # ------------------------------------------------------------------ #
    def add_common(sub):
        sub.add_argument(
            "--dataset", "-d",
            required=True,
            nargs="+",
            metavar="NAME",
            help="Dataset name(s) from datasets.yaml",
        )
        sub.add_argument(
            "--weights", "-w",
            required=True,
            nargs="+",
            metavar="PATH",
            help="Model weight file(s) (.pt)",
        )
        sub.add_argument(
            "--output", "-o",
            default=DEFAULT_OUTPUT,
            metavar="DIR",
            help=f"Output root directory (default: {DEFAULT_OUTPUT})",
        )
        sub.add_argument("--imgsz",   type=int,   default=None, help="Inference image size (default: 640)")
        sub.add_argument("--conf",    type=float, default=None, help="Detection confidence threshold (default: 0.25)")
        sub.add_argument("--max-det", type=int,   default=None, dest="max_det", help="Max detections per image/frame (default: 300)")
        sub.add_argument("--device",  type=str,   default=None, help="Device: mps | cuda | cpu (default: auto)")

    # ------------------------------------------------------------------ #
    # detect subcommand
    # ------------------------------------------------------------------ #
    detect_p = subparsers.add_parser(
        "detect",
        help="Run model.val() on image datasets (frames / labeled images)",
    )
    add_common(detect_p)

    # ------------------------------------------------------------------ #
    # count subcommand
    # ------------------------------------------------------------------ #
    count_p = subparsers.add_parser(
        "count",
        help="Run model.track() on videos and count fish by exit detection",
    )
    add_common(count_p)
    count_p.add_argument(
        "--exit-margin", type=float, default=None, dest="exit_margin",
        help="Exit zone as fraction of frame width (default: 0.35)",
    )
    count_p.add_argument(
        "--count-conf", type=float, default=None, dest="count_conf",
        help="Min confidence to record a detection for counting (default: 0.70)",
    )
    count_p.add_argument(
        "--majority-ratio", type=float, default=None, dest="majority_ratio",
        help="Min fraction of history needed for class vote (default: 0.70)",
    )
    count_p.add_argument(
        "--min-track-len", type=int, default=None, dest="min_track_len",
        help="Min high-conf frames a track needs before it can be counted (default: 10)",
    )
    count_p.add_argument(
        "--grayscale", action="store_true",
        help="Convert each frame to grayscale before inference (helps on overexposed/bright video)",
    )
    count_p.add_argument(
        "--no-display", action="store_true",
        help="Suppress the OpenCV preview window (useful for headless runs)",
    )

    return parser


# ============================================================================ #
# Helpers
# ============================================================================ #

def list_datasets(registry: DatasetRegistry) -> None:
    """Print a summary table of all registered datasets."""
    entries = registry.all_entries()
    col = 28
    print(f"\n{'Name':<{col}} {'Type':<12} {'Format':<8} Path / Key")
    print("-" * 80)
    for name in sorted(entries.keys()):
        e    = entries[name]
        dtype = e.get("type", "?")
        fmt  = e.get("format", "?")
        if fmt == "yolo":
            path = e.get("data_yaml", "")
        elif fmt == "coco":
            path = e.get("coco_dir", "")
        else:
            path = e.get("video_path", "")
        print(f"{name:<{col}} {dtype:<12} {fmt:<8} {path}")
    print()


def validate_weights(weights: list[str]) -> None:
    """Abort early if any weight file is missing."""
    for w in weights:
        if not Path(w).exists():
            sys.exit(f"Error: weights file not found: {w}")


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Change to project root so relative paths in datasets.yaml resolve correctly
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Load registry
    try:
        registry = DatasetRegistry.load(args.datasets)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error loading datasets.yaml: {e}")

    # --list-datasets (can be used without a subcommand)
    if args.list_datasets:
        list_datasets(registry)
        return

    if not args.subcommand:
        parser.print_help()
        return

    # Validate weights exist before loading any model
    validate_weights(args.weights)

    # Normalise: --dataset can be one or multiple names
    dataset_names = args.dataset  # already a list from nargs="+"

    for dataset_name in dataset_names:
        # Fetch entry (raises KeyError with helpful message if unknown)
        try:
            entry = registry.get(dataset_name)
        except KeyError as e:
            sys.exit(str(e))

        # Type guard: make sure the right subcommand is used for this dataset
        if args.subcommand == "detect" and entry.get("type") != "detection":
            sys.exit(
                f"Error: dataset '{dataset_name}' has type='{entry.get('type')}'. "
                "Use 'count' subcommand for counting datasets."
            )
        if args.subcommand == "count" and entry.get("type") != "counting":
            sys.exit(
                f"Error: dataset '{dataset_name}' has type='{entry.get('type')}'. "
                "Use 'detect' subcommand for detection datasets."
            )

        # Patch args.dataset for RunConfig (which expects a single name)
        args.dataset = dataset_name

        print(f"\n{'#'*60}")
        print(f"  Dataset : {dataset_name}  ({entry['type']} / {entry['format']})")
        print(f"  Models  : {args.weights}")
        print(f"  Output  : {args.output}/{dataset_name}/")
        print(f"{'#'*60}")

        config = RunConfig.from_cli(args, entry)
        print(f"  Device  : {config.device}")

        if args.subcommand == "detect":
            evaluator = DetectionEvaluator(config, registry)
        else:
            evaluator = CountingEvaluator(config, registry)

        evaluator.evaluate()


if __name__ == "__main__":
    main()
