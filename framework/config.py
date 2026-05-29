"""
RunConfig — a plain dataclass that holds every parameter for one evaluation run.

Built by RunConfig.from_cli(args, dataset_entry).
Precedence (highest wins):
  1. Explicit CLI flag
  2. Per-dataset override in datasets.yaml
  3. Hardcoded default in from_cli()

No filesystem access, no ultralytics imports — pure data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import torch


@dataclass(frozen=True)
class RunConfig:
    # --- identity ---
    dataset_name: str
    weights: List[str]

    # --- common inference ---
    device: str
    imgsz: int
    conf: float
    max_det: int
    output_root: str

    # --- detection-specific ---
    split: str = "test"

    # --- counting-specific ---
    exit_margin: float = 0.35
    count_conf: float = 0.70
    majority_ratio: float = 0.70
    min_track_len: int = 10
    grayscale: bool = False
    no_display: bool = False
    count_algorithm: str = "original"  # "original" | "zone" | "stitch" | "all"

    # --- adaptive track filtering (applies to ALL algorithms) ---
    min_track_time: float = 0.0       # seconds; 0 = disabled, use frame-count fallback
    min_track_distance: float = 0.0   # fraction of frame width; 0 = disabled
    min_detection_ratio: float = 0.0  # 0-1; min fraction of frames with high-conf detection; 0 = disabled

    # --- zone/stitch algorithm-specific (does NOT affect original) ---
    zone_entry_margin: float = 0.20
    zone_exit_margin: float = 0.10
    zone_absent_threshold: int = 5
    zone_min_track_length: int = 5

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_cli(args, dataset_entry: dict) -> "RunConfig":
        """
        Merge CLI args with dataset-level overrides from datasets.yaml.
        `args` is the Namespace from argparse.
        `dataset_entry` is the raw dict for one dataset from the registry.
        """
        def _pick(cli_val, yaml_key: str, default):
            """Return cli_val if explicitly set, else yaml override, else default."""
            yaml_val = dataset_entry.get(yaml_key)
            # argparse doesn't mark whether a value was explicitly supplied vs
            # defaulted, so we compare to the parser default stored in args.
            # Values equal to their parser default are considered "not explicitly set".
            if cli_val is not None and cli_val != default:
                return cli_val
            if yaml_val is not None:
                return yaml_val
            return default

        # Auto-detect device if not explicitly given
        if args.device:
            device = args.device
        elif torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        return RunConfig(
            dataset_name=args.dataset,
            weights=args.weights,
            device=device,
            imgsz=_pick(args.imgsz, "imgsz", 640),
            conf=_pick(args.conf, "conf", 0.25),
            max_det=_pick(args.max_det, "max_det", 300),
            output_root=args.output,
            split=dataset_entry.get("split", "test"),
            exit_margin=_pick(
                getattr(args, "exit_margin", None), "exit_margin", 0.35
            ),
            count_conf=_pick(
                getattr(args, "count_conf", None), "count_conf", 0.70
            ),
            majority_ratio=_pick(
                getattr(args, "majority_ratio", None), "majority_ratio", 0.70
            ),
            min_track_len=_pick(
                getattr(args, "min_track_len", None), "min_track_len", 10
            ),
            grayscale=bool(
                getattr(args, "grayscale", False) or dataset_entry.get("grayscale", False)
            ),
            no_display=getattr(args, "no_display", False),
            count_algorithm=getattr(args, "algorithm", "original") or "original",
            min_track_time=float(dataset_entry.get("min_track_time", 0.0)),
            min_track_distance=float(dataset_entry.get("min_track_distance", 0.0)),
            min_detection_ratio=float(dataset_entry.get("min_detection_ratio", 0.0)),
            zone_entry_margin=float(dataset_entry.get("zone_entry_margin", 0.20)),
            zone_exit_margin=float(dataset_entry.get("zone_exit_margin", 0.10)),
            zone_absent_threshold=int(dataset_entry.get("zone_absent_threshold", 5)),
            zone_min_track_length=int(dataset_entry.get("zone_min_track_length", 5)),
        )
