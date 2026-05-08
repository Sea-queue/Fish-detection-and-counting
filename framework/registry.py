"""
DatasetRegistry — loads datasets.yaml and provides named lookups.

Usage:
    registry = DatasetRegistry.load("datasets.yaml")
    entry    = registry.get("herring_hard")
    names    = registry.list_names()
    det_only = registry.list_by_type("detection")
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import yaml


# Required fields per (type, format) combination
_REQUIRED_FIELDS = {
    ("detection", "yolo"): ["data_yaml"],
    ("detection", "coco"): ["coco_dir", "category_map", "class_names", "cache_dir"],
    ("counting",  "video"): ["video_path"],
}


class DatasetRegistry:
    def __init__(self, entries: dict):
        self._entries = entries  # {name: raw_dict}

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, yaml_path: str) -> "DatasetRegistry":
        """Parse datasets.yaml and validate required fields for each entry."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(
                f"datasets.yaml not found at: {path.resolve()}\n"
                "Run from the project root, or pass --datasets /path/to/datasets.yaml"
            )

        with open(path) as f:
            raw = yaml.safe_load(f)

        datasets = raw.get("datasets", {})
        if not datasets:
            raise ValueError("datasets.yaml contains no 'datasets:' entries.")

        for name, entry in datasets.items():
            cls._validate(name, entry)

        return cls(datasets)

    # ------------------------------------------------------------------ #
    def get(self, name: str) -> dict:
        """Return the raw entry dict for a dataset name."""
        if name not in self._entries:
            available = ", ".join(sorted(self._entries.keys()))
            raise KeyError(
                f"Dataset '{name}' not found in datasets.yaml.\n"
                f"Available datasets: {available}"
            )
        return self._entries[name]

    def list_names(self) -> List[str]:
        return sorted(self._entries.keys())

    def list_by_type(self, type_: str) -> List[str]:
        return sorted(
            name for name, entry in self._entries.items()
            if entry.get("type") == type_
        )

    def all_entries(self) -> dict:
        return dict(self._entries)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate(name: str, entry: dict):
        """Raise ValueError with a clear message if a required field is missing."""
        dtype  = entry.get("type")
        fmt    = entry.get("format")

        if dtype not in ("detection", "counting"):
            raise ValueError(
                f"Dataset '{name}': 'type' must be 'detection' or 'counting', got '{dtype}'"
            )

        if fmt not in ("yolo", "coco", "video"):
            raise ValueError(
                f"Dataset '{name}': 'format' must be 'yolo', 'coco', or 'video', got '{fmt}'"
            )

        if dtype == "counting" and fmt != "video":
            raise ValueError(
                f"Dataset '{name}': counting datasets must have format: video"
            )
        if dtype == "detection" and fmt == "video":
            raise ValueError(
                f"Dataset '{name}': detection datasets cannot have format: video"
            )

        required = _REQUIRED_FIELDS.get((dtype, fmt), [])
        for field in required:
            if field not in entry:
                raise ValueError(
                    f"Dataset '{name}' (type={dtype}, format={fmt}) "
                    f"is missing required field: '{field}'"
                )
