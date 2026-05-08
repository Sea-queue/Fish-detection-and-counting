"""
Abstract base evaluator — Template Method pattern.

Every evaluation mode (detection, counting) subclasses Evaluator and
implements three hooks:

  setup()       — validate paths, run any pre-processing (e.g. COCO conversion)
  run(weights)  — run one model; return a flat metric dict
  summarize()   — write JSON/CSV summary, print results table

The public entry point is evaluate(), which orchestrates the three hooks.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import torch

from ..config import RunConfig


class Evaluator(ABC):

    def __init__(self, config: RunConfig, registry):
        """
        Parameters
        ----------
        config   : RunConfig frozen dataclass
        registry : DatasetRegistry — used by subclasses in setup()
        """
        self.config   = config
        self.registry = registry
        self.results: List[dict] = []

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def evaluate(self) -> List[dict]:
        """Run setup, then one model at a time, then summarize."""
        self.setup()
        for weights_path in self.config.weights:
            print(f"\n{'='*55}")
            print(f"  Model : {weights_path}")
            print(f"  Dataset: {self.config.dataset_name}")
            print(f"{'='*55}")
            result = self.run(weights_path)
            self.results.append(result)
        self.summarize()
        return self.results

    # ------------------------------------------------------------------ #
    # Template hooks — must be implemented by subclasses
    # ------------------------------------------------------------------ #
    @abstractmethod
    def setup(self) -> None:
        """
        Called once before any models are loaded.
        Validate paths, convert datasets, etc.
        Raise with a clear message if anything is wrong.
        """

    @abstractmethod
    def run(self, weights_path: str) -> dict:
        """
        Run evaluation for one set of weights.
        Return a flat dict of metrics/results.
        """

    @abstractmethod
    def summarize(self) -> None:
        """
        Called once after all models have been evaluated.
        Write JSON + CSV summaries and print the results table.
        """

    # ------------------------------------------------------------------ #
    # Shared helpers available to all subclasses
    # ------------------------------------------------------------------ #
    def _make_run_dir(self, weights_path: str) -> Path:
        """
        Build and create the output directory for one (dataset, weight) pair.

        Structure: {output_root}/{dataset_name}/{weight_stem}/
        """
        stem    = Path(weights_path).stem
        run_dir = Path(self.config.output_root) / self.config.dataset_name / stem
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _require_file(self, path: str | Path, label: str = "File") -> Path:
        """Raise FileNotFoundError with a clear message if path doesn't exist."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p.resolve()}")
        return p
