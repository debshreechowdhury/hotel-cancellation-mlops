"""Thin experiment-tracking wrapper around Weights & Biases.

If tracking.backend is "none", or wandb is missing, or WANDB_MODE=disabled,
every call becomes a no-op so the pipeline still runs (CI, offline grading).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.config import get


class Tracker:
    def __init__(self, cfg: dict, run_name: str | None = None):
        self.run = None
        self.url: str | None = None
        backend = get(cfg, "tracking.backend", "wandb")
        if backend != "wandb" or os.environ.get("WANDB_MODE") == "disabled":
            print("[tracking] disabled")
            return
        try:
            import wandb
        except ImportError:
            print("[tracking] wandb not installed — tracking disabled")
            return
        self._wandb = wandb
        self.run = wandb.init(
            project=get(cfg, "tracking.project", "hotel-cancellation"),
            entity=get(cfg, "tracking.entity"),
            name=run_name,
            config=cfg,
            job_type="train",
        )
        self.url = getattr(self.run, "url", None)
        print(f"[tracking] W&B run: {self.url}")

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if self.enabled:
            self.run.log(metrics)
            for k, v in metrics.items():
                self.run.summary[k] = v

    def log_figure(self, name: str, path: str | Path) -> None:
        if self.enabled:
            self.run.log({name: self._wandb.Image(str(path))})

    def log_table(self, name: str, df) -> None:
        if self.enabled:
            self.run.log({name: self._wandb.Table(dataframe=df)})

    def log_model(self, model_path: str | Path, metadata_path: str | Path,
                  metadata: dict) -> None:
        """Version the trained model as a W&B artifact."""
        if not self.enabled:
            return
        art = self._wandb.Artifact("cancellation-model", type="model",
                                   metadata={k: v for k, v in metadata.items()
                                             if isinstance(v, (str, int, float, bool, dict, list))})
        art.add_file(str(model_path))
        art.add_file(str(metadata_path))
        self.run.log_artifact(art, aliases=["latest"])

    def finish(self) -> None:
        if self.enabled:
            self.run.finish()
