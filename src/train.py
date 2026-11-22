"""End-to-end training pipeline:
ingest (GCS) -> split -> build pipeline (feature eng + preprocessing + model)
-> train -> evaluate (val + test) -> log to W&B -> save model + metadata.

Usage:
    python -m src.train
    python -m src.train --set model.type=random_forest
    python -m src.train --set data.source=local --set tracking.backend=none
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import evaluate
from src.config import feature_lists, get, load_config
from src.features import build_pipeline, input_columns
from src.ingest import load_dataset
from src.tracking import Tracker


def split_data(df: pd.DataFrame, cfg: dict):
    target = get(cfg, "data.target")
    X = df[input_columns(cfg)]
    y = df[target].astype(int)
    seed = get(cfg, "project.seed", 42)
    test_size = float(get(cfg, "split.test_size", 0.15))
    val_size = float(get(cfg, "split.val_size", 0.15))
    stratify = get(cfg, "split.stratify", True)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if stratify else None)
    # val_size is a fraction of the FULL dataset
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, random_state=seed,
        stratify=y_tmp if stratify else None)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def run(cfg: dict) -> dict:
    seed = get(cfg, "project.seed", 42)
    np.random.seed(seed)
    model_type = get(cfg, "model.type")
    threshold = float(get(cfg, "model.threshold", 0.5))
    run_name = f"{model_type}-{datetime.now().strftime('%m%d-%H%M%S')}"
    tracker = Tracker(cfg, run_name=run_name)

    try:
        # 1. Ingest
        df = load_dataset(cfg)

        # 2. Split
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, cfg)
        print(f"[train] split train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")
        tracker.log_metrics({"data/n_train": len(X_train), "data/n_val": len(X_val),
                             "data/n_test": len(X_test),
                             "data/cancel_rate": float(df[get(cfg, 'data.target')].mean())})

        # 3. Train
        pipeline = build_pipeline(cfg)
        print(f"[train] fitting {model_type} ...")
        pipeline.fit(X_train, y_train)

        # 4. Evaluate
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = {**evaluate.compute_metrics(y_val, val_proba, threshold, "val"),
                   **evaluate.compute_metrics(y_test, test_proba, threshold, "test")}
        tracker.log_metrics(metrics)
        for k, v in metrics.items():
            print(f"  {k:<16} {v:.4f}")

        reports = Path(get(cfg, "output.reports_dir", "reports"))
        reports.mkdir(parents=True, exist_ok=True)
        cm = evaluate.plot_confusion_matrix(y_test, test_proba, threshold, reports / "confusion_matrix.png")
        roc = evaluate.plot_roc(y_test, test_proba, reports / "roc_curve.png")
        tracker.log_figure("plots/confusion_matrix", cm)
        tracker.log_figure("plots/roc_curve", roc)

        imp = evaluate.feature_importance(
            pipeline, X_val, y_val,
            n_samples=int(get(cfg, "evaluation.permutation_importance_samples", 2000)),
            n_repeats=int(get(cfg, "evaluation.permutation_importance_repeats", 3)),
            seed=seed)
        imp.to_csv(reports / "feature_importance.csv", index=False)
        fi = evaluate.plot_importance(imp, reports / "feature_importance.png")
        tracker.log_figure("plots/feature_importance", fi)
        tracker.log_table("feature_importance", imp)

        # 5. Save model + metadata
        model_dir = Path(get(cfg, "output.model_dir", "models"))
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / get(cfg, "output.model_file", "model.joblib")
        meta_path = model_dir / get(cfg, "output.metadata_file", "metadata.json")
        joblib.dump(pipeline, model_path, compress=3)

        numeric, categorical, engineered = feature_lists(cfg)
        categories = {c: sorted(map(str, df[c].dropna().unique())) for c in categorical}
        metadata = {
            "model_type": model_type,
            "model_version": run_name,
            "threshold": threshold,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "wandb_run_url": tracker.url,
            "features": {"numeric": numeric, "categorical": categorical, "engineered": engineered},
            "categories": categories,
            "metrics": metrics,
            "top_features": imp.head(10)[["feature", "importance"]].to_dict(orient="records"),
        }
        meta_path.write_text(json.dumps(metadata, indent=2))
        tracker.log_model(model_path, meta_path, metadata)
        size_mb = model_path.stat().st_size / 1e6
        print(f"[train] saved {model_path} ({size_mb:.1f} MB) and {meta_path}")
        return metadata
    finally:
        tracker.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a config value (repeatable)")
    args = parser.parse_args()
    run(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
