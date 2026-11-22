"""Metrics and diagnostic plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (Docker / CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, proba, threshold: float, prefix: str) -> dict[str, float]:
    y_pred = (proba >= threshold).astype(int)
    return {
        f"{prefix}/accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}/precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}/recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}/f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}/roc_auc": float(roc_auc_score(y_true, proba)),
        f"{prefix}/pr_auc": float(average_precision_score(y_true, proba)),
    }


def plot_confusion_matrix(y_true, proba, threshold: float, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, (proba >= threshold).astype(int),
        display_labels=["Kept", "Cancelled"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Test confusion matrix")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_roc(y_true, proba, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax, name="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_title("Test ROC curve")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def feature_importance(pipeline, X: pd.DataFrame, y, n_samples: int, n_repeats: int,
                       seed: int) -> pd.DataFrame:
    """Model-agnostic permutation importance on raw input columns."""
    if len(X) > n_samples:
        idx = np.random.RandomState(seed).choice(len(X), n_samples, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    result = permutation_importance(pipeline, X, y, n_repeats=n_repeats,
                                    random_state=seed, scoring="roc_auc", n_jobs=1)
    return (pd.DataFrame({"feature": X.columns,
                          "importance": result.importances_mean,
                          "std": result.importances_std})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


def plot_importance(imp: pd.DataFrame, out: Path, top: int = 15) -> Path:
    top_df = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top_df["feature"], top_df["importance"], xerr=top_df["std"], color="#3b6fb6")
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Permutation feature importance (validation)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
