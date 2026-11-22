"""Feature engineering + model construction as a single sklearn Pipeline.

Everything that transforms raw booking fields lives inside the Pipeline, so the
saved model accepts the same raw JSON fields the API receives. That rules out
the most common bug in deployed ML: training/serving skew.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import feature_lists, get


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns. Must stay a module-level function so joblib can pickle it."""
    df = df.copy()
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    return df


def build_classifier(cfg: dict):
    model_type = get(cfg, "model.type")
    params = dict(get(cfg, f"model.params.{model_type}", {}) or {})
    seed = get(cfg, "project.seed", 42)
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if model_type == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    raise ValueError(f"Unknown model.type: {model_type!r}")


def build_pipeline(cfg: dict) -> Pipeline:
    numeric, categorical, engineered = feature_lists(cfg)
    min_freq = get(cfg, "features.min_category_frequency", None)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric + engineered),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=min_freq,
                                  sparse_output=False), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("engineer", FunctionTransformer(add_engineered_features)),
        ("preprocess", preprocessor),
        ("model", build_classifier(cfg)),
    ])


def input_columns(cfg: dict) -> list[str]:
    """Raw columns the pipeline needs at predict time."""
    numeric, categorical, _ = feature_lists(cfg)
    return numeric + categorical
