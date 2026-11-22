import joblib
import pandas as pd
import pytest

from src.config import load_config
from src.features import build_pipeline, input_columns
from src.preprocess import clean_raw
from src.train import split_data

LEAKY = ["reservation_status", "reservation_status_date", "assigned_room_type"]


def test_config_overrides_are_typed():
    cfg = load_config(overrides=["split.test_size=0.3", "model.type=random_forest"])
    assert cfg["split"]["test_size"] == 0.3
    assert cfg["model"]["type"] == "random_forest"


def test_clean_raw_removes_leakage_and_nans(raw_df):
    cfg = load_config()
    clean = clean_raw(raw_df, cfg)
    assert not any(c in clean.columns for c in LEAKY)
    assert clean[input_columns(cfg)].isna().sum().sum() == 0
    assert set(clean["is_canceled"].unique()) <= {0, 1}
    assert "Undefined" not in set(clean["meal"])


def test_split_sizes(raw_df):
    cfg = load_config()
    clean = clean_raw(raw_df, cfg)
    X_tr, X_va, X_te, *_ = split_data(clean, cfg)
    total = len(X_tr) + len(X_va) + len(X_te)
    assert total == len(clean)
    assert abs(len(X_te) / total - cfg["split"]["test_size"]) < 0.02
    assert abs(len(X_va) / total - cfg["split"]["val_size"]) < 0.02


@pytest.mark.parametrize("model_type", ["hist_gradient_boosting", "random_forest", "logistic_regression"])
def test_every_model_type_trains(raw_df, model_type):
    cfg = load_config(overrides=[f"model.type={model_type}",
                                 "model.params.random_forest.n_estimators=20"])
    clean = clean_raw(raw_df, cfg)
    X_tr, _, X_te, y_tr, _, _ = split_data(clean, cfg)
    pipe = build_pipeline(cfg).fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_include_country_flag(raw_df):
    cfg = load_config(overrides=["features.include_country=true"])
    assert "country" in input_columns(cfg)


def test_end_to_end_run_saves_artifacts(trained):
    model_dir = trained["model_dir"]
    assert (model_dir / "model.joblib").exists()
    assert (model_dir / "metadata.json").exists()
    meta = trained["metadata"]
    assert 0.0 <= meta["metrics"]["test/roc_auc"] <= 1.0
    # saved model accepts raw fields and handles unseen categories
    model = joblib.load(model_dir / "model.joblib")
    row = pd.DataFrame([{**{c: 0 for c in meta["features"]["numeric"]},
                         "hotel": "City Hotel", "arrival_date_month": "May", "meal": "BB",
                         "market_segment": "NEVER_SEEN", "distribution_channel": "TA/TO",
                         "reserved_room_type": "Z", "deposit_type": "No Deposit",
                         "customer_type": "Transient", "adults": 2}])
    p = model.predict_proba(row)[:, 1][0]
    assert 0.0 <= p <= 1.0
