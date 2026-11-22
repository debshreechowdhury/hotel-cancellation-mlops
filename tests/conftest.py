import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_sample_data import make  # noqa: E402
from src.config import load_config  # noqa: E402
from src.preprocess import clean_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return make(rows=1500, seed=1)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, raw_df):
    """Train a small model on synthetic data into a temp dir (no GCS, no W&B)."""
    from src.train import run

    tmp = tmp_path_factory.mktemp("run")
    clean_path = tmp / "clean.csv"
    cfg = load_config(overrides=[
        "data.source=local", f"data.clean_path={clean_path}",
        "tracking.backend=none", f"output.model_dir={tmp / 'models'}",
        f"output.reports_dir={tmp / 'reports'}", "model.params.hist_gradient_boosting.max_iter=50",
        "evaluation.permutation_importance_samples=300",
        "evaluation.permutation_importance_repeats=1",
    ])
    clean_raw(raw_df, cfg).to_csv(clean_path, index=False)
    metadata = run(cfg)
    return {"cfg": cfg, "metadata": metadata, "model_dir": tmp / "models"}
