"""Load config.yaml, apply env-var and CLI overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# env var -> dotted config key
ENV_OVERRIDES = {
    "DATA_SOURCE": "data.source",
    "GCS_BUCKET": "data.gcs_bucket",
    "GCS_BLOB": "data.gcs_blob",
    "MODEL_TYPE": "model.type",
    "TRACKING_BACKEND": "tracking.backend",
}


def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    node: Any = cfg
    for k in dotted_key.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def load_config(path: str | os.PathLike | None = None,
                overrides: Iterable[str] | None = None) -> dict:
    """Load YAML config.

    Precedence (lowest -> highest): config file, env vars, ``--set key=value``.
    Override values are parsed as YAML, so ``--set split.test_size=0.2`` gives a float.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg = copy.deepcopy(cfg)

    for env_name, key in ENV_OVERRIDES.items():
        if os.environ.get(env_name):
            _set_dotted(cfg, key, os.environ[env_name])

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))

    return cfg


def feature_lists(cfg: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (numeric, categorical, engineered) feature names after config flags."""
    numeric = list(get(cfg, "features.numeric", []))
    categorical = list(get(cfg, "features.categorical", []))
    engineered = list(get(cfg, "features.engineered", []))
    if get(cfg, "features.include_country", False) and "country" not in categorical:
        categorical.append("country")
    return numeric, categorical, engineered
