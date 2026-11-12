"""Shared helpers: config loading, logging, storage I/O, seeding.

Storage uses Parquet when pyarrow is installed (Colab has it) and falls back
to CSV otherwise, so the pipeline never breaks on a missing optional engine.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

try:  # pragma: no cover - depends on environment
    import pyarrow  # noqa: F401

    _PARQUET = True
except Exception:  # pragma: no cover
    _PARQUET = False

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror",
    "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western",
]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(cfg: dict, key: str) -> Path:
    """Absolute path of a configured folder (created if missing)."""
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str, cfg: dict | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    reports = (ROOT / cfg["paths"]["reports"]) if cfg else ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(reports / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_table(df: pd.DataFrame, folder: Path, name: str) -> Path:
    """Save as Parquet if possible, else CSV. `name` has no extension."""
    folder.mkdir(parents=True, exist_ok=True)
    if _PARQUET:
        path = folder / f"{name}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = folder / f"{name}.csv"
        df.to_csv(path, index=False)
    return path


def load_table(folder: Path, name: str) -> pd.DataFrame:
    """Load a table saved by save_table (datetime columns restored for CSV)."""
    pq, csv = folder / f"{name}.parquet", folder / f"{name}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        df = pd.read_csv(csv)
        for col in ("timestamp", "last_interaction", "first_interaction"):
            # only text columns are datetimes; numeric ones are epoch seconds
            # (parsed explicitly by the stage that owns them)
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
        return df
    raise FileNotFoundError(f"{name} not found in {folder}")
