"""Stage 4 - Data transformation.

* TIME-BASED split: the earliest `train_ratio` of the timeline is train, the rest is
  test (no random shuffling -> no look-ahead leakage)
* derived datasets built from TRAIN only: user activity summary, item popularity,
  sparse user-item interaction matrix (+ id <-> index mappings)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common import get_logger, load_config, load_table, resolve, save_table

log = None


def time_split(inter: pd.DataFrame, train_ratio: float):
    """Global time cutoff. Returns (train, test, cutoff_timestamp)."""
    cutoff = inter["timestamp"].quantile(train_ratio)
    train = inter[inter["timestamp"] <= cutoff].copy()
    test = inter[inter["timestamp"] > cutoff].copy()
    return train, test, cutoff


def user_activity(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("user_id")
    return pd.DataFrame({
        "n_interactions": g.size(),
        "avg_rating": g["rating"].mean(),
        "first_interaction": g["timestamp"].min(),
        "last_interaction": g["timestamp"].max(),
    }).reset_index()


def item_popularity(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("item_id")
    pop = pd.DataFrame({"n_ratings": g.size(), "avg_rating": g["rating"].mean()}).reset_index()
    pop["popularity_rank"] = pop["n_ratings"].rank(ascending=False, method="first").astype(int)
    return pop.sort_values("popularity_rank").reset_index(drop=True)


def build_matrix(train: pd.DataFrame):
    """Sparse user-item rating matrix + index mappings (train only)."""
    users = np.sort(train["user_id"].unique())
    items = np.sort(train["item_id"].unique())
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {it: i for i, it in enumerate(items)}
    rows = train["user_id"].map(u_idx).to_numpy()
    cols = train["item_id"].map(i_idx).to_numpy()
    mat = sp.csr_matrix((train["rating"].to_numpy(dtype=float), (rows, cols)),
                        shape=(len(users), len(items)))
    return mat, pd.DataFrame({"user_id": users, "user_idx": range(len(users))}), \
        pd.DataFrame({"item_id": items, "item_idx": range(len(items))})


def run(cfg: dict | None = None) -> dict:
    global log
    cfg = cfg or load_config()
    log = get_logger("transformation", cfg)
    proc = resolve(cfg, "processed")
    inter = load_table(proc, "interactions_clean")

    train, test, cutoff = time_split(inter, cfg["preparation"]["train_ratio"])
    assert train["timestamp"].max() <= test["timestamp"].min(), "time leakage in split"

    save_table(train, proc, "train")
    save_table(test, proc, "test")
    save_table(user_activity(train), proc, "user_activity_summary")
    save_table(item_popularity(train), proc, "item_popularity")
    mat, u_map, i_map = build_matrix(train)
    sp.save_npz(proc / "user_item_matrix.npz", mat)
    save_table(u_map, proc, "matrix_user_index")
    save_table(i_map, proc, "matrix_item_index")

    summary = {
        "cutoff_timestamp": str(cutoff),
        "train_rows": len(train), "test_rows": len(test),
        "train_users": int(train["user_id"].nunique()), "test_users": int(test["user_id"].nunique()),
        "test_users_cold_start": len(set(test["user_id"]) - set(train["user_id"])),
        "matrix_shape": list(mat.shape), "matrix_nnz": int(mat.nnz),
        "matrix_density_pct": round(100 * mat.nnz / (mat.shape[0] * mat.shape[1]), 4),
    }
    with open(resolve(cfg, "reports") / "split_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Transformation done: %s", summary)
    return summary


if __name__ == "__main__":
    run()
