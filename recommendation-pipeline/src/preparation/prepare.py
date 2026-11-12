"""Stage 3 - Data preparation (cleaning).

* repairs the warnings left by validation (missing / invalid age, gender, title, genres)
* standardises formats (casing, whitespace, genre tokens, UTC datetimes)
* removes duplicate (user, item) pairs, keeping the most recent interaction
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common import get_logger, load_config, load_table, resolve, save_table

log = None


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    users = users.copy()
    age = pd.to_numeric(users["age"], errors="coerce")
    valid_age = age.where((age >= 5) & (age <= 100))
    median_age = int(valid_age.median())
    users["age"] = valid_age.fillna(median_age).astype(int)
    users["gender"] = users["gender"].fillna("U").astype(str).str.strip().str.upper()
    users["occupation"] = users["occupation"].fillna("other").astype(str).str.strip().str.lower()
    users["zip_code"] = users["zip_code"].astype(str).str.strip()
    return users.drop_duplicates(subset="user_id").reset_index(drop=True)


def clean_items(items: pd.DataFrame) -> pd.DataFrame:
    items = items.copy()
    items["title"] = items["title"].fillna("Unknown title").astype(str).str.strip()
    items["genres"] = items["genres"].fillna("unknown").astype(str).str.strip()
    items["release_date"] = pd.to_datetime(items["release_date"], errors="coerce")
    items["release_year"] = items["release_date"].dt.year.astype("Int64")
    items = items.drop(columns="release_date")
    return items.drop_duplicates(subset="item_id").reset_index(drop=True)


def clean_interactions(inter: pd.DataFrame) -> pd.DataFrame:
    inter = inter.copy()
    inter["timestamp"] = pd.to_datetime(inter["timestamp"], unit="s")
    inter = inter.sort_values("timestamp")
    # one rating per (user, item): keep the latest
    inter = inter.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    return inter.sort_values("timestamp").reset_index(drop=True)


def run(cfg: dict | None = None) -> dict:
    global log
    cfg = cfg or load_config()
    log = get_logger("preparation", cfg)
    src = resolve(cfg, "processed") / "validated"
    out = resolve(cfg, "processed")

    users = clean_users(load_table(src, "users"))
    items = clean_items(load_table(src, "items"))
    raw_inter = load_table(src, "interactions")
    inter = clean_interactions(raw_inter)
    # defensive: everything must still resolve after cleaning
    inter = inter[inter["user_id"].isin(users["user_id"]) & inter["item_id"].isin(items["item_id"])]

    save_table(users, out, "users_clean")
    save_table(items, out, "items_clean")
    save_table(inter, out, "interactions_clean")

    summary = {
        "users": len(users), "items": len(items), "interactions": len(inter),
        "dedup_user_item_pairs_removed": int(len(raw_inter) - len(inter)),
        "min_timestamp": str(inter["timestamp"].min()), "max_timestamp": str(inter["timestamp"].max()),
        "rating_mean": round(float(inter["rating"].mean()), 4),
        "sparsity_pct": round(100 * (1 - len(inter) / (len(users) * len(items))), 3),
    }
    with open(resolve(cfg, "reports") / "preparation_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Preparation done: %s", summary)
    return summary


if __name__ == "__main__":
    run()
