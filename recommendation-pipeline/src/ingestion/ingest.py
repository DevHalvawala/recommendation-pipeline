"""Stage 1 - Data ingestion.

Loads users / items / interactions from MovieLens 100K (public dataset) or a
documented synthetic generator, writes them as raw CSVs, and (optionally)
injects known defects into the raw files so the validation stage has real
problems to detect. Pristine copies are kept in data/raw/original/.
"""
from __future__ import annotations

import io
import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from src.common import GENRES, get_logger, load_config, resolve, set_seed
from src.ingestion.synthetic import generate

log = None


def _download_movielens(url: str):
    log.info("Downloading MovieLens 100K from %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:
        blob = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(blob))

    users = pd.read_csv(
        zf.open("ml-100k/u.user"), sep="|", header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"], dtype={"zip_code": str},
    )
    item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    raw_items = pd.read_csv(zf.open("ml-100k/u.item"), sep="|", header=None,
                            names=item_cols, encoding="latin-1")
    flags = raw_items[GENRES].to_numpy()
    genres = ["|".join(g for g, f in zip(GENRES, row) if f) or "unknown" for row in flags]
    items = pd.DataFrame({
        "item_id": raw_items["item_id"],
        "title": raw_items["title"],
        "release_date": pd.to_datetime(raw_items["release_date"], format="%d-%b-%Y",
                                       errors="coerce").dt.strftime("%Y-%m-%d"),
        "genres": genres,
    })
    inter = pd.read_csv(zf.open("ml-100k/u.data"), sep="\t", header=None,
                        names=["user_id", "item_id", "rating", "timestamp"])
    return users, items, inter


def inject_errors(users, items, inter, seed: int):
    """Corrupt copies of the tables. Returns (users, items, inter, summary)."""
    rng = np.random.default_rng(seed)
    inter = inter.copy()
    users, items = users.copy(), items.copy()
    # object dtype so we can place strings like "abc" in numeric columns
    inter["rating"] = inter["rating"].astype(object)
    inter["timestamp"] = inter["timestamp"].astype(object)
    summary = {}

    def pick(n):
        return rng.choice(inter.index.to_numpy(), size=n, replace=False)

    used: set = set()

    def pick_new(n):
        idx = [i for i in pick(n * 3) if i not in used][:n]
        used.update(idx)
        return idx

    idx = pick_new(30); inter.loc[idx, "rating"] = np.nan; summary["null_rating"] = len(idx)
    idx = pick_new(25); inter.loc[idx, "rating"] = 0; summary["rating_zero"] = len(idx)
    idx = pick_new(25); inter.loc[idx, "rating"] = 7; summary["rating_above_max"] = len(idx)
    idx = pick_new(15); inter.loc[idx, "rating"] = "abc"; summary["rating_non_numeric"] = len(idx)
    idx = pick_new(20)
    inter.loc[idx, "timestamp"] = int(pd.Timestamp("2035-01-01").timestamp())
    summary["timestamp_future"] = len(idx)
    idx = pick_new(20); inter.loc[idx, "timestamp"] = 0; summary["timestamp_1970"] = len(idx)
    idx = pick_new(30); inter.loc[idx, "user_id"] = 99999; summary["orphan_user_id"] = len(idx)
    idx = pick_new(30); inter.loc[idx, "item_id"] = 88888; summary["orphan_item_id"] = len(idx)
    dups = inter.sample(40, random_state=seed)
    inter = pd.concat([inter, dups], ignore_index=True)
    summary["duplicate_interaction_rows"] = len(dups)

    # users / items
    ui = rng.choice(users.index.to_numpy(), size=8, replace=False)
    users.loc[ui[:4], "age"] = np.nan
    users.loc[ui[4:], "age"] = -5
    users = pd.concat([users, users.sample(5, random_state=seed)], ignore_index=True)
    summary.update({"user_null_age": 4, "user_negative_age": 4, "duplicate_user_rows": 5})
    ii = rng.choice(items.index.to_numpy(), size=5, replace=False)
    items.loc[ii, "title"] = np.nan
    items = pd.concat([items, items.sample(3, random_state=seed)], ignore_index=True)
    summary.update({"item_null_title": 5, "duplicate_item_rows": 3})
    return users, items, inter, summary


def run(cfg: dict | None = None) -> dict:
    global log
    cfg = cfg or load_config()
    log = get_logger("ingestion", cfg)
    set_seed(cfg["project"]["seed"])
    raw = resolve(cfg, "raw")
    orig = raw / "original"
    orig.mkdir(parents=True, exist_ok=True)

    source = cfg["data"]["source"]
    if source == "movielens":
        try:
            users, items, inter = _download_movielens(cfg["data"]["movielens_url"])
        except Exception as exc:  # network problems -> documented fallback
            log.warning("MovieLens download failed (%s). Falling back to synthetic data.", exc)
            source = "synthetic"
    if source == "synthetic":
        s = cfg["data"]["synthetic"]
        users, items, inter = generate(s["n_users"], s["n_items"], s["n_interactions"],
                                       cfg["project"]["seed"])

    users.to_csv(orig / "users.csv", index=False)
    items.to_csv(orig / "items.csv", index=False)
    inter.to_csv(orig / "interactions.csv", index=False)

    injected = {}
    if cfg["data"]["inject_errors"]:
        users, items, inter, injected = inject_errors(users, items, inter, cfg["project"]["seed"])
        log.info("Injected known defects into raw copies: %s", injected)

    users.to_csv(raw / "users.csv", index=False)
    items.to_csv(raw / "items.csv", index=False)
    inter.to_csv(raw / "interactions.csv", index=False)

    summary = {"source": source, "n_users": len(users), "n_items": len(items),
               "n_interactions": len(inter), "injected_defects": injected}
    with open(resolve(cfg, "reports") / "ingestion_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Ingestion done: %d users, %d items, %d interactions (source=%s)",
             len(users), len(items), len(inter), source)
    return summary


if __name__ == "__main__":
    run()
