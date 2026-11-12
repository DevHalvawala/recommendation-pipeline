"""Stage 5 - Feature management (custom lightweight feature store).

Offline store  : data/features/{user,item,interaction}_features (Parquet/CSV)
Registry       : FEATURE_REGISTRY below -> reports/feature_catalog.{md,csv}
Serving API    : FeatureStore.get_user_features / get_item_features /
                 get_user_genre_profile / get_item_genre_matrix
                 The SAME class is used by training, evaluation and prediction,
                 so a feature has exactly one definition.

Leakage control: every feature is computed from the TRAIN split only, and
"days since" features are measured from the last TRAIN timestamp (not today).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common import GENRES, get_logger, load_config, load_table, resolve, save_table

log = None
G = [g for g in GENRES]                       # genre vocabulary
GCOLS = [f"genre_{g}" for g in G]
PCOLS = [f"profile_{g}" for g in G]

FEATURE_REGISTRY = [
    # name, entity, description, formula, source, dtype
    ("user_avg_rating", "user", "Mean rating given by the user", "mean(rating) over user's train rows", "train", "float"),
    ("user_rating_std", "user", "Spread of the user's ratings (0 if one rating)", "std(rating), 0-filled", "train", "float"),
    ("user_activity_count", "user", "Number of interactions by the user", "count(rows) per user_id", "train", "int"),
    ("user_days_since_last_interaction", "user", "Days from user's last train interaction to the reference time",
     "(max(train.timestamp) - user's last timestamp) in days", "train", "float"),
    ("user_preferred_genre", "user", "Genre the user rated highest-weighted most often",
     "argmax over genres of sum(rating of items in genre)", "train + items", "str"),
    ("age / gender / occupation", "user", "Cleaned demographic attributes", "copied from users_clean", "users", "int/str/str"),
    ("profile_<Genre> (19 cols)", "user", "Genre taste profile used by the content part of the hybrid model",
     "L1-normalised mean genre vector of items rated >= threshold", "train + items", "float"),
    ("item_popularity", "item", "Number of train ratings for the item", "count(rows) per item_id", "train", "int"),
    ("item_avg_rating", "item", "Mean train rating of the item", "mean(rating) per item_id", "train", "float"),
    ("item_popularity_rank", "item", "Rank by popularity (1 = most popular)", "rank(item_popularity, desc)", "train", "int"),
    ("item_release_year", "item", "Release year of the item (may be null)", "year(release_date)", "items", "Int"),
    ("item_genres", "item", "Pipe-separated genre list", "copied from items_clean", "items", "str"),
    ("genre_<Genre> (19 cols)", "item", "Multi-hot genre vector", "1 if genre in item_genres else 0", "items", "int"),
    ("rating", "interaction", "Explicit rating", "copied", "train", "float"),
    ("interaction_recency_days", "interaction", "Age of the interaction at reference time",
     "(max(train.timestamp) - timestamp) in days", "train", "float"),
    ("rating_minus_user_avg", "interaction", "How much the rating deviates from the user's average",
     "rating - user_avg_rating", "train", "float"),
]


def build_features(train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame,
                   relevant_threshold: float = 4.0):
    ref = train["timestamp"].max()

    # ---------- item features
    g = train.groupby("item_id")
    pop = pd.DataFrame({"item_popularity": g.size(), "item_avg_rating": g["rating"].mean()})
    it = items[["item_id", "genres", "release_year"]].rename(
        columns={"genres": "item_genres", "release_year": "item_release_year"})
    it = it.merge(pop, left_on="item_id", right_index=True, how="left")
    it["item_popularity"] = it["item_popularity"].fillna(0).astype(int)
    it["item_avg_rating"] = it["item_avg_rating"].fillna(train["rating"].mean())
    it["item_popularity_rank"] = it["item_popularity"].rank(ascending=False, method="first").astype(int)
    for gname, col in zip(G, GCOLS):
        it[col] = it["item_genres"].map(lambda s, gn=gname: int(gn in s.split("|")))

    # ---------- user features
    gu = train.groupby("user_id")
    uf = pd.DataFrame({
        "user_avg_rating": gu["rating"].mean(),
        "user_rating_std": gu["rating"].std().fillna(0.0),
        "user_activity_count": gu.size(),
        "user_days_since_last_interaction": (ref - gu["timestamp"].max()).dt.total_seconds() / 86400,
    }).reset_index()

    gm = it.set_index("item_id")[GCOLS]
    tr = train[["user_id", "item_id", "rating"]].copy()
    genre_w = gm.reindex(tr["item_id"]).to_numpy() * tr["rating"].to_numpy()[:, None]
    weighted = pd.DataFrame(genre_w, columns=G).assign(user_id=tr["user_id"].to_numpy()).groupby("user_id").sum()
    pref = weighted.idxmax(axis=1).rename("user_preferred_genre")
    uf = uf.merge(pref, left_on="user_id", right_index=True, how="left")

    liked = tr[tr["rating"] >= relevant_threshold]
    lw = gm.reindex(liked["item_id"]).to_numpy()
    prof = pd.DataFrame(lw, columns=PCOLS).assign(user_id=liked["user_id"].to_numpy()).groupby("user_id").sum()
    prof = prof.div(prof.sum(axis=1).replace(0, 1), axis=0)
    uf = uf.merge(prof, left_on="user_id", right_index=True, how="left")
    uf[PCOLS] = uf[PCOLS].fillna(0.0)
    uf = uf.merge(users[["user_id", "age", "gender", "occupation"]], on="user_id", how="left")

    # ---------- interaction features
    inter = train[["user_id", "item_id", "rating", "timestamp"]].copy()
    inter["interaction_recency_days"] = (ref - inter["timestamp"]).dt.total_seconds() / 86400
    inter = inter.merge(uf[["user_id", "user_avg_rating"]], on="user_id", how="left")
    inter["rating_minus_user_avg"] = inter["rating"] - inter["user_avg_rating"]
    inter = inter.drop(columns="user_avg_rating")
    return uf, it, inter, ref


class FeatureStore:
    """Read API over the offline store - shared by training and prediction."""

    def __init__(self, features_dir):
        self.dir = features_dir
        self.users = load_table(features_dir, "user_features").set_index("user_id")
        self.items = load_table(features_dir, "item_features").set_index("item_id")
        self.global_defaults = {
            "user_avg_rating": float(self.users["user_avg_rating"].mean()),
            "user_activity_count": 0,
        }

    def get_user_features(self, user_ids) -> pd.DataFrame:
        """Unknown (cold-start) users get global defaults instead of errors."""
        df = self.users.reindex(list(user_ids))
        df["user_avg_rating"] = df["user_avg_rating"].fillna(self.global_defaults["user_avg_rating"])
        df["user_activity_count"] = df["user_activity_count"].fillna(0).astype(int)
        df["user_rating_std"] = df["user_rating_std"].fillna(0.0)
        df["user_preferred_genre"] = df["user_preferred_genre"].fillna("unknown")
        df[PCOLS] = df[PCOLS].fillna(0.0)
        return df

    def get_item_features(self, item_ids) -> pd.DataFrame:
        return self.items.reindex(list(item_ids))

    def get_item_genre_matrix(self, item_ids) -> np.ndarray:
        return self.items.reindex(list(item_ids))[GCOLS].fillna(0).to_numpy(dtype=float)

    def get_user_genre_profile(self, user_ids) -> np.ndarray:
        return self.get_user_features(user_ids)[PCOLS].to_numpy(dtype=float)


def write_catalog(reports_dir) -> None:
    cat = pd.DataFrame(FEATURE_REGISTRY,
                       columns=["feature", "entity", "description", "formula", "source", "dtype"])
    cat.to_csv(reports_dir / "feature_catalog.csv", index=False)
    lines = ["# Feature Catalog", "",
             "All features are computed from the **train split only**; time-based features are "
             "measured from the last train timestamp. Served through `FeatureStore`.", "",
             "| Feature | Entity | Description | Formula | Source | Type |", "|---|---|---|---|---|---|"]
    for r in FEATURE_REGISTRY:
        lines.append("| " + " | ".join(f"`{r[0]}`" if i == 0 else str(x) for i, x in enumerate(r)) + " |")
    (reports_dir / "feature_catalog.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: dict | None = None) -> dict:
    global log
    cfg = cfg or load_config()
    log = get_logger("features", cfg)
    proc, feats = resolve(cfg, "processed"), resolve(cfg, "features")
    train = load_table(proc, "train")
    users, items = load_table(proc, "users_clean"), load_table(proc, "items_clean")

    uf, it, inter, ref = build_features(train, users, items,
                                        cfg["preparation"]["relevant_rating_threshold"])
    save_table(uf, feats, "user_features")
    save_table(it, feats, "item_features")
    save_table(inter, feats, "interaction_features")
    write_catalog(resolve(cfg, "reports"))
    summary = {"reference_time": str(ref), "user_features_rows": len(uf), "user_feature_cols": uf.shape[1],
               "item_features_rows": len(it), "item_feature_cols": it.shape[1],
               "interaction_features_rows": len(inter)}
    with open(resolve(cfg, "reports") / "feature_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Feature build done: %s", summary)
    return summary


if __name__ == "__main__":
    run()
