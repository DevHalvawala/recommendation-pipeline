"""Synthetic MovieLens-like data generator (offline fallback).

Documented generative process (so the dataset is clearly explainable):
  * each item has 1-3 genres and a latent "quality" ~ N(0, 0.5)
  * each user has a preference weight per genre ~ Dirichlet
  * the chance a user interacts with an item grows with item popularity
    (Zipf-like) and with the user's genre affinity
  * rating = clip(round(3.3 + 2.0*(affinity - mean) + quality + noise), 1, 5)
  * timestamps are uniform over a two-year window
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import GENRES

_OCCUPATIONS = ["student", "engineer", "teacher", "artist", "writer", "doctor",
                "lawyer", "retired", "other", "scientist"]


def generate(n_users: int, n_items: int, n_interactions: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    real_genres = GENRES[1:]  # skip "unknown"

    # ---- items
    item_genre = np.zeros((n_items, len(real_genres)))
    for i in range(n_items):
        k = rng.integers(1, 4)
        item_genre[i, rng.choice(len(real_genres), size=k, replace=False)] = 1
    quality = rng.normal(0, 0.5, n_items)
    items = pd.DataFrame({
        "item_id": np.arange(1, n_items + 1),
        "title": [f"Movie {i}" for i in range(1, n_items + 1)],
        "release_date": pd.to_datetime("1990-01-01")
        + pd.to_timedelta(rng.integers(0, 365 * 30, n_items), unit="D"),
        "genres": ["|".join(g for g, f in zip(real_genres, row) if f) for row in item_genre],
    })
    items["release_date"] = items["release_date"].dt.strftime("%Y-%m-%d")

    # ---- users
    pref = rng.dirichlet(np.ones(len(real_genres)) * 0.4, n_users)
    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "age": rng.integers(15, 70, n_users),
        "gender": rng.choice(["M", "F"], n_users),
        "occupation": rng.choice(_OCCUPATIONS, n_users),
        "zip_code": rng.integers(10000, 99999, n_users).astype(str),
    })

    # ---- interactions
    affinity = pref @ item_genre.T                       # users x items
    affinity = affinity / affinity.max()
    pop = 1.0 / np.arange(1, n_items + 1) ** 0.6
    rng.shuffle(pop)
    prob = (0.3 * pop / pop.sum() + 0.7 * affinity / affinity.sum(axis=1, keepdims=True))
    prob = prob / prob.sum(axis=1, keepdims=True)

    per_user = rng.multinomial(n_interactions, np.ones(n_users) / n_users)
    rows = []
    start = pd.Timestamp("2022-01-01").timestamp()
    span = 365 * 2 * 86400
    for u in range(n_users):
        n = int(min(per_user[u], n_items))
        if n == 0:
            continue
        chosen = rng.choice(n_items, size=n, replace=False, p=prob[u])
        for it in chosen:
            score = 3.3 + 2.0 * (affinity[u, it] - affinity.mean()) * 3 + quality[it] + rng.normal(0, 0.6)
            rows.append((u + 1, it + 1, int(np.clip(round(score), 1, 5)),
                         int(start + rng.integers(0, span))))
    inter = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    return users, items, inter
