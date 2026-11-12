"""Ranking and rating metrics (pure numpy, unit-tested on toy examples)."""
from __future__ import annotations

import math

import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    return len(set(recommended[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Binary-relevance NDCG@k."""
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def catalog_coverage(all_recommendations: list[list], n_catalog_items: int) -> float:
    """Share of the catalogue that appears in at least one user's top-k list."""
    return len({i for recs in all_recommendations for i in recs}) / n_catalog_items


def intra_list_diversity(recommended: list, genre_lookup: dict) -> float:
    """Mean pairwise (1 - cosine similarity) of item genre vectors inside one list."""
    vecs = [genre_lookup[i] for i in recommended if i in genre_lookup]
    if len(vecs) < 2:
        return 0.0
    V = np.array(vecs, dtype=float)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.where(norms == 0, 1, norms)
    sim = V @ V.T
    n = len(V)
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sim[iu]))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))
