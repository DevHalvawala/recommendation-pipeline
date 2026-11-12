"""Recommender models.

All models share one interface so the evaluation stage is model-agnostic:
    fit(matrix, user_ids, item_ids, ...)   train on the sparse train matrix
    score_users(user_ids) -> (n_users, n_items) preference scores
    predict_rating(user_ids, item_ids)     explicit-rating prediction (RMSE/MAE)
    recommend(user_ids, k)                 top-k item ids, already-seen items removed

Models
  PopularityRecommender  baseline: most-rated items (ties -> higher mean rating)
  SVDRecommender         matrix factorisation (truncated SVD, scipy)
  HybridRecommender      SVD score blended with a genre content score built from
                         the FeatureStore (reuses the same features at prediction time)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


class BaseRecommender:
    name = "base"

    def _index(self, user_ids, item_ids):
        self.user_ids, self.item_ids = np.asarray(user_ids), np.asarray(item_ids)
        self.u_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.i_idx = {it: i for i, it in enumerate(self.item_ids)}

    # ---- shared logic -------------------------------------------------
    def recommend(self, user_ids, k: int = 10) -> dict:
        scores = self.score_users(user_ids).astype(float).copy()
        for row, u in enumerate(user_ids):
            if u in self.u_idx:  # remove items already seen in train
                seen = self.train_matrix.indices[
                    self.train_matrix.indptr[self.u_idx[u]]: self.train_matrix.indptr[self.u_idx[u] + 1]]
                scores[row, seen] = -np.inf
        top = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
        out = {}
        for row, u in enumerate(user_ids):
            order = top[row][np.argsort(-scores[row, top[row]])]
            out[u] = [int(self.item_ids[j]) for j in order]
        return out

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        raise NotImplementedError

    def score_users(self, user_ids) -> np.ndarray:
        raise NotImplementedError


class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def __init__(self, min_ratings: int = 5):
        self.min_ratings = min_ratings

    def fit(self, matrix: sp.csr_matrix, user_ids, item_ids, **_):
        self._index(user_ids, item_ids)
        self.train_matrix = matrix.tocsr()
        counts = np.asarray((matrix > 0).sum(axis=0)).ravel()
        sums = np.asarray(matrix.sum(axis=0)).ravel()
        self.global_mean = float(matrix.data.mean())
        m = 5.0                                           # shrinkage prior strength
        self.item_mean = (sums + m * self.global_mean) / (counts + m)
        eligible = counts >= self.min_ratings
        # score = popularity count, tiny tie-break by shrunk mean rating
        self.item_score = np.where(eligible, counts + self.item_mean / 10.0, -1e9)
        return self

    def score_users(self, user_ids) -> np.ndarray:
        return np.tile(self.item_score, (len(user_ids), 1))

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        cols = np.array([self.i_idx.get(i, -1) for i in item_ids])
        return np.where(cols >= 0, self.item_mean[np.clip(cols, 0, None)], self.global_mean)


class SVDRecommender(BaseRecommender):
    """Two factorisations on the same train matrix.

    * ranking model : PureSVD on the raw rating matrix (missing = 0), the standard
      strong choice for top-N ranking.
    * rating model  : global mean + regularised user/item biases + low-rank SVD of the
      residual matrix, clipped to [1, 5] - used for RMSE / MAE.
    """
    name = "svd_mf"

    def __init__(self, n_factors: int = 30, rating_factors: int = 10, bias_reg: float = 10.0,
                 seed: int = 42):
        self.n_factors, self.rating_factors = n_factors, rating_factors
        self.bias_reg, self.seed = bias_reg, seed

    def _svd(self, mat, n_factors=None):
        k = max(1, min(n_factors or self.n_factors, min(mat.shape) - 1))
        v0 = np.random.default_rng(self.seed).standard_normal(min(mat.shape))
        U, s, Vt = svds(mat.astype(float), k=k, v0=v0)
        return U, s, Vt

    def fit(self, matrix: sp.csr_matrix, user_ids, item_ids, **_):
        self._index(user_ids, item_ids)
        R = matrix.tocsr()
        self.train_matrix = R
        self.global_mean = float(R.data.mean())

        U, s, Vt = self._svd(R)                              # ranking factors
        self.rank_user = U * s
        self.rank_item = Vt

        coo = R.tocoo()
        n_u = np.bincount(coo.row, minlength=R.shape[0])
        n_i = np.bincount(coo.col, minlength=R.shape[1])
        bu, bi = np.zeros(R.shape[0]), np.zeros(R.shape[1])
        for _ in range(5):                                   # alternating regularised bias fit
            res = coo.data - self.global_mean - bu[coo.row]
            bi = np.bincount(coo.col, res, R.shape[1]) / (n_i + self.bias_reg)
            res = coo.data - self.global_mean - bi[coo.col]
            bu = np.bincount(coo.row, res, R.shape[0]) / (n_u + self.bias_reg)
        self.user_bias, self.item_bias = bu, bi
        resid = sp.csr_matrix((coo.data - self.global_mean - bu[coo.row] - bi[coo.col],
                               (coo.row, coo.col)), shape=R.shape)
        U2, s2, Vt2 = self._svd(resid, self.rating_factors)
        self.rate_user, self.rate_item = U2 * s2, Vt2
        return self

    def score_users(self, user_ids) -> np.ndarray:
        rows = [self.u_idx.get(u, -1) for u in user_ids]
        scores = np.zeros((len(rows), len(self.item_ids)))
        for r, ui in enumerate(rows):
            if ui >= 0:
                scores[r] = self.rank_user[ui] @ self.rank_item
            else:  # cold-start user -> fall back to item popularity
                scores[r] = np.asarray((self.train_matrix > 0).sum(axis=0)).ravel()
        return scores

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        out = np.full(len(user_ids), self.global_mean, dtype=float)
        for n, (u, it) in enumerate(zip(user_ids, item_ids)):
            ui, ii = self.u_idx.get(u), self.i_idx.get(it)
            if ui is not None:
                out[n] += self.user_bias[ui]
            if ii is not None:
                out[n] += self.item_bias[ii]
            if ui is not None and ii is not None:
                out[n] += self.rate_user[ui] @ self.rate_item[:, ii]
        return np.clip(out, 1.0, 5.0)


class HybridRecommender(BaseRecommender):
    """alpha * SVD score + (1 - alpha) * genre-content score (both min-max scaled per user)."""
    name = "hybrid_svd_content"

    def __init__(self, svd: SVDRecommender, store, alpha: float = 0.7):
        self.svd, self.store, self.alpha = svd, store, alpha

    def fit(self, matrix, user_ids, item_ids, **_):
        self._index(user_ids, item_ids)
        self.train_matrix = matrix.tocsr()
        self.global_mean = self.svd.global_mean
        self.item_genres = self.store.get_item_genre_matrix(self.item_ids)   # (items, genres)
        norms = np.linalg.norm(self.item_genres, axis=1, keepdims=True)
        self.item_genres_unit = self.item_genres / np.where(norms == 0, 1, norms)
        return self

    @staticmethod
    def _minmax(x):
        lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
        return (x - lo) / np.where(hi - lo == 0, 1, hi - lo)

    def score_users(self, user_ids) -> np.ndarray:
        cf = self._minmax(self.svd.score_users(user_ids))
        prof = self.store.get_user_genre_profile(user_ids)                 # same features as training
        pn = np.linalg.norm(prof, axis=1, keepdims=True)
        content = self._minmax((prof / np.where(pn == 0, 1, pn)) @ self.item_genres_unit.T)
        return self.alpha * cf + (1 - self.alpha) * content

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        return self.svd.predict_rating(user_ids, item_ids)
