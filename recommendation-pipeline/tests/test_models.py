import numpy as np
import scipy.sparse as sp

from src.training.models import PopularityRecommender, SVDRecommender


def _toy():
    rng = np.random.default_rng(0)
    dense = (rng.random((30, 20)) < 0.3) * rng.integers(1, 6, (30, 20))
    dense[:, 0] = 5                                  # item 0 rated by everybody
    return sp.csr_matrix(dense.astype(float)), np.arange(30), np.arange(100, 120)


def test_popularity_recommends_most_rated_and_skips_seen():
    m, u, i = _toy()
    model = PopularityRecommender(min_ratings=1).fit(m, u, i)
    recs = model.recommend([0], k=5)[0]
    seen = set(i[m[0].indices])
    assert not (set(recs) & seen)                    # never recommends already-seen items
    assert len(recs) == 5


def test_svd_is_deterministic_and_clips_ratings():
    m, u, i = _toy()
    a = SVDRecommender(n_factors=5, rating_factors=3, seed=1).fit(m, u, i)
    b = SVDRecommender(n_factors=5, rating_factors=3, seed=1).fit(m, u, i)
    assert a.recommend([0, 1], 5) == b.recommend([0, 1], 5)
    pred = a.predict_rating([0, 1, 999], [100, 101, 102])        # 999 = unseen user
    assert ((pred >= 1) & (pred <= 5)).all()


def test_cold_start_user_gets_recommendations():
    m, u, i = _toy()
    model = SVDRecommender(n_factors=5, rating_factors=3).fit(m, u, i)
    assert len(model.recommend([424242], 5)[424242]) == 5
