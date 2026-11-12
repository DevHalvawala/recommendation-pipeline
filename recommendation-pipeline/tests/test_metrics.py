import math

import pytest

from src.evaluation import metrics as M


def test_precision_recall_toy():
    rec, rel = [1, 2, 3, 4, 5], {2, 5, 9}
    assert M.precision_at_k(rec, rel, 5) == pytest.approx(2 / 5)
    assert M.recall_at_k(rec, rel, 5) == pytest.approx(2 / 3)
    assert M.recall_at_k(rec, set(), 5) == 0.0


def test_ndcg_perfect_and_worst():
    assert M.ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    assert M.ndcg_at_k([7, 8, 9], {1, 2, 3}, 3) == 0.0


def test_ndcg_known_value():
    # relevant item at rank 2 only, one relevant item in total
    assert M.ndcg_at_k([9, 1, 8], {1}, 3) == pytest.approx(1 / math.log2(3))


def test_coverage():
    assert M.catalog_coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)


def test_diversity_identical_vs_orthogonal():
    same = {1: [1, 0], 2: [1, 0]}
    orth = {1: [1, 0], 2: [0, 1]}
    assert M.intra_list_diversity([1, 2], same) == pytest.approx(0.0)
    assert M.intra_list_diversity([1, 2], orth) == pytest.approx(1.0)


def test_rmse_mae():
    assert M.rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(math.sqrt(4 / 3))
    assert M.mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
