import numpy as np
import pandas as pd
import pytest

from src.features.feature_store import build_features
from src.ingestion.synthetic import generate
from src.preparation import prepare
from src.transformation import transform


@pytest.fixture(scope="module")
def data():
    users, items, inter = generate(60, 50, 1500, seed=1)
    inter["timestamp"] = pd.to_datetime(inter["timestamp"], unit="s")
    items = prepare.clean_items(items)
    users = prepare.clean_users(users)
    return users, items, inter.sort_values("timestamp").reset_index(drop=True)


def test_time_split_has_no_leakage(data):
    _, _, inter = data
    train, test, cutoff = transform.time_split(inter, 0.8)
    assert train["timestamp"].max() <= cutoff < test["timestamp"].min()
    assert len(train) + len(test) == len(inter)
    assert 0.75 < len(train) / len(inter) < 0.85


def test_matrix_built_from_train_only(data):
    _, _, inter = data
    train, test, _ = transform.time_split(inter, 0.8)
    mat, u_map, i_map = transform.build_matrix(train)
    assert mat.nnz == len(train)
    assert set(i_map["item_id"]) <= set(train["item_id"])


def test_features_use_train_only_and_train_reference_time(data):
    users, items, inter = data
    train, test, _ = transform.time_split(inter, 0.8)
    uf, it, inf, ref = build_features(train, users, items)
    assert ref == train["timestamp"].max()
    assert uf["user_days_since_last_interaction"].min() >= 0       # never negative -> no future data
    assert it["item_popularity"].sum() == len(train)               # test rows never counted
    u = uf.iloc[0]
    expected = train[train["user_id"] == u["user_id"]]["rating"].mean()
    assert u["user_avg_rating"] == pytest.approx(expected)


def test_clean_users_repairs_age_and_gender():
    users = pd.DataFrame({"user_id": [1, 2, 3], "age": [30, np.nan, -5], "gender": ["m", None, "F"],
                          "occupation": ["A", None, "b"], "zip_code": ["1", "2", "3"]})
    out = prepare.clean_users(users)
    assert out["age"].between(5, 100).all()
    assert set(out["gender"]) == {"M", "U", "F"}
