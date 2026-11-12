import numpy as np
import pandas as pd

from src.common import load_config
from src.validation import validate


def _inter(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"]).astype(object)


def test_interaction_rules_flag_each_defect():
    cfg = load_config()
    ok_ts = 1_600_000_000
    df = _inter([
        (1, 1, 4, ok_ts),            # 0 valid
        (1, 2, 0, ok_ts),            # 1 rating too low
        (1, 3, 7, ok_ts),            # 2 rating too high
        (1, 4, "abc", ok_ts),        # 3 bad dtype
        (1, 5, np.nan, ok_ts),       # 4 missing rating
        (1, 6, 3, 0),                # 5 1970 timestamp
        (1, 7, 3, 4_000_000_000),    # 6 future timestamp
        (99, 1, 3, ok_ts),           # 7 orphan user
        (1, 99, 3, ok_ts),           # 8 orphan item
        (1, 1, 4, ok_ts),            # 9 duplicate of row 0
    ])
    reasons, counts = validate.validate_interactions(df, {1}, {1, 2, 3, 4, 5, 6, 7}, cfg)
    assert reasons[0] == ""
    assert "rating_out_of_range" in reasons[1] and "rating_out_of_range" in reasons[2]
    assert "invalid_rating_dtype" in reasons[3]
    assert "missing_rating" in reasons[4]
    assert "timestamp_before_min" in reasons[5]
    assert "timestamp_in_future" in reasons[6]
    assert "orphan_user_id" in reasons[7]
    assert "orphan_item_id" in reasons[8]
    assert "duplicate_row" in reasons[9]
    assert counts["duplicate_row"] == 1


def test_user_missing_age_is_warning_not_fatal():
    users = pd.DataFrame({"user_id": [1, 2, 2], "age": [30, np.nan, 40],
                          "gender": ["M", "F", "F"], "occupation": ["a", "b", "b"]})
    fatal, warn, counts = validate.validate_users(users)
    assert fatal[1] == "" and "missing_age" in warn[1]
    assert "duplicate_user_id" in fatal[2]
