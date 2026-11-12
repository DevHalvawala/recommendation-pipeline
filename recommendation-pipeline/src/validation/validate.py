"""Stage 2 - Data validation.

Rule-based checks written in plain pandas (no heavy dependency). Every rule
produces a named failure; bad rows are moved to data/errors/ with a
`failure_reason` column and the good rows are written to data/processed/validated/.

Checks: missing values, duplicates, data types, invalid ratings,
invalid timestamps, referential integrity (user_id / item_id must exist).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common import get_logger, load_config, resolve, save_table

log = None


def _flag(reasons: pd.Series, mask: pd.Series, label: str) -> None:
    """Append `label` to rows where mask is True."""
    mask = mask.fillna(False)
    reasons[mask] = reasons[mask].map(lambda s: f"{s};{label}" if s else label)


FATAL_USER = {"missing_user_id", "invalid_user_id_dtype", "duplicate_user_id"}
FATAL_ITEM = {"missing_item_id", "invalid_item_id_dtype", "duplicate_item_id"}


def validate_users(users: pd.DataFrame):
    """Returns (fatal_reasons, warning_reasons, counts).

    Fatal -> row rejected. Warning (repairable: missing/odd age or gender) ->
    row kept, logged, and repaired in the preparation stage.
    """
    fatal = pd.Series("", index=users.index)
    warn = pd.Series("", index=users.index)
    counts = {}
    uid = pd.to_numeric(users["user_id"], errors="coerce")
    age = pd.to_numeric(users["age"], errors="coerce")

    checks = {
        "missing_user_id": uid.isna(),
        "missing_age": users["age"].isna(),
        "invalid_age_dtype": users["age"].notna() & age.isna(),
        "age_out_of_range": age.notna() & ((age < 5) | (age > 100)),
        "missing_gender": users["gender"].isna(),
        "duplicate_user_id": users.duplicated(subset="user_id", keep="first"),
    }
    for label, mask in checks.items():
        counts[label] = int(mask.sum())
        _flag(fatal if label in FATAL_USER else warn, mask, label)
    return fatal, warn, counts


def validate_items(items: pd.DataFrame):
    fatal = pd.Series("", index=items.index)
    warn = pd.Series("", index=items.index)
    counts = {}
    checks = {
        "missing_item_id": pd.to_numeric(items["item_id"], errors="coerce").isna(),
        "missing_title": items["title"].isna(),
        "missing_genres": items["genres"].isna(),
        "duplicate_item_id": items.duplicated(subset="item_id", keep="first"),
    }
    for label, mask in checks.items():
        counts[label] = int(mask.sum())
        _flag(fatal if label in FATAL_ITEM else warn, mask, label)
    return fatal, warn, counts


def validate_interactions(inter: pd.DataFrame, valid_users: set, valid_items: set, cfg: dict):
    v = cfg["validation"]
    reasons = pd.Series("", index=inter.index)
    counts = {}

    rating = pd.to_numeric(inter["rating"], errors="coerce")
    ts = pd.to_numeric(inter["timestamp"], errors="coerce")
    uid = pd.to_numeric(inter["user_id"], errors="coerce")
    iid = pd.to_numeric(inter["item_id"], errors="coerce")

    min_ts = pd.Timestamp(v["min_timestamp"]).timestamp()
    now_ts = pd.Timestamp.now().timestamp()

    checks = {
        # missing values
        "missing_rating": inter["rating"].isna(),
        "missing_timestamp": inter["timestamp"].isna(),
        "missing_user_id": inter["user_id"].isna(),
        "missing_item_id": inter["item_id"].isna(),
        # data types
        "invalid_rating_dtype": inter["rating"].notna() & rating.isna(),
        "invalid_timestamp_dtype": inter["timestamp"].notna() & ts.isna(),
        # value ranges
        "rating_out_of_range": rating.notna() & ((rating < v["rating_min"]) | (rating > v["rating_max"])),
        "timestamp_before_min": ts.notna() & (ts < min_ts),
        "timestamp_in_future": ts.notna() & (ts > now_ts),
        # referential integrity
        "orphan_user_id": uid.notna() & ~uid.isin(valid_users),
        "orphan_item_id": iid.notna() & ~iid.isin(valid_items),
        # duplicates (same full row seen before)
        "duplicate_row": inter.duplicated(keep="first"),
    }
    for label, mask in checks.items():
        counts[label] = int(mask.sum())
        _flag(reasons, mask, label)
    return reasons, counts


def run(cfg: dict | None = None) -> dict:
    global log
    cfg = cfg or load_config()
    log = get_logger("validation", cfg)
    raw, errors = resolve(cfg, "raw"), resolve(cfg, "errors")
    out = resolve(cfg, "processed") / "validated"
    reports = resolve(cfg, "reports")

    users = pd.read_csv(raw / "users.csv", dtype={"zip_code": str})
    items = pd.read_csv(raw / "items.csv")
    inter = pd.read_csv(raw / "interactions.csv", dtype=object)  # object: detect bad types

    report = {"tables": {}}

    # users & items first -> they define the valid key sets for integrity checks
    u_fatal, u_warn, u_counts = validate_users(users)
    i_fatal, i_warn, i_counts = validate_items(items)
    good_users = users[u_fatal == ""].copy()
    good_items = items[i_fatal == ""].copy()
    good_users["age"] = pd.to_numeric(good_users["age"], errors="coerce")  # repaired later

    # referential integrity is checked against the surviving master records
    valid_users = set(pd.to_numeric(good_users["user_id"]).astype(int))
    valid_items = set(pd.to_numeric(good_items["item_id"]).astype(int))
    t_reasons, t_counts = validate_interactions(inter, valid_users, valid_items, cfg)
    good_inter = inter[t_reasons == ""].copy()

    for name, df, fatal, warn, counts, good in [
        ("users", users, u_fatal, u_warn, u_counts, good_users),
        ("items", items, i_fatal, i_warn, i_counts, good_items),
        ("interactions", inter, t_reasons, pd.Series("", index=inter.index), t_counts, good_inter),
    ]:
        bad = df[fatal != ""].copy()
        bad["failure_reason"] = fatal[fatal != ""]
        bad.to_csv(errors / f"{name}_errors.csv", index=False)
        warned = df[warn != ""].copy()
        warned["warning"] = warn[warn != ""]
        if len(warned):
            warned.to_csv(errors / f"{name}_warnings.csv", index=False)
        report["tables"][name] = {
            "rows_in": int(len(df)), "rows_valid": int(len(good)), "rows_rejected": int(len(bad)),
            "rows_repairable_warnings": int(len(warned)),
            "rejection_rate_pct": round(100 * len(bad) / max(len(df), 1), 3),
            "failures_by_check": counts,
        }
        log.info("%s: %d in, %d kept, %d rejected, %d warnings | %s", name, len(df), len(good),
                 len(bad), len(warned), {k: v for k, v in counts.items() if v})

    # write validated tables with proper dtypes
    good_users["user_id"] = good_users["user_id"].astype(int)
    good_items["item_id"] = good_items["item_id"].astype(int)
    for c in ("user_id", "item_id", "timestamp"):
        good_inter[c] = pd.to_numeric(good_inter[c]).astype("int64")
    good_inter["rating"] = pd.to_numeric(good_inter["rating"]).astype(float)
    save_table(good_users, out, "users")
    save_table(good_items, out, "items")
    save_table(good_inter, out, "interactions")

    with open(reports / "validation_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    _write_markdown(report, reports / "validation_report.md")
    return report


def _write_markdown(report: dict, path) -> None:
    lines = ["# Data Validation Report", ""]
    for name, t in report["tables"].items():
        lines += [f"## {name}", "",
                  f"- Rows in: **{t['rows_in']}**",
                  f"- Rows valid: **{t['rows_valid']}**",
                  f"- Rows rejected (moved to `data/errors/{name}_errors.csv`): "
                  f"**{t['rows_rejected']}** ({t['rejection_rate_pct']}%)",
                  f"- Rows kept with repairable warnings (`{name}_warnings.csv`): "
                  f"**{t['rows_repairable_warnings']}**",
                  "", "| Check | Rows flagged |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in t["failures_by_check"].items()]
        lines.append("")
    lines.append("_A row can fail several checks, so flagged counts may exceed rejected rows. "
                 "Fatal defects are rejected; repairable ones (missing/odd age, gender, title) are "
                 "kept, logged, and repaired in the preparation stage._")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
