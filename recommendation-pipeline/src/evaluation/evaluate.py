"""Stage 7 - Model evaluation.

Protocol
  * test users  : users with >= 1 *relevant* test interaction (rating >= threshold)
  * candidates  : whole catalogue minus items the user already rated in train
  * relevant    : test items rated >= threshold (default 4)
  * metrics@K   : Precision, Recall, NDCG, Coverage, Diversity (K from config)
  * rating task : RMSE / MAE of predicted ratings on ALL test interactions
Metrics are logged to the MLflow run created by the training stage.
"""
from __future__ import annotations

import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import get_logger, load_config, load_table, resolve
from src.evaluation import metrics as M
from src.features.feature_store import GCOLS, FeatureStore
from src.training.tracking import Tracker

log = None
MODEL_FILES = ["popularity", "svd_mf", "hybrid_svd_content"]


def evaluate_model(model, test: pd.DataFrame, k: int, threshold: float, store: FeatureStore,
                   max_users: int | None = None, warm_only: bool = True) -> dict:
    rel = test[test["rating"] >= threshold].groupby("user_id")["item_id"].apply(set)
    n_cold = int(sum(1 for u in rel.index if u not in model.u_idx))
    users = [u for u in rel.index if (u in model.u_idx or not warm_only)]
    if max_users:
        users = users[:max_users]
    recs = model.recommend(users, k)

    p = [M.precision_at_k(recs[u], rel[u], k) for u in users]
    r = [M.recall_at_k(recs[u], rel[u], k) for u in users]
    n = [M.ndcg_at_k(recs[u], rel[u], k) for u in users]

    genre_lookup = dict(zip(store.items.index, store.items[GCOLS].to_numpy()))
    div = [M.intra_list_diversity(recs[u], genre_lookup) for u in users]
    cov = M.catalog_coverage(list(recs.values()), len(model.item_ids))

    pred = model.predict_rating(test["user_id"].to_numpy(), test["item_id"].to_numpy())
    return {
        f"precision_at_{k}": float(np.mean(p)), f"recall_at_{k}": float(np.mean(r)),
        f"ndcg_at_{k}": float(np.mean(n)), "coverage": float(cov),
        "diversity": float(np.mean(div)),
        "rmse": M.rmse(test["rating"], pred), "mae": M.mae(test["rating"], pred),
        "n_eval_users": len(users), "n_cold_start_users_in_test": n_cold,
    }


def run(cfg: dict | None = None) -> pd.DataFrame:
    global log
    cfg = cfg or load_config()
    log = get_logger("evaluation", cfg)
    proc, models_dir, reports = resolve(cfg, "processed"), resolve(cfg, "models"), resolve(cfg, "reports")
    k = cfg["evaluation"]["k"]
    thr = cfg["preparation"]["relevant_rating_threshold"]

    test = load_table(proc, "test")
    store = FeatureStore(resolve(cfg, "features"))
    run_info = json.loads((models_dir / "run_ids.json").read_text())
    tracker = Tracker(cfg)

    rows = []
    for name in MODEL_FILES:
        with open(models_dir / f"{name}.pkl", "rb") as fh:
            model = pickle.load(fh)
        res = evaluate_model(model, test, k, thr, store, cfg["evaluation"]["max_users"],
                             cfg["evaluation"]["warm_users_only"])
        rows.append({"model": name, **res})
        log.info("%-20s %s", name, {a: round(b, 4) for a, b in res.items()})

    df = pd.DataFrame(rows).set_index("model")
    csv_path, md_path = reports / "model_evaluation.csv", reports / "model_evaluation.md"
    df.round(4).to_csv(csv_path)

    # bar chart of the ranking metrics
    plot_path = reports / "model_comparison.png"
    cols = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", "coverage", "diversity"]
    ax = df[cols].plot(kind="bar", figsize=(9, 4.5), rot=0)
    ax.set_title(f"Model comparison (K={k})")
    ax.set_ylabel("score")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    md = [f"# Model Evaluation (K={k}, relevant = rating >= {thr})", "",
          f"Evaluated on {int(df['n_eval_users'].iloc[0])} test users"
          f"{' (warm users only; ' + str(int(df['n_cold_start_users_in_test'].iloc[0])) + ' cold-start users excluded)' if cfg['evaluation']['warm_users_only'] else ''}; "
          "candidates exclude train-seen items.", "",
          df.round(4).reset_index().to_markdown(index=False) if _has_tabulate() else df.round(4).to_string()]
    md_path.write_text("\n".join(md), encoding="utf-8")

    for name in MODEL_FILES:
        with tracker.run(name, run_id=run_info["runs"][name]):
            tracker.log_metrics({a: b for a, b in df.loc[name].items()})
            tracker.log_artifact(csv_path, "evaluation")
            tracker.log_artifact(plot_path, "evaluation")
    return df


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    print(run())
