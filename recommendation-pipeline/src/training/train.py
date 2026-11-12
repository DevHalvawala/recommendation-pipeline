"""Stage 6 - Model training.

Trains the popularity baseline, the SVD matrix-factorisation model and the hybrid
model on the TRAIN split, stores them in models/, and logs parameters + the model
artifact to one MLflow run per model (run ids saved in models/run_ids.json so the
evaluation stage can add metrics to the same runs).
"""
from __future__ import annotations

import json
import pickle

import scipy.sparse as sp

from src.common import ROOT, get_logger, load_config, load_table, resolve, set_seed
from src.features.feature_store import FeatureStore
from src.training.models import HybridRecommender, PopularityRecommender, SVDRecommender
from src.training.tracking import Tracker

log = None


def run(cfg: dict | None = None) -> dict:
    global log
    cfg = cfg or load_config()
    log = get_logger("training", cfg)
    seed = cfg["project"]["seed"]
    set_seed(seed)
    proc, models_dir, reports = resolve(cfg, "processed"), resolve(cfg, "models"), resolve(cfg, "reports")

    matrix = sp.load_npz(proc / "user_item_matrix.npz").tocsr()
    uids = load_table(proc, "matrix_user_index")["user_id"].to_numpy()
    iids = load_table(proc, "matrix_item_index")["item_id"].to_numpy()
    store = FeatureStore(resolve(cfg, "features"))

    t = cfg["training"]
    pop = PopularityRecommender(min_ratings=t["popularity"]["min_ratings"]).fit(matrix, uids, iids)
    svd = SVDRecommender(n_factors=t["svd"]["n_factors"], rating_factors=t["svd"]["rating_factors"],
                         bias_reg=t["svd"]["bias_reg"], seed=seed).fit(matrix, uids, iids)
    hyb = HybridRecommender(svd, store, alpha=t["hybrid"]["alpha"]).fit(matrix, uids, iids)

    params = {
        "popularity": {"min_ratings": t["popularity"]["min_ratings"]},
        "svd_mf": {"n_factors": t["svd"]["n_factors"], "rating_factors": t["svd"]["rating_factors"],
                   "bias_reg": t["svd"]["bias_reg"], "seed": seed},
        "hybrid_svd_content": {"alpha": t["hybrid"]["alpha"], "n_factors": t["svd"]["n_factors"]},
    }
    tracker = Tracker(cfg)
    run_ids = {}
    common = {"train_matrix_shape": str(matrix.shape), "train_interactions": int(matrix.nnz),
              "split": f"time-based {cfg['preparation']['train_ratio']}"}
    for model in (pop, svd, hyb):
        path = models_dir / f"{model.name}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(model, fh)
        with tracker.run(model.name) as rid:
            tracker.log_params({**params[model.name], **common, "model_type": model.name})
            tracker.log_artifact(path, "model")
            tracker.log_artifact(ROOT / "configs" / "config.yaml", "config")
            cat = reports / "feature_catalog.csv"
            if cat.exists():
                tracker.log_artifact(cat, "features")
        run_ids[model.name] = rid
        log.info("Trained %-20s run_id=%s saved=%s", model.name, rid, path.name)

    with open(models_dir / "run_ids.json", "w") as fh:
        json.dump({"backend": tracker.backend, "runs": run_ids}, fh, indent=2)
    return {"backend": tracker.backend, "runs": run_ids}


if __name__ == "__main__":
    run()
