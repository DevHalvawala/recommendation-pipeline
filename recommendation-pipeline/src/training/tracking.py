"""Experiment tracking wrapper: MLflow when available, local JSON fallback otherwise.

The fallback keeps the pipeline runnable on machines without MLflow; in Colab
MLflow is installed via requirements.txt so the real backend is used.
Runs can be re-opened by run_id, so training logs params + model and the
evaluation stage later adds metrics to the SAME run.
"""
from __future__ import annotations

import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.common import ROOT, get_logger

try:  # pragma: no cover - environment dependent
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

EXPERIMENT = "recommendation-pipeline"


class Tracker:
    def __init__(self, cfg: dict):
        self.log = get_logger("tracking", cfg)
        self.root = ROOT / cfg["paths"]["mlruns"]
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = "local-json"
        self._run_dir: Path | None = None
        if mlflow is not None:
            try:
                mlflow.set_tracking_uri(f"sqlite:///{(self.root / 'mlflow.db').as_posix()}")
                if mlflow.get_experiment_by_name(EXPERIMENT) is None:
                    mlflow.create_experiment(EXPERIMENT,
                                             artifact_location=(self.root / "artifacts").as_uri())
                mlflow.set_experiment(EXPERIMENT)
                self.backend = "mlflow"
            except Exception as exc:  # pragma: no cover
                self.log.warning("MLflow init failed (%s) - using local JSON tracker", exc)
        self.log.info("Tracking backend: %s", self.backend)

    @contextmanager
    def run(self, run_name: str, run_id: str | None = None):
        if self.backend == "mlflow":
            with mlflow.start_run(run_id=run_id, run_name=None if run_id else run_name) as r:
                yield r.info.run_id
        else:
            rid = run_id or uuid.uuid4().hex
            self._run_dir = self.root / "local_runs" / rid
            self._run_dir.mkdir(parents=True, exist_ok=True)
            (self._run_dir / "name.txt").write_text(run_name)
            yield rid
            self._run_dir = None

    # ---- logging ------------------------------------------------------
    def _merge_json(self, fname: str, data: dict):
        p = self._run_dir / fname
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur.update(data)
        p.write_text(json.dumps(cur, indent=2, default=str))

    def log_params(self, params: dict):
        if self.backend == "mlflow":
            mlflow.log_params(params)
        else:
            self._merge_json("params.json", params)

    def log_metrics(self, metrics: dict):
        if self.backend == "mlflow":
            mlflow.log_metrics(metrics)
        else:
            self._merge_json("metrics.json", metrics)

    def set_tags(self, tags: dict):
        if self.backend == "mlflow":
            mlflow.set_tags(tags)
        else:
            self._merge_json("tags.json", tags)

    def log_artifact(self, path, artifact_dir: str | None = None):
        if self.backend == "mlflow":
            mlflow.log_artifact(str(path), artifact_path=artifact_dir)
        else:
            dest = self._run_dir / "artifacts" / (artifact_dir or "")
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, dest / Path(path).name)
