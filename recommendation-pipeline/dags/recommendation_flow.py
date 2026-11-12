"""Prefect orchestration of the recommendation pipeline.

DAG (each arrow = data dependency enforced by passing the upstream result):

  ingest -> validate -> prepare -> transform -> features -> train -> evaluate

Run:  python run_pipeline.py --engine prefect
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, task  # noqa: E402

from src.common import load_config  # noqa: E402
from src.orchestration.stages import NAMES, run_stage  # noqa: E402


def _make_task(name):
    @task(name=name, retries=1, retry_delay_seconds=5, log_prints=True)
    def _t(cfg, upstream=None):  # `upstream` (previous task's result) creates the dependency edge
        return run_stage(name, cfg)
    return _t


TASKS = [(name, _make_task(name)) for name in NAMES]


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str | None = None, overrides: dict | None = None):
    cfg = load_config(config_path)
    for section, values in (overrides or {}).items():
        cfg[section].update(values)
    upstream = None
    for _name, t in TASKS:
        upstream = t(cfg, upstream)          # sequential dependency chain
    return "pipeline finished"


if __name__ == "__main__":
    recommendation_pipeline()
