"""Single source of truth for the pipeline stages and their order.

Both the Prefect flow and the plain-Python runner import this list, and both
execute stages through `run_stage`, which records status + duration in
reports/run_status.json and draws docs/dag_run.png (a green/red DAG of the run).
"""
from __future__ import annotations

import json
import time
import traceback

from src.common import get_logger, resolve
from src.evaluation import evaluate
from src.features import feature_store
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.validation import validate

# (stage name, callable(cfg)) - each stage depends on the one before it.
STAGES = [
    ("ingest", ingest.run),
    ("validate", validate.run),
    ("prepare", prepare.run),
    ("transform", transform.run),
    ("features", feature_store.run),
    ("train", train.run),
    ("evaluate", evaluate.run),
]
FN = dict(STAGES)
NAMES = [n for n, _ in STAGES]


def _status_path(cfg):
    return resolve(cfg, "reports") / "run_status.json"


def _load(cfg) -> dict:
    p = _status_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {}


def run_stage(name: str, cfg: dict) -> str:
    """Run one stage, record its outcome, return a small marker string."""
    log = get_logger("orchestrator", cfg)
    status = {} if name == NAMES[0] else _load(cfg)      # first stage starts a fresh run record
    t0 = time.time()
    status[name] = {"state": "running"}
    _status_path(cfg).write_text(json.dumps(status, indent=2))
    try:
        FN[name](cfg)
    except Exception:
        status[name] = {"state": "failed", "seconds": round(time.time() - t0, 2),
                        "error": traceback.format_exc(limit=3)}
        _status_path(cfg).write_text(json.dumps(status, indent=2))
        _draw(cfg)
        raise
    status[name] = {"state": "success", "seconds": round(time.time() - t0, 2)}
    _status_path(cfg).write_text(json.dumps(status, indent=2))
    log.info("stage %-9s finished in %.1fs", name, status[name]["seconds"])
    if name == NAMES[-1]:
        _draw(cfg)
    return f"{name}: success"


def _draw(cfg) -> None:
    try:
        from src.reporting.dag_status import draw
        draw(_load(cfg))
    except Exception as exc:  # drawing must never break the pipeline
        get_logger("orchestrator", cfg).warning("could not draw DAG image: %s", exc)
