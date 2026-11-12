"""One-command entry point.

    python run_pipeline.py                       # Prefect if installed, else plain runner
    python run_pipeline.py --engine plain        # force the plain Python runner
    python run_pipeline.py --source synthetic    # offline: use generated data
    python run_pipeline.py --only features train evaluate
"""
from __future__ import annotations

import argparse
import time

from src.common import get_logger, load_config
from src.orchestration.stages import NAMES, run_stage


def run_plain(cfg: dict, only: list[str] | None = None) -> None:
    log = get_logger("orchestrator", cfg)
    t_all = time.time()
    for name in NAMES:
        if only and name not in only:
            continue
        log.info("=== stage: %s ===", name)
        run_stage(name, cfg)
    log.info("Pipeline finished in %.1fs", time.time() - t_all)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--engine", choices=["prefect", "plain", "auto"], default="auto")
    ap.add_argument("--source", choices=["movielens", "synthetic"], default=None)
    ap.add_argument("--only", nargs="*", default=None, help="run only these stages (plain engine)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    overrides = {}
    if args.source:
        cfg["data"]["source"] = args.source
        overrides = {"data": {"source": args.source}}

    engine = args.engine
    if engine in ("auto", "prefect"):
        try:
            import prefect  # noqa: F401
            engine = "prefect"
        except ImportError:
            if engine == "prefect":
                raise SystemExit("Prefect is not installed: pip install prefect")
            engine = "plain"

    if engine == "prefect" and not args.only:
        from dags.recommendation_flow import recommendation_pipeline
        recommendation_pipeline(args.config, overrides)
    else:
        run_plain(cfg, args.only)


if __name__ == "__main__":
    main()
