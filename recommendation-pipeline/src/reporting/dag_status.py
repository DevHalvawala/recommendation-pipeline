"""Draws docs/dag_run.png: the pipeline DAG coloured by the outcome of the last run."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.common import ROOT

ORDER = ["ingest", "validate", "prepare", "transform", "features", "train", "evaluate"]
COLOR = {"success": "#86efac", "failed": "#fca5a5", "running": "#fde68a", None: "#e5e7eb"}


def draw(status: dict, path=None):
    path = path or ROOT / "docs" / "dag_run.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 2.9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 2.9); ax.axis("off")
    w, h, gap = 1.62, 1.2, 0.34
    for i, name in enumerate(ORDER):
        st = status.get(name, {})
        state = st.get("state")
        x = 0.15 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.75), w, h, boxstyle="round,pad=0.03", fc=COLOR[state], ec="#14532d" if state == "success" else "#475569", lw=1.4))
        ax.text(x + w / 2, 1.5, name, ha="center", va="center", fontsize=11, fontweight="bold")
        label = f"{state or 'not run'}" + (f"\n{st['seconds']}s" if "seconds" in st else "")
        ax.text(x + w / 2, 1.08, label, ha="center", va="center", fontsize=9)
        if i < len(ORDER) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 1.35), (x + w + gap - 0.02, 1.35), arrowstyle="-|>", mutation_scale=14, color="#334155"))
    ok = all(status.get(n, {}).get("state") == "success" for n in ORDER)
    total = sum(s.get("seconds", 0) for s in status.values())
    ax.text(7, 2.55, "recommendation-pipeline  |  " + ("ALL TASKS SUCCEEDED" if ok else "RUN INCOMPLETE / FAILED")
            + f"  |  total {total:.1f}s", ha="center", fontsize=12, fontweight="bold",
            color="#166534" if ok else "#991b1b")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path
