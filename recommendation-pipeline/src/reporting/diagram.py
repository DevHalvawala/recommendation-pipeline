"""Draws the pipeline architecture diagram (docs/architecture.png) with matplotlib."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.common import ROOT

STAGES = [
    ("1. Ingestion", "MovieLens 100K\n(or synthetic)\n+ error injection", "data/raw/"),
    ("2. Validation", "6 rule families\nPandas checks\nreport + error rows", "data/errors/"),
    ("3. Preparation", "repair nulls, dedupe,\nstandardise formats", "data/processed/"),
    ("4. Transformation", "time-based split\nactivity, popularity,\nuser-item matrix", "data/processed/"),
    ("5. Features", "user / item /\ninteraction features\n+ catalog", "data/features/"),
    ("6. Training", "Popularity, SVD,\nHybrid\n+ MLflow tracking", "models/ , mlruns/"),
    ("7. Evaluation", "P@K R@K NDCG@K\nCoverage Diversity\nRMSE MAE", "reports/"),
]


def draw(path=None):
    path = path or ROOT / "docs" / "architecture.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.set_xlim(0, 15.4); ax.set_ylim(0, 4.6); ax.axis("off")
    w, h, gap, y = 1.85, 2.2, 0.28, 1.35
    colors = ["#dbeafe", "#fee2e2", "#fef3c7", "#fef3c7", "#dcfce7", "#ede9fe", "#ede9fe"]
    for i, (title, body, store) in enumerate(STAGES):
        x = 0.2 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=colors[i], ec="#334155", lw=1.3))
        ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.1, body, ha="center", va="center", fontsize=8.6)
        ax.text(x + w / 2, y - 0.35, store, ha="center", va="center", fontsize=8, style="italic", color="#475569")
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, y + h / 2), (x + w + gap - 0.02, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=14, color="#334155"))
    total = 0.2 + len(STAGES) * (w + gap) - gap
    ax.add_patch(FancyBboxPatch((0.2, 0.08), total - 0.2, 0.38, boxstyle="round,pad=0.02", fc="#f1f5f9", ec="#94a3b8"))
    ax.text(total / 2 + 0.1, 0.27, "Orchestration: Prefect flow (dags/recommendation_flow.py)  |  "
            "config: configs/config.yaml  |  one-command run: python run_pipeline.py",
            ha="center", va="center", fontsize=9)
    ax.text(total / 2 + 0.1, 4.25, "End-to-End Data Management Pipeline for a Recommendation System",
            ha="center", fontsize=13, fontweight="bold")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


if __name__ == "__main__":
    print(draw())
