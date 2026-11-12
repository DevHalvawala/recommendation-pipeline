"""Builds reports/final_report.pdf (2-3 page body) from the REAL outputs of the run,
then appends any screenshots found in screenshots/ as a captioned appendix.

Numbers are read from reports/*.json|csv, so the report can never disagree with the results.
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from src.common import ROOT, load_config, resolve

LOWER_IS_BETTER = {"rmse", "mae"}


def _j(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _table(data, col_widths=None, header=True, font=8):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def _best_sentence(df: pd.DataFrame, k: int) -> str:
    parts = []
    for col in [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", "coverage", "diversity", "rmse", "mae"]:
        best = df[col].idxmin() if col in LOWER_IS_BETTER else df[col].idxmax()
        parts.append(f"{col} -> {best} ({df.loc[best, col]:.4f})")
    return "; ".join(parts)


def _interpretation(ev: pd.DataFrame, k: int) -> str:
    """Interpretation sentences derived from the numbers (never hard-coded claims)."""
    ndcg, div, cov = f"ndcg_at_{k}", "diversity", "coverage"
    base = "popularity"
    others = [m for m in ev.index if m != base]
    out = []
    if base in ev.index and others:
        best = ev.loc[others, ndcg].idxmax()
        b0, b1 = ev.loc[base, ndcg], ev.loc[best, ndcg]
        lift = (b1 / b0 - 1) * 100 if b0 > 0 else float("nan")
        out.append(f"The best personalised model ({best}) changes NDCG@{k} from {b0:.4f} to {b1:.4f} "
                   f"({lift:+.1f}% relative to the popularity baseline).")
        out.append(f"The popularity baseline recommends essentially the same items to every user (only already-seen "
                   f"items differ), which is why its catalogue coverage is {ev.loc[base, cov]:.3f} versus "
                   f"{ev.loc[others, cov].max():.3f} for the best personalised model.")
        if ev.loc[base, div] > ev.loc[others, div].max():
            out.append("Its lists are nevertheless the most diverse in genre terms, i.e. personalisation trades some "
                       "intra-list diversity for relevance.")
        else:
            out.append("Personalised lists are also at least as diverse in genre terms as the popularity lists here.")
    return " ".join(out)


def build(cfg: dict | None = None, screenshots_dir: str | Path | None = None) -> Path:
    cfg = cfg or load_config()
    reports = resolve(cfg, "reports")
    rep = cfg["report"]
    k = cfg["evaluation"]["k"]
    ing, val = _j(reports / "ingestion_summary.json"), _j(reports / "validation_report.json")
    prep, split = _j(reports / "preparation_summary.json"), _j(reports / "split_summary.json")
    feat = _j(reports / "feature_summary.json")
    ev = pd.read_csv(reports / "model_evaluation.csv", index_col=0)

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceAfter=4)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11, spaceBefore=7, spaceAfter=3)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9, leading=11.5, spaceAfter=3)
    SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10)
    bullet = lambda txt: Paragraph(f"&bull; {txt}", ParagraphStyle("bl", parent=B, leftIndent=10, firstLineIndent=-8))

    st = []
    st.append(Paragraph("End-to-End Data Management Pipeline for a Recommendation System", H1))
    st.append(Paragraph(f"DM4ML Assignment 2 &nbsp;|&nbsp; {escape(str(rep['student_name']))} ({escape(str(rep['student_id']))}) "
                        f"&nbsp;|&nbsp; GitHub: <u>{escape(str(rep['github_url']))}</u>", SM))

    # 1 design
    st.append(Paragraph("1. Overview and design decisions", H2))
    st.append(Paragraph(
        f"The pipeline ingests <b>{ing.get('source', '?')}</b> data ({ing.get('n_users', '?')} users, "
        f"{ing.get('n_items', '?')} items, {ing.get('n_interactions', '?')} interactions), validates it, "
        "prepares and transforms it, serves features through a small feature store, trains three recommenders "
        "tracked in MLflow, evaluates them, and runs as a single Prefect flow "
        "(ingest -> validate -> prepare -> transform -> features -> train -> evaluate).", B))
    arch = ROOT / "docs" / "architecture.png"
    if arch.exists():
        from PIL import Image as PILImage
        aw, ah = PILImage.open(arch).size
        st.append(Image(str(arch), 17.2 * cm, 17.2 * cm * ah / aw))
    for t in [
        "<b>Config-driven & reproducible:</b> every path, threshold, seed and hyper-parameter lives in "
        "<i>configs/config.yaml</i>; the seed is fixed and the pipeline runs end-to-end with one command.",
        "<b>Validation separates fatal from repairable defects:</b> fatal rows (duplicates, bad types, out-of-range "
        "ratings, invalid timestamps, orphan keys) go to <i>data/errors/</i> with a failure_reason; repairable "
        "master-data gaps (missing age, title) are kept, logged as warnings and repaired in preparation. This "
        "prevents one bad user row from cascading into thousands of lost interactions.",
        "<b>Time-based split, no leakage:</b> the earliest "
        f"{int(cfg['preparation']['train_ratio'] * 100)}% of the timeline is train. All features, the interaction "
        "matrix and the models use train only; 'days since last interaction' is measured from the last train "
        "timestamp rather than today.",
        "<b>Custom feature layer instead of Feast:</b> Parquet offline store + a registry/catalog + one "
        "<i>FeatureStore</i> class used identically in training and prediction, with cold-start defaults.",
        "<b>Plain-pandas validation instead of Great Expectations/Pandera:</b> keeps dependencies light and every "
        "rule explicit and unit-tested.",
    ]:
        st.append(bullet(t))

    # 2 data quality
    st.append(Paragraph("2. Data quality and preparation", H2))
    rows = [["Table", "Rows in", "Rejected", "Warnings (repaired)"]]
    for name, t in val.get("tables", {}).items():
        rows.append([name, t["rows_in"], f"{t['rows_rejected']} ({t['rejection_rate_pct']}%)",
                     t["rows_repairable_warnings"]])
    st.append(_table(rows, [4 * cm, 3 * cm, 4 * cm, 4.5 * cm]))
    st.append(Spacer(1, 3))
    inj = ing.get("injected_defects", {})
    st.append(Paragraph(
        "Because the source data is already clean, known defects were injected into a copy of the raw files "
        f"({', '.join(f'{a}={b}' for a, b in list(inj.items())[:8])}, ...); the validator flags them with matching "
        f"counts (see <i>reports/validation_report.md</i>). After cleaning there are {prep.get('interactions', '?')} "
        f"interactions (sparsity {prep.get('sparsity_pct', '?')}%). Split: {split.get('train_rows', '?')} train / "
        f"{split.get('test_rows', '?')} test rows, cutoff {split.get('cutoff_timestamp', '?')}; "
        f"{split.get('test_users_cold_start', '?')} test users are cold-start.", B))

    # 3 features
    st.append(Paragraph("3. Feature management", H2))
    st.append(Paragraph(
        f"{feat.get('user_feature_cols', '?')} user columns, {feat.get('item_feature_cols', '?')} item columns and "
        "interaction features are materialised in <i>data/features/</i> and documented in "
        "<i>reports/feature_catalog.md</i> (name, entity, description, formula, source, type). Highlights: "
        "user average rating, activity count, days since last interaction, preferred genre, genre taste profile; "
        "item popularity, average rating, genre vector; interaction recency and rating deviation.", B))

    # 4 models
    st.append(Paragraph("4. Models and experiment tracking", H2))
    for t in [
        "<b>Popularity (baseline):</b> items ranked by train rating count (min. ratings threshold); rating "
        "prediction = shrunk item mean.",
        "<b>SVD matrix factorisation:</b> PureSVD on the sparse train matrix for top-N ranking; global mean + "
        "regularised user/item biases + low-rank residual SVD for rating prediction.",
        "<b>Hybrid:</b> blend of min-max scaled SVD scores and a genre content score computed from the "
        "FeatureStore user genre profile and item genre vectors (alpha from config).",
        "Parameters, model files, config, feature catalog, metrics and evaluation plots are logged to one MLflow "
        "run per model.",
    ]:
        st.append(bullet(t))

    # 5 results
    st.append(Paragraph(f"5. Evaluation results (K = {k})", H2))
    cols = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", "coverage", "diversity", "rmse", "mae"]
    hdr = ["Model", f"P@{k}", f"R@{k}", f"NDCG@{k}", "Coverage", "Diversity", "RMSE", "MAE"]
    rows = [hdr] + [[m] + [f"{ev.loc[m, c]:.4f}" for c in cols] for m in ev.index]
    st.append(_table(rows, [3.9 * cm] + [1.85 * cm] * 7))
    st.append(Spacer(1, 3))
    n_eval = int(ev["n_eval_users"].iloc[0])
    st.append(Paragraph(
        f"Protocol: {n_eval} test users with at least one relevant item (rating >= "
        f"{cfg['preparation']['relevant_rating_threshold']}); candidates exclude items seen in train; coverage = "
        "share of the catalogue recommended to anyone; diversity = mean pairwise genre dissimilarity inside a list. "
        f"Best per metric: {_best_sentence(ev, k)}.", B))
    st.append(Paragraph(_interpretation(ev, k), B))
    chart = reports / "model_comparison.png"
    if chart.exists():
        from PIL import Image as PILImage
        cw, ch = PILImage.open(chart).size
        st.append(Image(str(chart), 12.5 * cm, 12.5 * cm * ch / cw))

    # 6 limitations
    st.append(Paragraph("6. Limitations", H2))
    for t in [
        "Offline evaluation on a single time-based split; no online A/B test and no confidence intervals.",
        "Cold-start users/items fall back to popularity or genre profiles; there is no dedicated cold-start model.",
        "Custom feature store has no online serving layer, point-in-time joins or TTLs (Feast would add these).",
        "Implicit-feedback signal is derived from explicit ratings; hyper-parameters were not exhaustively tuned.",
        "MovieLens 100K is small; scaling to millions of interactions would need sparse/ALS libraries and a "
        "distributed orchestrator.",
    ]:
        st.append(bullet(t))

    out = reports / "final_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.4 * cm,
                            title="DM4ML Assignment 2 - Recommendation Pipeline")

    # appendix with screenshots (optional)
    shots = []
    sdir = Path(screenshots_dir) if screenshots_dir else ROOT / "screenshots"
    if sdir.exists():
        shots = sorted(p for p in sdir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if shots:
        st.append(PageBreak())
        st.append(Paragraph("Appendix - Execution screenshots", H1))
        for p in shots:
            from PIL import Image as PILImage
            w, h = PILImage.open(p).size
            max_w, max_h = 17.2 * cm, 11.5 * cm
            scale = min(max_w / w, max_h / h)
            cap = p.stem.replace("_", " ").replace("-", " ")
            st.append(KeepTogether([Paragraph(f"<b>{cap}</b>", B), Image(str(p), w * scale, h * scale),
                                    Spacer(1, 8)]))
    doc.build(st)
    return out


if __name__ == "__main__":
    print(build())
