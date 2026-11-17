"""Auto-report generator for the Air Quality Forecasting project (Phase 10).

Reads real data from ``data/aqi.db``, renders a Jinja2 Markdown template,
converts to HTML (with embedded CSS), and optionally produces a PDF via
``xhtml2pdf`` (pisa).  Never crashes — PDF failure is caught and logged.

Usage
-----
As a library::

    from src.report.generate import generate_report
    paths = generate_report()   # writes to outputs/

As a CLI::

    python -m src.report.generate

Public API
----------
``generate_report(config, out_dir, session) -> dict``
    Returns a dict of ``{"md": Path, "html": Path, "pdf": Path | None,
    "figures": {name: Path}}``.
"""
from __future__ import annotations

import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session

from src.config import CONFIG, resolve_path, ROOT
from src.data.db import (
    AQReading,
    DailyFeature,
    FeatureImportance,
    Metric,
    ModelRun,
    Prediction,
    Station,
    WeatherReading,
    get_engine,
    get_session,
)
from src.models.explain import residual_analysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TEMPLATE_DIR = Path(__file__).parent
_TEMPLATE_NAME = "template.md.j2"

_EMBEDDED_CSS = """
body {
    font-family: "Segoe UI", Arial, sans-serif;
    max-width: 1100px;
    margin: 0 auto;
    padding: 24px;
    color: #222;
    line-height: 1.6;
}
h1 { color: #1a3a5c; border-bottom: 3px solid #1a3a5c; padding-bottom: 8px; }
h2 { color: #2c5f8a; border-bottom: 1px solid #ccc; margin-top: 2em; }
h3 { color: #3a7abd; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 0.9em;
}
th {
    background-color: #1a3a5c;
    color: white;
    padding: 8px 12px;
    text-align: left;
}
td { padding: 7px 12px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) { background-color: #f5f8fb; }
tr:hover { background-color: #e8f0fa; }
img { max-width: 100%; height: auto; margin: 1em 0; border: 1px solid #ddd; padding: 4px; }
code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }
pre { background: #f4f4f4; padding: 12px; border-radius: 4px; overflow-x: auto; }
blockquote { border-left: 4px solid #1a3a5c; padding-left: 12px; color: #555; }
hr { border: none; border-top: 1px solid #ddd; margin: 2em 0; }
"""


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_report(
    config: dict[str, Any] = CONFIG,
    out_dir: Path | str | None = None,
    session: Session | None = None,
) -> dict[str, Any]:
    """Generate the full auto-report from real DB data.

    Parameters
    ----------
    config:
        Project configuration dict (defaults to the global singleton).
    out_dir:
        Directory where report files and figure PNGs are written.
        Defaults to ``outputs/`` relative to the project root.
    session:
        SQLAlchemy session to use.  A new session against the default DB is
        created if ``None``.

    Returns
    -------
    dict
        ``{"md": Path, "html": Path, "pdf": Path | None, "figures": {name: Path}}``.
    """
    # ---- Resolve output directory ----------------------------------------
    if out_dir is None:
        out_dir = resolve_path("outputs")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- DB session -------------------------------------------------------
    own_session = session is None
    if own_session:
        engine = get_engine()
        session = get_session(engine)

    try:
        context = _build_context(session, config, out_dir)
    finally:
        if own_session:
            session.close()

    # ---- Render Markdown --------------------------------------------------
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(_TEMPLATE_NAME)
    md_text = template.render(**context)

    md_path = out_dir / "report.md"
    md_path.write_text(md_text, encoding="utf-8")
    logger.info("Written Markdown report: %s", md_path)

    # ---- Markdown -> HTML ------------------------------------------------
    try:
        import markdown as md_lib
        html_body = md_lib.markdown(
            md_text,
            extensions=["tables", "fenced_code"],
        )
    except ImportError:
        logger.warning("markdown library not available; using plain pre-wrap HTML.")
        html_body = f"<pre>{md_text}</pre>"

    html_full = _wrap_html(html_body, title=f"AQI Report — {config['city']['name']}")
    html_path = out_dir / "report.html"
    html_path.write_text(html_full, encoding="utf-8")
    logger.info("Written HTML report: %s", html_path)

    # ---- HTML -> PDF (optional, never crash) -----------------------------
    # For PDF we use a version of the HTML with absolute image paths
    html_for_pdf = html_full.replace(
        'src="fig_', f'src="{out_dir.as_posix()}/fig_'
    )
    pdf_path: Path | None = out_dir / "report.pdf"
    try:
        from xhtml2pdf import pisa  # type: ignore

        with open(pdf_path, "wb") as pdf_fh:
            result = pisa.CreatePDF(html_for_pdf.encode("utf-8"), dest=pdf_fh)
        if result.err:
            logger.warning("xhtml2pdf reported errors (err=%s); PDF may be incomplete.", result.err)
        else:
            logger.info("Written PDF report: %s", pdf_path)
    except Exception as exc:  # pragma: no cover
        logger.warning("PDF generation failed (%s: %s) — skipping.", type(exc).__name__, exc)
        pdf_path = None

    # ---- Return paths ----------------------------------------------------
    result_paths: dict[str, Any] = {
        "md": md_path,
        "html": html_path,
        "pdf": pdf_path,
        "figures": context["_figure_paths"],
    }
    logger.info("Report generation complete: %s", result_paths)
    return result_paths


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(
    session: Session,
    config: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Query the DB and build all template context variables."""
    city_name = config["city"]["name"]
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- Stations --------------------------------------------------------
    stations_orm = session.query(Station).all()
    stations = [
        {
            "station_id": s.station_id,
            "station_name": s.station_name,
            "city": s.city,
            "lat": s.lat,
            "lon": s.lon,
        }
        for s in stations_orm
    ]

    # ---- Dataset summary -------------------------------------------------
    total_aq = session.query(AQReading).count()
    total_weather = session.query(WeatherReading).count()
    total_daily = session.query(DailyFeature).count()

    imputed_count = (
        session.query(AQReading).filter(AQReading.is_imputed == 1).count()
    )
    imputed_pct = round(imputed_count / max(total_aq, 1) * 100, 1)

    # Date range from aq_readings
    from sqlalchemy import func
    min_ts, max_ts = session.query(
        func.min(AQReading.timestamp), func.max(AQReading.timestamp)
    ).one()
    date_range_start = str(min_ts)[:10] if min_ts else "N/A"
    date_range_end = str(max_ts)[:10] if max_ts else "N/A"

    # Total predictions
    total_predictions = session.query(Prediction).count()

    # ---- Model runs with overall metrics ---------------------------------
    runs = session.query(ModelRun).order_by(ModelRun.run_id).all()
    model_comparison: list[dict[str, Any]] = []
    best_model: dict[str, Any] | None = None

    for run in runs:
        overall_m = (
            session.query(Metric)
            .filter(Metric.run_id == run.run_id, Metric.scope == "overall")
            .first()
        )
        if overall_m is None:
            continue
        row = {
            "run_id": run.run_id,
            "model_name": run.model_name,
            "feature_mode": run.feature_mode or "A",
            "is_best": bool(run.is_best),
            "mae": overall_m.mae or 0.0,
            "rmse": overall_m.rmse or 0.0,
            "r2": overall_m.r2 or 0.0,
            "mape": overall_m.mape or 0.0,
            "skill_vs_persistence": overall_m.skill_vs_persistence,
            "category_accuracy": overall_m.category_accuracy or 0.0,
            "event_f1": overall_m.event_f1 or 0.0,
        }
        model_comparison.append(row)
        if run.is_best:
            best_model = row

    if best_model is None and model_comparison:
        # Fallback: pick lowest RMSE
        best_model = min(model_comparison, key=lambda x: x["rmse"])
        best_model["is_best"] = True

    best_run_id: int = best_model["run_id"]  # type: ignore

    # ---- Best model metrics by season and station ------------------------
    all_best_metrics = (
        session.query(Metric)
        .filter(Metric.run_id == best_run_id)
        .all()
    )
    seasonal_metrics = [
        _metric_to_dict(m) for m in all_best_metrics if m.scope.startswith("season:")
    ]
    station_metrics = [
        _metric_to_dict(m) for m in all_best_metrics if m.scope.startswith("station:")
    ]

    # ---- Predictions for best run (first station) for plotting -----------
    first_station_id = stations[0]["station_id"] if stations else "DL001"
    preds_orm = (
        session.query(Prediction)
        .filter(
            Prediction.run_id == best_run_id,
            Prediction.station_id == first_station_id,
        )
        .order_by(Prediction.target_date)
        .all()
    )
    pred_dates = [p.target_date for p in preds_orm]
    y_true_arr = np.array([p.y_true for p in preds_orm], dtype=float)
    y_pred_arr = np.array([p.y_pred for p in preds_orm], dtype=float)

    # Test period date range
    if pred_dates:
        test_period_start = str(pred_dates[0])
        test_period_end = str(pred_dates[-1])
    else:
        test_period_start = test_period_end = "N/A"

    # ---- Feature importance for best run ---------------------------------
    fi_rows = (
        session.query(FeatureImportance)
        .filter(FeatureImportance.run_id == best_run_id)
        .order_by(FeatureImportance.permutation_importance.desc())
        .limit(20)
        .all()
    )
    fi_has_data = len(fi_rows) > 0

    # ---- Generate figures ------------------------------------------------
    figure_paths: dict[str, str] = {}

    # 1. Predicted vs actual line
    fig_pred = _plot_pred_vs_actual(
        pred_dates, y_true_arr, y_pred_arr,
        station_name=stations[0]["station_name"] if stations else first_station_id,
        out_path=out_dir / "fig_pred_vs_actual.png",
    )
    figure_paths["pred_vs_actual"] = "fig_pred_vs_actual.png"

    # 2. Scatter
    fig_scatter = _plot_scatter(
        y_true_arr, y_pred_arr,
        r2=best_model["r2"],
        out_path=out_dir / "fig_scatter.png",
    )
    figure_paths["scatter"] = "fig_scatter.png"

    # 3. Feature importance
    if fi_has_data:
        features = [fi.feature for fi in fi_rows]
        perm_imp = [fi.permutation_importance or 0.0 for fi in fi_rows]
        _plot_feature_importance(
            features, perm_imp,
            out_path=out_dir / "fig_feature_importance.png",
        )
    else:
        # Create placeholder figure
        _plot_placeholder("Feature importance data not available", out_dir / "fig_feature_importance.png")
    figure_paths["feature_importance"] = "fig_feature_importance.png"

    # 4. Residuals over time
    if len(pred_dates) > 0:
        ra = residual_analysis(y_true_arr, y_pred_arr, pred_dates)
        _plot_residuals(
            pred_dates, ra["residuals"],
            ljung_box_p=ra["ljung_box_p"],
            out_path=out_dir / "fig_residuals.png",
        )
    else:
        _plot_placeholder("No prediction data for residual analysis", out_dir / "fig_residuals.png")
    figure_paths["residuals"] = "fig_residuals.png"

    return {
        # Metadata
        "generated_at": generated_at,
        "city": city_name,
        "stations": stations,
        # Dataset summary
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "total_aq_readings": total_aq,
        "total_weather_readings": total_weather,
        "total_daily_features": total_daily,
        "imputed_count": imputed_count,
        "imputed_pct": imputed_pct,
        "total_predictions": total_predictions,
        # Model comparison
        "model_comparison": model_comparison,
        "best_model": best_model,
        "test_period_start": test_period_start,
        "test_period_end": test_period_end,
        # Scoped metrics
        "seasonal_metrics": seasonal_metrics,
        "station_metrics": station_metrics,
        # Figures (relative path names for embedding in MD)
        "figures": figure_paths,
        # Private: absolute Paths for the return dict
        "_figure_paths": {k: out_dir / v for k, v in figure_paths.items()},
    }


# ---------------------------------------------------------------------------
# Metric dict helper
# ---------------------------------------------------------------------------

def _metric_to_dict(m: Metric) -> dict[str, Any]:
    return {
        "scope": m.scope,
        "mae": m.mae or 0.0,
        "rmse": m.rmse or 0.0,
        "r2": m.r2 or 0.0,
        "mape": m.mape or 0.0,
        "skill_vs_persistence": m.skill_vs_persistence,
        "category_accuracy": m.category_accuracy,
        "event_precision": m.event_precision,
        "event_recall": m.event_recall,
        "event_f1": m.event_f1,
    }


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def _plot_pred_vs_actual(
    dates: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    station_name: str,
    out_path: Path,
) -> None:
    """Line chart of predicted vs actual PM2.5 over the test period."""
    fig, ax = plt.subplots(figsize=(12, 4))
    if len(dates) > 0:
        date_series = pd.to_datetime(dates)
        ax.plot(date_series, y_true, label="Actual PM2.5", color="#1a3a5c", linewidth=1.2, alpha=0.9)
        ax.plot(date_series, y_pred, label="Predicted PM2.5", color="#e74c3c", linewidth=1.2,
                linestyle="--", alpha=0.85)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title(f"Predicted vs Actual PM2.5 — {station_name} (Test Period)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved figure: %s", out_path)


def _plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r2: float,
    out_path: Path,
) -> None:
    """Scatter plot of predicted vs actual values with identity line."""
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(y_true) > 0:
        ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#2c5f8a", edgecolors="none")
        lo = min(y_true.min(), y_pred.min()) * 0.95
        hi = max(y_true.max(), y_pred.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="y = x (perfect)")
    ax.set_xlabel("Actual PM2.5 (µg/m³)")
    ax.set_ylabel("Predicted PM2.5 (µg/m³)")
    ax.set_title(f"Predicted vs Actual Scatter (R² = {r2:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved figure: %s", out_path)


def _plot_feature_importance(
    features: list[str],
    importances: list[float],
    out_path: Path,
    max_features: int = 15,
) -> None:
    """Horizontal bar chart of permutation feature importances."""
    n = min(len(features), max_features)
    feats = features[:n][::-1]
    imps = importances[:n][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.4)))
    bars = ax.barh(feats, imps, color="#2c5f8a", alpha=0.8)
    ax.set_xlabel("Permutation Importance (mean decrease in RMSE)")
    ax.set_title(f"Top {n} Feature Importances (Permutation)")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, imps):
        ax.text(val + max(imps) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved figure: %s", out_path)


def _plot_residuals(
    dates: list,
    residuals: list[float],
    ljung_box_p: float,
    out_path: Path,
) -> None:
    """Scatter/line plot of residuals over the test period."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [2, 1]})

    date_series = pd.to_datetime(dates)
    res_arr = np.array(residuals)

    # Residuals vs time
    ax1.scatter(date_series, res_arr, s=12, alpha=0.5, color="#e74c3c")
    ax1.axhline(0, color="black", linewidth=1.0)
    ax1.axhline(res_arr.std(), color="#999", linewidth=0.8, linestyle="--", label="+1σ")
    ax1.axhline(-res_arr.std(), color="#999", linewidth=0.8, linestyle="--", label="-1σ")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax1.set_ylabel("Residual (µg/m³)")
    ax1.set_title(
        f"Residuals Over Time  |  Ljung-Box p={ljung_box_p:.4f}"
        + (" (autocorrelation detected)" if ljung_box_p < 0.05 else " (no significant autocorrelation)")
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Residual histogram
    ax2.hist(res_arr, bins=30, color="#2c5f8a", alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=1.0)
    ax2.set_xlabel("Residual (µg/m³)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved figure: %s", out_path)


def _plot_placeholder(message: str, out_path: Path) -> None:
    """Write a simple placeholder figure."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="#888")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML wrapper
# ---------------------------------------------------------------------------

def _wrap_html(body: str, title: str = "Air Quality Report") -> str:
    """Wrap an HTML body fragment in a full HTML document with embedded CSS."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{title}</title>
<style>
{_EMBEDDED_CSS}
</style>
</head>
<body>
{body}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    """CLI: generate the report against the default DB."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    paths = generate_report()
    print("\nReport generated successfully:")
    for key, val in paths.items():
        if key == "figures":
            for fname, fpath in val.items():
                size = fpath.stat().st_size if fpath.exists() else 0
                print(f"  figure/{fname}: {fpath}  ({size:,} bytes)")
        elif val is not None:
            size = val.stat().st_size if isinstance(val, Path) and val.exists() else 0
            print(f"  {key}: {val}  ({size:,} bytes)")
        else:
            print(f"  {key}: (not generated)")


if __name__ == "__main__":
    _main()
