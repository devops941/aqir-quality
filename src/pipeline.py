"""End-to-end pipeline CLI for the Air Quality Forecasting project.

Usage
-----
    python -m src.pipeline --source synthetic|csv --config config.yaml
                           [--mode A|B|both] [--end-date YYYY-MM-DD]

Steps (Section 5 of the spec):
    load (synthetic|csv) -> validate -> clean -> merge -> resample daily
    -> add CPCB AQI -> store daily readings in SQLite
    -> build features (Mode A and/or B) -> store daily_features
    -> train_and_log (baselines, Ridge/OLS, Random Forest, optional HistGB)
    -> reconcile single global best model -> print summary.

The whole project runs end-to-end on synthetic data, so no download or API
access is ever required for development or a demo.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.config import CONFIG, ensure_dirs, load_config, resolve_path
from src.data import clean, ingest, synthetic
from src.data.db import (
    AQReading,
    DailyFeature,
    Metric,
    ModelRun,
    Station,
    WeatherReading,
    get_engine,
    get_session,
    init_db,
)
from src.features import aqi as aqi_mod
from src.features import build as build_mod
from src.models.train import train_and_log

_AQ_FIELDS = ["pm25", "pm10", "no2", "so2", "co", "o3"]
_WEATHER_FIELDS = [
    "temp_c", "humidity_pct", "wind_speed_ms", "wind_dir_deg",
    "pressure_hpa", "precip_mm", "cloud_pct", "blh_m",
]


# ----------------------------------------------------------------------
# Data assembly
# ----------------------------------------------------------------------
def load_raw(source: str, config: dict, end_date: str | None = None):
    """Load raw air-quality, weather and events frames from the chosen source."""
    if source == "synthetic":
        aq, weather, events = synthetic.generate_synthetic(config, end_date=end_date)
    elif source == "csv":
        raw = resolve_path("data_raw")
        aq = ingest.validate_air_quality(ingest.load_air_quality(raw / "air_quality.csv"))
        weather = ingest.validate_weather(ingest.load_weather(raw / "weather.csv"))
        events_path = raw / "events.csv"
        if events_path.exists():
            events = ingest.load_events(events_path)
        else:
            # No events file — use an empty DataFrame with the standard schema
            events = pd.DataFrame(columns=["date", "event_name", "flag"])
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f"Unknown source: {source!r}")
    return aq, weather, events


def build_daily(aq: pd.DataFrame, weather: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean, merge, resample to daily and attach the CPCB AQI."""
    aq_clean = clean.clean_air_quality(aq, config)
    weather_clean = clean.clean_weather(weather, config)
    merged = clean.merge_aq_weather(aq_clean, weather_clean)
    daily = clean.resample_daily(merged, config)
    daily = aqi_mod.add_aqi(daily, config)
    return daily


# ----------------------------------------------------------------------
# SQLite persistence
# ----------------------------------------------------------------------
def _to_py_date(value):
    return pd.Timestamp(value).date()


def _num(value):
    """Return a plain float or None (never NaN/np types) for JSON/DB storage."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value) if isinstance(value, (int, float)) else value


def store_stations(session, config: dict, daily: pd.DataFrame) -> None:
    """Upsert the station registry (config stations ∪ stations seen in the data)."""
    known = {s["station_id"]: s for s in config.get("stations", [])}
    for sid in daily["station_id"].unique():
        meta = known.get(sid, {"station_id": sid})
        session.merge(
            Station(
                station_id=sid,
                station_name=meta.get("station_name"),
                city=meta.get("city", config.get("city", {}).get("name")),
                lat=meta.get("lat"),
                lon=meta.get("lon"),
            )
        )
    session.commit()


def store_daily_readings(session, daily: pd.DataFrame) -> None:
    """Persist daily aggregates into aq_readings and weather_readings."""
    aq_rows, weather_rows = [], []
    for row in daily.itertuples(index=False):
        d = getattr(row, "date")
        ts = pd.Timestamp(d).to_pydatetime()
        sid = getattr(row, "station_id")
        aq_rows.append(
            AQReading(
                station_id=sid,
                timestamp=ts,
                pm25=_num(getattr(row, "pm25", None)),
                pm10=_num(getattr(row, "pm10", None)),
                no2=_num(getattr(row, "no2", None)),
                so2=_num(getattr(row, "so2", None)),
                co=_num(getattr(row, "co", None)),
                o3=_num(getattr(row, "o3", None)),
                aqi=_num(getattr(row, "aqi", None)),
                aqi_category=getattr(row, "aqi_category", None),
                is_imputed=int(getattr(row, "is_imputed", 0) or 0),
            )
        )
        weather_rows.append(
            WeatherReading(
                station_id=sid,
                timestamp=ts,
                **{f: _num(getattr(row, f, None)) for f in _WEATHER_FIELDS},
            )
        )
    session.add_all(aq_rows)
    session.add_all(weather_rows)
    session.commit()


def store_daily_features(session, feat: pd.DataFrame) -> None:
    """Persist the engineered feature table (feature_json + targets + split)."""
    fcols = build_mod.feature_columns(feat)
    rows = []
    for row in feat.itertuples(index=False):
        rec = row._asdict()
        feature_json = json.dumps({c: _num(rec.get(c)) for c in fcols})
        rows.append(
            DailyFeature(
                station_id=rec["station_id"],
                date=_to_py_date(rec["date"]),
                feature_json=feature_json,
                target_pm25_next=_num(rec.get("target_pm25_next")),
                target_aqi_next=_num(rec.get("target_aqi_next")),
                split=rec.get("split"),
            )
        )
    session.add_all(rows)
    session.commit()


def reconcile_best(session) -> int | None:
    """Mark the single lowest-overall-RMSE non-baseline run as is_best=1."""
    baselines = {"persistence", "seasonal_naive"}
    rows = session.execute(
        select(ModelRun.run_id, ModelRun.model_name, Metric.rmse)
        .join(Metric, Metric.run_id == ModelRun.run_id)
        .where(Metric.scope == "overall")
    ).all()
    candidates = [(rid, rmse) for rid, name, rmse in rows
                  if name not in baselines and rmse is not None]
    best_run = min(candidates, key=lambda t: t[1])[0] if candidates else None
    for run in session.scalars(select(ModelRun)).all():
        run.is_best = 1 if run.run_id == best_run else 0
    session.commit()
    return best_run


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_pipeline(source: str = "synthetic", config: dict | None = None,
                 mode: str = "both", end_date: str | None = None) -> dict:
    """Execute the full pipeline and return a summary dict."""
    config = config or CONFIG
    ensure_dirs()
    modes = ["A", "B"] if mode == "both" else [mode]

    print(f"[1/6] Loading raw data (source={source}) ...")
    aq, weather, events = load_raw(source, config, end_date=end_date)
    print(f"      air-quality rows={len(aq):,}  weather rows={len(weather):,}")

    print("[2/6] Cleaning + merging + resampling to daily + AQI ...")
    daily = build_daily(aq, weather, config)
    print(f"      daily rows={len(daily):,}  stations={daily['station_id'].nunique()}")

    print("[3/6] Initialising SQLite and storing readings ...")
    engine = get_engine()
    init_db(engine, drop=True)
    session = get_session(engine)
    try:
        store_stations(session, config, daily)
        store_daily_readings(session, daily)

        print("[4/6] Building feature tables ...")
        feat_by_mode = {m: build_mod.build_feature_table(daily, events, config, mode=m) for m in modes}
        # daily_features stores the headline Mode A table (used by the live forecast).
        store_daily_features(session, feat_by_mode.get("A", feat_by_mode[modes[0]]))
        for m, f in feat_by_mode.items():
            print(f"      Mode {m}: rows={len(f):,}  features={len(build_mod.feature_columns(f))}")

        print("[5/6] Training + logging models ...")
        results = {}
        for m in modes:
            results[m] = train_and_log(feat_by_mode[m], config, mode=m, session=session)
            print(f"      Mode {m}: best_run_id={results[m]['best_run_id']}  runs={len(results[m]['runs'])}")

        print("[6/6] Reconciling global best model ...")
        best_run = reconcile_best(session)
        best = session.get(ModelRun, best_run) if best_run else None
        if best is not None:
            overall = session.get(Metric, (best_run, "overall"))
            print(f"      BEST: run_id={best_run}  {best.model_name} (mode {best.feature_mode})  "
                  f"RMSE={overall.rmse:.2f}  R2={overall.r2:.3f}  skill={overall.skill_vs_persistence:.3f}")
        return {"best_run_id": best_run, "modes": modes, "results": results,
                "trained_at": datetime.now().isoformat(timespec="seconds")}
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Air Quality Forecasting pipeline")
    parser.add_argument("--source", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--mode", choices=["A", "B", "both"], default="both")
    parser.add_argument("--end-date", default=None, help="End date for synthetic data (YYYY-MM-DD)")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else CONFIG
    summary = run_pipeline(args.source, config, mode=args.mode, end_date=args.end_date)
    print(f"\nDone. best_run_id={summary['best_run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
