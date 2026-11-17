"""Live next-day AQI forecast for the Air Quality Forecasting project (Phase 8).

Public API (binding — see docs/INTERFACES.md):
    fetch_weather_forecast(lat, lon, config) -> pd.DataFrame  (cached, offline-safe)
    make_live_forecast(station_id, config, session) -> dict
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CONFIG, ROOT
from src.data.db import (
    AQReading,
    LiveForecast,
    ModelRun,
    Station,
    WeatherReading,
    get_engine,
    get_session,
    init_db,
)
from src.features.aqi import aqi_category, compute_subindex, add_aqi
from src.features.build import build_feature_table, feature_columns, season_of
from src.models.train import load_model

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_NON_BASELINE_MODELS = {"ridge", "random_forest", "hist_gb"}


def _category_color_health(
    category: str | None,
    config: dict[str, Any] = CONFIG,
) -> tuple[str | None, str | None]:
    """Return (color, health) for an AQI category from config."""
    if category is None:
        return None, None
    for cat in config.get("aqi", {}).get("categories", []):
        if cat["name"] == category:
            return cat["color"], cat["health"]
    return None, None


def _pm25_to_aqi_and_category(
    pm25: float,
    config: dict[str, Any] = CONFIG,
) -> tuple[float | None, str | None]:
    """Convert a PM2.5 concentration to (AQI sub-index, category)."""
    if pm25 is None or (isinstance(pm25, float) and math.isnan(pm25)):
        return None, None
    si = compute_subindex("pm25", pm25, config)
    if math.isnan(si):
        return None, None
    cat = aqi_category(si, config)
    return round(si, 1), cat


def _find_best_mode_a_run_id(session) -> int | None:
    """Query DB for the lowest-overall-RMSE Mode-A non-baseline run."""
    from sqlalchemy import select
    from src.data.db import Metric

    stmt = (
        select(ModelRun, Metric)
        .join(Metric, (ModelRun.run_id == Metric.run_id) & (Metric.scope == "overall"))
        .where(ModelRun.feature_mode == "A")
        .where(ModelRun.model_name.in_(list(_NON_BASELINE_MODELS)))
        .where(Metric.rmse.isnot(None))
        .order_by(Metric.rmse)
        .limit(1)
    )
    row = session.execute(stmt).first()
    if row is None:
        return None
    run, _ = row
    return run.run_id


def _load_daily_for_station(
    station_id: str,
    session,
    n_days: int = 45,
) -> pd.DataFrame:
    """Load recent daily AQ + weather readings and return a daily-merged frame.

    The returned DataFrame has columns matching the daily-merged contract:
    station_id, date, pm25, pm10, no2, so2, co, o3, temp_c, humidity_pct,
    wind_speed_ms, wind_dir_deg, pressure_hpa, precip_mm, cloud_pct, aqi, aqi_category.
    Rows are daily (one per calendar day), sorted ascending by date.
    """
    from sqlalchemy import select

    # Pull AQ readings (they are already daily in the DB for this project)
    aq_stmt = (
        select(AQReading)
        .where(AQReading.station_id == station_id)
        .order_by(AQReading.timestamp.desc())
        .limit(n_days + 5)
    )
    aq_rows = session.execute(aq_stmt).scalars().all()

    wx_stmt = (
        select(WeatherReading)
        .where(WeatherReading.station_id == station_id)
        .order_by(WeatherReading.timestamp.desc())
        .limit(n_days + 5)
    )
    wx_rows = session.execute(wx_stmt).scalars().all()

    if not aq_rows:
        return pd.DataFrame()

    aq_df = pd.DataFrame(
        [
            {
                "station_id": r.station_id,
                "date": pd.Timestamp(r.timestamp).normalize(),
                "pm25": r.pm25,
                "pm10": r.pm10,
                "no2": r.no2,
                "so2": r.so2,
                "co": r.co,
                "o3": r.o3,
                "aqi": r.aqi,
                "aqi_category": r.aqi_category,
            }
            for r in aq_rows
        ]
    )
    aq_df = aq_df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    wx_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(r.timestamp).normalize(),
                "temp_c": r.temp_c,
                "humidity_pct": r.humidity_pct,
                "wind_speed_ms": r.wind_speed_ms,
                "wind_dir_deg": r.wind_dir_deg,
                "pressure_hpa": r.pressure_hpa,
                "precip_mm": r.precip_mm,
                "cloud_pct": r.cloud_pct,
            }
            for r in wx_rows
        ]
    )
    wx_df = wx_df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    # Merge on date
    daily = aq_df.merge(wx_df, on="date", how="left")
    daily["station_id"] = station_id

    return daily.tail(n_days).reset_index(drop=True)


def _build_feature_row_for_latest_day(
    daily: pd.DataFrame,
    bundle: dict[str, Any],
    config: dict[str, Any] = CONFIG,
) -> pd.Series:
    """Build a single feature vector for the latest observed day t.

    Strategy: append a dummy next-day row so that build_feature_table does not
    drop day t (it drops the last row per station because target = shift(-1)
    produces NaN).  Then retrieve the row for date t.

    Parameters
    ----------
    daily:
        Daily-merged frame for a single station, sorted ascending by date.
    bundle:
        Model artifact bundle (contains feature_names and feature_defaults).
    config:
        Project config dict.

    Returns
    -------
    pd.Series
        Feature vector aligned to bundle['feature_names'], with any missing
        columns filled from bundle['feature_defaults'].
    """
    if daily.empty:
        defaults = bundle["feature_defaults"]
        return pd.Series({f: defaults.get(f, 0.0) for f in bundle["feature_names"]})

    latest_date = daily["date"].max()

    # --- Append a dummy next-day row so the latest day gets a target slot ---
    dummy_date = latest_date + pd.Timedelta(days=1)
    # Use the last row's values as a placeholder for the dummy
    dummy_row = daily.iloc[-1:].copy()
    dummy_row["date"] = dummy_date
    # Give it a non-NaN pm25 so dropna in build_feature_table keeps it
    dummy_row["pm25"] = daily["pm25"].median()
    daily_ext = pd.concat([daily, dummy_row], ignore_index=True)

    # --- make sure aqi columns exist (build_feature_table needs aqi for target) ---
    if "aqi" not in daily_ext.columns or daily_ext["aqi"].isna().all():
        daily_ext = add_aqi(daily_ext, config)

    # --- Build an empty events frame (no special events expected) ---
    events = pd.DataFrame(columns=["date", "event_name", "flag"])

    try:
        feat_df = build_feature_table(daily_ext, events, config=config, mode="A")
    except Exception as exc:
        warnings.warn(f"build_feature_table failed in live forecast: {exc}")
        defaults = bundle["feature_defaults"]
        return pd.Series({f: defaults.get(f, 0.0) for f in bundle["feature_names"]})

    # Retrieve the row corresponding to the latest observed day t
    feat_df["date"] = pd.to_datetime(feat_df["date"])
    row = feat_df[feat_df["date"] == latest_date]

    if row.empty:
        # Fall back to defaults
        defaults = bundle["feature_defaults"]
        return pd.Series({f: defaults.get(f, 0.0) for f in bundle["feature_names"]})

    row = row.iloc[0]

    # Align to bundle's feature_names, filling gaps with feature_defaults
    defaults = bundle["feature_defaults"]
    vec = {
        f: float(row[f]) if f in row.index and pd.notna(row[f]) else defaults.get(f, 0.0)
        for f in bundle["feature_names"]
    }
    return pd.Series(vec)


def _persist_forecast(
    session,
    station_id: str,
    issued_at: datetime,
    target_date: date,
    pm25_pred: float,
    aqi_pred: float | None,
    category: str | None,
    run_id: int,
) -> None:
    """Upsert a LiveForecast row in the database."""
    from sqlalchemy import select

    existing = session.execute(
        select(LiveForecast)
        .where(LiveForecast.station_id == station_id)
        .where(LiveForecast.target_date == target_date)
    ).scalars().first()

    if existing is not None:
        existing.issued_at = issued_at
        existing.pm25_pred = pm25_pred
        existing.aqi_pred = aqi_pred
        existing.category = category
        existing.run_id = run_id
    else:
        lf = LiveForecast(
            station_id=station_id,
            issued_at=issued_at,
            target_date=target_date,
            pm25_pred=pm25_pred,
            aqi_pred=aqi_pred,
            category=category,
            run_id=run_id,
        )
        session.add(lf)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_weather_forecast(
    lat: float,
    lon: float,
    config: dict[str, Any] = CONFIG,
) -> pd.DataFrame:
    """Fetch next-day weather forecast from Open-Meteo with on-disk caching.

    Degrades gracefully: returns an empty DataFrame if the API is unreachable
    or the response cannot be parsed.  Never raises; never blocks the demo.

    Parameters
    ----------
    lat, lon:
        Station coordinates.
    config:
        Project config dict.  Uses ``open_meteo.forecast_url`` and
        ``open_meteo.cache_dir``.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with columns matching the weather contract, indexed
        by date (next day).  Empty DataFrame on failure.
    """
    try:
        import urllib.request

        forecast_url = config.get("open_meteo", {}).get(
            "forecast_url", "https://api.open-meteo.com/v1/forecast"
        )
        cache_dir_rel = config.get("open_meteo", {}).get("cache_dir", "data/raw/open_meteo_cache")
        cache_dir = (ROOT / cache_dir_rel).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache key: lat/lon + today's date
        today_str = date.today().isoformat()
        cache_key = hashlib.md5(f"{lat}_{lon}_{today_str}".encode()).hexdigest()
        cache_file = cache_dir / f"forecast_{cache_key}.json"

        # Serve from cache if present
        if cache_file.exists():
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            params = (
                f"latitude={lat}&longitude={lon}"
                "&daily=temperature_2m_mean,relative_humidity_2m_mean,"
                "wind_speed_10m_mean,wind_direction_10m_dominant,"
                "surface_pressure_mean,precipitation_sum,cloud_cover_mean"
                "&forecast_days=2&timezone=Asia%2FKolkata"
            )
            url = f"{forecast_url}?{params}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = json.loads(resp.read().decode())
            cache_file.write_text(json.dumps(raw), encoding="utf-8")

        # Parse the response — take index 1 (tomorrow)
        daily = raw.get("daily", {})
        dates = daily.get("time", [])
        if len(dates) < 2:
            return pd.DataFrame()

        def _safe(key: str, idx: int = 1) -> float | None:
            vals = daily.get(key, [])
            return float(vals[idx]) if idx < len(vals) and vals[idx] is not None else None

        row = {
            "date": pd.Timestamp(dates[1]),
            "temp_c": _safe("temperature_2m_mean"),
            "humidity_pct": _safe("relative_humidity_2m_mean"),
            "wind_speed_ms": _safe("wind_speed_10m_mean"),
            "wind_dir_deg": _safe("wind_direction_10m_dominant"),
            "pressure_hpa": _safe("surface_pressure_mean"),
            "precip_mm": _safe("precipitation_sum"),
            "cloud_pct": _safe("cloud_cover_mean"),
        }
        return pd.DataFrame([row])

    except Exception as exc:
        warnings.warn(f"fetch_weather_forecast failed (offline mode): {exc}")
        return pd.DataFrame()


def make_live_forecast(
    station_id: str,
    config: dict[str, Any] = CONFIG,
    session=None,
) -> dict[str, Any]:
    """Generate and persist the next-day PM2.5 forecast for a station.

    Uses the best Mode-A model (lowest overall RMSE among non-baseline runs
    with feature_mode='A').  Builds the feature vector from the latest
    stored daily data, predicts PM2.5 for t+1, derives AQI category, and
    writes a LiveForecast row to the database.

    Parameters
    ----------
    station_id:
        Target station (e.g. ``"DL001"``).
    config:
        Project config dict.
    session:
        SQLAlchemy Session.  If ``None``, a temporary session is created and
        committed internally.

    Returns
    -------
    dict
        ``{station_id, station_name, issued_at, target_date, pm25_pred,
        aqi_pred, category, color, health, run_id, model_name}``
    """
    _own_session = session is None
    if _own_session:
        engine = get_engine()
        init_db(engine)
        session = get_session(engine)

    try:
        # --- 1. Find the best Mode-A run ---
        best_run_id = _find_best_mode_a_run_id(session)
        if best_run_id is None:
            raise ValueError("No Mode-A non-baseline model run found in the database.")

        # --- 2. Load the bundle ---
        bundle = load_model(run_id=best_run_id, session=session)

        # --- 3. Load station metadata ---
        station = session.get(Station, station_id)
        station_name = station.station_name if station else station_id

        # --- 4. Pull recent daily data ---
        daily = _load_daily_for_station(station_id, session, n_days=45)

        if daily.empty:
            raise ValueError(f"No data found for station {station_id!r}")

        latest_date = daily["date"].max()
        target_date = (latest_date + pd.Timedelta(days=1)).date()
        issued_at = datetime.now().replace(microsecond=0)

        # --- 5. Build feature vector for latest day t ---
        feat_vec = _build_feature_row_for_latest_day(daily, bundle, config)

        # --- 6. Predict ---
        X = pd.DataFrame([feat_vec[bundle["feature_names"]].to_dict()])
        pm25_pred_raw = float(bundle["estimator"].predict(X)[0])
        pm25_pred = round(max(pm25_pred_raw, 0.0), 2)

        # --- 7. Derive AQI + category + color + health ---
        aqi_pred, category = _pm25_to_aqi_and_category(pm25_pred, config)
        color, health = _category_color_health(category, config)

        # --- 8. Persist to live_forecasts ---
        _persist_forecast(
            session, station_id, issued_at, target_date,
            pm25_pred, aqi_pred, category, best_run_id,
        )
        if _own_session:
            session.commit()

        # --- 9. Return the contract dict ---
        return {
            "station_id": station_id,
            "station_name": station_name,
            "issued_at": issued_at,
            "target_date": target_date,
            "pm25_pred": pm25_pred,
            "aqi_pred": aqi_pred,
            "category": category,
            "color": color,
            "health": health,
            "run_id": best_run_id,
            "model_name": bundle.get("model_name"),
        }

    finally:
        if _own_session:
            session.close()
