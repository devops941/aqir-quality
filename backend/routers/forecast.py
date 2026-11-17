"""Forecast and what-if prediction endpoints."""
from __future__ import annotations

import math
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.deps import get_config, get_db
from backend.schemas import ForecastResponse, PredictRequest, PredictResponse
from src.config import CONFIG
from src.data.db import Station
from src.features.aqi import aqi_category, compute_subindex
from src.features.build import season_of

router = APIRouter()


def _load_best_mode_a_bundle(session) -> tuple[dict[str, Any], int]:
    """Load the best Mode-A model bundle.  Returns (bundle, run_id)."""
    from sqlalchemy import select
    from src.data.db import Metric, ModelRun
    from src.models.train import load_model

    _NON_BASELINE = {"ridge", "random_forest", "hist_gb"}
    stmt = (
        select(ModelRun, Metric)
        .join(Metric, (ModelRun.run_id == Metric.run_id) & (Metric.scope == "overall"))
        .where(ModelRun.feature_mode == "A")
        .where(ModelRun.model_name.in_(list(_NON_BASELINE)))
        .where(Metric.rmse.isnot(None))
        .order_by(Metric.rmse)
        .limit(1)
    )
    row = session.execute(stmt).first()
    if row is None:
        raise HTTPException(status_code=503, detail="No Mode-A model available.")
    run, _ = row
    bundle = load_model(run_id=run.run_id, session=session)
    return bundle, run.run_id


def _pm25_to_aqi_and_cat(pm25: float, config: dict) -> tuple[float | None, str | None]:
    si = compute_subindex("pm25", pm25, config)
    if math.isnan(si):
        return None, None
    cat = aqi_category(si, config)
    return round(si, 1), cat


def _cat_color_health(cat: str | None, config: dict) -> tuple[str | None, str | None]:
    if cat is None:
        return None, None
    for c in config.get("aqi", {}).get("categories", []):
        if c["name"] == cat:
            return c["color"], c["health"]
    return None, None


@router.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    station_id: str = Query(..., description="Station ID, e.g. DL001"),
    session: Session = Depends(get_db),
    config: dict = Depends(get_config),
) -> ForecastResponse:
    """Return the next-day forecast for a station.

    Calls ``make_live_forecast`` to compute (or refresh) the prediction on
    demand, then returns the result in the API contract shape.
    """
    # Verify station exists
    station = session.get(Station, station_id)
    if station is None:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found.")

    try:
        from src.forecast.live import make_live_forecast
        result = make_live_forecast(station_id, config=config, session=session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}") from exc

    return ForecastResponse(**result)


@router.post("/predict", response_model=PredictResponse)
def what_if_predict(
    body: PredictRequest,
    session: Session = Depends(get_db),
    config: dict = Depends(get_config),
) -> PredictResponse:
    """What-if prediction: assemble a Mode-A feature vector and predict PM2.5.

    Maps the request fields to the model's feature_names, filling absent
    features from the bundle's feature_defaults.  Derives AQI sub-index,
    category, color, and health from the predicted PM2.5.
    """
    import numpy as np
    import pandas as pd

    bundle, _ = _load_best_mode_a_bundle(session)
    defaults: dict[str, float] = bundle["feature_defaults"]
    feat_names: list[str] = bundle["feature_names"]

    # --- Determine calendar month & derived cyclic/season features ---
    month = body.month if body.month is not None else int(pd.Timestamp.now().month)
    season = season_of(month)
    month_sin = float(np.sin(2 * np.pi * month / 12.0))
    month_cos = float(np.cos(2 * np.pi * month / 12.0))

    # Wind sin/cos from wind_dir_deg (use default dir if not provided)
    wind_dir = (
        body.wind_dir_deg
        if body.wind_dir_deg is not None
        else defaults.get("wind_dir_deg", 0.0)
    )
    wind_sin = float(np.sin(np.deg2rad(wind_dir)))
    wind_cos = float(np.cos(np.deg2rad(wind_dir)))

    # --- Build an explicit override mapping ---
    # Map request fields to feature names (the lag-0 convention)
    override: dict[str, float] = {
        # Pollutant lags
        "pm25_lag0": body.recent_pm25,
        "pm25_lag1": body.pm25_lag1 if body.pm25_lag1 is not None else body.recent_pm25,
        "pm25_lag7": body.pm25_lag7 if body.pm25_lag7 is not None else body.recent_pm25,
        # Other pollutant lag-0 (use provided; lag1-7 filled from defaults)
        "pm10_lag0": body.pm10 if body.pm10 is not None else defaults.get("pm10_lag0", 0.0),
        "no2_lag0": body.no2 if body.no2 is not None else defaults.get("no2_lag0", 0.0),
        # Weather
        "temp_c": body.temp_c,
        "humidity_pct": body.humidity_pct,
        "wind_speed_ms": body.wind_speed_ms,
        "pressure_hpa": (
            body.pressure_hpa if body.pressure_hpa is not None
            else defaults.get("pressure_hpa", 1013.0)
        ),
        "precip_mm": body.precip_mm,
        "cloud_pct": (
            body.cloud_pct if body.cloud_pct is not None
            else defaults.get("cloud_pct", 0.0)
        ),
        # Calendar
        "month": float(month),
        "month_sin": month_sin,
        "month_cos": month_cos,
        # Season one-hots
        "season_winter": float(season == "winter"),
        "season_pre-monsoon": float(season == "pre-monsoon"),
        "season_monsoon": float(season == "monsoon"),
        "season_post-monsoon": float(season == "post-monsoon"),
        # Wind encoding
        "wind_sin": wind_sin,
        "wind_cos": wind_cos,
    }

    # Also propagate pm25_lag1 to rolling proxies (the defaults handle the rest)
    # Build rolling stats from provided lags if possible — otherwise use defaults
    # (best-effort: for a what-if we don't have full history, so defaults are fine)
    lag1 = override.get("pm25_lag1", body.recent_pm25)
    lag7 = override.get("pm25_lag7", body.recent_pm25)
    lag0 = body.recent_pm25
    # Simple rolling proxies
    override.setdefault("pm25_roll3_mean", (lag0 + lag1) / 2.0)
    override.setdefault("pm25_roll3_max", max(lag0, lag1))
    override.setdefault("pm25_roll7_mean", (lag0 + lag7) / 2.0)
    override.setdefault("pm25_roll7_max", max(lag0, lag7))

    # --- Assemble the feature row ---
    row = {f: override.get(f, defaults.get(f, 0.0)) for f in feat_names}
    X = pd.DataFrame([row])

    pm25_pred_raw = float(bundle["estimator"].predict(X)[0])
    pm25_pred = round(max(pm25_pred_raw, 0.0), 2)

    aqi_pred, cat = _pm25_to_aqi_and_cat(pm25_pred, config)
    color, health = _cat_color_health(cat, config)

    return PredictResponse(
        pm25_pred=pm25_pred,
        aqi_pred=aqi_pred,
        category=cat,
        color=color,
        health=health,
    )
