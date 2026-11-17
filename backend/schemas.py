"""Pydantic v2 request and response schemas for the Air Quality Forecasting API.

Every schema matches the exact JSON shapes defined in docs/API_CONTRACT.md.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# GET /api/stations
# ---------------------------------------------------------------------------

class StationSchema(BaseModel):
    """Single station row."""

    station_id: str
    station_name: Optional[str] = None
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# GET /api/history
# ---------------------------------------------------------------------------

class HistoryRow(BaseModel):
    """One daily observation row in the history response."""

    date: date
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    co: Optional[float] = None
    o3: Optional[float] = None
    aqi: Optional[float] = None
    aqi_category: Optional[str] = None
    temp_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    wind_dir_deg: Optional[float] = None
    pressure_hpa: Optional[float] = None
    precip_mm: Optional[float] = None
    cloud_pct: Optional[float] = None


class HistoryResponse(BaseModel):
    """Response for GET /api/history."""

    station_id: str
    rows: List[HistoryRow]


# ---------------------------------------------------------------------------
# GET /api/forecast
# ---------------------------------------------------------------------------

class ForecastResponse(BaseModel):
    """Next-day forecast from the best model."""

    station_id: str
    station_name: Optional[str] = None
    issued_at: datetime
    target_date: date
    pm25_pred: Optional[float] = None
    aqi_pred: Optional[float] = None
    category: Optional[str] = None
    color: Optional[str] = None
    health: Optional[str] = None
    run_id: Optional[int] = None
    model_name: Optional[str] = None


# ---------------------------------------------------------------------------
# GET /api/predictions
# ---------------------------------------------------------------------------

class PredictionRow(BaseModel):
    """One predicted-vs-actual row."""

    target_date: date
    y_true: Optional[float] = None
    y_pred: Optional[float] = None
    true_category: Optional[str] = None
    pred_category: Optional[str] = None

    model_config = {"from_attributes": True}


class PredictionsResponse(BaseModel):
    """Response for GET /api/predictions."""

    run_id: int
    station_id: str
    rows: List[PredictionRow]


# ---------------------------------------------------------------------------
# GET /api/models
# ---------------------------------------------------------------------------

class ModelRunSchema(BaseModel):
    """Model run row with flattened overall metrics."""

    run_id: int
    model_name: str
    feature_mode: Optional[str] = None
    trained_at: Optional[datetime] = None
    is_best: bool
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    skill_vs_persistence: Optional[float] = None
    category_accuracy: Optional[float] = None
    event_precision: Optional[float] = None
    event_recall: Optional[float] = None
    event_f1: Optional[float] = None


# ---------------------------------------------------------------------------
# GET /api/metrics
# ---------------------------------------------------------------------------

class MetricSchema(BaseModel):
    """One metric scope row."""

    run_id: int
    scope: str
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    skill_vs_persistence: Optional[float] = None
    category_accuracy: Optional[float] = None
    event_precision: Optional[float] = None
    event_recall: Optional[float] = None
    event_f1: Optional[float] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# GET /api/feature-importance
# ---------------------------------------------------------------------------

class FeatureImportanceSchema(BaseModel):
    """Feature importance row."""

    feature: str
    impurity_importance: Optional[float] = None
    permutation_importance: Optional[float] = None
    permutation_std: Optional[float] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# POST /api/predict  (what-if)
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """What-if prediction request.

    Required fields are validated for range; optional fields are filled from
    the model bundle's ``feature_defaults``.
    """

    # Required fields
    recent_pm25: float = Field(..., ge=0, description="Current-day PM2.5 (lag-0), µg/m³")
    temp_c: float = Field(..., ge=-20, le=60, description="Temperature in °C")
    humidity_pct: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    wind_speed_ms: float = Field(..., ge=0, le=60, description="Wind speed m/s")
    precip_mm: float = Field(..., ge=0, description="Precipitation mm")

    # Optional
    pm25_lag1: Optional[float] = Field(None, ge=0)
    pm25_lag7: Optional[float] = Field(None, ge=0)
    pm10: Optional[float] = Field(None, ge=0)
    no2: Optional[float] = Field(None, ge=0)
    wind_dir_deg: Optional[float] = Field(None, ge=0, le=360)
    pressure_hpa: Optional[float] = Field(None, ge=800, le=1100)
    cloud_pct: Optional[float] = Field(None, ge=0, le=100)
    month: Optional[int] = Field(None, ge=1, le=12)
    feature_mode: Optional[str] = Field(None, pattern="^[AB]$")


class PredictResponse(BaseModel):
    """What-if prediction result."""

    pm25_pred: float
    aqi_pred: Optional[float] = None
    category: Optional[str] = None
    color: Optional[str] = None
    health: Optional[str] = None


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Health-check response."""

    status: str
    db: bool
    best_run_id: Optional[int] = None


# ---------------------------------------------------------------------------
# POST /api/admin/refresh
# ---------------------------------------------------------------------------

class RefreshResponse(BaseModel):
    """Admin refresh response."""

    status: str
    message: str


# ---------------------------------------------------------------------------
# POST /api/admin/upload
# ---------------------------------------------------------------------------

class UploadAcceptedResponse(BaseModel):
    """202 response for POST /api/admin/upload — training job accepted."""

    status: str
    mode: str
    message: str
    started_at: datetime


# ---------------------------------------------------------------------------
# GET /api/admin/status
# ---------------------------------------------------------------------------

class JobStatusResponse(BaseModel):
    """Response for GET /api/admin/status."""

    state: str                     # idle | running | done | failed
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    best_run_id: Optional[int] = None
    returncode: Optional[int] = None
