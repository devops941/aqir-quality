"""Ingestion and validation layer for the Air Quality Forecasting project.

Provides loaders (CSV → standard DataFrame) and validators that raise
``ValueError`` on schema violations, wrong dtypes, or impossible physical
values.  All public functions are listed in docs/INTERFACES.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG

# ---------------------------------------------------------------------------
# Column contracts (must match docs/INTERFACES.md exactly)
# ---------------------------------------------------------------------------

_AQ_REQUIRED_COLS = [
    "station_id", "station_name", "city", "lat", "lon",
    "timestamp", "pm25", "pm10", "no2", "so2", "co", "o3",
]

_WEATHER_REQUIRED_COLS = [
    "station_id", "timestamp",
    "temp_c", "humidity_pct", "wind_speed_ms", "wind_dir_deg",
    "pressure_hpa", "precip_mm", "cloud_pct", "blh_m",
]

_EVENTS_REQUIRED_COLS = ["date", "event_name", "flag"]

# Physical sanity bounds for validation
_AQ_BOUNDS: dict[str, tuple[float, float]] = {
    "pm25": (0.0, 10_000.0),
    "pm10": (0.0, 10_000.0),
    "no2":  (0.0, 10_000.0),
    "so2":  (0.0, 10_000.0),
    "co":   (0.0, 1_000.0),   # mg/m³
    "o3":   (0.0, 10_000.0),
    "lat":  (-90.0, 90.0),
    "lon":  (-180.0, 180.0),
}

_WEATHER_BOUNDS: dict[str, tuple[float, float]] = {
    "temp_c":        (-80.0, 60.0),
    "humidity_pct":  (0.0,   100.0),
    "wind_speed_ms": (0.0,   100.0),
    "wind_dir_deg":  (0.0,   360.0),
    "pressure_hpa":  (800.0, 1100.0),
    "precip_mm":     (0.0,   1_000.0),
    "cloud_pct":     (0.0,   100.0),
    "blh_m":         (0.0,   10_000.0),
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_air_quality(path: str | Path) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame in the standard air_quality schema.

    The file must contain at least the required columns (order does not matter).
    Timestamps are parsed but *not* localised here—use ``parse_timestamps``
    separately when you need a specific timezone.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: station_id, station_name, city, lat, lon, timestamp,
        pm25, pm10, no2, so2, co, o3
    """
    df = pd.read_csv(
        Path(path),
        parse_dates=["timestamp"],
    )
    df = _reorder_columns(df, _AQ_REQUIRED_COLS)
    return df


def load_weather(path: str | Path) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame in the standard weather schema.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: station_id, timestamp, temp_c, humidity_pct, wind_speed_ms,
        wind_dir_deg, pressure_hpa, precip_mm, cloud_pct, blh_m
    """
    df = pd.read_csv(
        Path(path),
        parse_dates=["timestamp"],
    )
    df = _reorder_columns(df, _WEATHER_REQUIRED_COLS)
    return df


def load_events(path: str | Path) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame in the standard events schema.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: date (python date), event_name (str), flag (int 0/1)
    """
    df = pd.read_csv(Path(path), parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["flag"] = df["flag"].astype(int)
    df = _reorder_columns(df, _EVENTS_REQUIRED_COLS)
    return df


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_air_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an air_quality DataFrame; raises ``ValueError`` on failure.

    Checks performed
    ----------------
    1. All required columns are present.
    2. Numeric pollutant and coordinate columns have numeric dtype.
    3. Pollutant and coordinate values are within physically plausible bounds
       (NaN values in these columns are allowed—they represent missing data).

    Returns the original DataFrame if validation passes.
    """
    _check_required_columns(df, _AQ_REQUIRED_COLS)
    _check_numeric_columns(df, ["pm25", "pm10", "no2", "so2", "co", "o3", "lat", "lon"])
    _check_bounds(df, _AQ_BOUNDS)
    _check_timestamp_column(df, "timestamp")
    return df


def validate_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a weather DataFrame; raises ``ValueError`` on failure.

    Checks performed
    ----------------
    1. All required columns are present.
    2. Numeric weather columns have numeric dtype.
    3. Values within physically plausible bounds (NaN allowed for missing).

    Returns the original DataFrame if validation passes.
    """
    _check_required_columns(df, _WEATHER_REQUIRED_COLS)
    numeric_cols = [
        "temp_c", "humidity_pct", "wind_speed_ms", "wind_dir_deg",
        "pressure_hpa", "precip_mm", "cloud_pct", "blh_m",
    ]
    _check_numeric_columns(df, numeric_cols)
    _check_bounds(df, _WEATHER_BOUNDS)
    _check_timestamp_column(df, "timestamp")
    return df


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def parse_timestamps(
    df: pd.DataFrame,
    col: str = "timestamp",
    tz: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """Parse and optionally localise/convert a timestamp column in-place.

    If the column is already tz-aware it is converted to ``tz``; if it is
    tz-naive it is *localised* to ``tz``.  The resulting column is then
    converted back to tz-naive (IST) so the rest of the pipeline works with
    naive timestamps, matching the standard schema.

    Parameters
    ----------
    df:
        Input DataFrame.  Modified in-place and returned.
    col:
        Name of the datetime column to parse.
    tz:
        Target timezone (default ``"Asia/Kolkata"`` = IST).

    Returns
    -------
    pd.DataFrame
        The same DataFrame with the timestamp column updated.
    """
    df = df.copy()
    s = pd.to_datetime(df[col])
    if s.dt.tz is not None:
        # Already tz-aware: convert to target tz then strip
        s = s.dt.tz_convert(tz).dt.tz_localize(None)
    else:
        # Tz-naive: assume UTC, convert to IST, then strip
        s = s.dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
    df[col] = s
    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _reorder_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Return df with required columns first (preserving any extra columns)."""
    extra = [c for c in df.columns if c not in required]
    ordered = required + extra
    present = [c for c in ordered if c in df.columns]
    return df[present]


def _check_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )


def _check_numeric_columns(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Column '{col}' must have a numeric dtype, "
                f"got {df[col].dtype}."
            )


def _check_bounds(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> None:
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        # Only test non-NaN values
        valid = series.dropna()
        if (valid < lo).any():
            bad_vals = valid[valid < lo].values[:5]
            raise ValueError(
                f"Column '{col}' has values below physical minimum {lo}: "
                f"{bad_vals}"
            )
        if (valid > hi).any():
            bad_vals = valid[valid > hi].values[:5]
            raise ValueError(
                f"Column '{col}' has values above physical maximum {hi}: "
                f"{bad_vals}"
            )


def _check_timestamp_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        try:
            pd.to_datetime(df[col])
        except Exception as exc:
            raise ValueError(
                f"Column '{col}' cannot be parsed as datetime: {exc}"
            ) from exc
