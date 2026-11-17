"""Cleaning, merging and resampling pipeline for air quality and weather data.

Implements Section 6.1 of the project spec and respects all thresholds defined
in ``config['cleaning']``.  All public functions are listed in
docs/INTERFACES.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_air_quality(
    df: pd.DataFrame,
    config: dict = CONFIG,
) -> pd.DataFrame:
    """Clean an air_quality DataFrame according to the cleaning config.

    Steps (in order)
    ----------------
    1. Negative / impossible values (pm < pm_min or pm > pm_max) → NaN.
    2. Flag rows where PM2.5 > PM10 × pm25_over_pm10_ratio with a boolean
       ``pm25_over_pm10_flag`` column (the values are *not* set to NaN here;
       they are still suspicious but not guaranteed impossible).
    3. Remove stuck-sensor runs: if pm25 is identical for more than
       ``stuck_hours`` consecutive hours, set those values to NaN.
    4. Interpolate gaps ≤ ``interpolate_max_hours`` hours using linear
       interpolation; mark interpolated rows with ``is_imputed = 1``.
    5. Drop stations where the fraction of missing pm25 (after interpolation)
       exceeds ``drop_station_missing_pct`` %.

    Parameters
    ----------
    df:
        Standard air_quality DataFrame (must include station_id and timestamp).
    config:
        Project configuration dict.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with an additional ``is_imputed`` (int 0/1) column.
        Row order and index are reset.
    """
    cfg = config["cleaning"]
    pm_min: float = cfg["pm_min"]
    pm_max: float = cfg["pm_max"]
    ratio: float = cfg["pm25_over_pm10_ratio"]
    stuck_hours: int = cfg["stuck_hours"]
    interp_max: int = cfg["interpolate_max_hours"]
    drop_pct: float = cfg["drop_station_missing_pct"]

    pm_cols = ["pm25", "pm10", "no2", "so2", "co", "o3"]

    df = df.copy()

    # Step 1: Negatives / impossible PM values → NaN
    for col in ["pm25", "pm10"]:
        if col in df.columns:
            mask_neg = df[col] < pm_min
            mask_high = df[col] > pm_max
            df.loc[mask_neg | mask_high, col] = np.nan

    # Also clip other pollutants to non-negative
    for col in ["no2", "so2", "co", "o3"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Step 2: Flag PM2.5 > PM10 × ratio
    if "pm25" in df.columns and "pm10" in df.columns:
        df["pm25_over_pm10_flag"] = (
            (df["pm25"] > df["pm10"] * ratio)
            & df["pm25"].notna()
            & df["pm10"].notna()
        ).astype(int)
    else:
        df["pm25_over_pm10_flag"] = 0

    # Step 3: Remove stuck-sensor runs per station
    if "station_id" in df.columns:
        df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
        df = _remove_stuck_sensors(df, "pm25", stuck_hours)
    else:
        df = _remove_stuck_sensors(df, "pm25", stuck_hours)

    # Mark positions that are still NaN (before interpolation)
    if "pm25" in df.columns:
        was_missing_before = df["pm25"].isna()
    else:
        was_missing_before = pd.Series(False, index=df.index)

    # Step 4: Interpolate short gaps per station
    if "station_id" in df.columns:
        df = _interpolate_per_station(df, pm_cols, interp_max)
    else:
        for col in pm_cols:
            if col in df.columns:
                df[col] = _interpolate_series(df[col], interp_max)

    # Mark is_imputed: was NaN before interp but not NaN after
    if "pm25" in df.columns:
        now_filled = df["pm25"].notna() & was_missing_before
    else:
        now_filled = pd.Series(False, index=df.index)
    df["is_imputed"] = now_filled.astype(int)

    # Step 5: Drop stations with too much missing pm25
    if "station_id" in df.columns and "pm25" in df.columns:
        def _missing_pct(g: pd.DataFrame) -> float:
            return g["pm25"].isna().mean() * 100.0

        station_missing = df.groupby("station_id").apply(_missing_pct)
        keep_stations = station_missing[station_missing <= drop_pct].index
        df = df[df["station_id"].isin(keep_stations)].reset_index(drop=True)

    return df


def clean_weather(
    df: pd.DataFrame,
    config: dict = CONFIG,
) -> pd.DataFrame:
    """Clean a weather DataFrame: enforce sane bounds and interpolate short gaps.

    Physical bounds used
    --------------------
    temp_c: [-80, 60]  humidity_pct: [0, 100]  wind_speed_ms: [0, 100]
    wind_dir_deg: [0, 360]  pressure_hpa: [800, 1100]  precip_mm: [0, 1000]
    cloud_pct: [0, 100]  blh_m: [0, 10 000]

    Gaps ≤ ``interpolate_max_hours`` hours are linearly interpolated.
    """
    cfg = config["cleaning"]
    interp_max: int = cfg["interpolate_max_hours"]

    df = df.copy()

    bounds = {
        "temp_c":        (-80.0,  60.0),
        "humidity_pct":  (0.0,   100.0),
        "wind_speed_ms": (0.0,   100.0),
        "wind_dir_deg":  (0.0,   360.0),
        "pressure_hpa":  (800.0, 1100.0),
        "precip_mm":     (0.0,  1_000.0),
        "cloud_pct":     (0.0,   100.0),
        "blh_m":         (0.0, 10_000.0),
    }

    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    weather_cols = [
        "temp_c", "humidity_pct", "wind_speed_ms", "wind_dir_deg",
        "pressure_hpa", "precip_mm", "cloud_pct", "blh_m",
    ]
    if "station_id" in df.columns:
        df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
        df = _interpolate_per_station(df, weather_cols, interp_max)
    else:
        for col in weather_cols:
            if col in df.columns:
                df[col] = _interpolate_series(df[col], interp_max)

    return df


def merge_aq_weather(
    aq: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Merge air quality and weather DataFrames on (station_id, timestamp).

    Both DataFrames must have ``station_id`` and ``timestamp`` columns.
    The join is a left join: all AQ rows are kept; weather values are NaN
    where there is no matching weather reading.

    Parameters
    ----------
    aq:
        Cleaned air quality DataFrame.
    weather:
        Cleaned weather DataFrame.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all AQ columns plus weather columns.
    """
    # Drop duplicate weather columns that also exist in aq (beyond the join keys)
    weather_extra = [
        c for c in weather.columns
        if c not in {"station_id", "timestamp"}
    ]
    merged = aq.merge(
        weather[["station_id", "timestamp"] + weather_extra],
        on=["station_id", "timestamp"],
        how="left",
    )
    return merged.reset_index(drop=True)


def resample_daily(
    df: pd.DataFrame,
    config: dict = CONFIG,
) -> pd.DataFrame:
    """Resample an hourly merged DataFrame to daily aggregates.

    Aggregation rules
    -----------------
    - Pollutants (pm25, pm10, no2, so2, o3, aqi): daily **mean**.
    - CO (mg/m³): daily mean.
    - Precipitation (precip_mm): daily **sum**.
    - All other numeric columns: daily mean.
    - ``is_imputed``: daily max (1 if any hour was imputed).
    - ``aqi_category``: daily mode (most common category).

    The output has one row per (station_id × date) and contains a ``date``
    column (not a timestamp) so it matches the daily-merged contract in
    docs/INTERFACES.md.

    Parameters
    ----------
    df:
        Hourly merged DataFrame with at least station_id and timestamp.
    config:
        Project configuration dict.

    Returns
    -------
    pd.DataFrame
        Daily DataFrame with columns: station_id, date, plus all numeric
        columns aggregated as above.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()

    # Identify grouping and aggregation columns
    group_cols = ["station_id", "date"]

    # Build aggregation dict
    agg_spec: dict = {}

    # Station metadata columns (keep first value per group)
    meta_cols = ["station_name", "city", "lat", "lon"]
    for col in meta_cols:
        if col in df.columns:
            agg_spec[col] = "first"

    # Pollutant means
    pollutant_mean = ["pm25", "pm10", "no2", "so2", "co", "o3", "aqi"]
    for col in pollutant_mean:
        if col in df.columns:
            agg_spec[col] = "mean"

    # Precipitation: sum
    if "precip_mm" in df.columns:
        agg_spec["precip_mm"] = "sum"

    # Weather: mean
    weather_mean_cols = [
        "temp_c", "humidity_pct", "wind_speed_ms", "wind_dir_deg",
        "pressure_hpa", "cloud_pct", "blh_m",
    ]
    for col in weather_mean_cols:
        if col in df.columns:
            agg_spec[col] = "mean"

    # is_imputed: max
    if "is_imputed" in df.columns:
        agg_spec["is_imputed"] = "max"

    # aqi_category: mode
    if "aqi_category" in df.columns:
        agg_spec["aqi_category"] = _series_mode

    # Flags: max
    if "pm25_over_pm10_flag" in df.columns:
        agg_spec["pm25_over_pm10_flag"] = "max"

    daily = df.groupby(group_cols, as_index=False).agg(agg_spec)
    daily["date"] = daily["date"].dt.date

    return daily.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _series_mode(s: pd.Series) -> object:
    """Return the most frequent non-NaN value, or NaN if all are missing."""
    vals = s.dropna()
    if vals.empty:
        return np.nan
    return vals.mode().iloc[0]


def _interpolate_series(
    s: pd.Series,
    max_gap: int,
) -> pd.Series:
    """Linearly interpolate gaps up to ``max_gap`` consecutive NaNs."""
    # interpolate with limit fills gaps up to that length
    return s.interpolate(method="linear", limit=max_gap, limit_direction="forward")


def _interpolate_per_station(
    df: pd.DataFrame,
    cols: list[str],
    max_gap: int,
) -> pd.DataFrame:
    """Interpolate each column per station independently."""
    # Sort is assumed to have been done by the caller
    result_parts: list[pd.DataFrame] = []
    for _sid, group in df.groupby("station_id", sort=False):
        group = group.copy()
        for col in cols:
            if col in group.columns:
                group[col] = _interpolate_series(group[col], max_gap)
        result_parts.append(group)
    return pd.concat(result_parts, ignore_index=True)


def _remove_stuck_sensors(
    df: pd.DataFrame,
    col: str,
    stuck_hours: int,
) -> pd.DataFrame:
    """Set ``col`` to NaN where the same value persists for > stuck_hours hours.

    Works per station_id if that column is present.  Assumes the DataFrame is
    already sorted by (station_id, timestamp).
    """
    if col not in df.columns:
        return df

    df = df.copy()

    def _mask_stuck(series: pd.Series) -> pd.Series:
        """Return a boolean mask: True where the value is part of a stuck run."""
        s = series.copy()
        # Consecutive group IDs: increment whenever the value changes (NaN counts
        # as its own value—don't want to merge NaN runs with adjacent values)
        changed = s.ne(s.shift()) | (s.isna() & ~s.shift().isna()) | (~s.isna() & s.shift().isna())
        group_id = changed.cumsum()
        run_len = group_id.map(group_id.value_counts())
        # Stuck = same non-NaN value for > stuck_hours consecutive rows
        stuck_mask = (run_len > stuck_hours) & s.notna()
        return stuck_mask

    if "station_id" in df.columns:
        for sid, grp in df.groupby("station_id", sort=False):
            stuck = _mask_stuck(grp[col])
            df.loc[grp.index[stuck], col] = np.nan
    else:
        stuck = _mask_stuck(df[col])
        df.loc[stuck, col] = np.nan

    return df
