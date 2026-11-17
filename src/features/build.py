"""Feature engineering for the Air Quality Forecasting project (Phase 4).

Builds a tidy feature table from the daily-merged contract produced by
``src/data/clean.resample_daily`` + ``src/features/aqi.add_aqi``.

Public API (binding — see docs/INTERFACES.md):
    season_of(month) -> str
    build_feature_table(daily, events, config, mode) -> pd.DataFrame
    assign_splits(df, config) -> pd.DataFrame
    feature_columns(df) -> list[str]

Leakage guarantee
-----------------
* All pollutant lags use ``shift(k)`` with k >= 0 (lag-0 = current day t,
  lag-k = k days before t).  shift() pushes values *forward* in time, so
  the value visible at row t came from t-k.
* Rolling windows use ``.rolling(w).agg()`` without ``min_periods`` tricks —
  the window includes only rows at or before t.
* Mode-B ``<var>_next`` columns are the *only* exception and are clearly
  labelled; they are excluded from Mode A.
* Targets (target_pm25_next, target_aqi_next) are computed with shift(-1)
  within each station (next day's value as the label) — this is the
  prediction target, not an input feature.
* All computations are done per-station (groupby station_id), so station A's
  future values never bleed into station B's features.
* ``dropna`` is called at the end to remove rows that could not form the
  required lag-7 / rolling-7 window, or that have no next-day target.  Any
  row with a NaN in a feature or target column is dropped and documented in
  this docstring so callers are aware.

After ``dropna``, the returned DataFrame has NO NaN values in any feature
or target column.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.config import CONFIG

# ---------------------------------------------------------------------------
# Constants derived from the interface contract
# ---------------------------------------------------------------------------

_LAG_POLLUTANTS: list[str] = ["pm25", "pm10", "no2"]
_LAGS: list[int] = [0, 1, 2, 3, 7]
_ROLLING_WINDOWS: list[int] = [3, 7]
_ROLLING_STATS: list[str] = ["mean", "max", "std"]
_WEATHER_VARS: list[str] = [
    "temp_c", "humidity_pct", "wind_speed_ms", "pressure_hpa", "precip_mm", "cloud_pct"
]

# Columns that are identifiers, targets, or the split label — excluded from
# the feature-column list returned by ``feature_columns``.
_NON_FEATURE_COLS: set[str] = {
    "station_id", "date", "target_pm25_next", "target_aqi_next", "split"
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def season_of(month: int) -> str:
    """Map a calendar month (1-12) to the Indian meteorological season name.

    Parameters
    ----------
    month:
        Integer month (1 = January, 12 = December).

    Returns
    -------
    str
        One of ``"winter"``, ``"pre-monsoon"``, ``"monsoon"``,
        ``"post-monsoon"``.

    Raises
    ------
    ValueError
        If *month* is not in [1, 12].
    """
    if month not in range(1, 13):
        raise ValueError(f"month must be 1-12, got {month!r}")

    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "pre-monsoon"
    if month in (6, 7, 8, 9):
        return "monsoon"
    # 10, 11
    return "post-monsoon"


def build_feature_table(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
    mode: str = "A",
) -> pd.DataFrame:
    """Build the full feature table from the daily-merged contract.

    Per (station_id, date) produces:

    **Pollutant lags** — pm25, pm10, no2 at lags [0, 1, 2, 3, 7]
    ``<p>_lag<k>`` (lag0 = value at t; lag1 = value at t-1, etc.)

    **Rolling stats of pm25** — 3-day and 7-day mean/max/std ending at t
    ``pm25_roll3_mean``, ``pm25_roll3_max``, ``pm25_roll3_std``,
    ``pm25_roll7_mean``, ``pm25_roll7_max``, ``pm25_roll7_std``

    **Weather at t** — temp_c, humidity_pct, wind_speed_ms, pressure_hpa,
    precip_mm, cloud_pct (daily mean, already resampled by resample_daily)

    **Weather change** — ``<var>_chg`` = value(t) - value(t-1) for each
    weather var

    **Wind encoding** — ``wind_sin`` = sin(wind_dir_deg * π/180),
    ``wind_cos`` = cos(wind_dir_deg * π/180). Values in [-1, 1].
    Raw ``wind_dir_deg`` is NOT kept as a feature.

    **Calendar** — ``dow`` (0=Mon…6=Sun), ``month`` (1-12), ``is_weekend``
    (0/1), ``month_sin`` / ``month_cos`` (cyclic), plus one-hot season
    columns: ``season_winter``, ``season_pre-monsoon``, ``season_monsoon``,
    ``season_post-monsoon``.

    **Event flags** — ``is_festival`` (0/1), ``is_crop_burning`` (0/1).
    Crop burning = month in config['features']['crop_burning_months']
    OR an events row with event_name == 'crop_burning'.

    **Mode B only** — ``<var>_next`` for each weather var (observed day t+1
    weather used as a perfect-forecast proxy during training).

    **Targets** — ``target_pm25_next`` (shift(-1) of pm25 within station),
    ``target_aqi_next`` (shift(-1) of aqi within station).

    **Split** — added by :func:`assign_splits`.

    ``dropna`` policy
    -----------------
    After computing all features and targets, any row with a NaN in a
    feature or target column is dropped.  The main sources of NaN rows:

    * The first lag-7 / rolling-7 days per station (window warm-up).
    * The last row per station (no next-day target after shift(-1)).
    * Any station-day where the source daily frame had NaN pollutants
      (missing data not covered by interpolation).

    Parameters
    ----------
    daily:
        Daily-merged DataFrame (one row per station_id × date) containing
        at minimum: station_id, date, pm25, pm10, no2, aqi, temp_c,
        humidity_pct, wind_speed_ms, wind_dir_deg, pressure_hpa,
        precip_mm, cloud_pct.
    events:
        Events DataFrame with columns: date, event_name, flag.
    config:
        Project configuration dict.
    mode:
        ``"A"`` (default) — no future weather features.
        ``"B"`` — additionally adds ``<var>_next`` observed weather columns.

    Returns
    -------
    pd.DataFrame
        Tidy feature table sorted by (station_id, date) with columns:
        station_id, date, <feature cols>, target_pm25_next,
        target_aqi_next, split.  No NaNs in feature/target columns.
    """
    if mode not in ("A", "B"):
        raise ValueError(f"mode must be 'A' or 'B', got {mode!r}")

    feat_cfg = config.get("features", {})
    lag_pollutants: list[str] = feat_cfg.get("lag_pollutants", _LAG_POLLUTANTS)
    lags: list[int] = feat_cfg.get("lags", _LAGS)
    rolling_windows: list[int] = feat_cfg.get("rolling_windows", _ROLLING_WINDOWS)
    rolling_stats: list[str] = feat_cfg.get("rolling_stats", _ROLLING_STATS)
    weather_vars: list[str] = feat_cfg.get("weather_vars", _WEATHER_VARS)
    crop_burning_months: list[int] = feat_cfg.get("crop_burning_months", [10, 11])

    daily = daily.copy()
    # Ensure date column is comparable (some paths leave it as datetime.date)
    daily["date"] = pd.to_datetime(daily["date"])

    # ------------------------------------------------------------------
    # Pre-process events DataFrame
    # ------------------------------------------------------------------
    events = events.copy()
    events["date"] = pd.to_datetime(events["date"])

    # Festival flag: any event that is NOT crop_burning
    festival_dates: set = set(
        events.loc[events["event_name"] != "crop_burning", "date"]
    )

    # Crop-burning flag from events rows
    crop_burning_event_dates: set = set(
        events.loc[events["event_name"] == "crop_burning", "date"]
    )

    # ------------------------------------------------------------------
    # Process per station (preserves leakage guarantee)
    # ------------------------------------------------------------------
    all_parts: list[pd.DataFrame] = []

    for sid, grp in daily.groupby("station_id", sort=True):
        grp = grp.sort_values("date").copy()
        grp = grp.reset_index(drop=True)

        part: dict[str, pd.Series] = {}

        # Station + date identifiers
        part["station_id"] = grp["station_id"]
        part["date"] = grp["date"]

        # --------------------------------------------------------------
        # 1. Pollutant lags
        # --------------------------------------------------------------
        for p in lag_pollutants:
            if p not in grp.columns:
                continue
            for k in lags:
                # shift(k) moves values *forward* by k rows, meaning the
                # value at index i is the original value from index i-k
                # (i.e., k days before t).  lag-0 = current day t.
                part[f"{p}_lag{k}"] = grp[p].shift(k)

        # --------------------------------------------------------------
        # 2. Rolling stats of pm25
        # --------------------------------------------------------------
        if "pm25" in grp.columns:
            for w in rolling_windows:
                for stat in rolling_stats:
                    col_name = f"pm25_roll{w}_{stat}"
                    rolled = grp["pm25"].rolling(window=w, min_periods=w)
                    if stat == "mean":
                        part[col_name] = rolled.mean()
                    elif stat == "max":
                        part[col_name] = rolled.max()
                    elif stat == "std":
                        part[col_name] = rolled.std(ddof=1)

        # --------------------------------------------------------------
        # 3. Weather at t
        # --------------------------------------------------------------
        for var in weather_vars:
            if var in grp.columns:
                part[var] = grp[var]

        # --------------------------------------------------------------
        # 4. Weather change t-1 → t
        # --------------------------------------------------------------
        for var in weather_vars:
            if var in grp.columns:
                part[f"{var}_chg"] = grp[var] - grp[var].shift(1)

        # --------------------------------------------------------------
        # 5. Wind direction → sin/cos encoding
        # --------------------------------------------------------------
        if "wind_dir_deg" in grp.columns:
            rad = np.deg2rad(grp["wind_dir_deg"].to_numpy(dtype=float))
            part["wind_sin"] = pd.Series(np.sin(rad), index=grp.index)
            part["wind_cos"] = pd.Series(np.cos(rad), index=grp.index)

        # --------------------------------------------------------------
        # 6. Calendar features
        # --------------------------------------------------------------
        dates = pd.DatetimeIndex(grp["date"])

        part["dow"] = pd.Series(dates.day_of_week, index=grp.index)
        part["month"] = pd.Series(dates.month, index=grp.index)
        part["is_weekend"] = pd.Series(
            (dates.day_of_week >= 5).astype(int), index=grp.index
        )

        # Cyclic month encoding
        month_arr = dates.month.to_numpy(dtype=float)
        part["month_sin"] = pd.Series(
            np.sin(2 * np.pi * month_arr / 12.0), index=grp.index
        )
        part["month_cos"] = pd.Series(
            np.cos(2 * np.pi * month_arr / 12.0), index=grp.index
        )

        # Season one-hot columns
        seasons = dates.month.map(season_of)
        for s_name in ["winter", "pre-monsoon", "monsoon", "post-monsoon"]:
            part[f"season_{s_name}"] = pd.Series(
                (seasons == s_name).astype(int), index=grp.index
            )

        # --------------------------------------------------------------
        # 7. Event flags
        # --------------------------------------------------------------
        # is_festival
        part["is_festival"] = pd.Series(
            grp["date"].isin(festival_dates).astype(int).values,
            index=grp.index,
        )

        # is_crop_burning: month-based OR events-row-based
        is_crop_burn_month = dates.month.isin(crop_burning_months)
        is_crop_burn_event = grp["date"].isin(crop_burning_event_dates)
        part["is_crop_burning"] = pd.Series(
            (is_crop_burn_month | is_crop_burn_event.values).astype(int),
            index=grp.index,
        )

        # --------------------------------------------------------------
        # 8. Mode B: next-day weather columns
        # --------------------------------------------------------------
        if mode == "B":
            for var in weather_vars:
                if var in grp.columns:
                    part[f"{var}_next"] = grp[var].shift(-1)

        # --------------------------------------------------------------
        # 9. Targets
        # --------------------------------------------------------------
        if "pm25" in grp.columns:
            part["target_pm25_next"] = grp["pm25"].shift(-1)
        if "aqi" in grp.columns:
            part["target_aqi_next"] = grp["aqi"].shift(-1)

        all_parts.append(pd.DataFrame(part, index=grp.index))

    if not all_parts:
        return pd.DataFrame()

    result = pd.concat(all_parts, ignore_index=True)

    # ------------------------------------------------------------------
    # Assign chronological splits (adds 'split' column)
    # ------------------------------------------------------------------
    result = assign_splits(result, config)

    # ------------------------------------------------------------------
    # Drop rows with any NaN in feature/target columns
    # (window warm-up rows, last row per station without a next-day target,
    #  and rows where source data had NaN pollutants/weather)
    # ------------------------------------------------------------------
    target_cols = ["target_pm25_next", "target_aqi_next"]
    # Build feature column list (everything except identifiers/split/targets)
    feat_cols = feature_columns(result)
    drop_subset = feat_cols + target_cols
    # Only drop on columns that actually exist
    drop_subset = [c for c in drop_subset if c in result.columns]
    result = result.dropna(subset=drop_subset).reset_index(drop=True)

    # Ensure correct output ordering: station_id, date, features, targets, split
    id_cols = ["station_id", "date"]
    target_split_cols = ["target_pm25_next", "target_aqi_next", "split"]
    ordered_feat_cols = [
        c for c in result.columns
        if c not in set(id_cols) | set(target_split_cols)
    ]
    final_col_order = id_cols + ordered_feat_cols + target_split_cols
    result = result[[c for c in final_col_order if c in result.columns]]

    return result


def assign_splits(
    df: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
) -> pd.DataFrame:
    """Assign chronological train/val/test splits to a feature DataFrame.

    Uses the SAME date cutoffs for all stations so test dates are strictly
    later than train/val dates.  The split is based purely on the range of
    dates in *df*:

    * earliest 70% of the date span  → ``'train'``
    * next 15%                        → ``'val'``
    * last 15%                        → ``'test'``

    Parameters
    ----------
    df:
        DataFrame with a ``date`` column (datetime or date-like).
    config:
        Project configuration dict.  Uses ``config['models']['split']``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added/overwritten ``'split'`` column.
    """
    split_cfg = config.get("models", {}).get("split", {"train": 0.70, "val": 0.15, "test": 0.15})
    train_frac: float = split_cfg.get("train", 0.70)
    val_frac: float = split_cfg.get("val", 0.15)

    df = df.copy()
    dates = pd.to_datetime(df["date"])
    min_date = dates.min()
    max_date = dates.max()

    # Total span in days (inclusive of both endpoints)
    span = (max_date - min_date).total_seconds()

    # Compute cutoff timestamps
    train_end = min_date + pd.Timedelta(seconds=span * train_frac)
    val_end = min_date + pd.Timedelta(seconds=span * (train_frac + val_frac))

    split_labels = pd.Series("test", index=df.index, dtype=object)
    split_labels[dates <= train_end] = "train"
    split_labels[(dates > train_end) & (dates <= val_end)] = "val"

    df["split"] = split_labels
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (the X matrix columns).

    Excludes ``station_id``, ``date``, ``target_pm25_next``,
    ``target_aqi_next``, and ``split``.

    Parameters
    ----------
    df:
        Feature DataFrame produced by :func:`build_feature_table`.

    Returns
    -------
    list[str]
        Ordered list of feature column names.
    """
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _assert_in_m1_1(series: pd.Series, name: str) -> None:
    """Raise AssertionError if any value in *series* is outside [-1, 1]."""
    mn, mx = series.min(), series.max()
    if mn < -1.0 - 1e-9 or mx > 1.0 + 1e-9:
        raise AssertionError(
            f"{name} has values outside [-1, 1]: min={mn}, max={mx}"
        )
