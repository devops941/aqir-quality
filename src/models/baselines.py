"""Baseline models for the Air Quality Forecasting project (Phase 5).

Provides two simple baselines:

* **Persistence** — predict tomorrow's PM2.5 as today's PM2.5
  (uses the ``pm25_lag0`` column produced by the feature table).
* **Seasonal naive** — predict tomorrow's PM2.5 as PM2.5 from the
  same weekday last week (uses ``pm25_lag7``).

These baselines require NO training; they read directly from the
lag columns that are already present in the feature table.

Public API (binding — see docs/INTERFACES.md)::

    persistence(feat, split="test") -> pd.DataFrame
    seasonal_naive(feat, split="test") -> pd.DataFrame
    run_baseline(name, feat, split="test") -> pd.DataFrame
"""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Column names (match build_feature_table output)
# ---------------------------------------------------------------------------

_PERSISTENCE_COL = "pm25_lag0"   # PM2.5 at day t  → prediction for t+1
_SEASONAL_COL = "pm25_lag7"      # PM2.5 at day t-7 (same weekday last week)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def persistence(
    feat: pd.DataFrame,
    split: str = "test",
) -> pd.DataFrame:
    """Persistence baseline: y_hat(t+1) = pm25(t) = ``pm25_lag0``.

    Parameters
    ----------
    feat:
        Feature table produced by ``build_feature_table``, containing at
        minimum: ``station_id``, ``date``, ``pm25_lag0``,
        ``target_pm25_next``, ``split``.
    split:
        Which split to predict on (``"train"``, ``"val"``, or ``"test"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``station_id``, ``target_date``, ``y_true``, ``y_pred``.
        ``target_date`` is ``date + 1 day`` (the day being predicted).
    """
    subset = feat[feat["split"] == split].copy()
    return pd.DataFrame({
        "station_id": subset["station_id"].values,
        "target_date": (pd.to_datetime(subset["date"]) + pd.Timedelta(days=1)).dt.normalize().values,
        "y_true": subset["target_pm25_next"].values,
        "y_pred": subset[_PERSISTENCE_COL].values,
    }).reset_index(drop=True)


def seasonal_naive(
    feat: pd.DataFrame,
    split: str = "test",
) -> pd.DataFrame:
    """Seasonal-naive baseline: y_hat(t+1) = pm25(t-7) = ``pm25_lag7``.

    Uses PM2.5 from the same weekday last week as the forecast for the
    next day.

    Parameters
    ----------
    feat:
        Feature table produced by ``build_feature_table``, containing at
        minimum: ``station_id``, ``date``, ``pm25_lag7``,
        ``target_pm25_next``, ``split``.
    split:
        Which split to predict on.

    Returns
    -------
    pd.DataFrame
        Columns: ``station_id``, ``target_date``, ``y_true``, ``y_pred``.
    """
    subset = feat[feat["split"] == split].copy()
    return pd.DataFrame({
        "station_id": subset["station_id"].values,
        "target_date": (pd.to_datetime(subset["date"]) + pd.Timedelta(days=1)).dt.normalize().values,
        "y_true": subset["target_pm25_next"].values,
        "y_pred": subset[_SEASONAL_COL].values,
    }).reset_index(drop=True)


def run_baseline(
    name: str,
    feat: pd.DataFrame,
    split: str = "test",
) -> pd.DataFrame:
    """Run a named baseline and return its predictions DataFrame.

    Parameters
    ----------
    name:
        ``"persistence"`` or ``"seasonal_naive"``.
    feat:
        Feature table produced by ``build_feature_table``.
    split:
        Which split to predict on.

    Returns
    -------
    pd.DataFrame
        Columns: ``station_id``, ``target_date``, ``y_true``, ``y_pred``.

    Raises
    ------
    ValueError
        If *name* is not a recognised baseline.
    """
    if name == "persistence":
        return persistence(feat, split=split)
    if name == "seasonal_naive":
        return seasonal_naive(feat, split=split)
    raise ValueError(
        f"Unknown baseline {name!r}. "
        "Supported baselines: 'persistence', 'seasonal_naive'."
    )
