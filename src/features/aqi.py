"""CPCB Indian AQI calculation (Phase 3).

Implements the National Air Quality Index (NAQI) as published by the Central
Pollution Control Board of India (2014).  All breakpoints are loaded from
``config.yaml`` so they can be revised without touching this module.

Public API (binding — see docs/INTERFACES.md):
    compute_subindex(pollutant, concentration, config) -> float | nan
    compute_aqi(pollutants, config) -> tuple[float | None, str | None]
    aqi_category(aqi_value, config) -> str | None
    add_aqi(df, config) -> pd.DataFrame
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.config import CONFIG

# Pollutant columns recognised by add_aqi (order does not matter).
_POLLUTANT_COLS = ["pm25", "pm10", "no2", "so2", "co", "o3"]

# At least one of these must be present for a valid AQI reading.
_REQUIRED_PM = {"pm25", "pm10"}


def compute_subindex(
    pollutant: str,
    concentration: float,
    config: dict[str, Any] = CONFIG,
) -> float:
    """Compute the CPCB sub-index for a single pollutant.

    Uses linear interpolation between the breakpoint band that contains
    *concentration*::

        I = (I_hi - I_lo) / (C_hi - C_lo) * (C - C_lo) + I_lo

    Parameters
    ----------
    pollutant:
        Pollutant key matching a key in ``config['aqi']['breakpoints']``
        (e.g. ``"pm25"``, ``"co"``).
    concentration:
        Measured concentration in the units expected by the breakpoint table
        (µg/m³ for most pollutants; mg/m³ for CO).
    config:
        Loaded configuration dict (defaults to the module-level singleton).

    Returns
    -------
    float
        The sub-index value, clamped to 500 when the concentration exceeds
        the top breakpoint band's C_high.  Returns ``float('nan')`` when:
        * *concentration* is NaN or negative, or
        * *pollutant* has no breakpoint table in the config.
    """
    # Guard: invalid concentration values.
    try:
        c = float(concentration)
    except (TypeError, ValueError):
        return float("nan")

    if math.isnan(c) or c < 0:
        return float("nan")

    # Look up the breakpoint table.
    breakpoints: list[list[float]] | None = (
        config.get("aqi", {}).get("breakpoints", {}).get(pollutant)
    )
    if not breakpoints:
        return float("nan")

    # Scan bands from lowest to highest; take the first band where C_lo ≤ C ≤ C_hi.
    top_band = None
    for band in breakpoints:
        c_lo, c_hi, i_lo, i_hi = float(band[0]), float(band[1]), float(band[2]), float(band[3])
        if c_lo <= c <= c_hi:
            # Linear interpolation.
            if c_hi == c_lo:
                return float(i_lo)
            return (i_hi - i_lo) / (c_hi - c_lo) * (c - c_lo) + i_lo
        # Track the highest band so we can clamp if C exceeds it.
        if top_band is None or c_hi > top_band[1]:
            top_band = (c_lo, c_hi, i_lo, i_hi)

    # Concentration exceeds the top band's C_high — clamp to 500.
    return 500.0


def aqi_category(
    aqi_value: float,
    config: dict[str, Any] = CONFIG,
) -> str | None:
    """Map an AQI numeric value to its CPCB category name.

    Parameters
    ----------
    aqi_value:
        Numeric AQI (0–500).
    config:
        Loaded configuration dict.

    Returns
    -------
    str | None
        Category name (e.g. ``"Good"``, ``"Severe"``) or ``None`` if
        *aqi_value* is NaN or falls outside all defined category ranges.
    """
    try:
        v = float(aqi_value)
    except (TypeError, ValueError):
        return None

    if math.isnan(v):
        return None

    for cat in config.get("aqi", {}).get("categories", []):
        if cat["low"] <= v <= cat["high"]:
            return cat["name"]

    return None


def compute_aqi(
    pollutants: dict[str, float],
    config: dict[str, Any] = CONFIG,
) -> tuple[float | None, str | None]:
    """Compute the overall Indian AQI from a dict of pollutant concentrations.

    The AQI equals the **maximum** of the computable sub-indices.  The result
    is considered valid only when:

    1. At least ``config['aqi']['min_pollutants']`` sub-indices are computable
       (i.e. not NaN), *and*
    2. At least one of ``pm25`` or ``pm10`` is among those computable sub-indices.

    Parameters
    ----------
    pollutants:
        Mapping of pollutant name → concentration, e.g.
        ``{"pm25": 45.0, "pm10": 80.0, "no2": 55.0}``.
    config:
        Loaded configuration dict.

    Returns
    -------
    tuple[float | None, str | None]
        ``(aqi_value, category)`` when valid; ``(None, None)`` otherwise.
    """
    min_p: int = config.get("aqi", {}).get("min_pollutants", 3)

    sub_indices: dict[str, float] = {}
    for pollutant, concentration in pollutants.items():
        si = compute_subindex(pollutant, concentration, config)
        if not math.isnan(si):
            sub_indices[pollutant] = si

    # Validity checks.
    if len(sub_indices) < min_p:
        return (None, None)

    if not _REQUIRED_PM.intersection(sub_indices.keys()):
        return (None, None)

    aqi_val = max(sub_indices.values())
    category = aqi_category(aqi_val, config)
    return (aqi_val, category)


def add_aqi(
    df: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
) -> pd.DataFrame:
    """Add ``aqi`` and ``aqi_category`` columns to a copy of *df*.

    Applies :func:`compute_aqi` row-wise over whichever pollutant columns
    (``pm25``, ``pm10``, ``no2``, ``so2``, ``co``, ``o3``) are present in
    *df*.  Rows that do not meet the validity criteria will have ``NaN``/
    ``None`` in the new columns.

    Parameters
    ----------
    df:
        Input DataFrame (not modified — a copy is returned).
    config:
        Loaded configuration dict.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the original columns plus ``aqi`` (float) and
        ``aqi_category`` (object/str).
    """
    out = df.copy()

    # Only consider columns that are both in the standard list and present.
    present_cols = [c for c in _POLLUTANT_COLS if c in out.columns]

    def _row_aqi(row: pd.Series) -> pd.Series:
        poll_dict = {
            col: row[col]
            for col in present_cols
            if pd.notna(row[col])
        }
        aqi_val, cat = compute_aqi(poll_dict, config)
        return pd.Series({"aqi": aqi_val, "aqi_category": cat})

    if present_cols:
        result = out.apply(_row_aqi, axis=1)
        out["aqi"] = result["aqi"].astype(float)
        out["aqi_category"] = result["aqi_category"]
    else:
        out["aqi"] = float("nan")
        out["aqi_category"] = None

    return out
