"""Evaluation metrics for the Air Quality Forecasting project (Phase 7).

Computes regression, skill, category, event, and scoped metrics for
any model run.  All public functions are listed in docs/INTERFACES.md.

Design notes
------------
* ``regression_metrics`` guards against zero-valued y_true in MAPE by
  skipping those samples (NaN-safe approach).
* ``metrics_by_scope`` returns dicts whose keys match the ``Metric`` ORM
  table columns exactly, plus a "scope" key used as the primary-key partner.
* Season labels come from ``src.features.build.season_of`` so the mapping
  is consistent across the project.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.config import CONFIG
from src.features.aqi import aqi_category, compute_subindex
from src.features.build import season_of


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute standard regression metrics.

    Parameters
    ----------
    y_true:
        Ground-truth values.
    y_pred:
        Predicted values.

    Returns
    -------
    dict
        ``{"mae": float, "rmse": float, "r2": float, "mape": float}``.
        ``mape`` skips samples where ``y_true ≈ 0`` (absolute value < 1e-8)
        to avoid division by zero; if all samples are zero it returns NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE: skip where y_true is effectively zero
    nonzero_mask = np.abs(y_true) >= 1e-8
    if nonzero_mask.sum() == 0:
        mape = float("nan")
    else:
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
            * 100.0
        )

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def skill_score(rmse_model: float, rmse_persistence: float) -> float:
    """Compute the skill score relative to the persistence baseline.

    ``skill = 1 - RMSE_model / RMSE_persistence``

    A positive value means the model beats persistence; 0 means it matches;
    negative means it is worse than persistence.

    Parameters
    ----------
    rmse_model:
        RMSE of the model being evaluated.
    rmse_persistence:
        RMSE of the persistence baseline.

    Returns
    -------
    float
        Skill score in (-∞, 1].  Returns NaN if ``rmse_persistence`` is 0.
    """
    if rmse_persistence == 0.0:
        return float("nan")
    return 1.0 - rmse_model / rmse_persistence


def category_metrics(
    true_cat: pd.Series | list,
    pred_cat: pd.Series | list,
) -> dict[str, Any]:
    """Compute accuracy and confusion matrix for AQI categories.

    Parameters
    ----------
    true_cat:
        Sequence of true category labels.
    pred_cat:
        Sequence of predicted category labels.

    Returns
    -------
    dict
        ``{"accuracy": float, "confusion": {true_label: {pred_label: count}}}``.
    """
    true_arr = np.asarray(true_cat, dtype=object)
    pred_arr = np.asarray(pred_cat, dtype=object)

    # Accuracy
    accuracy = float(np.mean(true_arr == pred_arr))

    # Confusion matrix as a nested dict
    labels = sorted(set(np.concatenate([np.unique(true_arr), np.unique(pred_arr)])))
    cm = confusion_matrix(true_arr, pred_arr, labels=labels)
    confusion: dict[str, dict[str, int]] = {}
    for i, tl in enumerate(labels):
        confusion[str(tl)] = {str(pl): int(cm[i, j]) for j, pl in enumerate(labels)}

    return {"accuracy": accuracy, "confusion": confusion}


def event_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Compute precision, recall, and F1 for high-pollution events.

    A sample is a positive (event) when its value > ``threshold``.

    Parameters
    ----------
    y_true:
        Ground-truth PM2.5 values.
    y_pred:
        Predicted PM2.5 values.
    threshold:
        PM2.5 concentration threshold above which a day is an "event".

    Returns
    -------
    dict
        ``{"precision": float, "recall": float, "f1": float}``.
        Returns 0.0 for all metrics when no positive samples exist.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    true_pos = (y_true > threshold).astype(int)
    pred_pos = (y_pred > threshold).astype(int)

    # zero_division=0 returns 0 when there are no predicted/true positives
    precision = float(precision_score(true_pos, pred_pos, zero_division=0))
    recall = float(recall_score(true_pos, pred_pos, zero_division=0))
    f1 = float(f1_score(true_pos, pred_pos, zero_division=0))

    return {"precision": precision, "recall": recall, "f1": f1}


def _pm25_to_category(pm25_value: float, config: dict[str, Any] = CONFIG) -> str | None:
    """Map a PM2.5 concentration to its CPCB AQI category via pm25 sub-index."""
    si = compute_subindex("pm25", pm25_value, config)
    if math.isnan(si):
        return None
    return aqi_category(si, config)


def metrics_by_scope(
    preds: pd.DataFrame,
    feat: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
) -> list[dict[str, Any]]:
    """Compute metrics at multiple scopes: overall, per-station, per-season.

    Parameters
    ----------
    preds:
        Predictions DataFrame with columns:
        ``station_id, target_date, y_true, y_pred``.
        Optionally also ``true_category, pred_category``.
    feat:
        Feature table (used to look up ``persistence_rmse`` if needed; not
        directly used here — the caller passes ``skill_vs_persistence`` as
        ``persistence_rmse`` into this function).
    config:
        Project configuration dict.

    Returns
    -------
    list[dict]
        Each dict matches the ``Metric`` table columns:
        ``scope, mae, rmse, r2, mape, skill_vs_persistence,
        category_accuracy, event_precision, event_recall, event_f1``.
        The ``skill_vs_persistence`` field is ``None`` here; callers who
        know the persistence RMSE should fill it after calling this function,
        or pass ``persistence_rmse`` via the wrapper in ``train.py``.

    Notes
    -----
    ``target_date`` must be a date/datetime column so that the month can
    be extracted for season derivation.
    """
    threshold = float(config.get("models", {}).get("event_threshold_pm25", 90.0))

    preds = preds.copy()
    preds["target_date"] = pd.to_datetime(preds["target_date"])

    # Derive categories if not already present
    if "pred_category" not in preds.columns:
        preds["pred_category"] = preds["y_pred"].apply(
            lambda v: _pm25_to_category(v, config)
        )
    if "true_category" not in preds.columns:
        preds["true_category"] = preds["y_true"].apply(
            lambda v: _pm25_to_category(v, config)
        )

    # Derive season
    preds["_season"] = preds["target_date"].dt.month.apply(season_of)

    def _scope_metrics(subset: pd.DataFrame, scope: str) -> dict[str, Any]:
        """Build a single scope metric dict."""
        if len(subset) == 0:
            return {
                "scope": scope,
                "mae": None, "rmse": None, "r2": None, "mape": None,
                "skill_vs_persistence": None,
                "category_accuracy": None,
                "event_precision": None, "event_recall": None, "event_f1": None,
            }

        reg = regression_metrics(subset["y_true"].values, subset["y_pred"].values)

        # Category metrics (drop rows with None category)
        cat_subset = subset.dropna(subset=["true_category", "pred_category"])
        if len(cat_subset) >= 2:
            cat = category_metrics(
                cat_subset["true_category"].values,
                cat_subset["pred_category"].values,
            )
            cat_acc = cat["accuracy"]
        else:
            cat_acc = None

        # Event metrics
        ev = event_metrics(subset["y_true"].values, subset["y_pred"].values, threshold)

        return {
            "scope": scope,
            "mae": reg["mae"],
            "rmse": reg["rmse"],
            "r2": reg["r2"],
            "mape": reg["mape"],
            "skill_vs_persistence": None,  # filled by caller
            "category_accuracy": cat_acc,
            "event_precision": ev["precision"],
            "event_recall": ev["recall"],
            "event_f1": ev["f1"],
        }

    results: list[dict[str, Any]] = []

    # Overall
    results.append(_scope_metrics(preds, "overall"))

    # Per station
    for sid, grp in preds.groupby("station_id"):
        results.append(_scope_metrics(grp, f"station:{sid}"))

    # Per season
    for season, grp in preds.groupby("_season"):
        results.append(_scope_metrics(grp, f"season:{season}"))

    return results
