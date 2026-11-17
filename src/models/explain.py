"""Explainability tools for the Air Quality Forecasting project (Phase 7).

Provides impurity importance, permutation importance, partial dependence
curves, and residual analysis utilities.  All public functions are listed
in docs/INTERFACES.md.

Design notes
------------
* ``impurity_importance`` unwraps sklearn Pipelines to reach the underlying
  estimator's ``feature_importances_`` attribute.
* ``permutation_importance_df`` uses sklearn 1.9's
  ``sklearn.inspection.permutation_importance`` with ``n_jobs=1`` for
  reproducibility.
* ``partial_dependence_curves`` uses ``sklearn.inspection.partial_dependence``
  with ``kind="average"`` (the ICE average = PDP).
* ``residual_analysis`` uses ``statsmodels.stats.diagnostic.acorr_ljungbox``
  for the Ljung-Box test and ``scipy.stats.probplot`` for the QQ data.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.inspection import partial_dependence as sk_partial_dependence
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.pipeline import Pipeline
from statsmodels.stats.diagnostic import acorr_ljungbox


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def impurity_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract impurity (Gini / variance reduction) feature importances.

    Parameters
    ----------
    model:
        A fitted sklearn estimator or Pipeline whose final step exposes a
        ``feature_importances_`` attribute (e.g. RandomForestRegressor,
        HistGradientBoostingRegressor).
    feature_names:
        Ordered list of feature names matching the estimator's input columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``impurity_importance``.
        Sorted descending by ``impurity_importance``.

    Raises
    ------
    AttributeError
        If the (unwrapped) estimator does not have ``feature_importances_``.
    """
    # Unwrap Pipeline to get the actual estimator
    estimator = _unwrap_pipeline(model)

    importances = np.asarray(estimator.feature_importances_, dtype=float)
    df = pd.DataFrame({
        "feature": feature_names,
        "impurity_importance": importances,
    })
    return df.sort_values("impurity_importance", ascending=False).reset_index(drop=True)


def permutation_importance_df(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance on the provided (X, y) split.

    Parameters
    ----------
    model:
        Fitted sklearn estimator or Pipeline.
    X:
        Feature matrix (n_samples × n_features).
    y:
        Target vector.
    feature_names:
        Feature names matching the columns of X.
    n_repeats:
        Number of permutation repeats (default 10).
    random_state:
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``permutation_importance``, ``permutation_std``.
        Sorted descending by ``permutation_importance``.
    """
    result = sk_permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
        scoring="neg_root_mean_squared_error",
    )
    # importances_mean is the mean *decrease* in score (higher = more important)
    df = pd.DataFrame({
        "feature": feature_names,
        "permutation_importance": result.importances_mean,
        "permutation_std": result.importances_std,
    })
    return df.sort_values("permutation_importance", ascending=False).reset_index(drop=True)


def partial_dependence_curves(
    model: Any,
    X: pd.DataFrame,
    features: list[str],
) -> dict[str, dict[str, list]]:
    """Compute partial dependence curves for the specified features.

    Parameters
    ----------
    model:
        Fitted sklearn estimator or Pipeline.
    X:
        Feature DataFrame (used to determine value grid and for the PDP
        computation; must include all features the model was trained on).
    features:
        List of feature names for which to compute PDP curves.
        Default use-case: ``["wind_speed_ms", "humidity_pct", "temp_c"]``.

    Returns
    -------
    dict
        ``{feature_name: {"x": [grid_values], "y": [pd_values]}}``
        Values are plain Python lists (JSON-serialisable).
    """
    result: dict[str, dict[str, list]] = {}

    # Only compute for features that actually exist in X
    available = [f for f in features if f in X.columns]

    for feat in available:
        try:
            pd_result = sk_partial_dependence(
                model,
                X,
                features=[feat],
                kind="average",
                grid_resolution=50,
            )
            # pd_result["average"] shape: (1, n_grid_points)
            # pd_result["grid_values"] is a list of arrays; first element for single feature
            x_vals = pd_result["grid_values"][0].tolist()
            y_vals = pd_result["average"][0].tolist()
            result[feat] = {"x": x_vals, "y": y_vals}
        except Exception:
            # Feature may not be numeric or model may not support PDP — skip
            continue

    return result


def residual_analysis(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    dates: pd.Series | list,
) -> dict[str, Any]:
    """Compute residual diagnostics for a model's test-set predictions.

    Parameters
    ----------
    y_true:
        Ground-truth values.
    y_pred:
        Predicted values.
    dates:
        Date labels (same length as y_true/y_pred), used to identify
        worst-performing days.

    Returns
    -------
    dict with keys:
        ``residuals``: list of float (y_true - y_pred).
        ``ljung_box_p``: float, p-value from Ljung-Box test at lag 10
            (or maximum feasible lag).
        ``qq``: dict with ``"theoretical"`` and ``"sample"`` lists (from
            scipy.stats.probplot).
        ``worst_days``: list of dicts
            ``[{"date": str, "y_true": float, "y_pred": float,
               "abs_error": float}, ...]`` for the top-10 by absolute error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = (y_true - y_pred).tolist()

    # Ljung-Box test — use min(10, n//5) lags to stay feasible
    n = len(residuals)
    lags = min(10, max(1, n // 5))
    lb_result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    ljung_box_p = float(lb_result["lb_pvalue"].iloc[-1])

    # QQ plot data via scipy.stats.probplot (returns (osm, osr), (slope, intercept, r))
    qq_result = stats.probplot(np.asarray(residuals), dist="norm")
    theoretical = qq_result[0][0].tolist()   # theoretical quantiles
    sample = qq_result[0][1].tolist()         # ordered sample values

    # Worst 10 days by absolute error
    abs_errors = np.abs(y_true - y_pred)
    dates_arr = np.asarray(dates, dtype=object)
    top_idx = np.argsort(abs_errors)[::-1][:10]
    worst_days = [
        {
            "date": str(dates_arr[i]),
            "y_true": float(y_true[i]),
            "y_pred": float(y_pred[i]),
            "abs_error": float(abs_errors[i]),
        }
        for i in top_idx
    ]

    return {
        "residuals": residuals,
        "ljung_box_p": ljung_box_p,
        "qq": {"theoretical": theoretical, "sample": sample},
        "worst_days": worst_days,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _unwrap_pipeline(model: Any) -> Any:
    """Return the final estimator of a Pipeline, or the model itself."""
    if isinstance(model, Pipeline):
        return model[-1]
    return model
