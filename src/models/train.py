"""Model training and logging for the Air Quality Forecasting project (Phase 5-6).

Trains persistence, seasonal_naive, ridge (+ statsmodels OLS), random_forest,
optional hist_gb, and optional sarimax.  Logs every run to the SQLite database,
saves joblib bundles, and marks the best non-baseline model (lowest test RMSE)
with ``is_best=1``.

Public API (binding — see docs/INTERFACES.md)::

    train_and_log(feat, config, mode, session) -> dict
    load_model(run_id, session) -> dict

Model artifact format
---------------------
Every non-baseline model is saved as a dict bundle::

    {
        "estimator": fitted_sklearn_pipeline,
        "feature_names": [...],
        "feature_defaults": {feat: median_value},
        "mode": "A" | "B",
        "model_name": str,
        "target": "pm25",
    }

Saved to ``models/<model_name>_mode<MODE>.joblib``.
"""
from __future__ import annotations

import json
import math
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG, resolve_path
from src.data.db import FeatureImportance, Metric, ModelRun, Prediction, get_engine, get_session, init_db
from src.features.aqi import aqi_category, compute_subindex
from src.features.build import feature_columns
from src.models.baselines import run_baseline
from src.models.evaluate import metrics_by_scope, regression_metrics, skill_score
from src.models.explain import impurity_importance, permutation_importance_df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _pm25_to_category(pm25_value: float, config: dict[str, Any] = CONFIG) -> str | None:
    """Map PM2.5 concentration → CPCB category via the pm25 sub-index."""
    if pm25_value is None or (isinstance(pm25_value, float) and math.isnan(pm25_value)):
        return None
    si = compute_subindex("pm25", float(pm25_value), config)
    if math.isnan(si):
        return None
    return aqi_category(si, config)


def _ensure_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    """Create models/ and outputs/ directories; return their Paths."""
    root = Path(__file__).resolve().parents[2]
    models_dir = (root / config["paths"]["models"]).resolve()
    outputs_dir = (root / config["paths"]["outputs"]).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, outputs_dir


def _build_predictions_df(
    station_ids: np.ndarray,
    target_dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: dict[str, Any] = CONFIG,
) -> pd.DataFrame:
    """Build the predictions DataFrame with category columns."""
    pred_cats = [_pm25_to_category(v, config) for v in y_pred]
    true_cats = [_pm25_to_category(v, config) for v in y_true]
    return pd.DataFrame({
        "station_id": station_ids,
        "target_date": target_dates,
        "y_true": y_true,
        "y_pred": y_pred,
        "pred_category": pred_cats,
        "true_category": true_cats,
    })


def _log_run(
    session,
    model_name: str,
    mode: str,
    params: dict,
    model_path: str | None,
) -> int:
    """Insert a ModelRun row and return its run_id."""
    run = ModelRun(
        model_name=model_name,
        feature_mode=mode,
        params_json=json.dumps(params),
        trained_at=datetime.now(),
        model_path=model_path,
        is_best=0,
    )
    session.add(run)
    session.flush()
    return run.run_id


def _log_metrics(
    session,
    run_id: int,
    scope_metrics: list[dict],
    persistence_rmse: float | None = None,
) -> None:
    """Insert Metric rows for all scopes."""
    for sm in scope_metrics:
        svp = sm.get("skill_vs_persistence")
        if svp is None and persistence_rmse is not None:
            overall_rmse = sm.get("rmse")
            if overall_rmse is not None and sm["scope"] == "overall":
                svp = skill_score(overall_rmse, persistence_rmse)

        metric = Metric(
            run_id=run_id,
            scope=sm["scope"],
            mae=sm.get("mae"),
            rmse=sm.get("rmse"),
            r2=sm.get("r2"),
            mape=sm.get("mape"),
            skill_vs_persistence=svp,
            category_accuracy=sm.get("category_accuracy"),
            event_precision=sm.get("event_precision"),
            event_recall=sm.get("event_recall"),
            event_f1=sm.get("event_f1"),
        )
        session.add(metric)
    # Also add per-scope skill scores (for non-overall scopes, use the same persistence RMSE)
    if persistence_rmse is not None:
        for metric_obj in session.new:
            if isinstance(metric_obj, Metric) and metric_obj.run_id == run_id:
                if metric_obj.scope != "overall" and metric_obj.skill_vs_persistence is None:
                    if metric_obj.rmse is not None:
                        metric_obj.skill_vs_persistence = skill_score(
                            metric_obj.rmse, persistence_rmse
                        )


def _log_predictions(session, run_id: int, preds_df: pd.DataFrame) -> None:
    """Insert Prediction rows."""
    for _, row in preds_df.iterrows():
        pred = Prediction(
            run_id=run_id,
            station_id=str(row["station_id"]),
            target_date=pd.to_datetime(row["target_date"]).date(),
            y_true=float(row["y_true"]) if pd.notna(row["y_true"]) else None,
            y_pred=float(row["y_pred"]) if pd.notna(row["y_pred"]) else None,
            pred_category=row.get("pred_category"),
            true_category=row.get("true_category"),
        )
        session.add(pred)


def _log_feature_importance(
    session,
    run_id: int,
    imp_df: pd.DataFrame | None,
    perm_df: pd.DataFrame | None,
) -> None:
    """Insert FeatureImportance rows merging impurity + permutation results."""
    if imp_df is None and perm_df is None:
        return

    # Build a unified feature set
    all_features: set[str] = set()
    if imp_df is not None:
        all_features.update(imp_df["feature"].tolist())
    if perm_df is not None:
        all_features.update(perm_df["feature"].tolist())

    imp_map: dict[str, float] = {}
    if imp_df is not None:
        imp_map = dict(zip(imp_df["feature"], imp_df["impurity_importance"]))

    perm_mean_map: dict[str, float] = {}
    perm_std_map: dict[str, float] = {}
    if perm_df is not None:
        perm_mean_map = dict(zip(perm_df["feature"], perm_df["permutation_importance"]))
        perm_std_map = dict(zip(perm_df["feature"], perm_df["permutation_std"]))

    for feat in all_features:
        fi = FeatureImportance(
            run_id=run_id,
            feature=feat,
            impurity_importance=imp_map.get(feat),
            permutation_importance=perm_mean_map.get(feat),
            permutation_std=perm_std_map.get(feat),
        )
        session.add(fi)


def _save_bundle(
    models_dir: Path,
    model_name: str,
    mode: str,
    estimator: Any,
    feature_names: list[str],
    X_train: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    """Save the joblib bundle and return the file path string."""
    feature_defaults = {col: float(X_train[col].median()) for col in feature_names}
    bundle = {
        "estimator": estimator,
        "feature_names": feature_names,
        "feature_defaults": feature_defaults,
        "mode": mode,
        "model_name": model_name,
        "target": "pm25",
    }
    path = models_dir / f"{model_name}_mode{mode}.joblib"
    joblib.dump(bundle, path)
    return str(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_log(
    feat: pd.DataFrame,
    config: dict[str, Any] = CONFIG,
    mode: str = "A",
    session=None,
) -> dict[str, Any]:
    """Train all models, log to DB, save artifacts, return summary dict.

    Parameters
    ----------
    feat:
        Feature table produced by ``build_feature_table``, containing
        columns: ``station_id``, ``date``, feature columns, ``target_pm25_next``,
        ``target_aqi_next``, ``split``.
    config:
        Project configuration dict.
    mode:
        Feature mode: ``"A"`` or ``"B"``.
    session:
        SQLAlchemy Session.  If ``None``, a new session is created using the
        default engine from ``config['paths']['db']``.

    Returns
    -------
    dict
        ``{"best_run_id": int, "runs": [{run_id, model_name, rmse, ...}, ...]}``.
    """
    models_cfg = config.get("models", {})
    seed = config.get("project", {}).get("random_seed", 42)
    cv_splits = int(models_cfg.get("cv_splits", 5))
    models_dir, outputs_dir = _ensure_dirs(config)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    _own_session = session is None
    if _own_session:
        engine = get_engine()
        init_db(engine)
        session = get_session(engine)

    try:
        return _train_and_log_inner(
            feat=feat,
            config=config,
            mode=mode,
            session=session,
            seed=seed,
            cv_splits=cv_splits,
            models_dir=models_dir,
            outputs_dir=outputs_dir,
            models_cfg=models_cfg,
        )
    finally:
        if _own_session:
            session.close()


def _train_and_log_inner(
    feat: pd.DataFrame,
    config: dict[str, Any],
    mode: str,
    session,
    seed: int,
    cv_splits: int,
    models_dir: Path,
    outputs_dir: Path,
    models_cfg: dict,
) -> dict[str, Any]:
    """Inner implementation so the outer function can cleanly manage the session."""

    # ------------------------------------------------------------------
    # Split data
    # ------------------------------------------------------------------
    feat_cols = feature_columns(feat)
    target_col = "target_pm25_next"

    train_mask = feat["split"] == "train"
    val_mask = feat["split"] == "val"
    test_mask = feat["split"] == "test"

    X_train = feat.loc[train_mask, feat_cols]
    y_train = feat.loc[train_mask, target_col]

    X_val = feat.loc[val_mask, feat_cols]
    y_val = feat.loc[val_mask, target_col]

    X_test = feat.loc[test_mask, feat_cols]
    y_test = feat.loc[test_mask, target_col]

    # Test split metadata
    test_station_ids = feat.loc[test_mask, "station_id"].values
    test_dates = (
        pd.to_datetime(feat.loc[test_mask, "date"]) + pd.Timedelta(days=1)
    ).dt.normalize().values

    # ------------------------------------------------------------------
    # Baselines (run_baseline handles split selection internally)
    # ------------------------------------------------------------------
    runs_summary: list[dict] = []

    # Persistence baseline
    persist_preds = run_baseline("persistence", feat, split="test")
    persist_metrics = metrics_by_scope(persist_preds, feat, config)
    persist_overall_rmse = next(
        (m["rmse"] for m in persist_metrics if m["scope"] == "overall"), None
    )

    persist_run_id = _log_run(
        session, "persistence", mode,
        {"type": "persistence", "col": "pm25_lag0"}, None
    )
    _log_metrics(session, persist_run_id, persist_metrics, None)
    _log_predictions(session, persist_run_id, persist_preds)
    session.flush()

    runs_summary.append({
        "run_id": persist_run_id,
        "model_name": "persistence",
        "rmse": persist_overall_rmse,
        "is_baseline": True,
    })

    # Seasonal naive baseline
    sn_preds = run_baseline("seasonal_naive", feat, split="test")
    sn_metrics = metrics_by_scope(sn_preds, feat, config)
    sn_overall_rmse = next(
        (m["rmse"] for m in sn_metrics if m["scope"] == "overall"), None
    )

    sn_run_id = _log_run(
        session, "seasonal_naive", mode,
        {"type": "seasonal_naive", "col": "pm25_lag7"}, None
    )
    _log_metrics(session, sn_run_id, sn_metrics, persist_overall_rmse)
    _log_predictions(session, sn_run_id, sn_preds)
    session.flush()

    runs_summary.append({
        "run_id": sn_run_id,
        "model_name": "seasonal_naive",
        "rmse": sn_overall_rmse,
        "is_baseline": True,
    })

    # ------------------------------------------------------------------
    # Ridge regression (with StandardScaler; tuned over alphas via CV)
    # ------------------------------------------------------------------
    ridge_cfg = models_cfg.get("ridge", {})
    alphas = ridge_cfg.get("alphas", [0.1, 1.0, 10.0])

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # Grid search over alphas using train split only
    best_alpha = alphas[0]
    best_alpha_rmse = float("inf")
    for alpha in alphas:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        # Simple k-fold CV on training data
        fold_rmses = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]
            y_va = y_train.iloc[val_idx]
            pipeline.fit(X_tr, y_tr)
            y_va_pred = pipeline.predict(X_va)
            fold_rmse = math.sqrt(float(np.mean((y_va.values - y_va_pred) ** 2)))
            fold_rmses.append(fold_rmse)
        mean_rmse = float(np.mean(fold_rmses))
        if mean_rmse < best_alpha_rmse:
            best_alpha_rmse = mean_rmse
            best_alpha = alpha

    # Fit final ridge on full training set
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha)),
    ])
    ridge_pipeline.fit(X_train, y_train)

    # OLS (statsmodels) on standardized training data — run once
    _run_ols(X_train, y_train, mode, outputs_dir)

    # Ridge test predictions
    ridge_pred_vals = ridge_pipeline.predict(X_test)
    ridge_preds_df = _build_predictions_df(
        test_station_ids, test_dates, y_test.values, ridge_pred_vals, config
    )
    ridge_metrics = metrics_by_scope(ridge_preds_df, feat, config)
    ridge_overall_rmse = next(
        (m["rmse"] for m in ridge_metrics if m["scope"] == "overall"), None
    )

    # Fill skill_vs_persistence for all scopes
    for m in ridge_metrics:
        if m.get("rmse") is not None and persist_overall_rmse:
            m["skill_vs_persistence"] = skill_score(m["rmse"], persist_overall_rmse)

    ridge_path = _save_bundle(
        models_dir, "ridge", mode, ridge_pipeline, feat_cols, X_train, config
    )
    ridge_run_id = _log_run(
        session, "ridge", mode,
        {"alpha": best_alpha, "cv_splits": cv_splits}, ridge_path
    )
    _log_metrics(session, ridge_run_id, ridge_metrics, persist_overall_rmse)
    _log_predictions(session, ridge_run_id, ridge_preds_df)
    # Permutation importance (impurity is undefined for a linear model, so None)
    ridge_perm_df = permutation_importance_df(ridge_pipeline, X_test, y_test, feat_cols, n_repeats=10)
    _log_feature_importance(session, ridge_run_id, None, ridge_perm_df)
    session.flush()

    runs_summary.append({
        "run_id": ridge_run_id,
        "model_name": "ridge",
        "rmse": ridge_overall_rmse,
        "is_baseline": False,
        "params": {"alpha": best_alpha},
    })

    # ------------------------------------------------------------------
    # Random Forest (RandomizedSearchCV + TimeSeriesSplit on TRAIN)
    # ------------------------------------------------------------------
    rf_cfg = models_cfg.get("random_forest", {})
    rf_n_iter = int(rf_cfg.get("n_iter", 25))
    rf_param_grid = rf_cfg.get("param_grid", {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_leaf": [1, 4],
        "max_features": ["sqrt", 0.5],
    })

    # Convert null → None in max_depth (YAML null → None)
    if "max_depth" in rf_param_grid:
        rf_param_grid = dict(rf_param_grid)
        rf_param_grid["max_depth"] = [
            None if v is None else v for v in rf_param_grid["max_depth"]
        ]

    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=seed, n_jobs=1),
        param_distributions=rf_param_grid,
        n_iter=rf_n_iter,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring="neg_root_mean_squared_error",
        random_state=seed,
        refit=True,
        n_jobs=1,
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    rf_pred_vals = best_rf.predict(X_test)
    rf_preds_df = _build_predictions_df(
        test_station_ids, test_dates, y_test.values, rf_pred_vals, config
    )
    rf_metrics = metrics_by_scope(rf_preds_df, feat, config)
    rf_overall_rmse = next(
        (m["rmse"] for m in rf_metrics if m["scope"] == "overall"), None
    )
    for m in rf_metrics:
        if m.get("rmse") is not None and persist_overall_rmse:
            m["skill_vs_persistence"] = skill_score(m["rmse"], persist_overall_rmse)

    # Feature importance (impurity + permutation)
    rf_imp_df = impurity_importance(best_rf, feat_cols)
    rf_perm_df = permutation_importance_df(best_rf, X_test, y_test, feat_cols, n_repeats=10)

    rf_path = _save_bundle(
        models_dir, "random_forest", mode, best_rf, feat_cols, X_train, config
    )
    rf_run_id = _log_run(
        session, "random_forest", mode,
        rf_search.best_params_, rf_path
    )
    _log_metrics(session, rf_run_id, rf_metrics, persist_overall_rmse)
    _log_predictions(session, rf_run_id, rf_preds_df)
    _log_feature_importance(session, rf_run_id, rf_imp_df, rf_perm_df)
    session.flush()

    runs_summary.append({
        "run_id": rf_run_id,
        "model_name": "random_forest",
        "rmse": rf_overall_rmse,
        "is_baseline": False,
        "params": rf_search.best_params_,
        "permutation_importance": rf_perm_df,
    })

    # ------------------------------------------------------------------
    # HistGradientBoosting (optional)
    # ------------------------------------------------------------------
    hgb_run_id = None
    hgb_overall_rmse = None

    hgb_cfg = models_cfg.get("hist_gb", {})
    if hgb_cfg.get("enabled", False):
        hgb_run_id, hgb_overall_rmse = _train_hist_gb(
            X_train, y_train, X_test, y_test,
            test_station_ids, test_dates,
            feat_cols, feat, config, mode, models_dir, session,
            persist_overall_rmse, seed, cv_splits, hgb_cfg,
        )
        if hgb_run_id is not None:
            runs_summary.append({
                "run_id": hgb_run_id,
                "model_name": "hist_gb",
                "rmse": hgb_overall_rmse,
                "is_baseline": False,
            })

    # ------------------------------------------------------------------
    # SARIMAX (optional — skipped if disabled)
    # ------------------------------------------------------------------
    sarimax_cfg = models_cfg.get("sarimax", {})
    if sarimax_cfg.get("enabled", False):
        # Minimal SARIMAX implementation using statsmodels
        _train_sarimax(
            feat, config, mode, session, persist_overall_rmse, models_dir
        )

    # ------------------------------------------------------------------
    # Mark the best non-baseline model (lowest test RMSE)
    # ------------------------------------------------------------------
    non_baseline = [r for r in runs_summary if not r.get("is_baseline", False)]
    best_run_id: int | None = None

    if non_baseline:
        # Filter out None RMSEs
        valid = [r for r in non_baseline if r.get("rmse") is not None]
        if valid:
            best = min(valid, key=lambda r: r["rmse"])
            best_run_id = best["run_id"]

            # Update DB
            from src.data.db import ModelRun as MR
            for r in non_baseline:
                run_obj = session.get(MR, r["run_id"])
                if run_obj:
                    run_obj.is_best = 1 if r["run_id"] == best_run_id else 0

    session.commit()

    return {
        "best_run_id": best_run_id,
        "runs": [
            {k: v for k, v in r.items() if k != "permutation_importance"}
            for r in runs_summary
        ],
    }


def _run_ols(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    mode: str,
    outputs_dir: Path,
) -> None:
    """Fit a statsmodels OLS on standardised training data and save the summary."""
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        X_sm = sm.add_constant(X_scaled)

        ols_model = sm.OLS(y_train.values, X_sm).fit()
        summary_text = str(ols_model.summary())

        # VIF (collinear one-hot/cyclic features make this numerically noisy —
        # suppress the expected SingularMatrix/condition warnings).
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vif_data = []
                for i in range(X_scaled.shape[1]):
                    vif = variance_inflation_factor(X_scaled, i)
                    vif_data.append(f"  {X_train.columns[i]}: VIF={vif:.2f}")
            vif_text = "\n".join(vif_data)
            summary_text += f"\n\nVIF:\n{vif_text}"
        except Exception:
            pass  # VIF not critical

        out_path = outputs_dir / f"ols_summary_mode{mode}.txt"
        out_path.write_text(summary_text, encoding="utf-8")
    except Exception:
        # OLS is supplementary; never fail the whole training run
        pass


def _train_hist_gb(
    X_train, y_train, X_test, y_test,
    test_station_ids, test_dates,
    feat_cols, feat, config, mode, models_dir, session,
    persist_overall_rmse, seed, cv_splits, hgb_cfg,
):
    """Train HistGradientBoostingRegressor and log to DB."""
    try:
        from sklearn.model_selection import ParameterSampler

        param_grid = hgb_cfg.get("param_grid", {
            "max_iter": [200],
            "learning_rate": [0.05],
            "max_leaf_nodes": [31],
            "min_samples_leaf": [20],
        })

        tscv = TimeSeriesSplit(n_splits=cv_splits)
        rng = np.random.default_rng(seed)

        # Sample up to 9 candidates (keep test fast)
        n_candidates = min(9, len(list(ParameterSampler(param_grid, n_iter=9, random_state=seed))))
        best_params = None
        best_cv_rmse = float("inf")

        for params in ParameterSampler(param_grid, n_iter=n_candidates, random_state=seed):
            fold_rmses = []
            for tr_idx, va_idx in tscv.split(X_train):
                X_tr = X_train.iloc[tr_idx]
                y_tr = y_train.iloc[tr_idx]
                X_va = X_train.iloc[va_idx]
                y_va = y_train.iloc[va_idx]
                hgb = HistGradientBoostingRegressor(**params, random_state=seed)
                hgb.fit(X_tr, y_tr)
                fold_rmse = math.sqrt(float(np.mean((y_va.values - hgb.predict(X_va)) ** 2)))
                fold_rmses.append(fold_rmse)
            mean_rmse = float(np.mean(fold_rmses))
            if mean_rmse < best_cv_rmse:
                best_cv_rmse = mean_rmse
                best_params = params

        if best_params is None:
            best_params = {k: v[0] for k, v in param_grid.items()}

        hgb_model = HistGradientBoostingRegressor(**best_params, random_state=seed)
        hgb_model.fit(X_train, y_train)

        hgb_pred_vals = hgb_model.predict(X_test)
        hgb_preds_df = _build_predictions_df(
            test_station_ids, test_dates, y_test.values, hgb_pred_vals, config
        )
        hgb_metrics = metrics_by_scope(hgb_preds_df, feat, config)
        hgb_overall_rmse = next(
            (m["rmse"] for m in hgb_metrics if m["scope"] == "overall"), None
        )
        for m in hgb_metrics:
            if m.get("rmse") is not None and persist_overall_rmse:
                m["skill_vs_persistence"] = skill_score(m["rmse"], persist_overall_rmse)

        # Feature importance — HistGB has no feature_importances_; permutation only.
        hgb_imp_df = None
        hgb_perm_df = permutation_importance_df(hgb_model, X_test, y_test, feat_cols, n_repeats=5)

        hgb_path = _save_bundle(
            models_dir, "hist_gb", mode, hgb_model, feat_cols, X_train, config
        )
        hgb_run_id = _log_run(
            session, "hist_gb", mode, best_params, hgb_path
        )
        _log_metrics(session, hgb_run_id, hgb_metrics, persist_overall_rmse)
        _log_predictions(session, hgb_run_id, hgb_preds_df)
        _log_feature_importance(session, hgb_run_id, hgb_imp_df, hgb_perm_df)
        session.flush()

        return hgb_run_id, hgb_overall_rmse

    except Exception:
        warnings.warn(f"hist_gb training failed:\n{traceback.format_exc()}")
        return None, None


def _train_sarimax(feat, config, mode, session, persist_rmse, models_dir):
    """Optional SARIMAX — only called when enabled in config."""
    try:
        import statsmodels.api as sm

        sarimax_cfg = config["models"]["sarimax"]
        order = tuple(sarimax_cfg.get("order", [1, 0, 1]))
        seasonal_order = tuple(sarimax_cfg.get("seasonal_order", [1, 0, 1, 7]))

        # Use first station only for a univariate SARIMAX demo
        stations = feat["station_id"].unique()
        if len(stations) == 0:
            return

        sid = stations[0]
        station_feat = feat[feat["station_id"] == sid].sort_values("date")
        train_data = station_feat[station_feat["split"] == "train"]["target_pm25_next"]
        test_data = station_feat[station_feat["split"] == "test"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sarimax_model = sm.tsa.SARIMAX(
                train_data,
                order=order,
                seasonal_order=seasonal_order,
            ).fit(disp=False)

        # Forecast over the test horizon
        n_test = len(test_data)
        forecast = sarimax_model.forecast(steps=n_test)
        y_true_vals = test_data["target_pm25_next"].values
        y_pred_vals = forecast.values

        preds_df = pd.DataFrame({
            "station_id": test_data["station_id"].values,
            "target_date": (
                pd.to_datetime(test_data["date"]) + pd.Timedelta(days=1)
            ).dt.normalize().values,
            "y_true": y_true_vals,
            "y_pred": y_pred_vals,
        })

        sarimax_metrics = metrics_by_scope(preds_df, feat, config)
        sarimax_rmse = next(
            (m["rmse"] for m in sarimax_metrics if m["scope"] == "overall"), None
        )
        for m in sarimax_metrics:
            if m.get("rmse") is not None and persist_rmse:
                m["skill_vs_persistence"] = skill_score(m["rmse"], persist_rmse)

        sarimax_run_id = _log_run(
            session, "sarimax", mode,
            {"order": list(order), "seasonal_order": list(seasonal_order)}, None
        )
        _log_metrics(session, sarimax_run_id, sarimax_metrics, persist_rmse)
        _log_predictions(session, sarimax_run_id, preds_df)
        session.flush()

    except Exception:
        warnings.warn(f"SARIMAX training failed:\n{traceback.format_exc()}")


def load_model(
    run_id: int | None = None,
    session=None,
) -> dict[str, Any]:
    """Load a joblib model bundle by run_id (or the best run if run_id is None).

    Parameters
    ----------
    run_id:
        Primary key of the model run to load.  If ``None``, loads the run
        where ``is_best=1``.
    session:
        SQLAlchemy Session.  If ``None``, a temporary session is created.

    Returns
    -------
    dict
        The joblib bundle dict::

            {
                "estimator": fitted_pipeline,
                "feature_names": [...],
                "feature_defaults": {...},
                "mode": "A",
                "model_name": str,
                "target": "pm25",
            }

    Raises
    ------
    FileNotFoundError
        If the model path stored in the DB does not exist on disk.
    ValueError
        If no matching run is found in the DB.
    """
    _own_session = session is None
    if _own_session:
        engine = get_engine()
        session = get_session(engine)

    try:
        from src.data.db import ModelRun as MR

        if run_id is not None:
            run = session.get(MR, run_id)
        else:
            from sqlalchemy import select
            stmt = select(MR).where(MR.is_best == 1).limit(1)
            run = session.execute(stmt).scalars().first()

        if run is None:
            raise ValueError(
                f"No model run found (run_id={run_id}, is_best={run_id is None})"
            )

        if run.model_path is None:
            raise ValueError(
                f"Model run {run.run_id} ({run.model_name}) has no saved model file "
                "(baselines do not save a bundle)."
            )

        model_path = Path(run.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        bundle: dict = joblib.load(model_path)
        return bundle

    finally:
        if _own_session:
            session.close()
