"""Model runs, metrics, predictions, and feature-importance endpoints."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.deps import get_db
from backend.schemas import (
    FeatureImportanceSchema,
    MetricSchema,
    ModelRunSchema,
    PredictionRow,
    PredictionsResponse,
)
from src.data.db import FeatureImportance, Metric, ModelRun, Prediction

router = APIRouter()


@router.get("/models", response_model=List[ModelRunSchema])
def list_models(session: Session = Depends(get_db)) -> list[ModelRunSchema]:
    """Return all model runs with their overall metrics."""
    runs = session.execute(select(ModelRun)).scalars().all()
    result: list[ModelRunSchema] = []
    for run in runs:
        # Fetch the 'overall' metric row for this run
        metric = session.execute(
            select(Metric)
            .where(Metric.run_id == run.run_id)
            .where(Metric.scope == "overall")
        ).scalars().first()

        result.append(
            ModelRunSchema(
                run_id=run.run_id,
                model_name=run.model_name,
                feature_mode=run.feature_mode,
                trained_at=run.trained_at,
                is_best=bool(run.is_best),
                mae=metric.mae if metric else None,
                rmse=metric.rmse if metric else None,
                r2=metric.r2 if metric else None,
                mape=metric.mape if metric else None,
                skill_vs_persistence=metric.skill_vs_persistence if metric else None,
                category_accuracy=metric.category_accuracy if metric else None,
                event_precision=metric.event_precision if metric else None,
                event_recall=metric.event_recall if metric else None,
                event_f1=metric.event_f1 if metric else None,
            )
        )
    return result


@router.get("/metrics", response_model=List[MetricSchema])
def get_metrics(
    run_id: int = Query(..., description="Model run ID"),
    scope: Optional[str] = Query(None, description="Scope filter (overall / station:<id> / season:<name>)"),
    session: Session = Depends(get_db),
) -> list[MetricSchema]:
    """Return metric rows for a model run, optionally filtered by scope."""
    # Verify run exists
    run = session.get(ModelRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Model run {run_id} not found.")

    stmt = select(Metric).where(Metric.run_id == run_id)
    if scope is not None:
        stmt = stmt.where(Metric.scope == scope)
    metrics = session.execute(stmt).scalars().all()

    return [
        MetricSchema(
            run_id=m.run_id,
            scope=m.scope,
            mae=m.mae,
            rmse=m.rmse,
            r2=m.r2,
            mape=m.mape,
            skill_vs_persistence=m.skill_vs_persistence,
            category_accuracy=m.category_accuracy,
            event_precision=m.event_precision,
            event_recall=m.event_recall,
            event_f1=m.event_f1,
        )
        for m in metrics
    ]


@router.get("/feature-importance", response_model=List[FeatureImportanceSchema])
def get_feature_importance(
    run_id: Optional[int] = Query(None, description="Model run ID (defaults to best run)"),
    session: Session = Depends(get_db),
) -> list[FeatureImportanceSchema]:
    """Return feature importance rows, sorted by permutation_importance descending.

    If ``run_id`` is omitted, uses the globally best run (``is_best=1``).
    """
    if run_id is None:
        best_run = session.execute(
            select(ModelRun).where(ModelRun.is_best == 1).limit(1)
        ).scalars().first()
        if best_run is None:
            raise HTTPException(status_code=404, detail="No best model run found.")
        run_id = best_run.run_id

    # Verify run exists
    run = session.get(ModelRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Model run {run_id} not found.")

    rows = session.execute(
        select(FeatureImportance)
        .where(FeatureImportance.run_id == run_id)
        .order_by(FeatureImportance.permutation_importance.desc().nullslast())
    ).scalars().all()

    return [
        FeatureImportanceSchema(
            feature=r.feature,
            impurity_importance=r.impurity_importance,
            permutation_importance=r.permutation_importance,
            permutation_std=r.permutation_std,
        )
        for r in rows
    ]


@router.get("/predictions", response_model=PredictionsResponse)
def get_predictions(
    station_id: str = Query(..., description="Station ID"),
    run_id: Optional[int] = Query(None, description="Model run ID (defaults to best run)"),
    session: Session = Depends(get_db),
) -> PredictionsResponse:
    """Return predicted-vs-actual rows for a station and model run."""
    # Verify station exists
    from src.data.db import Station

    station = session.get(Station, station_id)
    if station is None:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found.")

    # Resolve run_id
    if run_id is None:
        best_run = session.execute(
            select(ModelRun).where(ModelRun.is_best == 1).limit(1)
        ).scalars().first()
        if best_run is None:
            raise HTTPException(status_code=404, detail="No best model run found.")
        resolved_run_id = best_run.run_id
    else:
        # Verify run exists
        run = session.get(ModelRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Model run {run_id} not found.")
        resolved_run_id = run_id

    rows = session.execute(
        select(Prediction)
        .where(Prediction.run_id == resolved_run_id)
        .where(Prediction.station_id == station_id)
        .order_by(Prediction.target_date)
    ).scalars().all()

    return PredictionsResponse(
        run_id=resolved_run_id,
        station_id=station_id,
        rows=[
            PredictionRow(
                target_date=r.target_date,
                y_true=r.y_true,
                y_pred=r.y_pred,
                true_category=r.true_category,
                pred_category=r.pred_category,
            )
            for r in rows
        ],
    )
