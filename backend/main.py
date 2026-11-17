"""FastAPI application entry point for the Air Quality Forecasting API.

Mounts all routers under /api, adds /health, enables permissive CORS for
localhost development, and serves the static frontend at / (mounted last so
it does not shadow /api or /docs).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.deps import get_config
from backend.routers import admin, forecast, models, stations, upload
from backend.schemas import HealthResponse
from src.config import ROOT
from src.data.db import ModelRun, get_engine, get_session
from sqlalchemy import select

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Air Quality Forecasting API",
    version="1.0.0",
    description=(
        "REST API for the Air Quality Forecasting project. "
        "Serves historical data, model metrics, feature importance, "
        "next-day forecasts, and what-if predictions."
    ),
)

# ---------------------------------------------------------------------------
# CORS — permissive for local development
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# /health endpoint (not under /api)
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    """Liveness / readiness check: verifies DB connectivity and reports best run."""
    engine = get_engine()
    db_ok = False
    best_run_id: int | None = None

    try:
        with get_session(engine) as session:
            # Simple connectivity check
            session.execute(text("SELECT 1"))
            db_ok = True

            # Find best run
            best_run = session.execute(
                select(ModelRun).where(ModelRun.is_best == 1).limit(1)
            ).scalars().first()
            if best_run:
                best_run_id = best_run.run_id
    except SQLAlchemyError:
        db_ok = False

    return HealthResponse(status="ok", db=db_ok, best_run_id=best_run_id)


# ---------------------------------------------------------------------------
# API routers — all under /api prefix
# ---------------------------------------------------------------------------

app.include_router(stations.router, prefix="/api", tags=["stations"])
app.include_router(forecast.router, prefix="/api", tags=["forecast"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(admin.router, prefix="/api", tags=["admin"])
app.include_router(upload.router, prefix="/api", tags=["admin"])

# ---------------------------------------------------------------------------
# Static frontend — mounted LAST so /api and /docs are not shadowed
# ---------------------------------------------------------------------------

_frontend_dir = ROOT / "frontend"
if _frontend_dir.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(_frontend_dir), html=True),
        name="frontend",
    )
