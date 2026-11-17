"""Admin endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.deps import get_config, get_db
from backend.schemas import RefreshResponse
from src.data.db import Station

router = APIRouter()


@router.post("/admin/refresh", response_model=RefreshResponse)
def refresh_forecasts(
    session: Session = Depends(get_db),
    config: dict = Depends(get_config),
) -> RefreshResponse:
    """Regenerate live forecasts for all stations.

    Calls ``make_live_forecast`` for every station in the database and
    returns the count of refreshed stations.
    """
    from src.forecast.live import make_live_forecast

    stations = session.execute(select(Station)).scalars().all()
    refreshed = 0
    errors: list[str] = []

    for station in stations:
        try:
            make_live_forecast(station.station_id, config=config, session=session)
            refreshed += 1
        except Exception as exc:
            errors.append(f"{station.station_id}: {exc}")

    if errors:
        message = (
            f"Live forecasts regenerated for {refreshed} stations. "
            f"Errors: {'; '.join(errors)}"
        )
    else:
        message = f"Live forecasts regenerated for {refreshed} stations."

    return RefreshResponse(status="ok", message=message)
