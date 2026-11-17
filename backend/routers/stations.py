"""Stations and history endpoints."""
from __future__ import annotations

from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.deps import get_db
from backend.schemas import HistoryResponse, HistoryRow, StationSchema
from src.data.db import AQReading, Station, WeatherReading

router = APIRouter()


@router.get("/stations", response_model=List[StationSchema])
def list_stations(session: Session = Depends(get_db)) -> list[StationSchema]:
    """Return all stations."""
    rows = session.execute(select(Station)).scalars().all()
    return [StationSchema.model_validate(r) for r in rows]


@router.get("/history", response_model=HistoryResponse)
def get_history(
    station_id: str = Query(..., description="Station ID, e.g. DL001"),
    start: Optional[date] = Query(None, description="Start date YYYY-MM-DD (inclusive)"),
    end: Optional[date] = Query(None, description="End date YYYY-MM-DD (inclusive)"),
    variables: Optional[str] = Query(
        None,
        description="Comma-separated variable names to include (default: all)",
    ),
    session: Session = Depends(get_db),
) -> HistoryResponse:
    """Return daily history rows for a station, joining AQ + weather readings.

    Rows are sorted ascending by date.  The ``variables`` parameter optionally
    restricts which columns are populated (others are set to ``None``).
    """
    # Verify station exists
    station = session.get(Station, station_id)
    if station is None:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found.")

    # Parse requested variables
    requested_vars: set[str] | None = None
    if variables:
        requested_vars = {v.strip() for v in variables.split(",") if v.strip()}

    # Fetch AQ readings
    aq_stmt = select(AQReading).where(AQReading.station_id == station_id)
    if start:
        aq_stmt = aq_stmt.where(AQReading.timestamp >= str(start))
    if end:
        # end is inclusive — add 1 day to cover the full day
        from datetime import timedelta
        end_dt = date.fromisoformat(str(end)) + timedelta(days=1)
        aq_stmt = aq_stmt.where(AQReading.timestamp < str(end_dt))
    aq_stmt = aq_stmt.order_by(AQReading.timestamp)
    aq_rows = session.execute(aq_stmt).scalars().all()

    # Fetch weather readings (same date range, same station)
    wx_stmt = select(WeatherReading).where(WeatherReading.station_id == station_id)
    if start:
        wx_stmt = wx_stmt.where(WeatherReading.timestamp >= str(start))
    if end:
        from datetime import timedelta
        end_dt = date.fromisoformat(str(end)) + timedelta(days=1)
        wx_stmt = wx_stmt.where(WeatherReading.timestamp < str(end_dt))
    wx_stmt = wx_stmt.order_by(WeatherReading.timestamp)
    wx_rows = session.execute(wx_stmt).scalars().all()

    # Build a date → weather mapping
    wx_map: dict[str, WeatherReading] = {}
    for w in wx_rows:
        day_key = w.timestamp.date().isoformat()
        wx_map[day_key] = w  # last entry per day wins (readings are daily)

    # Merge and build response rows
    result_rows: list[HistoryRow] = []
    for aq in aq_rows:
        day = aq.timestamp.date()
        day_key = day.isoformat()
        wx = wx_map.get(day_key)

        def _field(val: object, name: str) -> object:
            """Return val if name is requested (or no filter); else None."""
            if requested_vars is None or name in requested_vars:
                return val
            return None

        row = HistoryRow(
            date=day,
            pm25=_field(aq.pm25, "pm25"),
            pm10=_field(aq.pm10, "pm10"),
            no2=_field(aq.no2, "no2"),
            so2=_field(aq.so2, "so2"),
            co=_field(aq.co, "co"),
            o3=_field(aq.o3, "o3"),
            aqi=_field(aq.aqi, "aqi"),
            aqi_category=_field(aq.aqi_category, "aqi_category"),
            temp_c=_field(wx.temp_c if wx else None, "temp_c"),
            humidity_pct=_field(wx.humidity_pct if wx else None, "humidity_pct"),
            wind_speed_ms=_field(wx.wind_speed_ms if wx else None, "wind_speed_ms"),
            wind_dir_deg=_field(wx.wind_dir_deg if wx else None, "wind_dir_deg"),
            pressure_hpa=_field(wx.pressure_hpa if wx else None, "pressure_hpa"),
            precip_mm=_field(wx.precip_mm if wx else None, "precip_mm"),
            cloud_pct=_field(wx.cloud_pct if wx else None, "cloud_pct"),
        )
        result_rows.append(row)

    return HistoryResponse(station_id=station_id, rows=result_rows)
