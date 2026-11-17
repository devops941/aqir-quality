"""SQLAlchemy ORM models and helpers for the AQI SQLite database.

Mirrors the schema in Section 8 of the project spec exactly. This module is the
single source of truth for the database contract; every other module imports the
ORM classes and the ``get_engine`` / ``get_session`` / ``init_db`` helpers from
here.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from src.config import resolve_path


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Station(Base):
    __tablename__ = "stations"

    station_id: Mapped[str] = mapped_column(String, primary_key=True)
    station_name: Mapped[str | None] = mapped_column(String)
    city: Mapped[str | None] = mapped_column(String)
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)

    aq_readings: Mapped[list["AQReading"]] = relationship(back_populates="station")
    weather_readings: Mapped[list["WeatherReading"]] = relationship(back_populates="station")


class AQReading(Base):
    __tablename__ = "aq_readings"
    __table_args__ = (
        UniqueConstraint("station_id", "timestamp", name="uq_aq_station_ts"),
        Index("ix_aq_station_ts", "station_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[str] = mapped_column(ForeignKey("stations.station_id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    pm25: Mapped[float | None] = mapped_column(Float)
    pm10: Mapped[float | None] = mapped_column(Float)
    no2: Mapped[float | None] = mapped_column(Float)
    so2: Mapped[float | None] = mapped_column(Float)
    co: Mapped[float | None] = mapped_column(Float)
    o3: Mapped[float | None] = mapped_column(Float)
    aqi: Mapped[float | None] = mapped_column(Float)
    aqi_category: Mapped[str | None] = mapped_column(String)
    is_imputed: Mapped[int] = mapped_column(Integer, default=0)

    station: Mapped["Station"] = relationship(back_populates="aq_readings")


class WeatherReading(Base):
    __tablename__ = "weather_readings"
    __table_args__ = (
        UniqueConstraint("station_id", "timestamp", name="uq_weather_station_ts"),
        Index("ix_weather_station_ts", "station_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[str] = mapped_column(ForeignKey("stations.station_id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    temp_c: Mapped[float | None] = mapped_column(Float)
    humidity_pct: Mapped[float | None] = mapped_column(Float)
    wind_speed_ms: Mapped[float | None] = mapped_column(Float)
    wind_dir_deg: Mapped[float | None] = mapped_column(Float)
    pressure_hpa: Mapped[float | None] = mapped_column(Float)
    precip_mm: Mapped[float | None] = mapped_column(Float)
    cloud_pct: Mapped[float | None] = mapped_column(Float)
    blh_m: Mapped[float | None] = mapped_column(Float)

    station: Mapped["Station"] = relationship(back_populates="weather_readings")


class DailyFeature(Base):
    __tablename__ = "daily_features"

    station_id: Mapped[str] = mapped_column(ForeignKey("stations.station_id"), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    feature_json: Mapped[str | None] = mapped_column(String)
    target_pm25_next: Mapped[float | None] = mapped_column(Float)
    target_aqi_next: Mapped[float | None] = mapped_column(Float)
    split: Mapped[str | None] = mapped_column(String)  # 'train' | 'val' | 'test'


class ModelRun(Base):
    __tablename__ = "model_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String)
    feature_mode: Mapped[str | None] = mapped_column(String)  # 'A' | 'B'
    params_json: Mapped[str | None] = mapped_column(String)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime)
    model_path: Mapped[str | None] = mapped_column(String)
    is_best: Mapped[int] = mapped_column(Integer, default=0)

    metrics: Mapped[list["Metric"]] = relationship(back_populates="run")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="run")


class Metric(Base):
    __tablename__ = "metrics"

    run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.run_id"), primary_key=True)
    scope: Mapped[str] = mapped_column(String, primary_key=True)  # 'overall'|'station:<id>'|'season:<name>'
    mae: Mapped[float | None] = mapped_column(Float)
    rmse: Mapped[float | None] = mapped_column(Float)
    r2: Mapped[float | None] = mapped_column(Float)
    mape: Mapped[float | None] = mapped_column(Float)
    skill_vs_persistence: Mapped[float | None] = mapped_column(Float)
    category_accuracy: Mapped[float | None] = mapped_column(Float)
    event_precision: Mapped[float | None] = mapped_column(Float)
    event_recall: Mapped[float | None] = mapped_column(Float)
    event_f1: Mapped[float | None] = mapped_column(Float)

    run: Mapped["ModelRun"] = relationship(back_populates="metrics")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_pred_station_date", "station_id", "target_date"),)

    run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.run_id"), primary_key=True)
    station_id: Mapped[str] = mapped_column(ForeignKey("stations.station_id"), primary_key=True)
    target_date: Mapped[date] = mapped_column(Date, primary_key=True)
    y_true: Mapped[float | None] = mapped_column(Float)
    y_pred: Mapped[float | None] = mapped_column(Float)
    pred_category: Mapped[str | None] = mapped_column(String)
    true_category: Mapped[str | None] = mapped_column(String)

    run: Mapped["ModelRun"] = relationship(back_populates="predictions")


class FeatureImportance(Base):
    __tablename__ = "feature_importance"

    run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.run_id"), primary_key=True)
    feature: Mapped[str] = mapped_column(String, primary_key=True)
    impurity_importance: Mapped[float | None] = mapped_column(Float)
    permutation_importance: Mapped[float | None] = mapped_column(Float)
    permutation_std: Mapped[float | None] = mapped_column(Float)


class LiveForecast(Base):
    __tablename__ = "live_forecasts"

    station_id: Mapped[str] = mapped_column(ForeignKey("stations.station_id"), primary_key=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    target_date: Mapped[date] = mapped_column(Date, primary_key=True)
    pm25_pred: Mapped[float | None] = mapped_column(Float)
    aqi_pred: Mapped[float | None] = mapped_column(Float)
    category: Mapped[str | None] = mapped_column(String)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("model_runs.run_id"))


# ----------------------------------------------------------------------
# Engine / session helpers
# ----------------------------------------------------------------------
def get_engine(db_path: str | Path | None = None, *, echo: bool = False) -> Engine:
    """Create a SQLite engine. Defaults to the configured ``paths.db`` location."""
    if db_path is None:
        db_path = resolve_path("db", create_parent=True)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=echo, future=True)


def init_db(engine: Engine, *, drop: bool = False) -> None:
    """Create all tables (optionally dropping first)."""
    if drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def get_session(engine: Engine | None = None) -> Session:
    """Return a new ORM session bound to the given (or default) engine."""
    if engine is None:
        engine = get_engine()
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return factory()
