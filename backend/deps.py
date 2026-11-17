"""FastAPI dependency functions for the Air Quality Forecasting API.

Provides:
- ``get_db`` — SQLAlchemy session dependency (yield + close).
- ``get_config`` — returns the loaded project config dict.
"""
from __future__ import annotations

from typing import Any, Generator

from sqlalchemy.orm import Session

from src.config import CONFIG, load_config
from src.data.db import get_engine, get_session


# Module-level engine (created once at import time, reused across requests)
_engine = get_engine()


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session and close it after the request."""
    session: Session = get_session(_engine)
    try:
        yield session
    finally:
        session.close()


def get_config() -> dict[str, Any]:
    """Return the cached project configuration dict."""
    return CONFIG
