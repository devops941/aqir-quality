"""Upload + retrain endpoints.

POST /api/admin/upload  — validate CSVs, copy to data/raw/, launch pipeline
GET  /api/admin/status  — poll the running/done/failed job
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy import select

from backend.schemas import JobStatusResponse, UploadAcceptedResponse
from src.config import ROOT
from src.data import ingest
from src.data.db import ModelRun, get_engine, get_session

router = APIRouter()

# ---------------------------------------------------------------------------
# Job registry — single background training job
# ---------------------------------------------------------------------------

@dataclass
class _Job:
    state: str = "idle"           # idle | running | done | failed
    message: str = ""
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    returncode: Optional[int] = None
    proc: Optional[subprocess.Popen] = None   # type: ignore[type-arg]


_job = _Job()
_job_lock = threading.Lock()

# Output log path
_LOG_PATH = ROOT / "outputs" / "upload_train.log"


def _launch_pipeline(mode: str) -> None:
    """Start pipeline subprocess and update _job when it finishes."""
    global _job

    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(_LOG_PATH, "w", encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.pipeline", "--source", "csv", "--mode", mode],
        cwd=str(ROOT),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    with _job_lock:
        _job.proc = proc

    # Wait in a daemon thread so the HTTP response is not blocked
    def _wait():
        global _job
        rc = proc.wait()
        log_fh.close()
        with _job_lock:
            _job.returncode = rc
            _job.finished_at = datetime.now()
            _job.state = "done" if rc == 0 else "failed"
            if rc != 0:
                # Grab tail of log for the error message
                try:
                    lines = _LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
                    _job.message = "\n".join(lines[-15:])
                except Exception:
                    _job.message = f"Pipeline exited with code {rc}"

    t = threading.Thread(target=_wait, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# POST /api/admin/upload
# ---------------------------------------------------------------------------

@router.post("/admin/upload", response_model=UploadAcceptedResponse, status_code=202)
async def upload_and_train(
    air_quality: UploadFile = File(..., description="Air quality CSV (required)"),
    weather: UploadFile = File(..., description="Weather CSV (required)"),
    events: Optional[UploadFile] = File(None, description="Events CSV (optional)"),
    mode: str = Form("A", description="Feature mode: A | B | both"),
) -> UploadAcceptedResponse:
    """Validate uploaded CSVs then launch a background retrain pipeline."""

    # Validate mode
    if mode not in ("A", "B", "both"):
        raise HTTPException(status_code=422, detail=[
            {"loc": ["form", "mode"], "msg": "mode must be one of A, B, both", "type": "value_error"}
        ])

    # Check for a running job
    with _job_lock:
        if _job.proc is not None and _job.proc.poll() is None:
            raise HTTPException(status_code=409, detail="A training job is already running.")

    # -------------------------------------------------------------------
    # Validate uploaded files in temp paths (do NOT touch data/raw/ yet)
    # -------------------------------------------------------------------
    validation_errors = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # --- Air quality ---
        aq_tmp = tmp / "air_quality.csv"
        content = await air_quality.read()
        aq_tmp.write_bytes(content)
        try:
            aq_df = ingest.load_air_quality(aq_tmp)
            ingest.validate_air_quality(aq_df)
        except Exception as exc:
            validation_errors.append({"file": "air_quality", "error": str(exc)})

        # --- Weather ---
        wx_tmp = tmp / "weather.csv"
        content = await weather.read()
        wx_tmp.write_bytes(content)
        try:
            wx_df = ingest.load_weather(wx_tmp)
            ingest.validate_weather(wx_df)
        except Exception as exc:
            validation_errors.append({"file": "weather", "error": str(exc)})

        # --- Events (optional) ---
        events_tmp: Optional[Path] = None
        if events is not None:
            events_tmp = tmp / "events.csv"
            content = await events.read()
            events_tmp.write_bytes(content)
            try:
                ingest.load_events(events_tmp)
            except Exception as exc:
                validation_errors.append({"file": "events", "error": str(exc)})

        if validation_errors:
            raise HTTPException(status_code=422, detail=validation_errors)

        # -------------------------------------------------------------------
        # Validation passed — copy files to data/raw/
        # -------------------------------------------------------------------
        raw_dir = ROOT / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(aq_tmp, raw_dir / "air_quality.csv")
        shutil.copy2(wx_tmp, raw_dir / "weather.csv")

        if events_tmp is not None:
            shutil.copy2(events_tmp, raw_dir / "events.csv")
        else:
            # Write a header-only events.csv so the pipeline never crashes
            (raw_dir / "events.csv").write_text(
                "date,event_name,flag\n", encoding="utf-8"
            )

    # -------------------------------------------------------------------
    # Launch background pipeline
    # -------------------------------------------------------------------
    started = datetime.now()
    with _job_lock:
        _job.state = "running"
        _job.message = "Pipeline started."
        _job.started_at = started
        _job.finished_at = None
        _job.returncode = None
        _job.proc = None   # will be set by _launch_pipeline

    _launch_pipeline(mode)

    n_files = 2 + (1 if events is not None else 0)
    return UploadAcceptedResponse(
        status="accepted",
        mode=mode,
        message=f"Validated {n_files} files; training started.",
        started_at=started,
    )


# ---------------------------------------------------------------------------
# GET /api/admin/status
# ---------------------------------------------------------------------------

@router.get("/admin/status", response_model=JobStatusResponse)
def get_job_status() -> JobStatusResponse:
    """Poll the current upload/retrain job state."""
    with _job_lock:
        state = _job.state
        message = _job.message
        started_at = _job.started_at
        finished_at = _job.finished_at
        returncode = _job.returncode
        proc = _job.proc

    # Re-derive state from proc if still tracked as running
    if state == "running" and proc is not None:
        rc = proc.poll()
        if rc is not None:
            # Thread hasn't updated yet — be safe
            state = "done" if rc == 0 else "failed"
            returncode = rc

    best_run_id: Optional[int] = None
    if state == "done":
        try:
            engine = get_engine()
            with get_session(engine) as session:
                best = session.execute(
                    select(ModelRun).where(ModelRun.is_best == 1).limit(1)
                ).scalars().first()
                if best:
                    best_run_id = best.run_id
        except Exception:
            pass

    return JobStatusResponse(
        state=state,
        message=message or None,
        started_at=started_at,
        finished_at=finished_at,
        best_run_id=best_run_id,
        returncode=returncode,
    )
