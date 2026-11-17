# ============================================================
# Air Quality Forecasting — Docker image
# Base: python:3.12-slim (Debian Bookworm)
# xhtml2pdf is pure-Python; no GTK/Cairo/Pango needed.
# ============================================================

FROM python:3.12-slim

# --- System dependencies (minimal) -------------------------
# gcc/g++ only needed for packages that have no pre-built wheel
# on slim; most of our deps (numpy, pandas, scikit-learn) have
# manylinux wheels so this layer is small.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory --------------------------------------
WORKDIR /app

# --- Python dependencies (cached layer) --------------------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy project source ------------------------------------
# .dockerignore excludes .venv, __pycache__, data/aqi.db,
# models/*.joblib, outputs/*, .git
COPY . .

# --- Expose the FastAPI port --------------------------------
EXPOSE 8000

# --- Entrypoint --------------------------------------------
# Step 1: Run the synthetic pipeline to populate the DB and
#         train models (skipped automatically if aqi.db already
#         exists and contains model_runs; pass --rebuild to force).
# Step 2: Start the Uvicorn ASGI server.
#
# To use a pre-built DB, mount it: -v /host/data:/app/data
# To use real CSV data: mount data/raw/ and pass --source csv.
#
# Split into a shell CMD so users can override with docker run:
#   docker run <image> uvicorn backend.main:app --host 0.0.0.0 --port 8000
CMD ["sh", "-c", \
     "python -m src.pipeline --source synthetic && \
      python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"]
