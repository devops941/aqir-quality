# Air Quality Forecasting with Weather Variables

A next-day PM2.5 and Indian AQI forecasting system for multi-station urban monitoring networks. The project ingests or generates 3 years of daily air-quality and meteorological data for three Delhi monitoring stations, engineers time-series and weather features, trains and compares four forecasting models (persistence baseline, seasonal-naive baseline, Ridge regression, and Random Forest), and serves predictions through a FastAPI REST backend and a six-page static Plotly.js dashboard — with no Streamlit anywhere.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Data & ML | pandas 3.1, numpy 2.5, scikit-learn 1.9, statsmodels 0.15, scipy 1.18 |
| Backend API | FastAPI 0.163, Uvicorn 0.54, Pydantic v2 |
| Frontend | Vanilla HTML + CSS + JavaScript, Plotly.js (CDN) |
| Database | SQLite via SQLAlchemy 2.1 |
| Model persistence | joblib 1.6 |
| Visualization | matplotlib 3.11, seaborn 0.13 |
| Reporting | Jinja2 3.1, markdown 3.11, xhtml2pdf 0.2 |
| Config | PyYAML 6.0 |
| Testing | pytest 9.2, pytest-cov |
| Packaging | Docker (optional) |

---

## Architecture / Data Flow

```
Raw CSVs (data/raw/)  OR  Synthetic generator (src/data/synthetic.py)
        |
        v
[1] Ingestion & schema validation (src/data/ingest.py)
        |
        v
[2] Cleaning — negatives/impossible -> NaN, outlier flags,
    gap interpolation (<= 2 days), stuck-sensor removal (src/data/clean.py)
        |
        v
[3] Merge AQ + weather on (station_id, timestamp); resample to daily
        |
        v
[4] Indian AQI computation — CPCB sub-indices, AQI = max sub-index (src/features/aqi.py)
        |
        v
[5] Feature engineering — lags, rolling stats, weather, calendar,
    wind sin/cos, event flags; Mode A (strict) & Mode B (+next-day weather)
    — NO data leakage (src/features/build.py)
        |
        v
[6] SQLite storage — stations, aq_readings, weather_readings,
    daily_features, model_runs, metrics, predictions, feature_importance
        |
        v
[7] Modelling — chronological 70/15/15 split,
    TimeSeriesSplit CV, RandomizedSearchCV tuning (src/models/)
        |
        v
[8] Evaluation — MAE, RMSE, R2, MAPE, skill vs persistence,
    category accuracy, high-pollution event F1; by season & by station
        |
        v
[9] Explainability — impurity importance, permutation importance,
    partial dependence (src/models/explain.py)
        |
        v
[10] Auto-report — Markdown + HTML + PDF with figures (src/report/)
        |
        v
[11] FastAPI backend (backend/) <--> [12] Plotly.js dashboard (frontend/)
```

---

## Project Structure

```
air_quality_check/
├── backend/
│   ├── routers/
│   │   └── __init__.py
│   ├── deps.py
│   ├── schemas.py
│   └── __init__.py
├── data/
│   ├── raw/              # drop air_quality.csv / weather.csv / events.csv here
│   ├── processed/
│   └── aqi.db            # SQLite database (auto-created by pipeline)
├── docs/
│   ├── API_CONTRACT.md
│   └── INTERFACES.md
├── frontend/
│   ├── css/style.css
│   ├── js/               # api.js, home.js, history.js, performance.js,
│   │   └── ...           # explain.js, whatif.js, about.js, ui.js
│   ├── index.html        # Home / Forecast
│   ├── history.html
│   ├── performance.html
│   ├── explain.html
│   ├── whatif.html
│   └── about.html
├── models/               # saved joblib pipeline bundles
│   ├── ridge_modeA.joblib
│   ├── ridge_modeB.joblib
│   ├── random_forest_modeA.joblib
│   └── random_forest_modeB.joblib
├── notebooks/
├── outputs/              # report.md, report.html, report.pdf, PNG figures
├── src/
│   ├── config.py         # single-source YAML loader (CONFIG singleton)
│   ├── pipeline.py       # CLI orchestrator
│   ├── data/
│   │   ├── db.py         # SQLAlchemy ORM + engine/session helpers
│   │   ├── ingest.py
│   │   ├── clean.py
│   │   └── synthetic.py
│   ├── features/
│   │   ├── aqi.py        # CPCB AQI computation
│   │   └── build.py      # feature engineering (Mode A / B)
│   ├── forecast/
│   │   └── live.py       # live forecast from Open-Meteo + best model
│   ├── models/
│   │   ├── baselines.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── explain.py
│   └── report/
│       ├── generate.py   # generate_report() + CLI
│       └── template.md.j2
├── tests/
│   ├── test_aqi.py
│   ├── test_data_layer.py
│   ├── test_features.py
│   ├── test_metrics.py
│   ├── test_models.py
│   └── test_report.py
├── config.yaml           # master configuration (city, AQI breakpoints, model params)
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── docker-compose.yml
```

---

## Quickstart (Windows PowerShell)

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the full pipeline on synthetic data

```powershell
python -m src.pipeline --source synthetic
```

This generates 3 years of synthetic Delhi data, trains all models, and writes metrics and predictions to `data/aqi.db`.

### 3. Generate the auto-report

```powershell
python -m src.report.generate
```

Outputs `outputs/report.md`, `outputs/report.html`, `outputs/report.pdf`, and four PNG figures.

### 4. Start the API server

```powershell
python -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000 for the dashboard and http://127.0.0.1:8000/docs for the interactive Swagger UI.

---

## Using Real Data

Drop your data files into `data/raw/` using these schemas:

**`air_quality.csv`** — columns: `station_id, station_name, city, lat, lon, timestamp, pm25, pm10, no2, so2, co, o3`

**`weather.csv`** — columns: `station_id, timestamp, temp_c, humidity_pct, wind_speed_ms, wind_dir_deg, pressure_hpa, precip_mm, cloud_pct, blh_m`

**`events.csv`** — columns: `date, event_name, flag`

Sources: Kaggle "Air Quality Data in India (2015-2020)", CPCB CAAQMS portal, OpenAQ, Open-Meteo Historical Weather API.

Then run:

```powershell
python -m src.pipeline --source csv
```

---

## Features

- **Indian AQI (CPCB):** Sub-index computation for PM2.5, PM10, NO2, SO2, CO, O3; AQI = max sub-index; six categories (Good through Severe).
- **Feature engineering:** Pollutant lags (t, t-1, t-2, t-3, t-7), rolling 3/7-day mean/max/std, today's weather, weather change from t-1, wind sin/cos, day-of-week, month, season, festival/event flags.
- **Mode A — strict forecast:** Only information available at forecast time t (no future weather). Honest real-world evaluation.
- **Mode B — weather-assisted:** Adds next-day weather as features; uses observed weather as a proxy for a perfect forecast. Results are optimistic.
- **No data leakage:** All lags use `shift(k >= 1)`, rolling windows end at t, scalers fitted on training fold only, no random shuffling.
- **Synthetic data fallback:** The whole project runs end-to-end on generated data — no downloads required for development or demo.
- **Model comparison:** Persistence, seasonal-naive, Ridge, Random Forest — evaluated with MAE, RMSE, R2, MAPE, skill vs persistence, AQI category accuracy, and high-pollution event F1.
- **Explainability:** Impurity importance, permutation importance, partial dependence curves.
- **Auto-report:** Markdown + HTML + PDF report with four embedded figures, auto-generated from real DB data.

---

## Dashboard Pages

| Page | File | Description |
|------|------|-------------|
| Home / Forecast | `index.html` | Next-day PM2.5 and AQI forecast for all stations |
| Historical Trends | `history.html` | Interactive time-series of PM2.5, AQI, and weather |
| Model Performance | `performance.html` | Model comparison table, metrics by season and station |
| Explainability | `explain.html` | Feature importance, partial dependence, residuals |
| What-If Predictor | `whatif.html` | Interactive scenario tool — adjust weather/pollution inputs |
| About | `about.html` | Methodology, data sources, limitations |

---

## REST API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check; returns DB status and best run ID |
| GET | `/api/stations` | List all monitoring stations |
| GET | `/api/history` | Historical PM2.5/AQI + weather (`?station_id=&start=&end=&variables=`) |
| GET | `/api/forecast` | Next-day forecast from best model (`?station_id=`) |
| GET | `/api/predictions` | Test-period predictions (`?station_id=&run_id=`) |
| GET | `/api/models` | All model runs with overall metrics |
| GET | `/api/metrics` | Metrics by scope (`?run_id=&scope=`) |
| GET | `/api/feature-importance` | Feature importances sorted desc (`?run_id=`) |
| POST | `/api/predict` | What-if forecast from custom feature inputs |
| POST | `/api/admin/refresh` | Regenerate live forecasts for all stations |

Full request/response schemas are in `docs/API_CONTRACT.md`. Interactive docs at `/docs` (Swagger UI).

---

## Model Results (Real DB)

Best model: **Ridge Regression, Mode B** (run_id = 7, `is_best = 1`)

| Metric | Value |
|--------|-------|
| MAE | 6.549 µg/m³ |
| RMSE | 8.703 µg/m³ |
| R² | 0.9622 |
| MAPE | 39.69% |
| Skill vs Persistence | 0.2436 |
| Category Accuracy | 88.21% |
| Event F1 | 0.9192 |

**All models — overall test metrics:**

| Model | Mode | MAE | RMSE | R² | Skill vs Persistence |
|-------|------|-----|------|----|----------------------|
| Persistence | A | 8.630 | 11.506 | 0.9340 | — |
| Seasonal Naive | A | 10.629 | 14.327 | 0.8976 | -0.245 |
| Ridge | A | 6.639 | 8.841 | 0.9610 | 0.232 |
| Random Forest | A | 6.491 | 8.801 | 0.9614 | 0.235 |
| Persistence | B | 8.630 | 11.506 | 0.9340 | — |
| Seasonal Naive | B | 10.629 | 14.327 | 0.8976 | -0.245 |
| **Ridge (best)** | **B** | **6.549** | **8.703** | **0.9622** | **0.244** |
| Random Forest | B | 6.471 | 8.790 | 0.9615 | 0.236 |

Test period: 2026-06-07 to 2026-11-17 (492 predictions across 3 stations).

---

## Running Tests

```powershell
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=term-missing

# Run only report tests
python -m pytest tests/test_report.py -q
```

---

## Mode A vs Mode B

**Mode A — Strict Forecast (default):** Predicts next-day PM2.5 using only information known at time *t*: past pollutant readings (lags), rolling statistics, today's weather, calendar features, and event flags. This is what a real operational forecast at *t* could achieve.

**Mode B — Weather-Assisted:** Adds next-day weather variables (temperature, humidity, wind, precipitation) as features. During training, observed future weather is used as a perfect-forecast proxy. Mode B provides an **optimistic upper bound**; real deployments would need a weather forecast service (e.g., Open-Meteo Forecast API) which introduces additional error.

---

## Limitations

- Synthetic data only by default; real data requires manual CSV placement.
- Single city (Delhi); generalization to other cities is not validated.
- Mode B results with observed future weather are optimistic.
- Models are static; no automatic retraining on new data.
- No deep learning (LSTM/Transformer) — listed as future work.
- Indian AQI requires at least 3 pollutants; AQI may be missing in data-sparse periods.

---

## Future Work

- HistGradientBoosting and XGBoost/LightGBM comparison
- SARIMAX with exogenous weather variables
- LSTM and Transformer sequence models
- Satellite AOD (MODIS/VIIRS) integration
- Multi-city generalization
- Live data pipeline (CPCB/OpenAQ + Open-Meteo)
- Probabilistic forecasts (quantile regression, conformal prediction)

---

## Docker

```powershell
# Build the image
docker build -t air-quality-forecast .

# Run the container (runs pipeline once then starts API)
docker run -p 8000:8000 air-quality-forecast

# Or use docker-compose
docker-compose up
```

The container runs the synthetic pipeline on first start, then serves the API and dashboard at port 8000.

---

## License

This project was created as a final-year academic project. No license is applied; all rights reserved by the author.
