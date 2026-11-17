Air Quality Forecasting with Weather Variables
Type: Final-year project | Domain: Environment & Climate | Level: Intermediate
Stack: Python, Pandas, scikit-learn, statsmodels, FastAPI, SQLite, Plotly.js (HTML frontend). No
Streamlit.
Instructions for Claude Code: Read this whole document first. Build the project phase by
phase (Section 10). After each phase, run the tests and confirm the acceptance criteria before
moving on. Do not skip phases. Ask me only if something is truly blocking; otherwise use the
defaults stated here. Do NOT use Streamlit anywhere; the web layer is FastAPI + a static
HTML/JS frontend.
1. Problem Statement
Air pollution (PM2.5, PM10, NO2, SO2, CO, O3) affects health, and levels depend strongly on
weather: wind disperses pollutants, rain washes them out, low temperature and low boundary
layer trap them near the ground. People and authorities need a short-term forecast of air quality.
This project:
1. Builds a clean dataset of historical air-quality and meteorological observations.
2. Engineers time-series and weather features.
3. Trains and compares forecasting models (baseline, linear regression, Random Forest,
optional gradient boosting) for next-day (and optionally next-hour) PM2.5 / AQI.
4. Serves forecasts and model evidence through a REST API and a web dashboard.
2. Objectives
1. Ingest air-quality data (CPCB / OpenAQ / Kaggle India dataset) and weather data (Open-
Meteo / NASA POWER / IMD).
2. Clean, align and merge both sources on timestamp and station.
3. Compute the Indian AQI (CPCB) and AQI category from pollutant concentrations.
4. Engineer lag, rolling, calendar and weather features without data leakage.
5. Build baselines (persistence, seasonal naive) and models (Ridge/Linear regression,
Random Forest, optional HistGradientBoosting and SARIMAX).
6. Evaluate with MAE, RMSE, R^2, plus AQI-category accuracy and high-pollution-event
detection, using time-based cross-validation.
7. Explain which variables matter (feature importance, permutation importance, partial
dependence).
8. Provide a FastAPI backend and a browser dashboard with history, forecast, metrics and a
what-if predictor.
3. Scope
In scope: single-city, multi-station forecasting of next-day PM2.5 and AQI, weather-driven
features, model comparison, API, dashboard, report.
Out of scope: chemical transport models (CMAQ/WRF-Chem), satellite AOD, deep learning
(LSTM/Transformer; listed as future work), mobile app.
Default forecast task: predict next-day mean PM2.5 (and derived AQI category) for each station
using data up to the end of today. Optional second task: next-hour PM2.5 using hourly data.
4. Data Sources
Data
Source
Format
Notes
Air quality
(daily/hourly)
Kaggle "Air Quality Data in India
(2015-2020)" ( city_day.csv ,
station_day.csv ,
station_hour.csv ); CPCB CAAQMS
portal; OpenAQ API
CSV /
JSON
Pollutants: PM2.5, PM10, NO,
NO2, NOx, NH3, CO, SO2, O3,
Benzene, Toluene
Weather
Open-Meteo Historical Weather API
(free, no key); alternatively NASA
POWER or IMD
JSON /
CSV
Temperature, relative
humidity, wind speed and
direction, surface pressure,
precipitation, cloud cover,
boundary layer height (if
available)
Weather
forecast (for
live use)
Open-Meteo Forecast API
JSON
Used only in "live forecast"
mode, see Section 6
Festival / event
calendar
Hand-built CSV (Diwali, Bhogi, crop-
burning season flags, etc.)
CSV
Binary event flags
Default study area: one Indian city with good data coverage (configurable in config.yaml , e.g.
Delhi, Chennai, Bengaluru, Hyderabad). Choose the city with the most complete station data.
Fallback (mandatory): Build src/data/synthetic.py to generate 3 years of hourly data for 3
stations: PM2.5 with annual seasonality (higher in winter), daily cycle (rush hour peaks),
dependence on wind speed (negative), humidity, rain washout, temperature (negative),
autocorrelation, and random spikes. The whole project must run end-to-end on synthetic data so
development and demo never depend on downloads or API availability. Real data plugs in
through the same schemas.
Standard schemas
air_quality.csv : station_id, station_name, city, lat, lon, timestamp, pm25, pm10,
no2, so2, co, o3 (other pollutants optional) weather.csv : station_id (or city),
timestamp, temp_c, humidity_pct, wind_speed_ms, wind_dir_deg, pressure_hpa,
precip_mm, cloud_pct, blh_m (optional)
events.csv : date, event_name, flag
5. System Flow
Raw CSVs / APIs  or  Synthetic generator
|
6. Methods (precise definitions)
1. Cleaning: PM2.5 and PM10 must be >= 0 and below 2000 ug/m3; PM2.5 should not exceed
PM10 by a large margin (flag when PM2.5 > PM10 x 1.2); remove stuck sensor runs (same
v
[1] Ingestion & validation (schema, dtypes, timestamp parsing to UTC/IST,
duplicates)
|
v
[2] Cleaning (negative/impossible values -> NaN, outlier flags, gap handling:
interpolate <= 3 h / <= 2 days, drop stations with > 40% missing)
|
v
[3] Merge air quality + weather on (station, timestamp); resample to daily
(and hourly if enabled)
|
v
[4] AQI computation (CPCB sub-indices, AQI = max sub-index, category)
|
v
[5] Feature engineering (lags, rolling stats, weather, wind sin/cos,
calendar, events)  -- NO leakage
|
v
[6] Store in SQLite (stations, readings, features, targets)
|
v
[7] Modelling: time-based split -> baselines -> Ridge -> Random Forest ->
(HistGB, SARIMAX)
|   TimeSeriesSplit CV + RandomizedSearchCV tuning
v
[8] Evaluation: MAE, RMSE, R2, category accuracy, high-pollution recall,
error by season/station
|
v
[9] Explainability: feature importance, permutation importance, partial
dependence
|
v
[10] Save best model (joblib) + metrics + predictions to SQLite and /outputs
|
v
[11] FastAPI service (history, forecast, metrics, what-if predict)  -->  [12]
Web dashboard (HTML + Plotly.js)
value for more than 24 consecutive hours). Impute short gaps only; never impute the target
in the evaluation period.
2. Indian AQI (CPCB): compute sub-index for each pollutant by linear interpolation between
breakpoints; AQI = max sub-index; valid only if at least 3 pollutants are available and one of
them is PM2.5 or PM10. Categories: Good (0-50), Satisfactory (51-100), Moderate (101-200),
Poor (201-300), Very Poor (301-400), Severe (401-500). Store the breakpoint table in
config.yaml and verify it against the official CPCB document before final results. If only
PM2.5 is modelled, also report the PM2.5 concentration categories.
3. Target: y(t+1) = next-day mean PM2.5 (primary), optional AQI(t+1) and AQI
category(t+1) .
4. Feature engineering (for prediction made at end of day t):
Pollutant lags: PM2.5 at t, t-1, t-2, t-3, t-7; same for PM10, NO2.
Rolling stats: 3-day and 7-day mean/max/std of PM2.5.
Weather at day t: mean temperature, humidity, wind speed, pressure, total
precipitation, cloud cover; weather change from t-1 to t.
Wind direction as sin and cos (never raw degrees).
Calendar: day of week, month, weekend flag, season (winter / pre-monsoon / monsoon
/ post-monsoon), cyclic month encoding.
Events: festival and crop-burning-season flags.
Two feature modes (document both):
Mode A, strict forecast (default): only information available at time t (no future
weather). Honest evaluation of what a real forecast at t could do.
Mode B, with weather forecast: adds weather for day t+1 (in training use
observed weather as a proxy for a perfect forecast; in live mode use Open-Meteo
forecast). Report that Mode B results with observed future weather are
optimistic.
5. Leakage rules (must be enforced and tested): all lags use shift(k) with k >= 0 relative to
the prediction time; rolling windows must end at t; scaling/imputation statistics are fitted
only on the training fold; no random shuffling anywhere.
6. Split: chronological. Default: train = first 70%, validation = next 15%, test = last 15% (final
year held out). Cross-validation inside training uses TimeSeriesSplit(n_splits=5) .
7. Models:
Persistence baseline: y_hat(t+1) = PM2.5(t) .
Seasonal-naive baseline: y_hat(t+1) = PM2.5 on same weekday last week .
Ridge / Linear regression: standardized features, with statsmodels OLS also run
once for coefficient interpretation (p-values, VIF).
Random Forest Regressor: tuned with RandomizedSearchCV over n_estimators ,
max_depth , min_samples_leaf , max_features .
Optional: HistGradientBoostingRegressor , SARIMAX (statsmodels) with weather
as exogenous variables on one station for comparison.
8. Metrics: MAE, RMSE, R^2, MAPE (guard against zeros), skill score vs persistence 1 -
RMSE_model / RMSE_persistence ; for categories: accuracy and confusion matrix; for high-
pollution events (e.g. PM2.5 above 90 ug/m3 or AQI "Poor" and above): precision, recall, F1.
Also report errors by season, by station and by pollution level.
9. Explainability: Random Forest impurity importance, permutation importance on the test
set (preferred, less biased), partial dependence plots for wind speed, humidity, temperature;
optional SHAP.
10. Residual analysis: residuals vs time, residual autocorrelation (Ljung-Box), Q-Q plot, worst
10 error days with explanation (festival, stubble burning, missing weather).
7. Tech Stack
Layer
Choice
Reason
Language
Python 3.11
Standard for data science and
APIs
Data
pandas, numpy
Time series handling
ML
scikit-learn
Ridge, Random Forest, CV,
tuning, metrics
Statistics
statsmodels, scipy
OLS, SARIMAX, Ljung-Box
Weather API
requests / httpx
Open-Meteo
Backend
FastAPI + Uvicorn, Pydantic
REST API with automatic docs at
/docs
Frontend
Plain HTML + CSS + JavaScript, Plotly.js
(served as static files by FastAPI)
No Streamlit, no heavy
framework, easy to explain
Database
SQLite via SQLAlchemy
Zero setup; can be swapped to
PostgreSQL
Model
persistence
joblib
Save/load trained pipeline
Visualization
(offline)
matplotlib, seaborn
Report figures
Reporting
Jinja2 + markdown -> PDF (weasyprint)
Auto report
Scheduling
(optional)
APScheduler
Daily data refresh and retraining
Config
PyYAML
Cities, breakpoints,
hyperparameter grids
Testing
pytest, httpx TestClient
Unit and API tests
Quality
ruff, black
Lint, format
Layer
Choice
Reason
Packaging
(optional)
Docker
Easy demo deployment
8. Database Design (SQLite)
sql
CREATE TABLE stations (
station_id TEXT PRIMARY KEY,
station_name TEXT,
city TEXT,
lat REAL, lon REAL
);
CREATE TABLE aq_readings (
id INTEGER PRIMARY KEY AUTOINCREMENT,
station_id TEXT REFERENCES stations(station_id),
timestamp DATETIME NOT NULL,
pm25 REAL, pm10 REAL, no2 REAL, so2 REAL, co REAL, o3 REAL,
aqi REAL, aqi_category TEXT,
is_imputed INTEGER DEFAULT 0,
UNIQUE(station_id, timestamp)
);
CREATE TABLE weather_readings (
id INTEGER PRIMARY KEY AUTOINCREMENT,
station_id TEXT REFERENCES stations(station_id),
timestamp DATETIME NOT NULL,
temp_c REAL, humidity_pct REAL, wind_speed_ms REAL, wind_dir_deg REAL,
pressure_hpa REAL, precip_mm REAL, cloud_pct REAL, blh_m REAL,
UNIQUE(station_id, timestamp)
);
CREATE TABLE daily_features (
station_id TEXT REFERENCES stations(station_id),
date DATE,
feature_json TEXT,         -- engineered features for that date
target_pm25_next REAL,
target_aqi_next REAL,
split TEXT,                -- 'train' | 'val' | 'test'
PRIMARY KEY (station_id, date)
);
CREATE TABLE model_runs (
run_id INTEGER PRIMARY KEY AUTOINCREMENT,
model_name TEXT,           --
'persistence','seasonal_naive','ridge','random_forest','hist_gb','sarimax'
feature_mode TEXT,         -- 'A' or 'B'
params_json TEXT,
trained_at DATETIME,
model_path TEXT,
is_best INTEGER DEFAULT 0
);
CREATE TABLE metrics (
run_id INTEGER REFERENCES model_runs(run_id),
scope TEXT,                -- 'overall','station:<id>','season:<name>'
mae REAL, rmse REAL, r2 REAL, mape REAL, skill_vs_persistence REAL,
category_accuracy REAL, event_precision REAL, event_recall REAL, event_f1 
REAL,
PRIMARY KEY (run_id, scope)
);
CREATE TABLE predictions (
run_id INTEGER REFERENCES model_runs(run_id),
station_id TEXT REFERENCES stations(station_id),
target_date DATE,
y_true REAL, y_pred REAL, pred_category TEXT, true_category TEXT,
PRIMARY KEY (run_id, station_id, target_date)
);
CREATE TABLE feature_importance (
run_id INTEGER REFERENCES model_runs(run_id),
feature TEXT,
impurity_importance REAL,
permutation_importance REAL, permutation_std REAL,
PRIMARY KEY (run_id, feature)
);
CREATE TABLE live_forecasts (
station_id TEXT REFERENCES stations(station_id),
issued_at DATETIME,
target_date DATE,
pm25_pred REAL, aqi_pred REAL, category TEXT,
run_id INTEGER REFERENCES model_runs(run_id),
PRIMARY KEY (station_id, issued_at, target_date)
);
Indexes: aq_readings(station_id, timestamp) , weather_readings(station_id,
timestamp) , predictions(station_id, target_date) .
9. Backend API and Frontend
REST endpoints (FastAPI)
Method
Endpoint
Purpose
GET
/api/stations
List stations
GET
/api/history?
station_id=&start=&end=&variables=
Historical PM2.5/AQI and weather
GET
/api/forecast?station_id=
Next-day forecast (PM2.5, AQI, category)
using the best model
GET
/api/predictions?station_id=&run_id=
Test-period predicted vs actual
GET
/api/models
All model runs with metrics (comparison
table)
GET
/api/metrics?run_id=&scope=
Detailed metrics
GET
/api/feature-importance?run_id=
Importance values
POST
/api/predict
What-if: body with feature values (weather
+ recent PM2.5), returns predicted PM2.5
and category
POST
/api/admin/refresh
(Optional) fetch latest data and regenerate
forecast
GET
/health
Health check
All endpoints return JSON validated by Pydantic models; automatic Swagger docs at /docs .
Frontend pages (static HTML + Plotly.js, served from / )
1. Home / Forecast: station dropdown, tomorrow's predicted PM2.5 and AQI with colour-
coded category badge (green to maroon), short health message.
2. History: interactive time series of PM2.5 with weather overlays, date-range selector,
seasonal heatmap (month x year).
3. Model Performance: comparison table (baseline vs Ridge vs Random Forest), predicted-vs-
actual chart for the test year, scatter plot, error by season, confusion matrix.
4. Explainability: feature importance bar chart, permutation importance, partial dependence
curves.
5. What-if: form to enter wind speed, humidity, temperature, rain and recent PM2.5, calls
/api/predict , shows result.
6. About / Data: data sources, methodology summary, limitations.
Frontend folder: frontend/index.html, history.html, performance.html, explain.html,
whatif.html, css/style.css, js/*.js using fetch() against the API.
10. Project Structure
air-quality-forecasting/
|-- PROJECT_SPEC_AQI.md
|-- README.md
|-- requirements.txt
|-- config.yaml
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- aqi.db
|-- src/
|   |-- data/
|   |   |-- synthetic.py
|   |   |-- ingest.py          # CSV + API loaders, validation
|   |   |-- clean.py
|   |   `-- db.py              # SQLAlchemy models and helpers
|   |-- features/
|   |   |-- aqi.py             # CPCB AQI calculation
|   |   `-- build.py           # lags, rolling, weather, calendar, events
|   |-- models/
|   |   |-- baselines.py
|   |   |-- train.py           # Ridge, RF, HistGB, tuning, CV
|   |   |-- sarimax.py         # optional
|   |   |-- evaluate.py        # metrics, category + event metrics
|   |   `-- explain.py         # importance, PDP
|   |-- forecast/
|   |   `-- live.py            # fetch weather forecast, build features,
predict
|   |-- report/
|   |   |-- generate.py
|   |   `-- template.md.j2
|   `-- pipeline.py            # end-to-end CLI
|-- backend/
|   |-- main.py                # FastAPI app
|   |-- routers/
|   |-- schemas.py             # Pydantic models
|   `-- deps.py
|-- frontend/
|   |-- index.html, history.html, performance.html, explain.html, whatif.html
|   |-- css/style.css
Commands: python -m src.pipeline --source synthetic|csv --config config.yaml (build
data, train, evaluate), then uvicorn backend.main:app --reload , open
http://localhost:8000 .
|   `-- js/
|-- models/                    # saved joblib files
|-- tests/
|-- notebooks/                 # EDA
`-- outputs/
11. Development Plan (12 weeks)
Phase
Weeks
Deliverables
Acceptance criteria
1. Setup and
study
1
Repo, venv, requirements,
config.yaml, notes on AQI, PM2.5
drivers, time-series CV
pip install -r
requirements.txt works, pytest
runs
2. Data layer
2-3
synthetic.py , ingest.py ,
clean.py , db.py
3 synthetic stations x 3 years load
into SQLite; validation rejects bad
schema and impossible values;
tests pass
3. AQI and EDA
4
aqi.py CPCB calculation, EDA
notebook (seasonality, weather
correlations, missingness)
AQI matches hand-computed
examples for at least 10 test values;
EDA plots saved
4. Feature
engineering
5
build.py with lags, rolling,
weather, wind sin/cos, calendar,
events, Mode A and B
Leakage test passes (changing
future data never changes past
features); no NaNs after
documented imputation
5. Baselines and
Ridge
6
baselines.py , Ridge and OLS
Persistence and seasonal-naive
metrics computed; Ridge beats
persistence on synthetic data
6. Random
Forest and
tuning
7-8
train.py with TimeSeriesSplit
and RandomizedSearchCV,
optional HistGB and SARIMAX
RF tuned, models saved with
joblib, runs logged in model_runs
7. Evaluation
and
explainability
9
evaluate.py , explain.py ,
residual analysis
All metrics in metrics table;
permutation importance shows
wind speed as a key driver on
synthetic data
8. Backend API
10
FastAPI app with all endpoints and
live forecast module
API tests pass with TestClient;
/docs shows all routes
9. Frontend
11
HTML pages with Plotly.js charts
and what-if form
All pages load without console
errors and show data from the API
10. Report, tests,
polish
12
Auto report, README, coverage
over 70%, optional Dockerfile, real-
data run
python -m src.pipeline works
on real CSVs unchanged; demo
script ready
12. Testing Plan
Leakage tests: shifting or altering future rows must not change features computed for
earlier dates; train and test indexes never overlap and test dates are strictly later.
AQI tests: known concentration values return the expected sub-index and category,
including breakpoint edges.
Feature tests: lag and rolling features equal manually computed values on a small frame;
wind sin/cos in range.
Model tests: model beats persistence on synthetic data with a built-in weather dependence;
saved model loads and reproduces predictions.
Metric tests: MAE/RMSE on tiny arrays with hand-calculated answers.
API tests: each endpoint returns the right status and schema; /api/predict rejects
missing or out-of-range inputs with 422.
Smoke test: full pipeline on synthetic data runs end to end.
13. Risks and Mitigations
Risk
Mitigation
Missing sensor data and long gaps
Gap limits, station filtering, report coverage per station
Data leakage inflating scores
Strict shift rules, chronological split, leakage unit tests
Using observed future weather unfairly
Mode A is the default headline result; Mode B clearly
labelled
Class/extreme imbalance (few severe
days)
Report event recall and errors by pollution level, not only
overall RMSE
Random Forest cannot extrapolate
beyond training range
State limitation; compare with linear model and HistGB;
log-transform target as an experiment
Weather station and AQ station
mismatch
Use nearest grid point from Open-Meteo at station lat/lon
Open-Meteo/OpenAQ API downtime
Cache responses to disk; synthetic fallback; demo uses
stored data
Non-stationarity (policy change,
COVID-19 lockdown 2020)
Flag period, test with and without it, discuss in report
14. Final Deliverables Checklist
 Working code in repo with README and setup steps
 SQLite database with data, features, runs, metrics, predictions
 Trained model files and reproducible pipeline CLI
 FastAPI backend with Swagger docs
 Web dashboard (HTML + Plotly.js), no Streamlit
 Auto-generated evaluation report
 Final project report (Introduction, Literature Review, Data and Methodology,
Implementation, Results and Discussion, Conclusion and Future Work, References)
 Presentation slides and demo script
 Future work: LSTM/GRU or Temporal Fusion Transformer, satellite AOD features, multi-
day horizons, spatial interpolation across stations, uncertainty intervals (quantile forest),
mobile alerts
15. Claude Code Starter Prompts
Use in order:
1. 
Read PROJECT_SPEC_AQI.md. Set up the repo structure from Section 10,
requirements.txt, config.yaml and a README. Do Phase 1 only. Do not use
Streamlit.
2. 
Implement Phase 2 (data layer): synthetic generator, ingestion, cleaning,
SQLite schema from Section 8, with pytest tests. Run the tests.
3. 
Implement Phase 3: CPCB AQI calculation in aqi.py with config-driven
breakpoints and unit tests, plus an EDA notebook.
4. 
Implement Phase 4: feature engineering with Mode A and Mode B and the leakage
tests from Section 12.
5. 
Implement Phase 5 and 6: baselines, Ridge/OLS, Random Forest with
TimeSeriesSplit and RandomizedSearchCV, optional HistGB, log runs to
model_runs.
6. 
Implement Phase 7: evaluation (including category and high-pollution event
metrics), permutation importance, partial dependence and residual analysis.
7. 
Implement Phase 8: the FastAPI backend with all endpoints in Section 9,
Pydantic schemas, live forecast module with cached Open-Meteo calls, and API
tests.
8. 
Implement Phase 9: the static HTML + Plotly.js frontend pages listed in
Section 9, served by FastAPI. No Streamlit.
9. 
Generate the auto report, finish README, raise coverage to 70%+, add an
optional Dockerfile.
10. 
Add support for real data (Kaggle India AQ CSVs and Open-Meteo weather) using
the schemas in Section 4 and document how to prepare each file.
16. References (starting points)
CPCB: National Air Quality Index report (2014) and AQI breakpoint tables
Breiman, L. (2001): Random Forests
Hyndman & Athanasopoulos: Forecasting: Principles and Practice (time-series cross-
validation, baselines)
Pedregosa et al. (2011): scikit-learn
Open-Meteo, OpenAQ, CPCB CAAQMS portal documentation
Documentation: FastAPI, Pydantic, statsmodels, scikit-learn (TimeSeriesSplit,
permutation_importance)
