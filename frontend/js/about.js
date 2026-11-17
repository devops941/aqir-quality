/**
 * about.js — About / Data page (static content)
 */

'use strict';

(async function () {
  const content = await initPage({
    title: 'About & Data',
    activeHref: 'about.html',
    showStation: false,
  });

  if (content) renderAboutPage(content);
})();

function renderAboutPage(content) {
  content.innerHTML = `
    <!-- Hero -->
    <div class="section-header">
      <h2 class="section-title">Air Quality Forecasting Dashboard</h2>
      <p class="section-subtitle">
        Next-day PM2.5 and AQI prediction using machine learning on Indian urban air quality data.
      </p>
    </div>

    <!-- Data Sources -->
    <section class="about-section" aria-label="Data sources">
      <h2>Data Sources</h2>

      <h3>Air Quality Data — CPCB / Kaggle</h3>
      <p>
        Primary source: <strong>Kaggle "Air Quality Data in India (2015–2020)"</strong> which
        aggregates CPCB CAAQMS continuous monitoring data for Indian cities. Includes daily and
        hourly measurements of PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene,
        and Xylene from multiple ground stations.
      </p>
      <ul>
        <li><code>city_day.csv</code> — city-level daily averages</li>
        <li><code>station_day.csv</code> — station-level daily averages</li>
        <li><code>station_hour.csv</code> — station-level hourly readings</li>
      </ul>
      <p>
        A synthetic fallback generator (<code>src/data/synthetic.py</code>) creates 3 years of
        hourly data for 3 Delhi stations when real CSVs are not present, so the pipeline runs
        end-to-end without downloads.
      </p>

      <h3>Weather Data — Open-Meteo</h3>
      <p>
        Historical weather is fetched from the
        <a href="https://open-meteo.com/" target="_blank" rel="noopener">Open-Meteo Historical Weather API</a>
        (free, no API key required) at each station's latitude/longitude. Variables used:
      </p>
      <ul>
        <li>Temperature (°C)</li>
        <li>Relative humidity (%)</li>
        <li>Wind speed (m/s) and direction (°)</li>
        <li>Surface pressure (hPa)</li>
        <li>Total precipitation (mm)</li>
        <li>Cloud cover (%)</li>
      </ul>
      <p>
        Live-mode forecasts use the
        <a href="https://open-meteo.com/en/docs" target="_blank" rel="noopener">Open-Meteo Forecast API</a>
        to fetch next-day weather for Mode B.
      </p>

      <h3>Study Area</h3>
      <p>
        Default city: <strong>Delhi, India</strong> — three stations:
        Anand Vihar (DL001), R K Puram (DL002), and Punjabi Bagh (DL003).
        Station coordinates are defined in <code>config.yaml</code>.
      </p>
    </section>

    <!-- Methodology -->
    <section class="about-section" aria-label="Methodology">
      <h2>Methodology</h2>

      <h3>AQI Calculation</h3>
      <p>
        Indian AQI follows the <strong>CPCB National Air Quality Index (2014)</strong> standard.
        A sub-index is computed for each available pollutant using piecewise linear interpolation
        between breakpoint bands. The overall AQI is the maximum sub-index, and is valid only when
        at least 3 pollutants are measured, including at least one of PM2.5 or PM10.
      </p>

      <h3>AQI Categories</h3>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:var(--space-sm);margin:var(--space-md) 0;">
        ${[
          { name: 'Good',         range: '0–50',   color: '#2ecc71', health: 'Minimal impact.' },
          { name: 'Satisfactory', range: '51–100',  color: '#9acd32', health: 'Minor breathing discomfort to sensitive people.' },
          { name: 'Moderate',     range: '101–200', color: '#f1c40f', health: 'Breathing discomfort to people with lung/heart disease.' },
          { name: 'Poor',         range: '201–300', color: '#e67e22', health: 'Breathing discomfort to most on prolonged exposure.' },
          { name: 'Very Poor',    range: '301–400', color: '#e74c3c', health: 'Respiratory illness on prolonged exposure.' },
          { name: 'Severe',       range: '401–500', color: '#7e0023', health: 'Affects healthy people; serious impact on those with illness.' },
        ].map(c => `
          <div style="display:flex;align-items:flex-start;gap:var(--space-sm);padding:10px;
                      border-radius:var(--radius-md);border:1px solid ${c.color}20;
                      background:${c.color}0d;">
            <span style="display:inline-block;width:14px;height:14px;border-radius:50%;
                         background:${c.color};flex-shrink:0;margin-top:3px;"></span>
            <div>
              <strong style="color:${c.color};font-size:13px;">${c.name}</strong>
              <span style="color:var(--text-muted);font-size:12px;"> (${c.range})</span>
              <p style="font-size:12px;color:var(--text-muted);margin:2px 0 0;">${c.health}</p>
            </div>
          </div>`).join('')}
      </div>

      <h3>Feature Engineering</h3>
      <p>
        Features are engineered at the end of day <em>t</em> to predict PM2.5 for day <em>t+1</em>.
        All lags use <code>shift(k)</code> with k ≥ 1 to prevent data leakage:
      </p>
      <ul>
        <li><strong>Pollutant lags:</strong> PM2.5 at t, t−1, t−2, t−3, t−7; same for PM10 and NO2</li>
        <li><strong>Rolling statistics:</strong> 3-day and 7-day mean/max/std of PM2.5</li>
        <li><strong>Weather at day t:</strong> temperature, humidity, wind speed/direction, pressure, precipitation, cloud cover; change from t−1</li>
        <li><strong>Wind encoding:</strong> sin and cos of wind direction (cyclical)</li>
        <li><strong>Calendar:</strong> day-of-week, month, season, weekend flag, cyclic month encoding</li>
        <li><strong>Events:</strong> Diwali, crop-burning season (Oct–Nov North India)</li>
      </ul>

      <h3>Mode A vs Mode B</h3>
      <p>
        <strong>Mode A — Strict forecast (default):</strong> uses only information available at the
        end of day t. No future weather. This is the operationally honest evaluation.
      </p>
      <p>
        <strong>Mode B — With weather forecast:</strong> includes next-day weather (observed in
        training, forecasted in production). Mode B scores are optimistic upper bounds — a perfect
        weather forecast is assumed. Always compare headline results using Mode A.
      </p>

      <h3>Models</h3>
      <ul>
        <li><strong>Persistence baseline:</strong> PM2.5(t+1) = PM2.5(t)</li>
        <li><strong>Seasonal-naive baseline:</strong> same weekday from previous week</li>
        <li><strong>Ridge regression:</strong> L2-regularised linear model on standardised features</li>
        <li><strong>Random Forest:</strong> tuned with <code>RandomizedSearchCV</code> + <code>TimeSeriesSplit(n_splits=5)</code></li>
        <li><strong>HistGradientBoosting:</strong> (optional) gradient boosted trees</li>
        <li><strong>SARIMAX:</strong> (optional) seasonal ARIMA with weather as exogenous variables</li>
      </ul>

      <h3>Evaluation Metrics</h3>
      <ul>
        <li>MAE (Mean Absolute Error), RMSE, R², MAPE</li>
        <li>Skill score vs persistence: 1 − RMSE_model / RMSE_persistence</li>
        <li>AQI category accuracy and confusion matrix</li>
        <li>High-pollution event (PM2.5 ≥ 90 µg/m³): precision, recall, F1</li>
        <li>Error breakdowns by season, station, and pollution level</li>
      </ul>
    </section>

    <!-- Limitations -->
    <section class="about-section" aria-label="Limitations">
      <h2>Limitations &amp; Caveats</h2>
      <ul>
        <li>
          <strong>Random Forest cannot extrapolate</strong> beyond the range of values seen in training.
          Extreme pollution events may be under-estimated.
        </li>
        <li>
          <strong>Mode B results are optimistic</strong> — they assume perfect knowledge of next-day
          weather. Real-world forecast accuracy will be lower.
        </li>
        <li>
          <strong>Synthetic data</strong> captures known seasonal patterns but lacks the full complexity
          of real CPCB observations. Results on real data may differ.
        </li>
        <li>
          <strong>Single-step forecasting only</strong> — predicts 24 hours ahead. Multi-day horizons
          and uncertainty intervals are listed as future work.
        </li>
        <li>
          <strong>No satellite inputs</strong> — aerosol optical depth (AOD) from MODIS/VIIRS is not
          currently included.
        </li>
        <li>
          <strong>Non-stationarity:</strong> COVID-19 lockdowns (2020) dramatically reduced pollution;
          the model may behave unexpectedly on data from that period.
        </li>
      </ul>
    </section>

    <!-- Links -->
    <section class="about-section" aria-label="Links and references">
      <h2>Links</h2>
      <ul>
        <li><a href="/docs" target="_blank" rel="noopener">Swagger API Documentation (/docs)</a></li>
        <li><a href="https://cpcb.nic.in/" target="_blank" rel="noopener">CPCB — Central Pollution Control Board</a></li>
        <li><a href="https://open-meteo.com/" target="_blank" rel="noopener">Open-Meteo Weather API</a></li>
        <li><a href="https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india" target="_blank" rel="noopener">Kaggle — Air Quality Data in India</a></li>
        <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html" target="_blank" rel="noopener">scikit-learn: permutation_importance</a></li>
      </ul>

      <div style="margin-top:var(--space-xl);padding:var(--space-md);
                  background:linear-gradient(135deg,rgba(34,211,238,0.06) 0%,rgba(168,85,247,0.06) 100%);
                  border:1px solid rgba(34,211,238,0.15);border-radius:var(--radius-lg);">
        <p style="font-size:13px;color:var(--text-secondary);margin:0;">
          <strong style="background:var(--accent-grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
            Phase 9 — Frontend
          </strong>
          &nbsp;·&nbsp; Vanilla HTML + CSS + JavaScript + Plotly.js
          &nbsp;·&nbsp; Served as static files by FastAPI at <code>/</code>
          &nbsp;·&nbsp; API under <code>/api</code>
        </p>
      </div>
    </section>`;
}
