/**
 * home.js — Home / Forecast page logic
 * Shows tomorrow's forecast AQI badge, KPI cards, and 14-day sparkline.
 */

'use strict';

(async function () {
  const content = await initPage({
    title: 'Tomorrow\'s Forecast',
    activeHref: 'index.html',
    showStation: true,
    onStation: loadForecast,
  });

  makePlotsResponsive();
  renderSkeleton(content);
})();

function renderSkeleton(content) {
  if (!content) return;
  content.innerHTML = skeletonCards(4) + skeletonChart();
}

async function loadForecast(stationId) {
  const content = document.getElementById('page-content');
  if (!content || !stationId) return;

  content.innerHTML = skeletonCards(4) + skeletonChart();

  // Parallel fetch: forecast + history (last 14 rows)
  const [fRes, hRes] = await Promise.all([
    fetchForecast(stationId),
    fetchHistory(stationId, null, null, 'pm25,aqi'),
  ]);

  content.innerHTML = '';

  // ---- Forecast Section ----
  if (fRes.error) {
    content.innerHTML += apiErrorState(fRes.error);
  } else {
    content.innerHTML += buildForecastHero(fRes.data);
    content.innerHTML += buildKpiCards(fRes.data);
  }

  // ---- 14-day Sparkline ----
  if (hRes.error) {
    content.innerHTML += `<div class="chart-card"><div class="chart-body">${apiErrorState(hRes.error)}</div></div>`;
  } else {
    const rows = Array.isArray(hRes.data && hRes.data.rows) ? hRes.data.rows : [];
    const last14 = rows.slice(-14);
    content.innerHTML += buildSparklineCard(last14);
    if (last14.length > 0) {
      renderSparkline(last14);
    }
  }
}

function buildForecastHero(d) {
  if (!d) return emptyState({ icon: '📡', title: 'No forecast available', msg: 'Run the pipeline to generate forecasts.' });

  const aqiBadge = buildAqiBadge(d.aqi_pred, d.category, d.health);
  const issuedAt = fmtDatetime(d.issued_at);
  const targetDate = fmtDate(d.target_date);

  return `
    <section class="forecast-hero" aria-label="Tomorrow's air quality forecast">
      <div class="aqi-badge-wrap">
        ${aqiBadge}
      </div>
      <div>
        <div style="margin-bottom:var(--space-md);">
          <h2 class="section-title">
            Forecast for <span class="gradient-text">${escHtml(d.station_name || stationId)}</span>
          </h2>
          <p class="section-subtitle">Target date: <strong>${targetDate}</strong></p>
        </div>
        <div class="forecast-meta">
          <p>Issued at: <strong>${issuedAt}</strong></p>
          <p>Model: <strong>${escHtml(d.model_name || '—')}</strong> &nbsp;|&nbsp; Run ID: <strong>${d.run_id || '—'}</strong></p>
        </div>
      </div>
    </section>`;
}

function buildKpiCards(d) {
  if (!d) return '';

  const cfg = getCategoryConfig(d.category) || getCategoryByAqi(d.aqi_pred);
  const color = cfg ? cfg.color : '#94a3b8';

  const cards = [
    { label: 'PM2.5 Predicted', value: fmt(d.pm25_pred, 1), unit: 'µg/m³' },
    { label: 'AQI Predicted',   value: fmt(d.aqi_pred, 0),  unit: '' },
    { label: 'Category',        value: escHtml(d.category || '—'), unit: '', style: `color:${color};` },
    { label: 'Model Used',      value: escHtml((d.model_name || '—').replace(/_/g, ' ')), unit: '', small: true },
  ];

  const html = cards.map(c => `
    <article class="kpi-card" aria-label="${c.label}: ${c.value}${c.unit}">
      <div class="kpi-label">${c.label}</div>
      <div class="kpi-value${c.small ? '" style="font-size:22px;' : ''}" style="${c.style || ''}">
        ${c.value}<span class="kpi-unit">${c.unit}</span>
      </div>
    </article>`).join('');

  return `<div class="kpi-grid" role="list" aria-label="Forecast key metrics">${html}</div>`;
}

function buildSparklineCard(rows) {
  const count = rows.length;
  return `
    <section class="chart-card" aria-label="Recent PM2.5 trend">
      <div class="chart-header">
        <div>
          <div class="chart-title">Recent PM2.5 Trend</div>
          <div class="chart-subtitle">Last ${count} days of observations</div>
        </div>
      </div>
      <div class="chart-body">
        ${count === 0
          ? emptyState({ icon: '📉', title: 'No historical data', msg: 'No data yet — run the pipeline to populate history.' })
          : '<div id="sparkline-chart" class="chart-plot" aria-label="14-day PM2.5 line chart"></div>'
        }
      </div>
    </section>`;
}

function renderSparkline(rows) {
  const el = document.getElementById('sparkline-chart');
  if (!el || !rows.length) return;

  const dates  = rows.map(r => r.date);
  const pm25   = rows.map(r => r.pm25);
  const aqiVals = rows.map(r => r.aqi);

  const trace = {
    x: dates,
    y: pm25,
    name: 'PM2.5 µg/m³',
    type: 'scatter',
    mode: 'lines+markers',
    line: { color: '#22d3ee', width: 2.5, shape: 'spline' },
    marker: { size: 5, color: '#22d3ee' },
    fill: 'tozeroy',
    fillcolor: 'rgba(34,211,238,0.07)',
    hovertemplate: '<b>%{x}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>',
  };

  // AQI as secondary if available
  const aqiTrace = aqiVals.some(v => v != null) ? {
    x: dates,
    y: aqiVals,
    name: 'AQI',
    type: 'scatter',
    mode: 'lines',
    line: { color: '#a855f7', width: 1.5, dash: 'dot' },
    yaxis: 'y2',
    hovertemplate: '<b>%{x}</b><br>AQI: %{y:.0f}<extra></extra>',
  } : null;

  const traces = aqiTrace ? [trace, aqiTrace] : [trace];

  const layout = mergeLayout({
    yaxis:  { title: { text: 'PM2.5 (µg/m³)', font: { size: 11 } } },
    yaxis2: aqiTrace ? {
      title: { text: 'AQI', font: { size: 11 } },
      overlaying: 'y', side: 'right',
      gridcolor: 'rgba(255,255,255,0.03)',
    } : undefined,
    margin: { t: 10, r: aqiTrace ? 60 : 20, b: 50, l: 60 },
    hovermode: 'x unified',
  });

  Plotly.newPlot(el, traces, layout, PLOTLY_CONFIG);
}
