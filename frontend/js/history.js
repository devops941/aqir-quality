/**
 * history.js — History page logic
 * - Station + date-range filter → GET /api/history
 * - Interactive PM2.5 time series with weather overlays
 * - Month×year seasonal heatmap
 */

'use strict';

(async function () {
  const content = await initPage({
    title: 'Historical Data',
    activeHref: 'history.html',
    showStation: true,
    onStation: (id) => { currentStation = id; onFilterChange(); },
  });

  makePlotsResponsive();
  renderHistoryLayout(content);
})();

let currentStation = '';

function renderHistoryLayout(content) {
  if (!content) return;
  content.innerHTML = `
    <!-- Date Range Controls -->
    <section class="chart-card mb-xl" aria-label="Date range filter">
      <div class="chart-header">
        <div>
          <div class="chart-title">Date Range</div>
          <div class="chart-subtitle">Filter historical observations</div>
        </div>
      </div>
      <div class="chart-body">
        <div class="date-row">
          <div class="form-group">
            <label for="date-start" class="form-label">From</label>
            <input type="date" id="date-start" class="form-control" aria-label="Start date" />
          </div>
          <div class="form-group">
            <label for="date-end" class="form-label">To</label>
            <input type="date" id="date-end" class="form-control" aria-label="End date" />
          </div>
          <div class="form-group" style="align-self:flex-end;">
            <button class="btn btn-primary" id="apply-filter-btn" onclick="onFilterChange()">
              Apply Filter
            </button>
          </div>
        </div>
        <!-- Toggle overlays -->
        <div style="margin-top:var(--space-md);display:flex;gap:var(--space-md);flex-wrap:wrap;">
          <label style="font-size:13px;color:var(--text-secondary);display:flex;align-items:center;gap:6px;cursor:pointer;">
            <input type="checkbox" id="overlay-temp"  checked onchange="onFilterChange()"> Temperature
          </label>
          <label style="font-size:13px;color:var(--text-secondary);display:flex;align-items:center;gap:6px;cursor:pointer;">
            <input type="checkbox" id="overlay-wind"  onchange="onFilterChange()"> Wind Speed
          </label>
          <label style="font-size:13px;color:var(--text-secondary);display:flex;align-items:center;gap:6px;cursor:pointer;">
            <input type="checkbox" id="overlay-humid" onchange="onFilterChange()"> Humidity
          </label>
        </div>
      </div>
    </section>

    <!-- Time Series Chart -->
    <section class="chart-card mb-xl" aria-label="PM2.5 time series">
      <div class="chart-header">
        <div>
          <div class="chart-title">PM2.5 Over Time</div>
          <div class="chart-subtitle">With optional weather overlays on secondary axis</div>
        </div>
      </div>
      <div class="chart-body">
        <div id="timeseries-chart" class="chart-plot" style="min-height:360px;"
             aria-label="PM2.5 time series chart"></div>
      </div>
    </section>

    <!-- Seasonal Heatmap -->
    <section class="chart-card" aria-label="Seasonal heatmap">
      <div class="chart-header">
        <div>
          <div class="chart-title">Seasonal Heatmap</div>
          <div class="chart-subtitle">Mean PM2.5 by month × year</div>
        </div>
      </div>
      <div class="chart-body">
        <div id="heatmap-chart" class="chart-plot" style="min-height:300px;"
             aria-label="PM2.5 seasonal heatmap"></div>
      </div>
    </section>`;

  // Set default date range: last 90 days
  const today = new Date();
  const ago90 = new Date(today);
  ago90.setDate(ago90.getDate() - 90);
  document.getElementById('date-end').value   = today.toISOString().slice(0, 10);
  document.getElementById('date-start').value = ago90.toISOString().slice(0, 10);
}

async function onFilterChange() {
  if (!currentStation) return;
  const start  = document.getElementById('date-start')?.value || '';
  const end    = document.getElementById('date-end')?.value   || '';

  showChartLoading('timeseries-chart');
  showChartLoading('heatmap-chart');

  const { data, error } = await fetchHistory(currentStation, start, end);

  if (error) {
    setChartError('timeseries-chart', error);
    setChartError('heatmap-chart', error);
    return;
  }

  const rows = (data && Array.isArray(data.rows)) ? data.rows : [];

  if (rows.length === 0) {
    setChartEmpty('timeseries-chart');
    setChartEmpty('heatmap-chart');
    return;
  }

  renderTimeSeries(rows);
  renderHeatmap(rows);
}

function showChartLoading(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:300px;gap:var(--space-sm);color:var(--text-secondary);">
    <span class="spinner"></span> Loading…</div>`;
}

function setChartError(id, msg) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = apiErrorState(msg);
}

function setChartEmpty(id) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = emptyState({ icon: '📉', title: 'No data', msg: 'No data for this date range.' });
}

function renderTimeSeries(rows) {
  const el = document.getElementById('timeseries-chart');
  if (!el) return;
  el.innerHTML = '';

  const showTemp  = document.getElementById('overlay-temp')?.checked;
  const showWind  = document.getElementById('overlay-wind')?.checked;
  const showHumid = document.getElementById('overlay-humid')?.checked;

  const dates = rows.map(r => r.date);
  const pm25  = rows.map(r => r.pm25);

  const traces = [{
    x: dates,
    y: pm25,
    name: 'PM2.5 µg/m³',
    type: 'scatter',
    mode: 'lines',
    line: { color: '#22d3ee', width: 2 },
    fill: 'tozeroy',
    fillcolor: 'rgba(34,211,238,0.06)',
    hovertemplate: '<b>%{x}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>',
  }];

  const hasY2 = showTemp || showWind || showHumid;

  if (showTemp) {
    traces.push({
      x: dates, y: rows.map(r => r.temp_c),
      name: 'Temp (°C)', type: 'scatter', mode: 'lines',
      line: { color: '#f59e0b', width: 1.5, dash: 'dot' }, yaxis: 'y2',
      hovertemplate: '<b>%{x}</b><br>Temp: %{y:.1f}°C<extra></extra>',
    });
  }
  if (showWind) {
    traces.push({
      x: dates, y: rows.map(r => r.wind_speed_ms),
      name: 'Wind (m/s)', type: 'scatter', mode: 'lines',
      line: { color: '#10b981', width: 1.5, dash: 'dashdot' }, yaxis: 'y2',
      hovertemplate: '<b>%{x}</b><br>Wind: %{y:.1f} m/s<extra></extra>',
    });
  }
  if (showHumid) {
    traces.push({
      x: dates, y: rows.map(r => r.humidity_pct),
      name: 'Humidity (%)', type: 'scatter', mode: 'lines',
      line: { color: '#6366f1', width: 1.5, dash: 'dash' }, yaxis: 'y2',
      hovertemplate: '<b>%{x}</b><br>Humidity: %{y:.0f}%<extra></extra>',
    });
  }

  const layout = mergeLayout({
    yaxis:  { title: { text: 'PM2.5 (µg/m³)' }, rangemode: 'tozero' },
    yaxis2: hasY2 ? {
      title: { text: 'Weather' },
      overlaying: 'y', side: 'right',
      gridcolor: 'rgba(255,255,255,0.0)',
    } : undefined,
    margin: { t: 10, r: hasY2 ? 70 : 20, b: 50, l: 60 },
    hovermode: 'x unified',
    xaxis: { rangeslider: { visible: true, thickness: 0.06 } },
  });

  Plotly.newPlot(el, traces, layout, PLOTLY_CONFIG);
}

function renderHeatmap(rows) {
  const el = document.getElementById('heatmap-chart');
  if (!el) return;
  el.innerHTML = '';

  // Build month×year grid
  const grid = {};  // { year: { month: [values] } }
  for (const r of rows) {
    if (r.pm25 == null) continue;
    const d = new Date(r.date + 'T00:00:00');
    const y = d.getFullYear();
    const m = d.getMonth(); // 0-indexed
    if (!grid[y]) grid[y] = {};
    if (!grid[y][m]) grid[y][m] = [];
    grid[y][m].push(r.pm25);
  }

  const years  = Object.keys(grid).sort();
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  if (years.length === 0) {
    el.innerHTML = emptyState({ icon: '📅', title: 'No heatmap data' });
    return;
  }

  const z = months.map((_, mi) =>
    years.map(y => {
      const vals = (grid[y] && grid[y][mi]) || [];
      if (vals.length === 0) return null;
      return vals.reduce((a, b) => a + b, 0) / vals.length;
    })
  );

  const trace = {
    z,
    x: years,
    y: months,
    type: 'heatmap',
    colorscale: [
      [0,    '#2ecc71'],
      [0.1,  '#9acd32'],
      [0.3,  '#f1c40f'],
      [0.55, '#e67e22'],
      [0.75, '#e74c3c'],
      [1.0,  '#7e0023'],
    ],
    zmin: 0, zmax: 500,
    colorbar: { title: 'PM2.5 µg/m³', titleside: 'right', tickfont: { size: 11 } },
    hoverongaps: false,
    hovertemplate: '<b>%{y} %{x}</b><br>Mean PM2.5: %{z:.1f} µg/m³<extra></extra>',
  };

  const layout = mergeLayout({
    margin: { t: 10, r: 90, b: 50, l: 60 },
    xaxis: { type: 'category', title: { text: 'Year' } },
    yaxis: { autorange: 'reversed', title: { text: 'Month' } },
  });

  Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}
