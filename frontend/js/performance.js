/**
 * performance.js — Model Performance page logic
 * - GET /api/models → comparison table (best row highlighted)
 * - Station dropdown + GET /api/predictions → predicted-vs-actual line + scatter
 * - GET /api/metrics (all scopes) → error-by-season bar chart
 */

'use strict';

let perfStation = '';
let bestRunId   = null;

(async function () {
  const content = await initPage({
    title: 'Model Performance',
    activeHref: 'performance.html',
    showStation: true,
    onStation: (id) => { perfStation = id; loadPredictions(id); },
  });

  makePlotsResponsive();
  if (content) {
    renderPerformanceLayout(content);
    await loadModels();
  }
})();

function renderPerformanceLayout(content) {
  content.innerHTML = `
    <!-- Models Table -->
    <section aria-label="Model comparison">
      <div class="section-header">
        <h2 class="section-title">Model Comparison</h2>
        <p class="section-subtitle">All trained runs — best row highlighted</p>
      </div>
      <div id="models-table-wrap" class="table-wrap mb-xl" role="region" aria-label="Model metrics table">
        ${skeletonCards(1)}
      </div>
    </section>

    <!-- Predicted vs Actual -->
    <section aria-label="Predicted vs actual">
      <div class="section-header">
        <h2 class="section-title">Predicted vs Actual — Test Period</h2>
        <p class="section-subtitle">Best model on the held-out test set</p>
      </div>
      <div class="chart-card mb-xl">
        <div class="chart-body">
          <div id="pred-actual-chart" class="chart-plot" style="min-height:340px;"
               aria-label="Predicted vs actual PM2.5 line chart"></div>
        </div>
      </div>
    </section>

    <!-- Scatter + Season -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:var(--space-xl);"
         class="two-col-charts">
      <section aria-label="Scatter plot">
        <div class="section-header" style="margin-bottom:var(--space-md);">
          <h2 class="section-title" style="font-size:17px;">y_true vs y_pred Scatter</h2>
        </div>
        <div class="chart-card">
          <div class="chart-body">
            <div id="scatter-chart" class="chart-plot" style="min-height:300px;"
                 aria-label="Scatter plot of true vs predicted PM2.5"></div>
          </div>
        </div>
      </section>

      <section aria-label="Error by season">
        <div class="section-header" style="margin-bottom:var(--space-md);">
          <h2 class="section-title" style="font-size:17px;">Error by Season</h2>
        </div>
        <div class="chart-card">
          <div class="chart-body">
            <div id="season-chart" class="chart-plot" style="min-height:300px;"
                 aria-label="MAE by season bar chart"></div>
          </div>
        </div>
      </section>
    </div>`;

  // Make two-col responsive
  const twoCol = content.querySelector('.two-col-charts');
  if (window.innerWidth <= 820 && twoCol) {
    twoCol.style.gridTemplateColumns = '1fr';
  }
  let _twoColTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(_twoColTimer);
    _twoColTimer = setTimeout(() => {
      if (!twoCol) return;
      twoCol.style.gridTemplateColumns = window.innerWidth <= 820 ? '1fr' : '1fr 1fr';
    }, 150);
  });
}

/* ---- Models Table ---- */
async function loadModels() {
  const wrap = document.getElementById('models-table-wrap');
  if (!wrap) return;

  const { data, error } = await fetchModels();

  if (error) {
    wrap.innerHTML = apiErrorState(error);
    return;
  }

  if (!Array.isArray(data) || data.length === 0) {
    wrap.innerHTML = emptyState({ icon: '🏋️', title: 'No model runs yet', msg: 'Run the training pipeline first.' });
    return;
  }

  // Find best run for downstream charts
  const best = data.find(r => r.is_best) || data[0];
  if (best) bestRunId = best.run_id;

  const headers = ['Model', 'Mode', 'MAE', 'RMSE', 'R²', 'Skill', 'Cat Acc', 'F1'];
  const headerRow = headers.map(h => `<th scope="col">${h}</th>`).join('');

  const rows = data.map(r => {
    const isBest = r.is_best;
    const cells = [
      `<strong>${escHtml((r.model_name||'').replace(/_/g,' '))}</strong>`,
      `<span class="badge badge-purple">${escHtml(r.feature_mode||'—')}</span>`,
      fmt(r.mae, 2),
      fmt(r.rmse, 2),
      fmt(r.r2, 3),
      fmt(r.skill_vs_persistence, 3),
      r.category_accuracy != null ? (r.category_accuracy * 100).toFixed(1) + '%' : '—',
      fmt(r.event_f1, 3),
    ].map(v => `<td>${v}</td>`).join('');
    return `<tr class="${isBest ? 'best-row' : ''}">${cells}</tr>`;
  }).join('');

  wrap.innerHTML = `
    <table class="data-table" role="table" aria-label="Model performance comparison">
      <thead><tr>${headerRow}</tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

  // Load predictions for default station if already set
  if (perfStation && bestRunId != null) {
    await loadPredictions(perfStation);
  }
}

/* ---- Predicted vs Actual ---- */
async function loadPredictions(stationId) {
  if (!stationId) return;

  const predEl    = document.getElementById('pred-actual-chart');
  const scatterEl = document.getElementById('scatter-chart');
  const seasonEl  = document.getElementById('season-chart');

  [predEl, scatterEl, seasonEl].forEach(el => {
    if (el) el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:200px;gap:8px;color:var(--text-secondary);">
      <span class="spinner"></span> Loading…</div>`;
  });

  const { data, error } = await fetchPredictions(stationId, bestRunId);

  if (error) {
    [predEl, scatterEl].forEach(el => { if (el) el.innerHTML = apiErrorState(error); });
  } else {
    const rows = (data && Array.isArray(data.rows)) ? data.rows : [];
    if (rows.length === 0) {
      [predEl, scatterEl].forEach(el => {
        if (el) el.innerHTML = emptyState({ icon: '📊', title: 'No predictions', msg: 'No test-period predictions in the database.' });
      });
    } else {
      renderPredVsActual(rows, predEl);
      renderScatter(rows, scatterEl);
    }
  }

  // Season chart from /api/metrics
  if (bestRunId != null && seasonEl) {
    await loadSeasonChart(bestRunId, seasonEl);
  }
}

function renderPredVsActual(rows, el) {
  if (!el) return;
  el.innerHTML = '';

  const dates  = rows.map(r => r.target_date);
  const yTrue  = rows.map(r => r.y_true);
  const yPred  = rows.map(r => r.y_pred);

  const traces = [
    {
      x: dates, y: yTrue, name: 'Actual PM2.5',
      type: 'scatter', mode: 'lines',
      line: { color: '#22d3ee', width: 2 },
      hovertemplate: '<b>%{x}</b><br>Actual: %{y:.1f} µg/m³<extra></extra>',
    },
    {
      x: dates, y: yPred, name: 'Predicted PM2.5',
      type: 'scatter', mode: 'lines',
      line: { color: '#a855f7', width: 2, dash: 'dot' },
      hovertemplate: '<b>%{x}</b><br>Predicted: %{y:.1f} µg/m³<extra></extra>',
    },
  ];

  const layout = mergeLayout({
    yaxis: { title: { text: 'PM2.5 (µg/m³)' } },
    hovermode: 'x unified',
    margin: { t: 10, r: 20, b: 50, l: 60 },
    legend: { x: 0, y: 1, xanchor: 'left', yanchor: 'top' },
  });

  Plotly.newPlot(el, traces, layout, PLOTLY_CONFIG);
}

function renderScatter(rows, el) {
  if (!el) return;
  el.innerHTML = '';

  const yTrue = rows.map(r => r.y_true);
  const yPred = rows.map(r => r.y_pred);
  const maxVal = Math.max(...yTrue, ...yPred, 1);

  const traces = [
    {
      x: yTrue, y: yPred,
      type: 'scatter', mode: 'markers',
      name: 'Predictions',
      marker: { color: '#22d3ee', size: 5, opacity: 0.65 },
      hovertemplate: 'True: %{x:.1f}<br>Pred: %{y:.1f}<extra></extra>',
    },
    {
      x: [0, maxVal], y: [0, maxVal],
      type: 'scatter', mode: 'lines',
      name: 'Perfect fit',
      line: { color: '#a855f7', width: 1.5, dash: 'dash' },
      hoverinfo: 'skip',
    },
  ];

  const layout = mergeLayout({
    xaxis: { title: { text: 'Actual PM2.5 (µg/m³)' } },
    yaxis: { title: { text: 'Predicted PM2.5 (µg/m³)' } },
    margin: { t: 10, r: 20, b: 60, l: 70 },
    showlegend: true,
  });

  Plotly.newPlot(el, traces, layout, PLOTLY_CONFIG);
}

async function loadSeasonChart(runId, el) {
  const { data, error } = await fetchMetrics(runId);

  if (error) {
    el.innerHTML = apiErrorState(error);
    return;
  }

  const rows = Array.isArray(data) ? data : [];
  const seasonRows = rows.filter(r => r.scope && r.scope.startsWith('season:'));

  if (seasonRows.length === 0) {
    el.innerHTML = emptyState({ icon: '🌦️', title: 'No seasonal metrics', msg: 'Train the model to generate per-season metrics.' });
    return;
  }

  el.innerHTML = '';

  const seasons = seasonRows.map(r => r.scope.replace('season:', '').replace(/_/g, ' '));
  const maes    = seasonRows.map(r => r.mae);
  const rmses   = seasonRows.map(r => r.rmse);

  const traces = [
    {
      x: seasons, y: maes, name: 'MAE',
      type: 'bar', marker: { color: '#22d3ee' },
      hovertemplate: '%{x}<br>MAE: %{y:.2f}<extra></extra>',
    },
    {
      x: seasons, y: rmses, name: 'RMSE',
      type: 'bar', marker: { color: '#a855f7' },
      hovertemplate: '%{x}<br>RMSE: %{y:.2f}<extra></extra>',
    },
  ];

  const layout = mergeLayout({
    barmode: 'group',
    yaxis: { title: { text: 'Error (µg/m³)' } },
    xaxis: { title: { text: 'Season' } },
    margin: { t: 10, r: 20, b: 60, l: 70 },
  });

  Plotly.newPlot(el, traces, layout, PLOTLY_CONFIG);
}
