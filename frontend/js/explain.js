/**
 * explain.js — Explainability page logic
 * - Run selector (defaults to best run)
 * - GET /api/feature-importance → horizontal bar chart
 *   (permutation importance with std error bars + impurity as second trace)
 * - Note about partial dependence
 */

'use strict';

let allModels = [];

(async function () {
  const content = await initPage({
    title: 'Explainability',
    activeHref: 'explain.html',
    showStation: false,
  });

  makePlotsResponsive();
  if (content) {
    renderExplainLayout(content);
    await loadModelsForSelector();
  }
})();

function renderExplainLayout(content) {
  content.innerHTML = `
    <!-- Run Selector -->
    <section class="chart-card mb-xl" aria-label="Model run selector">
      <div class="chart-body">
        <div style="display:flex;align-items:center;gap:var(--space-md);flex-wrap:wrap;">
          <div class="form-group" style="flex:1;min-width:200px;margin-bottom:0;">
            <label for="run-select" class="form-label">Select Model Run</label>
            <select id="run-select" class="form-control" onchange="loadFeatureImportance()"
                    aria-label="Select model run for feature importance">
              <option value="">Loading runs…</option>
            </select>
          </div>
          <div class="form-group" style="margin-bottom:0;align-self:flex-end;">
            <button class="btn btn-primary" onclick="loadFeatureImportance()">
              Load Importances
            </button>
          </div>
        </div>
      </div>
    </section>

    <!-- Feature Importance Chart -->
    <section class="chart-card mb-xl" aria-label="Feature importance chart">
      <div class="chart-header">
        <div>
          <div class="chart-title">Feature Importance — Top Features</div>
          <div class="chart-subtitle">Permutation importance (with std error bars) vs impurity importance</div>
        </div>
      </div>
      <div class="chart-body">
        <div id="importance-chart" class="chart-plot" style="min-height:420px;"
             aria-label="Horizontal bar chart of feature importances"></div>
      </div>
    </section>

    <!-- Note about PDP -->
    <section class="about-section" aria-label="Methodology note">
      <h2>Partial Dependence & Interpretation Notes</h2>
      <p>
        <strong>Permutation importance</strong> (preferred) measures the drop in model accuracy when
        a feature's values are randomly shuffled on the test set. A large drop means the feature is
        genuinely relied upon by the model, not just correlated with the target.
      </p>
      <p>
        <strong>Impurity importance</strong> (mean decrease in node impurity) is computed during
        Random Forest training and can be biased toward high-cardinality or noisy features.
        Use permutation importance as the primary signal.
      </p>
      <h3>Partial Dependence Plots</h3>
      <p>
        Partial dependence plots (PDPs) show the marginal effect of a single feature on the
        predicted PM2.5 while averaging over all other features. Key drivers to inspect:
      </p>
      <ul>
        <li><strong>Wind speed</strong> — negative relationship expected (higher wind → better dispersion)</li>
        <li><strong>Humidity</strong> — positive at high values (hygroscopic growth of PM)</li>
        <li><strong>Temperature</strong> — negative in winter (cold = trapped boundary layer)</li>
        <li><strong>PM2.5 lag-1</strong> — strong autocorrelation; dominates short-term forecasts</li>
      </ul>
      <p class="text-muted" style="font-size:12px;margin-top:var(--space-md);">
        Note: PDPs assume feature independence, which may not hold for correlated weather variables.
        SHAP interaction plots are listed as future work.
      </p>
    </section>`;
}

async function loadModelsForSelector() {
  const sel = document.getElementById('run-select');
  if (!sel) return;

  const { data, error } = await fetchModels();

  if (error || !Array.isArray(data) || data.length === 0) {
    sel.innerHTML = '<option value="">No runs available</option>';
    if (error) showToast(error, 'error');
    // Try loading with no run_id (best)
    loadFeatureImportance();
    return;
  }

  allModels = data;
  sel.innerHTML = data.map(r => {
    const label = `${(r.model_name||'').replace(/_/g,' ')} | Mode ${r.feature_mode||'?'} | Run ${r.run_id}${r.is_best ? ' ★ Best' : ''}`;
    return `<option value="${r.run_id}" ${r.is_best ? 'selected' : ''}>${label}</option>`;
  }).join('');

  loadFeatureImportance();
}

async function loadFeatureImportance() {
  const el    = document.getElementById('importance-chart');
  const runId = document.getElementById('run-select')?.value;
  if (!el) return;

  el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:300px;gap:8px;color:var(--text-secondary);">
    <span class="spinner"></span> Loading importances…</div>`;

  const { data, error } = await fetchFeatureImportance(runId || undefined);

  if (error) {
    el.innerHTML = apiErrorState(error);
    return;
  }

  if (!Array.isArray(data) || data.length === 0) {
    el.innerHTML = emptyState({ icon: '🧪', title: 'No importances', msg: 'Train a model to see feature importance.' });
    return;
  }

  el.innerHTML = '';
  renderImportanceChart(data, el);
}

function renderImportanceChart(feats, el) {
  // Sort by permutation_importance desc, take top 20
  const sorted = [...feats]
    .sort((a, b) => (b.permutation_importance || 0) - (a.permutation_importance || 0))
    .slice(0, 20);

  const names  = sorted.map(f => f.feature);
  const perm   = sorted.map(f => f.permutation_importance);
  const permStd = sorted.map(f => f.permutation_std || 0);
  const impurity = sorted.map(f => f.impurity_importance);

  const traces = [
    {
      y: names, x: perm,
      name: 'Permutation Importance',
      type: 'bar', orientation: 'h',
      marker: {
        color: perm.map((_, i) => {
          const alpha = 0.4 + 0.6 * (1 - i / sorted.length);
          return `rgba(34,211,238,${alpha.toFixed(2)})`;
        }),
        line: { color: '#22d3ee', width: 1 },
      },
      error_x: {
        type: 'data',
        array: permStd,
        visible: true,
        color: 'rgba(255,255,255,0.3)',
        thickness: 1.5,
        width: 4,
      },
      hovertemplate: '<b>%{y}</b><br>Permutation: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>',
    },
    {
      y: names, x: impurity,
      name: 'Impurity Importance',
      type: 'bar', orientation: 'h',
      marker: { color: 'rgba(168,85,247,0.55)', line: { color: '#a855f7', width: 1 } },
      hovertemplate: '<b>%{y}</b><br>Impurity: %{x:.4f}<extra></extra>',
    },
  ];

  const layout = mergeLayout({
    barmode: 'group',
    yaxis: {
      autorange: 'reversed',
      tickfont: { size: 11 },
      automargin: true,
    },
    xaxis: { title: { text: 'Importance Score' } },
    margin: { t: 10, r: 20, b: 60, l: 160 },
    legend: { x: 0.5, y: -0.12, xanchor: 'center', yanchor: 'top', orientation: 'h' },
    height: Math.max(400, sorted.length * 26 + 100),
  });

  Plotly.newPlot(el, traces, layout, PLOTLY_CONFIG);
}
