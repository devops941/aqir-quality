/**
 * whatif.js — What-if predictor page logic
 * - Form with all required/optional fields
 * - Client-side validation before POST /api/predict
 * - Display predicted PM2.5 + AQI badge + category + health
 */

'use strict';

(async function () {
  const content = await initPage({
    title: 'What-if Predictor',
    activeHref: 'whatif.html',
    showStation: false,
  });

  if (content) renderWhatIfPage(content);
})();

/* ---- Field definitions ---- */
const FIELDS = [
  // Required
  { id: 'recent_pm25', label: 'Recent PM2.5 (µg/m³)', type: 'number', required: true, min: 0,    max: 2000, step: 0.1,  hint: 'Today\'s mean PM2.5. Required, ≥ 0.' },
  { id: 'temp_c',       label: 'Temperature (°C)',     type: 'number', required: true, min: -20,  max: 60,   step: 0.1,  hint: 'Mean daily temperature. Required, −20 to 60.' },
  { id: 'humidity_pct', label: 'Humidity (%)',         type: 'number', required: true, min: 0,    max: 100,  step: 0.1,  hint: 'Relative humidity 0–100. Required.' },
  { id: 'wind_speed_ms',label: 'Wind Speed (m/s)',     type: 'number', required: true, min: 0,    max: 60,   step: 0.1,  hint: 'Surface wind speed. Required, 0–60.' },
  { id: 'precip_mm',    label: 'Precipitation (mm)',   type: 'number', required: true, min: 0,    max: 500,  step: 0.1,  hint: 'Daily total precipitation. Required, ≥ 0.' },
  // Optional
  { id: 'pm25_lag1',    label: 'PM2.5 Lag-1 (µg/m³)', type: 'number', required: false, min: 0,  max: 2000, step: 0.1,  hint: 'Yesterday\'s PM2.5 (optional).' },
  { id: 'pm25_lag7',    label: 'PM2.5 Lag-7 (µg/m³)', type: 'number', required: false, min: 0,  max: 2000, step: 0.1,  hint: 'PM2.5 7 days ago (optional).' },
  { id: 'pm10',         label: 'PM10 (µg/m³)',         type: 'number', required: false, min: 0,  max: 2000, step: 0.1,  hint: 'Today\'s PM10 (optional).' },
  { id: 'no2',          label: 'NO₂ (µg/m³)',          type: 'number', required: false, min: 0,  max: 500,  step: 0.1,  hint: 'Today\'s NO2 (optional).' },
  { id: 'wind_dir_deg', label: 'Wind Direction (°)',   type: 'number', required: false, min: 0,  max: 360,  step: 1,    hint: 'Degrees 0–360 (optional).' },
  { id: 'pressure_hpa', label: 'Pressure (hPa)',       type: 'number', required: false, min: 800,max: 1200, step: 0.1,  hint: 'Surface pressure (optional).' },
  { id: 'cloud_pct',    label: 'Cloud Cover (%)',      type: 'number', required: false, min: 0,  max: 100,  step: 0.1,  hint: 'Cloud cover 0–100% (optional).' },
  { id: 'month',        label: 'Month (1–12)',         type: 'number', required: false, min: 1,  max: 12,   step: 1,    hint: 'Calendar month 1–12 (optional; defaults to today).' },
];

function renderWhatIfPage(content) {
  const today = new Date();

  const requiredFields = FIELDS.filter(f => f.required);
  const optionalFields = FIELDS.filter(f => !f.required);

  function fieldHtml(f) {
    const reqAttr = f.required ? 'required aria-required="true"' : '';
    const minAttr = f.min != null ? `min="${f.min}"` : '';
    const maxAttr = f.max != null ? `max="${f.max}"` : '';
    return `
      <div class="form-group">
        <label for="${f.id}" class="form-label">
          ${f.label}${f.required ? ' <span style="color:#ef4444;">*</span>' : ''}
        </label>
        <input type="${f.type}" id="${f.id}" name="${f.id}"
               class="form-control"
               ${minAttr} ${maxAttr} step="${f.step}"
               ${reqAttr}
               placeholder="${f.required ? 'Required' : 'Optional'}"
               aria-label="${f.label}" />
        <span class="form-hint">${f.hint}</span>
        <span class="form-error" id="${f.id}-error">Please enter a valid value.</span>
      </div>`;
  }

  content.innerHTML = `
    <div class="two-panel">
      <!-- Form Panel -->
      <div>
        <div class="section-header">
          <h2 class="section-title">Enter Conditions</h2>
          <p class="section-subtitle">Required fields marked with <span style="color:#ef4444;">*</span></p>
        </div>

        <form id="whatif-form" class="chart-card card-body" novalidate
              onsubmit="handlePredictSubmit(event)"
              aria-label="What-if prediction form">

          <fieldset style="border:none;padding:0;">
            <legend class="card-title" style="margin-bottom:var(--space-md);">Required Inputs</legend>
            <div class="form-grid">
              ${requiredFields.map(fieldHtml).join('')}
            </div>
          </fieldset>

          <hr class="section-divider">

          <fieldset style="border:none;padding:0;">
            <legend class="card-title" style="margin-bottom:var(--space-md);">Optional Inputs</legend>
            <div class="form-grid">
              ${optionalFields.map(fieldHtml).join('')}
            </div>
            <div class="form-group form-grid-full">
              <label for="feature_mode" class="form-label">Feature Mode</label>
              <select id="feature_mode" class="form-control" aria-label="Feature mode A or B">
                <option value="A" selected>Mode A — Strict forecast (no future weather)</option>
                <option value="B">Mode B — With next-day weather forecast</option>
              </select>
              <span class="form-hint">Mode A is the honest evaluation default.</span>
            </div>
          </fieldset>

          <div style="margin-top:var(--space-lg);display:flex;gap:var(--space-md);align-items:center;flex-wrap:wrap;">
            <button type="submit" class="btn btn-primary" id="predict-btn">
              <span id="predict-btn-text">Run Prediction</span>
              <span id="predict-spinner" class="spinner" style="display:none;"></span>
            </button>
            <button type="button" class="btn btn-ghost" onclick="resetForm()">Reset</button>
          </div>

          <div id="form-api-error" style="display:none;margin-top:var(--space-md);"
               role="alert" aria-live="polite"></div>
        </form>
      </div>

      <!-- Result Panel -->
      <div>
        <div class="section-header">
          <h2 class="section-title">Prediction Result</h2>
          <p class="section-subtitle">AQI category and health advisory</p>
        </div>
        <div id="result-panel" class="chart-card card-body" aria-live="polite" aria-label="Prediction result">
          <div class="empty-state" style="min-height:300px;">
            <div class="empty-icon" aria-hidden="true">🔮</div>
            <div class="empty-title">Fill in the form and click Run</div>
            <p class="empty-msg">Predicted PM2.5 and AQI will appear here.</p>
          </div>
        </div>

        <!-- Methodology note -->
        <div class="about-section" style="margin-top:var(--space-xl);">
          <h2>How it works</h2>
          <p>
            This predictor calls <code>POST /api/predict</code> with your entered values.
            The backend loads the best trained model, builds the same feature vector used during
            training, and returns the next-day PM2.5 estimate plus the derived AQI category.
          </p>
          <h3>Mode A vs Mode B</h3>
          <p>
            <strong>Mode A</strong> uses only information available at the end of today (no future
            weather) — this is the operationally honest setting.
            <strong>Mode B</strong> includes next-day weather forecast inputs, which improves accuracy
            but requires a forecast; treat Mode B scores as optimistic upper bounds.
          </p>
        </div>
      </div>
    </div>`;

  // Pre-fill month with current month
  const monthInput = document.getElementById('month');
  if (monthInput) monthInput.value = today.getMonth() + 1;

  // Add real-time validation
  FIELDS.forEach(f => {
    const el = document.getElementById(f.id);
    if (el) {
      el.addEventListener('blur', () => validateField(f));
      el.addEventListener('input', () => {
        el.classList.remove('invalid');
        const errEl = document.getElementById(f.id + '-error');
        if (errEl) errEl.style.display = 'none';
      });
    }
  });
}

function validateField(fieldDef) {
  const el = document.getElementById(fieldDef.id);
  if (!el) return true;

  const val = el.value.trim();
  const errEl = document.getElementById(fieldDef.id + '-error');

  if (fieldDef.required && val === '') {
    el.classList.add('invalid');
    if (errEl) { errEl.textContent = `${fieldDef.label} is required.`; errEl.style.display = 'block'; }
    return false;
  }

  if (val !== '') {
    const num = parseFloat(val);
    if (isNaN(num)) {
      el.classList.add('invalid');
      if (errEl) { errEl.textContent = 'Must be a number.'; errEl.style.display = 'block'; }
      return false;
    }
    if (fieldDef.min != null && num < fieldDef.min) {
      el.classList.add('invalid');
      if (errEl) { errEl.textContent = `Minimum value is ${fieldDef.min}.`; errEl.style.display = 'block'; }
      return false;
    }
    if (fieldDef.max != null && num > fieldDef.max) {
      el.classList.add('invalid');
      if (errEl) { errEl.textContent = `Maximum value is ${fieldDef.max}.`; errEl.style.display = 'block'; }
      return false;
    }
  }

  el.classList.remove('invalid');
  if (errEl) errEl.style.display = 'none';
  return true;
}

function validateAllFields() {
  let allValid = true;
  for (const f of FIELDS) {
    if (!validateField(f)) allValid = false;
  }
  return allValid;
}

function collectFormData() {
  const body = {};
  for (const f of FIELDS) {
    const el = document.getElementById(f.id);
    if (!el || el.value.trim() === '') continue;
    body[f.id] = parseFloat(el.value);
  }
  const modeEl = document.getElementById('feature_mode');
  if (modeEl) body.feature_mode = modeEl.value;
  return body;
}

async function handlePredictSubmit(evt) {
  evt.preventDefault();

  if (!validateAllFields()) {
    showToast('Please fix validation errors before submitting.', 'error');
    return;
  }

  const btn     = document.getElementById('predict-btn');
  const btnText = document.getElementById('predict-btn-text');
  const spinner = document.getElementById('predict-spinner');
  const errDiv  = document.getElementById('form-api-error');

  // Show loading
  if (btn) btn.disabled = true;
  if (btnText) btnText.textContent = 'Running…';
  if (spinner) spinner.style.display = 'inline-block';
  if (errDiv) errDiv.style.display = 'none';

  const body = collectFormData();
  const { data, error, status } = await postPredict(body);

  // Reset button
  if (btn) btn.disabled = false;
  if (btnText) btnText.textContent = 'Run Prediction';
  if (spinner) spinner.style.display = 'none';

  if (error) {
    if (status === 422) {
      // Validation error from backend
      let msg = 'Validation error: ';
      if (typeof error === 'string') {
        msg += error;
      } else if (Array.isArray(error)) {
        msg += error.map(e => `${e.loc?.join('.')}: ${e.msg}`).join('; ');
      } else {
        msg += JSON.stringify(error);
      }
      if (errDiv) {
        errDiv.innerHTML = `<div class="empty-state error-state" style="padding:var(--space-md);min-height:auto;">
          <span>⚠️ ${escHtml(msg)}</span></div>`;
        errDiv.style.display = 'block';
      }
    } else {
      if (errDiv) {
        errDiv.innerHTML = `<div class="empty-state error-state" style="padding:var(--space-md);min-height:auto;">
          <span>🔌 ${escHtml(error)}</span></div>`;
        errDiv.style.display = 'block';
      }
      showToast(error, 'error');
    }
    return;
  }

  renderResult(data);
}

function renderResult(d) {
  const panel = document.getElementById('result-panel');
  if (!panel) return;

  const cfg   = getCategoryConfig(d.category) || getCategoryByAqi(d.aqi_pred);
  const color = d.color || (cfg ? cfg.color : '#94a3b8');
  const health = d.health || (cfg ? cfg.health : '');

  panel.innerHTML = `
    <div style="text-align:center;padding:var(--space-lg);">
      ${buildAqiBadge(d.aqi_pred, d.category, health)}

      <div style="margin-top:var(--space-xl);display:grid;grid-template-columns:1fr 1fr;gap:var(--space-md);text-align:left;">
        <article class="kpi-card" aria-label="Predicted PM2.5">
          <div class="kpi-label">Predicted PM2.5</div>
          <div class="kpi-value" style="font-size:28px;">
            ${fmt(d.pm25_pred, 1)}<span class="kpi-unit"> µg/m³</span>
          </div>
        </article>
        <article class="kpi-card" aria-label="Predicted AQI">
          <div class="kpi-label">Predicted AQI</div>
          <div class="kpi-value" style="font-size:28px;color:${color};">
            ${fmt(d.aqi_pred, 0)}
          </div>
        </article>
      </div>

      <p style="margin-top:var(--space-lg);font-size:12px;color:var(--text-muted);">
        Note: predictions are based on the trained model's learned patterns.
        Extreme or unseen input combinations may yield less reliable results.
      </p>
    </div>`;

  showToast('Prediction complete!', 'success');
}

function resetForm() {
  const form = document.getElementById('whatif-form');
  if (form) form.reset();

  FIELDS.forEach(f => {
    const el = document.getElementById(f.id);
    if (el) el.classList.remove('invalid');
    const errEl = document.getElementById(f.id + '-error');
    if (errEl) errEl.style.display = 'none';
  });

  const errDiv = document.getElementById('form-api-error');
  if (errDiv) errDiv.style.display = 'none';

  const panel = document.getElementById('result-panel');
  if (panel) {
    panel.innerHTML = `
      <div class="empty-state" style="min-height:300px;">
        <div class="empty-icon" aria-hidden="true">🔮</div>
        <div class="empty-title">Fill in the form and click Run</div>
        <p class="empty-msg">Predicted PM2.5 and AQI will appear here.</p>
      </div>`;
  }

  // Re-fill month
  const monthInput = document.getElementById('month');
  if (monthInput) monthInput.value = new Date().getMonth() + 1;
}
