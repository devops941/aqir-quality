/**
 * upload.js — Upload & Retrain page logic
 * - Three file-drop zones (air_quality, weather, events)
 * - Forecast mode selector (A / B / both)
 * - Template download links (header-only CSV, generated client-side)
 * - POST /api/admin/upload -> validation errors or 202 accepted
 * - Poll GET /api/admin/status every 2s while running
 * - Show success (best_run_id) or failure (log tail) when done/failed
 */

'use strict';

/* ---- Schema column definitions (matches PROJECT_SPEC_AQI.md Section 4) ---- */
const SCHEMAS = {
  air_quality: {
    label: 'Air Quality CSV',
    required: true,
    columns: ['station_id', 'station_name', 'city', 'lat', 'lon', 'timestamp',
              'pm25', 'pm10', 'no2', 'so2', 'co', 'o3'],
    hint: 'Daily or hourly pollution readings per station.',
  },
  weather: {
    label: 'Weather CSV',
    required: true,
    columns: ['station_id', 'timestamp', 'temp_c', 'humidity_pct', 'wind_speed_ms',
              'wind_dir_deg', 'pressure_hpa', 'precip_mm', 'cloud_pct', 'blh_m'],
    hint: 'Matching weather observations per station & timestamp.',
  },
  events: {
    label: 'Events CSV',
    required: false,
    columns: ['date', 'event_name', 'flag'],
    hint: 'Optional. Festival / crop-burning flags. If omitted an empty file is created automatically.',
  },
};

/* ---- Generate a header-only CSV data: URL for template downloads ---- */
function makeTemplateUrl(columns) {
  const csv = columns.join(',') + '\n';
  return 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
}

/* ---- Main page init ---- */
(async function () {
  const content = await initPage({
    title: 'Upload & Retrain',
    activeHref: 'upload.html',
    showStation: false,
  });

  if (content) renderUploadPage(content);
})();

/* ---- State ---- */
let _pollTimer = null;
let _elapsedTimer = null;
let _elapsedSec = 0;

function stopPolling() {
  if (_pollTimer)   { clearInterval(_pollTimer);   _pollTimer = null; }
  if (_elapsedTimer){ clearInterval(_elapsedTimer); _elapsedTimer = null; }
}

/* ---- Render ---- */
function renderUploadPage(content) {

  /* Build drop-zone HTML for each schema */
  function dropZoneHtml(key) {
    const s = SCHEMAS[key];
    const req = s.required ? '<span style="color:#ef4444;" aria-label="required"> *</span>' : ' <span style="color:var(--text-muted);font-size:12px;">(optional)</span>';
    const cols = s.columns.map(c => `<code>${c}</code>`).join(', ');
    const tmplUrl = makeTemplateUrl(s.columns);
    return `
      <div class="card card-body upload-zone" id="zone-${key}">
        <div class="upload-zone-header">
          <span class="upload-zone-title">${escHtml(s.label)}${req}</span>
          <a class="upload-template-link"
             href="${tmplUrl}"
             download="${key}_template.csv"
             aria-label="Download ${s.label} template CSV">
            Download template
          </a>
        </div>
        <label class="upload-drop-area" for="file-${key}" id="droparea-${key}">
          <input type="file" id="file-${key}" name="${key}" accept=".csv,text/csv"
                 aria-label="${s.label} file input" style="display:none;" />
          <span class="upload-drop-icon" aria-hidden="true">📂</span>
          <span class="upload-drop-label" id="label-${key}">Click or drag a CSV file here</span>
        </label>
        <p class="upload-schema-hint">
          <strong>Expected columns:</strong> ${cols}
        </p>
        <p class="upload-schema-hint">${escHtml(s.hint)}</p>
      </div>`;
  }

  content.innerHTML = `
    <div class="upload-page">
      <section class="card card-body upload-intro" aria-label="Introduction">
        <h2 class="section-title" style="margin-bottom:8px;">Upload Your Data &amp; Retrain the Model</h2>
        <p style="color:var(--text-secondary);margin:0;">
          Upload air quality and weather CSVs matching the schemas below. The backend will validate them,
          copy them to <code>data/raw/</code>, and launch the ML pipeline. Live status is shown below.
        </p>
      </section>

      <form id="upload-form" novalidate>
        <!-- Drop zones -->
        <div class="upload-zones-grid">
          ${dropZoneHtml('air_quality')}
          ${dropZoneHtml('weather')}
          ${dropZoneHtml('events')}
        </div>

        <!-- Mode selector -->
        <div class="card card-body upload-mode-card">
          <label class="form-label" for="mode-select">
            Forecast mode
          </label>
          <select id="mode-select" name="mode" class="form-control" style="max-width:220px;" aria-describedby="mode-hint">
            <option value="A" selected>A — Strict forecast (fastest)</option>
            <option value="B">B — With weather forecast</option>
            <option value="both">Both A &amp; B (slowest)</option>
          </select>
          <p id="mode-hint" class="upload-schema-hint" style="margin-top:8px;">
            Mode A uses only information available at forecast time (recommended). Mode B adds day-ahead weather data (optimistic). "Both" trains two sets of models.
          </p>
        </div>

        <!-- Submit -->
        <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
          <button type="submit" id="submit-btn" class="btn btn-primary upload-submit-btn">
            Validate &amp; Train
          </button>
          <span id="submit-note" style="color:var(--text-muted);font-size:13px;"></span>
        </div>
      </form>

      <!-- Error panel -->
      <div id="error-panel" class="upload-error-panel" style="display:none;" role="alert" aria-live="assertive"></div>

      <!-- Status panel -->
      <div id="status-panel" class="upload-status-panel" style="display:none;" aria-live="polite"></div>
    </div>`;

  /* ---- Wire up drop areas ---- */
  for (const key of Object.keys(SCHEMAS)) {
    const input     = document.getElementById(`file-${key}`);
    const dropArea  = document.getElementById(`droparea-${key}`);
    const labelEl   = document.getElementById(`label-${key}`);

    // Click to open file picker
    dropArea.addEventListener('click', () => input.click());

    // Keyboard activation
    dropArea.setAttribute('tabindex', '0');
    dropArea.setAttribute('role', 'button');
    dropArea.setAttribute('aria-label', `Choose ${SCHEMAS[key].label}`);
    dropArea.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
    });

    // File chosen via input
    input.addEventListener('change', () => {
      if (input.files && input.files[0]) {
        labelEl.textContent = input.files[0].name;
        dropArea.classList.add('has-file');
      }
    });

    // Drag-and-drop
    dropArea.addEventListener('dragover', e => { e.preventDefault(); dropArea.classList.add('drag-over'); });
    dropArea.addEventListener('dragleave', () => dropArea.classList.remove('drag-over'));
    dropArea.addEventListener('drop', e => {
      e.preventDefault();
      dropArea.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f) {
        const dt = new DataTransfer();
        dt.items.add(f);
        input.files = dt.files;
        labelEl.textContent = f.name;
        dropArea.classList.add('has-file');
      }
    });
  }

  /* ---- Form submission ---- */
  document.getElementById('upload-form').addEventListener('submit', async e => {
    e.preventDefault();
    await handleSubmit();
  });
}

/* ---- Submit handler ---- */
async function handleSubmit() {
  const submitBtn  = document.getElementById('submit-btn');
  const submitNote = document.getElementById('submit-note');
  const errorPanel = document.getElementById('error-panel');
  const statusPanel= document.getElementById('status-panel');

  hidePanel(errorPanel);
  hidePanel(statusPanel);

  const aqFile     = document.getElementById('file-air_quality').files[0];
  const wxFile     = document.getElementById('file-weather').files[0];
  const evFile     = document.getElementById('file-events').files[0];
  const mode       = document.getElementById('mode-select').value;

  /* Client-side: require the two mandatory files */
  if (!aqFile || !wxFile) {
    const missing = [];
    if (!aqFile) missing.push('Air Quality CSV');
    if (!wxFile) missing.push('Weather CSV');
    showErrorPanel(errorPanel, [{
      file: 'form',
      error: `Required files missing: ${missing.join(', ')}.`,
    }]);
    return;
  }

  /* Disable button while uploading */
  submitBtn.disabled = true;
  submitBtn.textContent = 'Uploading…';
  submitNote.textContent = '';

  const fd = new FormData();
  fd.append('air_quality', aqFile);
  fd.append('weather', wxFile);
  if (evFile) fd.append('events', evFile);
  fd.append('mode', mode);

  let res, body;
  try {
    res  = await fetch('/api/admin/upload', { method: 'POST', body: fd });
    body = await res.json().catch(() => null);
  } catch (err) {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Validate & Train';
    showErrorPanel(errorPanel, [{ file: 'network', error: 'Cannot reach the API. Is the backend running?' }]);
    return;
  }

  if (res.status === 409) {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Validate & Train';
    showErrorPanel(errorPanel, [{ file: 'server', error: 'A training job is already running. Wait for it to finish.' }]);
    return;
  }

  if (res.status === 422) {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Validate & Train';
    const detail = body && body.detail ? body.detail : [{ file: 'unknown', error: JSON.stringify(body) }];
    showErrorPanel(errorPanel, Array.isArray(detail) ? detail : [{ file: 'validation', error: String(detail) }]);
    return;
  }

  if (!res.ok) {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Validate & Train';
    showErrorPanel(errorPanel, [{ file: 'server', error: `Unexpected error ${res.status}.` }]);
    return;
  }

  /* 202 Accepted — start polling */
  submitBtn.textContent = 'Training…';
  _elapsedSec = 0;
  showRunningPanel(statusPanel, body, 0);

  _elapsedTimer = setInterval(() => {
    _elapsedSec += 1;
    updateElapsed(statusPanel, _elapsedSec);
  }, 1000);

  _pollTimer = setInterval(async () => {
    const { data: status, error } = await apiFetch('/api/admin/status');
    if (error || !status) return; // transient error — keep polling

    if (status.state === 'running') {
      updateElapsed(statusPanel, _elapsedSec);
      return;
    }

    /* Done or failed */
    stopPolling();
    submitBtn.disabled = false;
    submitBtn.textContent = 'Validate & Train';

    if (status.state === 'done') {
      showDonePanel(statusPanel, status);
    } else {
      showFailedPanel(statusPanel, status);
    }
  }, 2000);
}

/* ---- Panel helpers ---- */

function hidePanel(el) {
  el.style.display = 'none';
  el.innerHTML = '';
}

function showErrorPanel(panel, errors) {
  panel.style.display = 'block';
  const rows = errors.map(e => `
    <div class="upload-error-row">
      <span class="upload-error-file">${escHtml(e.file || 'unknown')}</span>
      <span class="upload-error-msg">${escHtml(e.error || e.msg || JSON.stringify(e))}</span>
    </div>`).join('');
  panel.innerHTML = `
    <div class="upload-error-header">Validation failed — no training was started.</div>
    ${rows}`;
}

function showRunningPanel(panel, body, elapsed) {
  panel.style.display = 'block';
  panel.innerHTML = `
    <div class="upload-status-running">
      <span class="upload-spinner" aria-hidden="true"></span>
      <div>
        <div class="upload-status-title">Training in progress (mode ${escHtml(body.mode || '?')})</div>
        <div class="upload-status-sub" id="elapsed-label">Elapsed: ${elapsed}s</div>
        <div class="upload-status-hint">The pipeline is running in the background. This page polls every 2 seconds.</div>
      </div>
    </div>`;
}

function updateElapsed(panel, sec) {
  const el = panel.querySelector('#elapsed-label');
  if (el) el.textContent = `Elapsed: ${sec}s`;
}

function showDonePanel(panel, status) {
  const runId = status.best_run_id != null ? `Best model: run #${status.best_run_id}` : 'Training complete.';
  panel.innerHTML = `
    <div class="upload-status-done">
      <span aria-hidden="true" style="font-size:28px;line-height:1;">&#10003;</span>
      <div>
        <div class="upload-status-title">Training complete!</div>
        <div class="upload-status-sub">${escHtml(runId)}</div>
        <div class="upload-status-hint" style="margin-top:8px;">
          Refresh the <a href="index.html" class="upload-link">Forecast</a> or
          <a href="performance.html" class="upload-link">Performance</a> pages to see updated results.
        </div>
      </div>
    </div>`;
}

function showFailedPanel(panel, status) {
  const tail = status.message ? `<pre class="upload-log-tail">${escHtml(status.message)}</pre>` : '';
  panel.innerHTML = `
    <div class="upload-status-failed">
      <span aria-hidden="true" style="font-size:28px;line-height:1;">&#10005;</span>
      <div style="flex:1;">
        <div class="upload-status-title">Training failed (exit code ${status.returncode ?? '?'})</div>
        <div class="upload-status-hint">Last log lines:</div>
        ${tail}
      </div>
    </div>`;
}
