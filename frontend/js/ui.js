/**
 * ui.js — Shared UI utilities
 * - Build sidebar + topbar (injected into every page)
 * - Theme toggle with localStorage persistence
 * - Populate station <select> from /api/stations
 * - Active-nav highlight
 * - Toast / empty / skeleton helpers
 * - Plotly dark template
 */

'use strict';

/* ============================================================
   THEME MANAGEMENT
   ============================================================ */
const THEME_KEY = 'aqf-theme';

function getCurrentTheme() {
  return localStorage.getItem(THEME_KEY) || 'dark';
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(THEME_KEY, theme);
  // Update toggle button text/icons
  document.querySelectorAll('.theme-toggle').forEach(btn => {
    const icon = btn.querySelector('.icon');
    const label = btn.querySelector('.label');
    if (icon)  icon.textContent = theme === 'dark' ? '☀️' : '🌙';
    if (label) label.textContent = theme === 'dark' ? 'Light' : 'Dark';
  });
}

function toggleTheme() {
  const next = getCurrentTheme() === 'dark' ? 'light' : 'dark';
  applyTheme(next);
}

/* ============================================================
   SIDEBAR MARKUP
   ============================================================ */
const NAV_LINKS = [
  { href: 'index.html',       icon: '🏠', label: 'Forecast',    id: 'nav-home' },
  { href: 'history.html',     icon: '📈', label: 'History',     id: 'nav-history' },
  { href: 'performance.html', icon: '📊', label: 'Performance', id: 'nav-performance' },
  { href: 'explain.html',     icon: '🧠', label: 'Explain',     id: 'nav-explain' },
  { href: 'whatif.html',      icon: '🔮', label: 'What-if',     id: 'nav-whatif' },
  { href: 'upload.html',      icon: '📤', label: 'Upload',      id: 'nav-upload' },
  { href: 'about.html',       icon: 'ℹ️',  label: 'About',       id: 'nav-about' },
];

function buildSidebar(activeHref) {
  const linksHtml = NAV_LINKS.map(l => {
    const isActive = l.href === activeHref;
    return `
      <a href="${l.href}" class="nav-link${isActive ? ' active' : ''}" id="${l.id}"
         aria-current="${isActive ? 'page' : 'false'}">
        <span class="nav-icon" aria-hidden="true">${l.icon}</span>
        <span>${l.label}</span>
      </a>`;
  }).join('');

  return `
    <aside class="sidebar" id="sidebar" role="navigation" aria-label="Main navigation">
      <a href="index.html" class="sidebar-logo" aria-label="Air Quality Forecasting home">
        <div class="sidebar-logo-icon" aria-hidden="true">🌬️</div>
        <div class="sidebar-logo-text">
          AirQuality
          <span>Forecasting Dashboard</span>
        </div>
      </a>
      <nav class="sidebar-nav" aria-label="Pages">
        <div class="nav-section-label">Navigation</div>
        ${linksHtml}
      </nav>
      <div class="sidebar-footer">
        <span>Phase 9 · FastAPI + Plotly.js</span>
      </div>
    </aside>
    <div class="sidebar-overlay" id="sidebar-overlay" aria-hidden="true"></div>`;
}

/* ============================================================
   TOPBAR MARKUP
   ============================================================ */
function buildTopbar(pageTitle, showStationSelect = true) {
  const stationControl = showStationSelect ? `
    <div class="station-select-wrap" role="group" aria-label="Station selector">
      <select id="station-select" class="form-control" aria-label="Select monitoring station"
              style="min-width:200px; font-size:13px; padding:8px 36px 8px 30px;">
        <option value="">Loading stations…</option>
      </select>
    </div>` : '';

  return `
    <header class="topbar" role="banner">
      <h1 class="topbar-title">${pageTitle}</h1>
      <div class="topbar-controls">
        ${stationControl}
        <button class="theme-toggle" id="theme-toggle-top" onclick="toggleTheme()"
                aria-label="Toggle colour theme" title="Toggle dark/light mode">
          <span class="icon" aria-hidden="true">☀️</span>
          <span class="label">Light</span>
        </button>
      </div>
    </header>`;
}

/* ============================================================
   MOBILE TOPBAR
   ============================================================ */
function buildMobileTopbar(pageTitle) {
  return `
    <div class="mobile-topbar" role="banner" aria-label="Mobile header">
      <button class="hamburger" id="hamburger-btn" aria-label="Open navigation menu"
              aria-expanded="false" aria-controls="sidebar">
        ☰
      </button>
      <span style="font-family:var(--font-heading);font-weight:600;font-size:15px;flex:1;">${pageTitle}</span>
      <button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme"
              style="padding:5px 10px;">
        <span class="icon" aria-hidden="true">☀️</span>
      </button>
    </div>`;
}

/* ============================================================
   PAGE BOOTSTRAP — call once per page
   ============================================================ */
/**
 * initPage({ title, activeHref, showStationSelect, onStationChange })
 * Injects sidebar, topbar, mobile bar into the DOM.
 * Sets up theme, hamburger, and station select.
 *
 * @param {object} opts
 * @param {string} opts.title             - Page title shown in topbar
 * @param {string} opts.activeHref        - e.g. 'index.html'
 * @param {boolean} [opts.showStation]    - show station dropdown (default true)
 * @param {function} [opts.onStation]     - callback(stationId) when station changes
 */
async function initPage(opts = {}) {
  const { title = 'Dashboard', activeHref = '', showStation = true, onStation } = opts;

  // Apply saved theme before first paint
  applyTheme(getCurrentTheme());

  // Inject structural markup
  const layout = document.getElementById('app-layout');
  if (!layout) return;

  layout.innerHTML =
    buildMobileTopbar(title) +
    buildSidebar(activeHref) +
    `<div class="main-content">
       ${buildTopbar(title, showStation)}
       <main class="page-content" id="page-content" role="main"></main>
     </div>`;

  // Hamburger / sidebar toggle
  const hamburger = document.getElementById('hamburger-btn');
  const sidebar    = document.getElementById('sidebar');
  const overlay    = document.getElementById('sidebar-overlay');

  if (hamburger && sidebar && overlay) {
    hamburger.addEventListener('click', () => {
      const isOpen = sidebar.classList.toggle('open');
      overlay.classList.toggle('open', isOpen);
      hamburger.setAttribute('aria-expanded', isOpen);
    });
    overlay.addEventListener('click', () => {
      sidebar.classList.remove('open');
      overlay.classList.remove('open');
      hamburger.setAttribute('aria-expanded', 'false');
    });
  }

  // Toast container
  if (!document.getElementById('toast-container')) {
    const tc = document.createElement('div');
    tc.id = 'toast-container';
    tc.setAttribute('role', 'region');
    tc.setAttribute('aria-live', 'polite');
    tc.setAttribute('aria-label', 'Notifications');
    document.body.appendChild(tc);
  }

  // Populate station select
  if (showStation) {
    await populateStationSelect(onStation);
  }

  return document.getElementById('page-content');
}

/* ============================================================
   STATION SELECT POPULATION
   ============================================================ */
async function populateStationSelect(onChange) {
  const sel = document.getElementById('station-select');
  if (!sel) return;

  const { data, error } = await fetchStations();

  if (error || !Array.isArray(data) || data.length === 0) {
    sel.innerHTML = '<option value="">No stations available</option>';
    if (error) showToast('Could not load stations: ' + error, 'error');
    return;
  }

  sel.innerHTML = data.map(s =>
    `<option value="${s.station_id}">${s.station_name} (${s.city})</option>`
  ).join('');

  if (onChange) {
    sel.addEventListener('change', () => onChange(sel.value));
    // Fire immediately with first station
    onChange(sel.value);
  }
}

/** Get currently selected station_id */
function getSelectedStation() {
  const sel = document.getElementById('station-select');
  return sel ? sel.value : '';
}

/* ============================================================
   TOAST HELPER
   ============================================================ */
/**
 * @param {string} message
 * @param {'info'|'success'|'error'} [type]
 * @param {number} [duration] ms
 */
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.setAttribute('role', 'alert');
  const icons = { info: 'ℹ️', success: '✅', error: '⚠️' };
  toast.innerHTML = `<span aria-hidden="true">${icons[type] || 'ℹ️'}</span> ${escHtml(message)}`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(8px)';
    toast.style.transition = 'opacity 0.3s, transform 0.3s';
    setTimeout(() => toast.remove(), 310);
  }, duration);
}

/* ============================================================
   SKELETON / EMPTY / ERROR HELPERS
   ============================================================ */
function skeletonCards(count = 4) {
  return `<div class="kpi-grid">${'<div class="kpi-card skeleton skeleton-card"></div>'.repeat(count)}</div>`;
}

function skeletonChart() {
  return `<div class="skeleton skeleton-chart"></div>`;
}

/**
 * emptyState({ icon, title, msg, isError })
 */
function emptyState({ icon = '📭', title = 'No data yet', msg = 'Run the pipeline to populate data.', isError = false } = {}) {
  return `
    <div class="empty-state card card-body${isError ? ' error-state' : ''}">
      <div class="empty-icon" aria-hidden="true">${icon}</div>
      <div class="empty-title">${escHtml(title)}</div>
      <p class="empty-msg">${escHtml(msg)}</p>
    </div>`;
}

function apiErrorState(errorMsg) {
  return emptyState({
    icon: '🔌',
    title: 'API Unavailable',
    msg: errorMsg || 'Cannot reach the backend. Run: uvicorn backend.main:app --reload',
    isError: true,
  });
}

/* ============================================================
   PLOTLY DARK TEMPLATE
   ============================================================ */
const PLOTLY_TEMPLATE = {
  layout: {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0)',
    font: { family: 'Inter, system-ui, sans-serif', color: '#94a3b8', size: 12 },
    colorway: ['#22d3ee', '#a855f7', '#f59e0b', '#10b981', '#ef4444', '#3b82f6'],
    xaxis: {
      gridcolor:    'rgba(255,255,255,0.05)',
      linecolor:    'rgba(255,255,255,0.08)',
      tickcolor:    'rgba(255,255,255,0.08)',
      zerolinecolor:'rgba(255,255,255,0.08)',
    },
    yaxis: {
      gridcolor:    'rgba(255,255,255,0.05)',
      linecolor:    'rgba(255,255,255,0.08)',
      tickcolor:    'rgba(255,255,255,0.08)',
      zerolinecolor:'rgba(255,255,255,0.08)',
    },
    legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: '#94a3b8' } },
    hoverlabel: {
      bgcolor:     '#1a2538',
      bordercolor: 'rgba(255,255,255,0.1)',
      font: { color: '#f1f5f9', size: 12 },
    },
    margin: { t: 20, r: 20, b: 50, l: 60 },
  },
};

/** Plotly config object — responsive + no logo */
const PLOTLY_CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
};

/** Merge PLOTLY_TEMPLATE.layout into a user layout object */
function mergeLayout(userLayout = {}) {
  return Object.assign({}, PLOTLY_TEMPLATE.layout, userLayout, {
    xaxis: Object.assign({}, PLOTLY_TEMPLATE.layout.xaxis, userLayout.xaxis),
    yaxis: Object.assign({}, PLOTLY_TEMPLATE.layout.yaxis, userLayout.yaxis),
    legend: Object.assign({}, PLOTLY_TEMPLATE.layout.legend, userLayout.legend),
  });
}

/* ============================================================
   PLOTLY RESIZE HELPER
   ============================================================ */
function makePlotsResponsive() {
  let _resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(() => {
      document.querySelectorAll('.chart-plot').forEach(el => {
        if (el._fullLayout) {
          try { Plotly.relayout(el, { autosize: true }); } catch (_) {}
        }
      });
    }, 150);
  });
}

/* ============================================================
   UTILITY HELPERS
   ============================================================ */
/** Escape HTML to prevent XSS */
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Format number to fixed decimals, returning '—' for null/NaN */
function fmt(val, decimals = 1) {
  if (val == null || isNaN(val)) return '—';
  return Number(val).toFixed(decimals);
}

/** Format date string YYYY-MM-DD → "Nov 18, 2026" */
function fmtDate(str) {
  if (!str) return '—';
  try {
    return new Date(str + 'T00:00:00').toLocaleDateString('en-IN', {
      year: 'numeric', month: 'short', day: 'numeric'
    });
  } catch (_) { return str; }
}

/** Format ISO datetime string */
function fmtDatetime(str) {
  if (!str) return '—';
  try {
    return new Date(str).toLocaleString('en-IN', {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit'
    });
  } catch (_) { return str; }
}

/** Build an AQI badge element string */
function buildAqiBadge(aqiValue, categoryName, healthMsg) {
  const cfg = getCategoryConfig(categoryName) || getCategoryByAqi(aqiValue);
  const cssClass = categoryToCssClass(cfg ? cfg.name : categoryName);
  const bgClass  = categoryToBgClass(cfg ? cfg.name : categoryName);
  const color    = cfg ? cfg.color : '#94a3b8';
  const health   = healthMsg || (cfg ? cfg.health : '');

  return `
    <div class="aqi-badge-wrap">
      <div class="aqi-badge ${cssClass}"
           style="border-color:${color};color:${color};background:${color}18;"
           aria-label="AQI ${aqiValue} — ${cfg ? cfg.name : categoryName}">
        <span class="aqi-badge-number">${aqiValue != null ? Math.round(aqiValue) : '—'}</span>
        <span class="aqi-badge-label">AQI</span>
      </div>
      <span class="aqi-category-pill ${bgClass}" style="background:${color};">${escHtml(cfg ? cfg.name : (categoryName || '—'))}</span>
      <p class="aqi-health-msg" aria-live="polite">${escHtml(health)}</p>
    </div>`;
}

/* ============================================================
   PLOTLY LIGHT-THEME AWARE RERENDER
   ============================================================ */
function getPlotlyBgStyle() {
  return document.documentElement.getAttribute('data-theme') === 'light'
    ? { paper_bgcolor: 'rgba(240,244,248,0)', plot_bgcolor: 'rgba(240,244,248,0)', 'font.color': '#475569' }
    : {};
}
