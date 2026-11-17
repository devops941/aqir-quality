/**
 * api.js — Fetch helpers, AQI color map, category health text
 * Base URL: same origin (FastAPI serves frontend at / and API at /api)
 */

'use strict';

/* ---- Base URL (same-origin, no trailing slash) ---- */
const API_BASE = '';

/* ---- AQI Category Config (matches config.yaml aqi.categories) ---- */
const AQI_CATEGORIES = [
  { name: 'Good',         low: 0,   high: 50,  color: '#2ecc71', health: 'Minimal impact.' },
  { name: 'Satisfactory', low: 51,  high: 100, color: '#9acd32', health: 'Minor breathing discomfort to sensitive people.' },
  { name: 'Moderate',     low: 101, high: 200, color: '#f1c40f', health: 'Breathing discomfort to people with lung/heart disease.' },
  { name: 'Poor',         low: 201, high: 300, color: '#e67e22', health: 'Breathing discomfort to most on prolonged exposure.' },
  { name: 'Very Poor',    low: 301, high: 400, color: '#e74c3c', health: 'Respiratory illness on prolonged exposure.' },
  { name: 'Severe',       low: 401, high: 500, color: '#7e0023', health: 'Affects healthy people; serious impact on those with illness.' },
];

/** Map category name string to config object. */
function getCategoryConfig(categoryName) {
  if (!categoryName) return null;
  const norm = categoryName.trim().toLowerCase();
  return AQI_CATEGORIES.find(c => c.name.toLowerCase() === norm) || null;
}

/** Given AQI value (number), return matching category config. */
function getCategoryByAqi(aqiValue) {
  if (aqiValue == null || isNaN(aqiValue)) return null;
  return AQI_CATEGORIES.find(c => aqiValue >= c.low && aqiValue <= c.high) || AQI_CATEGORIES[AQI_CATEGORIES.length - 1];
}

/** CSS class name from category name e.g. "Very Poor" → "aqi-very-poor" */
function categoryToCssClass(categoryName) {
  if (!categoryName) return '';
  return 'aqi-' + categoryName.trim().toLowerCase().replace(/\s+/g, '-');
}

/** CSS bg class e.g. "Very Poor" → "bg-very-poor" */
function categoryToBgClass(categoryName) {
  if (!categoryName) return '';
  return 'bg-' + categoryName.trim().toLowerCase().replace(/\s+/g, '-');
}

/* ---- Generic Fetch with Error Handling ---- */

/**
 * apiFetch(path, options)
 * Returns { data, error } — never throws.
 * @param {string} path - e.g. "/api/stations"
 * @param {object} [options] - fetch options
 */
async function apiFetch(path, options = {}) {
  try {
    const url = API_BASE + path;
    const res = await fetch(url, {
      headers: { 'Accept': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const err = await res.json();
        detail = err.detail || JSON.stringify(err);
      } catch (_) { /* ignore */ }
      return { data: null, error: detail, status: res.status };
    }
    const data = await res.json();
    return { data, error: null, status: res.status };
  } catch (err) {
    const msg = err instanceof TypeError ? 'Cannot reach the API — make sure the backend is running.' : err.message;
    return { data: null, error: msg, status: 0 };
  }
}

/**
 * apiPost(path, body)
 * JSON POST wrapper. Returns { data, error }.
 */
async function apiPost(path, body) {
  return apiFetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/* ---- Named API Calls ---- */

/** GET /api/stations */
async function fetchStations() {
  return apiFetch('/api/stations');
}

/** GET /api/forecast?station_id= */
async function fetchForecast(stationId) {
  return apiFetch(`/api/forecast?station_id=${encodeURIComponent(stationId)}`);
}

/**
 * GET /api/history
 * @param {string} stationId
 * @param {string} [start] YYYY-MM-DD
 * @param {string} [end]   YYYY-MM-DD
 * @param {string} [variables] comma-separated e.g. "pm25,aqi"
 */
async function fetchHistory(stationId, start, end, variables) {
  const params = new URLSearchParams({ station_id: stationId });
  if (start)     params.set('start', start);
  if (end)       params.set('end', end);
  if (variables) params.set('variables', variables);
  return apiFetch(`/api/history?${params.toString()}`);
}

/**
 * GET /api/predictions
 * @param {string} stationId
 * @param {number|null} [runId] optional
 */
async function fetchPredictions(stationId, runId) {
  const params = new URLSearchParams({ station_id: stationId });
  if (runId != null) params.set('run_id', runId);
  return apiFetch(`/api/predictions?${params.toString()}`);
}

/** GET /api/models */
async function fetchModels() {
  return apiFetch('/api/models');
}

/**
 * GET /api/metrics
 * @param {number} runId
 * @param {string} [scope] e.g. "overall", "season:winter"
 */
async function fetchMetrics(runId, scope) {
  const params = new URLSearchParams({ run_id: runId });
  if (scope) params.set('scope', scope);
  return apiFetch(`/api/metrics?${params.toString()}`);
}

/**
 * GET /api/feature-importance
 * @param {number} [runId] optional
 */
async function fetchFeatureImportance(runId) {
  const params = runId != null ? `?run_id=${runId}` : '';
  return apiFetch(`/api/feature-importance${params}`);
}

/**
 * POST /api/predict (what-if)
 * @param {object} body
 */
async function postPredict(body) {
  return apiPost('/api/predict', body);
}

/** GET /health */
async function fetchHealth() {
  return apiFetch('/health');
}

/* ---- Exports (for modules; also available globally if loaded via <script>) ---- */
// In plain-HTML context these are global functions — no export needed.
