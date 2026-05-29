"use strict";

const API = "/dualing-simulation/api";

// Human labels + units for the parameter fields. Anything not listed falls back
// to a prettified key name with no unit, so new params still render.
const LABELS = {
  date: ["Date (YYYY-MM-DD or 'today')", ""],
  start_hour: ["Start hour of day", "h"],
  duration_s: ["Duration", "s"],
  dt_s: ["Engine step", "s"],
  cloud_factor: ["Cloud factor (1=clear)", ""],

  name: ["Name", ""],
  inverter_max_kw: ["Inverter max", "kW"],
  max_kw_per_phase: ["Max per phase", "kW"],
  three_phase: ["Three-phase", ""],
  phase: ["Phase", ""],
  pv_capacity_kw: ["PV capacity", "kW"],
  pv_tilt_deg: ["PV tilt", "°"],
  pv_azimuth_deg: ["PV azimuth (0=N)", "°"],
  battery_capacity_kwh: ["Battery capacity", "kWh"],
  soc_min_pct: ["SOC min", "%"],
  soc_max_pct: ["SOC max", "%"],
  soc_initial_pct: ["SOC initial", "%"],
  round_trip_efficiency: ["Round-trip eff.", ""],
  sample_interval_s: ["Control sample interval", "s"],
  deadband_w: ["Deadband", "W"],
  ramp_rate_w_per_s: ["Ramp rate", "W/s"],
  target_poc_import_w: ["Target POC import", "W"],
  proportional_gain: ["Proportional gain", ""],

  seed: ["RNG seed", ""],
  base_w_per_phase: ["Base load per phase", "W"],
  noise_w: ["Load noise", "W"],
  peaks_per_hour: ["Peaks per hour", ""],
  peak_min_kw: ["Peak min", "kW"],
  peak_max_kw: ["Peak max", "kW"],
  peak_min_s: ["Peak min duration", "s"],
  peak_max_s: ["Peak max duration", "s"],
};

let charts = {};

function prettyLabel(key) {
  const l = LABELS[key];
  const name = l ? l[0] : key.replace(/_/g, " ");
  const unit = l ? l[1] : "";
  return unit ? `${name} (${unit})` : name;
}

// Build one form control for a (key, value) pair, remembering the value's type
// so we can coerce it back on submit. Returns the wrapping element.
function makeField(prefix, key, value) {
  const wrap = document.createElement("label");
  wrap.className = "field";
  const span = document.createElement("span");
  span.textContent = prettyLabel(key);
  wrap.appendChild(span);

  const id = `${prefix}.${key}`;

  if (typeof value === "boolean") {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = value;
    input.dataset.path = id;
    input.dataset.kind = "bool";
    wrap.classList.add("field-bool");
    wrap.appendChild(input);
    return wrap;
  }

  if (key === "phase") {
    const select = document.createElement("select");
    select.dataset.path = id;
    select.dataset.kind = "phase";
    for (const opt of ["", "A", "B", "C"]) {
      const o = document.createElement("option");
      o.value = opt;
      o.textContent = opt === "" ? "(none / 3-phase)" : opt;
      if ((value || "") === opt) o.selected = true;
      select.appendChild(o);
    }
    wrap.appendChild(select);
    return wrap;
  }

  if (Array.isArray(value)) {
    const row = document.createElement("div");
    row.className = "triple";
    value.forEach((v, i) => {
      const input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.value = v;
      input.dataset.path = `${id}[${i}]`;
      input.dataset.kind = "number";
      row.appendChild(input);
    });
    wrap.appendChild(row);
    return wrap;
  }

  const input = document.createElement("input");
  if (typeof value === "number") {
    input.type = "number";
    input.step = "any";
    input.dataset.kind = "number";
  } else {
    input.type = "text";
    input.dataset.kind = "string";
  }
  input.value = value;
  input.dataset.path = id;
  wrap.appendChild(input);
  return wrap;
}

function renderFields(containerId, prefix, obj, skip = []) {
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  for (const [key, value] of Object.entries(obj)) {
    if (skip.includes(key)) continue;
    el.appendChild(makeField(prefix, key, value));
  }
}

function populateForm(defaults) {
  // Top-level sim window fields (everything that isn't a nested system/load).
  const sim = {
    date: defaults.date,
    start_hour: defaults.start_hour,
    duration_s: defaults.duration_s,
    dt_s: defaults.dt_s,
    cloud_factor: defaults.cloud_factor,
  };
  renderFields("sim-fields", "sim", sim);
  renderFields("estore-fields", "estore", defaults.estore);
  renderFields("solax-fields", "solax", defaults.solax);
  renderFields("load-fields", "load", defaults.load);
}

// Walk every control and rebuild the nested payload the API expects.
function collectPayload() {
  const sim = {};
  const estore = {};
  const solax = {};
  const load = {};
  const buckets = { sim, estore, solax, load };

  for (const ctrl of document.querySelectorAll("#params [data-path]")) {
    const path = ctrl.dataset.path;
    const [group, rest] = path.split(/\.(.+)/);
    const bucket = buckets[group];
    if (!bucket) continue;

    // Array element like base_w_per_phase[1]
    const arrMatch = rest.match(/^(.+)\[(\d+)\]$/);
    let value;
    switch (ctrl.dataset.kind) {
      case "bool": value = ctrl.checked; break;
      case "number": value = parseFloat(ctrl.value); break;
      case "phase": value = ctrl.value === "" ? null : ctrl.value; break;
      default: value = ctrl.value;
    }

    if (arrMatch) {
      const k = arrMatch[1];
      const idx = parseInt(arrMatch[2], 10);
      (bucket[k] = bucket[k] || [])[idx] = value;
    } else {
      bucket[rest] = value;
    }
  }

  return { ...sim, estore, solax, load };
}

// ---- time axis helpers -------------------------------------------------

function clockLabels(t_s, startHour) {
  const base = startHour * 3600;
  return t_s.map((t) => {
    const secs = base + t;
    const h = Math.floor(secs / 3600) % 24;
    const m = Math.floor((secs % 3600) / 60);
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
  });
}

const PALETTE = {
  estore: "#2563eb",
  solax: "#16a34a",
  target: "#9333ea",
  poc: "#dc2626",
  amber: "#d97706",
};

function lineChart(canvasId, title, labels, datasets, yLabel) {
  if (charts[canvasId]) charts[canvasId].destroy();
  const ctx = document.getElementById(canvasId).getContext("2d");
  charts[canvasId] = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      elements: { point: { radius: 0 } },
      plugins: {
        title: { display: true, text: title },
        legend: { position: "bottom" },
        decimation: { enabled: true, algorithm: "lttb", samples: 600 },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 12, autoSkip: true } },
        y: { title: { display: !!yLabel, text: yLabel } },
      },
    },
  });
}

function ds(label, data, color, opts = {}) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: color,
    borderWidth: 1.5,
    tension: 0,
    ...opts,
  };
}

function renderCharts(payload, r) {
  const labels = clockLabels(r.t_s, payload.start_hour);

  lineChart("chart-poc", "Power at point of connection (import +)", labels, [
    ds("POC total", r.poc_total_w, PALETTE.poc),
    ds("Target", r.poc_total_w.map(() => r.target_w), PALETTE.target,
       { borderDash: [6, 4], borderWidth: 1 }),
  ], "W");

  lineChart("chart-ac", "Inverter AC output", labels, [
    ds("eStore", r.estore_ac_kw, PALETTE.estore),
    ds("SolaX (total)", r.solax_ac_total_kw, PALETTE.solax),
  ], "kW");

  lineChart("chart-soc", "Battery state of charge", labels, [
    ds("eStore", r.estore_soc_pct, PALETTE.estore),
    ds("SolaX", r.solax_soc_pct, PALETTE.solax),
  ], "%");

  lineChart("chart-pv", "Solar PV generation (dashed = curtailed)", labels, [
    ds("eStore PV", r.pv_estore_kw, PALETTE.estore),
    ds("SolaX PV", r.pv_solax_kw, PALETTE.solax),
    ds("eStore curtailed", r.estore_curtail_kw, PALETTE.amber, { borderDash: [4, 3] }),
    ds("SolaX curtailed", r.solax_curtail_kw, PALETTE.poc, { borderDash: [4, 3] }),
  ], "kW");
}

// Integrate a 1 Hz kW series to kWh using the sample spacing.
function energyKwh(series, t_s) {
  const dt_h = t_s.length > 1 ? (t_s[1] - t_s[0]) / 3600 : 0;
  return series.reduce((s, x) => s + x, 0) * dt_h;
}

function mean(a) {
  return a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
}

function renderSummary(r) {
  // Skip the startup transient: judge settling on the last two-thirds.
  const tail = r.poc_total_w.slice(Math.floor(r.poc_total_w.length / 3));
  const cards = [
    ["Mean POC (settled)", `${mean(tail).toFixed(0)} W`],
    ["Target", `${r.target_w.toFixed(0)} W`],
    ["Peak POC import", `${Math.max(...r.poc_total_w).toFixed(0)} W`],
    ["Min POC (export)", `${Math.min(...r.poc_total_w).toFixed(0)} W`],
    ["eStore final SOC", `${r.estore_soc_pct.at(-1).toFixed(1)} %`],
    ["SolaX final SOC", `${r.solax_soc_pct.at(-1).toFixed(1)} %`],
    ["PV curtailed", `${(energyKwh(r.estore_curtail_kw, r.t_s) + energyKwh(r.solax_curtail_kw, r.t_s)).toFixed(2)} kWh`],
  ];
  document.getElementById("summary").innerHTML = cards
    .map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join("");
}

function renderMetrics(m) {
  const el = document.getElementById("metrics");
  const title = document.getElementById("metrics-title");
  if (!m) {
    el.innerHTML = "";
    title.hidden = true;
    return;
  }
  const period = m.poc_dominant_period_s > 0
    ? `${m.poc_dominant_period_s.toFixed(1)} s`
    : "—";
  const cards = [
    ["RMS error vs target", `${m.poc_rms_error_w.toFixed(0)} W`],
    ["Peak-to-peak swing", `${m.poc_peak_to_peak_w.toFixed(0)} W`],
    ["Oscillation period", period],
    ["Target crossings", `${m.target_crossings_per_min.toFixed(1)} /min`],
    ["Control effort", `${m.control_effort_kw.toFixed(1)} kW`],
    ["Export caused", `${m.export_energy_kwh.toFixed(2)} kWh (${m.export_fraction_pct.toFixed(0)}%)`],
  ];
  title.hidden = false;
  el.innerHTML = cards
    .map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join("");
}

async function runSimulation(ev) {
  if (ev) ev.preventDefault();
  const btn = document.getElementById("run-btn");
  const status = document.getElementById("status-line");
  const payload = collectPayload();
  btn.disabled = true;
  status.textContent = "Running…";
  const t0 = performance.now();
  try {
    const resp = await fetch(`${API}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const r = await resp.json();
    renderSummary(r);
    renderMetrics(r.metrics);
    renderCharts(payload, r);
    const ms = (performance.now() - t0).toFixed(0);
    status.textContent = `Done — ${r.t_s.length} samples over ${payload.duration_s}s (${ms} ms).`;
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

let DEFAULTS = null;

async function loadDefaults() {
  const r = await fetch(`${API}/defaults`);
  DEFAULTS = await r.json();
  populateForm(DEFAULTS);
}

async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const j = await r.json();
    document.getElementById("health").textContent = `ok · phase ${j.phase} · ${j.stage}`;
  } catch (e) {
    document.getElementById("health").textContent = "error: " + e.message;
  }
}

document.getElementById("params").addEventListener("submit", runSimulation);
document.getElementById("reset-btn").addEventListener("click", () => {
  if (DEFAULTS) populateForm(DEFAULTS);
});

checkHealth();
loadDefaults();
