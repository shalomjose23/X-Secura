"use strict";

// ── state ──────────────────────────────────────────────────────────────────
const state = {
  currentPage : 1,
  perPage     : 50,
  totalRecords: 0,
  minRisk     : 0,
  search      : "",
  chartRisk    : null,
  chartFlagged : null,
  chartCompare : null,
};

// ── helpers ────────────────────────────────────────────────────────────────
function showSpinner(msg = "Analysing …") {
  document.querySelector(".spinner-label").textContent = msg;
  document.getElementById("spinner").classList.add("visible");
}
function hideSpinner() {
  document.getElementById("spinner").classList.remove("visible");
}

async function apiFetch(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

function riskClass(score) {
  if (score >= 0.6) return "risk-high";
  if (score >= 0.3) return "risk-medium";
  return "risk-low";
}

function riskLabel(score) {
  if (score >= 0.6) return "HIGH";
  if (score >= 0.3) return "MED";
  return "LOW";
}

// ── view switching ─────────────────────────────────────────────────────────
let logsimHasRunOnce = false;

function switchView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(`view-${name}`).classList.add("active");
  // detail views have no sidebar nav button — guard against null
  const navBtn = document.querySelector(`[data-view="${name}"]`);
  if (navBtn) navBtn.classList.add("active");
  if (name === "temporal") loadTemporalCharts();
  if (name === "logsim" && !logsimHasRunOnce) runLogSim();
}

document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => switchView(btn.dataset.view));
});

// ══════════════════════════════════════════════════════════════════════════
// 1. RISK OVERVIEW
// ══════════════════════════════════════════════════════════════════════════
async function loadOverview(page = 1) {
  showSpinner("Loading risk data …");
  state.currentPage = page;

  const url = `/api/risk_overview?page=${page}&per_page=${state.perPage}`
            + `&min_risk=${state.minRisk}&search=${encodeURIComponent(state.search)}`;
  try {
    const data = await apiFetch(url);
    state.totalRecords = data.total;
    renderOverviewTable(data.rows, page);
    renderPagination(data.total, page, data.per_page);
    updateStatBar(data.rows, data.total);
  } catch(e) {
    console.error(e);
  } finally {
    hideSpinner();
  }
}

function updateStatBar(rows, total) {
  const flagged  = rows.filter(r => r.prediction === 1).length;
  const insiders = rows.filter(r => r.is_insider  === 1).length;
  const avgRisk  = rows.length ? (rows.reduce((s,r) => s + r.risk_score, 0) / rows.length).toFixed(3) : "—";
  document.getElementById("s-total").textContent   = total.toLocaleString();
  document.getElementById("s-flagged").textContent  = flagged;
  document.getElementById("s-insiders").textContent = insiders;
  document.getElementById("s-avg-risk").textContent = avgRisk;
}

function renderOverviewTable(rows, page) {
  const tbody = document.getElementById("risk-tbody");
  tbody.innerHTML = "";
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="12" style="text-align:center;color:var(--text-muted);padding:32px">No records found.</td></tr>`;
    return;
  }
  rows.forEach((r, i) => {
    const rank  = (page - 1) * state.perPage + i + 1;
    const rc    = riskClass(r.risk_score);
    const rl    = riskLabel(r.risk_score);
    const fl    = r.prediction === 1
      ? `<span class="label-pill label-insider">FLAGGED</span>`
      : `<span class="label-pill label-normal">—</span>`;
    const il    = r.is_insider === 1
      ? `<span class="label-pill label-insider">INSIDER</span>`
      : `<span class="label-pill label-normal">NORMAL</span>`;
    tbody.insertAdjacentHTML("beforeend", `
      <tr>
        <td class="rank-cell">${rank}</td>
        <td class="user-cell">${r.user}</td>
        <td>${r.day}</td>
        <td><span class="risk-pill ${rc}">${r.risk_score.toFixed(4)} &nbsp;${rl}</span></td>
        <td>${fl}</td>
        <td>${il}</td>
        <td>${r.unique_pcs}</td>
        <td>${r.file_count}</td>
        <td>${r.email_count}</td>
        <td>${r.http_count}</td>
        <td>${r.device_count}</td>
        <td><button class="detail-btn" data-id="${r.user_day_id}">Drill-down</button></td>
      </tr>`);
  });

  // drill-down buttons
  tbody.querySelectorAll(".detail-btn").forEach(btn => {
    btn.addEventListener("click", () => loadUserDayDetail(+btn.dataset.id));
  });
}

function renderPagination(total, current, perPage) {
  const pages = Math.ceil(total / perPage);
  const pg    = document.getElementById("pagination");
  pg.innerHTML = "";
  if (pages <= 1) return;

  const addBtn = (label, page, cls = "") => {
    const b = document.createElement("button");
    b.className = "page-btn" + (cls ? " " + cls : "");
    b.textContent = label;
    if (cls !== "active") b.addEventListener("click", () => loadOverview(page));
    pg.appendChild(b);
  };

  if (current > 1)   addBtn("← Prev", current - 1);
  const start = Math.max(1, current - 2);
  const end   = Math.min(pages, start + 4);
  for (let p = start; p <= end; p++) addBtn(p, p, p === current ? "active" : "");
  if (current < pages) addBtn("Next →", current + 1);

  const info = document.createElement("span");
  info.className = "page-info";
  info.textContent = `${total.toLocaleString()} records`;
  pg.appendChild(info);
}

// filter controls
document.getElementById("apply-filter").addEventListener("click", () => {
  state.minRisk = parseFloat(document.getElementById("min-risk-input").value) || 0;
  state.search  = document.getElementById("search-input").value.trim();
  loadOverview(1);
});
document.getElementById("search-input").addEventListener("keydown", e => {
  if (e.key === "Enter") document.getElementById("apply-filter").click();
});

// ══════════════════════════════════════════════════════════════════════════
// 2. USER-DAY DETAIL
// ══════════════════════════════════════════════════════════════════════════
async function loadUserDayDetail(uid) {
  showSpinner("Computing SHAP explanations …");
  try {
    const res  = await fetch(`/api/user_day/${uid}`);
    const data = await res.json();
    if (!res.ok) {
      // show the actual server error message in the detail panel
      switchView("detail");
      document.getElementById("detail-title").textContent = "Error loading record";
      document.getElementById("detail-summary").innerHTML =
        `<div style="color:var(--accent-red);font-family:var(--font-mono);font-size:12px;padding:12px">
           <strong>Server error ${res.status}</strong><br><br>${data.error || JSON.stringify(data)}
         </div>`;
      document.getElementById("detail-force-plot").innerHTML = "";
      document.getElementById("detail-feat-table").innerHTML  = "";
      console.error("API error", res.status, data);
      return;
    }
    renderDetailView(data);
    switchView("detail");
  } catch(e) {
    console.error("loadUserDayDetail failed:", e);
  } finally {
    hideSpinner();
  }
}

function renderDetailView(data) {
  document.getElementById("detail-title").textContent =
    `User ${data.user}  ·  Day ${data.day}`;

  // summary card
  const rc = riskClass(data.risk_score);
  const rl = riskLabel(data.risk_score);
  document.getElementById("detail-summary").innerHTML = `
    <h3 class="card-title">Record Summary</h3>
    <div style="margin-bottom:12px;margin-top:8px">
      <span class="risk-pill ${rc}" style="font-size:16px;padding:6px 14px">
        ${data.risk_score.toFixed(4)} &nbsp; ${rl}
      </span>
    </div>
    ${field("User",       data.user)}
    ${field("Day",        data.day)}
    ${field("Flagged",    data.prediction === 1 ? "YES — SUSPICIOUS" : "No")}
    ${field("True Label", data.is_insider  === 1 ? "INSIDER"         : "Normal")}
  `;

  // force plot
  const fp = document.getElementById("detail-force-plot");
  fp.innerHTML = `<img src="data:image/png;base64,${data.force_plot}" alt="SHAP force plot"/>`;

  // feature values
  const fg = document.getElementById("detail-feat-table");
  fg.innerHTML = Object.entries(data.feat_vals)
    .map(([k,v]) => `<div class="feat-cell">
      <div class="feat-name">${k}</div>
      <div class="feat-value">${typeof v === "number" ? v.toFixed(2) : v}</div>
    </div>`).join("");
}

function field(k, v) {
  return `<div class="summary-field">
    <span class="summary-key">${k}</span>
    <span class="summary-val">${v}</span>
  </div>`;
}

document.getElementById("back-btn").addEventListener("click", () => switchView("overview"));

// ══════════════════════════════════════════════════════════════════════════
// 3. TEMPORAL TRENDS
// ══════════════════════════════════════════════════════════════════════════
async function loadTemporalCharts() {
  if (state._temporalLoaded) return;
  showSpinner("Building temporal charts …");
  try {
    const d = await apiFetch("/api/temporal_trend");
    buildTemporalCharts(d);
    state._temporalLoaded = true;
  } catch(e) { console.error(e); }
  finally { hideSpinner(); }
}

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: true,
  plugins: {
    legend: { labels: { color: "#8b92a8", font: { size: 11 } } },
    tooltip: { backgroundColor: "#1b1f2e", titleColor: "#e8ecf4", bodyColor: "#8b92a8" },
  },
  scales: {
    x: { ticks: { color: "#525a72", maxTicksLimit: 12, font: { size: 10 } },
         grid:  { color: "#1f2435" } },
    y: { ticks: { color: "#525a72", font: { size: 10 } },
         grid:  { color: "#1f2435" } },
  },
};

function buildTemporalCharts(d) {
  const days = d.days;

  // destroy old charts
  ["chartRisk","chartFlagged","chartCompare"].forEach(k => {
    if (state[k]) { state[k].destroy(); state[k] = null; }
  });

  // avg risk
  state.chartRisk = new Chart(document.getElementById("chart-risk"), {
    type: "line",
    data: {
      labels: days,
      datasets: [{ label: "Avg Risk Score", data: d.avg_risk,
        borderColor: "#4f6ef7", backgroundColor: "rgba(79,110,247,0.1)",
        fill: true, tension: 0.3, pointRadius: 0 }],
    },
    options: { ...CHART_DEFAULTS },
  });

  // flagged per day
  state.chartFlagged = new Chart(document.getElementById("chart-flagged"), {
    type: "bar",
    data: {
      labels: days,
      datasets: [{ label: "Flagged", data: d.flagged,
        backgroundColor: "rgba(224,82,82,0.6)", borderColor: "#e05252", borderWidth: 1 }],
    },
    options: { ...CHART_DEFAULTS },
  });

  // compare insider vs flagged
  state.chartCompare = new Chart(document.getElementById("chart-compare"), {
    type: "line",
    data: {
      labels: days,
      datasets: [
        { label: "Flagged (Model)", data: d.flagged,
          borderColor: "#4f6ef7", backgroundColor: "rgba(79,110,247,0.08)",
          fill: true, tension: 0.3, pointRadius: 0 },
        { label: "True Insiders",  data: d.insider_count,
          borderColor: "#e05252", backgroundColor: "rgba(224,82,82,0.08)",
          fill: true, tension: 0.3, pointRadius: 4 },
      ],
    },
    options: { ...CHART_DEFAULTS },
  });
}

// ══════════════════════════════════════════════════════════════════════════
// 4. GLOBAL SHAP
// ══════════════════════════════════════════════════════════════════════════
document.getElementById("load-shap-btn").addEventListener("click", async () => {
  showSpinner("Computing global SHAP (sampled) …");
  try {
    const data = await apiFetch("/api/shap_global");

    document.getElementById("shap-plot-wrap").innerHTML =
      `<img src="data:image/png;base64,${data.image}" alt="SHAP bar chart"/>`;

    const list = document.getElementById("shap-ranking");
    list.innerHTML = data.features.slice(0, 15).map(f =>
      `<li><code style="font-family:var(--font-mono);color:var(--accent-blue);font-size:11px">${f}</code></li>`
    ).join("");
  } catch(e) { alert("SHAP computation failed."); console.error(e); }
  finally { hideSpinner(); }
});

// ══════════════════════════════════════════════════════════════════════════
// 5. SCENARIO TESTING
// ══════════════════════════════════════════════════════════════════════════
async function initScenarios() {
  try {
    const keys = await apiFetch("/api/scenarios");
    const grid = document.getElementById("scenario-grid");
    grid.innerHTML = "";

    // fetch labels by running each briefly — use static map for UI speed
    const labels = {
      normal_activity    : { label: "Normal Activity",        desc: "Typical 9-to-5 employee with moderate email and file usage." },
      data_exfiltration  : { label: "Data Exfiltration",      desc: "Large USB transfer at 23:45 with abnormal file and device activity." },
      email_exfiltration : { label: "Email Exfiltration",     desc: "Mass emails to external addresses with high attachment count." },
      off_hours_browsing : { label: "Off-Hours Web Browsing", desc: "Heavy HTTP activity between midnight and 4 AM." },
    };

    keys.forEach(key => {
      const info = labels[key] || { label: key, desc: "" };
      const card = document.createElement("div");
      card.className = "scenario-card";
      card.dataset.key = key;
      card.innerHTML = `<div class="sc-label">${info.label}</div>
                        <div class="sc-desc">${info.desc}</div>`;
      card.addEventListener("click", () => runScenario(key, card));
      grid.appendChild(card);
    });
  } catch(e) { console.error(e); }
}

async function runScenario(key, cardEl) {
  document.querySelectorAll(".scenario-card").forEach(c => c.classList.remove("selected"));
  cardEl.classList.add("selected");

  showSpinner("Running scenario …");
  try {
    const data = await apiFetch(`/api/scenario/${key}`);
    renderScenarioResult(data);
  } catch(e) { alert("Scenario failed."); console.error(e); }
  finally { hideSpinner(); }
}

function renderScenarioResult(data) {
  const wrap = document.getElementById("scenario-result");
  wrap.style.display = "block";

  document.getElementById("sc-result-title").textContent =
    `Result: ${data.label}`;
  document.getElementById("sc-result-desc").textContent = data.description;

  const rc  = data.risk_score >= 0.5 ? "badge-risk-high"   : "badge-risk-normal";
  const flc = data.prediction === 1  ? "badge-flagged"      : "badge-safe";
  const flt = data.prediction === 1  ? "⚠ FLAGGED AS SUSPICIOUS" : "✓ Predicted Normal";

  document.getElementById("sc-badges").innerHTML = `
    <span class="badge ${rc}">Risk: ${data.risk_score.toFixed(4)}</span>
    <span class="badge ${flc}">${flt}</span>`;

  document.getElementById("sc-plot-wrap").innerHTML =
    `<img src="data:image/png;base64,${data.image}" alt="Scenario SHAP chart"/>`;

  wrap.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ══════════════════════════════════════════════════════════════════════════
// 6. LOG SIMULATION — fully automatic backend pipeline
// ══════════════════════════════════════════════════════════════════════════
// Flow: page load (first visit to this tab) → POST-less GET /api/log_sim/run
// → backend generates N random user-day logs → parses → engineers features
// → scores through XGBoost → runs SHAP → returns ranked table.
// Drill-down fetches /api/log_sim/detail/<id> for the cached record's
// full SHAP force plot + raw log + engineered features.
// (logsimHasRunOnce is declared earlier, alongside switchView)

document.getElementById("logsim-run-btn").addEventListener("click", () => runLogSim());

async function runLogSim() {
  const nInput = document.getElementById("logsim-count-input");
  let n = parseInt(nInput.value, 10);
  if (isNaN(n)) n = 30;
  n = Math.max(5, Math.min(50, n));
  nInput.value = n;

  clearLogsimError();
  showSpinner(`Generating & scoring ${n} synthetic user-days …`);

  const tbody = document.getElementById("logsim-tbody");
  tbody.innerHTML = `<tr><td colspan="12" style="text-align:center;color:var(--text-muted);padding:32px">
    Running simulation …</td></tr>`;

  try {
    const data = await apiFetch(`/api/log_sim/run?n=${n}`);
    renderLogsimTable(data.rows);
    updateLogsimStats(data);
    logsimHasRunOnce = true;
  } catch(e) {
    showLogsimError("Simulation failed. Check server logs.");
    console.error(e);
  } finally {
    hideSpinner();
  }
}

function updateLogsimStats(data) {
  document.getElementById("ls-total").textContent    = data.total;
  document.getElementById("ls-flagged").textContent  = data.flagged_count;
  document.getElementById("ls-avg-risk").textContent = data.avg_risk.toFixed(3);
}

function renderLogsimTable(rows) {
  const tbody = document.getElementById("logsim-tbody");
  tbody.innerHTML = "";

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="12" style="text-align:center;color:var(--text-muted);padding:32px">
      No records generated.</td></tr>`;
    return;
  }

  rows.forEach((r, i) => {
    const rc = riskClass(r.risk_score);
    const rl = riskLabel(r.risk_score);
    const fl = r.prediction === 1
      ? `<span class="label-pill label-insider">FLAGGED</span>`
      : `<span class="label-pill label-normal">—</span>`;

    tbody.insertAdjacentHTML("beforeend", `
      <tr>
        <td class="rank-cell">${i + 1}</td>
        <td class="user-cell">${r.user}</td>
        <td>${r.day}</td>
        <td><span class="risk-pill ${rc}">${r.risk_score.toFixed(4)} &nbsp;${rl}</span></td>
        <td>${fl}</td>
        <td>${r.file_count}</td>
        <td>${r.email_count}</td>
        <td>${r.http_count}</td>
        <td>${r.device_count}</td>
        <td>${r.unique_pcs}</td>
        <td><button class="detail-btn" data-id="${r.sim_id}">Drill-down</button></td>
      </tr>`);
  });

  tbody.querySelectorAll(".detail-btn").forEach(btn => {
    btn.addEventListener("click", () => loadLogsimDetail(+btn.dataset.id));
  });
}

async function loadLogsimDetail(simId) {
  showSpinner("Loading SHAP explanation …");
  try {
    const data = await apiFetch(`/api/log_sim/detail/${simId}`);
    renderLogsimDetail(data);
    switchView("logsim-detail");
  } catch(e) {
    showLogsimError("Failed to load record detail.");
    console.error(e);
  } finally {
    hideSpinner();
  }
}

function renderLogsimDetail(data) {
  document.getElementById("logsim-detail-title").textContent =
    `User ${data.user}  ·  Day ${data.day}  ·  ${data.scenario_label}`;

  const rc = riskClass(data.risk_score);
  const rl = riskLabel(data.risk_score);
  document.getElementById("logsim-detail-summary").innerHTML = `
    <h3 class="card-title">Record Summary</h3>
    <div style="margin-bottom:12px;margin-top:8px">
      <span class="risk-pill ${rc}" style="font-size:16px;padding:6px 14px">
        ${data.risk_score.toFixed(4)} &nbsp; ${rl}
      </span>
    </div>
    ${field("User",     data.user)}
    ${field("Day",      data.day)}
    ${field("Scenario Generated", data.scenario_label)}
    ${field("Flagged",  data.prediction === 1 ? "YES — SUSPICIOUS" : "No")}
  `;

  document.getElementById("logsim-detail-plot").innerHTML =
    `<img src="data:image/png;base64,${data.image}" alt="SHAP force plot"/>`;

  const fg = document.getElementById("logsim-detail-feat-grid");
  fg.innerHTML = Object.entries(data.engineered)
    .map(([k, v]) => `<div class="feat-cell">
      <div class="feat-name">${k}</div>
      <div class="feat-value">${v}</div>
    </div>`).join("");

  document.getElementById("logsim-detail-csv").textContent      = data.csv_text;
  document.getElementById("logsim-detail-readable").textContent = data.readable_text;
}

document.getElementById("logsim-back-btn").addEventListener("click", () => switchView("logsim"));

function showLogsimError(msg) {
  document.getElementById("logsim-error").textContent = msg;
}
function clearLogsimError() {
  document.getElementById("logsim-error").textContent = "";
}

// ── init ───────────────────────────────────────────────────────────────────
loadOverview(1);
initScenarios();