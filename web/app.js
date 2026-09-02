/**
 * AetherQuant-MT5 — SOTA Institutional Live Terminal Frontend Controller
 * 100% Real MT5 Terminal & 23-Channel MacroSuperPatchTST Telemetry
 */

let activeSymbol = "EURUSD";
let priceChart = null;
let currentLogFilter = "all";
let rawLogsCache = [];

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initChart();
  
  // Initial data fetches
  fetchTelemetry();
  fetchBars(activeSymbol);
  fetchEconomicCalendar();
  fetchLogs();

  // High-frequency live polling
  setInterval(fetchTelemetry, 1500);
  setInterval(fetchEconomicCalendar, 30000);
  setInterval(fetchLogs, 2000);
  setInterval(() => fetchBars(activeSymbol), 10000);
});

// ─────────────────────────────────────────────────────────────────────────────
// 1. CLOCK
// ─────────────────────────────────────────────────────────────────────────────
function initClock() {
  const clockEl = document.getElementById("clock");
  function update() {
    const now = new Date();
    clockEl.textContent = now.toUTCString().split(" ")[4] + " UTC";
  }
  update();
  setInterval(update, 1000);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. CHART CONTROLLER (MULTI-ASSET READY)
// ─────────────────────────────────────────────────────────────────────────────
function initChart() {
  const ctx = document.getElementById("priceChart").getContext("2d");
  
  const gradient = ctx.createLinearGradient(0, 0, 0, 280);
  gradient.addColorStop(0, "rgba(0, 240, 255, 0.20)");
  gradient.addColorStop(1, "rgba(0, 240, 255, 0.00)");

  priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "H1 Close",
          data: [],
          borderColor: "#00f0ff",
          backgroundColor: gradient,
          borderWidth: 2.2,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: "#00f0ff",
          fill: true,
          tension: 0.2,
        },
        {
          label: "Active Entry",
          data: [],
          borderColor: "rgba(255, 255, 255, 0.7)",
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
        },
        {
          label: "Emergency Disaster SL (2.5x ATR)",
          data: [],
          borderColor: "rgba(255, 51, 102, 0.85)",
          borderWidth: 1.5,
          borderDash: [3, 3],
          pointRadius: 0,
          fill: false,
        },
        {
          label: "Target Alpha TP",
          data: [],
          borderColor: "rgba(0, 255, 136, 0.85)",
          borderWidth: 1.5,
          borderDash: [3, 3],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(10, 15, 25, 0.95)",
          titleColor: "#94a3b8",
          bodyColor: "#f3f6fb",
          borderColor: "rgba(255, 255, 255, 0.12)",
          borderWidth: 1,
          padding: 10,
          bodyFont: { family: "JetBrains Mono", size: 11 },
          titleFont: { family: "Inter", size: 11 }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.03)" },
          ticks: {
            color: "#475569",
            font: { family: "JetBrains Mono", size: 10 },
            maxTicksLimit: 8
          }
        },
        y: {
          position: "right",
          grid: { color: "rgba(255, 255, 255, 0.03)" },
          ticks: {
            color: "#94a3b8",
            font: { family: "JetBrains Mono", size: 10 },
            callback: (val) => val >= 100 ? val.toFixed(2) : val.toFixed(5)
          }
        }
      }
    }
  });
}

function switchAsset(symbol) {
  activeSymbol = symbol;
  
  // Update Tab Styling
  document.querySelectorAll(".asset-tab").forEach(tab => {
    if (tab.getAttribute("data-symbol") === symbol) {
      tab.classList.add("active");
    } else {
      tab.classList.remove("active");
    }
  });

  document.getElementById("stat-symbol").textContent = symbol;
  fetchBars(symbol);
}

async function fetchBars(symbol = "EURUSD") {
  try {
    const res = await fetch(`/api/bars?symbol=${encodeURIComponent(symbol)}&count=60`);
    const data = await res.json();
    const bars = data.bars || [];

    if (bars.length === 0 || !priceChart) return;

    const labels = bars.map(b => b.time ? b.time.substring(11, 16) : "");
    const closes = bars.map(b => b.close);

    const latest = bars[bars.length - 1];
    document.getElementById("stat-price").textContent = latest.close >= 100 ? latest.close.toFixed(2) : latest.close.toFixed(5);
    document.getElementById("stat-open").textContent = latest.open >= 100 ? latest.open.toFixed(2) : latest.open.toFixed(5);
    document.getElementById("stat-high").textContent = latest.high >= 100 ? latest.high.toFixed(2) : latest.high.toFixed(5);
    document.getElementById("stat-low").textContent = latest.low >= 100 ? latest.low.toFixed(2) : latest.low.toFixed(5);

    priceChart.data.labels = labels;
    priceChart.data.datasets[0].data = closes;
    priceChart.data.datasets[0].label = `${symbol} H1 Close`;

    // Check if open position exists for this symbol to draw SL/TP bands
    const posRes = await fetch("/api/status");
    const statusData = await posRes.json();
    const openPositions = statusData.open_positions || [];
    const activePos = openPositions.find(p => (p.symbol || "").toUpperCase().includes(symbol.toUpperCase()));

    if (activePos && activePos.price_open) {
      priceChart.data.datasets[1].data = new Array(closes.length).fill(activePos.price_open);
      priceChart.data.datasets[2].data = new Array(closes.length).fill(activePos.sl);
      priceChart.data.datasets[3].data = new Array(closes.length).fill(activePos.tp);
    } else {
      priceChart.data.datasets[1].data = [];
      priceChart.data.datasets[2].data = [];
      priceChart.data.datasets[3].data = [];
    }

    priceChart.update();
  } catch (err) {
    console.error("Error fetching bars:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. TELEMETRY & DOM SYNC
// ─────────────────────────────────────────────────────────────────────────────
async function fetchTelemetry() {
  const startTime = performance.now();
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const latency = Math.round(performance.now() - startTime);

    document.getElementById("latency").textContent = `${latency}ms`;

    // 1. Account & KPI Ribbon
    const acc = data.account || {};
    if (acc.login) {
      document.getElementById("account-id").textContent = `#${acc.login}`;
      document.getElementById("kpi-equity").textContent = `$${Number(acc.equity).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      document.getElementById("kpi-balance-sub").textContent = `Balance: $${Number(acc.balance).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      document.getElementById("kpi-leverage").textContent = `1:${acc.leverage || 100} MT5`;
    }

    // 2. Open Positions Table
    const posList = data.open_positions || [];
    const posBadge = document.getElementById("pos-count-badge");
    posBadge.textContent = `${posList.length} Open ${posList.length === 1 ? "Position" : "Positions"}`;

    const tbody = document.getElementById("positions-body");
    let totalFloating = 0.0;

    if (posList.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" class="text-center py-4 text-dim">No active positions (Portfolio risk budget available: 0.60%).</td></tr>`;
      document.getElementById("kpi-floating-pnl").textContent = "$0.00";
      document.getElementById("kpi-floating-pnl").className = "kpi-main text-muted";
    } else {
      let rowsHtml = "";
      posList.forEach(p => {
        totalFloating += Number(p.profit || 0);
        const isBuy = (p.type || "").toUpperCase() === "BUY";
        const sideClass = isBuy ? "badge-green" : "badge-coral";
        const pnlClass = p.profit >= 0 ? "text-success" : "text-coral";
        const pnlSign = p.profit >= 0 ? "+" : "";
        const isFx = (p.symbol || "").includes("EUR");
        const dec = isFx ? 5 : 2;

        rowsHtml += `
          <tr>
            <td class="mono font-bold text-white">#${p.ticket}</td>
            <td class="mono font-bold text-cyan">${p.symbol}</td>
            <td><span class="badge ${sideClass}">${p.type}</span></td>
            <td class="mono">${Number(p.volume).toFixed(2)}</td>
            <td class="mono">${Number(p.price_open).toFixed(dec)}</td>
            <td class="mono text-white">${Number(p.price_current).toFixed(dec)}</td>
            <td class="mono text-coral">${Number(p.sl).toFixed(dec)}</td>
            <td class="mono text-success">${Number(p.tp).toFixed(dec)}</td>
            <td class="mono font-bold ${pnlClass}">${pnlSign}$${Number(p.profit).toFixed(2)}</td>
            <td><span class="badge badge-ai">Model Dynamic Exit</span></td>
          </tr>
        `;
      });
      tbody.innerHTML = rowsHtml;

      const pnlSign = totalFloating >= 0 ? "+" : "";
      const pnlEl = document.getElementById("kpi-floating-pnl");
      pnlEl.textContent = `${pnlSign}$${totalFloating.toFixed(2)}`;
      pnlEl.className = totalFloating >= 0 ? "kpi-main text-success" : "kpi-main text-coral";
    }

    // 3. Performance & Closed Deals
    const state = data.daemon_state || {};
    const perf = state.performance || {};
    const netPnl = perf.net_pnl_usd || 0;
    const pnlTag = netPnl >= 0 ? `+$${netPnl.toFixed(2)}` : `-$${Math.abs(netPnl).toFixed(2)}`;
    document.getElementById("kpi-closed-pnl").innerHTML = `Realized: <strong class="${netPnl >= 0 ? "text-success" : "text-coral"}">${pnlTag}</strong> (${perf.wins || 0}W / ${perf.losses || 0}L)`;

    // 4. Update 23-Channel Signal Tiles
    const sigs = state.signals || {};
    ["EURUSD", "NAS100", "WTI"].forEach(sym => {
      const sData = sigs[sym];
      const actionEl = document.getElementById(`sig-action-${sym}`);
      const fcstEl = document.getElementById(`sig-fcst-${sym}`);
      if (sData && actionEl && fcstEl) {
        const ret = sData.forecast_return || 0;
        const sig = sData.signal || "FLAT";
        fcstEl.textContent = `Forecast: ${ret >= 0 ? "+" : ""}${ret.toFixed(6)}`;
        if (sig === "BUY") {
          actionEl.textContent = "BUY (LONG)";
          actionEl.className = "tile-action text-success";
        } else if (sig === "SELL") {
          actionEl.textContent = "SELL (SHORT)";
          actionEl.className = "tile-action text-coral";
        } else {
          actionEl.textContent = "FLAT (NO SIGNAL)";
          actionEl.className = "tile-action text-muted";
        }
      }
    });

  } catch (err) {
    console.error("Error in fetchTelemetry:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. ECONOMIC CALENDAR PIPELINE
// ─────────────────────────────────────────────────────────────────────────────
async function fetchEconomicCalendar() {
  try {
    const res = await fetch("/api/economic_calendar");
    const data = await res.json();
    const events = data.events || [];
    const isBlackout = data.is_blackout || false;

    const shieldBadge = document.getElementById("shield-status-badge");
    if (isBlackout) {
      shieldBadge.textContent = "🛑 NEWS BLACKOUT ACTIVE";
      shieldBadge.className = "badge badge-coral";
    } else {
      shieldBadge.textContent = "🛡️ SHIELD IDLE (MARKET OPEN)";
      shieldBadge.className = "badge badge-green";
    }

    const container = document.getElementById("macro-events-list");
    if (events.length === 0) {
      container.innerHTML = `<div class="macro-loading text-dim text-center py-2">No Tier-1 releases scheduled in the next 72 hours.</div>`;
      return;
    }

    let html = "";
    events.forEach(ev => {
      const isImminent = ev.hours_until <= 3.0;
      const statusBadge = isImminent ? `<span class="badge badge-coral">T-${Math.max(0, ev.hours_until)}h</span>` : `<span class="badge badge-dim">in ${ev.hours_until}h</span>`;
      const cleanName = ev.name.replace(/^(USD|EUR|GBP)\s+/, "");
      const impactClass = ev.impact === 3 ? "badge-coral" : "badge-amber";
      const impactText = ev.impact === 3 ? "HIGH" : "MED";

      html += `
        <div class="macro-event-item">
          <div class="event-left">
            <span class="event-cur">${ev.currency}</span>
            <span class="event-name">${cleanName}</span>
          </div>
          <div class="event-right">
            ${statusBadge}
            <span class="badge ${impactClass}">${impactText}</span>
          </div>
        </div>
      `;
    });
    container.innerHTML = html;
  } catch (err) {
    console.error("Error fetching economic calendar:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. LOG CONSOLE WITH FILTERS
// ─────────────────────────────────────────────────────────────────────────────
async function fetchLogs() {
  try {
    const res = await fetch("/api/logs?lines=60");
    const data = await res.json();
    rawLogsCache = data.logs || [];
    renderLogs();
  } catch (err) {
    console.error("Error fetching logs:", err);
  }
}

function filterLogs(category) {
  currentLogFilter = category;
  document.querySelectorAll(".filter-btn").forEach(btn => {
    if (btn.getAttribute("data-filter") === category) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });
  renderLogs();
}

function renderLogs() {
  const terminal = document.getElementById("log-terminal");
  if (!terminal) return;

  const filtered = rawLogsCache.filter(line => {
    if (currentLogFilter === "all") return true;
    if (currentLogFilter === "order") return line.includes("ORDER") || line.includes("send_market_order");
    if (currentLogFilter === "win") return line.includes("WIN") || line.includes("DYNAMIC MODEL EXIT") || line.includes("CLOSED DEAL");
    if (currentLogFilter === "shield") return line.includes("SHIELD") || line.includes("COOLDOWN") || line.includes("FREEZE") || line.includes("Blocked");
    return true;
  });

  if (filtered.length === 0) {
    terminal.innerHTML = `<div class="log-line text-dim">No log events found for category: ${currentLogFilter.toUpperCase()}</div>`;
    return;
  }

  terminal.innerHTML = filtered.map(line => {
    let cls = "log-line";
    if (line.includes("WIN") || line.includes("🎉")) cls += " log-win";
    else if (line.includes("LOSS") || line.includes("🛑")) cls += " log-loss";
    else if (line.includes("ORDER") || line.includes("🚀")) cls += " log-order";
    else if (line.includes("COOLDOWN") || line.includes("FREEZE") || line.includes("Blocked")) cls += " log-shield";
    else if (line.includes("DYNAMIC MODEL EXIT") || line.includes("🎯")) cls += " log-exit";
    return `<div class="${cls}">${escapeHtml(line)}</div>`;
  }).join("");

  terminal.scrollTop = terminal.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
