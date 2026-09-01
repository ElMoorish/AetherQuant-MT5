/**
 * EA AI — Institutional Quantitative Live Terminal Frontend Controller
 * 100% Real MT5 Terminal & Daemon Telemetry
 */

let priceChart = null;
let lastLogCount = 0;

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initChart();
  fetchTelemetry();
  fetchBars();
  fetchLogs();

  // High-frequency telemetry polling every 1.5 seconds
  setInterval(fetchTelemetry, 1500);
  setInterval(fetchLogs, 2000);
  setInterval(fetchBars, 15000);
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
// 2. CHART INITIALIZATION
// ─────────────────────────────────────────────────────────────────────────────
function initChart() {
  const ctx = document.getElementById("priceChart").getContext("2d");
  
  priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "EURUSD H1 Close",
          data: [],
          borderColor: "#00f0ff",
          backgroundColor: "rgba(0, 240, 255, 0.06)",
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: true,
          tension: 0.15,
        },
        {
          label: "Entry Price",
          data: [],
          borderColor: "rgba(255, 255, 255, 0.6)",
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
        },
        {
          label: "Hard Stop Loss",
          data: [],
          borderColor: "rgba(255, 0, 85, 0.8)",
          borderWidth: 1.5,
          borderDash: [3, 3],
          pointRadius: 0,
          fill: false,
        },
        {
          label: "Hard Take Profit",
          data: [],
          borderColor: "rgba(0, 255, 136, 0.8)",
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
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: "index",
          intersect: false,
          backgroundColor: "rgba(13, 18, 29, 0.95)",
          titleColor: "#8e9bb0",
          bodyColor: "#f0f4f8",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1,
          padding: 10,
          bodyFont: { family: "JetBrains Mono" }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.03)" },
          ticks: {
            color: "#4e5d78",
            font: { family: "JetBrains Mono", size: 10 },
            maxTicksLimit: 8
          }
        },
        y: {
          position: "right",
          grid: { color: "rgba(255, 255, 255, 0.03)" },
          ticks: {
            color: "#8e9bb0",
            font: { family: "JetBrains Mono", size: 10 },
            callback: (val) => val.toFixed(5)
          }
        }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. FETCH TELEMETRY & UPDATE DOM
// ─────────────────────────────────────────────────────────────────────────────
async function fetchTelemetry() {
  const startTime = performance.now();
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const latency = Math.round(performance.now() - startTime);

    document.getElementById("latency").textContent = `${latency}ms`;

    // 1. Header & Account
    const acc = data.account || {};
    if (acc.login) {
      document.getElementById("account-id").textContent = `#${acc.login}`;
      document.getElementById("kpi-equity").textContent = `$${Number(acc.equity).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      document.getElementById("kpi-balance-sub").textContent = `Balance: $${Number(acc.balance).toLocaleString("en-US", { minimumFractionDigits: 2 })}`;
      document.getElementById("kpi-margin-level").textContent = `${acc.margin_level > 9999 ? "10,000%" : acc.margin_level + "%"}`;
      document.getElementById("kpi-free-margin").textContent = `Free Margin: $${Number(acc.free_margin).toLocaleString("en-US", { minimumFractionDigits: 2 })}`;
    }

    // 2. Open Positions Table
    const posList = data.open_positions || [];
    const posBody = document.getElementById("positions-body");
    const posCountEl = document.getElementById("pos-count");
    posCountEl.textContent = `${posList.length} Open Position${posList.length === 1 ? "" : "s"}`;

    let totalFloatingPnL = 0;
    let totalPips = 0;

    if (posList.length === 0) {
      posBody.innerHTML = `<tr><td colspan="10" class="text-dim" style="text-align:center; padding: 24px;">No active positions open &bull; Daemon scanning market bars</td></tr>`;
    } else {
      let rowsHtml = "";
      posList.forEach(p => {
        totalFloatingPnL += p.profit;
        totalPips += p.pips;

        const isProfit = p.profit >= 0;
        const pnlClass = isProfit ? "text-success" : "text-danger";
        const typeClass = p.type === "BUY" ? "text-success" : "text-danger";

        rowsHtml += `
          <tr>
            <td class="mono font-bold">#${p.ticket}</td>
            <td><strong>${p.symbol}</strong></td>
            <td><span class="badge ${p.type === 'BUY' ? 'badge-success' : 'badge-live'}">${p.type}</span></td>
            <td class="mono">${p.volume}</td>
            <td class="mono">${p.price_open.toFixed(5)}</td>
            <td class="mono font-bold">${p.price_current.toFixed(5)}</td>
            <td class="mono text-danger">${p.sl.toFixed(5)}</td>
            <td class="mono text-success">${p.tp.toFixed(5)}</td>
            <td class="mono font-bold ${pnlClass}">${p.pips > 0 ? "+" : ""}${p.pips}</td>
            <td class="mono font-bold ${pnlClass}">${p.profit >= 0 ? "+$" : "-$"}${Math.abs(p.profit).toFixed(2)}</td>
          </tr>
        `;
      });
      posBody.innerHTML = rowsHtml;
    }

    const pnlEl = document.getElementById("kpi-floating-pnl");
    pnlEl.textContent = `${totalFloatingPnL >= 0 ? "+$" : "-$"}${Math.abs(totalFloatingPnL).toFixed(2)}`;
    pnlEl.className = `kpi-value ${totalFloatingPnL >= 0 ? "text-success" : "text-danger"}`;
    document.getElementById("kpi-floating-pips").textContent = `${totalPips > 0 ? "+" : ""}${totalPips.toFixed(1)} pips total`;

    // 3. AI Intelligence & Daemon Decision
    const daemon = data.daemon_state || {};
    const dec = daemon.last_decision || {};

    if (dec.signal) {
      const sigBox = document.getElementById("ai-direction-box");
      const sigText = document.getElementById("ai-signal-text");
      sigText.textContent = dec.signal;

      if (dec.signal === "BUY") {
        sigBox.className = "signal-direction-box buy-box";
      } else if (dec.signal === "SELL") {
        sigBox.className = "signal-direction-box";
      }

      document.getElementById("ai-forecast-val").textContent = `${dec.forecast_return > 0 ? "+" : ""}${dec.forecast_return.toFixed(6)}`;
      document.getElementById("ai-sl-pts").textContent = `${dec.sl_points.toFixed(1)} pts (${(dec.sl_points / 10).toFixed(1)} pips)`;
      document.getElementById("ai-tp-pts").textContent = `${dec.tp_points.toFixed(1)} pts (1:2 RR)`;
      document.getElementById("ai-lot-size").textContent = `${dec.lot_size.toFixed(2)} Lots`;
      document.getElementById("kpi-risk-dollar").textContent = `Max Risk: $${(acc.equity * (dec.risk_pct || 0.0025)).toFixed(2)} / trade`;
    }

    // Overlay active positions onto price chart if available
    if (priceChart && posList.length > 0) {
      const active = posList[0];
      const count = priceChart.data.labels.length;
      priceChart.data.datasets[1].data = Array(count).fill(active.price_open);
      priceChart.data.datasets[2].data = Array(count).fill(active.sl);
      priceChart.data.datasets[3].data = Array(count).fill(active.tp);
      priceChart.update("none");
    }

  } catch (err) {
    console.error("Telemetry fetch error:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. FETCH BARS FOR CANDLE/LINE CHART
// ─────────────────────────────────────────────────────────────────────────────
async function fetchBars() {
  try {
    const res = await fetch("/api/bars?count=48");
    const data = await res.json();
    const bars = data.bars || [];

    if (bars.length > 0 && priceChart) {
      const labels = bars.map(b => b.time ? b.time.split(" ")[1]?.slice(0, 5) || b.time : "");
      const closes = bars.map(b => b.close);

      priceChart.data.labels = labels;
      priceChart.data.datasets[0].data = closes;
      priceChart.update();
    }
  } catch (err) {
    console.error("Bars fetch error:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. FETCH & STREAM TERMINAL LOGS
// ─────────────────────────────────────────────────────────────────────────────
async function fetchLogs() {
  try {
    const res = await fetch("/api/logs?lines=30");
    const data = await res.json();
    const logs = data.logs || [];
    const logBox = document.getElementById("terminal-logs");

    if (logs.length > 0) {
      let html = "";
      logs.forEach(line => {
        let cls = "log-line";
        if (line.includes("[INFO]")) cls += " log-info";
        if (line.includes("ORDER") || line.includes("LIVE") || line.includes("BUY") || line.includes("SELL")) cls += " log-trade";
        if (line.includes("WARNING")) cls += " log-warn";
        if (line.includes("ERROR")) cls += " log-err";

        html += `<div class="${cls}">${escapeHtml(line)}</div>`;
      });
      logBox.innerHTML = html;
      logBox.scrollTop = logBox.scrollHeight;
    }
  } catch (err) {
    console.error("Logs fetch error:", err);
  }
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
