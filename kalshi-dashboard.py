#!/usr/bin/env python3
"""
Kalshi AutoTrader Dashboard — port 8888
Real-time paper trading monitor with charts and API for feedback loop.
"""

import json
import os
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

DATA_DIR = Path(__file__).parent / "data" / "trading"
TRADE_LOG   = DATA_DIR / "kalshi-unified-trades.jsonl"
CYCLE_LOG   = DATA_DIR / "kalshi-unified-cycles.jsonl"
PAPER_STATE = DATA_DIR / "paper-trade-state.json"
AUTOTRADER  = Path(__file__).parent / "kalshi-autotrader.py"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_trades():
    trades = []
    if not TRADE_LOG.exists():
        return trades
    for line in TRADE_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            trades.append(json.loads(line))
        except Exception:
            pass
    return trades

def load_cycles():
    cycles = []
    if not CYCLE_LOG.exists():
        return cycles
    for line in CYCLE_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cycles.append(json.loads(line))
        except Exception:
            pass
    return cycles

def load_paper_state():
    if PAPER_STATE.exists():
        try:
            return json.loads(PAPER_STATE.read_text())
        except Exception:
            pass
    return {}

def trader_pid():
    import subprocess
    r = subprocess.run(["pgrep", "-f", "kalshi-autotrader"], capture_output=True, text=True)
    pids = r.stdout.strip().split()
    return int(pids[0]) if pids else None

# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics():
    trades = load_trades()
    state  = load_paper_state()

    buys = [t for t in trades if t.get("action") in ("BUY_YES", "BUY_NO")]
    settled = [t for t in buys if t.get("result_status") in ("won", "lost")]
    won = [t for t in settled if t.get("result_status") == "won"]
    lost = [t for t in settled if t.get("result_status") == "lost"]
    pending = [t for t in buys if t.get("result_status") not in ("won", "lost")]

    total_cost = sum(t.get("cost_cents", 0) for t in settled)
    gross_profit = sum(
        t.get("contracts", 0) * (100 - t.get("price_cents", 50))
        for t in won
    )
    gross_loss = sum(t.get("cost_cents", 0) for t in lost)
    net_pnl = gross_profit - gross_loss
    roi_pct = (net_pnl / total_cost * 100) if total_cost > 0 else 0

    buy_yes = [t for t in settled if t.get("action") == "BUY_YES"]
    buy_no  = [t for t in settled if t.get("action") == "BUY_NO"]
    yes_wr  = len([t for t in buy_yes if t["result_status"] == "won"]) / len(buy_yes) * 100 if buy_yes else 0
    no_wr   = len([t for t in buy_no  if t["result_status"] == "won"]) / len(buy_no)  * 100 if buy_no  else 0

    # By category
    by_cat = defaultdict(lambda: {"trades": 0, "won": 0, "pnl": 0})
    for t in settled:
        cat = t.get("category") or _infer_category(t.get("ticker", ""))
        by_cat[cat]["trades"] += 1
        if t["result_status"] == "won":
            by_cat[cat]["won"] += 1
            by_cat[cat]["pnl"] += t.get("contracts", 0) * (100 - t.get("price_cents", 50))
        else:
            by_cat[cat]["pnl"] -= t.get("cost_cents", 0)

    # Bankroll timeline from paper state trade_history
    history = state.get("trade_history", [])
    starting = state.get("starting_balance_cents", 10000)
    bankroll_series = []
    running = starting
    for h in sorted(history, key=lambda x: x.get("settled_at", x.get("opened_at", ""))):
        running += h.get("pnl_cents", 0)
        bankroll_series.append({
            "t": h.get("settled_at") or h.get("opened_at"),
            "v": running / 100
        })

    # If no history in state, rebuild from trades log
    if not bankroll_series and settled:
        running = starting / 100
        for t in sorted(settled, key=lambda x: x.get("settled_at") or x.get("timestamp")):
            if t["result_status"] == "won":
                pnl = t.get("contracts", 0) * (100 - t.get("price_cents", 50)) / 100
            else:
                pnl = -t.get("cost_cents", 0) / 100
            running += pnl
            bankroll_series.append({
                "t": t.get("settled_at") or t.get("timestamp"),
                "v": round(running, 2)
            })

    current_balance = state.get("current_balance_cents", starting) / 100

    return {
        "total_trades": len(buys),
        "settled": len(settled),
        "pending": len(pending),
        "won": len(won),
        "lost": len(lost),
        "win_rate": round(len(won) / len(settled) * 100, 1) if settled else 0,
        "yes_win_rate": round(yes_wr, 1),
        "no_win_rate": round(no_wr, 1),
        "yes_trades": len(buy_yes),
        "no_trades": len(buy_no),
        "roi_pct": round(roi_pct, 2),
        "net_pnl_usd": round(net_pnl / 100, 2),
        "gross_profit_usd": round(gross_profit / 100, 2),
        "gross_loss_usd": round(gross_loss / 100, 2),
        "current_balance_usd": round(current_balance, 2),
        "starting_balance_usd": round(starting / 100, 2),
        "bankroll_series": bankroll_series,
        "by_category": dict(by_cat),
        "avg_edge": round(sum(t.get("edge", 0) for t in settled) / len(settled) * 100, 2) if settled else 0,
        "trader_running": trader_pid() is not None,
        "trader_pid": trader_pid(),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

def _infer_category(ticker):
    t = ticker.upper()
    if any(x in t for x in ("BTC", "ETH", "SOL", "CRYPTO")):
        return "crypto"
    if any(x in t for x in ("NBA", "NFL", "MLB", "NHL", "NCAA", "SOCCER", "CBB")):
        return "sports"
    if any(x in t for x in ("WEATHER", "TEMP", "SNOW", "RAIN")):
        return "weather"
    return "other"

def get_recent_trades(limit=50):
    trades = load_trades()
    buys = [t for t in trades if t.get("action") in ("BUY_YES", "BUY_NO")]
    buys.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    result = []
    for t in buys[:limit]:
        status = t.get("result_status", "pending")
        price  = t.get("price_cents", 0)
        contr  = t.get("contracts", 0)
        if status == "won":
            pnl = contr * (100 - price) / 100
        elif status == "lost":
            pnl = -t.get("cost_cents", 0) / 100
        else:
            pnl = None
        result.append({
            "ts":        t.get("timestamp", "")[:19].replace("T", " "),
            "ticker":    t.get("ticker", ""),
            "title":     (t.get("title") or t.get("ticker", ""))[:60],
            "action":    t.get("action", ""),
            "price":     price,
            "contracts": contr,
            "cost_usd":  round(t.get("cost_cents", 0) / 100, 2),
            "edge":      round(t.get("edge", 0) * 100, 1),
            "status":    status,
            "pnl_usd":   round(pnl, 2) if pnl is not None else None,
            "settled_at": (t.get("settled_at") or "")[:19].replace("T", " "),
            "category":  t.get("category") or _infer_category(t.get("ticker", "")),
            "forecast":  round(t.get("forecast_prob", 0) * 100, 1),
        })
    return result

def get_roi_series():
    """ROI% cumulative over time for chart."""
    trades = load_trades()
    settled = [t for t in trades if t.get("action") in ("BUY_YES","BUY_NO")
               and t.get("result_status") in ("won","lost")]
    settled.sort(key=lambda x: x.get("settled_at") or x.get("timestamp"))
    total_cost = 0
    total_pnl  = 0
    series = []
    for i, t in enumerate(settled):
        cost = t.get("cost_cents", 0)
        total_cost += cost
        if t["result_status"] == "won":
            pnl = t.get("contracts", 0) * (100 - t.get("price_cents", 50))
        else:
            pnl = -cost
        total_pnl += pnl
        roi = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        series.append({
            "t": (t.get("settled_at") or t.get("timestamp", ""))[:19].replace("T"," "),
            "v": round(roi, 2),
            "n": i + 1,
        })
    return series

def get_config_params():
    """Read key params from kalshi-autotrader.py for display."""
    params = {}
    if not AUTOTRADER.exists():
        return params
    src = AUTOTRADER.read_text()
    for key in ["MIN_EDGE_BUY_NO", "MIN_EDGE_BUY_YES", "KELLY_FRACTION",
                "MAX_BET_CENTS", "CALIBRATION_FACTOR", "MAX_POSITIONS",
                "MAX_PRICE_CENTS", "VIRTUAL_BALANCE"]:
        m = re.search(rf'^{key}\s*=\s*([^\s#\n]+)', src, re.MULTILINE)
        if m:
            params[key] = m.group(1)
    return params

# ── API endpoints ──────────────────────────────────────────────────────────────

@app.route("/api/metrics")
def api_metrics():
    return jsonify(compute_metrics())

@app.route("/api/trades")
def api_trades():
    return jsonify(get_recent_trades(100))

@app.route("/api/roi_series")
def api_roi_series():
    return jsonify(get_roi_series())

@app.route("/api/status")
def api_status():
    pid = trader_pid()
    m   = compute_metrics()
    return jsonify({
        "trader_running": pid is not None,
        "trader_pid": pid,
        "current_balance_usd": m["current_balance_usd"],
        "roi_pct": m["roi_pct"],
        "win_rate": m["win_rate"],
        "settled_trades": m["settled"],
        "pending_trades": m["pending"],
        "last_updated": m["last_updated"],
        "config": get_config_params(),
    })

@app.route("/api/config")
def api_config():
    return jsonify(get_config_params())

# ── HTML Dashboard ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kalshi AutoTrader Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --green: #3fb950; --red: #f85149; --blue: #58a6ff;
    --yellow: #e3b341; --text: #c9d1d9; --muted: #8b949e;
  }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; }
  .stat-card { text-align: center; padding: 1.2rem; }
  .stat-val  { font-size: 2rem; font-weight: 700; line-height: 1.1; }
  .stat-lbl  { font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-top: .3rem; }
  .green { color: var(--green); } .red { color: var(--red); }
  .blue  { color: var(--blue);  } .yellow { color: var(--yellow); }
  .badge-yes { background: rgba(63,185,80,.2); color: var(--green); border:1px solid var(--green); }
  .badge-no  { background: rgba(248,81,73,.2);  color: var(--red);   border:1px solid var(--red);  }
  .badge-won { background: rgba(63,185,80,.15); color: var(--green); }
  .badge-lost{ background: rgba(248,81,73,.15); color: var(--red);   }
  .badge-pending { background: rgba(88,166,255,.15); color: var(--blue); }
  .table { color: var(--text); } .table td,.table th { border-color: var(--border); font-size:.85rem; }
  .table thead th { color: var(--muted); font-weight:500; }
  canvas { max-height: 260px; }
  .status-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:6px; }
  .dot-green { background:var(--green); box-shadow:0 0 6px var(--green); animation: pulse 2s infinite; }
  .dot-red   { background:var(--red); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .navbar { background: var(--card) !important; border-bottom: 1px solid var(--border); }
  .section-title { font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin-bottom:.8rem; }
  .pnl-pos { color: var(--green); font-weight:600; }
  .pnl-neg { color: var(--red);   font-weight:600; }
  #refresh-badge { font-size:.7rem; }
</style>
</head>
<body>

<nav class="navbar navbar-dark px-3 py-2 mb-4">
  <span class="navbar-brand fw-bold">⚡ Kalshi AutoTrader</span>
  <div class="d-flex align-items-center gap-3">
    <span id="trader-status" class="small"></span>
    <span id="refresh-badge" class="badge bg-secondary">auto-refresh 30s</span>
    <span id="last-update" class="text-muted" style="font-size:.75rem"></span>
  </div>
</nav>

<div class="container-fluid px-4">

  <!-- Stat cards -->
  <div class="row g-3 mb-4" id="stat-cards">
    <div class="col-6 col-md-3 col-xl-2">
      <div class="card stat-card">
        <div class="stat-val blue" id="s-balance">—</div>
        <div class="stat-lbl">Bankroll</div>
      </div>
    </div>
    <div class="col-6 col-md-3 col-xl-2">
      <div class="card stat-card">
        <div class="stat-val" id="s-roi">—</div>
        <div class="stat-lbl">ROI %</div>
      </div>
    </div>
    <div class="col-6 col-md-3 col-xl-2">
      <div class="card stat-card">
        <div class="stat-val" id="s-wr">—</div>
        <div class="stat-lbl">Win Rate</div>
      </div>
    </div>
    <div class="col-6 col-md-3 col-xl-2">
      <div class="card stat-card">
        <div class="stat-val" id="s-pnl">—</div>
        <div class="stat-lbl">Net P&L</div>
      </div>
    </div>
    <div class="col-6 col-md-3 col-xl-2">
      <div class="card stat-card">
        <div class="stat-val yellow" id="s-trades">—</div>
        <div class="stat-lbl">Trades Settled</div>
      </div>
    </div>
    <div class="col-6 col-md-3 col-xl-2">
      <div class="card stat-card">
        <div class="stat-val" id="s-pending">—</div>
        <div class="stat-lbl">Open Positions</div>
      </div>
    </div>
  </div>

  <!-- BUY_YES vs BUY_NO breakdown -->
  <div class="row g-3 mb-4">
    <div class="col-md-6 col-xl-3">
      <div class="card stat-card">
        <div class="stat-val green" id="s-no-wr">—</div>
        <div class="stat-lbl">BUY_NO Win Rate (<span id="s-no-count">0</span> trades)</div>
      </div>
    </div>
    <div class="col-md-6 col-xl-3">
      <div class="card stat-card">
        <div class="stat-val yellow" id="s-yes-wr">—</div>
        <div class="stat-lbl">BUY_YES Win Rate (<span id="s-yes-count">0</span> trades)</div>
      </div>
    </div>
    <div class="col-md-6 col-xl-3">
      <div class="card stat-card">
        <div class="stat-val muted" id="s-avg-edge">—</div>
        <div class="stat-lbl">Avg Edge</div>
      </div>
    </div>
    <div class="col-md-6 col-xl-3">
      <div class="card stat-card">
        <div class="stat-val" id="s-gross">—</div>
        <div class="stat-lbl">Gross P / L</div>
      </div>
    </div>
  </div>

  <!-- Charts row 1 -->
  <div class="row g-3 mb-4">
    <div class="col-lg-6">
      <div class="card p-3">
        <div class="section-title">Bankroll ($)</div>
        <canvas id="chart-bankroll"></canvas>
      </div>
    </div>
    <div class="col-lg-6">
      <div class="card p-3">
        <div class="section-title">ROI Cumulativo (%)</div>
        <canvas id="chart-roi"></canvas>
      </div>
    </div>
  </div>

  <!-- Charts row 2 -->
  <div class="row g-3 mb-4">
    <div class="col-lg-4">
      <div class="card p-3">
        <div class="section-title">Win Rate per Categoria</div>
        <canvas id="chart-category"></canvas>
      </div>
    </div>
    <div class="col-lg-4">
      <div class="card p-3">
        <div class="section-title">Distribuzione Edge (%)</div>
        <canvas id="chart-edge"></canvas>
      </div>
    </div>
    <div class="col-lg-4">
      <div class="card p-3">
        <div class="section-title">Trades per Giorno</div>
        <canvas id="chart-daily"></canvas>
      </div>
    </div>
  </div>

  <!-- Config params -->
  <div class="row g-3 mb-4">
    <div class="col-12">
      <div class="card p-3">
        <div class="section-title">Parametri Algo Attivi</div>
        <div class="row g-2" id="config-params"></div>
      </div>
    </div>
  </div>

  <!-- Trades table -->
  <div class="card p-3 mb-4">
    <div class="section-title">Ultimi Trade</div>
    <div class="table-responsive">
      <table class="table table-hover mb-0">
        <thead>
          <tr>
            <th>Data</th><th>Ticker</th><th>Azione</th><th>Prezzo</th>
            <th>Contratti</th><th>Costo</th><th>Edge</th><th>Forecast</th>
            <th>Stato</th><th>P&L</th><th>Categoria</th>
          </tr>
        </thead>
        <tbody id="trades-tbody"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
const chartDefaults = {
  responsive: true, maintainAspectRatio: true,
  plugins: { legend: { labels: { color: '#c9d1d9', font: { size: 11 } } } },
  scales: {
    x: { ticks: { color: '#8b949e', font:{size:10} }, grid: { color: '#21262d' } },
    y: { ticks: { color: '#8b949e', font:{size:10} }, grid: { color: '#21262d' } }
  }
};

let charts = {};
function mkChart(id, cfg) {
  const ctx = document.getElementById(id).getContext('2d');
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(ctx, cfg);
}

function fmtUSD(v) { return (v>=0?'+':'')+v.toFixed(2)+'$'; }
function fmtPct(v) { return (v>=0?'+':'')+v.toFixed(1)+'%'; }

async function refresh() {
  const [m, roi, trades, cfg] = await Promise.all([
    fetch('/api/metrics').then(r=>r.json()),
    fetch('/api/roi_series').then(r=>r.json()),
    fetch('/api/trades').then(r=>r.json()),
    fetch('/api/config').then(r=>r.json()),
  ]);

  // Trader status
  const dot = m.trader_running
    ? '<span class="status-dot dot-green"></span><span class="green">TRADING</span>'
    : '<span class="status-dot dot-red"></span><span class="red">OFFLINE</span>';
  document.getElementById('trader-status').innerHTML = dot + (m.trader_pid ? ` <small class="text-muted">(PID ${m.trader_pid})</small>` : '');
  document.getElementById('last-update').textContent = 'aggiornato ' + new Date().toLocaleTimeString('it-IT');

  // Stat cards
  const roiEl = document.getElementById('s-roi');
  roiEl.textContent = fmtPct(m.roi_pct);
  roiEl.className = 'stat-val ' + (m.roi_pct >= 0 ? 'green' : 'red');

  const pnlEl = document.getElementById('s-pnl');
  pnlEl.textContent = fmtUSD(m.net_pnl_usd);
  pnlEl.className = 'stat-val ' + (m.net_pnl_usd >= 0 ? 'green' : 'red');

  document.getElementById('s-balance').textContent = '$' + m.current_balance_usd.toFixed(2);
  document.getElementById('s-wr').textContent = m.win_rate + '%';
  document.getElementById('s-wr').className = 'stat-val ' + (m.win_rate >= 50 ? 'green' : 'yellow');
  document.getElementById('s-trades').textContent = m.settled;
  document.getElementById('s-pending').textContent = m.pending;
  document.getElementById('s-no-wr').textContent = m.no_win_rate + '%';
  document.getElementById('s-yes-wr').textContent = m.yes_win_rate + '%';
  document.getElementById('s-no-count').textContent = m.no_trades;
  document.getElementById('s-yes-count').textContent = m.yes_trades;
  document.getElementById('s-avg-edge').textContent = m.avg_edge + '%';
  document.getElementById('s-avg-edge').className = 'stat-val ' + (m.avg_edge > 0 ? 'blue' : 'muted');
  document.getElementById('s-gross').innerHTML =
    `<span class="green">+${m.gross_profit_usd.toFixed(2)}$</span> / <span class="red">-${m.gross_loss_usd.toFixed(2)}$</span>`;

  // --- Chart: Bankroll ---
  const bkSeries = m.bankroll_series.length ? m.bankroll_series : roi.map((r,i)=>({t:r.t, v: m.starting_balance_usd + (r.v/100*m.starting_balance_usd)}));
  if (bkSeries.length) {
    mkChart('chart-bankroll', {
      type: 'line',
      data: {
        labels: bkSeries.map(p=>p.t),
        datasets: [{
          label: 'Bankroll ($)', data: bkSeries.map(p=>p.v),
          borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,.1)',
          fill: true, tension: 0.3, pointRadius: bkSeries.length<30?4:1,
        }]
      },
      options: { ...chartDefaults,
        plugins: { ...chartDefaults.plugins, legend:{display:false} },
        scales: {
          x: { ...chartDefaults.scales.x, ticks:{...chartDefaults.scales.x.ticks, maxTicksLimit:8} },
          y: { ...chartDefaults.scales.y, ticks:{...chartDefaults.scales.y.ticks, callback:v=>'$'+v.toFixed(0)} }
        }
      }
    });
  } else {
    document.getElementById('chart-bankroll').parentElement.innerHTML += '<p class="text-muted text-center mt-3" style="font-size:.8rem">Nessun dato ancora — aspetta che i trade si settilino</p>';
  }

  // --- Chart: ROI ---
  if (roi.length) {
    const roiColor = roi[roi.length-1].v >= 0 ? '#3fb950' : '#f85149';
    mkChart('chart-roi', {
      type: 'line',
      data: {
        labels: roi.map(p=>p.t),
        datasets: [{
          label: 'ROI %', data: roi.map(p=>p.v),
          borderColor: roiColor, backgroundColor: roiColor+'22',
          fill: true, tension: 0.3, pointRadius: roi.length<30?4:1,
        }]
      },
      options: { ...chartDefaults,
        plugins: { ...chartDefaults.plugins, legend:{display:false} },
        scales: {
          x: { ...chartDefaults.scales.x, ticks:{...chartDefaults.scales.x.ticks, maxTicksLimit:8} },
          y: { ...chartDefaults.scales.y, ticks:{...chartDefaults.scales.y.ticks, callback:v=>v+'%'} }
        }
      }
    });
  }

  // --- Chart: Category WR ---
  const cats = Object.entries(m.by_category).filter(([,v])=>v.trades>0);
  if (cats.length) {
    mkChart('chart-category', {
      type: 'bar',
      data: {
        labels: cats.map(([k])=>k),
        datasets: [
          { label: 'Win Rate %', data: cats.map(([,v])=>v.trades>0?Math.round(v.won/v.trades*100):0),
            backgroundColor: cats.map(([,v])=>{const wr=v.trades>0?v.won/v.trades:0; return wr>=0.5?'rgba(63,185,80,.7)':'rgba(248,81,73,.7)'}) },
          { label: 'P&L ($)', data: cats.map(([,v])=>+(v.pnl/100).toFixed(2)),
            backgroundColor: cats.map(([,v])=>v.pnl>=0?'rgba(88,166,255,.4)':'rgba(248,81,73,.4)'),
            yAxisID: 'y2' }
        ]
      },
      options: { ...chartDefaults,
        scales: {
          x: chartDefaults.scales.x,
          y:  { ...chartDefaults.scales.y, title:{display:true,text:'WR%',color:'#8b949e',font:{size:10}} },
          y2: { ...chartDefaults.scales.y, position:'right', title:{display:true,text:'P&L$',color:'#8b949e',font:{size:10}},
                grid:{drawOnChartArea:false} }
        }
      }
    });
  }

  // --- Chart: Edge distribution ---
  const edges = trades.filter(t=>t.edge!=null).map(t=>t.edge);
  if (edges.length) {
    const buckets = Array(10).fill(0);
    edges.forEach(e=>{const i=Math.min(9,Math.floor(e/2)); buckets[i]++;});
    mkChart('chart-edge', {
      type: 'bar',
      data: {
        labels: Array(10).fill(0).map((_,i)=>`${i*2}-${i*2+2}%`),
        datasets: [{ label:'Trades', data:buckets, backgroundColor:'rgba(88,166,255,.6)' }]
      },
      options: { ...chartDefaults,
        plugins:{...chartDefaults.plugins, legend:{display:false}} }
    });
  }

  // --- Chart: Trades per day ---
  const byDay = {};
  trades.forEach(t=>{
    const d = (t.ts||'').slice(0,10);
    if (d) byDay[d] = (byDay[d]||0)+1;
  });
  const days = Object.keys(byDay).sort().slice(-14);
  if (days.length) {
    mkChart('chart-daily', {
      type: 'bar',
      data: {
        labels: days,
        datasets:[{ label:'Trades/giorno', data:days.map(d=>byDay[d]),
          backgroundColor:'rgba(227,179,65,.6)' }]
      },
      options:{ ...chartDefaults,
        plugins:{...chartDefaults.plugins,legend:{display:false}} }
    });
  }

  // --- Config params ---
  const cfgEl = document.getElementById('config-params');
  cfgEl.innerHTML = Object.entries(cfg).map(([k,v])=>`
    <div class="col-6 col-md-4 col-xl-2">
      <div style="background:#0d1117;border:1px solid var(--border);border-radius:8px;padding:.5rem .75rem">
        <div style="font-size:.65rem;color:var(--muted);text-transform:uppercase">${k.replace(/_/g,' ')}</div>
        <div style="font-size:1rem;font-weight:600;color:var(--blue)">${v}</div>
      </div>
    </div>`).join('');

  // --- Trades table ---
  const tbody = document.getElementById('trades-tbody');
  tbody.innerHTML = trades.map(t=>{
    const statusClass = t.status==='won'?'badge-won':t.status==='lost'?'badge-lost':'badge-pending';
    const actionClass = t.action==='BUY_NO'?'badge-no':'badge-yes';
    const pnlHtml = t.pnl_usd!=null
      ? `<span class="${t.pnl_usd>=0?'pnl-pos':'pnl-neg'}">${fmtUSD(t.pnl_usd)}</span>`
      : '<span class="text-muted">—</span>';
    return `<tr>
      <td style="white-space:nowrap;color:var(--muted)">${t.ts}</td>
      <td style="font-size:.75rem;font-family:monospace">${t.ticker.slice(0,28)}</td>
      <td><span class="badge ${actionClass}" style="font-size:.7rem">${t.action}</span></td>
      <td>${t.price}¢</td>
      <td>${t.contracts}</td>
      <td>$${t.cost_usd.toFixed(2)}</td>
      <td>${t.edge}%</td>
      <td>${t.forecast}%</td>
      <td><span class="badge ${statusClass}" style="font-size:.7rem">${t.status}</span></td>
      <td>${pnlHtml}</td>
      <td><span style="font-size:.75rem;color:var(--muted)">${t.category}</span></td>
    </tr>`;
  }).join('');
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""

@app.route("/")
def dashboard():
    return render_template_string(HTML)

if __name__ == "__main__":
    print("🚀 Kalshi Dashboard → http://localhost:8889")
    app.run(host="0.0.0.0", port=8889, debug=False)
