#!/usr/bin/env python3
"""
Kalshi AutoTrader Dashboard — port 8888
Real-time paper trading monitor backed by SQLite.
"""

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, render_template_string

import db as _db

app = Flask(__name__)

DATA_DIR    = Path(__file__).parent / "data" / "trading"
PAPER_STATE = DATA_DIR / "paper-trade-state.json"
AUTOTRADER  = Path(__file__).parent / "kalshi-autotrader.py"

# Ensure DB and schema exist at startup
_db.init_db()


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_paper_state() -> dict:
    if PAPER_STATE.exists():
        try:
            return json.loads(PAPER_STATE.read_text())
        except Exception:
            pass
    return {}


def trader_pid() -> int | None:
    r = subprocess.run(["pgrep", "-f", "kalshi-autotrader"], capture_output=True, text=True)
    pids = r.stdout.strip().split()
    return int(pids[0]) if pids else None


def get_config_params() -> dict:
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


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics() -> dict:
    m     = _db.get_metrics()
    state = load_paper_state()

    starting_cents = state.get("starting_balance_cents", 10000)
    current_cents  = state.get("current_balance_cents", starting_cents)

    # Build bankroll series from SQLite cumulative PnL
    bk_raw  = m.get("bankroll_series_raw", [])
    bk_base = starting_cents / 100
    bankroll_series = [
        {"t": t[:19].replace("T", " ") if t else "", "v": round(bk_base + cum_pnl / 100, 2)}
        for t, cum_pnl in bk_raw
    ]

    pid = trader_pid()
    return {
        "total_trades":       m["total_trades"],
        "settled":            m["settled"],
        "pending":            m["pending"],
        "won":                m["won"],
        "lost":               m["lost"],
        "win_rate":           m["win_rate"],
        "yes_win_rate":       m["yes_win_rate"],
        "no_win_rate":        m["no_win_rate"],
        "yes_trades":         m["yes_trades"],
        "no_trades":          m["no_trades"],
        "roi_pct":            m["roi_pct"],
        "net_pnl_usd":        round(m["net_pnl_cents"] / 100, 2),
        "gross_profit_usd":   round(m["gross_profit_cents"] / 100, 2),
        "gross_loss_usd":     round(m["gross_loss_cents"] / 100, 2),
        "current_balance_usd":  round(current_cents / 100, 2),
        "starting_balance_usd": round(starting_cents / 100, 2),
        "bankroll_series":    bankroll_series,
        "by_category":        m["by_category"],
        "avg_edge":           m["avg_edge"],
        "trader_running":     pid is not None,
        "trader_pid":         pid,
        "last_updated":       datetime.now(timezone.utc).isoformat(),
    }


def get_recent_trades(limit: int = 100) -> list:
    rows = _db.get_trades(limit=limit)
    result = []
    for t in rows:
        status = t.get("result_status", "pending")
        price  = t.get("price_cents", 0)
        contr  = t.get("contracts", 0)
        pnl_c  = t.get("pnl_cents")
        pnl    = round(pnl_c / 100, 2) if pnl_c is not None else None
        result.append({
            "ts":        (t.get("timestamp") or "")[:19].replace("T", " "),
            "ticker":    t.get("ticker", ""),
            "title":     (t.get("title") or t.get("ticker", ""))[:60],
            "action":    t.get("action", ""),
            "price":     price,
            "contracts": contr,
            "cost_usd":  round(t.get("cost_cents", 0) / 100, 2),
            "edge":      round((t.get("edge") or 0) * 100, 1),
            "status":    status,
            "pnl_usd":   pnl,
            "settled_at": (t.get("settled_at") or "")[:19].replace("T", " "),
            "category":  t.get("category") or "other",
            "forecast":  round((t.get("forecast_prob") or 0) * 100, 1),
        })
    return result

# ── API endpoints ──────────────────────────────────────────────────────────────

@app.route("/api/metrics")
def api_metrics():
    return jsonify(compute_metrics())

@app.route("/api/trades")
def api_trades():
    return jsonify(get_recent_trades(100))

@app.route("/api/roi_series")
def api_roi_series():
    return jsonify(_db.get_roi_series())

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

@app.route("/api/cycle_series")
def api_cycle_series():
    return jsonify(_db.get_cycle_series())

@app.route("/api/daily_stats")
def api_daily_stats():
    return jsonify(_db.get_daily_stats())

@app.route("/api/hourly_distribution")
def api_hourly_distribution():
    return jsonify(_db.get_hourly_distribution())

@app.route("/api/cycle_hourly")
def api_cycle_hourly():
    return jsonify(_db.get_cycle_hourly_distribution())

# ── HTML Dashboard ─────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>⚡ Kalshi AutoTrader</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/apexcharts@3.45.1/dist/apexcharts.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.13.3/dist/cdn.min.js"></script>
<style>
/* ── CSS Variables ───────────────────────────────────────── */
:root{
  --bg:#030712;--bg2:#0c1120;--surface:#0f1729;
  --card:rgba(15,23,41,0.85);--border:rgba(99,179,237,0.1);--border2:rgba(99,179,237,0.22);
  --cyan:#06b6d4;--purple:#8b5cf6;--green:#10b981;--red:#ef4444;--yellow:#f59e0b;--blue:#3b82f6;
  --text:#e2e8f0;--text2:#94a3b8;--text3:#475569;
  --glow-c:0 0 18px rgba(6,182,212,.45);--glow-g:0 0 18px rgba(16,185,129,.45);--glow-r:0 0 18px rgba(239,68,68,.45);
}
.light{
  --bg:#f0f4f8;--bg2:#e2e8f0;--surface:#fff;--card:rgba(255,255,255,.92);
  --border:rgba(99,179,237,.18);--border2:rgba(99,179,237,.38);
  --text:#1e293b;--text2:#475569;--text3:#94a3b8;
}
/* ── Reset / Base ────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  font-family:'Space Grotesk',system-ui,sans-serif;
  background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;
  transition:background .3s,color .3s;
}
/* grid bg */
body::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(6,182,212,.025) 1px,transparent 1px),
    linear-gradient(90deg,rgba(6,182,212,.025) 1px,transparent 1px);
  background-size:48px 48px;
}
.light body::after{
  background-image:linear-gradient(rgba(6,182,212,.06) 1px,transparent 1px),
    linear-gradient(90deg,rgba(6,182,212,.06) 1px,transparent 1px);
}
/* ── Header ──────────────────────────────────────────────── */
.hdr{
  position:sticky;top:0;z-index:200;
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
  background:rgba(3,7,18,.88);border-bottom:1px solid var(--border2);
  padding:.85rem 1.5rem;
}
.light .hdr{background:rgba(240,244,248,.9)}
.hdr-inner{max-width:1640px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;gap:1rem}
.logo{display:flex;align-items:center;gap:.75rem}
.logo-icon{
  width:38px;height:38px;border-radius:10px;
  background:linear-gradient(135deg,var(--cyan),var(--purple));
  display:flex;align-items:center;justify-content:center;font-size:1.15rem;
  box-shadow:var(--glow-c);flex-shrink:0;
}
.logo-name{
  font-size:1.05rem;font-weight:700;letter-spacing:-.02em;
  background:linear-gradient(90deg,var(--cyan),var(--purple));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.logo-sub{font-size:.6rem;color:var(--text2);text-transform:uppercase;letter-spacing:.15em;margin-top:1px}
.hdr-right{display:flex;align-items:center;gap:.85rem}
.spill{
  display:flex;align-items:center;gap:.45rem;
  padding:.3rem .8rem;border-radius:999px;font-size:.72rem;font-weight:600;letter-spacing:.05em;border:1px solid;
}
.spill.live{background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.4);color:var(--green)}
.spill.off{background:rgba(239,68,68,.1);border-color:rgba(239,68,68,.3);color:var(--red)}
.pdot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);animation:blink 1.6s ease-in-out infinite}
.off .pdot{background:var(--red);box-shadow:0 0 6px var(--red);animation:none}
@keyframes blink{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.8)}}
.tbtn{
  width:36px;height:36px;border-radius:9px;border:1px solid var(--border2);
  background:var(--surface);color:var(--text);cursor:pointer;
  display:flex;align-items:center;justify-content:center;font-size:.95rem;transition:.2s;
}
.tbtn:hover{border-color:var(--cyan);box-shadow:var(--glow-c)}
.lupd{font-size:.67rem;color:var(--text3);font-family:'JetBrains Mono',monospace}
/* ── Layout ──────────────────────────────────────────────── */
.wrap{max-width:1640px;margin:0 auto;padding:1.5rem 1.5rem 3rem;position:relative;z-index:1}
/* ── Card ────────────────────────────────────────────────── */
.card{
  background:var(--card);border:1px solid var(--border);border-radius:16px;
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  transition:border-color .2s,box-shadow .2s;position:relative;overflow:hidden;
}
.card::before{
  content:'';position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(135deg,rgba(6,182,212,.03) 0%,transparent 55%);
}
.card:hover{border-color:var(--border2)}
/* ── Stat Grid ───────────────────────────────────────────── */
.sgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:.9rem;margin:1.25rem 0}
.sc{padding:1.2rem;text-align:center}
.sc.feat{background:linear-gradient(135deg,rgba(6,182,212,.1),rgba(139,92,246,.07));border-color:rgba(6,182,212,.28)}
.sv{font-size:1.9rem;font-weight:700;line-height:1.1;letter-spacing:-.03em;font-variant-numeric:tabular-nums}
.sl{font-size:.64rem;color:var(--text2);text-transform:uppercase;letter-spacing:.12em;margin-top:.4rem}
.ss{font-size:.7rem;color:var(--text3);margin-top:.2rem;font-family:'JetBrains Mono',monospace}
/* colours */
.cyan{color:var(--cyan)}.green{color:var(--green)}.red{color:var(--red)}
.yellow{color:var(--yellow)}.purple{color:var(--purple)}.blue{color:var(--blue)}.muted{color:var(--text2)}
.gg{text-shadow:0 0 14px rgba(16,185,129,.65)}
.gr{text-shadow:0 0 14px rgba(239,68,68,.65)}
.gc{text-shadow:0 0 14px rgba(6,182,212,.65)}
/* ── Section Label ───────────────────────────────────────── */
.sec{display:flex;align-items:center;gap:.75rem;margin:1.5rem 0 .75rem;
  font-size:.63rem;text-transform:uppercase;letter-spacing:.15em;color:var(--text3)}
.sec::after{content:'';flex:1;height:1px;background:var(--border)}
/* ── Chart Grids ─────────────────────────────────────────── */
.cg2{display:grid;grid-template-columns:1fr 1fr;gap:.9rem;margin-bottom:.9rem}
.cg3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.9rem;margin-bottom:.9rem}
@media(max-width:1100px){.cg3{grid-template-columns:1fr 1fr}}
@media(max-width:720px){.cg2,.cg3{grid-template-columns:1fr}.sgrid{grid-template-columns:repeat(2,1fr)}}
.cc{padding:1.15rem}
.ct{font-size:.63rem;text-transform:uppercase;letter-spacing:.12em;color:var(--text2);margin-bottom:.9rem;
  display:flex;align-items:center;gap:.45rem}
.ct::before{content:'';width:3px;height:13px;border-radius:2px;background:linear-gradient(var(--cyan),var(--purple))}
/* ── Config Pills ────────────────────────────────────────── */
.cpills{display:flex;flex-wrap:wrap;gap:.55rem;margin-top:.25rem}
.cpill{display:flex;flex-direction:column;padding:.45rem .8rem;border-radius:9px;
  border:1px solid var(--border2);background:rgba(6,182,212,.05);min-width:115px}
.ck{font-size:.58rem;text-transform:uppercase;letter-spacing:.1em;color:var(--text3);font-family:'JetBrains Mono',monospace}
.cv{font-size:.95rem;font-weight:600;color:var(--cyan);font-family:'JetBrains Mono',monospace;margin-top:.12rem}
/* ── Trades Table ────────────────────────────────────────── */
.twrap{overflow-x:auto}
.tt{width:100%;border-collapse:collapse;font-size:.81rem}
.tt th{
  padding:.65rem .85rem;text-align:left;font-size:.61rem;
  text-transform:uppercase;letter-spacing:.1em;color:var(--text3);
  border-bottom:1px solid var(--border);white-space:nowrap;
  background:rgba(0,0,0,.15);
}
.light .tt th{background:rgba(0,0,0,.04)}
.tt td{padding:.58rem .85rem;border-bottom:1px solid rgba(99,179,237,.05);vertical-align:middle}
.tt tr:hover td{background:rgba(6,182,212,.04)}
.mono{font-family:'JetBrains Mono',monospace;font-size:.73rem}
.bdg{
  display:inline-flex;align-items:center;padding:.18rem .55rem;
  border-radius:5px;font-size:.62rem;font-weight:600;letter-spacing:.05em;border:1px solid;
}
.bw{background:rgba(16,185,129,.12);color:var(--green);border-color:rgba(16,185,129,.3)}
.bl{background:rgba(239,68,68,.12);color:var(--red);border-color:rgba(239,68,68,.3)}
.bp{background:rgba(6,182,212,.12);color:var(--cyan);border-color:rgba(6,182,212,.3)}
.by{background:rgba(16,185,129,.12);color:var(--green);border-color:rgba(16,185,129,.3)}
.bn{background:rgba(239,68,68,.12);color:var(--red);border-color:rgba(239,68,68,.3)}
.pp{color:var(--green);font-weight:600;font-family:'JetBrains Mono',monospace}
.pn{color:var(--red);font-weight:600;font-family:'JetBrains Mono',monospace}
/* scrollbar */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
</style>
</head>
<body x-data="dash()" x-init="init()" :class="dark?'':'light'">

<!-- HEADER -->
<header class="hdr">
  <div class="hdr-inner">
    <div class="logo">
      <div class="logo-icon">⚡</div>
      <div>
        <div class="logo-name">KALSHI AUTOTRADER</div>
        <div class="logo-sub">Paper Trading · AI Powered · Self-Tuning</div>
      </div>
    </div>
    <div class="hdr-right">
      <div :class="'spill ' + (m.trader_running ? 'live' : 'off')">
        <div class="pdot"></div>
        <span x-text="m.trader_running ? 'LIVE' : 'OFFLINE'"></span>
        <span x-show="m.trader_pid" class="mono" x-text="'#'+m.trader_pid" style="opacity:.55;font-size:.62rem"></span>
      </div>
      <div class="lupd" x-text="upd"></div>
      <button class="tbtn" @click="toggleDark()" :title="dark?'Light mode':'Dark mode'">
        <span x-text="dark ? '☀️' : '🌙'"></span>
      </button>
    </div>
  </div>
</header>

<div class="wrap">

  <!-- STAT CARDS -->
  <div class="sgrid">
    <div class="card sc feat">
      <div class="sv cyan" :class="{'gc': m.current_balance_usd > m.starting_balance_usd}" x-text="'$'+(m.current_balance_usd||0).toFixed(2)">—</div>
      <div class="sl">Bankroll</div>
      <div class="ss" x-text="'start $'+(m.starting_balance_usd||0).toFixed(2)"></div>
    </div>
    <div class="card sc">
      <div class="sv" :class="(m.roi_pct||0)>=0?'green gg':'red gr'"
           x-text="((m.roi_pct||0)>=0?'+':'')+(m.roi_pct||0).toFixed(1)+'%'">—</div>
      <div class="sl">ROI</div>
      <div class="ss" x-text="((m.net_pnl_usd||0)>=0?'+':'')+' $'+Math.abs(m.net_pnl_usd||0).toFixed(2)+' P&L'"></div>
    </div>
    <div class="card sc">
      <div class="sv" :class="(m.win_rate||0)>=50?'green':'yellow'"
           x-text="(m.win_rate||0)+'%'">—</div>
      <div class="sl">Win Rate</div>
      <div class="ss" x-text="(m.won||0)+'W / '+(m.lost||0)+'L'"></div>
    </div>
    <div class="card sc">
      <div class="sv yellow" x-text="m.settled||0">—</div>
      <div class="sl">Settled</div>
      <div class="ss" x-text="(m.pending||0)+' open'"></div>
    </div>
    <div class="card sc">
      <div class="sv green" x-text="(m.no_win_rate||0)+'%'">—</div>
      <div class="sl">BUY_NO WR</div>
      <div class="ss" x-text="(m.no_trades||0)+' trades'"></div>
    </div>
    <div class="card sc">
      <div class="sv yellow" x-text="(m.yes_win_rate||0)+'%'">—</div>
      <div class="sl">BUY_YES WR</div>
      <div class="ss" x-text="(m.yes_trades||0)+' trades'"></div>
    </div>
    <div class="card sc">
      <div class="sv" :class="(m.avg_edge||0)>0?'cyan gc':'muted'" x-text="(m.avg_edge||0)+'%'">—</div>
      <div class="sl">Avg Edge</div>
      <div class="ss">per trade</div>
    </div>
    <div class="card sc">
      <div class="sv" style="font-size:1.25rem">
        <span class="green" x-text="'+$'+(m.gross_profit_usd||0).toFixed(2)"></span>
        <span class="muted" style="font-size:.8em"> / </span>
        <span class="red"   x-text="'-$'+(m.gross_loss_usd||0).toFixed(2)"></span>
      </div>
      <div class="sl">Gross P / L</div>
    </div>
  </div>

  <!-- CHARTS ROW 1 -->
  <div class="sec">Performance</div>
  <div class="cg2">
    <div class="card cc"><div class="ct">Bankroll ($)</div><div id="ch-bankroll"></div></div>
    <div class="card cc"><div class="ct">ROI Cumulativo (%)</div><div id="ch-roi"></div></div>
  </div>

  <!-- CHARTS ROW 2 -->
  <div class="cg3">
    <div class="card cc"><div class="ct">Win Rate per Categoria</div><div id="ch-cat"></div></div>
    <div class="card cc"><div class="ct">Distribuzione Edge (%)</div><div id="ch-edge"></div></div>
    <div class="card cc"><div class="ct">Trades per Giorno</div><div id="ch-daily"></div></div>
  </div>

  <!-- TIME SERIES SECTION -->
  <div class="sec">Andamento nel Tempo</div>

  <!-- Balance from cycles (most granular) -->
  <div class="card cc" style="margin-bottom:.9rem">
    <div class="ct">Bankroll per Ciclo (ogni 5 min)</div>
    <div id="ch-cycle-balance"></div>
  </div>

  <!-- Daily P&L + Win Rate -->
  <div class="cg2">
    <div class="card cc"><div class="ct">P&L per Giorno ($)</div><div id="ch-daily-pnl"></div></div>
    <div class="card cc"><div class="ct">Win Rate per Giorno (%)</div><div id="ch-daily-wr"></div></div>
  </div>

  <!-- Hourly distribution -->
  <div class="cg2">
    <div class="card cc"><div class="ct">Attività per Ora del Giorno (cicli)</div><div id="ch-hour-cycles"></div></div>
    <div class="card cc"><div class="ct">Trade Piazzati per Ora</div><div id="ch-hour-trades"></div></div>
  </div>

  <!-- CONFIG PARAMS -->
  <div class="sec">Parametri Algo Attivi</div>
  <div class="card" style="padding:1.15rem;margin-bottom:.9rem">
    <div class="cpills">
      <template x-for="[k,v] in Object.entries(cfg)" :key="k">
        <div class="cpill">
          <div class="ck" x-text="k.replace(/_/g,' ')"></div>
          <div class="cv" x-text="v"></div>
        </div>
      </template>
    </div>
  </div>

  <!-- TRADES TABLE -->
  <div class="sec">Ultimi Trade</div>
  <div class="card" style="padding:1.15rem;margin-bottom:.9rem">
    <div class="twrap">
      <table class="tt">
        <thead><tr>
          <th>Data</th><th>Ticker</th><th>Azione</th><th>Prezzo</th>
          <th>Contratti</th><th>Costo</th><th>Edge</th><th>Forecast</th>
          <th>Stato</th><th>P&L</th><th>Cat</th>
        </tr></thead>
        <tbody>
          <template x-for="t in trades" :key="t.ts+t.ticker">
            <tr>
              <td class="mono" style="color:var(--text3);white-space:nowrap" x-text="t.ts"></td>
              <td class="mono" style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                  x-text="t.ticker.slice(0,26)" :title="t.title"></td>
              <td><span class="bdg" :class="t.action==='BUY_NO'?'bn':'by'" x-text="t.action"></span></td>
              <td class="mono" x-text="t.price+'¢'"></td>
              <td class="mono" x-text="t.contracts"></td>
              <td class="mono" x-text="'$'+t.cost_usd.toFixed(2)"></td>
              <td class="mono" :class="t.edge>2?'green':t.edge>0?'yellow':'red'" x-text="t.edge+'%'"></td>
              <td class="mono" x-text="t.forecast+'%'"></td>
              <td><span class="bdg" :class="t.status==='won'?'bw':t.status==='lost'?'bl':'bp'" x-text="t.status"></span></td>
              <td>
                <span x-show="t.pnl_usd!==null" :class="t.pnl_usd>=0?'pp':'pn'"
                      x-text="(t.pnl_usd>=0?'+':'')+'$'+(t.pnl_usd||0).toFixed(2)"></span>
                <span x-show="t.pnl_usd===null" style="color:var(--text3)">—</span>
              </td>
              <td style="color:var(--text3);font-size:.73rem" x-text="t.category"></td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>
  </div>

</div><!-- /wrap -->

<script>
function dash(){
  return {
    dark: localStorage.getItem('kd')!=='light',
    m:{trader_running:false,current_balance_usd:0,starting_balance_usd:100,
       roi_pct:0,win_rate:0,settled:0,pending:0,won:0,lost:0,
       no_win_rate:0,yes_win_rate:0,no_trades:0,yes_trades:0,
       avg_edge:0,net_pnl_usd:0,gross_profit_usd:0,gross_loss_usd:0,
       bankroll_series:[],by_category:{},trader_pid:null},
    trades:[],roi:[],cfg:{},upd:'—',_ch:{},
    cycles:[],daily:[],hourly:[],cycleHourly:[],

    init(){ this.refresh(); setInterval(()=>this.refresh(),30000); },

    toggleDark(){
      this.dark=!this.dark;
      localStorage.setItem('kd',this.dark?'dark':'light');
      setTimeout(()=>this.charts(),80);
    },

    async refresh(){
      try{
        const [m,roi,tr,cfg,cycles,daily,hourly,cycleHourly]=await Promise.all([
          fetch('/api/metrics').then(r=>r.json()),
          fetch('/api/roi_series').then(r=>r.json()),
          fetch('/api/trades').then(r=>r.json()),
          fetch('/api/config').then(r=>r.json()),
          fetch('/api/cycle_series').then(r=>r.json()),
          fetch('/api/daily_stats').then(r=>r.json()),
          fetch('/api/hourly_distribution').then(r=>r.json()),
          fetch('/api/cycle_hourly').then(r=>r.json()),
        ]);
        this.m=m; this.roi=roi; this.trades=tr; this.cfg=cfg;
        this.cycles=cycles; this.daily=daily;
        this.hourly=hourly; this.cycleHourly=cycleHourly;
        this.upd=new Date().toLocaleTimeString('it-IT');
        this.$nextTick(()=>this.charts());
      }catch(e){console.error(e)}
    },

    base(){
      const d=this.dark;
      return{
        chart:{background:'transparent',fontFamily:"'Space Grotesk',system-ui",
               toolbar:{show:false},animations:{enabled:true,speed:500}},
        theme:{mode:d?'dark':'light'},
        grid:{borderColor:d?'rgba(99,179,237,.07)':'rgba(0,0,0,.07)',strokeDashArray:4},
        tooltip:{theme:d?'dark':'light'},
        xaxis:{labels:{style:{colors:d?'#64748b':'#94a3b8',fontSize:'10px'}},
               axisBorder:{show:false},axisTicks:{show:false}},
        yaxis:{labels:{style:{colors:d?'#64748b':'#94a3b8',fontSize:'10px'}}},
        legend:{labels:{colors:d?'#94a3b8':'#475569'}},
      };
    },

    mk(id,opts){
      if(this._ch[id]){this._ch[id].destroy();}
      const el=document.getElementById(id);
      if(!el)return;
      this._ch[id]=new ApexCharts(el,opts);
      this._ch[id].render();
    },

    charts(){
      const m=this.m,roi=this.roi,tr=this.trades,b=this.base();
      const cycles=this.cycles,daily=this.daily,hourly=this.hourly,ch=this.cycleHourly;

      /* Bankroll */
      const bk=m.bankroll_series&&m.bankroll_series.length?m.bankroll_series:
        roi.map(r=>({t:r.t,v:+(m.starting_balance_usd+r.v/100*m.starting_balance_usd).toFixed(2)}));
      this.mk('ch-bankroll',{...b,
        chart:{...b.chart,type:'area',height:210},
        series:[{name:'Bankroll ($)',data:bk.map(p=>({x:p.t,y:p.v}))}],
        stroke:{curve:'smooth',width:2},
        fill:{type:'gradient',gradient:{shadeIntensity:1,opacityFrom:.42,opacityTo:.03,stops:[0,100]}},
        colors:['#06b6d4'],
        markers:{size:bk.length<25?4:0,colors:['#06b6d4'],strokeWidth:0},
        yaxis:{...b.yaxis,labels:{...b.yaxis.labels,formatter:v=>'$'+v.toFixed(0)}},
        xaxis:{...b.xaxis,type:'category',labels:{...b.xaxis.labels,rotate:-30,maxHeight:55,
               formatter:v=>v?v.slice(5,16):''}},
      });

      /* ROI */
      const lv=roi.length?roi[roi.length-1].v:0;
      const rc=lv>=0?'#10b981':'#ef4444';
      this.mk('ch-roi',{...b,
        chart:{...b.chart,type:'area',height:210},
        series:[{name:'ROI (%)',data:roi.map(p=>({x:p.t,y:p.v}))}],
        stroke:{curve:'smooth',width:2},
        fill:{type:'gradient',gradient:{shadeIntensity:1,opacityFrom:.38,opacityTo:.02,stops:[0,100]}},
        colors:[rc],
        markers:{size:roi.length<25?4:0,colors:[rc],strokeWidth:0},
        yaxis:{...b.yaxis,labels:{...b.yaxis.labels,formatter:v=>v.toFixed(1)+'%'}},
        xaxis:{...b.xaxis,type:'category',labels:{...b.xaxis.labels,rotate:-30,maxHeight:55,
               formatter:v=>v?v.slice(5,16):''}},
        annotations:{yaxis:[{y:0,borderColor:this.dark?'#334155':'#cbd5e1',strokeDashArray:4}]},
      });

      /* Category WR */
      const cats=Object.entries(m.by_category||{}).filter(([,v])=>v.trades>0);
      if(cats.length){
        const wd=cats.map(([,v])=>v.trades>0?Math.round(v.won/v.trades*100):0);
        this.mk('ch-cat',{...b,
          chart:{...b.chart,type:'bar',height:210},
          series:[{name:'Win Rate %',data:wd}],
          xaxis:{...b.xaxis,categories:cats.map(([k])=>k)},
          colors:wd.map(v=>v>=50?'#10b981':'#ef4444'),
          plotOptions:{bar:{borderRadius:6,distributed:true}},
          legend:{show:false},
          yaxis:{...b.yaxis,max:100,labels:{...b.yaxis.labels,formatter:v=>v+'%'}},
        });
      }

      /* Edge Distribution */
      const edges=tr.filter(t=>t.edge!=null).map(t=>t.edge);
      if(edges.length){
        const bkts=Array(10).fill(0);
        edges.forEach(e=>{const i=Math.min(9,Math.floor(e/2));bkts[i]++;});
        this.mk('ch-edge',{...b,
          chart:{...b.chart,type:'bar',height:210},
          series:[{name:'Trades',data:bkts}],
          xaxis:{...b.xaxis,categories:Array(10).fill(0).map((_,i)=>`${i*2}-${i*2+2}%`)},
          colors:['#8b5cf6'],plotOptions:{bar:{borderRadius:4}},legend:{show:false},
        });
      }

      /* Trades per Day */
      const bd={};
      tr.forEach(t=>{const d=(t.ts||'').slice(0,10);if(d)bd[d]=(bd[d]||0)+1;});
      const days=Object.keys(bd).sort().slice(-14);
      if(days.length){
        this.mk('ch-daily',{...b,
          chart:{...b.chart,type:'bar',height:210},
          series:[{name:'Trades',data:days.map(d=>bd[d])}],
          xaxis:{...b.xaxis,categories:days.map(d=>d.slice(5))},
          colors:['#f59e0b'],plotOptions:{bar:{borderRadius:4}},legend:{show:false},
        });
      }

      /* ── TIME SERIES CHARTS ─────────────────────────────── */

      /* Balance from cycles (per-cycle granularity) */
      if(cycles&&cycles.length){
        const N=cycles.length;
        // x-axis: abbreviated time label based on span
        const first=new Date(cycles[0].t.replace(' ','T')+'Z');
        const last=new Date(cycles[N-1].t.replace(' ','T')+'Z');
        const spanH=(last-first)/3600000;
        const fmt=t=>{
          const d=new Date(t.replace(' ','T')+'Z');
          if(spanH<=6) return d.toLocaleTimeString('it-IT',{hour:'2-digit',minute:'2-digit',timeZone:'UTC'});
          if(spanH<=48) return d.toLocaleString('it-IT',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',timeZone:'UTC'});
          return d.toLocaleDateString('it-IT',{month:'2-digit',day:'2-digit',timeZone:'UTC'});
        };
        const cycleLabels=cycles.map(c=>fmt(c.t));
        const balData=cycles.map(c=>({x:fmt(c.t),y:c.b}));
        const tpData=cycles.map(c=>c.tp);
        this.mk('ch-cycle-balance',{...b,
          chart:{...b.chart,type:'area',height:240},
          series:[{name:'Bankroll ($)',data:balData}],
          stroke:{curve:'smooth',width:2},
          fill:{type:'gradient',gradient:{shadeIntensity:1,opacityFrom:.45,opacityTo:.04,stops:[0,100]}},
          colors:['#8b5cf6'],
          markers:{size:N<60?3:0,colors:['#8b5cf6'],strokeWidth:0},
          yaxis:{...b.yaxis,labels:{...b.yaxis.labels,formatter:v=>'$'+v.toFixed(2)}},
          xaxis:{...b.xaxis,type:'category',tickAmount:Math.min(N,16),
                 labels:{...b.xaxis.labels,rotate:-30,maxHeight:60,
                         formatter:(v,i)=>{
                           if(N<=20)return v;
                           const step=Math.floor(N/12)||1;
                           return cycleLabels.indexOf(v)%step===0?v:'';
                         }}},
          tooltip:{y:{formatter:v=>'$'+v.toFixed(2)},x:{show:true}},
          annotations:{yaxis:[{
            y:m.starting_balance_usd,
            borderColor:'#f59e0b',strokeDashArray:5,
            label:{text:'Start $'+m.starting_balance_usd.toFixed(0),
                   style:{background:'rgba(245,158,11,.15)',color:'#f59e0b',fontSize:'10px'}}
          }]},
        });
      }

      /* Daily P&L */
      if(daily&&daily.length){
        this.mk('ch-daily-pnl',{...b,
          chart:{...b.chart,type:'bar',height:220},
          series:[{name:'P&L ($)',data:daily.map(d=>d.pnl_usd)}],
          xaxis:{...b.xaxis,categories:daily.map(d=>d.day.slice(5))},
          colors:daily.map(d=>d.pnl_usd>=0?'#10b981':'#ef4444'),
          plotOptions:{bar:{borderRadius:5,distributed:true,
            colors:{ranges:[{from:-9999,to:-0.001,color:'#ef4444'},{from:0,to:9999,color:'#10b981'}]}}},
          legend:{show:false},
          yaxis:{...b.yaxis,labels:{...b.yaxis.labels,formatter:v=>(v>=0?'+':'')+v.toFixed(2)}},
          annotations:{yaxis:[{y:0,borderColor:this.dark?'#334155':'#cbd5e1',strokeDashArray:3}]},
        });
      }

      /* Daily Win Rate */
      if(daily&&daily.length){
        const dwr=daily.filter(d=>d.won+d.lost>0);
        if(dwr.length){
          this.mk('ch-daily-wr',{...b,
            chart:{...b.chart,type:'line',height:220},
            series:[
              {name:'Win Rate %',data:dwr.map(d=>d.win_rate),type:'line'},
              {name:'Trades',data:dwr.map(d=>d.trades),type:'bar'},
            ],
            xaxis:{...b.xaxis,categories:dwr.map(d=>d.day.slice(5))},
            colors:['#06b6d4','#8b5cf6'],
            stroke:{curve:'smooth',width:[2,0]},
            plotOptions:{bar:{borderRadius:4,columnWidth:'55%'}},
            yaxis:[
              {...b.yaxis,max:100,labels:{...b.yaxis.labels,formatter:v=>v+'%'}},
              {opposite:true,labels:{style:{colors:this.dark?'#64748b':'#94a3b8',fontSize:'10px'},formatter:v=>v+'t'}},
            ],
            annotations:{yaxis:[{y:50,borderColor:'#f59e0b',strokeDashArray:4,
              label:{text:'50%',style:{background:'rgba(245,158,11,.15)',color:'#f59e0b',fontSize:'9px'}}}]},
          });
        }
      }

      /* Hourly cycle distribution */
      if(ch&&ch.length){
        const active=ch.filter(h=>h.cycles>0);
        if(active.length){
          const maxC=Math.max(...ch.map(h=>h.cycles),1);
          this.mk('ch-hour-cycles',{...b,
            chart:{...b.chart,type:'bar',height:220},
            series:[{name:'Cicli',data:ch.map(h=>h.cycles)}],
            xaxis:{...b.xaxis,categories:ch.map(h=>h.hour+':00')},
            colors:ch.map(h=>{const i=h.cycles/maxC;return `rgba(6,182,212,${0.2+i*0.8})`;}),
            plotOptions:{bar:{borderRadius:4,distributed:true}},
            legend:{show:false},
            tooltip:{y:{formatter:v=>v+' cicli'}},
          });
        }
      }

      /* Hourly trade distribution */
      if(hourly&&hourly.length){
        const maxT=Math.max(...hourly.map(h=>h.trades),1);
        const hasAny=hourly.some(h=>h.trades>0);
        if(hasAny){
          this.mk('ch-hour-trades',{...b,
            chart:{...b.chart,type:'bar',height:220},
            series:[{name:'Trade',data:hourly.map(h=>h.trades)}],
            xaxis:{...b.xaxis,categories:hourly.map(h=>h.hour+':00')},
            colors:hourly.map(h=>{const i=h.trades/maxT;return `rgba(139,92,246,${0.2+i*0.8})`;}),
            plotOptions:{bar:{borderRadius:4,distributed:true}},
            legend:{show:false},
            tooltip:{y:{formatter:(v,{dataPointIndex:i})=>{
              const h=hourly[i];
              return v+' trade'+(h.win_rate>0?' | WR '+h.win_rate+'%':'');
            }}},
          });
        }
      }
    },
  };
}
</script>
</body>
</html>"""

@app.route("/")
def dashboard():
    return render_template_string(HTML)

if __name__ == "__main__":
    print("🚀 Kalshi Dashboard → http://localhost:8888")
    app.run(host="0.0.0.0", port=8888, debug=False)
