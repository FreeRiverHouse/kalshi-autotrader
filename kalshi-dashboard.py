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
from flask import Flask, jsonify, render_template_string, request

import db as _db

app = Flask(__name__)

DATA_DIR    = Path(__file__).parent / "data" / "trading"
_SSD_DATA   = Path("/Volumes/DATI-SSD/kalshi-logs")
PAPER_STATE = (_SSD_DATA / "paper-trade-state.json") if _SSD_DATA.exists() else DATA_DIR / "paper-trade-state.json"
WATCHDOG_STATE = (_SSD_DATA / "watchdog-last.json") if _SSD_DATA.exists() else DATA_DIR / "watchdog-last.json"
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

def load_watchdog_state() -> dict:
    if WATCHDOG_STATE.exists():
        try:
            return json.loads(WATCHDOG_STATE.read_text())
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
        "watchdog":           load_watchdog_state(),
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

@app.route("/api/calibration")
def api_calibration():
    return jsonify(_db.get_calibration_data())

@app.route("/api/streaks")
def api_streaks():
    return jsonify(_db.get_streak_analysis())

@app.route("/api/edge_distribution")
def api_edge_distribution():
    return jsonify(_db.get_edge_distribution())

@app.route("/api/risk_metrics")
def api_risk_metrics():
    return jsonify(_db.get_risk_metrics())

@app.route("/api/all")
def api_all():
    """Single endpoint for dashboard: returns all data in one request."""
    m   = compute_metrics()
    ps  = load_paper_state()
    wd  = load_watchdog_state()
    pid = trader_pid()
    return jsonify({
        "metrics":         m,
        "trades":          get_recent_trades(100),
        "roi_series":      _db.get_roi_series(),
        "cycle_series":    _db.get_cycle_series(500),
        "daily_stats":     _db.get_daily_stats(),
        "calibration":     _db.get_calibration_data(),
        "streaks":         _db.get_streak_analysis(),
        "edge_dist":       _db.get_edge_distribution(),
        "risk":            _db.get_risk_metrics(),
        "config":          get_config_params(),
        "paper_state": {
            "current_balance_usd":  round((ps.get("current_balance_cents") or 0) / 100, 2),
            "starting_balance_usd": round((ps.get("starting_balance_cents") or 10000) / 100, 2),
            "open_positions":       len(ps.get("positions") or []),
        },
        "trader_running": pid is not None,
        "trader_pid":     pid,
        "watchdog":       wd,
        "last_updated":   datetime.now(timezone.utc).isoformat(),
    })

@app.route("/api/start_trader", methods=["POST"])
def api_start_trader():
    """Start the trader via launchctl (uses existing KeepAlive LaunchAgent)."""
    pid = trader_pid()
    if pid is not None:
        return jsonify({"ok": False, "reason": "already_running", "pid": pid})
    try:
        r = subprocess.run(
            ["launchctl", "start", "com.frh.kalshi-autotrader"],
            capture_output=True, text=True, timeout=10
        )
        import time; time.sleep(2)
        new_pid = trader_pid()
        if new_pid:
            return jsonify({"ok": True, "pid": new_pid})
        return jsonify({"ok": False, "reason": r.stderr.strip() or "not started after 2s"})
    except Exception as e:
        return jsonify({"ok": False, "reason": str(e)}), 500


@app.route("/api/reset_paper", methods=["POST"])
def api_reset_paper():
    """Reset ONLY paper-trade-state.json (balance + positions). SQLite data is NEVER touched."""
    data = request.get_json(silent=True) or {}
    if not data.get("confirm"):
        return jsonify({"ok": False, "reason": "missing confirm=true"}), 400

    fresh = {
        "starting_balance_cents": 10000,
        "current_balance_cents":  10000,
        "positions": [],
        "stats": {"wins": 0, "losses": 0, "total_pnl_cents": 0},
        "trade_history": [],
        "reset_at": datetime.now(timezone.utc).isoformat(),
        "reset_reason": data.get("reason", "manual reset from dashboard"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    PAPER_STATE.write_text(json.dumps(fresh, indent=2))
    return jsonify({"ok": True, "new_balance_usd": 100.0})


# CORS for iframe embedding
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

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
/* ── Start Trader Button ─────────────────────────────────── */
.start-btn{
  display:flex;align-items:center;gap:.45rem;
  padding:.35rem .9rem;border-radius:8px;border:1px solid rgba(16,185,129,.45);
  background:rgba(16,185,129,.12);color:var(--green);cursor:pointer;font-size:.72rem;
  font-weight:700;letter-spacing:.06em;transition:.2s;font-family:inherit;
}
.start-btn:hover{background:rgba(16,185,129,.22);box-shadow:0 0 12px rgba(16,185,129,.3)}
.start-btn:disabled{opacity:.5;cursor:not-allowed}
.last-act{font-size:.65rem;color:var(--text3);font-family:'JetBrains Mono',monospace}
/* ── Recent Bets Live Feed ───────────────────────────────── */
.live-feed{display:flex;flex-direction:column;gap:.4rem;max-height:360px;overflow-y:auto}
.bet-row{
  display:flex;align-items:center;gap:.75rem;padding:.6rem .9rem;border-radius:10px;
  border:1px solid var(--border);background:var(--card);font-size:.78rem;
  transition:border-color .2s;
}
.bet-row:hover{border-color:var(--border2)}
.bet-row.won{border-left:3px solid var(--green)}
.bet-row.lost{border-left:3px solid var(--red)}
.bet-row.pending{border-left:3px solid var(--cyan)}
.bet-time{font-size:.65rem;color:var(--text3);font-family:'JetBrains Mono',monospace;white-space:nowrap;min-width:72px}
.bet-ticker{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--text2);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;min-width:0}
.bet-pnl{font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:700;white-space:nowrap}
/* ── Reset Danger Zone ───────────────────────────────────── */
.danger-zone{border:1px solid rgba(239,68,68,.25);border-radius:14px;padding:1rem 1.4rem;margin-top:.9rem}
.danger-title{font-size:.6rem;text-transform:uppercase;letter-spacing:.15em;color:var(--red);margin-bottom:.6rem}
.reset-btn{
  padding:.38rem .9rem;border-radius:8px;border:1px solid rgba(239,68,68,.4);
  background:rgba(239,68,68,.08);color:var(--red);cursor:pointer;font-size:.72rem;
  font-weight:700;letter-spacing:.06em;transition:.2s;font-family:inherit;
}
.reset-btn:hover{background:rgba(239,68,68,.18);box-shadow:0 0 10px rgba(239,68,68,.2)}
/* ── Paper Banner ────────────────────────────────────────── */
.paper-banner{
  border:1px solid rgba(245,158,11,.35);
  background:linear-gradient(135deg,rgba(245,158,11,.08),rgba(245,158,11,.03));
  border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:1rem;
  display:flex;align-items:center;flex-wrap:wrap;gap:1.2rem;
}
.paper-banner .pb-tag{
  display:flex;align-items:center;gap:.5rem;font-size:.75rem;font-weight:700;
  letter-spacing:.12em;color:var(--yellow);text-transform:uppercase;
}
.paper-banner .pb-dot{
  width:9px;height:9px;border-radius:50%;background:var(--yellow);
  box-shadow:0 0 8px var(--yellow);animation:blink 1.8s ease-in-out infinite;
}
.paper-banner .pb-stats{display:flex;flex-wrap:wrap;gap:1.5rem;margin-left:auto}
.paper-banner .pb-stat{text-align:right}
.paper-banner .pb-val{font-size:1.4rem;font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums;font-family:'JetBrains Mono',monospace}
.paper-banner .pb-lbl{font-size:.6rem;text-transform:uppercase;letter-spacing:.12em;color:var(--text3)}
/* ── Risk Cards ──────────────────────────────────────────── */
.risk-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:.75rem;margin-bottom:.9rem}
.rcard{border:1px solid var(--border);border-radius:12px;padding:1rem;text-align:center;
       background:var(--card);backdrop-filter:blur(18px)}
.rval{font-size:1.55rem;font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums;font-family:'JetBrains Mono',monospace}
.rlbl{font-size:.6rem;text-transform:uppercase;letter-spacing:.12em;color:var(--text3);margin-top:.3rem}
/* ── Golden Config Table ─────────────────────────────────── */
.cfg-tbl{width:100%;border-collapse:collapse;font-size:.8rem}
.cfg-tbl th{padding:.55rem .85rem;text-align:left;font-size:.6rem;text-transform:uppercase;
  letter-spacing:.1em;color:var(--text3);border-bottom:1px solid var(--border)}
.cfg-tbl td{padding:.52rem .85rem;border-bottom:1px solid rgba(99,179,237,.04);font-family:'JetBrains Mono',monospace;font-size:.78rem}
.cfg-tbl tr:hover td{background:rgba(6,182,212,.04)}
.cfg-ok{color:var(--green)}.cfg-warn{color:var(--red);font-weight:700}
/* ── Streak Pill ─────────────────────────────────────────── */
.streak-pill{
  display:inline-flex;align-items:center;gap:.4rem;
  padding:.4rem .9rem;border-radius:999px;font-size:.78rem;font-weight:700;border:1px solid;
}
.streak-win{background:rgba(16,185,129,.12);border-color:rgba(16,185,129,.35);color:var(--green)}
.streak-loss{background:rgba(239,68,68,.12);border-color:rgba(239,68,68,.35);color:var(--red)}
.streak-none{background:rgba(99,179,237,.08);border-color:rgba(99,179,237,.25);color:var(--cyan)}
/* ── Calibration ─────────────────────────────────────────── */
.calib-empty{color:var(--text3);font-size:.8rem;padding:1.5rem;text-align:center}
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
      <!-- Start Trader button — only when offline -->
      <button x-show="!m.trader_running" class="start-btn" :disabled="starting"
              @click="startTrader()">
        <span x-text="starting ? '⏳ Starting…' : '▶ Start Trader'"></span>
      </button>
      <!-- Last trade activity -->
      <div class="last-act" x-show="lastActText" x-text="'⏱ '+lastActText"></div>
      <div class="lupd" x-text="upd"></div>
      <button class="tbtn" @click="toggleDark()" :title="dark?'Light mode':'Dark mode'">
        <span x-text="dark ? '☀️' : '🌙'"></span>
      </button>
    </div>
  </div>
</header>

<div class="wrap">

  <!-- RECENT BETS LIVE FEED -->
  <div class="sec">Ultime Bet <span style="color:var(--cyan);margin-left:.4rem" x-text="trades.length?'('+trades.length+')':''"></span></div>
  <div class="card" style="padding:1rem 1.15rem;margin-bottom:.9rem">
    <div x-show="trades.length===0" style="color:var(--text3);font-size:.8rem;text-align:center;padding:1.5rem">
      Nessun trade ancora — l'algo sta scansionando mercati…
    </div>
    <div class="live-feed">
      <template x-for="t in trades.slice(0,20)" :key="t.ts+t.ticker">
        <div class="bet-row" :class="t.status">
          <div class="bet-time" x-text="relTime(t.ts)"></div>
          <span class="bdg" :class="t.action==='BUY_NO'?'bn':'by'" x-text="t.action" style="flex-shrink:0"></span>
          <div class="bet-ticker" :title="t.title" x-text="t.ticker.slice(0,30)"></div>
          <div class="mono" style="color:var(--text3);white-space:nowrap" x-text="t.price+'¢ × '+t.contracts"></div>
          <div class="mono" style="color:var(--text2);white-space:nowrap" x-text="'edge '+t.edge+'%'"></div>
          <span class="bdg" :class="t.status==='won'?'bw':t.status==='lost'?'bl':'bp'" x-text="t.status" style="flex-shrink:0"></span>
          <div class="bet-pnl" x-show="t.pnl_usd!==null"
               :class="t.pnl_usd>=0?'pp':'pn'"
               x-text="(t.pnl_usd>=0?'+':'')+'$'+(t.pnl_usd||0).toFixed(2)"></div>
          <div style="color:var(--text3);font-size:.7rem;white-space:nowrap" x-show="t.pnl_usd===null">⏳</div>
        </div>
      </template>
    </div>
  </div>

  <!-- PAPER MODE BANNER -->
  <div class="paper-banner">
    <div class="pb-tag"><div class="pb-dot"></div>Paper Trading Mode</div>
    <div class="pb-stats">
      <div class="pb-stat">
        <div class="pb-val yellow" x-text="'$'+(paper.current_balance_usd||0).toFixed(2)">—</div>
        <div class="pb-lbl">Balance / <span x-text="'$'+(paper.starting_balance_usd||100).toFixed(0)"></span> start</div>
      </div>
      <div class="pb-stat">
        <div class="pb-val" :class="(m.roi_pct||0)>=0?'green':'red'"
             x-text="((m.roi_pct||0)>=0?'+':'')+(m.roi_pct||0).toFixed(1)+'%'">—</div>
        <div class="pb-lbl">Paper ROI</div>
      </div>
      <div class="pb-stat">
        <div class="pb-val cyan" x-text="(paper.open_positions||0)">—</div>
        <div class="pb-lbl">Open Positions</div>
      </div>
      <div class="pb-stat">
        <div class="pb-val" :class="m.trader_running?'green':'red'" x-text="m.trader_running?'RUNNING':'STOPPED'">—</div>
        <div class="pb-lbl">Trader</div>
      </div>
    </div>
  </div>

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

  <!-- RISK METRICS + STREAKS -->
  <div class="sec">Risk Metrics &amp; Streaks</div>
  <div class="risk-grid">
    <div class="rcard">
      <div class="rval" :class="(risk.sharpe||0)>=1?'green':(risk.sharpe||0)>=0?'yellow':'red'"
           x-text="(risk.sharpe||0).toFixed(2)">—</div>
      <div class="rlbl">Sharpe Ratio</div>
    </div>
    <div class="rcard">
      <div class="rval" :class="(risk.sortino||0)>=1?'green':(risk.sortino||0)>=0?'yellow':'red'"
           x-text="(risk.sortino||0).toFixed(2)">—</div>
      <div class="rlbl">Sortino Ratio</div>
    </div>
    <div class="rcard">
      <div class="rval" :class="(risk.calmar||0)>=1?'green':(risk.calmar||0)>=0?'yellow':'red'"
           x-text="(risk.calmar||0).toFixed(2)">—</div>
      <div class="rlbl">Calmar Ratio</div>
    </div>
    <div class="rcard">
      <div class="rval red" x-text="(risk.maxDrawdownPct||0).toFixed(1)+'%'">—</div>
      <div class="rlbl">Max Drawdown</div>
    </div>
    <div class="rcard">
      <div class="rval" :class="(risk.profitFactor||0)>=1?'green':'red'"
           x-text="(risk.profitFactor||0).toFixed(2)">—</div>
      <div class="rlbl">Profit Factor</div>
    </div>
    <div class="rcard">
      <div class="rval cyan"
           x-text="risk.avgDurationHours!=null?(risk.avgDurationHours||0).toFixed(1)+'h':'—'">—</div>
      <div class="rlbl">Avg Duration</div>
    </div>
    <div class="rcard" style="grid-column:span 2">
      <div style="display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:.75rem;padding:.2rem 0">
        <span class="streak-pill" :class="streaks.currentType==='win'?'streak-win':streaks.currentType==='loss'?'streak-loss':'streak-none'">
          Current:
          <span x-text="streaks.currentType==='win'?'🔥 '+streaks.current+'W':streaks.currentType==='loss'?'❄️ '+streaks.current+'L':'—'"></span>
        </span>
        <span class="streak-pill streak-win">Best: <span x-text="streaks.longestWin+'W'"></span></span>
        <span class="streak-pill streak-loss">Worst: <span x-text="streaks.longestLoss+'L'"></span></span>
      </div>
      <div class="rlbl" style="text-align:center;margin-top:.5rem">Win/Loss Streaks</div>
    </div>
  </div>

  <!-- FORECAST CALIBRATION -->
  <div class="sec">Forecast Calibration</div>
  <div class="cg2">
    <div class="card cc">
      <div class="ct">Predicted vs Actual Win Rate per Prob Bin</div>
      <div x-show="calibration && calibration.length>0" id="ch-calibration"></div>
      <div x-show="!calibration || calibration.length===0" class="calib-empty">
        Nessun dato calibration — servono trade settled con forecast_prob
      </div>
    </div>
    <div class="card cc">
      <div class="ct">Edge Distribution: BUY_YES vs BUY_NO</div>
      <div id="ch-edge-dist"></div>
    </div>
  </div>

  <!-- CONFIG VS GOLDEN -->
  <div class="sec">Config vs Golden Config</div>
  <div class="card" style="padding:1.15rem;margin-bottom:.9rem">
    <table class="cfg-tbl">
      <thead><tr><th>Parametro</th><th>Attuale</th><th>Golden</th><th>Delta</th></tr></thead>
      <tbody>
        <template x-for="[k,gv] in Object.entries(golden)" :key="k">
          <tr>
            <td style="color:var(--text2)" x-text="k.replace(/_/g,' ')"></td>
            <td :class="cfg[k] && cfg[k]!=gv?'cfg-warn':'cfg-ok'" x-text="cfg[k]||'—'"></td>
            <td style="color:var(--text3)" x-text="gv"></td>
            <td>
              <span x-show="cfg[k] && cfg[k]!=gv" class="bdg bl">DRIFT</span>
              <span x-show="!cfg[k]||cfg[k]==gv" class="bdg bw">OK</span>
            </td>
          </tr>
        </template>
      </tbody>
    </table>
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

  <!-- DANGER ZONE -->
  <div class="danger-zone">
    <div class="danger-title">⚠️ Danger Zone</div>
    <div style="display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap">
      <button class="reset-btn" @click="resetPaper()" :disabled="resetting">
        <span x-text="resetting ? '⏳ Resettando…' : '🔄 Reset Paper Mode ($100)'"></span>
      </button>
      <span style="font-size:.7rem;color:var(--text3)">
        Azzera balance e posizioni paper. I dati SQLite (storico trade) NON vengono cancellati.
      </span>
    </div>
    <div x-show="resetMsg" x-text="resetMsg" style="margin-top:.6rem;font-size:.75rem;color:var(--green)"></div>
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
    paper:{current_balance_usd:0,starting_balance_usd:100,open_positions:0},
    trades:[],roi:[],cfg:{},upd:'—',_ch:{},
    cycles:[],daily:[],hourly:[],cycleHourly:[],
    calibration:[],streaks:{longestWin:0,longestLoss:0,current:0,currentType:'none'},
    risk:{sharpe:0,sortino:0,calmar:0,maxDrawdownPct:0,profitFactor:0,avgDurationHours:null},
    edgeDist:{yes:[],no:[]},
    starting:false, resetting:false, resetMsg:'',

    // GOLDEN CONFIG for comparison
    golden:{MIN_EDGE_BUY_NO:'3%',MIN_EDGE_BUY_YES:'8%',KELLY_FRACTION:'0.25',
            MAX_BET_CENTS:'500',MAX_POSITIONS:'15',DAILY_LOSS_LIMIT_CENTS:'500'},

    get lastActText(){
      if(!this.trades||!this.trades.length) return '';
      const ts=this.trades[0].ts; if(!ts) return '';
      const d=new Date(ts.replace(' ','T')+'Z');
      const diff=Math.floor((Date.now()-d)/1000);
      if(diff<60)   return diff+'s ago';
      if(diff<3600) return Math.floor(diff/60)+'m ago';
      return Math.floor(diff/3600)+'h ago';
    },

    relTime(ts){
      if(!ts) return '—';
      const d=new Date(ts.replace(' ','T')+'Z');
      const diff=Math.floor((Date.now()-d)/1000);
      if(diff<60)   return diff+'s fa';
      if(diff<3600) return Math.floor(diff/60)+'m fa';
      if(diff<86400) return Math.floor(diff/3600)+'h fa';
      return ts.slice(5,16);
    },

    async startTrader(){
      this.starting=true;
      try{
        const r=await fetch('/api/start_trader',{method:'POST'}).then(x=>x.json());
        if(r.ok){ await this.refresh(); }
        else{ alert('Start failed: '+r.reason); }
      }catch(e){alert('Error: '+e);}
      this.starting=false;
    },

    async resetPaper(){
      if(!confirm('CONFERMA: azzerare paper mode? Balance torna a $100, posizioni cancellate. I trade storici SQLite rimangono intatti.')) return;
      this.resetting=true; this.resetMsg='';
      try{
        const r=await fetch('/api/reset_paper',{
          method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({confirm:true,reason:'manual reset from dashboard'})
        }).then(x=>x.json());
        if(r.ok){
          this.resetMsg='✅ Reset completato — balance: $100.00';
          await this.refresh();
        } else { this.resetMsg='❌ '+r.reason; }
      }catch(e){this.resetMsg='❌ '+e;}
      this.resetting=false;
    },

    init(){ this.refresh(); setInterval(()=>this.refresh(),15000); },

    toggleDark(){
      this.dark=!this.dark;
      localStorage.setItem('kd',this.dark?'dark':'light');
      setTimeout(()=>this.charts(),80);
    },

    async refresh(){
      try{
        const d=await fetch('/api/all').then(r=>r.json());
        this.m=d.metrics;
        this.roi=d.roi_series||[];
        this.trades=d.trades||[];
        this.cfg=d.config||{};
        this.cycles=d.cycle_series||[];
        this.daily=d.daily_stats||[];
        this.paper=d.paper_state||this.paper;
        this.calibration=d.calibration||[];
        this.streaks=d.streaks||this.streaks;
        this.risk=d.risk||this.risk;
        this.edgeDist=d.edge_dist||this.edgeDist;
        // hourly comes from daily_stats; cycle_hourly needs separate fetch
        const [hourly,cycleHourly]=await Promise.all([
          fetch('/api/hourly_distribution').then(r=>r.json()),
          fetch('/api/cycle_hourly').then(r=>r.json()),
        ]);
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

      /* Bankroll — fallback to cycle data when no settled trades yet */
      let bk=m.bankroll_series&&m.bankroll_series.length?m.bankroll_series:null;
      if(!bk&&cycles&&cycles.length) bk=cycles.map(c=>({t:c.t,v:c.b}));
      if(!bk&&roi&&roi.length) bk=roi.map(r=>({t:r.t,v:+(m.starting_balance_usd+r.v/100*m.starting_balance_usd).toFixed(2)}));
      if(bk&&bk.length){
        const bkLabel=m.bankroll_series&&m.bankroll_series.length?'Bankroll ($)':'Balance per Ciclo ($)';
        this.mk('ch-bankroll',{...b,
          chart:{...b.chart,type:'area',height:210},
          series:[{name:bkLabel,data:bk.map(p=>({x:p.t,y:p.v}))}],
          stroke:{curve:'smooth',width:2},
          fill:{type:'gradient',gradient:{shadeIntensity:1,opacityFrom:.42,opacityTo:.03,stops:[0,100]}},
          colors:['#06b6d4'],
          markers:{size:bk.length<40?4:0,colors:['#06b6d4'],strokeWidth:0},
          yaxis:{...b.yaxis,labels:{...b.yaxis.labels,formatter:v=>'$'+v.toFixed(2)}},
          xaxis:{...b.xaxis,type:'category',tickAmount:Math.min(bk.length,12),
                 labels:{...b.xaxis.labels,rotate:-30,maxHeight:55,formatter:v=>v?v.slice(5,16):''}},
          annotations:{yaxis:[{y:m.starting_balance_usd,borderColor:'#f59e0b',strokeDashArray:4,
            label:{text:'Start',style:{background:'rgba(245,158,11,.12)',color:'#f59e0b',fontSize:'9px'}}}]},
        });
      }

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

      /* Forecast Calibration */
      const cal=this.calibration||[];
      if(cal.length){
        this.mk('ch-calibration',{...b,
          chart:{...b.chart,type:'bar',height:230},
          series:[
            {name:'Predicted %',data:cal.map(c=>c.predicted)},
            {name:'Actual WR %',data:cal.map(c=>c.actual)},
          ],
          xaxis:{...b.xaxis,categories:cal.map(c=>c.bin),
                 labels:{...b.xaxis.labels,rotate:-30,maxHeight:60}},
          colors:['#94a3b8','#10b981'],
          plotOptions:{bar:{borderRadius:3,columnWidth:'70%'}},
          yaxis:{...b.yaxis,max:100,labels:{...b.yaxis.labels,formatter:v=>v+'%'}},
          tooltip:{y:{formatter:v=>v.toFixed(1)+'%'}},
          annotations:{yaxis:[{y:50,borderColor:'#f59e0b',strokeDashArray:4,
            label:{text:'50%',style:{background:'rgba(245,158,11,.1)',color:'#f59e0b',fontSize:'9px'}}}]},
        });
      }

      /* Edge Distribution by Action */
      const ed=this.edgeDist||{};
      const yBkts=(ed.yes||[]).map(b2=>b2.count);
      const nBkts=(ed.no||[]).map(b2=>b2.count);
      const eBins=(ed.yes||ed.no||[]).map(b2=>b2.bucket);
      if(eBins.length){
        this.mk('ch-edge-dist',{...b,
          chart:{...b.chart,type:'bar',height:230},
          series:[
            {name:'BUY_YES',data:yBkts},
            {name:'BUY_NO', data:nBkts},
          ],
          xaxis:{...b.xaxis,categories:eBins,
                 labels:{...b.xaxis.labels,rotate:-30,maxHeight:60}},
          colors:['#10b981','#ef4444'],
          plotOptions:{bar:{borderRadius:3,columnWidth:'65%'}},
          yaxis:{...b.yaxis,labels:{...b.yaxis.labels,formatter:v=>v}},
          tooltip:{y:{formatter:(v,{seriesIndex:si,dataPointIndex:di})=>{
            const src=si===0?(ed.yes||[]):(ed.no||[]);
            const wr=(src[di]||{}).win_rate||0;
            return v+' trades (WR '+wr+'%)';
          }}},
        });
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
    print("🚀 Kalshi Dashboard → http://localhost:8887")
    app.run(host="::", port=8887, debug=False)
