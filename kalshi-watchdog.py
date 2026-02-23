#!/usr/bin/env python3
"""
Kalshi AutoTrader Watchdog
Runs every 30 minutes (via LaunchAgent or cron).

Checks:
  1. Trader process is alive → restarts if dead
  2. Dashboard (port 8888) is responding → restarts if down
  3. Recent cycle activity (last 2h should have cycles)
  4. Paper balance sanity
  5. Calls Claude API for AI analysis & recommendations
  6. Appends structured report to watchdog.log
"""

import json
import os
import sys
import time
import sqlite3
import subprocess
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

BASE        = Path(__file__).parent
DB_PATH     = BASE / "data" / "trading" / "trades.db"
PAPER_STATE = BASE / "data" / "trading" / "paper-trade-state.json"
TRADER_SCRIPT = BASE / "kalshi-autotrader.py"
DASHBOARD_SCRIPT = BASE / "kalshi-dashboard.py"
WATCHDOG_LOG = BASE / "data" / "trading" / "watchdog.log"
PYTHON      = "/opt/homebrew/bin/python3.11"
DASHBOARD_URL = "http://localhost:8888"
def _load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    env_file = Path.home() / ".clawdbot" / ".env.trading"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""

ANTHROPIC_API_KEY = _load_api_key()

# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")


# ── Health Checks ─────────────────────────────────────────────────────────────

def check_trader() -> dict:
    r = subprocess.run(["pgrep", "-f", "kalshi-autotrader.py"], capture_output=True, text=True)
    pids = [p for p in r.stdout.strip().split() if p.isdigit()]
    return {"running": bool(pids), "pid": int(pids[0]) if pids else None}


def check_dashboard() -> dict:
    try:
        with urllib.request.urlopen(f"{DASHBOARD_URL}/api/status", timeout=5) as r:
            data = json.loads(r.read())
            return {"running": True, "balance_usd": data.get("current_balance_usd"), "data": data}
    except Exception as e:
        return {"running": False, "error": str(e)}


def check_db_activity() -> dict:
    if not DB_PATH.exists():
        return {"ok": False, "reason": "DB missing"}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        # Cycles in last 2h
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        recent_cycles = conn.execute(
            "SELECT COUNT(*) FROM cycles WHERE timestamp > ?", (cutoff,)
        ).fetchone()[0]

        # Total trades
        total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE result_status='pending'"
        ).fetchone()[0]
        settled = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE result_status IN ('won','lost')"
        ).fetchone()[0]

        # Last cycle timestamp
        last_cycle_row = conn.execute(
            "SELECT timestamp FROM cycles ORDER BY id DESC LIMIT 1"
        ).fetchone()
        last_cycle = last_cycle_row[0] if last_cycle_row else None

        # Last trade timestamp
        last_trade_row = conn.execute(
            "SELECT timestamp FROM trades ORDER BY id DESC LIMIT 1"
        ).fetchone()
        last_trade = last_trade_row[0] if last_trade_row else None

        # Win rate
        won  = conn.execute("SELECT COUNT(*) FROM trades WHERE result_status='won'").fetchone()[0]
        wr   = round(won / settled * 100, 1) if settled else 0

        # ROI
        net_pnl = conn.execute(
            "SELECT COALESCE(SUM(pnl_cents),0) FROM trades WHERE result_status IN ('won','lost')"
        ).fetchone()[0]
        total_cost = conn.execute(
            "SELECT COALESCE(SUM(cost_cents),0) FROM trades WHERE result_status IN ('won','lost')"
        ).fetchone()[0]
        roi = round(net_pnl / total_cost * 100, 2) if total_cost else 0

        conn.close()

        return {
            "ok": True,
            "recent_cycles_2h": recent_cycles,
            "total_trades": total,
            "pending": pending,
            "settled": settled,
            "win_rate": wr,
            "roi_pct": roi,
            "net_pnl_cents": net_pnl,
            "last_cycle": last_cycle,
            "last_trade": last_trade,
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def check_paper_state() -> dict:
    if not PAPER_STATE.exists():
        return {"ok": False}
    try:
        s = json.loads(PAPER_STATE.read_text())
        balance = s.get("current_balance_cents", 0) / 100
        start   = s.get("starting_balance_cents", 10000) / 100
        return {
            "ok": True,
            "balance_usd": balance,
            "starting_usd": start,
            "roi_pct": round((balance - start) / start * 100, 2) if start else 0,
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ── Restart helpers ───────────────────────────────────────────────────────────

def restart_trader():
    log("Restarting trader…", "WARN")
    subprocess.Popen(
        [PYTHON, "-u", str(TRADER_SCRIPT), "--loop", "300"],
        stdout=open(BASE / "data" / "trading" / "kalshi-autotrader.log", "a"),
        stderr=subprocess.STDOUT,
        cwd=str(BASE),
    )
    time.sleep(3)
    status = check_trader()
    log(f"Trader restart → running={status['running']} pid={status['pid']}", "INFO")
    return status["running"]


def restart_dashboard():
    log("Restarting dashboard…", "WARN")
    subprocess.Popen(
        [PYTHON, "-u", str(DASHBOARD_SCRIPT)],
        stdout=open("/tmp/kalshi-dashboard.log", "a"),
        stderr=subprocess.STDOUT,
        cwd=str(BASE),
    )
    time.sleep(3)
    status = check_dashboard()
    log(f"Dashboard restart → running={status['running']}", "INFO")
    return status["running"]


# ── Claude Analysis ───────────────────────────────────────────────────────────

def claude_analysis(report: dict) -> str:
    if not ANTHROPIC_API_KEY:
        return "(ANTHROPIC_API_KEY not set — skipping AI analysis)"
    try:
        prompt = f"""Sei il watchdog del Kalshi AutoTrader. Analizza questo report e dai un parere in 3-5 bullet points:
- Il trader sta girando bene?
- Il ROI è in linea con le aspettative?
- Qualcosa da ottimizzare o da controllare?
- Azioni consigliate (se necessarie)

REPORT:
{json.dumps(report, indent=2, default=str)}

Rispondi in italiano, breve e diretto. Max 200 parole."""

        payload = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()

        is_oauth = ANTHROPIC_API_KEY.startswith("sk-ant-oat")
        headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if is_oauth:
            headers["authorization"] = f"Bearer {ANTHROPIC_API_KEY}"
            headers["anthropic-beta"] = "oauth-2025-04-20"
        else:
            headers["x-api-key"] = ANTHROPIC_API_KEY

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = json.loads(r.read())
            return resp["content"][0]["text"]
    except Exception as e:
        return f"(Claude error: {e})"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("Watchdog run starting")

    trader    = check_trader()
    dashboard = check_dashboard()
    db_info   = check_db_activity()
    paper     = check_paper_state()

    # ── Auto-fix: restart dead processes ──────────────────────────────────────
    if not trader["running"]:
        log("Trader is DOWN — restarting", "WARN")
        restart_trader()
        trader = check_trader()

    if not dashboard["running"]:
        log("Dashboard is DOWN — restarting", "WARN")
        restart_dashboard()
        dashboard = check_dashboard()

    # ── Warnings ──────────────────────────────────────────────────────────────
    if db_info.get("ok") and db_info.get("recent_cycles_2h", 0) == 0:
        log("⚠️  No cycles in last 2h — trader may be stalled", "WARN")

    if paper.get("ok") and paper.get("balance_usd", 100) < 50:
        log(f"⚠️  Balance critically low: ${paper['balance_usd']:.2f}", "WARN")

    # ── Build report ──────────────────────────────────────────────────────────
    report = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "trader":     trader,
        "dashboard":  {"running": dashboard["running"], "balance_usd": dashboard.get("balance_usd")},
        "db":         db_info,
        "paper":      paper,
        "status": "OK" if (trader["running"] and dashboard["running"] and db_info.get("ok")) else "DEGRADED",
    }

    log(f"Status: {report['status']} | "
        f"Trader PID={trader.get('pid')} | "
        f"Balance=${paper.get('balance_usd', '?')} | "
        f"Trades settled={db_info.get('settled', 0)} pending={db_info.get('pending', 0)} | "
        f"ROI={db_info.get('roi_pct', 0)}% | "
        f"WR={db_info.get('win_rate', 0)}% | "
        f"Cycles(2h)={db_info.get('recent_cycles_2h', 0)}")

    # ── AI analysis ───────────────────────────────────────────────────────────
    analysis = claude_analysis(report)
    log(f"Claude says:\n{analysis}")

    # ── Write JSON snapshot ────────────────────────────────────────────────────
    snap_path = BASE / "data" / "trading" / "watchdog-last.json"
    report["claude_analysis"] = analysis
    snap_path.write_text(json.dumps(report, indent=2, default=str))

    log("Watchdog run complete")
    log("=" * 60)

    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
