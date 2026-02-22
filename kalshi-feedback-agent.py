#!/usr/bin/env python3
"""
Kalshi Feedback Loop Agent
Reads metrics from dashboard API every N cycles and fine-tunes algo params.
Run as: python3 kalshi-feedback-agent.py [--interval 1800]
"""

import json
import re
import sys
import time
import argparse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DASHBOARD_URL = "http://localhost:8889"
AUTOTRADER    = Path(__file__).parent / "kalshi-autotrader.py"
LOG_FILE      = Path(__file__).parent / "data" / "trading" / "feedback-agent.log"

# ── Tuning rules ──────────────────────────────────────────────────────────────
# Each rule: (condition_fn, param, new_value, reason)
# Conditions receive the metrics dict. Applied only if >= MIN_SETTLED trades.

MIN_SETTLED_FOR_TUNING = 20   # Don't tune with fewer than N settled trades

RULES = [
    # BUY_NO performing well (>75% WR, >10 trades) → slightly lower min edge to get more volume
    {
        "name": "no_wr_high_lower_edge",
        "condition": lambda m: m["no_win_rate"] >= 75 and m["no_trades"] >= 10,
        "param": "MIN_EDGE_BUY_NO",
        "target": "0.015",
        "reason": "BUY_NO WR >= 75% → lower min edge to 1.5% for more volume",
    },
    # BUY_NO performing poorly (<50% WR) → raise min edge
    {
        "name": "no_wr_low_raise_edge",
        "condition": lambda m: m["no_win_rate"] < 50 and m["no_trades"] >= 10,
        "param": "MIN_EDGE_BUY_NO",
        "target": "0.04",
        "reason": "BUY_NO WR < 50% → raise min edge to 4% for higher conviction",
    },
    # BUY_YES catastrophic (<15% WR with >10 trades) → raise threshold significantly
    {
        "name": "yes_wr_very_low",
        "condition": lambda m: m["yes_win_rate"] < 15 and m["yes_trades"] >= 10,
        "param": "MIN_EDGE_BUY_YES",
        "target": "0.08",
        "reason": "BUY_YES WR < 15% → raise min edge to 8%",
    },
    # ROI positive and >10 trades → can be slightly more aggressive on Kelly
    {
        "name": "roi_positive_increase_kelly",
        "condition": lambda m: m["roi_pct"] > 5 and m["settled"] >= 20,
        "param": "KELLY_FRACTION",
        "target": "0.30",
        "reason": "ROI > 5% with 20+ trades → bump Kelly to 0.30",
    },
    # ROI negative with >20 trades → get more conservative
    {
        "name": "roi_negative_lower_kelly",
        "condition": lambda m: m["roi_pct"] < -10 and m["settled"] >= 20,
        "param": "KELLY_FRACTION",
        "target": "0.20",
        "reason": "ROI < -10% → lower Kelly to 0.20",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_metrics():
    try:
        with urllib.request.urlopen(f"{DASHBOARD_URL}/api/status", timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:
        return None

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

def read_param(param):
    src = AUTOTRADER.read_text()
    m = re.search(rf'^{param}\s*=\s*([^\s#\n]+)', src, re.MULTILINE)
    return m.group(1) if m else None

def set_param(param, new_val):
    src = AUTOTRADER.read_text()
    new_src, n = re.subn(
        rf'^({param}\s*=\s*)([^\s#\n]+)',
        lambda m: m.group(1) + new_val,
        src,
        flags=re.MULTILINE
    )
    if n == 0:
        log(f"  ⚠️  Param {param} not found in script")
        return False
    AUTOTRADER.write_text(new_src)
    return True

def restart_trader():
    import subprocess
    # Kill existing
    subprocess.run(["pkill", "-f", "kalshi-autotrader.py"], capture_output=True)
    time.sleep(2)
    # Restart
    log_path = Path(__file__).parent / "kalshi-autotrader.log"
    p = subprocess.Popen(
        [sys.executable, "-u", str(AUTOTRADER), "--loop", "300"],
        stdout=open(log_path, "a"),
        stderr=subprocess.STDOUT,
        cwd=str(AUTOTRADER.parent)
    )
    log(f"  ✅ Trader restarted (PID {p.pid})")
    return p.pid

# ── Main loop ─────────────────────────────────────────────────────────────────

def run_check(dry_run=False):
    log("🔍 Feedback agent — checking metrics...")
    m = fetch_metrics()
    if not m:
        log("  ❌ Dashboard not reachable (is it running on :8888?)")
        return

    log(f"  ROI={m['roi_pct']:+.1f}%  WR={m['win_rate']:.0f}%  "
        f"NO_WR={m['roi_pct']:+.0f}%  settled={m['settled_trades']}  "
        f"balance=${m['current_balance_usd']:.2f}")

    if m["settled_trades"] < MIN_SETTLED_FOR_TUNING:
        log(f"  ⏸  Only {m['settled_trades']} settled trades — need {MIN_SETTLED_FOR_TUNING} before tuning")
        return

    applied = []
    for rule in RULES:
        try:
            if rule["condition"](m):
                current = read_param(rule["param"])
                if current == rule["target"]:
                    continue  # already at target
                log(f"  📐 RULE '{rule['name']}': {rule['param']} {current} → {rule['target']}")
                log(f"      Reason: {rule['reason']}")
                if not dry_run:
                    if set_param(rule["param"], rule["target"]):
                        applied.append(f"{rule['param']}={rule['target']}")
        except Exception as e:
            log(f"  ⚠️  Rule '{rule['name']}' error: {e}")

    if applied and not dry_run:
        log(f"  ✏️  Applied: {', '.join(applied)} — restarting trader...")
        restart_trader()
    elif applied and dry_run:
        log(f"  [DRY-RUN] Would apply: {', '.join(applied)}")
    else:
        log("  ✅ No tuning needed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=1800,
                        help="Check interval in seconds (default: 1800 = 30 min)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without applying")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit")
    args = parser.parse_args()

    log(f"🤖 Kalshi Feedback Agent started (interval={args.interval}s, dry_run={args.dry_run})")
    run_check(dry_run=args.dry_run)
    if args.once:
        return

    while True:
        time.sleep(args.interval)
        run_check(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
