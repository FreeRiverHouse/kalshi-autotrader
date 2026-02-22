#!/usr/bin/env python3
"""
Kalshi Win Rate Alert - Checks win rate and creates alert file if low.
Used by heartbeat to send Telegram notifications.

Usage:
    python kalshi-winrate-alert.py             # Check and update alert status
    python kalshi-winrate-alert.py --status    # Just show status
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ============== CONFIG ==============
LOW_WINRATE_THRESHOLD = 0.40  # 40%
MIN_TRADES_FOR_ALERT = 10  # Need at least 10 trades to trigger alert
ALERT_FILE = Path(__file__).parent / "kalshi-low-winrate.alert"
SETTLEMENTS_FILE = Path(__file__).parent / "kalshi-settlements.json"
COOLDOWN_HOURS = 6  # Don't spam alerts


def load_settlements() -> dict:
    """Load settlements data."""
    if SETTLEMENTS_FILE.exists():
        try:
            with open(SETTLEMENTS_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def check_and_alert():
    """Check win rate and update alert file."""
    settlements = load_settlements()
    summary = settlements.get("summary", {})
    
    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    total = wins + losses
    pnl_cents = summary.get("total_pnl_cents", 0)
    pending = summary.get("pending", 0)
    
    if total == 0:
        print("📊 No settled trades yet.")
        return None
    
    win_rate = wins / total
    
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wins": wins,
        "losses": losses,
        "total": total,
        "win_rate": win_rate,
        "pnl_cents": pnl_cents,
        "pending": pending,
        "threshold": LOW_WINRATE_THRESHOLD,
        "is_low": win_rate < LOW_WINRATE_THRESHOLD and total >= MIN_TRADES_FOR_ALERT
    }
    
    print(f"📊 Trading Performance")
    print(f"   Wins: {wins}")
    print(f"   Losses: {losses}")
    print(f"   Win Rate: {win_rate*100:.1f}%")
    print(f"   PnL: ${pnl_cents/100:.2f}")
    print(f"   Pending: {pending}")
    print(f"   Threshold: {LOW_WINRATE_THRESHOLD*100:.0f}%")
    
    # Check cooldown
    if ALERT_FILE.exists():
        try:
            with open(ALERT_FILE) as f:
                old_alert = json.load(f)
            old_ts = datetime.fromisoformat(old_alert.get("timestamp", "2000-01-01T00:00:00+00:00"))
            now = datetime.now(timezone.utc)
            hours_since = (now - old_ts).total_seconds() / 3600
            
            if hours_since < COOLDOWN_HOURS:
                print(f"\n⏰ Cooldown active ({hours_since:.1f}h of {COOLDOWN_HOURS}h)")
                return result
        except (json.JSONDecodeError, KeyError):
            pass
    
    if result["is_low"]:
        print(f"\n🚨 LOW WIN RATE ALERT! {win_rate*100:.1f}% < {LOW_WINRATE_THRESHOLD*100:.0f}%")
        # Write alert file
        with open(ALERT_FILE, "w") as f:
            json.dump(result, f, indent=2)
        print(f"📝 Alert written to {ALERT_FILE}")
    elif total < MIN_TRADES_FOR_ALERT:
        print(f"\n⏳ Not enough trades yet ({total} < {MIN_TRADES_FOR_ALERT})")
        if ALERT_FILE.exists():
            ALERT_FILE.unlink()
            print(f"🗑️ Cleared old alert")
    else:
        print(f"\n✅ Win Rate OK")
        if ALERT_FILE.exists():
            ALERT_FILE.unlink()
            print(f"🗑️ Cleared old alert")
    
    return result


def get_status():
    """Just show current status without writing alerts."""
    settlements = load_settlements()
    summary = settlements.get("summary", {})
    
    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    total = wins + losses
    pnl_cents = summary.get("total_pnl_cents", 0)
    
    print(f"📊 Kalshi Win Rate Status")
    if total > 0:
        print(f"   Wins: {wins} | Losses: {losses}")
        print(f"   Win Rate: {wins/total*100:.1f}%")
        print(f"   PnL: ${pnl_cents/100:.2f}")
    else:
        print("   No settled trades")
    
    if ALERT_FILE.exists():
        print(f"\n⚠️ Active low win rate alert!")
    else:
        print(f"\n✅ No active alerts")


if __name__ == "__main__":
    if "--status" in sys.argv:
        get_status()
    else:
        check_and_alert()
