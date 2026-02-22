#!/usr/bin/env python3
"""
Kalshi PnL Threshold Alert

Monitors daily and weekly PnL and creates alerts when thresholds are crossed.
Designed to run every hour via cron, creates .alert files for heartbeat pickup.

Thresholds (configurable):
- Daily profit target: +$50
- Daily loss limit: -$30
- Weekly profit milestone: +$200
- Weekly loss warning: -$100

Features:
- Only alerts once per threshold per period (tracks alerted state)
- Includes context: current position count, win rate, streak
- Resets alert state at midnight (daily) and Sunday (weekly)
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Configuration
THRESHOLDS = {
    "daily_profit_target": 50.00,     # +$50 daily profit
    "daily_loss_limit": -30.00,       # -$30 daily loss
    "weekly_profit_milestone": 200.00, # +$200 weekly profit
    "weekly_loss_warning": -100.00,   # -$100 weekly loss
}

# Files
SCRIPT_DIR = Path(__file__).parent
TRADES_FILE = SCRIPT_DIR / "kalshi-trades-v2.jsonl"
TRADES_FILE_V1 = SCRIPT_DIR / "kalshi-trades.jsonl"
STATE_FILE = SCRIPT_DIR / "kalshi-pnl-alert-state.json"
ALERT_FILE = SCRIPT_DIR / "kalshi-pnl-threshold.alert"

def get_pst_now():
    """Get current time in PST."""
    pst_offset = timedelta(hours=-8)
    return datetime.now(timezone.utc) + pst_offset

def get_pst_date(dt):
    """Get PST date from datetime."""
    pst_offset = timedelta(hours=-8)
    pst_time = dt + pst_offset if dt.tzinfo else dt
    return pst_time.strftime('%Y-%m-%d')

def get_week_start(dt):
    """Get the start of the week (Sunday) for a given date."""
    pst_offset = timedelta(hours=-8)
    pst_time = dt + pst_offset if dt.tzinfo else dt
    days_since_sunday = pst_time.weekday() + 1  # Monday=0, so Sunday=6+1=7, but we want 0
    if pst_time.weekday() == 6:  # Sunday
        days_since_sunday = 0
    week_start = pst_time - timedelta(days=days_since_sunday)
    return week_start.strftime('%Y-%m-%d')

def load_trades():
    """Load all trades from JSONL file."""
    trades_file = TRADES_FILE if TRADES_FILE.exists() else TRADES_FILE_V1
    if not trades_file.exists():
        return []
    
    trades = []
    with open(trades_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("type") == "trade":
                    trades.append(entry)
            except:
                pass
    return trades

def calculate_pnl_stats(trades, start_date: str, end_date: str = None):
    """
    Calculate PnL statistics for trades within a date range.
    Returns: dict with total_pnl, wins, losses, pending, win_rate
    
    PnL calculation for Kalshi:
    - Won trade: profit = (100 * contracts) - cost_cents (you get $1 per contract minus what you paid)
    - Lost trade: profit = -cost_cents (you lose what you paid)
    """
    stats = {
        "total_pnl_cents": 0,
        "wins": 0,
        "losses": 0,
        "pending": 0,
        "total_settled": 0,
    }
    
    for trade in trades:
        ts = trade.get("timestamp", "")
        if not ts:
            continue
        
        try:
            trade_date = get_pst_date(datetime.fromisoformat(ts.replace('Z', '+00:00')))
        except:
            continue
        
        # Check date range
        if trade_date < start_date:
            continue
        if end_date and trade_date > end_date:
            continue
        
        result = trade.get("result_status", "pending")
        contracts = trade.get("contracts", 0)
        cost_cents = trade.get("cost_cents", 0)
        
        if result in ("won", "win"):
            stats["wins"] += 1
            stats["total_settled"] += 1
            # Won: get $1 per contract minus what we paid
            pnl = (100 * contracts) - cost_cents
            stats["total_pnl_cents"] += pnl
        elif result in ("lost", "loss"):
            stats["losses"] += 1
            stats["total_settled"] += 1
            # Lost: lose what we paid
            stats["total_pnl_cents"] -= cost_cents
        else:
            stats["pending"] += 1
    
    stats["win_rate"] = (stats["wins"] / stats["total_settled"] * 100) if stats["total_settled"] > 0 else 0
    stats["total_pnl"] = stats["total_pnl_cents"] / 100
    
    return stats

def load_alert_state():
    """Load the alert state (which thresholds have been alerted)."""
    if not STATE_FILE.exists():
        return {"daily_alerted": {}, "weekly_alerted": {}, "last_daily_reset": None, "last_weekly_reset": None}
    
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return {"daily_alerted": {}, "weekly_alerted": {}, "last_daily_reset": None, "last_weekly_reset": None}

def save_alert_state(state):
    """Save the alert state."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def check_and_reset_state(state):
    """Check if we need to reset alert state (new day/week)."""
    now = get_pst_now()
    today = now.strftime('%Y-%m-%d')
    week_start = get_week_start(now)
    
    # Reset daily alerts if new day
    if state.get("last_daily_reset") != today:
        state["daily_alerted"] = {}
        state["last_daily_reset"] = today
    
    # Reset weekly alerts if new week
    if state.get("last_weekly_reset") != week_start:
        state["weekly_alerted"] = {}
        state["last_weekly_reset"] = week_start
    
    return state

def create_alert(threshold_name: str, pnl: float, stats: dict, period: str):
    """Create an alert file for heartbeat pickup."""
    now = get_pst_now()
    
    # Determine emoji and message based on threshold
    if "profit" in threshold_name or "milestone" in threshold_name:
        emoji = "🎉" if pnl >= 0 else "📈"
        sentiment = "reached" if pnl >= 0 else "hit"
    else:
        emoji = "⚠️"
        sentiment = "hit"
    
    # Format message
    threshold_display = threshold_name.replace("_", " ").title()
    message = f"""💰 **PnL Alert: {threshold_display}!** {emoji}

📊 **{period} Performance:**
• P&L: **${pnl:+.2f}**
• Win Rate: {stats['win_rate']:.1f}%
• Trades: {stats['wins']}W / {stats['losses']}L ({stats['pending']} pending)

⏰ {now.strftime('%Y-%m-%d %H:%M')} PST

🎯 Threshold ${THRESHOLDS[threshold_name]:+.2f} {sentiment}!"""
    
    # Write alert file
    with open(ALERT_FILE, 'w') as f:
        f.write(message)
    
    print(f"✅ Alert created: {threshold_name}")

def check_thresholds():
    """Main function to check PnL thresholds and create alerts."""
    now = get_pst_now()
    today = now.strftime('%Y-%m-%d')
    week_start = get_week_start(now)
    
    print(f"📊 Checking PnL thresholds at {now.strftime('%Y-%m-%d %H:%M')} PST")
    print(f"   Today: {today}")
    print(f"   Week start: {week_start}")
    
    # Load trades
    trades = load_trades()
    if not trades:
        print("❌ No trades found")
        return
    
    # Calculate stats
    daily_stats = calculate_pnl_stats(trades, today)
    weekly_stats = calculate_pnl_stats(trades, week_start)
    
    print(f"\n📈 Daily PnL: ${daily_stats['total_pnl']:+.2f} ({daily_stats['wins']}W/{daily_stats['losses']}L)")
    print(f"📈 Weekly PnL: ${weekly_stats['total_pnl']:+.2f} ({weekly_stats['wins']}W/{weekly_stats['losses']}L)")
    
    # Load and reset state if needed
    state = load_alert_state()
    state = check_and_reset_state(state)
    
    alerts_created = 0
    
    # Check daily thresholds
    daily_pnl = daily_stats["total_pnl"]
    
    # Daily profit target
    if daily_pnl >= THRESHOLDS["daily_profit_target"]:
        if "daily_profit_target" not in state["daily_alerted"]:
            create_alert("daily_profit_target", daily_pnl, daily_stats, "Daily")
            state["daily_alerted"]["daily_profit_target"] = True
            alerts_created += 1
    
    # Daily loss limit
    if daily_pnl <= THRESHOLDS["daily_loss_limit"]:
        if "daily_loss_limit" not in state["daily_alerted"]:
            create_alert("daily_loss_limit", daily_pnl, daily_stats, "Daily")
            state["daily_alerted"]["daily_loss_limit"] = True
            alerts_created += 1
    
    # Check weekly thresholds
    weekly_pnl = weekly_stats["total_pnl"]
    
    # Weekly profit milestone
    if weekly_pnl >= THRESHOLDS["weekly_profit_milestone"]:
        if "weekly_profit_milestone" not in state["weekly_alerted"]:
            create_alert("weekly_profit_milestone", weekly_pnl, weekly_stats, "Weekly")
            state["weekly_alerted"]["weekly_profit_milestone"] = True
            alerts_created += 1
    
    # Weekly loss warning
    if weekly_pnl <= THRESHOLDS["weekly_loss_warning"]:
        if "weekly_loss_warning" not in state["weekly_alerted"]:
            create_alert("weekly_loss_warning", weekly_pnl, weekly_stats, "Weekly")
            state["weekly_alerted"]["weekly_loss_warning"] = True
            alerts_created += 1
    
    # Save state
    save_alert_state(state)
    
    if alerts_created == 0:
        print("\n✅ No threshold crossings detected")
    else:
        print(f"\n🔔 Created {alerts_created} alert(s)")

if __name__ == "__main__":
    check_thresholds()
