#!/usr/bin/env python3
"""
Kalshi Daily Trading Summary
Generates a daily summary for Telegram notification.
Output is written to kalshi-daily-summary.txt for cron delivery.
"""

import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path

TRADES_FILE = Path(__file__).parent / "kalshi-trades.jsonl"
V2_TRADES_FILE = Path(__file__).parent / "kalshi-trades-v2.jsonl"
OUTPUT_FILE = Path(__file__).parent / "kalshi-daily-summary.txt"
STREAK_RECORDS_FILE = Path(__file__).parent / "kalshi-streak-records.json"
STOP_LOSS_LOG = Path(__file__).parent / "kalshi-stop-loss.log"

def parse_timestamp(ts):
    """Parse ISO timestamp."""
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))

def get_pst_date(dt):
    """Convert datetime to PST date string."""
    pst_offset = timedelta(hours=-8)
    pst_time = dt + pst_offset
    return pst_time.strftime('%Y-%m-%d')

def calculate_streaks() -> dict:
    """
    Calculate current streak and longest streaks from trade log.
    Returns dict with: current_streak, current_type, longest_win, longest_loss
    """
    # Prefer v2, fallback to v1
    trades_file = V2_TRADES_FILE if V2_TRADES_FILE.exists() else TRADES_FILE
    
    if not trades_file.exists():
        return {"current_streak": 0, "current_type": None, "longest_win": 0, "longest_loss": 0}
    
    settled_trades = []
    with open(trades_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("type") == "trade":
                    result = entry.get("result_status", "pending")
                    if result in ("won", "lost", "win", "loss"):
                        settled_trades.append(entry)
            except:
                pass
    
    if not settled_trades:
        return {"current_streak": 0, "current_type": None, "longest_win": 0, "longest_loss": 0}
    
    # Sort by timestamp
    settled_trades.sort(key=lambda x: x.get("timestamp", ""))
    
    current_streak = 0
    current_type = None
    longest_win = 0
    longest_loss = 0
    
    for trade in settled_trades:
        result = trade.get("result_status", "")
        is_win = result in ("won", "win")
        is_loss = result in ("lost", "loss")
        
        if is_win:
            if current_type == "win":
                current_streak += 1
            else:
                current_streak = 1
                current_type = "win"
            longest_win = max(longest_win, current_streak)
        elif is_loss:
            if current_type == "loss":
                current_streak += 1
            else:
                current_streak = 1
                current_type = "loss"
            longest_loss = max(longest_loss, current_streak)
    
    return {
        "current_streak": current_streak,
        "current_type": current_type,
        "longest_win": longest_win,
        "longest_loss": longest_loss
    }

def load_streak_records() -> dict:
    """Load saved streak records for comparison."""
    try:
        with open(STREAK_RECORDS_FILE) as f:
            return json.load(f)
    except:
        return {"longest_win_streak": 0, "longest_loss_streak": 0}

def analyze_stop_losses_today(today_pst: str) -> dict:
    """
    Analyze stop-losses triggered today.
    Returns dict with: count, total_loss_cents, avg_loss_pct
    """
    if not STOP_LOSS_LOG.exists():
        return {"count": 0, "total_loss_cents": 0, "avg_loss_pct": 0}
    
    today_stops = []
    with open(STOP_LOSS_LOG) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("type") != "stop_loss":
                    continue
                # Parse timestamp and check date
                ts = entry.get("timestamp", "")
                if ts:
                    dt = parse_timestamp(ts)
                    if get_pst_date(dt) == today_pst:
                        today_stops.append(entry)
            except:
                pass
    
    if not today_stops:
        return {"count": 0, "total_loss_cents": 0, "avg_loss_pct": 0}
    
    total_loss = 0
    total_loss_pct = 0
    
    for sl in today_stops:
        # Calculate loss from stop-loss
        entry_price = sl.get("entry_price", 0) or sl.get("entry_price_cents", 0)
        exit_price = sl.get("exit_price", 0) or sl.get("exit_price_cents", 0)
        contracts = sl.get("contracts", 1)
        loss_pct = sl.get("loss_pct", 0)
        
        # Actual loss at exit
        loss_cents = (entry_price - exit_price) * contracts
        total_loss += loss_cents
        total_loss_pct += loss_pct
    
    count = len(today_stops)
    avg_loss_pct = total_loss_pct / count if count > 0 else 0
    
    return {
        "count": count,
        "total_loss_cents": total_loss,
        "avg_loss_pct": avg_loss_pct
    }

def analyze_daily():
    """Analyze today's trades and generate summary."""
    # Get today's date in PST
    now_utc = datetime.now(timezone.utc)
    today_pst = get_pst_date(now_utc)
    
    trades = []
    settled_won = 0
    settled_lost = 0
    pending = 0
    total_cost = 0
    pnl = 0
    
    with open(TRADES_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get('type') != 'trade' or entry.get('order_status') != 'executed':
                    continue
                    
                trade_date = get_pst_date(parse_timestamp(entry['timestamp']))
                if trade_date != today_pst:
                    continue
                    
                trades.append(entry)
                cost = entry.get('cost_cents', 0)
                total_cost += cost
                
                result = entry.get('result_status', 'pending')
                if result == 'won':
                    settled_won += 1
                    # NO bet wins: payout = contracts * 100 - cost
                    contracts = entry.get('contracts', 1)
                    pnl += (contracts * 100) - cost
                elif result == 'lost':
                    settled_lost += 1
                    pnl -= cost
                else:
                    pending += 1
                    
            except (json.JSONDecodeError, KeyError):
                continue
    
    total_trades = len(trades)
    settled = settled_won + settled_lost
    win_rate = (settled_won / settled * 100) if settled > 0 else 0
    
    # Get streak data
    streaks = calculate_streaks()
    records = load_streak_records()
    
    # Format current streak
    if streaks["current_type"] == "win":
        streak_emoji = "🔥"
        streak_text = f"{streak_emoji} {streaks['current_streak']}W"
    elif streaks["current_type"] == "loss":
        streak_emoji = "❄️"
        streak_text = f"{streak_emoji} {streaks['current_streak']}L"
    else:
        streak_text = "—"
    
    # Get stop-loss stats for today
    stop_loss_stats = analyze_stop_losses_today(today_pst)
    
    # Format stop-loss section
    if stop_loss_stats["count"] > 0:
        stop_loss_section = f"""
**Stop-Losses Today:**
🛑 Triggered: {stop_loss_stats['count']}
📉 Total loss: ${stop_loss_stats['total_loss_cents']/100:.2f}
📊 Avg loss: {stop_loss_stats['avg_loss_pct']:.1f}%"""
    else:
        stop_loss_section = ""
    
    # Generate summary message
    summary = f"""📊 **Kalshi Daily Summary** ({today_pst})

🎯 **Trades Today:** {total_trades}
💰 **Total Wagered:** ${total_cost/100:.2f}

**Settled Results:**
✅ Won: {settled_won}
❌ Lost: {settled_lost}
⏳ Pending: {pending}

**Win Rate:** {win_rate:.1f}%
**Daily PnL:** ${pnl/100:+.2f}
{stop_loss_section}

**Streaks:**
Current: {streak_text}
🏆 Best Win: {streaks['longest_win']}
💀 Worst Loss: {streaks['longest_loss']}

---
_Autotrader v2 • Black-Scholes Model_"""
    
    # Write to file for cron
    with open(OUTPUT_FILE, 'w') as f:
        f.write(summary)
    
    print(summary)
    return summary

if __name__ == "__main__":
    analyze_daily()
