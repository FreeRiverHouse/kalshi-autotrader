#!/usr/bin/env python3
"""
Kalshi Volume Anomaly Alert Script

Detects unusual trading volume:
- Alert if today > 2x 7-day average (high activity)
- Alert if today < 0.5x 7-day average (low activity)

Creates kalshi-volume-anomaly.alert for heartbeat pickup.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "trading"
ALERT_FILE = Path(__file__).parent / "kalshi-volume-anomaly.alert"
STATE_FILE = DATA_DIR / "volume-anomaly-state.json"
LOOKBACK_DAYS = 7
HIGH_THRESHOLD = 2.0  # 2x average = high volume
LOW_THRESHOLD = 0.5   # 0.5x average = low volume


def count_trades_for_date(date_str: str) -> int:
    """Count executed trades for a specific date."""
    log_file = DATA_DIR / f"kalshi-trades-{date_str}.jsonl"
    
    if not log_file.exists():
        return 0
    
    count = 0
    with open(log_file, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("type") == "trade" and entry.get("order_status") == "executed":
                    count += 1
            except json.JSONDecodeError:
                continue
    
    return count


def get_volume_history() -> dict:
    """Get trade counts for the last N days."""
    today = datetime.now()
    history = {}
    
    for i in range(LOOKBACK_DAYS + 1):  # +1 to include today
        date = today - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        history[date_str] = count_trades_for_date(date_str)
    
    return history


def load_state() -> dict:
    """Load previous alert state."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_alert_date": None, "last_alert_type": None}


def save_state(state: dict):
    """Save alert state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main():
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # Get volume history
    history = get_volume_history()
    today_volume = history.get(today_str, 0)
    
    # Calculate 7-day average (excluding today)
    past_volumes = [v for d, v in history.items() if d != today_str and v > 0]
    
    if not past_volumes:
        print("Not enough historical data for volume analysis")
        return
    
    avg_volume = sum(past_volumes) / len(past_volumes)
    
    print(f"📊 Volume Analysis:")
    print(f"   Today ({today_str}): {today_volume} trades")
    print(f"   7-day average: {avg_volume:.1f} trades")
    print(f"   Ratio: {today_volume / avg_volume:.2f}x" if avg_volume > 0 else "   Ratio: N/A")
    
    # Only check after some trading has occurred (not early morning)
    current_hour = datetime.now().hour
    if current_hour < 10 and today_volume < 10:
        print("   ⏰ Too early to assess volume anomaly")
        return
    
    # Load state
    state = load_state()
    
    # Determine if anomaly
    alert_type = None
    alert_msg = None
    
    if avg_volume > 0:
        ratio = today_volume / avg_volume
        
        if ratio >= HIGH_THRESHOLD:
            alert_type = "high"
            alert_msg = f"""📈 HIGH VOLUME ALERT

Today's trading volume is unusually high!

📊 Stats:
- Today: {today_volume} trades
- 7-day avg: {avg_volume:.1f} trades
- Ratio: {ratio:.2f}x average

📅 History:
"""
            for date, vol in sorted(history.items(), reverse=True)[:7]:
                marker = "👉" if date == today_str else "  "
                alert_msg += f"{marker} {date}: {vol} trades\n"
            
            alert_msg += """
🔍 Possible causes:
- High market volatility
- More opportunities found
- Algorithm change effects

Check autotrader logs for details."""

        elif ratio <= LOW_THRESHOLD and today_volume > 0:
            alert_type = "low"
            alert_msg = f"""📉 LOW VOLUME ALERT

Today's trading volume is unusually low!

📊 Stats:
- Today: {today_volume} trades
- 7-day avg: {avg_volume:.1f} trades
- Ratio: {ratio:.2f}x average

📅 History:
"""
            for date, vol in sorted(history.items(), reverse=True)[:7]:
                marker = "👉" if date == today_str else "  "
                alert_msg += f"{marker} {date}: {vol} trades\n"
            
            alert_msg += """
🔍 Possible causes:
- Low market volatility
- Few opportunities meeting criteria
- Potential autotrader issue

Verify autotrader is running correctly."""

    # Only alert once per day per type
    if alert_type:
        if state.get("last_alert_date") == today_str and state.get("last_alert_type") == alert_type:
            print(f"   ⏭️ Already alerted for {alert_type} volume today")
            return
        
        print(f"   🚨 Creating {alert_type} volume alert!")
        
        # Write alert file
        with open(ALERT_FILE, "w") as f:
            f.write(alert_msg)
        
        # Update state
        state["last_alert_date"] = today_str
        state["last_alert_type"] = alert_type
        save_state(state)
        
        print(f"   ✅ Alert written to {ALERT_FILE}")
    else:
        print("   ✅ Volume is normal")


if __name__ == "__main__":
    main()
