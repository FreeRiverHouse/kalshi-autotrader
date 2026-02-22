#!/usr/bin/env python3
"""
Hourly snapshot of trading stats for trend analysis.
Saves snapshots to data/trading/snapshots/YYYY-MM-DD.jsonl
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

TRADES_FILE = Path('/Users/mattia/Projects/Onde/scripts/kalshi-trades.jsonl')
SNAPSHOTS_DIR = Path('/Users/mattia/Projects/Onde/data/trading/snapshots')

def load_trades():
    """Load all trades from JSONL file."""
    trades = []
    if not TRADES_FILE.exists():
        return trades
    
    with open(TRADES_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return trades

def calculate_stats(trades):
    """Calculate trading statistics."""
    if not trades:
        return {
            'total_trades': 0,
            'won': 0,
            'lost': 0,
            'pending': 0,
            'win_rate': 0.0,
            'pnl': 0.0,
            'btc_trades': 0,
            'eth_trades': 0,
            'yes_trades': 0,
            'no_trades': 0,
        }
    
    won = sum(1 for t in trades if t.get('result_status') == 'won')
    lost = sum(1 for t in trades if t.get('result_status') == 'lost')
    pending = len(trades) - won - lost
    settled = won + lost
    
    # Calculate PnL
    pnl = 0.0
    for t in trades:
        price = t.get('price', 0) or 0
        contracts = t.get('contracts', 1) or 1
        status = t.get('result_status', '')
        
        if status == 'won':
            pnl += (100 - price) * contracts / 100
        elif status == 'lost':
            pnl -= price * contracts / 100
    
    # Count by asset
    btc_trades = sum(1 for t in trades if t.get('asset', 'BTC') == 'BTC' or 'KXBTCD' in t.get('ticker', ''))
    eth_trades = sum(1 for t in trades if t.get('asset') == 'ETH' or 'KXETHD' in t.get('ticker', ''))
    
    # Count by side
    yes_trades = sum(1 for t in trades if t.get('side', '').upper() == 'YES')
    no_trades = sum(1 for t in trades if t.get('side', '').upper() == 'NO')
    
    return {
        'total_trades': len(trades),
        'won': won,
        'lost': lost,
        'pending': pending,
        'win_rate': round((won / settled * 100) if settled > 0 else 0, 2),
        'pnl': round(pnl, 2),
        'btc_trades': btc_trades,
        'eth_trades': eth_trades,
        'yes_trades': yes_trades,
        'no_trades': no_trades,
    }

def get_today_trades(trades):
    """Filter trades from today (UTC)."""
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    return [t for t in trades if t.get('timestamp', '').startswith(today)]

def save_snapshot(stats, today_stats):
    """Save snapshot to daily JSONL file."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    now = datetime.now(timezone.utc)
    date_str = now.strftime('%Y-%m-%d')
    snapshot_file = SNAPSHOTS_DIR / f'{date_str}.jsonl'
    
    snapshot = {
        'timestamp': now.isoformat(),
        'hour': now.hour,
        'all_time': stats,
        'today': today_stats,
    }
    
    with open(snapshot_file, 'a') as f:
        f.write(json.dumps(snapshot) + '\n')
    
    print(f"Saved snapshot to {snapshot_file}")
    print(f"  All-time: {stats['total_trades']} trades, {stats['win_rate']}% win rate, ${stats['pnl']:.2f} PnL")
    print(f"  Today: {today_stats['total_trades']} trades, {today_stats['win_rate']}% win rate, ${today_stats['pnl']:.2f} PnL")

def main():
    trades = load_trades()
    today_trades = get_today_trades(trades)
    
    all_time_stats = calculate_stats(trades)
    today_stats = calculate_stats(today_trades)
    
    save_snapshot(all_time_stats, today_stats)

if __name__ == '__main__':
    main()
