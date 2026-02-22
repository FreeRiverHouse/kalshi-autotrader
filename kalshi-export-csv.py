#!/usr/bin/env python3
"""
Kalshi Trade Export to CSV
Export kalshi-trades.jsonl to CSV format for spreadsheet analysis.
"""

import json
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

TRADES_FILE = Path(__file__).parent / "kalshi-trades.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "exports"

def parse_timestamp(ts):
    """Parse ISO timestamp."""
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))

def get_pst_datetime(dt):
    """Convert datetime to PST datetime string."""
    pst_offset = timedelta(hours=-8)
    pst_time = dt + pst_offset
    return pst_time.strftime('%Y-%m-%d %H:%M:%S')

def export_trades(days=None, output_file=None):
    """Export trades to CSV.
    
    Args:
        days: Only export last N days (None = all)
        output_file: Custom output path (None = auto-generate)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    trades = []
    now = datetime.now().astimezone()
    cutoff = now - timedelta(days=days) if days else None
    
    with open(TRADES_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get('type') != 'trade':
                    continue
                    
                if entry.get('order_status') != 'executed':
                    continue
                
                trade_time = parse_timestamp(entry['timestamp'])
                if cutoff and trade_time < cutoff:
                    continue
                
                trades.append(entry)
            except (json.JSONDecodeError, KeyError):
                continue
    
    if not trades:
        print("No trades found!")
        return None
    
    # Determine output file
    if output_file:
        output_path = Path(output_file)
    else:
        suffix = f"-last{days}d" if days else "-all"
        output_path = OUTPUT_DIR / f"kalshi-trades{suffix}-{now.strftime('%Y%m%d')}.csv"
    
    # CSV columns
    fieldnames = [
        'timestamp_utc',
        'timestamp_pst',
        'ticker',
        'side',
        'contracts',
        'avg_price_cents',
        'cost_cents',
        'cost_usd',
        'result',
        'pnl_cents',
        'pnl_usd',
        'order_id',
        'fill_id',
    ]
    
    # Calculate PnL
    def calc_pnl(trade):
        result = trade.get('result_status', 'pending')
        cost = trade.get('cost_cents', 0)
        contracts = trade.get('contracts', 1)
        
        if result == 'won':
            return (contracts * 100) - cost
        elif result == 'lost':
            return -cost
        return 0
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for trade in sorted(trades, key=lambda x: x['timestamp']):
            pnl = calc_pnl(trade)
            cost = trade.get('cost_cents', 0)
            
            writer.writerow({
                'timestamp_utc': trade.get('timestamp', ''),
                'timestamp_pst': get_pst_datetime(parse_timestamp(trade['timestamp'])),
                'ticker': trade.get('ticker', ''),
                'side': trade.get('side', ''),
                'contracts': trade.get('contracts', 1),
                'avg_price_cents': trade.get('avg_price', 0),
                'cost_cents': cost,
                'cost_usd': f"{cost/100:.2f}",
                'result': trade.get('result_status', 'pending'),
                'pnl_cents': pnl,
                'pnl_usd': f"{pnl/100:.2f}",
                'order_id': trade.get('order_id', ''),
                'fill_id': trade.get('fill_id', ''),
            })
    
    print(f"✅ Exported {len(trades)} trades to: {output_path}")
    return output_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export Kalshi trades to CSV')
    parser.add_argument('--days', type=int, help='Only export last N days')
    parser.add_argument('--output', '-o', type=str, help='Custom output file path')
    args = parser.parse_args()
    
    export_trades(days=args.days, output_file=args.output)

if __name__ == "__main__":
    main()
