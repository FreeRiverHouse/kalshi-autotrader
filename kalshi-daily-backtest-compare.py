#!/usr/bin/env python3
"""
Kalshi Daily Backtest vs Live Comparison
Compares live trading results with what backtest would have predicted.

Run daily via cron to track model drift.

Usage: python3 scripts/kalshi-daily-backtest-compare.py [date]
"""

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

DATA_DIR = Path("data/trading")


def load_trades_for_date(date_str):
    """Load trades for a specific date"""
    trades_file = DATA_DIR / f"kalshi-trades-{date_str}.jsonl"
    trades = []
    if trades_file.exists():
        with open(trades_file) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    trades.append(d)
                except:
                    continue
    return trades


def load_forecasts():
    """Load opus forecasts"""
    forecasts = []
    f = DATA_DIR / "opus-forecasts.jsonl"
    if f.exists():
        with open(f) as fh:
            for line in fh:
                try:
                    forecasts.append(json.loads(line.strip()))
                except:
                    continue
    return forecasts


def analyze_day(date_str):
    """Analyze a single day's performance"""
    trades = load_trades_for_date(date_str)

    # Filter to actual executions and opportunities
    opportunities = [t for t in trades if t.get("type") == "opportunity"]
    executions = [t for t in trades if t.get("type") == "execution"]
    scans = [t for t in trades if t.get("type") == "scan_summary"]

    print(f"📊 Daily Report: {date_str}")
    print(f"{'=' * 50}")

    if not trades:
        print("  No trade data for this date")
        return

    print(f"  Scans:         {len(scans)}")
    print(f"  Opportunities: {len(opportunities)}")
    print(f"  Executions:    {len(executions)}")

    if opportunities:
        edges = [t["edge"] for t in opportunities if "edge" in t]
        if edges:
            print(f"  Avg edge:      {sum(edges)/len(edges):+.1%}")
            print(f"  Max edge:      {max(edges):+.1%}")

    # Edge distribution for this day
    if opportunities:
        yes_count = sum(1 for t in opportunities if t.get("side") == "yes")
        no_count = sum(1 for t in opportunities if t.get("side") == "no")
        print(f"  YES opps:      {yes_count}")
        print(f"  NO opps:       {no_count}")

    # Unique tickers
    tickers = set(t.get("ticker", "") for t in opportunities)
    print(f"  Unique markets: {len(tickers)}")

    # Top opportunities
    if opportunities:
        top = sorted(opportunities, key=lambda t: t.get("edge", 0), reverse=True)[:5]
        print(f"\n  Top 5 opportunities:")
        for t in top:
            ticker = t.get("ticker", "?")[:35]
            edge = t.get("edge", 0)
            side = t.get("side", "?")
            prob = t.get("our_prob", 0)
            print(f"    {edge:+.1%} | {side:3s} | prob={prob:.0%} | {ticker}")


def main():
    import sys

    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        # Yesterday by default
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")

    analyze_day(date_str)

    # Also show comparison with today
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today_str != date_str:
        print(f"\n{'=' * 50}")
        analyze_day(today_str)

    # Load forecasts and compare
    forecasts = load_forecasts()
    if forecasts:
        opus = [f for f in forecasts if f.get("type") == "forecast"]
        settled = [f for f in forecasts if f.get("type") == "settled"]
        print(f"\n📈 Opus Forecasts: {len(opus)} total, {len(settled)} settled")


if __name__ == "__main__":
    main()
