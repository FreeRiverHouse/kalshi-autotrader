#!/usr/bin/env python3
"""
Kalshi Edge Distribution Analysis
Reads trade history and shows edge distribution to spot outliers.

Usage: python3 scripts/kalshi-edge-analysis.py [days]
"""

import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta

DATA_DIR = Path("data/trading")

def load_trades(days=30):
    """Load trade data from jsonl files"""
    trades = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    
    for f in sorted(DATA_DIR.glob("kalshi-trades-2026-*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    d = json.loads(line.strip())
                    if d.get("type") == "opportunity" and "edge" in d:
                        trades.append(d)
                except:
                    continue
    
    # Also check v3 trades
    v3_file = DATA_DIR / "kalshi-v3-trades.jsonl"
    if v3_file.exists():
        with open(v3_file) as fh:
            for line in fh:
                try:
                    d = json.loads(line.strip())
                    if "edge" in d:
                        trades.append(d)
                except:
                    continue
    
    return trades

def text_histogram(values, bins=20, width=50):
    """Simple text-based histogram"""
    if not values:
        print("  No data")
        return
    
    mn, mx = min(values), max(values)
    if mn == mx:
        print(f"  All values = {mn:.3f}")
        return
    
    bin_size = (mx - mn) / bins
    counts = [0] * bins
    
    for v in values:
        idx = min(int((v - mn) / bin_size), bins - 1)
        counts[idx] += 1
    
    max_count = max(counts)
    
    for i in range(bins):
        lo = mn + i * bin_size
        hi = lo + bin_size
        bar_len = int(counts[i] / max_count * width) if max_count > 0 else 0
        bar = "█" * bar_len
        label = f"{lo:+6.1%} to {hi:+6.1%}"
        print(f"  {label} | {bar} {counts[i]}")

def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    
    trades = load_trades(days)
    print(f"📊 Edge Distribution Analysis (last {days} days)")
    print(f"   Loaded {len(trades)} opportunities\n")
    
    if not trades:
        print("No trade data found")
        return
    
    edges = [t["edge"] for t in trades]
    sides = [t.get("side", "?") for t in trades]
    
    # Overall stats
    print(f"{'='*60}")
    print(f"  Mean edge:   {sum(edges)/len(edges):+.3f}")
    print(f"  Median:      {sorted(edges)[len(edges)//2]:+.3f}")
    print(f"  Min:         {min(edges):+.3f}")
    print(f"  Max:         {max(edges):+.3f}")
    print(f"  Std dev:     {(sum((e - sum(edges)/len(edges))**2 for e in edges)/len(edges))**0.5:.3f}")
    print(f"{'='*60}\n")
    
    # Edge histogram
    print("📈 Edge Distribution:")
    text_histogram(edges, bins=15)
    
    # By side
    yes_edges = [t["edge"] for t in trades if t.get("side") == "yes"]
    no_edges = [t["edge"] for t in trades if t.get("side") == "no"]
    
    if yes_edges:
        print(f"\n📗 YES edges ({len(yes_edges)} trades):")
        print(f"   Mean: {sum(yes_edges)/len(yes_edges):+.3f}")
        text_histogram(yes_edges, bins=10)
    
    if no_edges:
        print(f"\n📕 NO edges ({len(no_edges)} trades):")
        print(f"   Mean: {sum(no_edges)/len(no_edges):+.3f}")
        text_histogram(no_edges, bins=10)
    
    # Outliers (>2 std dev)
    mean = sum(edges) / len(edges)
    std = (sum((e - mean)**2 for e in edges) / len(edges)) ** 0.5
    outliers = [t for t in trades if abs(t["edge"] - mean) > 2 * std]
    
    if outliers:
        print(f"\n⚠️  Outliers (>{2*std:.3f} from mean): {len(outliers)}")
        for o in outliers[:10]:
            ticker = o.get("ticker", "?")[:40]
            print(f"   Edge: {o['edge']:+.3f} | {o.get('side','?'):3s} | {ticker}")
    
    # Edge by our_prob bucket
    print(f"\n📊 Edge by Confidence Bucket:")
    buckets = {"<30%": [], "30-50%": [], "50-70%": [], "70-90%": [], ">90%": []}
    for t in trades:
        p = t.get("our_prob", 0.5)
        if p < 0.3: buckets["<30%"].append(t["edge"])
        elif p < 0.5: buckets["30-50%"].append(t["edge"])
        elif p < 0.7: buckets["50-70%"].append(t["edge"])
        elif p < 0.9: buckets["70-90%"].append(t["edge"])
        else: buckets[">90%"].append(t["edge"])
    
    for bucket, vals in buckets.items():
        if vals:
            avg = sum(vals) / len(vals)
            print(f"   {bucket:>6s}: {len(vals):3d} trades, avg edge {avg:+.3f}")

if __name__ == "__main__":
    main()
