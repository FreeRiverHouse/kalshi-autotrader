#!/usr/bin/env python3
"""
Kalshi Forecast Tracker - Compare Opus forecasts vs actual market results.
Reads from data/trading/opus-forecasts.jsonl and checks settled markets.

Usage: python3 scripts/kalshi-forecast-tracker.py
"""

import json
import requests
from pathlib import Path
from datetime import datetime, timezone

BASE_URL = "https://api.elections.kalshi.com"
FORECASTS_FILE = Path("data/trading/opus-forecasts.jsonl")

def get_market_result(ticker):
    """Check if a market has settled and get the result"""
    try:
        resp = requests.get(f"{BASE_URL}/trade-api/v2/markets/{ticker}", timeout=5)
        if resp.status_code == 200:
            m = resp.json().get("market", {})
            return {
                "status": m.get("status", ""),
                "result": m.get("result", ""),
                "expiration_value": m.get("expiration_value", ""),
                "last_price": m.get("last_price", 0),
            }
    except:
        pass
    return None

def analyze_forecasts():
    """Load forecasts and check results"""
    if not FORECASTS_FILE.exists():
        print("No forecasts file found")
        return
    
    forecasts = []
    with open(FORECASTS_FILE) as f:
        for line in f:
            try:
                forecasts.append(json.loads(line.strip()))
            except:
                continue
    
    print(f"Loaded {len(forecasts)} forecast entries\n")
    
    # Track accuracy
    settled = 0
    correct = 0
    total_edge = 0
    total_pnl = 0
    
    for fc in forecasts:
        if fc.get("type") in ["btc_analysis"]:
            continue  # Skip analysis entries
        
        market = fc.get("market", "")
        if not market or "/" in market:
            continue
            
        title = fc.get("title", market)
        my_forecast = fc.get("my_forecast", None)
        market_price = fc.get("market_yes_price", 0)
        position = fc.get("position", "none")
        edge = fc.get("edge", 0)
        
        if my_forecast is None:
            continue
        
        # Check result
        result = get_market_result(market)
        
        status_icon = "⏳"
        if result and result["result"]:
            settled += 1
            won = result["result"] == "yes"
            
            if my_forecast > 0.5 and won:
                correct += 1
                status_icon = "✅"
            elif my_forecast <= 0.5 and not won:
                correct += 1
                status_icon = "✅"
            else:
                status_icon = "❌"
            
            # PnL calc if we had a position
            if "YES" in position:
                count = 1
                price = market_price
                try:
                    parts = position.split("x")
                    if len(parts) > 1:
                        count = int(parts[1].split("@")[0].strip())
                        price = int(parts[1].split("@")[1].strip().replace("c",""))
                except:
                    pass
                
                if won:
                    pnl = count * (100 - price)
                else:
                    pnl = -count * price
                total_pnl += pnl
        elif result:
            status_icon = "⏳" if result["status"] == "active" else "🔒"
        
        print(f"{status_icon} {title}")
        print(f"   My forecast: {my_forecast:.0%} | Market: {market_price}¢ | Edge: {edge:+.0%} | Position: {position}")
        if result and result["result"]:
            print(f"   Result: {result['result'].upper()}")
        print()
    
    if settled > 0:
        accuracy = correct / settled * 100
        print(f"\n{'='*60}")
        print(f"📊 FORECAST ACCURACY")
        print(f"   Settled: {settled} | Correct: {correct} | Accuracy: {accuracy:.1f}%")
        print(f"   Est. PnL: ${total_pnl/100:+.2f}")
        print(f"{'='*60}")
    else:
        print(f"\n⏳ No markets settled yet. Check back later.")

if __name__ == "__main__":
    analyze_forecasts()
