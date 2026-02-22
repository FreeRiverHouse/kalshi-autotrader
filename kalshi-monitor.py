#!/usr/bin/env python3
"""
Kalshi Trade Monitor - Check and report on trade results

Usage:
    python kalshi-monitor.py           # Full report
    python kalshi-monitor.py --update  # Update pending trades and report
    python kalshi-monitor.py --live    # Continuous monitoring
"""

import json
import time
import base64
import argparse
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import re

# Load credentials
script_dir = Path(__file__).parent
with open(script_dir / "kalshi-autotrader.py") as f:
    content = f.read()

API_KEY = re.search(r'API_KEY_ID = "([^"]+)"', content).group(1)
KEY_MATCH = re.search(r'PRIVATE_KEY = """(.+?)"""', content, re.DOTALL)
PRIVATE_KEY = KEY_MATCH.group(1).strip()
BASE_URL = "https://api.elections.kalshi.com"

TRADE_LOG_V1 = script_dir / "kalshi-trades.jsonl"
TRADE_LOG_V2 = script_dir / "kalshi-trades-v2.jsonl"


def sign_request(method, path, timestamp):
    key = serialization.load_pem_private_key(PRIVATE_KEY.encode(), password=None)
    message = f"{timestamp}{method}{path}".encode('utf-8')
    sig = key.sign(message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
    return base64.b64encode(sig).decode('utf-8')


def api_request(method, path):
    timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
    signature = sign_request(method, path.split('?')[0], timestamp)
    headers = {
        "KALSHI-ACCESS-KEY": API_KEY,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }
    r = requests.get(BASE_URL + path, headers=headers, timeout=10)
    return r.json()


def get_balance():
    return api_request("GET", "/trade-api/v2/portfolio/balance")


def get_positions():
    return api_request("GET", "/trade-api/v2/portfolio/positions").get("market_positions", [])


def get_fills(limit=200):
    return api_request("GET", f"/trade-api/v2/portfolio/fills?limit={limit}").get("fills", [])


def get_market(ticker):
    return api_request("GET", f"/trade-api/v2/markets/{ticker}").get("market", {})


def analyze_fills():
    """Analyze all fills and check settlement status"""
    fills = get_fills(200)
    
    # Group by ticker
    by_ticker = defaultdict(lambda: {"buys": 0, "cost": 0, "side": None})
    
    for f in fills:
        ticker = f["ticker"]
        if f.get("action") == "buy":
            by_ticker[ticker]["buys"] += f["count"]
            by_ticker[ticker]["cost"] += f["count"] * f["price"]
            by_ticker[ticker]["side"] = f.get("side")
    
    results = {"wins": 0, "losses": 0, "pending": 0, "pnl": 0, "at_risk": 0}
    details = []
    
    for ticker, data in sorted(by_ticker.items()):
        if data["buys"] == 0:
            continue
            
        market = get_market(ticker)
        status = market.get("status", "unknown")
        result = market.get("result")
        
        detail = {
            "ticker": ticker,
            "side": data["side"],
            "contracts": data["buys"],
            "cost": data["cost"],
            "status": status,
            "result": result
        }
        
        if status == "finalized" and result:
            we_won = (data["side"] == result)
            if we_won:
                pnl = (100 * data["buys"]) - data["cost"]
                results["wins"] += 1
                detail["outcome"] = "WIN"
            else:
                pnl = -data["cost"]
                results["losses"] += 1
                detail["outcome"] = "LOSS"
            results["pnl"] += pnl
            detail["pnl"] = pnl
        else:
            results["pending"] += 1
            results["at_risk"] += data["cost"]
            detail["outcome"] = "PENDING"
        
        details.append(detail)
    
    return results, details


def print_report():
    """Print full trade report"""
    print("=" * 70)
    print("📊 KALSHI TRADE MONITOR")
    print("=" * 70)
    
    # Balance
    bal = get_balance()
    cash = bal.get("balance", 0) / 100
    portfolio = bal.get("portfolio_value", 0) / 100
    print(f"\n💰 Account: ${cash:.2f} cash + ${portfolio:.2f} portfolio = ${cash+portfolio:.2f} total")
    
    # Positions
    positions = get_positions()
    print(f"📋 Open positions: {len(positions)}")
    
    # Analyze all trades
    print("\n🔍 Analyzing trades...")
    results, details = analyze_fills()
    
    # Summary
    total = results["wins"] + results["losses"]
    win_rate = results["wins"] / total * 100 if total > 0 else 0
    
    print(f"\n{'='*70}")
    print("📈 RESULTS SUMMARY")
    print("="*70)
    print(f"✅ Wins:     {results['wins']}")
    print(f"❌ Losses:   {results['losses']}")
    print(f"⏳ Pending:  {results['pending']}")
    print(f"📊 Win Rate: {win_rate:.1f}%")
    print(f"💰 P/L:      ${results['pnl']/100:+.2f}")
    print(f"⚠️  At Risk: ${results['at_risk']/100:.2f}")
    
    # Details by status
    print(f"\n{'='*70}")
    print("📋 SETTLED TRADES")
    print("="*70)
    
    settled = [d for d in details if d["outcome"] != "PENDING"]
    for d in sorted(settled, key=lambda x: x["ticker"]):
        emoji = "✅" if d["outcome"] == "WIN" else "❌"
        print(f"{emoji} {d['ticker']}: {d['side'].upper()} {d['contracts']}x → {d['result'].upper()} = ${d['pnl']/100:+.2f}")
    
    print(f"\n{'='*70}")
    print("⏳ PENDING TRADES")
    print("="*70)
    
    pending = [d for d in details if d["outcome"] == "PENDING"]
    for d in sorted(pending, key=lambda x: x["ticker"]):
        print(f"⏳ {d['ticker']}: {d['side'].upper()} {d['contracts']}x @ {d['cost']}¢")


def live_monitor(interval=60):
    """Continuous monitoring"""
    print("🔴 LIVE MONITORING (Ctrl+C to stop)")
    
    last_results = None
    while True:
        results, _ = analyze_fills()
        
        if last_results and results != last_results:
            # Something changed!
            if results["wins"] > last_results["wins"]:
                print(f"\n🎉 WIN! Total: {results['wins']}W/{results['losses']}L | P/L: ${results['pnl']/100:+.2f}")
            elif results["losses"] > last_results["losses"]:
                print(f"\n💔 Loss. Total: {results['wins']}W/{results['losses']}L | P/L: ${results['pnl']/100:+.2f}")
        
        last_results = results.copy()
        
        # Status line
        bal = get_balance()
        total = (bal.get("balance", 0) + bal.get("portfolio_value", 0)) / 100
        print(f"\r⏱️  {datetime.now().strftime('%H:%M:%S')} | ${total:.2f} | {results['wins']}W/{results['losses']}L | P/L: ${results['pnl']/100:+.2f}", end="", flush=True)
        
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Continuous monitoring")
    args = parser.parse_args()
    
    try:
        if args.live:
            live_monitor()
        else:
            print_report()
    except KeyboardInterrupt:
        print("\n\n👋 Bye!")
