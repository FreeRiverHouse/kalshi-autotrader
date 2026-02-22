#!/usr/bin/env python3
"""
Kalshi v3 Settlement Tracker

Tracks settlement of sports/parlay markets by querying Kalshi API directly
for market results. Works with ANY Kalshi market (not just crypto).

Usage:
  python kalshi-v3-settlement-tracker.py           # Process all pending trades
  python kalshi-v3-settlement-tracker.py --stats    # Show stats only
  python kalshi-v3-settlement-tracker.py --loop 600 # Check every 10 minutes
"""

import json
import sys
import time
import base64
import argparse
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# Reuse credentials from autotrader
PROJECT_ROOT = Path(__file__).parent.parent
TRADES_FILE = PROJECT_ROOT / "data" / "trading" / "kalshi-v3-trades.jsonl"
SETTLEMENTS_FILE = PROJECT_ROOT / "data" / "trading" / "kalshi-v3-settlements.json"

# Credentials — loaded from environment or .kalshi-private-key.pem
API_KEY_ID = os.environ.get("KALSHI_API_KEY_ID", "4308d1ca-585e-4b73-be82-5c0968b9a59a")
_key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.kalshi-private-key.pem')
if os.path.exists(_key_file):
    with open(_key_file) as _f:
        PRIVATE_KEY = _f.read().strip()
elif os.environ.get("KALSHI_PRIVATE_KEY"):
    PRIVATE_KEY = os.environ["KALSHI_PRIVATE_KEY"]
else:
    print("❌ Kalshi private key not found!")
    sys.exit(1)

BASE_URL = "https://api.elections.kalshi.com"


def sign_request(method: str, path: str, timestamp: str) -> str:
    private_key = serialization.load_pem_private_key(PRIVATE_KEY.encode(), password=None)
    message = f"{timestamp}{method}{path}".encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')


def kalshi_api(method: str, path: str) -> dict:
    url = f"{BASE_URL}{path}"
    timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
    signature = sign_request(method, path.split('?')[0], timestamp)
    headers = {
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code >= 400:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def get_market_result(ticker: str) -> dict:
    """Query Kalshi API for a market's settlement result."""
    result = kalshi_api("GET", f"/trade-api/v2/markets/{ticker}")
    if "error" in result:
        return {"status": "error", "error": result["error"]}
    
    market = result.get("market", result)
    status = market.get("status", "")
    result_val = market.get("result", "")
    
    if result_val in ("yes", "no"):
        return {
            "status": "settled",
            "result": result_val,
            "market_status": status,
        }
    elif status in ("finalized", "settled"):
        return {
            "status": "settled",
            "result": result_val or "unknown",
            "market_status": status,
        }
    else:
        return {
            "status": "active",
            "market_status": status,
        }


def load_settlements() -> dict:
    if SETTLEMENTS_FILE.exists():
        try:
            with open(SETTLEMENTS_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {"trades": {}, "summary": {"wins": 0, "losses": 0, "pending": 0, "total_pnl_cents": 0}}


def save_settlements(data: dict):
    SETTLEMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTLEMENTS_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def determine_outcome(action: str, result: str, contracts: int, price_cents: int) -> dict:
    """
    Determine win/loss for a trade.
    
    action: BUY_YES or BUY_NO
    result: 'yes' or 'no' (market settlement)
    contracts: number of contracts
    price_cents: price per contract in cents
    """
    cost_cents = contracts * price_cents
    
    if action == "BUY_YES":
        if result == "yes":
            payout_cents = contracts * 100
            return {"won": True, "pnl_cents": payout_cents - cost_cents, "cost_cents": cost_cents}
        else:
            return {"won": False, "pnl_cents": -cost_cents, "cost_cents": cost_cents}
    elif action == "BUY_NO":
        if result == "no":
            payout_cents = contracts * 100
            return {"won": True, "pnl_cents": payout_cents - cost_cents, "cost_cents": cost_cents}
        else:
            return {"won": False, "pnl_cents": -cost_cents, "cost_cents": cost_cents}
    
    return {"won": False, "pnl_cents": 0, "cost_cents": 0}


def process_settlements():
    """Check all traded tickers against Kalshi API for settlement."""
    if not TRADES_FILE.exists():
        print(f"Trade file not found: {TRADES_FILE}")
        return
    
    # Load existing settlements
    settlements = load_settlements()
    if "trades" not in settlements:
        settlements["trades"] = {}
    if "summary" not in settlements:
        settlements["summary"] = {"wins": 0, "losses": 0, "pending": 0, "total_pnl_cents": 0}
    
    # Load all unique executed trades (dedup: take FIRST trade per ticker)
    trades_by_ticker = {}
    with open(TRADES_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("action") in ("BUY_YES", "BUY_NO"):
                    ticker = entry.get("ticker")
                    if ticker and ticker not in trades_by_ticker:
                        trades_by_ticker[ticker] = entry
            except json.JSONDecodeError:
                continue
    
    # Filter to unsettled tickers
    pending_tickers = {
        t: d for t, d in trades_by_ticker.items()
        if settlements["trades"].get(t, {}).get("status") != "settled"
    }
    
    print(f"\n{'='*60}")
    print(f"📊 Kalshi v3 Settlement Tracker")
    print(f"{'='*60}")
    print(f"Total unique tickers traded: {len(trades_by_ticker)}")
    print(f"Already settled: {len(trades_by_ticker) - len(pending_tickers)}")
    print(f"Pending to check: {len(pending_tickers)}")
    
    if not pending_tickers:
        print("\nAll trades settled! Showing stats...")
        show_stats(settlements)
        return
    
    newly_settled = 0
    still_pending = 0
    errors = 0
    
    for ticker, trade in pending_tickers.items():
        result = get_market_result(ticker)
        
        if result["status"] == "settled":
            market_result = result["result"]
            action = trade["action"]
            contracts = trade.get("contracts", 1)
            price_cents = trade.get("price_cents", 50)
            
            outcome = determine_outcome(action, market_result, contracts, price_cents)
            
            settlements["trades"][ticker] = {
                "status": "settled",
                "market_result": market_result,
                "action": action,
                "contracts": contracts,
                "price_cents": price_cents,
                "won": outcome["won"],
                "pnl_cents": outcome["pnl_cents"],
                "cost_cents": outcome["cost_cents"],
                "title": trade.get("title", ""),
                "edge": trade.get("edge", 0),
                "settled_at": datetime.now(timezone.utc).isoformat(),
            }
            
            # Update summary
            if outcome["won"]:
                settlements["summary"]["wins"] += 1
            else:
                settlements["summary"]["losses"] += 1
            settlements["summary"]["total_pnl_cents"] += outcome["pnl_cents"]
            
            emoji = "✅" if outcome["won"] else "❌"
            print(f"  {emoji} {ticker[:50]}... → {market_result.upper()} (we: {action}) PnL: ${outcome['pnl_cents']/100:.2f}")
            newly_settled += 1
            
        elif result["status"] == "active":
            still_pending += 1
        else:
            errors += 1
        
        time.sleep(0.3)  # Rate limit
    
    settlements["summary"]["pending"] = still_pending
    settlements["summary"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    
    save_settlements(settlements)
    
    print(f"\n  Newly settled: {newly_settled}")
    print(f"  Still pending: {still_pending}")
    print(f"  Errors: {errors}")
    
    show_stats(settlements)


def show_stats(settlements: dict = None):
    """Show detailed statistics."""
    if settlements is None:
        settlements = load_settlements()
    
    settled = {k: v for k, v in settlements.get("trades", {}).items() if v.get("status") == "settled"}
    
    if not settled:
        print("\nNo settled trades yet.")
        return
    
    wins = sum(1 for v in settled.values() if v.get("won"))
    losses = sum(1 for v in settled.values() if not v.get("won"))
    total = wins + losses
    total_pnl = sum(v.get("pnl_cents", 0) for v in settled.values())
    
    print(f"\n{'='*60}")
    print(f"📈 SETTLEMENT STATS ({total} settled)")
    print(f"{'='*60}")
    print(f"  Wins:     {wins}")
    print(f"  Losses:   {losses}")
    print(f"  Win Rate: {wins/total*100:.1f}%" if total > 0 else "  Win Rate: N/A")
    print(f"  PnL:      ${total_pnl/100:.2f}")
    
    # Breakdown by action
    for action in ["BUY_YES", "BUY_NO"]:
        sub = {k: v for k, v in settled.items() if v.get("action") == action}
        if sub:
            w = sum(1 for v in sub.values() if v.get("won"))
            l = len(sub) - w
            pnl = sum(v.get("pnl_cents", 0) for v in sub.values())
            print(f"\n  {action}: {len(sub)} trades, {w}W/{l}L ({w/len(sub)*100:.0f}% WR), PnL: ${pnl/100:.2f}")
    
    # Breakdown by edge bucket
    print(f"\n  Edge buckets:")
    for lo, hi, label in [(0, 0.03, "<3%"), (0.03, 0.05, "3-5%"), (0.05, 0.10, "5-10%"), (0.10, 1.0, ">10%")]:
        sub = {k: v for k, v in settled.items() if lo <= v.get("edge", 0) < hi}
        if sub:
            w = sum(1 for v in sub.values() if v.get("won"))
            pnl = sum(v.get("pnl_cents", 0) for v in sub.values())
            print(f"    {label}: {len(sub)} trades, {w}W/{len(sub)-w}L ({w/len(sub)*100:.0f}% WR), PnL: ${pnl/100:.2f}")
    
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Kalshi v3 Settlement Tracker")
    parser.add_argument("--stats", action="store_true", help="Show stats only")
    parser.add_argument("--loop", type=int, default=0, help="Loop interval in seconds")
    args = parser.parse_args()
    
    if args.stats:
        show_stats()
        return
    
    if args.loop > 0:
        print(f"🔄 Loop mode: checking every {args.loop}s")
        while True:
            try:
                process_settlements()
                print(f"\n⏰ Next check in {args.loop}s...")
                time.sleep(args.loop)
            except KeyboardInterrupt:
                print("\n👋 Stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(30)
    else:
        process_settlements()


if __name__ == "__main__":
    main()
