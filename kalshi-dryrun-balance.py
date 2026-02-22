#!/usr/bin/env python3
"""
Paper Balance Tracker for Dry-Run Trading

Tracks virtual balance for dry-run trades to simulate real P&L.
Uses settlement prices from CoinGecko to determine trade outcomes.

Usage:
  python3 scripts/kalshi-dryrun-balance.py [--reset] [--status]
  
Options:
  --reset    Reset paper balance to $100
  --status   Show current paper balance status
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

# Config
SCRIPT_DIR = Path(__file__).parent
DRYRUN_LOG = SCRIPT_DIR / "kalshi-trades-dryrun.jsonl"
BALANCE_FILE = SCRIPT_DIR / "kalshi-dryrun-balance.json"
INITIAL_BALANCE = 10000  # $100.00 in cents

def load_balance():
    """Load paper balance state"""
    if BALANCE_FILE.exists():
        with open(BALANCE_FILE, 'r') as f:
            return json.load(f)
    return {
        "balance_cents": INITIAL_BALANCE,
        "initial_balance_cents": INITIAL_BALANCE,
        "total_trades": 0,
        "settled_trades": 0,
        "wins": 0,
        "losses": 0,
        "pending_trades": [],
        "pnl_cents": 0,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

def save_balance(state):
    """Save paper balance state"""
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(BALANCE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def get_settlement_price(asset, settle_time):
    """Get historical price for settlement"""
    # Parse settle time
    if isinstance(settle_time, str):
        settle_dt = datetime.fromisoformat(settle_time.replace('Z', '+00:00'))
    else:
        settle_dt = settle_time
    
    # Use CoinGecko historical API
    coin_id = "bitcoin" if asset == "BTC" else "ethereum"
    date_str = settle_dt.strftime("%d-%m-%Y")
    
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/history"
        params = {"date": date_str, "localization": "false"}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            price = data.get("market_data", {}).get("current_price", {}).get("usd")
            if price:
                return price
    except Exception as e:
        print(f"Warning: Could not fetch historical price: {e}")
    
    # Fallback: current price
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": coin_id, "vs_currencies": "usd"}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json().get(coin_id, {}).get("usd")
    except:
        pass
    
    return None

def parse_ticker(ticker):
    """Parse KXBTCD-25JAN30-B98000 → asset, strike, direction"""
    parts = ticker.split("-")
    if len(parts) < 3:
        return None, None, None
    
    asset = "BTC" if "KXBTCD" in parts[0] else "ETH"
    
    # Parse strike (B98000 = $98,000 boundary)
    strike_part = parts[-1]
    if strike_part.startswith("B"):
        strike = int(strike_part[1:])
    else:
        strike = None
    
    return asset, strike, "above"

def settle_trade(trade, state):
    """Settle a single trade based on actual price"""
    ticker = trade.get("ticker", "")
    asset, strike, _ = parse_ticker(ticker)
    
    if not asset or not strike:
        return None
    
    # Get settlement time (hourly contract expires at the hour)
    trade_time = datetime.fromisoformat(trade["timestamp"].replace('Z', '+00:00'))
    expiry_str = trade.get("expiry_time", "")
    
    if expiry_str:
        settle_time = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
    else:
        # Assume 1 hour from trade
        settle_time = trade_time + timedelta(hours=1)
    
    # Check if trade is settled (expiry passed)
    now = datetime.now(timezone.utc)
    if settle_time > now:
        return None  # Still pending
    
    # Get settlement price
    settle_price = get_settlement_price(asset, settle_time)
    if not settle_price:
        return None
    
    # Determine outcome
    price_above = settle_price >= strike
    side = trade.get("side", "no").lower()
    
    if side == "yes":
        won = price_above
    else:  # NO bet
        won = not price_above
    
    # Calculate P&L
    price_cents = trade.get("price_cents", 50)
    contracts = trade.get("contracts", 1)
    
    if won:
        profit_cents = (100 - price_cents) * contracts
        result = "won"
    else:
        profit_cents = -price_cents * contracts
        result = "lost"
    
    return {
        "result": result,
        "pnl_cents": profit_cents,
        "settle_price": settle_price,
        "strike": strike
    }

def process_dryrun_trades(state):
    """Process dry-run trade log and update balance"""
    if not DRYRUN_LOG.exists():
        print("No dry-run trades found")
        return state
    
    # Load trades
    processed_ids = set(t.get("id") for t in state.get("pending_trades", []))
    settled_ids = set()
    
    # Load all trades from log
    trades = []
    with open(DRYRUN_LOG, 'r') as f:
        for line in f:
            try:
                trade = json.loads(line.strip())
                trades.append(trade)
            except:
                continue
    
    print(f"Found {len(trades)} dry-run trades")
    
    new_pending = []
    newly_settled = 0
    
    for trade in trades:
        trade_id = trade.get("timestamp", "") + trade.get("ticker", "")
        
        # Skip already settled
        if trade.get("settled"):
            continue
        
        # Try to settle
        result = settle_trade(trade, state)
        
        if result:
            # Trade settled
            state["settled_trades"] += 1
            state["pnl_cents"] += result["pnl_cents"]
            state["balance_cents"] += result["pnl_cents"]
            
            if result["result"] == "won":
                state["wins"] += 1
            else:
                state["losses"] += 1
            
            newly_settled += 1
            print(f"  Settled: {trade.get('ticker')} → {result['result']} "
                  f"(P&L: ${result['pnl_cents']/100:.2f}, "
                  f"settle=${result['settle_price']:,.0f}, strike=${result['strike']:,})")
        else:
            # Still pending
            new_pending.append(trade)
    
    state["pending_trades"] = new_pending
    state["total_trades"] = len(trades)
    
    if newly_settled:
        print(f"\nSettled {newly_settled} trades this run")
    
    return state

def print_status(state):
    """Print current paper balance status"""
    balance = state["balance_cents"] / 100
    initial = state["initial_balance_cents"] / 100
    pnl = state["pnl_cents"] / 100
    pnl_pct = (pnl / initial * 100) if initial > 0 else 0
    
    total = state["settled_trades"]
    wins = state["wins"]
    losses = state["losses"]
    win_rate = (wins / total * 100) if total > 0 else 0
    
    pending = len(state.get("pending_trades", []))
    
    print("\n" + "="*50)
    print("📊 PAPER TRADING STATUS")
    print("="*50)
    print(f"💰 Balance:     ${balance:,.2f}")
    print(f"📈 Total P&L:   ${pnl:+,.2f} ({pnl_pct:+.1f}%)")
    print(f"📊 Win Rate:    {win_rate:.1f}% ({wins}W/{losses}L)")
    print(f"🎯 Trades:      {state['total_trades']} total, {total} settled, {pending} pending")
    print(f"⏱️  Updated:     {state['last_updated'][:19]}")
    print("="*50 + "\n")

def main():
    if "--reset" in sys.argv:
        state = {
            "balance_cents": INITIAL_BALANCE,
            "initial_balance_cents": INITIAL_BALANCE,
            "total_trades": 0,
            "settled_trades": 0,
            "wins": 0,
            "losses": 0,
            "pending_trades": [],
            "pnl_cents": 0,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        save_balance(state)
        print("✅ Paper balance reset to $100.00")
        return
    
    state = load_balance()
    
    if "--status" not in sys.argv:
        state = process_dryrun_trades(state)
        save_balance(state)
    
    print_status(state)

if __name__ == "__main__":
    main()
