#!/usr/bin/env python3
"""
Kalshi Opus Trader - Direct trading script for use by Clawdbot/Opus
No external LLM needed - the agent IS the forecaster.

Usage: python3 scripts/kalshi-opus-trader.py scan     # Scan and show opportunities
       python3 scripts/kalshi-opus-trader.py trade TICKER yes|no PRICE COUNT  # Place order
       python3 scripts/kalshi-opus-trader.py positions  # Show current positions
       python3 scripts/kalshi-opus-trader.py balance    # Show balance
"""

import sys
import json
import requests
import base64
from datetime import datetime, timezone
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

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

def sign_request(method, path, timestamp):
    private_key = serialization.load_pem_private_key(PRIVATE_KEY.encode(), password=None)
    message = f"{timestamp}{method}{path}".encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')

def api(method, path, body=None):
    ts = str(int(datetime.now(timezone.utc).timestamp() * 1000))
    sig = sign_request(method, path, ts)
    headers = {
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json"
    }
    url = BASE_URL + path
    if method == "POST":
        r = requests.post(url, headers=headers, json=body, timeout=10)
    else:
        r = requests.get(url, headers=headers, timeout=10)
    return r.json()

def get_balance():
    bal = api("GET", "/trade-api/v2/portfolio/balance")
    return bal.get('balance', 0), bal.get('portfolio_value', 0)

def get_positions():
    pos = api("GET", "/trade-api/v2/portfolio/positions?limit=50&settlement_status=unsettled")
    return pos.get("market_positions", [])

def scan_events(event_tickers):
    """Scan specific events for tradeable markets"""
    all_markets = []
    for ticker in event_tickers:
        resp = requests.get(f'{BASE_URL}/trade-api/v2/events/{ticker}?with_nested_markets=true')
        if resp.status_code == 200:
            event = resp.json().get('event', {})
            markets = event.get('markets', [])
            for m in markets:
                vol = m.get('volume', 0) or 0
                yb = m.get('yes_bid', 0) or 0
                ya = m.get('yes_ask', 0) or 0
                oi = m.get('open_interest', 0) or 0
                if vol > 0 and ya > 0:
                    all_markets.append({
                        'ticker': m['ticker'],
                        'title': m.get('title', '')[:80],
                        'yes_bid': yb,
                        'yes_ask': ya,
                        'no_bid': m.get('no_bid', 0) or 0,
                        'no_ask': m.get('no_ask', 0) or 0,
                        'volume': vol,
                        'oi': oi,
                        'event': ticker,
                        'event_title': event.get('title', '')[:60],
                        'close': m.get('close_time', ''),
                    })
    return sorted(all_markets, key=lambda x: x['volume'], reverse=True)

def place_order(ticker, side, price_cents, count):
    """Place a limit order"""
    body = {
        "ticker": ticker,
        "action": "buy",
        "side": side,
        "type": "limit",
        "count": count,
    }
    if side == "yes":
        body["yes_price"] = price_cents
    else:
        body["no_price"] = price_cents
    
    result = api("POST", "/trade-api/v2/portfolio/orders", body)
    return result

# Series tickers to scan (auto-discover active events)
SERIES_TO_SCAN = [
    # Crypto
    'KXBTCD', 'KXBTC', 'KXETHD', 'KXETH', 'KXSOL',
    # Economics
    'KXCPI', 'KXFED', 'KXPPI', 'KXNFP', 'KXRETAILSALES',
    'KXUMICH', 'KXJOBLESS', 'KXPCEPRICE', 'KXGDP',
    # Weather
    'HIGHNY', 'HIGHLA', 'HIGHCHI', 'RAINNYC', 'RAINLA',
    # Politics/World
    'KXNEWPOPE', 'KXTRUMPAPPROVAL', 'KXGOVTSHUTDOWN',
    # Tech
    'KXDEEPSEEKR2RELEASE',
]

# Static event tickers (don't use series lookup)
STATIC_EVENTS = [
    'KXNEWPOPE-70', 'KXNEWPOPE-35',
    'KXCPI-26FEB', 'KXFED-26MAR',
]

def discover_events(series_list):
    """Discover active events from series tickers"""
    event_tickers = set()
    for series in series_list:
        try:
            resp = requests.get(f'{BASE_URL}/trade-api/v2/events?limit=3&series_ticker={series}&with_nested_markets=false', timeout=5)
            if resp.status_code == 200:
                for e in resp.json().get('events', []):
                    event_tickers.add(e['event_ticker'])
        except:
            pass
    return list(event_tickers)

DEFAULT_EVENTS = STATIC_EVENTS  # Will be extended by discover_events at runtime

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    
    if cmd == "discover":
        events = discover_events(SERIES_TO_SCAN)
        print(f"Discovered {len(events)} active events:")
        for e in sorted(events):
            print(f"  {e}")
        sys.exit(0)
    
    if cmd == "balance":
        bal, pv = get_balance()
        print(f"Balance: ${bal/100:.2f}")
        print(f"Portfolio value: ${pv/100:.2f}")
        
    elif cmd == "positions":
        positions = get_positions()
        if not positions:
            print("No open positions")
        for p in positions:
            print(json.dumps(p, indent=2))
            
    elif cmd == "scan":
        # Auto-discover active events then scan them
        discovered = discover_events(SERIES_TO_SCAN)
        all_events = list(set(STATIC_EVENTS + discovered))
        markets = scan_events(all_events)
        print(f"Found {len(markets)} markets\n")
        for m in markets[:30]:
            spread = m['yes_ask'] - m['yes_bid']
            print(f"YES:{m['yes_bid']:3d}/{m['yes_ask']:3d}¢ (sp:{spread:2d}) Vol:{m['volume']:>8,} OI:{m['oi']:>6,} | {m['event']}")
            print(f"  {m['title']}")
            print()
            
    elif cmd == "trade":
        if len(sys.argv) < 6:
            print("Usage: trade TICKER yes|no PRICE COUNT")
            sys.exit(1)
        ticker = sys.argv[2]
        side = sys.argv[3]
        price = int(sys.argv[4])
        count = int(sys.argv[5])
        
        print(f"Placing order: {side.upper()} {ticker} x{count} @ {price}¢")
        result = place_order(ticker, side, price, count)
        if "order" in result:
            o = result["order"]
            print(f"✅ Status: {o.get('status')}")
            print(f"   Filled: {o.get('fill_count')}/{o.get('initial_count')}")
        else:
            print(f"❌ Error: {json.dumps(result)[:300]}")
    else:
        print(f"Unknown command: {cmd}")
        print("Commands: scan, trade, positions, balance")
