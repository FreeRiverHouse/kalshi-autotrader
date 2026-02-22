#!/usr/bin/env python3
"""
Kalshi Multi-Agent Trading Bot (Grok Style)

Architecture:
  FORECASTER → CRITIC → TRADER
  
- Forecaster: Estimates true probability using market data + logic
- Critic: Identifies flaws, missing context, overconfidence
- Trader: Makes final BUY/SKIP decision with position sizing

Usage:
    python kalshi-multiagent.py              # Analyze top opportunities
    python kalshi-multiagent.py --execute    # Actually trade
"""

import requests
import json
import sys
from datetime import datetime, timezone
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# === CONFIG ===
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

# Trading params
MIN_CONFIDENCE = 0.60  # Minimum 60% confidence to trade
MIN_EDGE = 0.10  # Minimum 10% edge
BEAST_MODE_EDGE = 0.30  # 30%+ edge triggers beast mode
MAX_POSITION_PCT = 0.05  # Max 5% portfolio per position
KELLY_FRACTION = 0.25  # Quarter Kelly


def sign_request(method, path, timestamp):
    private_key = serialization.load_pem_private_key(PRIVATE_KEY.encode(), password=None)
    message = f"{timestamp}{method}{path}".encode('utf-8')
    signature = private_key.sign(message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
    return base64.b64encode(signature).decode('utf-8')


def api_get(path):
    timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
    signature = sign_request("GET", path, timestamp)
    headers = {"KALSHI-ACCESS-KEY": API_KEY_ID, "KALSHI-ACCESS-SIGNATURE": signature, "KALSHI-ACCESS-TIMESTAMP": timestamp}
    try:
        return requests.get(f"{BASE_URL}{path}", headers=headers, timeout=10).json()
    except:
        return {}


def api_post(path, body):
    timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
    signature = sign_request("POST", path, timestamp)
    headers = {"KALSHI-ACCESS-KEY": API_KEY_ID, "KALSHI-ACCESS-SIGNATURE": signature, "KALSHI-ACCESS-TIMESTAMP": timestamp, "Content-Type": "application/json"}
    try:
        return requests.post(f"{BASE_URL}{path}", headers=headers, json=body, timeout=10).json()
    except:
        return {}


def get_crypto_prices():
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd,usd_24h_change", timeout=5)
        data = resp.json()
        return {
            "btc": data["bitcoin"]["usd"],
            "btc_change": data["bitcoin"].get("usd_24h_change", 0),
            "eth": data["ethereum"]["usd"],
            "eth_change": data["ethereum"].get("usd_24h_change", 0)
        }
    except:
        return {"btc": 0, "eth": 0, "btc_change": 0, "eth_change": 0}


# =============================================================================
# AGENT 1: FORECASTER
# =============================================================================

def forecaster_agent(market, prices):
    """
    FORECASTER: Estimates true probability of outcome
    
    Uses:
    - Current asset price vs threshold
    - Price momentum (24h change)
    - Time to expiry
    - Historical volatility patterns
    """
    ticker = market.get("ticker", "")
    
    # Determine asset type and threshold
    if "KXBTCD" in ticker:
        asset = "btc"
        asset_price = prices["btc"]
        momentum = prices["btc_change"]
        volatility_daily = 0.03  # ~3% daily BTC volatility
    elif "KXETHD" in ticker:
        asset = "eth"
        asset_price = prices["eth"]
        momentum = prices["eth_change"]
        volatility_daily = 0.04  # ~4% daily ETH volatility
    else:
        return None
    
    # Extract threshold from ticker
    if "-T" not in ticker:
        return None
    try:
        threshold = float(ticker.split("-T")[1])
    except:
        return None
    
    # Calculate distance from threshold (in standard deviations)
    distance_pct = (asset_price - threshold) / asset_price
    distance_std = distance_pct / volatility_daily
    
    # Base probability from current price position
    # If price is 2 std above threshold, very likely to stay above
    if distance_std > 3:
        base_prob = 0.95
    elif distance_std > 2:
        base_prob = 0.90
    elif distance_std > 1:
        base_prob = 0.80
    elif distance_std > 0.5:
        base_prob = 0.70
    elif distance_std > 0:
        base_prob = 0.60
    elif distance_std > -0.5:
        base_prob = 0.40
    elif distance_std > -1:
        base_prob = 0.30
    elif distance_std > -2:
        base_prob = 0.20
    else:
        base_prob = 0.10
    
    # Momentum adjustment
    # Positive momentum → higher chance of staying above
    momentum_factor = 1 + (momentum / 100) * 0.1  # 1% momentum = 0.1% prob shift
    adjusted_prob = min(0.98, max(0.02, base_prob * momentum_factor))
    
    # Confidence based on how clear the signal is
    confidence = min(1.0, abs(distance_std) / 2)
    
    return {
        "ticker": ticker,
        "asset": asset,
        "asset_price": asset_price,
        "threshold": threshold,
        "distance_std": distance_std,
        "momentum": momentum,
        "estimated_prob_yes": adjusted_prob,
        "confidence": confidence,
        "reasoning": f"{asset.upper()} ${asset_price:,.0f} is {distance_std:.1f}σ from ${threshold:,.0f}, momentum {momentum:+.1f}%"
    }


# =============================================================================
# AGENT 2: CRITIC
# =============================================================================

def critic_agent(forecast):
    """
    CRITIC: Identifies flaws and adjusts confidence
    
    Checks:
    - Overconfidence on volatile assets
    - Missing context (news events, market conditions)
    - Edge cases and tail risks
    """
    issues = []
    confidence_penalty = 0
    
    # Check for overconfidence
    if forecast["confidence"] > 0.9 and abs(forecast["distance_std"]) < 2:
        issues.append("Overconfident: distance not extreme enough")
        confidence_penalty += 0.1
    
    # Check momentum reliability
    if abs(forecast["momentum"]) > 5:
        issues.append(f"High momentum ({forecast['momentum']:+.1f}%) may reverse")
        confidence_penalty += 0.05
    
    # Check for close-to-threshold positions
    if abs(forecast["distance_std"]) < 0.5:
        issues.append("Too close to threshold - high uncertainty")
        confidence_penalty += 0.15
    
    # Crypto-specific risks
    if forecast["asset"] in ["btc", "eth"]:
        issues.append("Crypto volatile - unexpected moves possible")
        confidence_penalty += 0.05
    
    # Adjust confidence
    adjusted_confidence = max(0.1, forecast["confidence"] - confidence_penalty)
    
    # Determine if we should proceed
    proceed = len(issues) < 3 and adjusted_confidence >= 0.4
    
    return {
        **forecast,
        "critic_issues": issues,
        "adjusted_confidence": adjusted_confidence,
        "critic_proceed": proceed,
        "critic_summary": f"{len(issues)} issues, confidence {adjusted_confidence:.0%}"
    }


# =============================================================================
# AGENT 3: TRADER
# =============================================================================

def trader_agent(analysis, market, bankroll):
    """
    TRADER: Makes final decision and calculates position size
    
    Decides:
    - BUY YES / BUY NO / SKIP
    - Position size (Kelly-based)
    - Beast mode activation
    """
    if not analysis["critic_proceed"]:
        return {
            **analysis,
            "decision": "SKIP",
            "reason": "Critic rejected",
            "position_size": 0
        }
    
    # Get market prices
    yes_ask = market.get("yes_ask", 0)
    yes_bid = market.get("yes_bid", 0)
    
    if yes_ask == 0:
        return {**analysis, "decision": "SKIP", "reason": "No ask price", "position_size": 0}
    
    # Determine direction based on our estimated probability
    estimated_prob = analysis["estimated_prob_yes"]
    implied_prob_yes = yes_ask / 100
    implied_prob_no = (100 - yes_bid) / 100 if yes_bid > 0 else 1.0
    
    # Calculate edge
    edge_yes = estimated_prob - implied_prob_yes
    edge_no = (1 - estimated_prob) - implied_prob_no
    
    # Choose best direction
    if edge_yes > edge_no and edge_yes > MIN_EDGE:
        direction = "YES"
        edge = edge_yes
        price = yes_ask
    elif edge_no > edge_yes and edge_no > MIN_EDGE:
        direction = "NO"
        edge = edge_no
        price = 100 - yes_bid if yes_bid > 0 else 100
    else:
        return {**analysis, "decision": "SKIP", "reason": f"Insufficient edge (YES:{edge_yes:.1%}, NO:{edge_no:.1%})", "position_size": 0}
    
    # Check confidence threshold
    if analysis["adjusted_confidence"] < MIN_CONFIDENCE:
        return {**analysis, "decision": "SKIP", "reason": f"Low confidence ({analysis['adjusted_confidence']:.0%})", "position_size": 0}
    
    # Calculate position size (Kelly)
    b = (100 - price) / price  # odds
    p = estimated_prob if direction == "YES" else (1 - estimated_prob)
    q = 1 - p
    kelly = (b * p - q) / b if b > 0 else 0
    kelly = max(0, kelly)
    
    # Apply safety factor
    safe_kelly = kelly * KELLY_FRACTION
    
    # Cap at max position size
    max_bet = bankroll * MAX_POSITION_PCT
    position_size = min(max_bet, safe_kelly * bankroll)
    
    # Beast mode for high edge
    beast_mode = edge >= BEAST_MODE_EDGE
    if beast_mode:
        position_size = min(bankroll * 0.10, position_size * 1.5)  # 1.5x size, max 10%
    
    # Calculate contracts
    contracts = max(1, int(position_size * 100 / price))
    
    return {
        **analysis,
        "decision": f"BUY {direction}",
        "direction": direction,
        "price": price,
        "edge": edge,
        "kelly_fraction": safe_kelly,
        "position_size": position_size,
        "contracts": contracts,
        "beast_mode": beast_mode,
        "reason": f"Edge {edge:.1%}, Confidence {analysis['adjusted_confidence']:.0%}"
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def analyze_market(market, prices, bankroll):
    """Run full multi-agent analysis on a market"""
    # Step 1: Forecaster
    forecast = forecaster_agent(market, prices)
    if not forecast:
        return None
    
    # Step 2: Critic
    critique = critic_agent(forecast)
    
    # Step 3: Trader
    decision = trader_agent(critique, market, bankroll)
    
    return decision


def run():
    execute_mode = "--execute" in sys.argv
    
    print("=" * 60)
    print("KALSHI MULTI-AGENT BOT (Grok Style)")
    print("=" * 60)
    
    # Get current state
    prices = get_crypto_prices()
    print(f"\n📈 BTC: ${prices['btc']:,.0f} ({prices['btc_change']:+.1f}%)")
    print(f"📈 ETH: ${prices['eth']:,.0f} ({prices['eth_change']:+.1f}%)")
    
    balance = api_get("/trade-api/v2/portfolio/balance")
    cash = balance.get("balance", 0) / 100
    portfolio = balance.get("portfolio_value", 0) / 100
    print(f"\n💰 Cash: ${cash:.2f} | Portfolio: ${portfolio:.2f}")
    
    if cash < 0.10:
        print("⚠️ Insufficient cash")
        return
    
    # Get markets
    opportunities = []
    
    for series in ["KXBTCD", "KXETHD"]:
        markets = api_get(f"/trade-api/v2/markets?limit=30&status=open&series_ticker={series}")
        for m in markets.get("markets", []):
            result = analyze_market(m, prices, cash)
            if result and result["decision"].startswith("BUY"):
                opportunities.append(result)
    
    # Sort by edge
    opportunities.sort(key=lambda x: x.get("edge", 0), reverse=True)
    
    print(f"\n🎯 ANALYSIS RESULTS ({len(opportunities)} opportunities)")
    print("-" * 60)
    
    for opp in opportunities[:5]:
        beast = "🔥" if opp.get("beast_mode") else ""
        print(f"\n{beast} {opp['decision']} @ {opp['price']}¢ on {opp['ticker'][-25:]}")
        print(f"   📊 {opp['reasoning']}")
        print(f"   🎯 Edge: {opp['edge']:.1%} | Confidence: {opp['adjusted_confidence']:.0%}")
        print(f"   💰 Size: ${opp['position_size']:.2f} ({opp['contracts']} contracts)")
        if opp.get("critic_issues"):
            print(f"   ⚠️ Issues: {', '.join(opp['critic_issues'][:2])}")
    
    # Execute if requested
    if execute_mode and opportunities:
        print("\n" + "=" * 60)
        print("🚀 EXECUTING TRADES")
        print("=" * 60)
        
        for opp in opportunities[:3]:
            if cash < 0.10:
                break
            
            order = {
                "ticker": opp["ticker"],
                "action": "buy",
                "side": opp["direction"].lower(),
                "count": opp["contracts"],
                "type": "limit",
                "yes_price": opp["price"] if opp["direction"] == "YES" else 100 - opp["price"]
            }
            
            result = api_post("/trade-api/v2/portfolio/orders", order)
            status = result.get("order", {}).get("status", "unknown")
            
            if status == "executed":
                cost = result.get("order", {}).get("taker_fill_cost", 0) / 100
                cash -= cost
                print(f"✅ {opp['decision']} @ {opp['price']}¢ | ${cost:.2f}")
            else:
                print(f"⏳ {status}: {opp['decision']}")
        
        # Final balance
        balance = api_get("/trade-api/v2/portfolio/balance")
        print(f"\n💰 Final: ${balance.get('balance', 0)/100:.2f} cash, ${balance.get('portfolio_value', 0)/100:.2f} portfolio")
    
    elif not execute_mode:
        print("\n💡 Run with --execute to place trades")


if __name__ == "__main__":
    run()
