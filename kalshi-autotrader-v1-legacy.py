#!/usr/bin/env python3
"""
Kalshi AutoTrader - Automated BTC/ETH Trading Bot
Inspired by the Polymarket Clawdbot success ($100 → $347 overnight)

Features:
- Real-time price monitoring
- Volatility detection
- Sentiment analysis (Fear & Greed Index)
- Automated trading via Kalshi API
- Risk management with Kelly criterion

Usage:
    python kalshi-autotrader.py              # Run in monitor mode
    python kalshi-autotrader.py --live       # Run with live trading
    python kalshi-autotrader.py --backtest   # Backtest strategy
"""

import requests
import json
import sys
import time
import os
from datetime import datetime, timezone, timedelta
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import statistics
from pathlib import Path

# ============== CONFIG ==============
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

# Paper/Dry-run mode — logs trades but does NOT place real orders
DRY_RUN = True  # 🔴 PAPER MODE — set to False to enable real trading

# Trading parameters
MIN_EDGE = 0.15  # 15% minimum edge to trade (conservative!)
MAX_EDGE = 0.20  # TRADE-005: 20% max edge — edges above this are model errors, not real alpha
MAX_POSITION_PCT = 0.05  # Max 5% of portfolio per position (smaller bets!)
MAX_POSITIONS = 50  # More positions allowed
KELLY_FRACTION = 0.08  # Conservative Kelly for risk management (was 0.15)
MIN_BET_CENTS = 5  # Minimum bet size (micro!)
VOLATILITY_WINDOW = 10  # Minutes for volatility calc
MIN_TIME_TO_EXPIRY_MINUTES = 30  # Skip markets expiring in less than 30 min

# Alert settings
LOW_BALANCE_THRESHOLD_CENTS = 500  # $5.00
LOW_BALANCE_ALERT_FILE = Path(__file__).parent / "kalshi-low-balance.alert"
LOW_BALANCE_ALERT_COOLDOWN = 3600  # 1 hour between alerts

# Daily loss limit settings
DAILY_LOSS_LIMIT_CENTS = 100  # $1.00 max daily loss
DAILY_LOSS_PAUSE_FILE = Path(__file__).parent / "kalshi-daily-pause.json"

# Logging/tracking
TRADE_LOG_FILE = "scripts/kalshi-trades.jsonl"  # Track all decisions
SKIP_LOG_FILE = "scripts/kalshi-skips.jsonl"  # Track skipped opportunities for pattern analysis
LAST_LOW_BALANCE_ALERT = None  # Track when we last alerted

# ============== LOGGING FUNCTIONS ==============

def log_decision(decision_type: str, ticker: str, details: dict):
    """Log trading decision with full context"""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": decision_type,  # "trade", "skip", "opportunity", "error"
        "ticker": ticker,
        **details
    }
    try:
        log_path = Path(TRADE_LOG_FILE)
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to log: {e}")


def log_skip(ticker: str, reason: str, details: dict):
    """Log skipped opportunity to separate file for pattern analysis.
    
    This helps identify:
    - What edges we're consistently missing
    - If our MIN_EDGE is too conservative
    - Which strikes/assets have more near-misses
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "reason": reason,
        **details
    }
    try:
        log_path = Path(SKIP_LOG_FILE)
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to log skip: {e}")


def get_trade_stats() -> dict:
    """Calculate win/loss stats from trade log"""
    stats = {"total": 0, "wins": 0, "losses": 0, "pending": 0, "profit_cents": 0}
    try:
        log_path = Path(TRADE_LOG_FILE)
        if not log_path.exists():
            return stats
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("type") == "trade":
                    stats["total"] += 1
                    status = entry.get("result_status", "pending")
                    if status == "win":
                        stats["wins"] += 1
                        stats["profit_cents"] += entry.get("profit_cents", 0)
                    elif status == "loss":
                        stats["losses"] += 1
                        stats["profit_cents"] -= entry.get("cost_cents", 0)
                    else:
                        stats["pending"] += 1
    except:
        pass
    return stats


def calculate_daily_pnl() -> dict:
    """Calculate today's PnL from trade log. Returns dict with spent, won, net_pnl in cents."""
    today = datetime.now(timezone.utc).date()
    result = {"spent_cents": 0, "won_cents": 0, "trades_today": 0, "wins": 0, "losses": 0}
    
    try:
        log_path = Path(TRADE_LOG_FILE)
        if not log_path.exists():
            return result
        
        with open(log_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("type") != "trade":
                        continue
                    
                    # Parse timestamp
                    ts_str = entry.get("timestamp", "")
                    if not ts_str:
                        continue
                    trade_date = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).date()
                    
                    # Only count today's trades
                    if trade_date != today:
                        continue
                    
                    result["trades_today"] += 1
                    cost = entry.get("cost_cents", 0)
                    result["spent_cents"] += cost
                    
                    status = entry.get("result_status", "pending")
                    if status == "won":
                        result["wins"] += 1
                        # Won amount = contracts * (100 - price) for NO bets
                        contracts = entry.get("contracts", 0)
                        price = entry.get("price_cents", 0)
                        result["won_cents"] += contracts * (100 - price)
                    elif status == "lost":
                        result["losses"] += 1
                except:
                    continue
    except:
        pass
    
    result["net_pnl_cents"] = result["won_cents"] - result["spent_cents"]
    return result


def check_daily_loss_limit() -> tuple:
    """Check if daily loss limit has been hit. Returns (is_paused, pnl_info)."""
    pnl = calculate_daily_pnl()
    net_pnl = pnl["net_pnl_cents"]
    
    # If we've lost more than the limit, pause
    if net_pnl < -DAILY_LOSS_LIMIT_CENTS:
        pause_info = {
            "paused_at": datetime.now(timezone.utc).isoformat(),
            "reason": "daily_loss_limit",
            "net_pnl_cents": net_pnl,
            "limit_cents": -DAILY_LOSS_LIMIT_CENTS,
            "trades_today": pnl["trades_today"],
            "message": f"🛑 PAUSED: Daily loss ${abs(net_pnl)/100:.2f} exceeds limit ${DAILY_LOSS_LIMIT_CENTS/100:.2f}"
        }
        try:
            with open(DAILY_LOSS_PAUSE_FILE, "w") as f:
                json.dump(pause_info, f, indent=2)
        except:
            pass
        return True, pnl
    
    # Check if pause file exists from earlier today
    if DAILY_LOSS_PAUSE_FILE.exists():
        try:
            with open(DAILY_LOSS_PAUSE_FILE) as f:
                pause_data = json.load(f)
            pause_date = datetime.fromisoformat(pause_data.get("paused_at", "")).date()
            today = datetime.now(timezone.utc).date()
            if pause_date == today:
                # Still paused today
                return True, pnl
            else:
                # New day, remove pause file
                DAILY_LOSS_PAUSE_FILE.unlink()
        except:
            DAILY_LOSS_PAUSE_FILE.unlink()  # Corrupted, remove
    
    return False, pnl


def check_low_balance_alert(cash_cents: int) -> bool:
    """Check if balance is low and create alert file if needed. Returns True if alert created."""
    global LAST_LOW_BALANCE_ALERT
    
    if cash_cents >= LOW_BALANCE_THRESHOLD_CENTS:
        # Balance OK - remove alert file if exists
        if LOW_BALANCE_ALERT_FILE.exists():
            LOW_BALANCE_ALERT_FILE.unlink()
        return False
    
    # Balance is low - check cooldown
    now = time.time()
    if LAST_LOW_BALANCE_ALERT and (now - LAST_LOW_BALANCE_ALERT) < LOW_BALANCE_ALERT_COOLDOWN:
        return False  # Already alerted recently
    
    # Write alert file
    alert_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cash_cents": cash_cents,
        "threshold_cents": LOW_BALANCE_THRESHOLD_CENTS,
        "message": f"🚨 LOW BALANCE ALERT! Cash ${cash_cents/100:.2f} < ${LOW_BALANCE_THRESHOLD_CENTS/100:.2f}"
    }
    try:
        with open(LOW_BALANCE_ALERT_FILE, "w") as f:
            json.dump(alert_data, f, indent=2)
        LAST_LOW_BALANCE_ALERT = now
        print(f"\n🚨 LOW BALANCE ALERT! Written to {LOW_BALANCE_ALERT_FILE}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to write balance alert: {e}")
        return False


def check_time_to_expiry(market: dict) -> tuple:
    """Check if market has enough time to expiry. Returns (is_ok, minutes_left)"""
    try:
        close_time_str = market.get("close_time") or market.get("expiration_time")
        if not close_time_str:
            return True, 999  # No expiry info, allow
        
        # Parse ISO timestamp
        close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        minutes_left = (close_time - now).total_seconds() / 60
        
        return minutes_left >= MIN_TIME_TO_EXPIRY_MINUTES, minutes_left
    except:
        return True, 999  # On error, allow


# ============== API FUNCTIONS ==============

def sign_request(method: str, path: str, timestamp: str) -> str:
    """Sign request with RSA-PSS"""
    private_key = serialization.load_pem_private_key(PRIVATE_KEY.encode(), password=None)
    message = f"{timestamp}{method}{path}".encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')


def api_request(method: str, path: str, body: dict = None) -> dict:
    """Make authenticated API request"""
    timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
    signature = sign_request(method, path, timestamp)
    headers = {
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}{path}"
    
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            resp = requests.post(url, headers=headers, json=body, timeout=10)
        else:
            raise ValueError(f"Unknown method: {method}")
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def get_balance() -> dict:
    """Get account balance"""
    return api_request("GET", "/trade-api/v2/portfolio/balance")


def get_positions() -> list:
    """Get current positions"""
    result = api_request("GET", "/trade-api/v2/portfolio/positions")
    return result.get("market_positions", [])


def search_markets(series: str = None, limit: int = 50) -> list:
    """Search for markets"""
    path = f"/trade-api/v2/markets?limit={limit}&status=open"
    if series:
        path += f"&series_ticker={series}"
    result = api_request("GET", path)
    return result.get("markets", [])


def place_order(ticker: str, side: str, count: int, price_cents: int) -> dict:
    """Place a limit order"""
    body = {
        "ticker": ticker,
        "action": "buy",
        "side": side,
        "count": count,
        "type": "limit",
        "yes_price": price_cents if side == "yes" else 100 - price_cents
    }
    return api_request("POST", "/trade-api/v2/portfolio/orders", body)


# ============== DATA FUNCTIONS ==============

def get_crypto_prices() -> dict:
    """Get current BTC/ETH prices"""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd",
            timeout=5
        )
        data = resp.json()
        return {
            "btc": data["bitcoin"]["usd"],
            "eth": data["ethereum"]["usd"]
        }
    except:
        return None


def get_fear_greed_index() -> dict:
    """Get crypto Fear & Greed Index (sentiment)"""
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        data = resp.json()
        return {
            "value": int(data["data"][0]["value"]),
            "classification": data["data"][0]["value_classification"]
        }
    except:
        return {"value": 50, "classification": "Neutral"}


def get_btc_volatility() -> float:
    """Get BTC 24h volatility from price history"""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1",
            timeout=10
        )
        data = resp.json()
        prices = [p[1] for p in data["prices"]]
        if len(prices) < 2:
            return 0.02
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return statistics.stdev(returns) * (24 ** 0.5)  # Annualize to daily
    except:
        return 0.02  # Default 2% volatility


# ============== STRATEGY FUNCTIONS ==============

def calculate_implied_prob(yes_price: int) -> float:
    """Convert yes price (cents) to implied probability"""
    return yes_price / 100.0


def calculate_edge(our_prob: float, market_prob: float, side: str) -> float:
    """Calculate our edge vs market"""
    if side == "yes":
        return our_prob - market_prob
    else:
        return (1 - our_prob) - (1 - market_prob)


def kelly_size(edge: float, odds: float, bankroll: float) -> int:
    """Calculate Kelly criterion bet size in cents"""
    if edge <= 0 or odds <= 0:
        return 0
    
    # Kelly formula: f = (bp - q) / b
    # where b = odds-1, p = win prob, q = 1-p
    p = edge + 0.5  # Our estimated win probability
    q = 1 - p
    b = odds
    
    kelly_pct = (b * p - q) / b
    kelly_pct = max(0, min(kelly_pct, 1))  # Bound between 0-1
    
    # Apply fraction and position limit
    bet_pct = kelly_pct * KELLY_FRACTION
    bet_pct = min(bet_pct, MAX_POSITION_PCT)
    
    bet_cents = int(bankroll * bet_pct)
    return max(MIN_BET_CENTS, bet_cents)


def get_btc_momentum() -> float:
    """Get BTC momentum (% change over last 4 hours). Positive = uptrend."""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1",
            timeout=10
        )
        data = resp.json()
        prices = [p[1] for p in data["prices"]]
        if len(prices) < 20:
            return 0.0
        # Compare now vs 4 hours ago (roughly 16 data points with 15-min intervals)
        recent = prices[-1]
        old = prices[-min(16, len(prices))]
        momentum = (recent - old) / old
        return momentum
    except:
        return 0.0  # No momentum data


def estimate_btc_prob(current_price: float, strike: float, hours_to_expiry: float, volatility: float, sentiment: int, momentum: float = 0.0) -> float:
    """
    Estimate probability BTC will be above strike at expiry
    Uses Black-Scholes inspired model with volatility and drift adjustment
    
    Key insight: With 2% daily vol, BTC can easily move ±2-3% in a day.
    Cheap NO bets aren't free money - they're priced cheap because market expects UP.
    """
    import math
    
    # Distance to strike as % of current price
    distance_pct = (strike - current_price) / current_price
    
    # Expected move based on volatility (sigma * sqrt(time))
    # Assuming hours_to_expiry=24 means daily vol applies directly
    time_factor = min(hours_to_expiry / 24, 1.0)  # Cap at 1 day
    expected_move = volatility * math.sqrt(time_factor)
    
    # How many standard deviations is the strike from current price?
    # Positive = strike is above current price (need upward move)
    if expected_move > 0.001:
        z_score = distance_pct / expected_move
    else:
        z_score = distance_pct * 100  # Fallback
    
    # Add drift adjustment for momentum (positive momentum = expect higher prices)
    # Scale momentum effect: +1% 4hr momentum → shift expected value up ~0.5%
    drift = momentum * 0.5
    z_score_adjusted = z_score - (drift / expected_move if expected_move > 0.001 else 0)
    
    # Sentiment adjustment: Greed (>50) = bullish, Fear (<50) = bearish
    # Max ±0.3 z-score adjustment
    sentiment_adj = (sentiment - 50) / 150
    z_score_adjusted -= sentiment_adj
    
    # Convert z-score to probability using approximate normal CDF
    # P(BTC > strike) = P(Z > z_score) = 1 - Phi(z_score)
    # Using approximation: Phi(z) ≈ 1/(1 + exp(-1.7 * z))
    prob_below_strike = 1 / (1 + math.exp(-1.7 * z_score_adjusted))
    prob_above_strike = 1 - prob_below_strike
    
    # Bound probabilities
    return max(0.05, min(0.95, prob_above_strike))


def find_opportunities(markets: list, btc_price: float, eth_price: float, volatility: float, sentiment: int, momentum: float = 0.0) -> list:
    """Find trading opportunities in the markets"""
    opportunities = []
    skipped_expiry = 0
    skipped_low_edge = 0
    
    # Higher edge required for NO bets - they're often traps during rallies
    MIN_EDGE_NO = max(MIN_EDGE, 0.25)  # At least 25% edge for NO
    
    for m in markets:
        ticker = m.get("ticker", "")
        yes_bid = m.get("yes_bid", 0)
        yes_ask = m.get("yes_ask", 0)
        subtitle = m.get("subtitle", "")
        
        if not yes_bid or not yes_ask:
            continue
        
        # Check time to expiry
        expiry_ok, minutes_left = check_time_to_expiry(m)
        if not expiry_ok:
            skipped_expiry += 1
            log_skip(ticker, "too_close_to_expiry", {
                "minutes_left": round(minutes_left, 1),
                "min_required": MIN_TIME_TO_EXPIRY_MINUTES,
                "subtitle": subtitle
            })
            continue
        
        # Parse BTC markets
        if "KXBTCD" in ticker and "$" in subtitle:
            try:
                # Extract strike price from subtitle like "$88,750 or above"
                strike_str = subtitle.split("$")[1].split(" ")[0].replace(",", "")
                strike = float(strike_str)
                
                # Estimate our probability (now with momentum!)
                our_prob = estimate_btc_prob(btc_price, strike, 24, volatility, sentiment, momentum)
                market_prob_yes = calculate_implied_prob(yes_ask)
                market_prob_no = calculate_implied_prob(100 - yes_bid)
                
                # Check YES opportunity
                edge_yes = our_prob - market_prob_yes
                if edge_yes > MIN_EDGE:
                    opp = {
                        "ticker": ticker,
                        "side": "yes",
                        "price": yes_ask,
                        "edge": edge_yes,
                        "our_prob": our_prob,
                        "market_prob": market_prob_yes,
                        "strike": strike,
                        "current": btc_price,
                        "asset": "BTC",
                        "minutes_to_expiry": round(minutes_left, 1)
                    }
                    opportunities.append(opp)
                    log_decision("opportunity", ticker, {
                        "side": "yes", "edge": round(edge_yes, 4),
                        "our_prob": round(our_prob, 3), "market_prob": round(market_prob_yes, 3),
                        "price": yes_ask, "strike": strike, "btc_price": btc_price
                    })
                else:
                    skipped_low_edge += 1
                    # Log detailed skip info for pattern analysis
                    log_skip(ticker, "low_edge_yes", {
                        "side": "yes",
                        "edge": round(edge_yes, 4),
                        "edge_needed": MIN_EDGE,
                        "edge_gap": round(MIN_EDGE - edge_yes, 4),
                        "our_prob": round(our_prob, 3),
                        "market_prob": round(market_prob_yes, 3),
                        "price": yes_ask,
                        "strike": strike,
                        "current_price": btc_price,
                        "asset": "BTC",
                        "minutes_to_expiry": round(minutes_left, 1)
                    })
                
                # Check NO opportunity (higher edge required - NO bets are contrarian)
                edge_no = (1 - our_prob) - market_prob_no
                if edge_no > MIN_EDGE_NO:
                    opp = {
                        "ticker": ticker,
                        "side": "no",
                        "price": 100 - yes_bid,
                        "edge": edge_no,
                        "our_prob": 1 - our_prob,
                        "market_prob": market_prob_no,
                        "strike": strike,
                        "current": btc_price,
                        "asset": "BTC",
                        "minutes_to_expiry": round(minutes_left, 1)
                    }
                    opportunities.append(opp)
                    log_decision("opportunity", ticker, {
                        "side": "no", "edge": round(edge_no, 4),
                        "our_prob": round(1 - our_prob, 3), "market_prob": round(market_prob_no, 3),
                        "price": 100 - yes_bid, "strike": strike, "btc_price": btc_price
                    })
                else:
                    skipped_low_edge += 1
                    # Log detailed skip info for pattern analysis
                    log_skip(ticker, "low_edge_no", {
                        "side": "no",
                        "edge": round(edge_no, 4),
                        "edge_needed": MIN_EDGE_NO,
                        "edge_gap": round(MIN_EDGE_NO - edge_no, 4),
                        "our_prob": round(1 - our_prob, 3),
                        "market_prob": round(market_prob_no, 3),
                        "price": 100 - yes_bid,
                        "strike": strike,
                        "current_price": btc_price,
                        "asset": "BTC",
                        "minutes_to_expiry": round(minutes_left, 1)
                    })
            except:
                continue
    
    # Log summary
    if skipped_expiry or skipped_low_edge:
        log_decision("scan_summary", "ALL", {
            "skipped_expiry": skipped_expiry,
            "skipped_low_edge": skipped_low_edge,
            "opportunities_found": len(opportunities),
            "min_edge": MIN_EDGE,
            "min_time_to_expiry": MIN_TIME_TO_EXPIRY_MINUTES
        })
        print(f"   ⏰ Skipped {skipped_expiry} markets too close to expiry")
        print(f"   📉 Skipped {skipped_low_edge} markets with edge < {MIN_EDGE*100:.0f}%")
    
    # TRADE-005: Filter out suspiciously high edges (likely model errors)
    # Backtest found edges >25% were false positives. Cap at MAX_EDGE.
    before_cap = len(opportunities)
    opportunities = [o for o in opportunities if o.get("edge", 0) <= MAX_EDGE]
    if before_cap > len(opportunities):
        capped = before_cap - len(opportunities)
        print(f"   ⚠️ Edge cap: filtered {capped} trades with edge >{MAX_EDGE*100:.0f}% (model error)")
    
    # Sort by edge (best first)
    opportunities.sort(key=lambda x: x["edge"], reverse=True)
    return opportunities


# ============== MAIN TRADING LOOP ==============

def trading_cycle(live_mode: bool = False):
    """Run one trading cycle"""
    print(f"\n{'='*60}")
    print(f"🤖 KALSHI AUTOTRADER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Strategy: Kelly={KELLY_FRACTION}, MinEdge={MIN_EDGE*100:.0f}%, MinExpiry={MIN_TIME_TO_EXPIRY_MINUTES}min")
    print(f"{'='*60}")
    
    # Show trade stats
    stats = get_trade_stats()
    if stats["total"] > 0:
        win_rate = stats["wins"] / max(1, stats["wins"] + stats["losses"]) * 100
        print(f"\n📊 Trade History: {stats['total']} total | {stats['wins']}W/{stats['losses']}L | {win_rate:.0f}% win rate")
        print(f"   Pending: {stats['pending']} | P/L: ${stats['profit_cents']/100:+.2f}")
    
    # Get account status
    balance = get_balance()
    cash_cents = balance.get("balance", 0)
    portfolio_cents = balance.get("portfolio_value", 0)
    print(f"\n💰 Cash: ${cash_cents/100:.2f}")
    print(f"📊 Portfolio: ${portfolio_cents/100:.2f}")
    
    # Check for low balance alert
    check_low_balance_alert(cash_cents)
    
    if cash_cents < MIN_BET_CENTS:
        print("❌ Insufficient cash for trading")
        return
    
    # Get market data
    prices = get_crypto_prices()
    if not prices:
        print("❌ Failed to get crypto prices")
        return
    
    btc = prices["btc"]
    eth = prices["eth"]
    print(f"\n📈 BTC: ${btc:,.0f}")
    print(f"📈 ETH: ${eth:,.0f}")
    
    # Get sentiment
    fng = get_fear_greed_index()
    sentiment = fng["value"]
    print(f"😱 Fear & Greed: {sentiment} ({fng['classification']})")
    
    # Get volatility
    volatility = get_btc_volatility()
    print(f"📉 BTC Volatility (24h): {volatility*100:.1f}%")
    
    # Get momentum (4h trend)
    momentum = get_btc_momentum()
    trend = "📈 UP" if momentum > 0.005 else "📉 DOWN" if momentum < -0.005 else "➡️ FLAT"
    print(f"🔄 BTC Momentum (4h): {momentum*100:+.2f}% {trend}")
    
    # Get current positions
    positions = get_positions()
    position_tickers = [p.get("ticker") for p in positions]
    print(f"\n📋 Current positions: {len(positions)}")
    
    if len(positions) >= MAX_POSITIONS:
        print("⚠️ Max positions reached")
        return
    
    # Search BTC markets
    btc_markets = search_markets(series="KXBTCD")
    print(f"🔍 Found {len(btc_markets)} BTC markets")
    
    # Find opportunities (now momentum-aware!)
    opportunities = find_opportunities(btc_markets, btc, eth, volatility, sentiment, momentum)
    print(f"🎯 Found {len(opportunities)} opportunities (YES edge>{MIN_EDGE*100:.0f}%, NO edge>25%)")
    
    if not opportunities:
        print("No good opportunities right now")
        return
    
    # Show top opportunities
    print("\n🏆 Top Opportunities:")
    for i, opp in enumerate(opportunities[:5]):
        print(f"  {i+1}. {opp['ticker']}")
        print(f"     {opp['side'].upper()} @ {opp['price']}¢ | Edge: {opp['edge']*100:.1f}%")
        print(f"     Strike: ${opp['strike']:,.0f} | Current: ${opp['current']:,.0f}")
    
    if not live_mode:
        print("\n⚠️ DRY RUN - No trades executed (use --live for real trading)")
        return
    
    # Execute best trade
    best = opportunities[0]
    if best["ticker"] in position_tickers:
        print(f"⏭️ Already have position in {best['ticker']}")
        if len(opportunities) > 1:
            best = opportunities[1]
        else:
            return
    
    # Calculate bet size
    odds = (100 - best["price"]) / best["price"]  # Potential profit / cost
    bet_cents = kelly_size(best["edge"], odds, cash_cents)
    contracts = bet_cents // best["price"]
    
    if contracts < 1:
        print("❌ Bet size too small")
        return
    
    print(f"\n🚀 EXECUTING TRADE:")
    print(f"   {best['side'].upper()} {contracts} contracts @ {best['price']}¢")
    print(f"   Ticker: {best['ticker']}")
    print(f"   Edge: {best['edge']*100:.1f}%")
    print(f"   Time to expiry: {best.get('minutes_to_expiry', '?')} min")
    
    if DRY_RUN:
        print(f"📝 [PAPER MODE] Would place order — NOT executing")
        result = {"order": {"status": "paper", "taker_fill_cost": contracts * best["price"]}}
    else:
        result = place_order(best["ticker"], best["side"], contracts, best["price"])
    
    # Build trade reason explanation
    momentum_dir = "bullish" if momentum > 0.005 else "bearish" if momentum < -0.005 else "neutral"
    edge_pct = best["edge"] * 100
    reason_parts = [f"{edge_pct:.1f}% edge"]
    if momentum_dir != "neutral":
        reason_parts.append(f"{momentum_dir} momentum ({momentum*100:+.1f}%)")
    if sentiment < 30:
        reason_parts.append(f"extreme fear ({sentiment})")
    elif sentiment > 70:
        reason_parts.append(f"extreme greed ({sentiment})")
    if volatility > 0.03:
        reason_parts.append(f"high vol ({volatility*100:.1f}%)")
    trade_reason = " | ".join(reason_parts)
    
    order = result.get("order", {})
    trade_log = {
        "side": best["side"],
        "contracts": contracts,
        "price_cents": best["price"],
        "edge": round(best["edge"], 4),
        "our_prob": round(best["our_prob"], 3),
        "market_prob": round(best["market_prob"], 3),
        "strike": best.get("strike"),
        "current_price": best.get("current"),
        "minutes_to_expiry": best.get("minutes_to_expiry"),
        "order_status": order.get("status", "unknown"),
        "cost_cents": order.get("taker_fill_cost", 0),
        # Context for why this trade was taken
        "reason": trade_reason,
        "momentum": round(momentum, 4),
        "volatility": round(volatility, 4),
        "sentiment": sentiment,
        "asset": best.get("asset", "BTC")
    }
    
    if order.get("status") == "executed":
        print(f"✅ EXECUTED! Cost: ${order.get('taker_fill_cost', 0)/100:.2f}")
        trade_log["result_status"] = "pending"  # Will be updated when resolved
        log_decision("trade", best["ticker"], trade_log)
    else:
        print(f"📋 Order status: {order.get('status', 'unknown')}")
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            trade_log["error"] = result["error"]
        log_decision("trade_attempt", best["ticker"], trade_log)


def run_autotrader(live_mode: bool = False, interval_seconds: int = 300):
    """Run the autotrader continuously"""
    if DRY_RUN:
        live_mode = False  # DRY_RUN config overrides --live flag
    print("🤖 Starting Kalshi AutoTrader...")
    if DRY_RUN:
        print("   ⚠️  *** PAPER MODE *** — NO real orders will be placed!")
    print(f"   Mode: {'LIVE' if live_mode else 'PAPER / DRY RUN'}")
    print(f"   Interval: {interval_seconds}s")
    print(f"   Daily loss limit: ${DAILY_LOSS_LIMIT_CENTS/100:.2f}")
    print("   Press Ctrl+C to stop\n")
    
    while True:
        try:
            # Check daily loss limit before each cycle
            is_paused, pnl = check_daily_loss_limit()
            if is_paused:
                print(f"\n🛑 TRADING PAUSED - Daily loss limit reached!")
                print(f"   Today's PnL: ${pnl['net_pnl_cents']/100:+.2f}")
                print(f"   Trades today: {pnl['trades_today']} ({pnl['wins']}W/{pnl['losses']}L)")
                print(f"   Limit: -${DAILY_LOSS_LIMIT_CENTS/100:.2f}")
                print(f"   Trading will resume tomorrow (UTC midnight)")
                print(f"\n⏰ Checking again in {interval_seconds}s...")
                time.sleep(interval_seconds)
                continue
            
            # Show daily PnL status
            print(f"\n📊 Daily PnL: ${pnl['net_pnl_cents']/100:+.2f} | Trades: {pnl['trades_today']} | Limit: -${DAILY_LOSS_LIMIT_CENTS/100:.2f}")
            
            trading_cycle(live_mode)
            print(f"\n⏰ Next cycle in {interval_seconds}s...")
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\n\n👋 Stopping autotrader...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(60)  # Wait 1 min on error


# ============== MAIN ==============

if __name__ == "__main__":
    if "--live" in sys.argv:
        run_autotrader(live_mode=True, interval_seconds=300)
    elif "--backtest" in sys.argv:
        print("Backtest mode not implemented yet")
    else:
        # Single dry run cycle
        trading_cycle(live_mode=False)
