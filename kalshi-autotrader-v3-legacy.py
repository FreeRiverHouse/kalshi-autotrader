#!/usr/bin/env python3
"""
Kalshi AutoTrader v3 - Claude-as-Forecaster (Grok Strategy Clone)

Based on PredictionArena / ryanfrigo/kalshi-ai-trading-bot architecture:
  FORECASTER → CRITIC → TRADER

Uses Claude API to estimate true probability of ANY Kalshi market (not just crypto).
Multi-agent pipeline: Forecaster estimates, Critic validates, Trader executes.

When no LLM API key is available, falls back to a built-in heuristic forecaster
that uses market structure, sport-specific models, and known biases to estimate
true probabilities without any external API calls.

Author: Clawd
Date: 2026-02-15
"""

import os
import sys
import json
import time
import math
import re
import argparse
import base64
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ============== CONFIGURATION ==============

# Kalshi API credentials (reuse from v2)
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

# Claude API configuration
# Supports multiple providers: Anthropic direct, OpenRouter, or any OpenAI-compatible endpoint
def get_llm_config():
    """
    Get LLM configuration. Priority:
    1. ANTHROPIC_API_KEY env var → Anthropic direct
    2. OPENROUTER_API_KEY env var → OpenRouter
    3. LLM_API_KEY + LLM_BASE_URL env vars → custom endpoint
    4. ~/.clawdbot/.env.trading file
    5. Fallback instructions
    """
    # 1. Direct Anthropic key
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key and key.startswith("sk-ant-api"):
        return {
            "provider": "anthropic",
            "api_key": key,
            "base_url": "https://api.anthropic.com/v1/messages",
            "model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            "headers": {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        }
    
    # 2. OpenRouter
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return {
            "provider": "openrouter",
            "api_key": key,
            "base_url": "https://openrouter.ai/api/v1/chat/completions",
            "model": os.environ.get("CLAUDE_MODEL", "anthropic/claude-sonnet-4"),
            "headers": {
                "Authorization": f"Bearer {key}",
                "content-type": "application/json"
            }
        }
    
    # 3. Custom OpenAI-compatible endpoint
    key = os.environ.get("LLM_API_KEY")
    base = os.environ.get("LLM_BASE_URL")
    if key and base:
        return {
            "provider": "openai-compatible",
            "api_key": key,
            "base_url": base,
            "model": os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514"),
            "headers": {
                "Authorization": f"Bearer {key}",
                "content-type": "application/json"
            }
        }
    
    # 4. Check .env.trading file
    env_trading = Path.home() / ".clawdbot" / ".env.trading"
    if env_trading.exists():
        env_vars = {}
        with open(env_trading) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    env_vars[k.strip()] = v.strip()
        
        if "ANTHROPIC_API_KEY" in env_vars and env_vars["ANTHROPIC_API_KEY"].startswith("sk-ant-"):
            key = env_vars["ANTHROPIC_API_KEY"]
            auth_header = {"Authorization": f"Bearer {key}"} if key.startswith("sk-ant-oat") else {"x-api-key": key}
            return {
                "provider": "anthropic",
                "api_key": key,
                "base_url": "https://api.anthropic.com/v1/messages",
                "model": env_vars.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
                "headers": {
                    **auth_header,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
            }
        
        if "OPENROUTER_API_KEY" in env_vars:
            key = env_vars["OPENROUTER_API_KEY"]
            return {
                "provider": "openrouter",
                "api_key": key,
                "base_url": "https://openrouter.ai/api/v1/chat/completions",
                "model": env_vars.get("CLAUDE_MODEL", "anthropic/claude-sonnet-4"),
                "headers": {
                    "Authorization": f"Bearer {key}",
                    "content-type": "application/json"
                }
            }
    
    # 5. OAuth token (sk-ant-oat*) — use Bearer auth instead of x-api-key
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key and key.startswith("sk-ant-oat"):
        return {
            "provider": "anthropic",
            "api_key": key,
            "base_url": "https://api.anthropic.com/v1/messages",
            "model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            "headers": {
                "Authorization": f"Bearer {key}",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        }
    
    return None

LLM_CONFIG = get_llm_config()

# Trading parameters
# ── DATA-DRIVEN (132 settled trades, 2026-07-16) ──
# BUY_NO overall: 76% WR, +$15.00 → AMAZING
#   <3% edge: 78% WR (BEST bucket!)  → keep MIN_EDGE low for BUY_NO
#   5-10%: 57% WR, >10%: 75% WR    → high edge BUY_NO is good too
# BUY_YES overall: 19% WR, -$1.31 → BAD
#   <3%: 12% WR, 3-5%: 9% WR       → terrible at low edge
#   5-10%: 38% WR (+$3.78)          → the ONLY profitable bucket
#   >10%: 7% WR                     → catastrophically overconfident
# STRATEGY: Different thresholds for BUY_NO (low bar) vs BUY_YES (high bar)
MIN_EDGE_BUY_NO = 0.01    # 1% min for BUY_NO — 78% WR even at tiny edges
MIN_EDGE_BUY_YES = 0.05   # 5% min for BUY_YES — only profitable bucket
MIN_EDGE = 0.01           # Global minimum (legacy compat, real logic in make_trade_decision)
PARLAY_ONLY_NO = True     # On multi-leg parlays, primarily take BUY_NO positions
PARLAY_YES_EXCEPTION = True  # Allow BUY_YES on 2-leg parlays if edge > 5%
MAX_EDGE_CAP = 0.10       # Cap at 10% (was 6%, but BUY_NO does well at 5-10% range)
MAX_POSITION_PCT = 0.05   # 5% max per position
KELLY_FRACTION = 0.15     # Aggressive in paper mode - collect more data
MIN_BET_CENTS = 5         # Minimum 5 cents per contract
MAX_BET_CENTS = 500       # Maximum $5 per contract
MAX_POSITIONS = 15        # Max concurrent positions

# Market scanning filters
MIN_VOLUME = 200           # Min volume in contracts
MIN_LIQUIDITY = 1000       # Min open interest / liquidity
MAX_DAYS_TO_EXPIRY = 30    # Max 30 days until expiry
MIN_DAYS_TO_EXPIRY = 0.02  # ~30 minutes minimum

# Price filters - avoid extremes where market is very confident
MIN_PRICE_CENTS = 5        # Don't trade markets priced < 5¢
MAX_PRICE_CENTS = 95       # Don't trade markets priced > 95¢

# Event tickers to scan for individual game markets
# Format: series_ticker prefixes to look for
SPORTS_EVENT_TICKERS = [
    # NCAA Basketball (CBB)
    "KXCBBSPREAD", "KXCBBTOTAL", "KXCBBML",
    "KXNCAABSPREAD", "KXNCAABTOTAL",
    # NBA
    "KXNBASPREAD", "KXNBATOTAL", "KXNBAML",
    "KXNBAPOINTS", "KXNBAOU",
    # NFL
    "KXNFLSPREAD", "KXNFLTOTAL", "KXNFLML",
    # NHL
    "KXNHLSPREAD", "KXNHLTOTAL", "KXNHLML",
    # MLB
    "KXMLBSPREAD", "KXMLBTOTAL", "KXMLBML",
    # Soccer / MLS
    "KXSOCCERML", "KXSOCCERTOTAL",
    # Esports / multi-game parlays
    "KXMVESPORTSMULTIGAMEEXTENDED",
    "KXMVESPORTSMULTIGAME",
    "KXMULTISPORT",
    # Generic sports
    "KXSPORTSSPREAD", "KXSPORTSTOTAL",
]

# Logging
PROJECT_ROOT = Path(__file__).parent.parent
TRADE_LOG_FILE = PROJECT_ROOT / "data" / "trading" / "kalshi-v3-trades.jsonl"
CYCLE_LOG_FILE = PROJECT_ROOT / "data" / "trading" / "kalshi-v3-cycles.jsonl"

# ============== DATA CLASSES ==============

@dataclass
class MarketInfo:
    """Parsed Kalshi market info."""
    ticker: str
    title: str
    subtitle: str
    category: str
    yes_price: int          # in cents
    no_price: int           # in cents
    volume: int
    open_interest: int
    expiry: str             # ISO datetime
    status: str
    result: str             # "" if not settled
    yes_bid: int = 0
    yes_ask: int = 0
    last_price: int = 0
    
    @property
    def market_prob(self) -> float:
        """Market-implied probability (from yes price)."""
        return self.yes_price / 100.0
    
    @property
    def days_to_expiry(self) -> float:
        """Days until expiry."""
        try:
            exp = datetime.fromisoformat(self.expiry.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            return max(0, (exp - now).total_seconds() / 86400)
        except Exception:
            return 999

@dataclass 
class ForecastResult:
    """Result from Claude forecaster."""
    probability: float         # 0-1
    reasoning: str
    confidence: str           # "low", "medium", "high"
    key_factors: list = field(default_factory=list)
    raw_response: str = ""
    model_used: str = ""
    tokens_used: int = 0

@dataclass
class CriticResult:
    """Result from Claude critic."""
    adjusted_probability: float  # 0-1, after considering flaws
    major_flaws: list = field(default_factory=list)
    minor_flaws: list = field(default_factory=list)
    should_trade: bool = True
    reasoning: str = ""
    raw_response: str = ""
    tokens_used: int = 0

@dataclass
class TradeDecision:
    """Final trade decision."""
    action: str               # "BUY_YES", "BUY_NO", "SKIP"
    edge: float               # Expected edge
    kelly_size: float         # Kelly-suggested position size (fraction)
    contracts: int            # Number of contracts to buy
    price_cents: int          # Price per contract
    reason: str
    forecast: Optional[ForecastResult] = None
    critic: Optional[CriticResult] = None


# ============== KALSHI API ==============

def sign_request(method: str, path: str, timestamp: str) -> str:
    """Sign request with RSA-PSS for Kalshi API."""
    private_key = serialization.load_pem_private_key(PRIVATE_KEY.encode(), password=None)
    message = f"{timestamp}{method}{path}".encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')


def kalshi_api(method: str, path: str, body: dict = None, max_retries: int = 3) -> dict:
    """Make authenticated Kalshi API request."""
    url = f"{BASE_URL}{path}"
    
    for attempt in range(max_retries):
        timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
        signature = sign_request(method, path.split('?')[0], timestamp)
        headers = {
            "KALSHI-ACCESS-KEY": API_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }
        
        try:
            if method == "GET":
                resp = requests.get(url, headers=headers, timeout=15)
            elif method == "POST":
                resp = requests.post(url, headers=headers, json=body, timeout=10)
            elif method == "DELETE":
                resp = requests.delete(url, headers=headers, timeout=10)
            else:
                return {"error": f"Unknown method {method}"}
            
            if resp.status_code >= 500:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {"error": f"Server error {resp.status_code}"}
            
            return resp.json()
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"error": "Timeout"}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"error": "Connection error"}
        except Exception as e:
            return {"error": str(e)}
    
    return {"error": "Max retries exceeded"}


def get_balance() -> float:
    """Get account balance in dollars."""
    result = kalshi_api("GET", "/trade-api/v2/portfolio/balance")
    if "error" in result:
        print(f"❌ Balance error: {result['error']}")
        return 0.0
    balance_cents = result.get("balance", 0)
    return balance_cents / 100.0


def get_positions() -> list:
    """Get current open positions."""
    result = kalshi_api("GET", "/trade-api/v2/portfolio/positions?limit=100&settlement_status=unsettled")
    if "error" in result:
        print(f"❌ Positions error: {result['error']}")
        return []
    return result.get("market_positions", [])


def scan_markets(cursor: str = None, limit: int = 200) -> tuple:
    """
    Scan Kalshi markets with pagination.
    Returns (markets_list, next_cursor).
    """
    path = f"/trade-api/v2/markets?limit={limit}"
    if cursor:
        path += f"&cursor={cursor}"
    
    result = kalshi_api("GET", path)
    if "error" in result:
        print(f"❌ Markets scan error: {result['error']}")
        return [], None
    
    markets = result.get("markets", [])
    next_cursor = result.get("cursor", None)
    return markets, next_cursor


def get_market_orderbook(ticker: str) -> dict:
    """Get orderbook for a specific market."""
    result = kalshi_api("GET", f"/trade-api/v2/markets/{ticker}/orderbook")
    if "error" in result:
        return {}
    return result.get("orderbook", result)


def place_order(ticker: str, side: str, price_cents: int, count: int, dry_run: bool = True) -> dict:
    """
    Place an order on Kalshi.
    side: "yes" or "no"
    price_cents: Price in cents (1-99)
    count: Number of contracts
    """
    if dry_run:
        return {
            "dry_run": True,
            "ticker": ticker,
            "side": side,
            "price": price_cents,
            "count": count,
            "status": "simulated"
        }
    
    body = {
        "ticker": ticker,
        "action": "buy",
        "side": side,
        "type": "limit",
        "count": count,
        "yes_price": price_cents if side == "yes" else (100 - price_cents),
    }
    
    result = kalshi_api("POST", "/trade-api/v2/portfolio/orders", body=body)
    return result


# ============== CLAUDE API ==============

def call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> dict:
    """
    Call LLM API for forecasting/critiquing.
    Supports Anthropic direct, OpenRouter, and OpenAI-compatible endpoints.
    Returns dict with 'content', 'tokens_used', 'model'.
    """
    if not LLM_CONFIG:
        return {"error": "No LLM API key configured. Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY", "content": "", "tokens_used": 0}
    
    if "error" in LLM_CONFIG:
        return {"error": LLM_CONFIG["error"], "content": "", "tokens_used": 0}
    
    provider = LLM_CONFIG["provider"]
    headers = LLM_CONFIG["headers"]
    model = LLM_CONFIG["model"]
    base_url = LLM_CONFIG["base_url"]
    
    try:
        if provider == "anthropic":
            # Anthropic Messages API format
            body = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}]
            }
            resp = requests.post(base_url, headers=headers, json=body, timeout=60)
            
            if resp.status_code != 200:
                return {"error": f"Anthropic API {resp.status_code}: {resp.text[:300]}", "content": "", "tokens_used": 0}
            
            data = resp.json()
            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content += block["text"]
            
            usage = data.get("usage", {})
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            
            return {"content": content, "tokens_used": tokens, "model": data.get("model", model)}
        
        else:
            # OpenRouter / OpenAI-compatible format
            body = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            resp = requests.post(base_url, headers=headers, json=body, timeout=60)
            
            if resp.status_code != 200:
                return {"error": f"LLM API {resp.status_code}: {resp.text[:300]}", "content": "", "tokens_used": 0}
            
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            tokens = usage.get("total_tokens", 0) or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
            
            return {"content": content, "tokens_used": tokens, "model": data.get("model", model)}
    
    except Exception as e:
        return {"error": str(e), "content": "", "tokens_used": 0}


# ============== FORECASTER AGENT ==============

def forecast_market(market: MarketInfo) -> ForecastResult:
    """
    Use Claude to estimate the true probability of a market outcome.
    This is the core of the strategy - Claude analyzes the market question
    and provides its probability estimate.
    """
    system_prompt = """You are an expert forecaster for prediction markets. Your job is to estimate the TRUE probability of events.

Rules:
1. Think step by step about base rates, current conditions, and relevant factors
2. Be calibrated - don't be overconfident. If you're uncertain, your probability should reflect that
3. Consider the time remaining until expiry
4. Account for both sides - what could make YES happen and what could make NO happen
5. Give your final probability as a precise number between 1% and 99%

IMPORTANT: End your response with exactly this format on its own line:
PROBABILITY: XX%
CONFIDENCE: [low/medium/high]
KEY_FACTORS: [factor1], [factor2], [factor3]"""

    expiry_str = market.expiry
    try:
        exp_dt = datetime.fromisoformat(market.expiry.replace('Z', '+00:00'))
        expiry_str = exp_dt.strftime("%Y-%m-%d %H:%M UTC")
        time_left = exp_dt - datetime.now(timezone.utc)
        if time_left.total_seconds() < 3600:
            time_desc = f"{int(time_left.total_seconds() / 60)} minutes"
        elif time_left.days > 0:
            time_desc = f"{time_left.days} days, {time_left.seconds // 3600} hours"
        else:
            time_desc = f"{time_left.seconds // 3600} hours, {(time_left.seconds % 3600) // 60} minutes"
    except Exception:
        time_desc = "unknown"

    user_prompt = f"""Analyze this prediction market and estimate the true probability:

MARKET: {market.title}
{f'DETAILS: {market.subtitle}' if market.subtitle else ''}
CATEGORY: {market.category}
CURRENT YES PRICE: {market.yes_price}¢ (market implies {market.market_prob:.0%} probability)
CURRENT NO PRICE: {market.no_price}¢
VOLUME: {market.volume:,} contracts
OPEN INTEREST: {market.open_interest:,}
EXPIRY: {expiry_str} ({time_desc} remaining)
TICKER: {market.ticker}

Today's date is {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}.

What is the TRUE probability that this market resolves YES? Think step by step, considering:
- Base rates for this type of event
- Current conditions and recent news/data
- Time remaining and what needs to happen
- Whether the market price seems too high or too low

Remember to end with PROBABILITY: XX% on its own line."""

    result = call_claude(system_prompt, user_prompt, max_tokens=1500)
    
    if "error" in result and result["error"]:
        print(f"   ⚠️ Forecaster error: {result['error']}")
        return ForecastResult(
            probability=market.market_prob,  # fallback to market price
            reasoning=f"Error: {result['error']}",
            confidence="low",
            raw_response="",
            model_used="error",
            tokens_used=0
        )
    
    content = result["content"]
    
    # Parse probability
    prob = parse_probability(content)
    if prob is None:
        prob = market.market_prob  # fallback
    
    # Parse confidence
    confidence = "medium"
    conf_match = re.search(r'CONFIDENCE:\s*(low|medium|high)', content, re.IGNORECASE)
    if conf_match:
        confidence = conf_match.group(1).lower()
    
    # Parse key factors
    key_factors = []
    factors_match = re.search(r'KEY_FACTORS:\s*(.+)', content, re.IGNORECASE)
    if factors_match:
        factors_str = factors_match.group(1)
        key_factors = [f.strip() for f in factors_str.split(',')]
    
    return ForecastResult(
        probability=prob,
        reasoning=content,
        confidence=confidence,
        key_factors=key_factors,
        raw_response=content,
        model_used=result.get("model", LLM_CONFIG.get("model", "unknown") if LLM_CONFIG else "unknown"),
        tokens_used=result.get("tokens_used", 0)
    )


def parse_probability(text: str) -> Optional[float]:
    """Parse probability from Claude's response."""
    # Try exact format first: "PROBABILITY: XX%"
    match = re.search(r'PROBABILITY:\s*(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100.0
    
    # Try "my estimate is XX%"
    match = re.search(r'(?:estimate|probability|chance|likelihood)\s+(?:is|of|at)\s+(?:approximately\s+)?(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100.0
    
    # Try standalone "XX%" near end of text
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
    if matches:
        # Take the last percentage mentioned (usually the final answer)
        val = float(matches[-1])
        if 1 <= val <= 99:
            return val / 100.0
    
    return None


# ============== HEURISTIC FORECASTER (NO LLM NEEDED) ==============

def classify_market_type(market: MarketInfo) -> str:
    """
    Classify a Kalshi market into a type for heuristic forecasting.
    Returns one of: 'combo', 'spread', 'total', 'moneyline', 'generic'
    """
    ticker = market.ticker.upper()
    title = market.title.lower()
    subtitle = (market.subtitle or "").lower()
    category = (market.category or "").upper()
    
    # Combo / Parlay markets
    if "MULTIGAME" in ticker or "MULTI" in ticker or "PARLAY" in ticker:
        return "combo"
    if "parlay" in title or "combo" in title or ("and" in title and any(w in title for w in ["win", "beat", "cover", "over"])):
        return "combo"
    # Detect multi-leg from title: comma-separated legs starting with "yes "/"no "
    # e.g. "yes Iowa St.,no Murray St. wins by over 13.5 Points"
    segments = [s.strip() for s in title.split(',')]
    leg_count = sum(1 for s in segments if s.startswith('yes ') or s.startswith('no '))
    if leg_count >= 2:
        return "combo"
    # Detect multi-leg from title: "X and Y and Z"
    and_count = title.count(" and ")
    if and_count >= 1 and any(w in title for w in ["win", "beat", "cover", "over", "under", "score"]):
        return "combo"
    
    # Spread markets
    if "SPREAD" in ticker or "SPREAD" in category:
        return "spread"
    if "spread" in title or re.search(r'[+-]\d+\.?5?\s*(points?|pts?)', title):
        return "spread"
    if "cover" in title:
        return "spread"
    
    # Total / Over-Under markets
    if "TOTAL" in ticker or "TOTAL" in category or "OU" in ticker:
        return "total"
    if "over" in title and ("under" in title or re.search(r'over\s+\d', title)):
        return "total"
    if "total" in title and any(w in title for w in ["points", "goals", "runs", "score"]):
        return "total"
    
    # Moneyline markets
    if "ML" in ticker or "MONEYLINE" in category:
        return "moneyline"
    if any(w in title for w in ["to win", "will win", "wins", "beat", "defeats"]):
        if "and" not in title:  # Not a combo
            return "moneyline"
    
    return "generic"


def estimate_combo_legs(market: MarketInfo) -> int:
    """Estimate the number of legs in a combo/parlay market from the title.
    
    Kalshi parlay titles use COMMAS to separate legs, e.g.:
    'yes Iowa St.,no Murray St. wins by over 13.5 Points'
    'no Belmont wins by over 11.5 Points,no Campbell wins by over 4.5 Points,...'
    
    Each leg starts with 'yes ' or 'no '.
    """
    title = market.title
    
    # PRIMARY: Count legs by 'yes '/'no ' prefixes after commas
    # Split on comma and count segments that start with yes/no
    segments = [s.strip() for s in title.split(',')]
    leg_count = sum(1 for s in segments if s.lower().startswith('yes ') or s.lower().startswith('no '))
    if leg_count >= 2:
        return leg_count
    
    # FALLBACK: Count " and " separators (some older format)
    and_count = title.lower().count(" and ")
    if and_count >= 1:
        return and_count + 1
    
    # Check for numbered legs like "3-leg" or "4-team"
    leg_match = re.search(r'(\d+)[\s-]*(leg|team|game|pick)', title)
    if leg_match:
        return int(leg_match.group(1))
    
    # Default: assume 2-leg if we detected combo but can't count
    return 2


def logistic_spread_prob(spread: float) -> float:
    """
    Convert a point spread to a win probability using a logistic function.
    
    Based on historical sports data:
    - Spread of 0 → 50% (pick 'em)
    - Each point of spread ≈ 3% probability shift (in NFL/NBA)
    - Uses logistic curve for smooth mapping
    
    spread > 0 means the team is favored (e.g., -3.5 → they're favored by 3.5)
    We pass the absolute spread value; positive = favored.
    """
    # k controls steepness. ~0.15-0.2 works well for most sports
    # NFL: k ≈ 0.15, NBA: k ≈ 0.12, CBB: k ≈ 0.10
    k = 0.15
    prob = 1.0 / (1.0 + math.exp(-k * spread))
    return prob


def extract_spread_from_title(title: str) -> Optional[float]:
    """Extract point spread from market title."""
    # Match patterns like "+3.5", "-7", "+1.5 points", etc.
    match = re.search(r'([+-]\d+\.?\d*)\s*(points?|pts?)?', title)
    if match:
        return float(match.group(1))
    
    # Match "by X or more" patterns
    match = re.search(r'by\s+(\d+\.?\d*)\s+or\s+more', title, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    return None


def extract_total_line(title: str) -> Optional[float]:
    """Extract the over/under total line from market title."""
    # Match "over X.5", "under X", "total X.5"
    match = re.search(r'(?:over|under|total)\s+(\d+\.?\d*)', title, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # Match "X.5 points" pattern
    match = re.search(r'(\d+\.5)\s*(?:points?|goals?|runs?)', title, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    return None


def detect_sport(market: MarketInfo) -> str:
    """Detect which sport a market is about."""
    ticker = market.ticker.upper()
    title = market.title.lower()
    category = (market.category or "").upper()
    
    if "NBA" in ticker or "NBA" in category or "nba" in title:
        return "nba"
    if "NFL" in ticker or "NFL" in category or "nfl" in title or "football" in title:
        return "nfl"
    if "CBB" in ticker or "NCAAB" in ticker or "CBB" in category or "ncaa" in title or "college basketball" in title:
        return "cbb"
    if "NHL" in ticker or "NHL" in category or "nhl" in title or "hockey" in title:
        return "nhl"
    if "MLB" in ticker or "MLB" in category or "mlb" in title or "baseball" in title:
        return "mlb"
    if "SOCCER" in ticker or "soccer" in title or "mls" in title:
        return "soccer"
    
    return "unknown"


# Historical average totals by sport (approximate)
SPORT_AVG_TOTALS = {
    "nba": 224.0,    # NBA average total ~224 points
    "nfl": 44.0,     # NFL average total ~44 points
    "cbb": 140.0,    # College basketball ~140 points
    "nhl": 5.8,      # NHL ~5.8 goals
    "mlb": 8.5,      # MLB ~8.5 runs
    "soccer": 2.5,   # Soccer ~2.5 goals
    "unknown": 100.0 # Fallback
}

# Spread steepness by sport (how much each point matters)
SPORT_SPREAD_K = {
    "nba": 0.10,     # NBA points matter less (higher scoring)
    "nfl": 0.15,     # NFL each point more impactful
    "cbb": 0.09,     # College basketball high variance
    "nhl": 0.45,     # Hockey goals are rare, each matters a lot
    "mlb": 0.35,     # Baseball runs matter significantly
    "soccer": 0.55,  # Soccer goals very impactful
    "unknown": 0.15
}


def heuristic_forecast(market: MarketInfo) -> ForecastResult:
    """
    Built-in heuristic forecaster that estimates probability WITHOUT any LLM API.
    
    Uses market structure, sport-specific models, and known biases:
    1. COMBO/PARLAY: Product of individual leg probabilities
    2. SPREAD: Logistic function based on point spread
    3. TOTAL: Comparison to historical averages
    4. MONEYLINE: Market price + favorite-longshot bias correction
    5. GENERIC: Market price with slight regression to 50%
    
    Returns ForecastResult with probability and confidence level.
    """
    market_type = classify_market_type(market)
    sport = detect_sport(market)
    market_prob = market.market_prob
    title = market.title.lower()
    
    prob = market_prob  # default: trust market
    confidence = "low"
    reasoning_parts = []
    key_factors = []
    
    if market_type == "combo":
        # ── COMBO/PARLAY MARKETS ──
        # Data-driven approach based on settled trades:
        # BUY_NO at 81.8% WR means parlays are systematically overpriced (YES too high)
        # BUY_YES at 9.7% WR confirms the overpricing
        num_legs = estimate_combo_legs(market)
        
        if num_legs >= 2:
            # ── PER-LEG ANALYSIS ──
            # Parse individual legs from title to determine leg types
            # Title format: "yes Team A,no Over 147.5 points scored,yes Team B"
            segments = [s.strip() for s in market.title.split(',')]
            leg_probs = []
            leg_descriptions = []
            
            for seg in segments:
                seg_lower = seg.lower()
                # Classify each leg
                if re.search(r'over\s+\d+\.?\d*\s+(?:points|goals|runs)', seg_lower):
                    # Over/under total leg — totals are ~50/50, slight over-bias from public
                    # Check if it's explicitly "under" (no-side)
                    if seg_lower.startswith('no ') and 'over' in seg_lower:
                        # "no Over X" = betting the under → slight edge (public bets overs)
                        leg_prob = 0.52
                    else:
                        # "yes Over X" = betting the over → slightly overbet by public
                        leg_prob = 0.48
                    leg_descriptions.append(f"total({seg[:30]})")
                elif re.search(r'wins?\s+by\s+over\s+\d', seg_lower):
                    # Spread leg: "Team wins by over X.5 Points"
                    spread_val = None
                    sm = re.search(r'by\s+over\s+(\d+\.?\d*)', seg_lower)
                    if sm:
                        spread_val = float(sm.group(1))
                    if spread_val is not None:
                        k = SPORT_SPREAD_K.get(sport, 0.15)
                        # Covering by X+ points → prob from logistic
                        cover_prob = 1.0 / (1.0 + math.exp(k * spread_val))
                        # If "no Team wins by over X" → betting AGAINST covering
                        if seg_lower.startswith('no '):
                            leg_prob = 1.0 - cover_prob
                        else:
                            leg_prob = cover_prob
                        # Adjustment: public bias favors favorites covering
                        leg_prob *= 0.97
                    else:
                        leg_prob = 0.50
                    leg_descriptions.append(f"spread({seg[:30]})")
                elif seg_lower.startswith('yes ') or seg_lower.startswith('no '):
                    # Moneyline leg: "yes TeamName" or "no TeamName"
                    # Without external data, assume:
                    # "yes Team" = betting team wins = parlay maker chose favorite ≈ 60%
                    # "no Team" = betting team loses = parlay maker chose underdog loss ≈ 55%
                    if seg_lower.startswith('yes '):
                        leg_prob = 0.60  # Favorites win ~60%
                    else:
                        leg_prob = 0.55  # "no Team" is less clear, slightly lower confidence
                    leg_descriptions.append(f"ml({seg[:30]})")
                else:
                    leg_prob = 0.55
                    leg_descriptions.append(f"unknown({seg[:30]})")
                
                leg_probs.append(max(0.20, min(0.85, leg_prob)))
            
            # If we couldn't parse legs, fall back to implied per-leg
            if len(leg_probs) < 2:
                implied_per_leg = market_prob ** (1.0 / num_legs)
                # Apply favorite-longshot bias correction
                if implied_per_leg > 0.65:
                    per_leg_adj = implied_per_leg * 1.02
                elif implied_per_leg < 0.55:
                    per_leg_adj = implied_per_leg * 0.92
                else:
                    per_leg_adj = implied_per_leg * 0.97
                per_leg_adj = min(0.88, max(0.40, per_leg_adj))
                leg_probs = [per_leg_adj] * num_legs
            
            # Correlation boost: same-sport legs are correlated
            title_lower = title
            sport_keywords = ['nba', 'nfl', 'nhl', 'mlb', 'ncaa', 'cbb', 'premier league', 'la liga', 'serie a']
            same_sport = sum(1 for kw in sport_keywords if kw in title_lower) >= 1
            corr_per_leg = 1.04 if same_sport else 1.02
            correlation_boost = corr_per_leg ** (num_legs - 1)
            
            # True combo probability = product of individual leg probs × correlation
            true_prob = 1.0
            for lp in leg_probs:
                true_prob *= lp
            true_prob *= correlation_boost
            
            # Favorite-longshot bias on the OVERALL parlay price:
            # Longshot parlays (yes < 20¢) are most overpriced, then < 30¢
            if market.yes_price < 20:
                # Extreme longshots: ~22% overpricing discount
                true_prob = true_prob * 0.78
            elif market.yes_price < 30:
                # Longshot parlays: 15% overpricing discount
                true_prob = true_prob * 0.85
            elif market.yes_price > 70:
                # Favorites in parlays slightly underpriced
                true_prob = true_prob * 1.05
            
            # Additional leg penalty: each extra leg compounds overpricing
            if num_legs >= 6:
                true_prob *= 0.82  # 6+ legs: extreme overpricing
            elif num_legs >= 4:
                true_prob *= 0.88  # 4-5 legs: heavy overpricing
            elif num_legs >= 3:
                true_prob *= 0.93  # 3 legs: moderate overpricing
            
            prob = max(0.02, min(0.98, true_prob))
            confidence = "medium" if num_legs <= 3 else "low"
            
            leg_detail = ", ".join(f"{lp:.0%}" for lp in leg_probs[:5])
            if len(leg_probs) > 5:
                leg_detail += f"... ({len(leg_probs)} total)"
            
            reasoning_parts.append(
                f"Combo/parlay with {num_legs} legs (parsed from title). "
                f"Per-leg probs: [{leg_detail}]. "
                f"Correlation boost: {correlation_boost:.3f} ({'same-sport' if same_sport else 'mixed'}). "
                f"Combined: {prob:.1%}. "
                f"Market price: {market_prob:.1%}. "
                f"{'LONGSHOT OVERPRICING applied.' if market.yes_price < 30 else ''}"
            )
            key_factors = [
                f"{num_legs}-leg parlay",
                f"Legs: [{leg_detail}]",
                f"{'Longshot overpriced' if market.yes_price < 30 else 'Fair-priced parlay'}",
                f"Corr: {correlation_boost:.2f}"
            ]
        else:
            prob = market_prob
            reasoning_parts.append("Could not determine number of legs, using market price.")
            key_factors = ["Unknown combo structure"]
    
    elif market_type == "spread":
        # ── SPREAD MARKETS ──
        # Use logistic function based on the point spread
        spread = extract_spread_from_title(market.title)
        k = SPORT_SPREAD_K.get(sport, 0.15)
        
        if spread is not None:
            # Determine if market is for the favored team covering
            is_covering = "cover" in title or spread < 0
            
            # Convert spread to probability
            # Negative spread = favorite (e.g., -3.5 means favored by 3.5)
            spread_prob = 1.0 / (1.0 + math.exp(-k * (-spread)))
            
            # The spread is designed to be ~50/50, but public betting bias
            # means favorites often have slightly <50% covering probability
            # (the line moves with public money, not sharp money)
            public_bias_adjustment = -0.015  # slight edge to underdogs covering
            
            prob = max(0.05, min(0.95, spread_prob + public_bias_adjustment))
            confidence = "medium"
            
            reasoning_parts.append(
                f"Spread market: {spread:+.1f} points ({sport.upper()}). "
                f"Logistic model (k={k}): {spread_prob:.1%}. "
                f"After public bias adjustment: {prob:.1%}. "
                f"Market price: {market_prob:.1%}."
            )
            key_factors = [
                f"Spread: {spread:+.1f}",
                f"Sport: {sport.upper()}",
                "Public betting bias",
                "Logistic model"
            ]
        else:
            # Can't extract spread, use market price with regression
            prob = 0.7 * market_prob + 0.3 * 0.5
            confidence = "low"
            reasoning_parts.append(
                f"Spread market but couldn't extract line. "
                f"Regressing market price {market_prob:.1%} toward 50%: {prob:.1%}."
            )
            key_factors = ["Unknown spread", "Mean reversion"]
    
    elif market_type == "total":
        # ── TOTAL / OVER-UNDER MARKETS ──
        total_line = extract_total_line(market.title)
        avg_total = SPORT_AVG_TOTALS.get(sport, 100.0)
        
        is_over = "over" in title
        is_under = "under" in title
        
        if total_line is not None:
            # How far is the line from the average?
            # Lines above average → under is slightly more likely
            # Lines below average → over is slightly more likely
            deviation = (total_line - avg_total) / avg_total
            
            # Base: totals are designed to be ~50/50
            # Known bias: public tends to bet overs (they're more fun)
            # This means overs are slightly overpriced
            base_prob = 0.50
            
            # Adjust based on deviation from average
            # If line is well above average, under is more likely
            adjustment = -deviation * 0.15  # small adjustment
            
            if is_over:
                prob = base_prob - adjustment - 0.02  # over bias: public loves overs
            elif is_under:
                prob = base_prob + adjustment + 0.02  # under slight edge
            else:
                prob = base_prob
            
            prob = max(0.10, min(0.90, prob))
            confidence = "medium"
            
            reasoning_parts.append(
                f"Total market: line at {total_line} ({sport.upper()}, avg: {avg_total}). "
                f"Deviation from avg: {deviation:+.1%}. "
                f"{'Over' if is_over else 'Under' if is_under else 'Unknown side'}: {prob:.1%}. "
                f"Includes public over-betting bias. Market: {market_prob:.1%}."
            )
            key_factors = [
                f"Line: {total_line}",
                f"Avg: {avg_total}",
                "Public over-bias",
                f"{'Over' if is_over else 'Under'}"
            ]
        else:
            prob = 0.7 * market_prob + 0.3 * 0.5
            confidence = "low"
            reasoning_parts.append(
                f"Total market but couldn't extract line. "
                f"Regressing market price toward 50%: {prob:.1%}."
            )
            key_factors = ["Unknown total line", "Mean reversion"]
    
    elif market_type == "moneyline":
        # ── MONEYLINE MARKETS ──
        # Use market price as baseline, then correct for favorite-longshot bias
        #
        # Known bias: on prediction markets and sportsbooks alike,
        # longshots (low probability outcomes) tend to be overpriced.
        # Favorites tend to be slightly underpriced.
        #
        # Correction model (empirical):
        # - If market_prob > 0.5: team is favorite → true prob slightly higher
        # - If market_prob < 0.5: team is underdog → true prob slightly lower
        
        if market_prob > 0.60:
            # Strong favorite: market often underprices
            bias_correction = 0.02 + (market_prob - 0.60) * 0.05
            prob = min(0.95, market_prob + bias_correction)
            confidence = "medium"
            key_factors.append("Favorite-longshot bias: favorite underpriced")
        elif market_prob < 0.40:
            # Underdog: market often overprices
            bias_correction = 0.02 + (0.40 - market_prob) * 0.05
            prob = max(0.05, market_prob - bias_correction)
            confidence = "medium"
            key_factors.append("Favorite-longshot bias: underdog overpriced")
        else:
            # Near 50/50: less bias to exploit
            prob = market_prob
            confidence = "low"
            key_factors.append("Near toss-up, minimal bias")
        
        reasoning_parts.append(
            f"Moneyline market ({sport.upper()}). "
            f"Market implied: {market_prob:.1%}. "
            f"After favorite-longshot bias correction: {prob:.1%}. "
            f"{'Favorite' if market_prob > 0.5 else 'Underdog'} side."
        )
        key_factors.extend(["Market efficiency", f"Sport: {sport.upper()}"])
    
    else:
        # ── GENERIC MARKETS ──
        # For non-sports or unrecognized markets:
        # Slight regression toward 50% (markets can overreact)
        # But mostly trust the market
        
        regression_strength = 0.10  # 10% pull toward 50%
        prob = (1 - regression_strength) * market_prob + regression_strength * 0.5
        confidence = "low"
        
        reasoning_parts.append(
            f"Generic market type. Market implied: {market_prob:.1%}. "
            f"Applying {regression_strength:.0%} regression to 50%: {prob:.1%}. "
            f"Low confidence — no sport-specific model available."
        )
        key_factors = ["Generic model", "Mean reversion", "No specific edge"]
    
    # Build full reasoning string
    reasoning = (
        f"[HEURISTIC FORECASTER — no LLM API]\n"
        f"Market type: {market_type.upper()}\n"
        f"Sport: {sport.upper()}\n\n"
        + "\n".join(reasoning_parts)
    )
    
    return ForecastResult(
        probability=prob,
        reasoning=reasoning,
        confidence=confidence,
        key_factors=key_factors[:4],
        raw_response=reasoning,
        model_used=f"heuristic-{market_type}",
        tokens_used=0
    )


def heuristic_critique(market: MarketInfo, forecast: ForecastResult) -> CriticResult:
    """
    Built-in heuristic critic that validates the heuristic forecast.
    No LLM needed. Uses simple sanity checks.
    """
    major_flaws = []
    minor_flaws = []
    should_trade = True
    adj_prob = forecast.probability
    
    market_prob = market.market_prob
    edge = abs(forecast.probability - market_prob)
    market_type = classify_market_type(market)
    
    # Sanity check 1: Very large edge — threshold depends on market type
    # Combo/parlay markets: large edges are EXPECTED (parlays are systematically
    # overpriced due to compounding house edge on each leg). Don't veto these.
    # Non-parlay markets: be suspicious of very large edges (>40%).
    if market_type == "combo":
        # Parlays: edges of 20-30% are normal and expected.
        # Only flag truly extreme edges (>50%) as suspicious.
        if edge > 0.50:
            minor_flaws.append(f"Very large parlay edge ({edge:.0%}) — double check")
            adj_prob = 0.7 * forecast.probability + 0.3 * market_prob
        # Otherwise: trust the heuristic, parlays ARE overpriced
    else:
        # Non-parlay: be more cautious, but 20% was too aggressive a threshold.
        # Raise to 40% — edges of 20-40% can happen on spreads/totals with
        # public betting bias.
        if edge > 0.40:
            major_flaws.append(f"Very large edge ({edge:.0%}) — heuristic may be wrong")
            adj_prob = 0.6 * forecast.probability + 0.4 * market_prob
            should_trade = False
        elif edge > 0.25:
            minor_flaws.append(f"Large edge ({edge:.0%}) — proceed with caution")
            adj_prob = 0.8 * forecast.probability + 0.2 * market_prob
    
    # Sanity check 2: Low volume → less reliable market price → less reliable edge
    if market.volume < 500:
        minor_flaws.append(f"Low volume ({market.volume}) — market may be inefficient")
    
    # Sanity check 3: Very close to expiry → more volatile
    if market.days_to_expiry < 0.1:  # Less than ~2.4 hours
        minor_flaws.append(f"Very close to expiry ({market.days_to_expiry:.2f}d)")
        # Close to expiry, market is probably right
        adj_prob = 0.7 * market_prob + 0.3 * forecast.probability
    
    # Sanity check 4: Heuristic confidence is low + small edge → skip
    # (Raised threshold: data shows <3% edge has 66.7% WR, so small edges are fine
    #  when confidence is medium; only skip low-confidence + tiny edge)
    if forecast.confidence == "low" and edge < 0.05:
        should_trade = False
        major_flaws.append("Low confidence heuristic + small edge")
    
    # Sanity check 5: Generic model shouldn't override market
    if "heuristic-generic" in (forecast.model_used or ""):
        if edge > 0.05:
            should_trade = False
            major_flaws.append("Generic model — no specific insight to trade on")
    
    reasoning = (
        f"[HEURISTIC CRITIC]\n"
        f"Market type: {market_type}. Edge: {edge:.1%}. "
        f"Major flaws: {', '.join(major_flaws) if major_flaws else 'None'}. "
        f"Minor flaws: {', '.join(minor_flaws) if minor_flaws else 'None'}. "
        f"Should trade: {'Yes' if should_trade else 'No'}."
    )
    
    return CriticResult(
        adjusted_probability=adj_prob,
        major_flaws=major_flaws,
        minor_flaws=minor_flaws,
        should_trade=should_trade,
        reasoning=reasoning,
        raw_response=reasoning,
        tokens_used=0
    )


# ============== CRITIC AGENT ==============

def critique_forecast(market: MarketInfo, forecast: ForecastResult) -> CriticResult:
    """
    Second Claude call to critically evaluate the forecast.
    Looks for flaws, missing context, overconfidence, etc.
    """
    system_prompt = """You are a critical analyst reviewing probability forecasts for prediction markets. Your job is to find flaws, missing context, and potential errors in forecasts.

Rules:
1. Look for overconfidence or underconfidence
2. Check if important factors were missed
3. Consider contrarian perspectives
4. Check if the reasoning has logical gaps
5. Provide an ADJUSTED probability if the original seems wrong

IMPORTANT: End your response with exactly:
ADJUSTED_PROBABILITY: XX%
MAJOR_FLAWS: [flaw1], [flaw2] (or NONE)
SHOULD_TRADE: [yes/no]"""

    edge = forecast.probability - market.market_prob
    
    user_prompt = f"""Review this forecast for a prediction market:

MARKET: {market.title}
{f'DETAILS: {market.subtitle}' if market.subtitle else ''}
MARKET PRICE: {market.yes_price}¢ (implies {market.market_prob:.0%})
VOLUME: {market.volume:,} contracts

FORECASTER'S ESTIMATE: {forecast.probability:.1%}
FORECASTER'S CONFIDENCE: {forecast.confidence}
FORECASTER'S REASONING:
{forecast.reasoning[:2000]}

The forecaster sees a {abs(edge):.1%} edge ({'YES is underpriced' if edge > 0 else 'NO is underpriced'}).

Questions to evaluate:
1. Is the forecaster being overconfident? (Common with AI forecasters)
2. Are there important factors they missed?
3. Does the market price actually incorporate information the forecaster doesn't have?
4. Is the edge large enough to overcome transaction costs and uncertainty?
5. Are there any logical flaws in the reasoning?

End with ADJUSTED_PROBABILITY, MAJOR_FLAWS, and SHOULD_TRADE."""

    result = call_claude(system_prompt, user_prompt, max_tokens=1200)
    
    if "error" in result and result["error"]:
        print(f"   ⚠️ Critic error: {result['error']}")
        return CriticResult(
            adjusted_probability=forecast.probability,
            should_trade=False,
            reasoning=f"Error: {result['error']}",
            tokens_used=0
        )
    
    content = result["content"]
    
    # Parse adjusted probability
    adj_prob = parse_probability(content)
    if adj_prob is None:
        # If critic didn't give adjusted prob, use forecaster's
        adj_prob = forecast.probability
    
    # Parse major flaws
    major_flaws = []
    flaws_match = re.search(r'MAJOR_FLAWS:\s*(.+)', content, re.IGNORECASE)
    if flaws_match:
        flaws_str = flaws_match.group(1).strip()
        if flaws_str.upper() != "NONE":
            major_flaws = [f.strip() for f in flaws_str.split(',') if f.strip()]
    
    # Parse should_trade
    should_trade = True
    trade_match = re.search(r'SHOULD_TRADE:\s*(yes|no)', content, re.IGNORECASE)
    if trade_match:
        should_trade = trade_match.group(1).lower() == "yes"
    
    # Parse minor flaws (anything mentioned as concerns)
    minor_flaws = []
    
    return CriticResult(
        adjusted_probability=adj_prob,
        major_flaws=major_flaws,
        minor_flaws=minor_flaws,
        should_trade=should_trade,
        reasoning=content,
        raw_response=content,
        tokens_used=result.get("tokens_used", 0)
    )


# ============== TRADER AGENT ==============

def calculate_kelly(prob: float, price_cents: int) -> float:
    """
    Kelly criterion for position sizing.
    
    f* = (bp - q) / b
    where:
      b = net odds (payout / stake - 1)
      p = probability of winning
      q = 1 - p
    
    Returns fraction of bankroll to bet (before applying KELLY_FRACTION cap).
    """
    if price_cents <= 0 or price_cents >= 100:
        return 0.0
    
    # Buying at price_cents, payout is 100 cents
    # b = (100 - price_cents) / price_cents = net odds
    b = (100 - price_cents) / price_cents
    p = prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Apply fractional Kelly
    kelly *= KELLY_FRACTION
    
    # Clamp
    return max(0.0, min(kelly, MAX_POSITION_PCT))


def make_trade_decision(market: MarketInfo, forecast: ForecastResult, critic: CriticResult, balance: float) -> TradeDecision:
    """
    Compare Claude's probability vs market price and decide whether to trade.
    Uses the critic's adjusted probability (blended with forecaster's).
    """
    # Blend forecaster and critic probabilities
    # Weight: 60% forecaster, 40% critic (forecaster has more analysis)
    final_prob = 0.6 * forecast.probability + 0.4 * critic.adjusted_probability
    
    market_prob = market.market_prob
    edge_yes = final_prob - market_prob      # Edge for buying YES
    edge_no = (1 - final_prob) - (1 - market_prob)  # Edge for buying NO (= -edge_yes)
    
    # Determine best side with SPLIT THRESHOLDS (data-driven)
    # BUY_NO at <3% edge: 78% WR → low bar. BUY_YES at <5%: ~10% WR → high bar.
    if edge_yes > 0:
        # Would be BUY_YES → need higher threshold
        if edge_yes < MIN_EDGE_BUY_YES:
            return TradeDecision(
                action="SKIP",
                edge=edge_yes,
                kelly_size=0,
                contracts=0,
                price_cents=0,
                reason=f"BUY_YES edge too small: {edge_yes:+.1%} (need >{MIN_EDGE_BUY_YES:.0%} for YES)",
                forecast=forecast,
                critic=critic
            )
    else:
        # Would be BUY_NO → lower threshold
        if abs(edge_yes) < MIN_EDGE_BUY_NO:
            return TradeDecision(
                action="SKIP",
                edge=edge_yes,
                kelly_size=0,
                contracts=0,
                price_cents=0,
                reason=f"BUY_NO edge too small: {abs(edge_yes):+.1%} (need >{MIN_EDGE_BUY_NO:.0%} for NO)",
                forecast=forecast,
                critic=critic
            )
    
    # Check critic veto
    if not critic.should_trade:
        return TradeDecision(
            action="SKIP",
            edge=edge_yes,
            kelly_size=0,
            contracts=0,
            price_cents=0,
            reason=f"Critic vetoed: {', '.join(critic.major_flaws) if critic.major_flaws else 'general concerns'}",
            forecast=forecast,
            critic=critic
        )
    
    # Check for major flaws
    if len(critic.major_flaws) >= 2:
        return TradeDecision(
            action="SKIP",
            edge=edge_yes,
            kelly_size=0,
            contracts=0,
            price_cents=0,
            reason=f"Too many flaws ({len(critic.major_flaws)}): {', '.join(critic.major_flaws[:2])}",
            forecast=forecast,
            critic=critic
        )
    
    # Check confidence — but only block BUY_YES (BUY_NO works at low conf)
    if forecast.confidence == "low" and edge_yes > 0 and abs(edge_yes) < 0.10:
        return TradeDecision(
            action="SKIP",
            edge=edge_yes,
            kelly_size=0,
            contracts=0,
            price_cents=0,
            reason=f"Low confidence + moderate edge ({edge_yes:+.1%})",
            forecast=forecast,
            critic=critic
        )
    
    # Data-driven edge cap: forecaster overconfident at high edges (0% WR on >10%)
    if abs(edge_yes) > MAX_EDGE_CAP:
        # Clip to cap — don't trust large perceived edges
        edge_yes = MAX_EDGE_CAP if edge_yes > 0 else -MAX_EDGE_CAP
        # Recalculate final_prob from capped edge
        final_prob = market_prob + edge_yes

    # Decide YES or NO
    market_type = classify_market_type(market)
    if edge_yes > 0:
        # YES is underpriced → buy YES
        # But for parlays: data shows BUY_YES has 9.7% WR → mostly skip
        if PARLAY_ONLY_NO and market_type == "combo":
            # Exception: allow BUY_YES on 2-leg parlays with edge > 5%
            num_legs = estimate_combo_legs(market)
            allow_yes = (
                PARLAY_YES_EXCEPTION 
                and num_legs <= 2 
                and edge_yes > 0.05
                and market.yes_price >= 30  # Not a longshot
            )
            if not allow_yes:
                return TradeDecision(
                    action="SKIP",
                    edge=edge_yes,
                    kelly_size=0,
                    contracts=0,
                    price_cents=0,
                    reason=f"Parlay BUY_YES blocked (data: 9.7% WR). legs={num_legs}, edge={edge_yes:.1%}. Need 2-leg + >5% edge + yes>=30¢.",
                    forecast=forecast,
                    critic=critic
                )
        action = "BUY_YES"
        side_price = market.yes_price
        edge = edge_yes
    else:
        # NO is underpriced → buy NO
        action = "BUY_NO"
        side_price = market.no_price
        edge = abs(edge_yes)
    
    # Calculate position size with Kelly
    kelly_frac = calculate_kelly(final_prob if action == "BUY_YES" else (1 - final_prob), side_price)
    
    if kelly_frac <= 0:
        return TradeDecision(
            action="SKIP",
            edge=edge,
            kelly_size=0,
            contracts=0,
            price_cents=side_price,
            reason=f"Kelly says no bet (edge={edge:+.1%} but odds unfavorable)",
            forecast=forecast,
            critic=critic
        )
    
    # Calculate number of contracts
    bet_dollars = balance * kelly_frac
    bet_cents = int(bet_dollars * 100)
    
    # Clamp bet size
    cost_per_contract = side_price  # cents
    if cost_per_contract <= 0:
        cost_per_contract = 1
    
    contracts = max(1, bet_cents // cost_per_contract)
    
    # Don't exceed max bet — HARD CAP: 5% of bankroll (dynamic) or MAX_BET_CENTS (flat), whichever is lower
    max_bet_dynamic = int(balance * MAX_POSITION_PCT * 100)  # 5% of bankroll in cents
    effective_max = min(MAX_BET_CENTS, max_bet_dynamic)
    total_cost_cents = contracts * cost_per_contract
    if total_cost_cents > effective_max:
        contracts = effective_max // cost_per_contract
    
    # Ensure at least 1 contract if the cost per contract is affordable
    # (within 10% of bankroll — generous for paper trading / data collection)
    max_single_contract_cents = int(balance * 0.10 * 100)  # 10% of bankroll
    if contracts <= 0 and cost_per_contract <= max_single_contract_cents and cost_per_contract >= MIN_BET_CENTS:
        contracts = 1
    
    if contracts <= 0:
        return TradeDecision(
            action="SKIP",
            edge=edge,
            kelly_size=kelly_frac,
            contracts=0,
            price_cents=side_price,
            reason=f"Position too small (cost/contract={cost_per_contract}¢, max={effective_max}¢, bankroll=${balance:.2f})",
            forecast=forecast,
            critic=critic
        )
    
    return TradeDecision(
        action=action,
        edge=edge,
        kelly_size=kelly_frac,
        contracts=contracts,
        price_cents=side_price,
        reason=f"Edge={edge:.1%}, Kelly={kelly_frac:.3f}, Final_prob={final_prob:.1%} vs Market={market_prob:.1%}",
        forecast=forecast,
        critic=critic
    )


# ============== MARKET SCANNER ==============

def parse_market(raw: dict) -> Optional[MarketInfo]:
    """Parse raw Kalshi API market into MarketInfo."""
    try:
        ticker = raw.get("ticker", "")
        title = raw.get("title", "") or raw.get("event_title", "")
        subtitle = raw.get("subtitle", "") or raw.get("yes_sub_title", "")
        category = raw.get("category", "") or raw.get("series_ticker", "")
        
        yes_price = raw.get("yes_bid", 0) or raw.get("last_price", 50)
        no_price = 100 - yes_price if yes_price else 50
        
        # Try different field names for yes/no prices
        if "yes_ask" in raw and raw["yes_ask"]:
            yes_ask = raw["yes_ask"]
        else:
            yes_ask = yes_price
            
        if "yes_bid" in raw and raw["yes_bid"]:
            yes_bid = raw["yes_bid"]
        else:
            yes_bid = yes_price
        
        volume = raw.get("volume", 0) or 0
        open_interest = raw.get("open_interest", 0) or 0
        
        # Liquidity proxy: use open_interest or volume
        liquidity = max(open_interest, volume)
        
        expiry = raw.get("close_time", "") or raw.get("expiration_time", "")
        status = raw.get("status", "")
        result = raw.get("result", "") or ""
        last_price = raw.get("last_price", 0) or 0
        
        return MarketInfo(
            ticker=ticker,
            title=title,
            subtitle=subtitle,
            category=category,
            yes_price=yes_price,
            no_price=no_price,
            volume=volume,
            open_interest=open_interest,
            expiry=expiry,
            status=status,
            result=result,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            last_price=last_price
        )
    except Exception as e:
        return None


def filter_markets(markets: list) -> list:
    """Filter markets based on trading criteria."""
    filtered = []
    
    for m in markets:
        # Must have a ticker and title
        if not m.ticker or not m.title:
            continue
        
        # Status must be open/active
        if m.status not in ("open", "active", ""):
            continue
        
        # Must not be settled
        if m.result:
            continue
        
        # Volume filter
        if m.volume < MIN_VOLUME:
            continue
        
        # Liquidity filter (open_interest or volume)
        liquidity = max(m.open_interest, m.volume)
        if liquidity < MIN_LIQUIDITY:
            continue
        
        # Time to expiry
        dte = m.days_to_expiry
        if dte > MAX_DAYS_TO_EXPIRY:
            continue
        if dte < MIN_DAYS_TO_EXPIRY:
            continue
        
        # Price filters - skip extreme prices where market is very confident
        if m.yes_price < MIN_PRICE_CENTS or m.yes_price > MAX_PRICE_CENTS:
            continue
        
        filtered.append(m)
    
    return filtered


def scan_event_markets(event_ticker: str) -> list:
    """
    Scan markets for a specific event ticker / series.
    Uses the series_ticker filter to find individual game markets.
    """
    path = f"/trade-api/v2/markets?limit=200&series_ticker={event_ticker}&status=open"
    result = kalshi_api("GET", path)
    if "error" in result:
        return []
    return result.get("markets", [])


def scan_all_markets() -> list:
    """
    Scan all open Kalshi markets with pagination.
    Also specifically scans for individual sports game markets (spreads, totals, MLs).
    Returns filtered list of MarketInfo objects.
    """
    all_markets = []
    seen_tickers = set()
    cursor = None
    page = 0
    max_pages = 20  # Safety limit
    
    print("📡 Scanning Kalshi markets (general)...")
    
    while page < max_pages:
        raw_markets, next_cursor = scan_markets(cursor=cursor, limit=200)
        
        if not raw_markets:
            break
        
        for raw in raw_markets:
            parsed = parse_market(raw)
            if parsed and parsed.ticker not in seen_tickers:
                all_markets.append(parsed)
                seen_tickers.add(parsed.ticker)
        
        page += 1
        print(f"   Page {page}: {len(raw_markets)} markets (total unique: {len(all_markets)})")
        
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        
        time.sleep(0.3)  # Rate limit respect
    
    # Also scan specific sports event tickers for individual game markets
    print(f"\n📡 Scanning {len(SPORTS_EVENT_TICKERS)} sports event tickers...")
    sports_found = 0
    for event_ticker in SPORTS_EVENT_TICKERS:
        raw_markets = scan_event_markets(event_ticker)
        for raw in raw_markets:
            parsed = parse_market(raw)
            if parsed and parsed.ticker not in seen_tickers:
                all_markets.append(parsed)
                seen_tickers.add(parsed.ticker)
                sports_found += 1
        time.sleep(0.2)  # Rate limit
    
    if sports_found > 0:
        print(f"   Found {sports_found} additional sports markets")
    else:
        print(f"   No additional sports markets found via event tickers")
    
    # Filter
    filtered = filter_markets(all_markets)
    print(f"   Filtered: {len(filtered)}/{len(all_markets)} markets pass criteria")
    
    return filtered


# ============== MARKET SCORING / PRIORITIZATION ==============

def score_market(market: MarketInfo) -> float:
    """
    Score a market for how promising it is to analyze.
    Higher score = analyze first.
    
    Factors:
    - Volume (more volume = more liquid = better)
    - Price near 50% (more room for edge)
    - Days to expiry (sweet spot: 1-14 days)
    - Open interest
    """
    score = 0.0
    
    # Volume score (log scale, max ~10 points)
    if market.volume > 0:
        score += min(10, math.log10(market.volume) * 2)
    
    # Price distance from extremes (max 10 points, peak at 50¢)
    price_mid = abs(market.yes_price - 50)
    score += max(0, 10 - price_mid * 0.2)
    
    # Days to expiry sweet spot (max 5 points, peak at 3-7 days)
    dte = market.days_to_expiry
    if 1 <= dte <= 7:
        score += 5
    elif 7 < dte <= 14:
        score += 3
    elif 0.1 <= dte < 1:
        score += 2
    else:
        score += 1
    
    # Open interest bonus
    if market.open_interest > 0:
        score += min(5, math.log10(market.open_interest + 1) * 1.5)
    
    return score


# ============== LOGGING ==============

def log_trade(market: MarketInfo, decision: TradeDecision, order_result: dict, dry_run: bool):
    """Log trade to JSONL file."""
    TRADE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "ticker": market.ticker,
        "title": market.title,
        "category": market.category,
        "market_price_yes": market.yes_price,
        "market_price_no": market.no_price,
        "volume": market.volume,
        "open_interest": market.open_interest,
        "expiry": market.expiry,
        "days_to_expiry": round(market.days_to_expiry, 2),
        "action": decision.action,
        "edge": round(decision.edge, 4),
        "kelly_size": round(decision.kelly_size, 4),
        "contracts": decision.contracts,
        "price_cents": decision.price_cents,
        "reason": decision.reason,
        "forecast_prob": round(decision.forecast.probability, 4) if decision.forecast else None,
        "forecast_confidence": decision.forecast.confidence if decision.forecast else None,
        "forecast_key_factors": decision.forecast.key_factors if decision.forecast else [],
        "critic_adjusted_prob": round(decision.critic.adjusted_probability, 4) if decision.critic else None,
        "critic_major_flaws": decision.critic.major_flaws if decision.critic else [],
        "critic_should_trade": decision.critic.should_trade if decision.critic else None,
        "total_tokens": (
            (decision.forecast.tokens_used if decision.forecast else 0) +
            (decision.critic.tokens_used if decision.critic else 0)
        ),
        "order_result": order_result
    }
    
    with open(TRADE_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_cycle(stats: dict):
    """Log cycle summary."""
    CYCLE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **stats
    }
    
    with open(CYCLE_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============== MAIN TRADING CYCLE ==============

def run_cycle(dry_run: bool = True, max_markets_to_analyze: int = 20, max_trades: int = 10):
    """
    Run one complete trading cycle:
    1. Scan all markets
    2. Score and rank them
    3. For top N: Forecast → Critique → Trade decision
    4. Execute trades (or log dry run)
    """
    cycle_start = time.time()
    
    print("=" * 70)
    print(f"🤖 KALSHI AUTOTRADER v3 - Claude-as-Forecaster")
    print(f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'🧪 DRY RUN MODE' if dry_run else '🔴 LIVE TRADING MODE'}")
    print("=" * 70)
    
    # Check LLM API configuration
    llm_available = True
    use_heuristic = False
    if not LLM_CONFIG or "error" in (LLM_CONFIG or {}):
        llm_available = False
        use_heuristic = True
        print("⚠️  NO LLM API KEY - Using HEURISTIC FORECASTER (built-in)")
        if LLM_CONFIG and "error" in LLM_CONFIG:
            print(f"   Reason: {LLM_CONFIG['error']}")
        print("   Using sport-specific heuristic models as fallback")
        print("   For better forecasts, set one of:")
        print("   • export ANTHROPIC_API_KEY=sk-ant-api03-...")
        print("   • export OPENROUTER_API_KEY=sk-or-...")
        print("   • Add key to ~/.clawdbot/.env.trading")
        print()
    else:
        provider = LLM_CONFIG['provider']
        model = LLM_CONFIG['model']
        key_preview = LLM_CONFIG['api_key'][:20] + "..."
        print(f"✅ LLM: {provider} / {model} ({key_preview})")
        
        # Check if it's an OAuth token (won't work for API calls)
        if LLM_CONFIG['api_key'].startswith("sk-ant-oat"):
            print("⚠️  OAuth token detected (sk-ant-oat) — won't work for API calls")
            print("   Falling back to HEURISTIC FORECASTER")
            llm_available = False
            use_heuristic = True
    
    # Get balance
    balance = get_balance()
    print(f"💰 Balance: ${balance:.2f}")
    
    if balance <= 0 and not dry_run:
        print("❌ No balance available for trading!")
        return
    
    # Use a virtual balance for dry run (if balance too low for any real trade)
    if dry_run and balance < 1.0:
        balance = 100.0
        print(f"   📝 Using virtual balance: ${balance:.2f} (paper mode)")
    
    # Get current positions
    positions = get_positions()
    num_positions = len(positions)
    print(f"📊 Open positions: {num_positions}/{MAX_POSITIONS}")
    
    if num_positions >= MAX_POSITIONS:
        print("⚠️ Max positions reached, skipping this cycle")
        return
    
    # Scan markets
    markets = scan_all_markets()
    
    if not markets:
        print("❌ No tradeable markets found!")
        return
    
    # Score and rank markets
    scored = [(m, score_market(m)) for m in markets]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Select top markets to analyze
    top_markets = scored[:max_markets_to_analyze]
    
    print(f"\n🎯 TOP {len(top_markets)} MARKETS TO ANALYZE:")
    print("-" * 70)
    for i, (m, score) in enumerate(top_markets, 1):
        print(f"  {i:2d}. [{score:.1f}] {m.ticker}")
        print(f"      {m.title[:80]}")
        print(f"      YES: {m.yes_price}¢ | Vol: {m.volume:,} | DTE: {m.days_to_expiry:.1f}d | Cat: {m.category}")
    print("-" * 70)
    
    # Analyze each market: Forecast → Critique → Decide
    trades_executed = 0
    trades_skipped = 0
    total_tokens = 0
    
    # Track existing position tickers to avoid doubling up
    existing_tickers = set()
    for pos in positions:
        existing_tickers.add(pos.get("ticker", ""))
    
    # DEDUP: Also load tickers from trade log to avoid re-trading in paper mode
    # In dry_run mode, positions aren't tracked by Kalshi API, so we track locally
    if dry_run and TRADE_LOG_FILE.exists():
        try:
            with open(TRADE_LOG_FILE) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("action") in ("BUY_YES", "BUY_NO") and entry.get("dry_run"):
                            existing_tickers.add(entry.get("ticker", ""))
                    except json.JSONDecodeError:
                        continue
            print(f"📋 Dedup: {len(existing_tickers)} tickers already traded (paper mode)")
        except Exception as e:
            print(f"⚠️ Dedup file read error: {e}")
    
    for i, (market, score) in enumerate(top_markets, 1):
        if trades_executed >= max_trades:
            print(f"\n⏹️ Max trades ({max_trades}) reached, stopping analysis")
            break
        
        # Skip if we already have a position
        if market.ticker in existing_tickers:
            print(f"\n⏭️ [{i}/{len(top_markets)}] {market.ticker} - Already have position, skipping")
            continue
        
        print(f"\n{'='*50}")
        print(f"🔍 [{i}/{len(top_markets)}] Analyzing: {market.ticker}")
        print(f"   {market.title[:80]}")
        print(f"   Market price: YES {market.yes_price}¢ / NO {market.no_price}¢")
        
        # Step 1: FORECASTER (LLM or heuristic)
        if use_heuristic:
            market_type = classify_market_type(market)
            sport = detect_sport(market)
            print(f"   🧮 Heuristic forecasting ({market_type}/{sport})...")
            forecast = heuristic_forecast(market)
        else:
            print(f"   🧠 LLM Forecasting...")
            forecast = forecast_market(market)
        total_tokens += forecast.tokens_used
        print(f"   📊 Forecast: {forecast.probability:.1%} (confidence: {forecast.confidence})")
        print(f"   📝 Key factors: {', '.join(forecast.key_factors[:3]) if forecast.key_factors else 'N/A'}")
        
        # Quick edge check before spending tokens on critic
        # Use MIN_EDGE_BUY_NO (lowest bar) for quick check — fine-grained filter in make_trade_decision
        quick_edge = abs(forecast.probability - market.market_prob)
        if quick_edge < MIN_EDGE_BUY_NO * 0.5:
            print(f"   ⏭️ Quick skip: edge {quick_edge:.1%} too small")
            trades_skipped += 1
            
            # Log the skip
            skip_decision = TradeDecision(
                action="SKIP",
                edge=quick_edge,
                kelly_size=0,
                contracts=0,
                price_cents=0,
                reason=f"Quick edge check: {quick_edge:.1%} < {MIN_EDGE*0.5:.1%} threshold",
                forecast=forecast
            )
            log_trade(market, skip_decision, {}, dry_run)
            continue
        
        # Step 2: CRITIC (LLM or heuristic)
        if use_heuristic:
            print(f"   🔎 Heuristic critique...")
            critic = heuristic_critique(market, forecast)
        else:
            print(f"   🔎 LLM Critiquing forecast...")
            critic = critique_forecast(market, forecast)
        total_tokens += critic.tokens_used
        print(f"   📊 Critic adjusted: {critic.adjusted_probability:.1%}")
        print(f"   🚨 Major flaws: {', '.join(critic.major_flaws) if critic.major_flaws else 'None'}")
        print(f"   ✅ Should trade: {'Yes' if critic.should_trade else 'No'}")
        
        # Step 3: TRADE DECISION
        decision = make_trade_decision(market, forecast, critic, balance)
        
        print(f"\n   📋 DECISION: {decision.action}")
        print(f"   📝 {decision.reason}")
        
        if decision.action == "SKIP":
            trades_skipped += 1
            log_trade(market, decision, {}, dry_run)
            continue
        
        # Step 4: EXECUTE
        side = "yes" if decision.action == "BUY_YES" else "no"
        
        print(f"   💰 Executing: {decision.action} × {decision.contracts} @ {decision.price_cents}¢")
        print(f"      Cost: ${decision.contracts * decision.price_cents / 100:.2f}")
        
        order_result = place_order(
            ticker=market.ticker,
            side=side,
            price_cents=decision.price_cents,
            count=decision.contracts,
            dry_run=dry_run
        )
        
        if dry_run:
            print(f"   🧪 DRY RUN: Order simulated")
        else:
            if "error" in order_result:
                print(f"   ❌ Order failed: {order_result['error']}")
            else:
                print(f"   ✅ Order placed! ID: {order_result.get('order', {}).get('order_id', 'N/A')}")
        
        trades_executed += 1
        existing_tickers.add(market.ticker)
        
        log_trade(market, decision, order_result, dry_run)
        
        # Brief pause between trades
        time.sleep(1)
    
    # Cycle summary
    cycle_duration = time.time() - cycle_start
    
    forecaster_mode = "heuristic" if use_heuristic else "llm"
    
    print(f"\n{'='*70}")
    print(f"📊 CYCLE SUMMARY")
    print(f"   Forecaster: {'🧮 HEURISTIC (built-in)' if use_heuristic else '🧠 LLM (Claude)'}")
    print(f"   Duration: {cycle_duration:.1f}s")
    print(f"   Markets scanned: {len(markets)}")
    print(f"   Markets analyzed: {min(len(top_markets), max_markets_to_analyze)}")
    print(f"   Trades executed: {trades_executed}")
    print(f"   Trades skipped: {trades_skipped}")
    if not use_heuristic:
        print(f"   Total tokens used: {total_tokens:,}")
        print(f"   Est. API cost: ${total_tokens * 0.000004:.4f}")
    print(f"{'='*70}")
    
    # Log cycle stats
    log_cycle({
        "dry_run": dry_run,
        "forecaster_mode": forecaster_mode,
        "duration_seconds": round(cycle_duration, 1),
        "markets_scanned": len(markets),
        "markets_analyzed": min(len(top_markets), max_markets_to_analyze),
        "trades_executed": trades_executed,
        "trades_skipped": trades_skipped,
        "total_tokens": total_tokens,
        "balance": balance,
        "positions": num_positions
    })


# ============== CLI ==============

def _override_config(min_edge: float, kelly_fraction: float):
    """Override module-level config variables."""
    global MIN_EDGE, KELLY_FRACTION
    MIN_EDGE = min_edge
    KELLY_FRACTION = kelly_fraction

def main():
    parser = argparse.ArgumentParser(
        description="Kalshi AutoTrader v3 - Claude-as-Forecaster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kalshi-autotrader-v3.py                    # Dry run, analyze top 10
  python kalshi-autotrader-v3.py --live              # Live trading
  python kalshi-autotrader-v3.py --markets 20        # Analyze more markets
  python kalshi-autotrader-v3.py --max-trades 5      # Allow more trades per cycle
  python kalshi-autotrader-v3.py --loop 300           # Run every 5 minutes
        """
    )
    
    parser.add_argument("--live", action="store_true", 
                       help="Enable LIVE trading (default: dry run)")
    parser.add_argument("--markets", type=int, default=10,
                       help="Max markets to analyze per cycle (default: 10)")
    parser.add_argument("--max-trades", type=int, default=3,
                       help="Max trades per cycle (default: 3)")
    parser.add_argument("--loop", type=int, default=0,
                       help="Loop interval in seconds (0 = single run)")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE,
                       help=f"Minimum edge to trade (default: {MIN_EDGE})")
    parser.add_argument("--kelly", type=float, default=KELLY_FRACTION,
                       help=f"Kelly fraction (default: {KELLY_FRACTION})")
    
    args = parser.parse_args()
    
    # Override module-level config from args
    _override_config(args.min_edge, args.kelly)
    
    dry_run = not args.live
    
    if args.live:
        print("⚠️  LIVE TRADING MODE ENABLED!")
        print("    Press Ctrl+C within 5 seconds to abort...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)
    
    if args.loop > 0:
        print(f"🔄 Loop mode: running every {args.loop}s")
        while True:
            try:
                run_cycle(
                    dry_run=dry_run,
                    max_markets_to_analyze=args.markets,
                    max_trades=args.max_trades
                )
                print(f"\n⏰ Next cycle in {args.loop}s...")
                time.sleep(args.loop)
            except KeyboardInterrupt:
                print("\n\n👋 Stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Cycle error: {e}")
                traceback.print_exc()
                time.sleep(30)  # Wait before retry
    else:
        run_cycle(
            dry_run=dry_run,
            max_markets_to_analyze=args.markets,
            max_trades=args.max_trades
        )


if __name__ == "__main__":
    main()
