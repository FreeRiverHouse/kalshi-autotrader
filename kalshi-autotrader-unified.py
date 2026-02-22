#!/usr/bin/env python3
"""
Kalshi AutoTrader - Unified Edition
Consolidation of v1 (crypto), v2 (infra+weather+momentum), v3 (LLM pipeline)

Architecture: FORECASTER → CRITIC → TRADER
├─ LLM forecaster (Claude via Anthropic/OpenRouter) when API key available
├─ Heuristic forecaster (sport-specific + crypto models) as fallback
├─ Weather forecaster (NWS integration from v2)
├─ Crypto signal enrichment (sentiment, momentum, regime from v2)
└─ Kelly criterion position sizing with conservative fraction

Features from each version:
  V1: Core crypto probability model, fear & greed, momentum basics
  V2: Weather/NWS, crypto news sentiment, regime detection, momentum multi-TF,
      latency tracking, rate limiting, circuit breaker, market holidays, VIX,
      portfolio concentration, stop-loss, dynamic volatility, OHLC caching
  V3: Forecaster→Critic→Trader pipeline, heuristic sport models, multi-market
      scanning (not just crypto), split BUY_YES/BUY_NO thresholds, parlay analysis

Paper mode by default (DRY_RUN = True). Use --live for real trading.
Virtual $100 balance when paper mode and balance < $1.

Author: Clawd (unified from v1+v2+v3)
"""

import os
import sys
import json
import time
import math
import re
import argparse
import base64
import signal
import logging
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ============================================================================
# STRUCTURED LOGGING (JSON)
# ============================================================================

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Add extra fields if present
        for key in ("ticker", "action", "edge", "contracts", "price_cents",
                     "cost_cents", "balance", "positions", "cycle_id", "duration_s",
                     "error_type", "market_type", "alert_type", "component"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(json_log_path: Path = None, console_level: int = logging.INFO):
    """Set up structured JSON logging + human-readable console output."""
    root_logger = logging.getLogger("autotrader")
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_fmt = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # JSON file handler
    if json_log_path:
        json_log_path.parent.mkdir(parents=True, exist_ok=True)
        json_handler = logging.FileHandler(str(json_log_path), encoding="utf-8")
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(json_handler)

    return root_logger


# Initialize logger (will be configured in main())
log = logging.getLogger("autotrader")

# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================

class GracefulShutdown:
    """Handle SIGTERM/SIGINT for graceful shutdown."""

    def __init__(self):
        self.should_stop = False
        self.in_trade = False
        self.current_cycle = 0
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        self._original_sigint = signal.getsignal(signal.SIGINT)

    def install_handlers(self):
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        if self.in_trade:
            log.warning(f"🛑 {sig_name} received during trade execution — finishing current trade, then stopping",
                        extra={"component": "shutdown", "alert_type": "graceful_shutdown"})
        else:
            log.warning(f"🛑 {sig_name} received — stopping after current cycle",
                        extra={"component": "shutdown", "alert_type": "graceful_shutdown"})
        self.should_stop = True

        # Write shutdown alert
        _write_alert("shutdown", {
            "signal": sig_name,
            "cycle": self.current_cycle,
            "in_trade": self.in_trade,
            "message": f"Graceful shutdown initiated by {sig_name}",
        })

    def check_stop(self) -> bool:
        return self.should_stop

    def enter_trade(self):
        self.in_trade = True

    def exit_trade(self):
        self.in_trade = False

    def cleanup(self):
        """Restore original signal handlers."""
        signal.signal(signal.SIGTERM, self._original_sigterm)
        signal.signal(signal.SIGINT, self._original_sigint)


# Global shutdown handler
shutdown = GracefulShutdown()

# ============================================================================
# REAL-TIME ALERTS (.alert files)
# ============================================================================

ALERT_DIR = Path(__file__).parent
ERROR_COUNTER = defaultdict(int)  # Track error clusters
ERROR_WINDOW_START = time.time()
ERROR_CLUSTER_THRESHOLD = 5  # N errors in window = alert
ERROR_CLUSTER_WINDOW_SEC = 300  # 5 minute window

DRAWDOWN_THRESHOLDS = [0.05, 0.10, 0.20]  # 5%, 10%, 20% drawdown alerts
DRAWDOWN_ALERTED = set()  # Track which thresholds already alerted


def _write_alert(alert_type: str, data: dict):
    """Write a .alert file for monitoring tools to pick up."""
    alert_file = ALERT_DIR / f"kalshi-{alert_type}.alert"
    alert_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": alert_type,
        "severity": data.get("severity", "warning"),
        **data,
    }
    try:
        with open(alert_file, "w") as f:
            json.dump(alert_data, f, indent=2)
        log.info(f"🚨 Alert written: {alert_file.name}",
                 extra={"component": "alerts", "alert_type": alert_type})
    except Exception as e:
        log.error(f"Failed to write alert file: {e}")


def check_drawdown_alert(balance: float, peak_balance: float):
    """Alert if drawdown exceeds thresholds."""
    if peak_balance <= 0:
        return
    drawdown = (peak_balance - balance) / peak_balance
    for threshold in DRAWDOWN_THRESHOLDS:
        if drawdown >= threshold and threshold not in DRAWDOWN_ALERTED:
            DRAWDOWN_ALERTED.add(threshold)
            severity = "critical" if threshold >= 0.20 else ("warning" if threshold >= 0.10 else "info")
            _write_alert("drawdown", {
                "severity": severity,
                "drawdown_pct": round(drawdown * 100, 2),
                "threshold_pct": round(threshold * 100, 2),
                "balance": round(balance, 2),
                "peak_balance": round(peak_balance, 2),
                "message": f"Drawdown {drawdown:.1%} exceeds {threshold:.0%} threshold",
            })
            log.warning(f"🚨 DRAWDOWN ALERT: {drawdown:.1%} (threshold: {threshold:.0%}) | "
                        f"Balance: ${balance:.2f} / Peak: ${peak_balance:.2f}",
                        extra={"component": "risk", "alert_type": "drawdown",
                               "balance": balance})


def record_error(error_type: str, details: str = ""):
    """Track errors and alert on error clusters."""
    global ERROR_WINDOW_START
    now = time.time()

    # Reset window if expired
    if now - ERROR_WINDOW_START > ERROR_CLUSTER_WINDOW_SEC:
        ERROR_COUNTER.clear()
        ERROR_WINDOW_START = now

    ERROR_COUNTER[error_type] += 1
    total_errors = sum(ERROR_COUNTER.values())

    if total_errors >= ERROR_CLUSTER_THRESHOLD:
        _write_alert("error-cluster", {
            "severity": "critical",
            "total_errors": total_errors,
            "window_seconds": ERROR_CLUSTER_WINDOW_SEC,
            "error_types": dict(ERROR_COUNTER),
            "latest_error": error_type,
            "latest_details": details[:500],
            "message": f"{total_errors} errors in {ERROR_CLUSTER_WINDOW_SEC}s window",
        })
        log.error(f"🚨 ERROR CLUSTER: {total_errors} errors in {ERROR_CLUSTER_WINDOW_SEC}s",
                  extra={"component": "alerts", "alert_type": "error_cluster",
                         "error_type": error_type})
        # Reset after alerting
        ERROR_COUNTER.clear()
        ERROR_WINDOW_START = now


# ============================================================================
# POSITION & RISK LIMITS
# ============================================================================

# Per-market exposure limits
MAX_EXPOSURE_PER_MARKET_CENTS = 1000   # $10 max exposure per single market
MAX_EXPOSURE_PER_CATEGORY_PCT = 0.30   # 30% of portfolio in one category
MAX_CONCURRENT_POSITIONS = 15          # Hard cap on open positions
DAILY_LOSS_CAP_PCT = 0.10              # 10% of portfolio as daily loss cap
MAX_DAILY_TRADES = 50                  # Circuit breaker on trade count

# Peak balance tracking file
PEAK_BALANCE_FILE = Path(__file__).parent / "kalshi-peak-balance.json"


def load_peak_balance() -> float:
    """Load historical peak balance for drawdown calculation."""
    try:
        if PEAK_BALANCE_FILE.exists():
            with open(PEAK_BALANCE_FILE) as f:
                data = json.load(f)
            return data.get("peak_balance", 0)
    except Exception:
        pass
    return 0


def save_peak_balance(balance: float):
    """Save peak balance if new high."""
    current_peak = load_peak_balance()
    if balance > current_peak:
        try:
            with open(PEAK_BALANCE_FILE, "w") as f:
                json.dump({
                    "peak_balance": round(balance, 2),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
        except Exception:
            pass


def check_position_risk_limits(market, decision, balance: float,
                                positions: list, daily_pnl: dict) -> tuple:
    """
    Comprehensive risk check before placing a trade.
    Returns (allowed: bool, reason: str).
    """
    # 1. Max concurrent positions
    if len(positions) >= MAX_CONCURRENT_POSITIONS:
        return False, f"Max concurrent positions ({MAX_CONCURRENT_POSITIONS}) reached"

    # 2. Per-market exposure check
    cost_cents = decision.contracts * decision.price_cents
    if cost_cents > MAX_EXPOSURE_PER_MARKET_CENTS:
        return False, (f"Per-market exposure ${cost_cents/100:.2f} exceeds "
                       f"${MAX_EXPOSURE_PER_MARKET_CENTS/100:.2f} limit")

    # 3. Check existing exposure on same market
    existing_exposure = 0
    for pos in positions:
        if pos.get("ticker") == market.ticker:
            existing_exposure += abs(pos.get("market_exposure", 0))
    if existing_exposure + cost_cents > MAX_EXPOSURE_PER_MARKET_CENTS:
        return False, (f"Combined exposure on {market.ticker} would be "
                       f"${(existing_exposure + cost_cents)/100:.2f}")

    # 4. Category concentration check
    category = market.category or "unknown"
    cat_exposure = 0
    for pos in positions:
        # Approximate: count positions in same category
        if pos.get("ticker", "").split("-")[0] == market.ticker.split("-")[0]:
            cat_exposure += abs(pos.get("market_exposure", 0))
    total_portfolio_cents = int(balance * 100)
    if total_portfolio_cents > 0:
        cat_pct = (cat_exposure + cost_cents) / total_portfolio_cents
        if cat_pct > MAX_EXPOSURE_PER_CATEGORY_PCT:
            return False, (f"Category concentration {cat_pct:.0%} exceeds "
                           f"{MAX_EXPOSURE_PER_CATEGORY_PCT:.0%}")

    # 5. Dynamic daily loss cap (% of portfolio)
    dynamic_loss_cap = int(balance * DAILY_LOSS_CAP_PCT * 100)
    effective_cap = max(DAILY_LOSS_LIMIT_CENTS, dynamic_loss_cap)
    net_pnl = daily_pnl.get("net_pnl_cents", 0)
    if net_pnl < -effective_cap:
        return False, (f"Daily loss ${abs(net_pnl)/100:.2f} exceeds dynamic cap "
                       f"${effective_cap/100:.2f}")

    # 6. Max daily trade count
    trades_today = daily_pnl.get("trades_today", 0)
    if trades_today >= MAX_DAILY_TRADES:
        return False, f"Max daily trades ({MAX_DAILY_TRADES}) reached"

    return True, "OK"

# ============================================================================
# OPTIONAL MODULE IMPORTS (from v2 ecosystem)
# ============================================================================

# Crypto news sentiment (v2's T661 - Grok Fundamental strategy)
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("crypto_news_search",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto-news-search.py"))
    crypto_news_module = module_from_spec(spec)
    spec.loader.exec_module(crypto_news_module)
    get_crypto_sentiment = crypto_news_module.get_crypto_sentiment
    NEWS_SEARCH_AVAILABLE = True
except Exception:
    NEWS_SEARCH_AVAILABLE = False
    def get_crypto_sentiment(asset="both"):
        return {"sentiment": "neutral", "confidence": 0.5, "edge_adjustment": 0,
                "should_trade": True, "reasons": []}

# NWS weather forecast (v2's T422)
try:
    weather_spec = spec_from_file_location("nws_weather_forecast",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "nws-weather-forecast.py"))
    weather_module = module_from_spec(weather_spec)
    weather_spec.loader.exec_module(weather_module)
    parse_kalshi_weather_ticker = weather_module.parse_kalshi_weather_ticker
    calculate_weather_edge = weather_module.calculate_weather_edge
    fetch_weather_forecast = weather_module.fetch_forecast
    NWS_POINTS = weather_module.NWS_POINTS
    WEATHER_MODULE_AVAILABLE = True
except Exception:
    WEATHER_MODULE_AVAILABLE = False

# Market holiday checker (v2's T414)
try:
    holiday_spec = spec_from_file_location("check_market_holiday",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "check-market-holiday.py"))
    holiday_module = module_from_spec(holiday_spec)
    holiday_spec.loader.exec_module(holiday_module)
    is_market_holiday = holiday_module.is_holiday
    HOLIDAY_CHECK_AVAILABLE = True
except Exception:
    HOLIDAY_CHECK_AVAILABLE = False
    def is_market_holiday(check_date=None):
        return False, None

# ============================================================================
# CONFIGURATION
# ============================================================================

# Credentials — loaded from environment or .kalshi-private-key.pem
API_KEY_ID = os.environ.get("KALSHI_API_KEY_ID", "4308d1ca-585e-4b73-be82-5c0968b9a59a")
_key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.kalshi-private-key.pem')
if os.path.exists(_key_file):
    with open(_key_file) as _f:
        PRIVATE_KEY = _f.read().strip()
elif os.environ.get("KALSHI_PRIVATE_KEY"):
    PRIVATE_KEY = os.environ["KALSHI_PRIVATE_KEY"]
else:
    print("❌ Kalshi private key not found! Set KALSHI_PRIVATE_KEY env or create .kalshi-private-key.pem")
    sys.exit(1)

BASE_URL = "https://api.elections.kalshi.com"

# ── Paper / Live mode ──
DRY_RUN = True  # Paper mode by default. Use --live to override.
VIRTUAL_BALANCE = 100.0  # Virtual balance for paper mode when real balance < $1

# ── Trading parameters (data-driven from v3's 132 settled trades analysis) ──
# BUY_NO: 76% WR overall → low bar.  BUY_YES: 19% WR overall → high bar.
MIN_EDGE_BUY_NO  = 0.01   # 1% min for BUY_NO  (78% WR even at tiny edges)
MIN_EDGE_BUY_YES = 0.05   # 5% min for BUY_YES (only profitable bucket: 5-10%)
MIN_EDGE = 0.01            # Global minimum (legacy compat)
MAX_EDGE_CAP = 0.10        # Cap edges >10% (overconfident forecaster at >10%: 0% WR)
# ── Dynamic edge caps by market type (GROK-TRADE-003) ──
SINGLE_LEG_EDGE_CAP = 0.15        # 15% cap for single-leg markets
PARLAY_BASE_EDGE_CAP = 0.35       # 35% base cap for parlays (2 legs)
PARLAY_MAX_EDGE_CAP = 0.40        # 40% hard ceiling for parlays (any leg count)
PARLAY_LEG_SCALE_FACTOR = 0.02    # +2% per additional leg beyond 2
MAX_POSITION_PCT = 0.05    # Max 5% of portfolio per position
KELLY_FRACTION = 0.15      # Aggressive in paper mode for data collection
MIN_BET_CENTS = 5
MAX_BET_CENTS = 500
MAX_POSITIONS = 15

# ── Parlay strategy (from v3 data) ──
PARLAY_ONLY_NO = True       # On multi-leg parlays, primarily take BUY_NO
PARLAY_YES_EXCEPTION = True # Allow BUY_YES on 2-leg parlays if edge > 5%

# ── Market scanning filters ──
MIN_VOLUME = 200
MIN_LIQUIDITY = 1000
MAX_DAYS_TO_EXPIRY = 30
MIN_DAYS_TO_EXPIRY = 0.02  # ~30 minutes
MIN_PRICE_CENTS = 5
MAX_PRICE_CENTS = 95

# ── Circuit breaker / daily loss ──
CIRCUIT_BREAKER_THRESHOLD = 5  # Pause after N consecutive losses
CIRCUIT_BREAKER_COOLDOWN_HOURS = 4
DAILY_LOSS_LIMIT_CENTS = 500   # $5 daily loss limit

# ── Weather markets (v2's T422) ──
WEATHER_ENABLED = os.getenv("WEATHER_ENABLED", "false").lower() in ("true", "1", "yes")
WEATHER_CITIES = ["NYC", "CHI", "DEN"]
WEATHER_MAX_HOURS_TO_SETTLEMENT = 48
WEATHER_MIN_EDGE = 0.15
WEATHER_MIN_FORECAST_STRIKE_GAP = 2.0
WEATHER_MAX_MARKET_CONVICTION = 0.85
WEATHER_MIN_OUR_PROB = 0.05

# ── Sports event tickers to scan (from v3) ──
SPORTS_EVENT_TICKERS = [
    "KXCBBSPREAD", "KXCBBTOTAL", "KXCBBML",
    "KXNCAABSPREAD", "KXNCAABTOTAL",
    "KXNBASPREAD", "KXNBATOTAL", "KXNBAML",
    "KXNFLSPREAD", "KXNFLTOTAL", "KXNFLML",
    "KXNHLSPREAD", "KXNHLTOTAL", "KXNHLML",
    "KXMLBSPREAD", "KXMLBTOTAL", "KXMLBML",
    "KXSOCCERML", "KXSOCCERTOTAL",
    "KXMVESPORTSMULTIGAMEEXTENDED", "KXMVESPORTSMULTIGAME",
    "KXMULTISPORT",
]

# ── Crypto volatility defaults (v2 calibrated) ──
BTC_HOURLY_VOL = 0.003
ETH_HOURLY_VOL = 0.004
CRYPTO_FAT_TAIL_MULTIPLIER = 1.0  # Disabled after v2 disaster analysis

# ── Logging paths ──
PROJECT_ROOT = Path(__file__).parent.parent
TRADE_LOG_FILE = PROJECT_ROOT / "data" / "trading" / "kalshi-unified-trades.jsonl"
CYCLE_LOG_FILE = PROJECT_ROOT / "data" / "trading" / "kalshi-unified-cycles.jsonl"
SKIP_LOG_FILE  = PROJECT_ROOT / "data" / "trading" / "kalshi-unified-skips.jsonl"
# Also write to legacy location for compatibility
LEGACY_TRADE_LOG = Path(__file__).parent / "kalshi-trades.jsonl"

# ── Alert files (v2 compat) ──
CIRCUIT_BREAKER_STATE_FILE = Path(__file__).parent / "kalshi-circuit-breaker.json"
DAILY_LOSS_PAUSE_FILE = Path(__file__).parent / "kalshi-daily-pause.json"

# ── Latency tracking (from v2) ──
API_LATENCY_LOG = defaultdict(list)
LATENCY_PROFILE_WINDOW = 50

# ── Rate limit tracking (from v2) ──
API_RATE_LIMITS = {
    "kalshi": {"calls_per_hour": 0, "limit": 1000},
    "coingecko": {"calls_per_hour": 0, "limit": 30},
}
API_RATE_WINDOW_START = time.time()

# ── External API cache (from v2) ──
EXT_API_CACHE = {}
EXT_API_CACHE_TTL = 60

# ============================================================================
# LLM CONFIGURATION (from v3)
# ============================================================================

def get_llm_config():
    """Get LLM configuration. Priority: ANTHROPIC_API_KEY → OPENROUTER_API_KEY → .env.trading"""
    # 1. Direct Anthropic key
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key.startswith("sk-ant-api"):
        return {
            "provider": "anthropic", "api_key": key,
            "base_url": "https://api.anthropic.com/v1/messages",
            "model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            "headers": {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        }

    # 2. OpenRouter
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return {
            "provider": "openrouter", "api_key": key,
            "base_url": "https://openrouter.ai/api/v1/chat/completions",
            "model": os.environ.get("CLAUDE_MODEL", "anthropic/claude-sonnet-4"),
            "headers": {"Authorization": f"Bearer {key}", "content-type": "application/json"}
        }

    # 3. .env.trading file
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
            auth_h = {"Authorization": f"Bearer {key}"} if key.startswith("sk-ant-oat") else {"x-api-key": key}
            return {
                "provider": "anthropic", "api_key": key,
                "base_url": "https://api.anthropic.com/v1/messages",
                "model": env_vars.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
                "headers": {**auth_h, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            }
        if "OPENROUTER_API_KEY" in env_vars:
            key = env_vars["OPENROUTER_API_KEY"]
            return {
                "provider": "openrouter", "api_key": key,
                "base_url": "https://openrouter.ai/api/v1/chat/completions",
                "model": env_vars.get("CLAUDE_MODEL", "anthropic/claude-sonnet-4"),
                "headers": {"Authorization": f"Bearer {key}", "content-type": "application/json"}
            }

    # 4. OAuth token
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key.startswith("sk-ant-oat"):
        return {
            "provider": "anthropic", "api_key": key,
            "base_url": "https://api.anthropic.com/v1/messages",
            "model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            "headers": {"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01", "content-type": "application/json"}
        }

    return None

LLM_CONFIG = get_llm_config()

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MarketInfo:
    """Parsed Kalshi market info."""
    ticker: str
    title: str
    subtitle: str
    category: str
    yes_price: int          # cents
    no_price: int           # cents
    volume: int
    open_interest: int
    expiry: str             # ISO datetime
    status: str
    result: str
    yes_bid: int = 0
    yes_ask: int = 0
    last_price: int = 0

    @property
    def market_prob(self) -> float:
        return self.yes_price / 100.0

    @property
    def days_to_expiry(self) -> float:
        try:
            exp = datetime.fromisoformat(self.expiry.replace('Z', '+00:00'))
            return max(0, (exp - datetime.now(timezone.utc)).total_seconds() / 86400)
        except Exception:
            return 999

@dataclass
class ForecastResult:
    probability: float
    reasoning: str
    confidence: str  # "low", "medium", "high"
    key_factors: list = field(default_factory=list)
    raw_response: str = ""
    model_used: str = ""
    tokens_used: int = 0

@dataclass
class CriticResult:
    adjusted_probability: float
    major_flaws: list = field(default_factory=list)
    minor_flaws: list = field(default_factory=list)
    should_trade: bool = True
    reasoning: str = ""
    tokens_used: int = 0

@dataclass
class TradeDecision:
    action: str               # "BUY_YES", "BUY_NO", "SKIP"
    edge: float
    kelly_size: float
    contracts: int
    price_cents: int
    reason: str
    forecast: Optional[ForecastResult] = None
    critic: Optional[CriticResult] = None

# ============================================================================
# LATENCY & RATE LIMIT TRACKING (from v2)
# ============================================================================

def record_api_latency(endpoint: str, latency_ms: float):
    API_LATENCY_LOG[endpoint].append((time.time(), latency_ms))
    if len(API_LATENCY_LOG[endpoint]) > LATENCY_PROFILE_WINDOW:
        API_LATENCY_LOG[endpoint] = API_LATENCY_LOG[endpoint][-LATENCY_PROFILE_WINDOW:]

def get_avg_latency(endpoint: str) -> float:
    entries = API_LATENCY_LOG.get(endpoint, [])
    if not entries:
        return 0
    return sum(lat for _, lat in entries) / len(entries)

def record_api_call(source: str):
    global API_RATE_WINDOW_START
    if time.time() - API_RATE_WINDOW_START > 3600:
        for src in API_RATE_LIMITS:
            API_RATE_LIMITS[src]["calls_per_hour"] = 0
        API_RATE_WINDOW_START = time.time()
    if source in API_RATE_LIMITS:
        API_RATE_LIMITS[source]["calls_per_hour"] += 1

def get_cached_response(cache_key: str):
    if cache_key not in EXT_API_CACHE:
        return None
    cached_time, cached_data = EXT_API_CACHE[cache_key]
    if time.time() - cached_time > EXT_API_CACHE_TTL:
        del EXT_API_CACHE[cache_key]
        return None
    return cached_data

def set_cached_response(cache_key: str, data):
    EXT_API_CACHE[cache_key] = (time.time(), data)

# ============================================================================
# KALSHI API (merged from v2+v3, with latency tracking)
# ============================================================================

def sign_request(method: str, path: str, timestamp: str) -> str:
    private_key = serialization.load_pem_private_key(PRIVATE_KEY.encode(), password=None)
    message = f"{timestamp}{method}{path}".encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')


def kalshi_api(method: str, path: str, body: dict = None, max_retries: int = 3) -> dict:
    """Make authenticated Kalshi API request with retries and latency tracking."""
    url = f"{BASE_URL}{path}"
    endpoint_name = path.split("/")[-1].split("?")[0]
    if "orders" in path: endpoint_name = "order"
    elif "positions" in path: endpoint_name = "positions"
    elif "balance" in path: endpoint_name = "balance"
    elif "markets" in path: endpoint_name = "markets_search"

    total_start = time.time()
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
            else:
                return {"error": f"Unknown method {method}"}

            if resp.status_code >= 500:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {"error": f"Server error {resp.status_code}"}

            latency = (time.time() - total_start) * 1000
            record_api_latency(endpoint_name, latency)
            record_api_call("kalshi")
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
        log.error(f"❌ Balance error: {result['error']}",
                  extra={"component": "api", "error_type": "balance_fetch"})
        record_error("balance_fetch", result["error"])
        return 0.0
    return result.get("balance", 0) / 100.0


def get_positions() -> list:
    result = kalshi_api("GET", "/trade-api/v2/portfolio/positions?limit=100&settlement_status=unsettled")
    if "error" in result:
        return []
    return result.get("market_positions", [])


def place_order(ticker: str, side: str, price_cents: int, count: int, dry_run: bool = True) -> dict:
    if dry_run:
        return {"dry_run": True, "ticker": ticker, "side": side, "price": price_cents,
                "count": count, "status": "simulated"}
    body = {
        "ticker": ticker, "action": "buy", "side": side, "type": "limit",
        "count": count,
        "yes_price": price_cents if side == "yes" else (100 - price_cents),
    }
    return kalshi_api("POST", "/trade-api/v2/portfolio/orders", body=body)


# ============================================================================
# EXTERNAL DATA (from v2: crypto prices, OHLC, sentiment)
# ============================================================================

def get_crypto_prices() -> Optional[dict]:
    """Get BTC/ETH prices with caching and fallback sources."""
    cached = get_cached_response("crypto_prices")
    if cached:
        return cached

    # Try CoinGecko first
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd",
            timeout=5)
        record_api_call("coingecko")
        data = resp.json()
        result = {"btc": data["bitcoin"]["usd"], "eth": data["ethereum"]["usd"]}
        set_cached_response("crypto_prices", result)
        return result
    except Exception:
        pass

    # Fallback: Binance
    try:
        btc_resp = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=5)
        eth_resp = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT", timeout=5)
        result = {"btc": float(btc_resp.json()["price"]), "eth": float(eth_resp.json()["price"])}
        set_cached_response("crypto_prices", result)
        return result
    except Exception:
        return None


def get_fear_greed_index() -> dict:
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        data = resp.json()
        return {"value": int(data["data"][0]["value"]),
                "classification": data["data"][0]["value_classification"]}
    except Exception:
        return {"value": 50, "classification": "Neutral"}


def get_crypto_ohlc(coin_id: str = "bitcoin", days: int = 7) -> list:
    """Get OHLC data from CoinGecko (or cache)."""
    cache_key = f"ohlc_{coin_id}_{days}"
    cached = get_cached_response(cache_key)
    if cached:
        return cached

    # Try local cache file first (from v2's OHLC caching)
    cache_dir = PROJECT_ROOT / "data" / "ohlc"
    cache_file = cache_dir / f"{coin_id[:3]}-ohlc.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                data = json.load(f)
            updated = data.get("updated_at", "")
            if updated:
                age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(
                    updated.replace("Z", "+00:00"))).total_seconds() / 3600
                if age_hours < 24:
                    candles = [[c.get("timestamp"), c.get("open"), c.get("high"),
                                c.get("low"), c.get("close")] for c in data.get("candles", [])]
                    if candles:
                        set_cached_response(cache_key, candles)
                        return candles
        except Exception:
            pass

    # Fetch from CoinGecko
    try:
        resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}",
            timeout=10)
        record_api_call("coingecko")
        ohlc = resp.json()
        if isinstance(ohlc, list) and len(ohlc) > 0:
            set_cached_response(cache_key, ohlc)
            return ohlc
    except Exception:
        pass
    return []


# ============================================================================
# MOMENTUM & REGIME DETECTION (from v2)
# ============================================================================

def calculate_momentum(ohlc_data: list, timeframe: str) -> dict:
    """Calculate momentum for a single timeframe."""
    result = {"direction": 0, "strength": 0, "pct_change": 0}
    if not ohlc_data or len(ohlc_data) < 4:
        return result

    tf_candles = {"1h": 1, "4h": 4, "24h": 24}
    n = tf_candles.get(timeframe, 4)
    if len(ohlc_data) < n:
        n = len(ohlc_data)

    current = ohlc_data[-1][4]  # close
    old = ohlc_data[-n][4] if len(ohlc_data) >= n else ohlc_data[0][4]
    if not current or not old or old == 0:
        return result

    pct_change = (current - old) / old
    result["pct_change"] = pct_change
    result["direction"] = 1 if pct_change > 0.002 else (-1 if pct_change < -0.002 else 0)
    result["strength"] = min(1.0, abs(pct_change) / 0.03)  # Normalize: 3% = full strength
    return result


def get_multi_timeframe_momentum(ohlc_data: list) -> dict:
    """Get momentum across 1h, 4h, 24h timeframes."""
    result = {"timeframes": {}, "composite_direction": 0, "composite_strength": 0, "alignment": False}
    if not ohlc_data:
        return result

    weights = {"1h": 0.5, "4h": 0.3, "24h": 0.2}
    comp_dir = 0
    comp_str = 0

    for tf in ["1h", "4h", "24h"]:
        mom = calculate_momentum(ohlc_data, tf)
        result["timeframes"][tf] = mom
        comp_dir += mom["direction"] * weights[tf]
        comp_str += mom["strength"] * weights[tf]

    result["composite_direction"] = comp_dir
    result["composite_strength"] = comp_str

    # Check alignment (all same direction)
    dirs = [result["timeframes"][tf]["direction"] for tf in ["1h", "4h", "24h"]]
    result["alignment"] = all(d > 0 for d in dirs) or all(d < 0 for d in dirs)
    return result


def detect_market_regime(ohlc_data: list, momentum: dict) -> dict:
    """Detect regime: trending_bullish, trending_bearish, sideways, choppy."""
    result = {"regime": "sideways", "confidence": 0.5, "volatility": "normal", "dynamic_min_edge": MIN_EDGE}
    if not ohlc_data or len(ohlc_data) < 24:
        return result

    current_price = ohlc_data[-1][4]
    if not current_price:
        return result

    price_4h = ohlc_data[-4][4] if len(ohlc_data) >= 4 else current_price
    price_24h = ohlc_data[0][4] if len(ohlc_data) >= 24 else current_price
    change_4h = (current_price - price_4h) / price_4h if price_4h else 0
    change_24h = (current_price - price_24h) / price_24h if price_24h else 0

    # Volatility from candle ranges
    ranges = []
    for c in ohlc_data[-24:]:
        if c and len(c) >= 4 and c[3] > 0:
            ranges.append((c[2] - c[3]) / c[3])
    avg_range = sum(ranges) / len(ranges) if ranges else 0
    vol_class = ("very_low" if avg_range < 0.003 else "low" if avg_range < 0.005 else
                 "normal" if avg_range < 0.01 else "high" if avg_range < 0.02 else "very_high")
    result["volatility"] = vol_class

    mom_dir = momentum.get("composite_direction", 0)
    mom_str = momentum.get("composite_strength", 0)
    mom_aligned = momentum.get("alignment", False)

    is_bullish = change_4h > 0.005 and change_24h > 0.01 and mom_dir > 0.2
    is_bearish = change_4h < -0.005 and change_24h < -0.01 and mom_dir < -0.2

    if is_bullish and mom_aligned:
        result["regime"] = "trending_bullish"
        result["confidence"] = min(0.9, 0.5 + abs(change_24h) * 10 + mom_str * 0.3)
    elif is_bearish and mom_aligned:
        result["regime"] = "trending_bearish"
        result["confidence"] = min(0.9, 0.5 + abs(change_24h) * 10 + mom_str * 0.3)
    elif vol_class in ("high", "very_high") and abs(change_24h) < 0.02:
        result["regime"] = "choppy"
        result["confidence"] = 0.7
    else:
        result["regime"] = "sideways"
        result["confidence"] = 0.6

    # Dynamic min edge per regime
    if result["regime"] in ("trending_bullish", "trending_bearish"):
        result["dynamic_min_edge"] = 0.07
    elif result["regime"] == "choppy":
        result["dynamic_min_edge"] = 0.08
    else:
        result["dynamic_min_edge"] = 0.06

    if vol_class in ("high", "very_high"):
        result["dynamic_min_edge"] += 0.01

    result["dynamic_min_edge"] = max(0.03 if DRY_RUN else 0.05, min(0.20, result["dynamic_min_edge"]))
    return result


# ============================================================================
# LLM FORECASTER & CRITIC (from v3)
# ============================================================================

def call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> dict:
    """Call LLM API. Supports Anthropic, OpenRouter, OpenAI-compatible."""
    if not LLM_CONFIG:
        return {"error": "No LLM API key", "content": "", "tokens_used": 0}

    provider = LLM_CONFIG["provider"]
    try:
        if provider == "anthropic":
            body = {"model": LLM_CONFIG["model"], "max_tokens": max_tokens,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}]}
            resp = requests.post(LLM_CONFIG["base_url"], headers=LLM_CONFIG["headers"], json=body, timeout=60)
            if resp.status_code != 200:
                return {"error": f"API {resp.status_code}: {resp.text[:300]}", "content": "", "tokens_used": 0}
            data = resp.json()
            content = "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")
            usage = data.get("usage", {})
            return {"content": content, "tokens_used": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    "model": data.get("model", LLM_CONFIG["model"])}
        else:
            body = {"model": LLM_CONFIG["model"], "max_tokens": max_tokens,
                    "messages": [{"role": "system", "content": system_prompt},
                                 {"role": "user", "content": user_prompt}]}
            resp = requests.post(LLM_CONFIG["base_url"], headers=LLM_CONFIG["headers"], json=body, timeout=60)
            if resp.status_code != 200:
                return {"error": f"API {resp.status_code}: {resp.text[:300]}", "content": "", "tokens_used": 0}
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            return {"content": content,
                    "tokens_used": usage.get("total_tokens", 0) or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
                    "model": data.get("model", LLM_CONFIG["model"])}
    except Exception as e:
        return {"error": str(e), "content": "", "tokens_used": 0}


def parse_probability(text: str) -> Optional[float]:
    """Parse probability from LLM response."""
    match = re.search(r'PROBABILITY:\s*(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100.0
    match = re.search(r'(?:estimate|probability|chance)\s+(?:is|of|at)\s+(?:approximately\s+)?(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100.0
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
    if matches:
        val = float(matches[-1])
        if 1 <= val <= 99:
            return val / 100.0
    return None


def forecast_market_llm(market: MarketInfo, context: dict = None) -> ForecastResult:
    """Use Claude to estimate true probability (v3 pipeline)."""
    system_prompt = """You are an expert forecaster for prediction markets. Estimate the TRUE probability of events.

Rules:
1. Think step by step about base rates, current conditions, and relevant factors
2. Be calibrated - if uncertain, reflect that in your probability
3. Consider time remaining until expiry
4. Account for both sides

End your response with exactly:
PROBABILITY: XX%
CONFIDENCE: [low/medium/high]
KEY_FACTORS: [factor1], [factor2], [factor3]"""

    try:
        exp_dt = datetime.fromisoformat(market.expiry.replace('Z', '+00:00'))
        time_left = exp_dt - datetime.now(timezone.utc)
        if time_left.total_seconds() < 3600:
            time_desc = f"{int(time_left.total_seconds() / 60)} minutes"
        elif time_left.days > 0:
            time_desc = f"{time_left.days}d {time_left.seconds // 3600}h"
        else:
            time_desc = f"{time_left.seconds // 3600}h {(time_left.seconds % 3600) // 60}m"
        expiry_str = exp_dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        time_desc = "unknown"
        expiry_str = market.expiry

    # Build context from v2 signals if available
    extra_context = ""
    if context:
        if context.get("crypto_prices"):
            extra_context += f"\nCurrent crypto prices: BTC ${context['crypto_prices'].get('btc', 0):,.0f}, ETH ${context['crypto_prices'].get('eth', 0):,.0f}"
        if context.get("sentiment"):
            s = context["sentiment"]
            extra_context += f"\nFear & Greed Index: {s.get('value', 50)} ({s.get('classification', 'Neutral')})"
        if context.get("news_sentiment"):
            ns = context["news_sentiment"]
            extra_context += f"\nCrypto news sentiment: {ns.get('sentiment', 'neutral')} (conf: {ns.get('confidence', 0.5):.0%})"
        if context.get("momentum"):
            m = context["momentum"]
            extra_context += f"\nMomentum: composite_dir={m.get('composite_direction', 0):.2f}, aligned={m.get('alignment', False)}"
        if context.get("regime"):
            r = context["regime"]
            extra_context += f"\nMarket regime: {r.get('regime', 'unknown')} (conf: {r.get('confidence', 0):.0%}), vol: {r.get('volatility', 'normal')}"

    user_prompt = f"""Analyze this prediction market and estimate the true probability:

MARKET: {market.title}
{f'DETAILS: {market.subtitle}' if market.subtitle else ''}
CATEGORY: {market.category}
YES PRICE: {market.yes_price}¢ (implies {market.market_prob:.0%})
NO PRICE: {market.no_price}¢
VOLUME: {market.volume:,} contracts
EXPIRY: {expiry_str} ({time_desc} remaining)
TICKER: {market.ticker}
{extra_context}

Today: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

What is the TRUE probability this resolves YES?"""

    result = call_claude(system_prompt, user_prompt, max_tokens=1500)

    if result.get("error"):
        return ForecastResult(probability=market.market_prob, reasoning=f"Error: {result['error']}",
                              confidence="low", model_used="error", tokens_used=0)

    content = result["content"]
    prob = parse_probability(content) or market.market_prob
    confidence = "medium"
    conf_match = re.search(r'CONFIDENCE:\s*(low|medium|high)', content, re.IGNORECASE)
    if conf_match:
        confidence = conf_match.group(1).lower()
    key_factors = []
    factors_match = re.search(r'KEY_FACTORS:\s*(.+)', content, re.IGNORECASE)
    if factors_match:
        key_factors = [f.strip() for f in factors_match.group(1).split(',')]

    return ForecastResult(probability=prob, reasoning=content, confidence=confidence,
                          key_factors=key_factors, raw_response=content,
                          model_used=result.get("model", ""), tokens_used=result.get("tokens_used", 0))


def critique_forecast_llm(market: MarketInfo, forecast: ForecastResult) -> CriticResult:
    """Second LLM call to critically evaluate the forecast (v3 pipeline)."""
    system_prompt = """You are a critical analyst reviewing probability forecasts. Find flaws, missing context, overconfidence.

End with:
ADJUSTED_PROBABILITY: XX%
MAJOR_FLAWS: [flaw1], [flaw2] (or NONE)
SHOULD_TRADE: [yes/no]"""

    edge = forecast.probability - market.market_prob
    user_prompt = f"""Review this forecast:

MARKET: {market.title}
PRICE: {market.yes_price}¢ (implies {market.market_prob:.0%})
VOLUME: {market.volume:,}

FORECAST: {forecast.probability:.1%} (confidence: {forecast.confidence})
REASONING: {forecast.reasoning[:1500]}

Edge: {abs(edge):.1%} ({'YES underpriced' if edge > 0 else 'NO underpriced'})

Is the forecaster overconfident? Missing factors? Is the edge real?"""

    result = call_claude(system_prompt, user_prompt, max_tokens=1200)
    if result.get("error"):
        return CriticResult(adjusted_probability=forecast.probability, should_trade=False,
                            reasoning=f"Error: {result['error']}")

    content = result["content"]
    adj_prob = parse_probability(content) or forecast.probability
    major_flaws = []
    flaws_match = re.search(r'MAJOR_FLAWS:\s*(.+)', content, re.IGNORECASE)
    if flaws_match:
        flaws_str = flaws_match.group(1).strip()
        if flaws_str.upper() != "NONE":
            major_flaws = [f.strip() for f in flaws_str.split(',') if f.strip()]
    should_trade = True
    trade_match = re.search(r'SHOULD_TRADE:\s*(yes|no)', content, re.IGNORECASE)
    if trade_match:
        should_trade = trade_match.group(1).lower() == "yes"

    return CriticResult(adjusted_probability=adj_prob, major_flaws=major_flaws,
                        should_trade=should_trade, reasoning=content,
                        tokens_used=result.get("tokens_used", 0))


# ============================================================================
# HEURISTIC FORECASTER & CRITIC (from v3 - no LLM needed)
# ============================================================================

SPORT_AVG_TOTALS = {"nba": 224, "nfl": 44, "cbb": 140, "nhl": 5.8, "mlb": 8.5, "soccer": 2.5, "unknown": 100}
SPORT_SPREAD_K = {"nba": 0.10, "nfl": 0.15, "cbb": 0.09, "nhl": 0.45, "mlb": 0.35, "soccer": 0.55, "unknown": 0.15}


def classify_market_type(market: MarketInfo) -> str:
    """Classify into: combo, spread, total, moneyline, crypto, weather, generic."""
    ticker = market.ticker.upper()
    title = market.title.lower()

    # Crypto
    if any(x in ticker for x in ("KXBTC", "KXETH", "KXSOL")):
        return "crypto"

    # Weather
    if any(x in ticker for x in ("KXHIGH", "KXLOW")):
        return "weather"

    # Combo/Parlay
    if any(x in ticker for x in ("MULTIGAME", "MULTI", "PARLAY")):
        return "combo"
    segments = [s.strip() for s in market.title.split(',')]
    leg_count = sum(1 for s in segments if s.lower().startswith('yes ') or s.lower().startswith('no '))
    if leg_count >= 2:
        return "combo"
    if title.count(" and ") >= 1 and any(w in title for w in ["win", "beat", "cover", "over", "under"]):
        return "combo"

    # Spread
    if "SPREAD" in ticker or "spread" in title or "cover" in title:
        return "spread"
    if re.search(r'[+-]\d+\.?5?\s*(points?|pts?)', title):
        return "spread"

    # Total
    if "TOTAL" in ticker or "OU" in ticker:
        return "total"
    if "over" in title and re.search(r'over\s+\d', title):
        return "total"

    # Moneyline
    if "ML" in ticker:
        return "moneyline"
    if any(w in title for w in ["to win", "will win", "wins", "beat"]) and "and" not in title:
        return "moneyline"

    return "generic"


def detect_sport(market: MarketInfo) -> str:
    ticker = market.ticker.upper()
    title = market.title.lower()
    for sport, keywords in [("nba", ["NBA"]), ("nfl", ["NFL"]), ("cbb", ["CBB", "NCAAB"]),
                             ("nhl", ["NHL"]), ("mlb", ["MLB"]), ("soccer", ["SOCCER", "MLS"])]:
        if any(k in ticker or k.lower() in title for k in keywords):
            return sport
    return "unknown"


def estimate_combo_legs(market: MarketInfo) -> int:
    """Estimate number of legs in a combo/parlay."""
    segments = [s.strip() for s in market.title.split(',')]
    leg_count = sum(1 for s in segments if s.lower().startswith('yes ') or s.lower().startswith('no '))
    if leg_count >= 2:
        return leg_count
    and_count = market.title.lower().count(" and ")
    if and_count >= 1:
        return and_count + 1
    leg_match = re.search(r'(\d+)[\s-]*(leg|team|game|pick)', market.title)
    if leg_match:
        return int(leg_match.group(1))
    return 2


def heuristic_forecast(market: MarketInfo, context: dict = None) -> ForecastResult:
    """Built-in heuristic forecaster without LLM. Uses v3's sport models + v2's crypto model."""
    market_type = classify_market_type(market)
    sport = detect_sport(market)
    market_prob = market.market_prob
    title = market.title.lower()

    prob = market_prob
    confidence = "low"
    reasoning_parts = []
    key_factors = []

    if market_type == "crypto":
        # Use v2's proper log-normal probability model for crypto
        prob, confidence, reasoning_parts, key_factors = _heuristic_crypto(market, context)

    elif market_type == "combo":
        prob, confidence, reasoning_parts, key_factors = _heuristic_combo(market, sport)

    elif market_type == "spread":
        spread = None
        match = re.search(r'by\s+over\s+(\d+\.?\d*)', title)
        if match:
            spread = float(match.group(1))
        if spread is None:
            match = re.search(r'([+-]\d+\.?\d*)\s*(points?|pts?)?', title)
            if match:
                spread = float(match.group(1))
        k = SPORT_SPREAD_K.get(sport, 0.15)
        if spread is not None:
            spread_prob = 1.0 / (1.0 + math.exp(-k * (-spread)))
            prob = max(0.05, min(0.95, spread_prob - 0.015))
            confidence = "medium"
            reasoning_parts.append(f"Spread {spread:+.1f}, logistic prob {spread_prob:.1%}, adj {prob:.1%}")
            key_factors = [f"Spread: {spread:+.1f}", f"Sport: {sport}"]
        else:
            prob = 0.7 * market_prob + 0.3 * 0.5
            key_factors = ["Unknown spread"]

    elif market_type == "total":
        total_match = re.search(r'(?:over|under|total)\s+(\d+\.?\d*)', title, re.IGNORECASE)
        avg_total = SPORT_AVG_TOTALS.get(sport, 100)
        is_over = "over" in title
        if total_match:
            total_line = float(total_match.group(1))
            deviation = (total_line - avg_total) / avg_total
            base = 0.50
            adj = -deviation * 0.15
            if is_over:
                prob = base - adj - 0.02
            else:
                prob = base + adj + 0.02
            prob = max(0.10, min(0.90, prob))
            confidence = "medium"
            key_factors = [f"Line: {total_line}", f"Avg: {avg_total}", "Public over-bias"]
        else:
            prob = 0.7 * market_prob + 0.3 * 0.5

    elif market_type == "moneyline":
        if market_prob > 0.60:
            bias = 0.02 + (market_prob - 0.60) * 0.05
            prob = min(0.95, market_prob + bias)
            confidence = "medium"
            key_factors = ["Favorite underpriced"]
        elif market_prob < 0.40:
            bias = 0.02 + (0.40 - market_prob) * 0.05
            prob = max(0.05, market_prob - bias)
            confidence = "medium"
            key_factors = ["Underdog overpriced"]
        else:
            prob = market_prob
            key_factors = ["Near toss-up"]

    else:  # generic
        prob = 0.90 * market_prob + 0.10 * 0.5
        key_factors = ["Generic model", "Mean reversion"]

    reasoning = (f"[HEURISTIC] type={market_type}, sport={sport}\n" + "\n".join(reasoning_parts))
    return ForecastResult(probability=prob, reasoning=reasoning, confidence=confidence,
                          key_factors=key_factors[:4], model_used=f"heuristic-{market_type}", tokens_used=0)


def _heuristic_crypto(market: MarketInfo, context: dict = None) -> tuple:
    """Crypto-specific heuristic using v2's log-normal model."""
    title = market.title.lower()
    subtitle = (market.subtitle or "").lower()
    ticker = market.ticker.upper()
    market_prob = market.market_prob
    reasoning_parts = []
    key_factors = []

    # Detect asset
    asset = "btc" if "BTC" in ticker else ("eth" if "ETH" in ticker else "sol")

    # Get current price from context
    prices = (context or {}).get("crypto_prices", {})
    current_price = prices.get(asset, 0)
    if not current_price:
        return market_prob, "low", ["No price data"], ["No crypto price"]

    # Extract strike from subtitle (e.g., "$88,750 or above")
    strike = None
    for text in [subtitle, title]:
        if "$" in text:
            try:
                strike_str = text.split("$")[1].split(" ")[0].replace(",", "")
                strike = float(strike_str)
                break
            except Exception:
                pass
    if not strike:
        return market_prob, "low", ["Cannot parse strike"], ["No strike price"]

    # Time to expiry
    try:
        exp = datetime.fromisoformat(market.expiry.replace('Z', '+00:00'))
        minutes_left = max(1, (exp - datetime.now(timezone.utc)).total_seconds() / 60)
    except Exception:
        minutes_left = 60

    # Get hourly vol (use context's OHLC if available)
    hourly_vol = {"btc": BTC_HOURLY_VOL, "eth": ETH_HOURLY_VOL}.get(asset, 0.005)
    ohlc_data = (context or {}).get("ohlc", {}).get(asset, [])
    if ohlc_data and len(ohlc_data) >= 10:
        try:
            closes = [c[4] for c in ohlc_data[-24:] if c and len(c) >= 5 and c[4] and c[4] > 0]
            if len(closes) >= 2:
                log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
                mean_r = sum(log_returns) / len(log_returns)
                rv = math.sqrt(sum((r - mean_r) ** 2 for r in log_returns) / len(log_returns))
                default_vol = hourly_vol
                hourly_vol = max(default_vol * 0.5, min(default_vol * 2.5, rv))
        except Exception:
            pass

    # Log-normal probability model (from v2)
    T = minutes_left / 60.0
    sigma = hourly_vol * math.sqrt(T) * CRYPTO_FAT_TAIL_MULTIPLIER
    if sigma <= 0 or current_price <= 0 or strike <= 0:
        return market_prob, "low", ["Invalid model params"], ["Bad data"]

    log_ratio = math.log(current_price / strike)
    d2 = log_ratio / sigma - sigma / 2

    # Normal CDF approximation
    def norm_cdf(x):
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p_val = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x)
        t = 1.0 / (1.0 + p_val * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
        return 0.5 * (1.0 + sign * y)

    prob_above = max(0.05, min(0.95, norm_cdf(d2)))

    # Momentum adjustment
    momentum = (context or {}).get("momentum", {}).get(asset, {})
    mom_dir = momentum.get("composite_direction", 0)
    mom_adj = mom_dir * 0.03  # Max ±3% adjustment
    prob_above = max(0.05, min(0.95, prob_above + mom_adj))

    # Sentiment adjustment
    fng = (context or {}).get("sentiment", {})
    sentiment_val = fng.get("value", 50)
    sentiment_adj = (sentiment_val - 50) / 1500  # Very small effect
    prob_above += sentiment_adj

    # News sentiment adjustment
    news = (context or {}).get("news_sentiment")
    if news and news.get("edge_adjustment"):
        prob_above = max(0.05, min(0.95, prob_above + news["edge_adjustment"] * 0.5))

    prob_above = max(0.05, min(0.95, prob_above))

    distance_pct = abs(current_price - strike) / current_price * 100
    confidence = "medium" if distance_pct > 1 else "low"

    reasoning_parts = [
        f"Log-normal: P(>{strike:,.0f})={prob_above:.1%}, "
        f"current=${current_price:,.0f}, sigma={sigma:.4f}, T={T:.1f}h",
        f"Distance: {distance_pct:.2f}%, vol: {hourly_vol*100:.3f}%/hr",
    ]
    if mom_dir:
        reasoning_parts.append(f"Momentum adj: {mom_adj:+.3f}")
    key_factors = [f"Strike ${strike:,.0f}", f"Dist {distance_pct:.1f}%", f"{asset.upper()} vol"]

    return prob_above, confidence, reasoning_parts, key_factors


def _heuristic_combo(market: MarketInfo, sport: str) -> tuple:
    """Combo/parlay heuristic from v3."""
    market_prob = market.market_prob
    num_legs = estimate_combo_legs(market)

    segments = [s.strip() for s in market.title.split(',')]
    leg_probs = []
    k = SPORT_SPREAD_K.get(sport, 0.15)

    for seg in segments:
        seg_lower = seg.lower()
        if re.search(r'over\s+\d+\.?\d*\s+(?:points|goals|runs)', seg_lower):
            leg_prob = 0.52 if seg_lower.startswith('no ') else 0.48
        elif re.search(r'wins?\s+by\s+over\s+\d', seg_lower):
            sm = re.search(r'by\s+over\s+(\d+\.?\d*)', seg_lower)
            if sm:
                sv = float(sm.group(1))
                cover_prob = 1.0 / (1.0 + math.exp(k * sv))
                leg_prob = (1.0 - cover_prob) if seg_lower.startswith('no ') else cover_prob
                leg_prob *= 0.97
            else:
                leg_prob = 0.50
        elif seg_lower.startswith('yes '):
            leg_prob = 0.60
        elif seg_lower.startswith('no '):
            leg_prob = 0.55
        else:
            leg_prob = 0.55
        leg_probs.append(max(0.20, min(0.85, leg_prob)))

    if len(leg_probs) < 2:
        implied_per_leg = market_prob ** (1.0 / num_legs)
        per_leg_adj = implied_per_leg * (1.02 if implied_per_leg > 0.65 else 0.92 if implied_per_leg < 0.55 else 0.97)
        leg_probs = [min(0.88, max(0.40, per_leg_adj))] * num_legs

    # Correlation boost
    corr_boost = 1.04 ** (num_legs - 1) if sport != "unknown" else 1.02 ** (num_legs - 1)

    true_prob = 1.0
    for lp in leg_probs:
        true_prob *= lp
    true_prob *= corr_boost

    # Favorite-longshot bias on overall parlay
    if market.yes_price < 20:
        true_prob *= 0.78
    elif market.yes_price < 30:
        true_prob *= 0.85
    elif market.yes_price > 70:
        true_prob *= 1.05

    # Leg penalty
    if num_legs >= 6: true_prob *= 0.82
    elif num_legs >= 4: true_prob *= 0.88
    elif num_legs >= 3: true_prob *= 0.93

    prob = max(0.02, min(0.98, true_prob))
    confidence = "medium" if num_legs <= 3 else "low"
    leg_detail = ", ".join(f"{lp:.0%}" for lp in leg_probs[:5])

    return prob, confidence, [f"{num_legs}-leg parlay, per-leg: [{leg_detail}], combined: {prob:.1%}"], \
           [f"{num_legs}-leg parlay", f"Market: {market_prob:.1%}", f"Our: {prob:.1%}"]


def heuristic_critique(market: MarketInfo, forecast: ForecastResult) -> CriticResult:
    """Built-in heuristic critic (from v3)."""
    major_flaws = []
    minor_flaws = []
    should_trade = True
    adj_prob = forecast.probability
    market_prob = market.market_prob
    edge = abs(forecast.probability - market_prob)
    market_type = classify_market_type(market)

    if market_type == "combo":
        if edge > 0.50:
            minor_flaws.append(f"Very large parlay edge ({edge:.0%})")
            adj_prob = 0.7 * forecast.probability + 0.3 * market_prob
    else:
        if edge > 0.40:
            major_flaws.append(f"Very large edge ({edge:.0%})")
            adj_prob = 0.6 * forecast.probability + 0.4 * market_prob
            should_trade = False
        elif edge > 0.25:
            minor_flaws.append(f"Large edge ({edge:.0%})")
            adj_prob = 0.8 * forecast.probability + 0.2 * market_prob

    if market.volume < 500:
        minor_flaws.append(f"Low volume ({market.volume})")
    if market.days_to_expiry < 0.1:
        minor_flaws.append("Very close to expiry")
        adj_prob = 0.7 * market_prob + 0.3 * forecast.probability
    if forecast.confidence == "low" and edge < 0.05:
        should_trade = False
        major_flaws.append("Low confidence + small edge")
    if "heuristic-generic" in (forecast.model_used or "") and edge > 0.05:
        should_trade = False
        major_flaws.append("Generic model — no specific insight")

    return CriticResult(adjusted_probability=adj_prob, major_flaws=major_flaws,
                        minor_flaws=minor_flaws, should_trade=should_trade,
                        reasoning=f"Edge: {edge:.1%}, flaws: {len(major_flaws)} major",
                        tokens_used=0)


# ============================================================================
# TRADER (Kelly + Trade Decision from v3)
# ============================================================================

def calculate_kelly(prob: float, price_cents: int) -> float:
    """Kelly criterion for position sizing."""
    if price_cents <= 0 or price_cents >= 100:
        return 0.0
    b = (100 - price_cents) / price_cents
    p = prob
    q = 1 - p
    kelly = (b * p - q) / b
    kelly *= KELLY_FRACTION
    return max(0.0, min(kelly, MAX_POSITION_PCT))


def get_dynamic_edge_cap(market_type: str, num_legs: int = 1) -> float:
    """Return the edge cap based on market type and leg count (GROK-TRADE-003).
    
    - Single-leg markets: SINGLE_LEG_EDGE_CAP (15%)
    - Parlays: PARLAY_BASE_EDGE_CAP + PARLAY_LEG_SCALE_FACTOR per leg beyond 2,
      capped at PARLAY_MAX_EDGE_CAP (40%)
    """
    if market_type != "combo" or num_legs < 2:
        return SINGLE_LEG_EDGE_CAP
    # Parlay: base 35% + 2% per extra leg beyond 2, capped at 40%
    extra_legs = max(0, num_legs - 2)
    parlay_cap = PARLAY_BASE_EDGE_CAP + extra_legs * PARLAY_LEG_SCALE_FACTOR
    return min(parlay_cap, PARLAY_MAX_EDGE_CAP)


def make_trade_decision(market: MarketInfo, forecast: ForecastResult, critic: CriticResult,
                        balance: float) -> TradeDecision:
    """Compare probability vs market price and decide. Uses split thresholds (v3 data-driven)."""
    final_prob = 0.6 * forecast.probability + 0.4 * critic.adjusted_probability
    market_prob = market.market_prob
    edge_yes = final_prob - market_prob

    # Split thresholds
    if edge_yes > 0:
        if edge_yes < MIN_EDGE_BUY_YES:
            return TradeDecision(action="SKIP", edge=edge_yes, kelly_size=0, contracts=0,
                                price_cents=0, reason=f"BUY_YES edge {edge_yes:.1%} < {MIN_EDGE_BUY_YES:.0%}",
                                forecast=forecast, critic=critic)
    else:
        if abs(edge_yes) < MIN_EDGE_BUY_NO:
            return TradeDecision(action="SKIP", edge=edge_yes, kelly_size=0, contracts=0,
                                price_cents=0, reason=f"BUY_NO edge {abs(edge_yes):.1%} < {MIN_EDGE_BUY_NO:.0%}",
                                forecast=forecast, critic=critic)

    if not critic.should_trade:
        return TradeDecision(action="SKIP", edge=edge_yes, kelly_size=0, contracts=0, price_cents=0,
                            reason=f"Critic vetoed: {', '.join(critic.major_flaws[:2]) or 'concerns'}",
                            forecast=forecast, critic=critic)
    if len(critic.major_flaws) >= 2:
        return TradeDecision(action="SKIP", edge=edge_yes, kelly_size=0, contracts=0, price_cents=0,
                            reason=f"Too many flaws: {', '.join(critic.major_flaws[:2])}",
                            forecast=forecast, critic=critic)
    if forecast.confidence == "low" and edge_yes > 0 and abs(edge_yes) < 0.10:
        return TradeDecision(action="SKIP", edge=edge_yes, kelly_size=0, contracts=0, price_cents=0,
                            reason=f"Low conf + moderate YES edge ({edge_yes:+.1%})",
                            forecast=forecast, critic=critic)

    # Classify market and count legs for dynamic edge cap (GROK-TRADE-003)
    market_type = classify_market_type(market)
    num_legs = estimate_combo_legs(market) if market_type == "combo" else 1

    # Dynamic edge cap: single-leg 15%, parlays 35-40% scaled by legs
    effective_edge_cap = get_dynamic_edge_cap(market_type, num_legs)
    if abs(edge_yes) > effective_edge_cap:
        edge_yes = effective_edge_cap if edge_yes > 0 else -effective_edge_cap
        final_prob = market_prob + edge_yes

    # Parlay BUY_YES filter
    if edge_yes > 0 and PARLAY_ONLY_NO and market_type == "combo":
        allow = PARLAY_YES_EXCEPTION and num_legs <= 2 and edge_yes > 0.05 and market.yes_price >= 30
        if not allow:
            return TradeDecision(action="SKIP", edge=edge_yes, kelly_size=0, contracts=0, price_cents=0,
                                reason=f"Parlay BUY_YES blocked (9.7% WR)", forecast=forecast, critic=critic)

    # Decide side
    if edge_yes > 0:
        action, side_price, edge = "BUY_YES", market.yes_price, edge_yes
    else:
        action, side_price, edge = "BUY_NO", market.no_price, abs(edge_yes)

    # Kelly sizing
    kelly_frac = calculate_kelly(final_prob if action == "BUY_YES" else (1 - final_prob), side_price)
    if kelly_frac <= 0:
        return TradeDecision(action="SKIP", edge=edge, kelly_size=0, contracts=0, price_cents=side_price,
                            reason="Kelly says no bet", forecast=forecast, critic=critic)

    bet_cents = int(balance * kelly_frac * 100)
    cost_per = max(1, side_price)
    contracts = max(1, bet_cents // cost_per)

    # Cap
    max_bet_dynamic = int(balance * MAX_POSITION_PCT * 100)
    effective_max = min(MAX_BET_CENTS, max_bet_dynamic)
    if contracts * cost_per > effective_max:
        contracts = effective_max // cost_per

    # Ensure at least 1 contract if affordable
    if contracts <= 0 and cost_per <= int(balance * 0.10 * 100) and cost_per >= MIN_BET_CENTS:
        contracts = 1
    if contracts <= 0:
        return TradeDecision(action="SKIP", edge=edge, kelly_size=kelly_frac, contracts=0,
                            price_cents=side_price, reason="Position too small", forecast=forecast, critic=critic)

    return TradeDecision(action=action, edge=edge, kelly_size=kelly_frac, contracts=contracts,
                        price_cents=side_price,
                        reason=f"Edge={edge:.1%}, Kelly={kelly_frac:.3f}, prob={final_prob:.1%} vs mkt={market_prob:.1%}",
                        forecast=forecast, critic=critic)


# ============================================================================
# MARKET SCANNER (from v3 + v2 weather)
# ============================================================================

def parse_market(raw: dict) -> Optional[MarketInfo]:
    try:
        ticker = raw.get("ticker", "")
        title = raw.get("title", "") or raw.get("event_title", "")
        subtitle = raw.get("subtitle", "") or raw.get("yes_sub_title", "")
        category = raw.get("category", "") or raw.get("series_ticker", "")
        yes_price = raw.get("yes_bid", 0) or raw.get("last_price", 50)
        no_price = 100 - yes_price if yes_price else 50
        yes_ask = raw.get("yes_ask") or yes_price
        yes_bid = raw.get("yes_bid") or yes_price
        volume = raw.get("volume", 0) or 0
        oi = raw.get("open_interest", 0) or 0
        expiry = raw.get("close_time", "") or raw.get("expiration_time", "")
        status = raw.get("status", "")
        result = raw.get("result", "") or ""
        last_price = raw.get("last_price", 0) or 0
        return MarketInfo(ticker=ticker, title=title, subtitle=subtitle, category=category,
                         yes_price=yes_price, no_price=no_price, volume=volume, open_interest=oi,
                         expiry=expiry, status=status, result=result, yes_bid=yes_bid,
                         yes_ask=yes_ask, last_price=last_price)
    except Exception:
        return None


def filter_markets(markets: list) -> list:
    filtered = []
    for m in markets:
        if not m.ticker or not m.title:
            continue
        if m.status not in ("open", "active", ""):
            continue
        if m.result:
            continue
        if m.volume < MIN_VOLUME:
            continue
        if max(m.open_interest, m.volume) < MIN_LIQUIDITY:
            continue
        dte = m.days_to_expiry
        if dte > MAX_DAYS_TO_EXPIRY or dte < MIN_DAYS_TO_EXPIRY:
            continue
        if m.yes_price < MIN_PRICE_CENTS or m.yes_price > MAX_PRICE_CENTS:
            continue
        filtered.append(m)
    return filtered


def score_market(market: MarketInfo) -> float:
    score = 0.0
    if market.volume > 0:
        score += min(10, math.log10(market.volume) * 2)
    score += max(0, 10 - abs(market.yes_price - 50) * 0.2)
    dte = market.days_to_expiry
    if 1 <= dte <= 7: score += 5
    elif 7 < dte <= 14: score += 3
    elif 0.1 <= dte < 1: score += 2
    else: score += 1
    if market.open_interest > 0:
        score += min(5, math.log10(market.open_interest + 1) * 1.5)
    return score


def scan_all_markets() -> list:
    """Scan all open Kalshi markets with pagination + sports tickers."""
    all_markets = []
    seen = set()
    cursor = None

    log.info("📡 Scanning Kalshi markets...", extra={"component": "scanner"})
    for page in range(20):
        path = f"/trade-api/v2/markets?limit=200"
        if cursor:
            path += f"&cursor={cursor}"
        result = kalshi_api("GET", path)
        if "error" in result:
            break
        raw = result.get("markets", [])
        if not raw:
            break
        for r in raw:
            m = parse_market(r)
            if m and m.ticker not in seen:
                all_markets.append(m)
                seen.add(m.ticker)
        next_cursor = result.get("cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.3)

    log.info(f"   General: {len(all_markets)} markets")

    # Sports event tickers
    sports_found = 0
    for et in SPORTS_EVENT_TICKERS:
        path = f"/trade-api/v2/markets?limit=200&series_ticker={et}&status=open"
        result = kalshi_api("GET", path)
        if "error" not in result:
            for r in result.get("markets", []):
                m = parse_market(r)
                if m and m.ticker not in seen:
                    all_markets.append(m)
                    seen.add(m.ticker)
                    sports_found += 1
        time.sleep(0.2)
    if sports_found:
        log.info(f"   Sports: +{sports_found} additional markets")

    # Filter
    filtered = filter_markets(all_markets)
    log.info(f"   Filtered: {len(filtered)}/{len(all_markets)} pass criteria",
             extra={"component": "scanner"})
    return filtered


def find_weather_opportunities() -> list:
    """Find weather trading opportunities (from v2)."""
    if not WEATHER_MODULE_AVAILABLE or not WEATHER_ENABLED:
        return []

    opportunities = []
    city_series = {
        "NYC": ["KXHIGHNY", "KXLOWTNYC", "KXLOWNY"],
        "DEN": ["KXHIGHDEN", "KXLOWDEN"],
        "CHI": ["KXHIGHCHI", "KXLOWCHI"],
    }

    for city in WEATHER_CITIES:
        series_list = city_series.get(city, [])
        try:
            forecast = fetch_weather_forecast(city)
            if not forecast:
                continue
        except Exception:
            continue

        for series in series_list:
            try:
                path = f"/trade-api/v2/markets?series_ticker={series}&limit=20&status=open"
                resp_data = kalshi_api("GET", path)
                markets = resp_data.get("markets", [])

                for m in markets:
                    ticker = m.get("ticker")
                    title = m.get("title", "")
                    yes_bid = m.get("yes_bid")
                    yes_ask = m.get("yes_ask")
                    if not ticker or yes_bid is None:
                        continue
                    if yes_ask and (yes_ask <= 5 or yes_ask >= 95):
                        continue

                    parsed = parse_kalshi_weather_ticker(ticker, title)
                    if not parsed:
                        continue
                    edge_result = calculate_weather_edge(parsed, yes_bid)
                    if not edge_result or edge_result.get("edge", 0) < WEATHER_MIN_EDGE:
                        continue

                    our_prob = edge_result.get("calculated_probability", 0)
                    if our_prob < WEATHER_MIN_OUR_PROB:
                        continue

                    side = "yes" if edge_result.get("recommendation") == "BUY_YES" else "no"
                    price = yes_ask if side == "yes" else (100 - yes_bid if yes_bid else None)
                    if not price or price <= 0:
                        continue

                    market_info = parse_market(m)
                    if not market_info:
                        continue

                    opportunities.append({
                        "market": market_info,
                        "edge": edge_result["edge"],
                        "our_prob": our_prob,
                        "side": side,
                        "price": price,
                        "source": "weather",
                        "city": city,
                        "forecast_temp": edge_result.get("forecast_temp"),
                    })
            except Exception:
                continue

    opportunities.sort(key=lambda x: x["edge"], reverse=True)
    return opportunities


# ============================================================================
# CIRCUIT BREAKER & DAILY LOSS (from v1+v2)
# ============================================================================

def check_circuit_breaker() -> tuple:
    """Check consecutive loss circuit breaker. Returns (is_paused, losses, message)."""
    try:
        if CIRCUIT_BREAKER_STATE_FILE.exists():
            with open(CIRCUIT_BREAKER_STATE_FILE) as f:
                state = json.load(f)
            if state.get("paused"):
                paused_at = datetime.fromisoformat(state["paused_at"])
                cooldown = timedelta(hours=CIRCUIT_BREAKER_COOLDOWN_HOURS)
                if datetime.now(timezone.utc) < paused_at + cooldown:
                    remaining = (paused_at + cooldown - datetime.now(timezone.utc)).total_seconds() / 60
                    return True, state.get("losses", 0), f"Paused ({remaining:.0f}min left)"
                else:
                    CIRCUIT_BREAKER_STATE_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    # Count consecutive losses from trade log
    losses = 0
    try:
        if TRADE_LOG_FILE.exists():
            with open(TRADE_LOG_FILE) as f:
                lines = f.readlines()
            for line in reversed(lines):
                try:
                    entry = json.loads(line.strip())
                    status = entry.get("result_status", "pending")
                    if status == "won":
                        break
                    elif status == "lost":
                        losses += 1
                except Exception:
                    continue
    except Exception:
        pass

    if losses >= CIRCUIT_BREAKER_THRESHOLD:
        state = {"paused": True, "paused_at": datetime.now(timezone.utc).isoformat(),
                 "losses": losses, "reason": f"{losses} consecutive losses"}
        with open(CIRCUIT_BREAKER_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        return True, losses, f"TRIGGERED: {losses} consecutive losses"

    return False, losses, f"OK ({losses}/{CIRCUIT_BREAKER_THRESHOLD} losses)"


def check_daily_loss_limit() -> tuple:
    """Check daily loss limit. Returns (is_paused, pnl_info)."""
    today = datetime.now(timezone.utc).date()
    spent = 0
    won = 0
    trades = 0

    try:
        if TRADE_LOG_FILE.exists():
            with open(TRADE_LOG_FILE) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        ts = entry.get("timestamp", "")
                        if not ts:
                            continue
                        trade_date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
                        if trade_date != today:
                            continue
                        trades += 1
                        cost = entry.get("cost_cents", entry.get("contracts", 0) * entry.get("price_cents", 0))
                        spent += cost
                        if entry.get("result_status") == "won":
                            contracts = entry.get("contracts", 0)
                            price = entry.get("price_cents", 0)
                            won += contracts * (100 - price)
                    except Exception:
                        continue
    except Exception:
        pass

    net_pnl = won - spent
    pnl = {"net_pnl_cents": net_pnl, "trades_today": trades, "spent": spent, "won": won}

    if net_pnl < -DAILY_LOSS_LIMIT_CENTS:
        return True, pnl

    # Check pause file from earlier today
    if DAILY_LOSS_PAUSE_FILE.exists():
        try:
            with open(DAILY_LOSS_PAUSE_FILE) as f:
                data = json.load(f)
            pause_date = datetime.fromisoformat(data.get("paused_at", "")).date()
            if pause_date == today:
                return True, pnl
            else:
                DAILY_LOSS_PAUSE_FILE.unlink(missing_ok=True)
        except Exception:
            DAILY_LOSS_PAUSE_FILE.unlink(missing_ok=True)

    return False, pnl


# ============================================================================
# SETTLEMENT TRACKER
# ============================================================================

def update_trade_results():
    """Check settled markets and update trade log with results."""
    if not TRADE_LOG_FILE.exists():
        return {"updated": 0, "wins": 0, "losses": 0}

    updated_lines = []
    updated_count = 0
    wins = 0
    losses = 0

    try:
        with open(TRADE_LOG_FILE) as f:
            lines = f.readlines()

        for line in lines:
            try:
                entry = json.loads(line.strip())
                if entry.get("result_status") == "pending" and entry.get("ticker"):
                    # Check if market settled
                    result = kalshi_api("GET", f"/trade-api/v2/markets/{entry['ticker']}")
                    if "error" not in result:
                        market_result = result.get("result", "")
                        if market_result:
                            action = entry.get("action", "")
                            side = "yes" if action == "BUY_YES" else "no"
                            if (side == "yes" and market_result == "yes") or \
                               (side == "no" and market_result == "no"):
                                entry["result_status"] = "won"
                                wins += 1
                            else:
                                entry["result_status"] = "lost"
                                losses += 1
                            entry["settled_at"] = datetime.now(timezone.utc).isoformat()
                            entry["market_result"] = market_result
                            updated_count += 1
                    time.sleep(0.2)  # Rate limit
                updated_lines.append(json.dumps(entry) + "\n")
            except Exception:
                updated_lines.append(line)

        if updated_count > 0:
            with open(TRADE_LOG_FILE, "w") as f:
                f.writelines(updated_lines)

    except Exception as e:
        log.warning(f"⚠️ Settlement check error: {e}",
                    extra={"component": "settlement", "error_type": "settlement_check"})
        record_error("settlement_check", str(e))

    return {"updated": updated_count, "wins": wins, "losses": losses}


# ============================================================================
# LOGGING
# ============================================================================

def log_trade(market: MarketInfo, decision: TradeDecision, order_result: dict, dry_run: bool):
    """Log trade to JSONL files."""
    for log_path in [TRADE_LOG_FILE, LEGACY_TRADE_LOG]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "ticker": market.ticker,
            "title": market.title[:100],
            "category": market.category,
            "market_price_yes": market.yes_price,
            "market_price_no": market.no_price,
            "volume": market.volume,
            "expiry": market.expiry,
            "days_to_expiry": round(market.days_to_expiry, 2),
            "action": decision.action,
            "edge": round(decision.edge, 4),
            "kelly_size": round(decision.kelly_size, 4),
            "contracts": decision.contracts,
            "price_cents": decision.price_cents,
            "cost_cents": decision.contracts * decision.price_cents,
            "reason": decision.reason,
            "forecast_prob": round(decision.forecast.probability, 4) if decision.forecast else None,
            "forecast_confidence": decision.forecast.confidence if decision.forecast else None,
            "forecast_model": decision.forecast.model_used if decision.forecast else None,
            "critic_adj_prob": round(decision.critic.adjusted_probability, 4) if decision.critic else None,
            "critic_flaws": decision.critic.major_flaws if decision.critic else [],
            "total_tokens": ((decision.forecast.tokens_used if decision.forecast else 0) +
                             (decision.critic.tokens_used if decision.critic else 0)),
            "order_result": order_result,
            "result_status": "pending" if decision.action != "SKIP" else "skipped",
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def log_cycle(stats: dict):
    CYCLE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **stats}
    with open(CYCLE_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_skip(ticker: str, reason: str, details: dict = None):
    SKIP_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "ticker": ticker,
             "reason": reason, **(details or {})}
    with open(SKIP_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================================================================
# MAIN TRADING CYCLE
# ============================================================================

def run_cycle(dry_run: bool = True, max_markets: int = 20, max_trades: int = 5):
    """
    One complete trading cycle:
    1. Check safety (circuit breaker, daily loss, holiday, risk limits)
    2. Gather context (prices, momentum, sentiment, regime)
    3. Scan & rank markets
    4. For top N: Forecast → Critique → Decide → Risk Check → Execute
    5. Log everything (structured JSON)
    6. Check drawdown alerts
    """
    cycle_start = time.time()
    shutdown.current_cycle += 1
    cycle_id = f"cycle-{shutdown.current_cycle}-{int(cycle_start)}"

    log.info("=" * 70)
    log.info(f"🤖 KALSHI AUTOTRADER — Unified",
             extra={"component": "cycle", "cycle_id": cycle_id})
    log.info(f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"{'🧪 DRY RUN (PAPER)' if dry_run else '🔴 LIVE TRADING'}")
    log.info("=" * 70)

    # ── Graceful shutdown check ──
    if shutdown.check_stop():
        log.info("🛑 Shutdown requested — skipping cycle",
                 extra={"component": "shutdown", "cycle_id": cycle_id})
        return

    # ── Check holiday ──
    if HOLIDAY_CHECK_AVAILABLE:
        is_hol, hol_name = is_market_holiday()
        if is_hol:
            log.info(f"🎄 Market holiday: {hol_name} — skipping cycle",
                     extra={"component": "cycle", "cycle_id": cycle_id})
            return

    # ── Settlement tracker ──
    settlement = update_trade_results()
    if settlement["updated"]:
        log.info(f"📊 Settled: {settlement['updated']} trades ({settlement['wins']}W/{settlement['losses']}L)",
                 extra={"component": "settlement", "cycle_id": cycle_id})

    # ── Circuit breaker ──
    cb_paused, cb_losses, cb_msg = check_circuit_breaker()
    log.info(f"🔒 Circuit breaker: {cb_msg}",
             extra={"component": "risk", "cycle_id": cycle_id})
    if cb_paused:
        _write_alert("circuit-breaker", {
            "severity": "warning",
            "consecutive_losses": cb_losses,
            "message": cb_msg,
        })
        return

    # ── Daily loss limit ──
    dl_paused, dl_pnl = check_daily_loss_limit()
    log.info(f"📊 Daily PnL: ${dl_pnl['net_pnl_cents']/100:+.2f} ({dl_pnl['trades_today']} trades)",
             extra={"component": "risk", "cycle_id": cycle_id})
    if dl_paused:
        log.warning(f"🛑 Daily loss limit reached (-${DAILY_LOSS_LIMIT_CENTS/100:.2f})",
                    extra={"component": "risk", "alert_type": "daily_loss_limit"})
        _write_alert("daily-loss", {
            "severity": "critical",
            "net_pnl_cents": dl_pnl["net_pnl_cents"],
            "trades_today": dl_pnl["trades_today"],
            "limit_cents": DAILY_LOSS_LIMIT_CENTS,
            "message": f"Daily loss limit reached: ${abs(dl_pnl['net_pnl_cents'])/100:.2f}",
        })
        return

    # ── LLM status ──
    use_heuristic = True
    if LLM_CONFIG and not LLM_CONFIG.get("api_key", "").startswith("sk-ant-oat"):
        log.info(f"✅ LLM: {LLM_CONFIG['provider']} / {LLM_CONFIG['model']}",
                 extra={"component": "config"})
        use_heuristic = False
    else:
        log.info("⚠️  No LLM API — using HEURISTIC forecaster",
                 extra={"component": "config"})

    # ── Balance ──
    balance = get_balance()
    log.info(f"💰 Balance: ${balance:.2f}",
             extra={"component": "balance", "balance": balance})
    if dry_run and balance < 1.0:
        balance = VIRTUAL_BALANCE
        log.info(f"   📝 Using virtual balance: ${balance:.2f} (paper mode)")

    # ── Track peak balance & check drawdown ──
    save_peak_balance(balance)
    peak_balance = load_peak_balance()
    check_drawdown_alert(balance, peak_balance)

    # ── Positions ──
    positions = get_positions()
    num_positions = len(positions)
    log.info(f"📊 Open positions: {num_positions}/{MAX_CONCURRENT_POSITIONS}",
             extra={"component": "positions", "positions": num_positions})
    if num_positions >= MAX_CONCURRENT_POSITIONS:
        log.warning("⚠️ Max positions reached",
                    extra={"component": "risk", "positions": num_positions})
        return

    # ── Gather context (v2 signals) ──
    context = {}

    # Crypto prices
    prices = get_crypto_prices()
    if prices:
        context["crypto_prices"] = prices
        log.info(f"📈 BTC: ${prices['btc']:,.0f} | ETH: ${prices['eth']:,.0f}",
                 extra={"component": "market_data"})
    else:
        log.warning("⚠️ No crypto prices available",
                    extra={"component": "market_data"})
        record_error("crypto_prices", "Failed to fetch crypto prices")

    # Fear & Greed
    fng = get_fear_greed_index()
    context["sentiment"] = fng
    log.info(f"😱 Fear & Greed: {fng['value']} ({fng['classification']})",
             extra={"component": "market_data"})

    # News sentiment
    if NEWS_SEARCH_AVAILABLE:
        try:
            news = get_crypto_sentiment("both")
            context["news_sentiment"] = news
            icon = "🟢" if news["sentiment"] == "bullish" else ("🔴" if news["sentiment"] == "bearish" else "⚪")
            log.info(f"📰 News: {icon} {news['sentiment'].upper()} ({news['confidence']*100:.0f}%)",
                     extra={"component": "market_data"})
        except Exception as e:
            context["news_sentiment"] = None
            record_error("news_sentiment", str(e))
    else:
        context["news_sentiment"] = None

    # ── Graceful shutdown check ──
    if shutdown.check_stop():
        log.info("🛑 Shutdown requested during context gathering — exiting",
                 extra={"component": "shutdown"})
        return

    # OHLC + Momentum + Regime
    btc_ohlc = get_crypto_ohlc("bitcoin", 7)
    eth_ohlc = get_crypto_ohlc("ethereum", 7)
    btc_momentum = get_multi_timeframe_momentum(btc_ohlc)
    eth_momentum = get_multi_timeframe_momentum(eth_ohlc)
    context["ohlc"] = {"btc": btc_ohlc, "eth": eth_ohlc}
    context["momentum"] = {"btc": btc_momentum, "eth": eth_momentum}

    btc_regime = detect_market_regime(btc_ohlc, btc_momentum)
    eth_regime = detect_market_regime(eth_ohlc, eth_momentum)
    context["regime"] = {"btc": btc_regime, "eth": eth_regime}

    for asset, mom in [("BTC", btc_momentum), ("ETH", eth_momentum)]:
        cd = mom.get("composite_direction", 0)
        cs = mom.get("composite_strength", 0)
        aligned = "✓" if mom.get("alignment") else ""
        label = "BULL" if cd > 0.1 else ("BEAR" if cd < -0.1 else "FLAT")
        log.info(f"📊 {asset} Momentum: {label} (dir={cd:+.2f}, str={cs:.2f}) {aligned}",
                 extra={"component": "momentum"})

    for asset, regime in [("BTC", btc_regime), ("ETH", eth_regime)]:
        log.info(f"   {asset} Regime: {regime['regime']} ({regime['confidence']:.0%}), vol: {regime['volatility']}",
                 extra={"component": "regime"})

    # ── Scan markets ──
    markets = scan_all_markets()
    if not markets:
        log.warning("❌ No tradeable markets found!",
                    extra={"component": "scanner", "cycle_id": cycle_id})
        return

    # Score and rank
    scored = [(m, score_market(m)) for m in markets]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_markets = scored[:max_markets]

    log.info(f"\n🎯 TOP {len(top_markets)} MARKETS:")
    log.info("-" * 70)
    for i, (m, s) in enumerate(top_markets[:10], 1):
        log.info(f"  {i:2d}. [{s:.1f}] {m.ticker} — {m.title[:65]}")
        log.info(f"      YES:{m.yes_price}¢ Vol:{m.volume:,} DTE:{m.days_to_expiry:.1f}d")
    log.info("-" * 70)

    # ── Analyze markets: Forecast → Critique → Decide → Risk Check ──
    trades_executed = 0
    trades_skipped = 0
    risk_blocked = 0
    total_tokens = 0
    existing_tickers = {p.get("ticker", "") for p in positions}

    # Dedup from trade log (paper mode)
    if dry_run and TRADE_LOG_FILE.exists():
        try:
            with open(TRADE_LOG_FILE) as f:
                for line in f:
                    try:
                        e = json.loads(line.strip())
                        if e.get("action") in ("BUY_YES", "BUY_NO") and e.get("dry_run"):
                            existing_tickers.add(e.get("ticker", ""))
                    except Exception:
                        continue
        except Exception:
            pass

    for i, (market, score) in enumerate(top_markets, 1):
        # ── Graceful shutdown check (between markets) ──
        if shutdown.check_stop():
            log.info(f"🛑 Shutdown requested — stopping market analysis at {i}/{len(top_markets)}",
                     extra={"component": "shutdown", "cycle_id": cycle_id})
            break

        if trades_executed >= max_trades:
            log.info(f"\n⏹️ Max trades ({max_trades}) reached")
            break

        if market.ticker in existing_tickers:
            continue

        log.info(f"\n{'='*50}")
        log.info(f"🔍 [{i}/{len(top_markets)}] {market.ticker}",
                 extra={"component": "analysis", "ticker": market.ticker})
        log.info(f"   {market.title[:75]}")
        log.info(f"   YES:{market.yes_price}¢ NO:{market.no_price}¢ Vol:{market.volume:,}")

        # Build market-specific context
        mkt_context = dict(context)
        mkt_type = classify_market_type(market)
        if mkt_type == "crypto":
            asset = "btc" if "BTC" in market.ticker.upper() else "eth"
            mkt_context["momentum"] = context.get("momentum", {}).get(asset, {})
            mkt_context["regime"] = context.get("regime", {}).get(asset, {})

        # Step 1: FORECAST
        if use_heuristic:
            log.info(f"   🧮 Heuristic ({mkt_type}/{detect_sport(market)})...")
            forecast = heuristic_forecast(market, mkt_context)
        else:
            log.info(f"   🧠 LLM Forecasting...")
            forecast = forecast_market_llm(market, mkt_context)
        total_tokens += forecast.tokens_used
        log.info(f"   📊 Forecast: {forecast.probability:.1%} ({forecast.confidence})",
                 extra={"component": "forecast", "ticker": market.ticker})
        log.info(f"   📝 Factors: {', '.join(forecast.key_factors[:3]) if forecast.key_factors else 'N/A'}")

        # Quick edge check before critic
        quick_edge = abs(forecast.probability - market.market_prob)
        if quick_edge < MIN_EDGE_BUY_NO * 0.5:
            log.debug(f"   ⏭️ Quick skip: edge {quick_edge:.1%} too small",
                      extra={"component": "analysis", "ticker": market.ticker, "edge": quick_edge})
            log.info(f"   ⏭️ Quick skip: edge {quick_edge:.1%} too small")
            trades_skipped += 1
            log_skip(market.ticker, f"quick_edge_{quick_edge:.3f}", {"market_prob": market.market_prob})
            continue

        # Step 2: CRITIQUE
        if use_heuristic:
            critic = heuristic_critique(market, forecast)
        else:
            log.info(f"   🔎 LLM Critiquing...")
            critic = critique_forecast_llm(market, forecast)
        total_tokens += critic.tokens_used
        log.info(f"   📊 Critic: {critic.adjusted_probability:.1%} | Flaws: {len(critic.major_flaws)} | Trade: {'✅' if critic.should_trade else '❌'}",
                 extra={"component": "critic", "ticker": market.ticker})

        # Step 3: TRADE DECISION
        decision = make_trade_decision(market, forecast, critic, balance)
        log.info(f"   📋 DECISION: {decision.action} — {decision.reason}",
                 extra={"component": "decision", "ticker": market.ticker,
                        "action": decision.action, "edge": round(decision.edge, 4)})

        if decision.action == "SKIP":
            trades_skipped += 1
            log_trade(market, decision, {}, dry_run)
            continue

        # Step 3.5: RISK LIMITS CHECK (new!)
        risk_ok, risk_reason = check_position_risk_limits(
            market, decision, balance, positions, dl_pnl)
        if not risk_ok:
            log.warning(f"   🛡️ RISK BLOCKED: {risk_reason}",
                        extra={"component": "risk", "ticker": market.ticker,
                               "action": decision.action})
            risk_blocked += 1
            decision_blocked = TradeDecision(
                action="SKIP", edge=decision.edge, kelly_size=decision.kelly_size,
                contracts=0, price_cents=decision.price_cents,
                reason=f"Risk limit: {risk_reason}",
                forecast=forecast, critic=critic)
            log_trade(market, decision_blocked, {}, dry_run)
            continue

        # Step 4: EXECUTE (with graceful shutdown protection)
        shutdown.enter_trade()
        try:
            side = "yes" if decision.action == "BUY_YES" else "no"
            cost = decision.contracts * decision.price_cents
            log.info(f"   💰 {decision.action} × {decision.contracts} @ {decision.price_cents}¢ = ${cost/100:.2f}",
                     extra={"component": "execution", "ticker": market.ticker,
                            "action": decision.action, "contracts": decision.contracts,
                            "price_cents": decision.price_cents, "cost_cents": cost})

            order_result = place_order(market.ticker, side, decision.price_cents, decision.contracts, dry_run)

            if dry_run:
                log.info(f"   🧪 DRY RUN: Simulated",
                         extra={"component": "execution", "ticker": market.ticker})
            else:
                if "error" in order_result:
                    log.error(f"   ❌ Order failed: {order_result['error']}",
                              extra={"component": "execution", "ticker": market.ticker,
                                     "error_type": "order_failed"})
                    record_error("order_failed", f"{market.ticker}: {order_result['error']}")
                else:
                    log.info(f"   ✅ Placed! ID: {order_result.get('order', {}).get('order_id', 'N/A')}",
                             extra={"component": "execution", "ticker": market.ticker})

            trades_executed += 1
            existing_tickers.add(market.ticker)
            log_trade(market, decision, order_result, dry_run)
        finally:
            shutdown.exit_trade()

        time.sleep(1)

    # ── Weather opportunities ──
    if WEATHER_ENABLED and not shutdown.check_stop():
        weather_opps = find_weather_opportunities()
        if weather_opps:
            log.info(f"\n🌡️ Weather: {len(weather_opps)} opportunities found",
                     extra={"component": "weather"})
            for wo in weather_opps[:3]:
                if trades_executed >= max_trades or shutdown.check_stop():
                    break
                wm = wo["market"]
                if wm.ticker in existing_tickers:
                    continue
                log.info(f"   🌡️ {wm.ticker}: {wo['side'].upper()} @{wo['price']}¢ edge:{wo['edge']*100:.1f}% ({wo['city']})")
                # Simple forecast/critic for weather
                wf = ForecastResult(probability=wo["our_prob"], reasoning="NWS forecast",
                                    confidence="medium", key_factors=["NWS"], model_used="nws-weather")
                wc = CriticResult(adjusted_probability=wo["our_prob"], should_trade=True)
                wd = make_trade_decision(wm, wf, wc, balance)
                if wd.action != "SKIP":
                    # Risk check for weather trades too
                    risk_ok, risk_reason = check_position_risk_limits(
                        wm, wd, balance, positions, dl_pnl)
                    if not risk_ok:
                        log.warning(f"   🛡️ Weather trade risk blocked: {risk_reason}")
                        continue
                    shutdown.enter_trade()
                    try:
                        order = place_order(wm.ticker, wo["side"], wo["price"], wd.contracts, dry_run)
                        log_trade(wm, wd, order, dry_run)
                        trades_executed += 1
                        existing_tickers.add(wm.ticker)
                    finally:
                        shutdown.exit_trade()

    # ── Cycle summary ──
    duration = time.time() - cycle_start
    log.info(f"\n{'='*70}")
    log.info(f"📊 CYCLE SUMMARY",
             extra={"component": "summary", "cycle_id": cycle_id, "duration_s": round(duration, 1)})
    log.info(f"   Forecaster: {'🧮 HEURISTIC' if use_heuristic else '🧠 LLM'}")
    log.info(f"   Duration: {duration:.1f}s")
    log.info(f"   Markets scanned: {len(markets)}")
    log.info(f"   Markets analyzed: {min(len(top_markets), max_markets)}")
    log.info(f"   Trades executed: {trades_executed}")
    log.info(f"   Trades skipped: {trades_skipped}")
    log.info(f"   Risk blocked: {risk_blocked}")
    if not use_heuristic:
        log.info(f"   Tokens: {total_tokens:,} (~${total_tokens * 0.000004:.4f})")

    # Latency summary
    avg_lat = get_avg_latency("markets_search")
    if avg_lat > 0:
        log.info(f"   Avg API latency: {avg_lat:.0f}ms")
    log.info(f"{'='*70}")

    log_cycle({
        "dry_run": dry_run,
        "cycle_id": cycle_id,
        "forecaster": "heuristic" if use_heuristic else "llm",
        "duration_s": round(duration, 1),
        "markets_scanned": len(markets),
        "markets_analyzed": min(len(top_markets), max_markets),
        "trades_executed": trades_executed,
        "trades_skipped": trades_skipped,
        "risk_blocked": risk_blocked,
        "tokens": total_tokens,
        "balance": balance,
        "peak_balance": peak_balance,
        "positions": num_positions,
        "daily_pnl_cents": dl_pnl.get("net_pnl_cents", 0),
        "shutdown_requested": shutdown.check_stop(),
    })


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kalshi AutoTrader — Unified (v1+v2+v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kalshi-autotrader.py                      # Paper mode, single cycle
  python kalshi-autotrader.py --live                # Live trading
  python kalshi-autotrader.py --markets 30          # Analyze more markets
  python kalshi-autotrader.py --max-trades 10       # More trades per cycle
  python kalshi-autotrader.py --loop 300            # Run every 5 minutes
  python kalshi-autotrader.py --min-edge 0.03       # Custom min edge
  python kalshi-autotrader.py --kelly 0.10          # Custom Kelly fraction
  python kalshi-autotrader.py --json-log /tmp/at.log  # Custom JSON log path
        """)

    parser.add_argument("--live", action="store_true", help="Enable LIVE trading (default: paper)")
    parser.add_argument("--markets", type=int, default=20, help="Max markets to analyze (default: 20)")
    parser.add_argument("--max-trades", type=int, default=5, help="Max trades per cycle (default: 5)")
    parser.add_argument("--loop", type=int, default=0, help="Loop interval in seconds (0 = single run)")
    parser.add_argument("--min-edge", type=float, default=None, help="Override minimum edge")
    parser.add_argument("--kelly", type=float, default=None, help="Override Kelly fraction")
    parser.add_argument("--json-log", type=str, default=None, help="Path for structured JSON log file")
    parser.add_argument("--max-exposure", type=int, default=None, help="Max exposure per market in cents (default: 1000)")
    parser.add_argument("--daily-loss-cap-pct", type=float, default=None, help="Daily loss cap as %% of portfolio (default: 0.10)")

    args = parser.parse_args()

    # ── Setup structured logging ──
    global log
    json_log_path = Path(args.json_log) if args.json_log else (PROJECT_ROOT / "data" / "trading" / "kalshi-autotrader.jsonl")
    log = setup_logging(json_log_path=json_log_path)
    log.info("🚀 Autotrader starting",
             extra={"component": "init", "json_log": str(json_log_path)})

    # ── Install graceful shutdown handlers ──
    shutdown.install_handlers()
    log.info("🛡️ Graceful shutdown handlers installed",
             extra={"component": "init"})

    # Override globals
    global MIN_EDGE, MIN_EDGE_BUY_YES, MIN_EDGE_BUY_NO, KELLY_FRACTION, DRY_RUN
    global MAX_EXPOSURE_PER_MARKET_CENTS, DAILY_LOSS_CAP_PCT
    if args.min_edge is not None:
        MIN_EDGE = args.min_edge
        MIN_EDGE_BUY_YES = max(args.min_edge, MIN_EDGE_BUY_YES)
        MIN_EDGE_BUY_NO = args.min_edge
    if args.kelly is not None:
        KELLY_FRACTION = args.kelly
    if args.max_exposure is not None:
        MAX_EXPOSURE_PER_MARKET_CENTS = args.max_exposure
    if args.daily_loss_cap_pct is not None:
        DAILY_LOSS_CAP_PCT = args.daily_loss_cap_pct
    if args.live:
        DRY_RUN = False

    dry_run = not args.live

    # Log risk configuration
    log.info(f"🛡️ Risk limits: max_exposure/market=${MAX_EXPOSURE_PER_MARKET_CENTS/100:.2f}, "
             f"daily_loss_cap={DAILY_LOSS_CAP_PCT:.0%}, max_positions={MAX_CONCURRENT_POSITIONS}, "
             f"max_daily_trades={MAX_DAILY_TRADES}",
             extra={"component": "config"})

    if args.live:
        log.warning("⚠️  LIVE TRADING MODE! Press Ctrl+C within 5s to abort...",
                    extra={"component": "init"})
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("Aborted.")
            sys.exit(0)

    try:
        if args.loop > 0:
            log.info(f"🔄 Loop mode: every {args.loop}s",
                     extra={"component": "init"})
            while not shutdown.check_stop():
                try:
                    run_cycle(dry_run=dry_run, max_markets=args.markets, max_trades=args.max_trades)
                    if shutdown.check_stop():
                        break
                    log.info(f"\n⏰ Next cycle in {args.loop}s...")
                    # Sleep in small increments to check for shutdown
                    for _ in range(args.loop):
                        if shutdown.check_stop():
                            break
                        time.sleep(1)
                except KeyboardInterrupt:
                    log.info("\n\n👋 Stopped by user",
                             extra={"component": "shutdown"})
                    break
                except Exception as e:
                    log.error(f"\n❌ Cycle error: {e}",
                              extra={"component": "cycle", "error_type": "cycle_crash"})
                    record_error("cycle_crash", str(e))
                    traceback.print_exc()
                    time.sleep(30)

            log.info("🏁 Autotrader stopped gracefully",
                     extra={"component": "shutdown"})
        else:
            run_cycle(dry_run=dry_run, max_markets=args.markets, max_trades=args.max_trades)
    finally:
        # Cleanup
        shutdown.cleanup()
        log.info("👋 Autotrader shutdown complete",
                 extra={"component": "shutdown"})


if __name__ == "__main__":
    main()
