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
import traceback
import signal
import logging
import logging.handlers
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# SQLite DB layer (zero extra deps — uses stdlib sqlite3)
try:
    import db as _db
    _db.init_db()
    _DB_AVAILABLE = True
except Exception as _db_err:
    print(f"⚠️  SQLite DB unavailable: {_db_err}")
    _DB_AVAILABLE = False

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
_key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.kalshi-private-key.pem')
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
MIN_EDGE_BUY_NO  = 0.02   # 2% min for BUY_NO  (paper mode: collect data aggressively)
MIN_EDGE_BUY_YES = 0.04   # 4% min for BUY_YES (lowered for data collection in paper mode)
CALIBRATION_FACTOR = 0.65  # baseline; overridden dynamically


def _compute_dynamic_calibration_factor(
    trade_history: list,
    window: int = 30,
    min_trades: int = 10,
    factor_min: float = 0.5,
    factor_max: float = 0.8,
    baseline: float = 0.65,
) -> float:
    """Rolling backtest over last `window` closed trades to adapt CALIBRATION_FACTOR.

    Logic:
        - Collects the last `window` trades that have a recorded outcome
          (resolved_prob vs predicted_prob).
        - Computes mean signed error: error = resolved - predicted.
        - If LLM is systematically over-confident (predicted > resolved),
          error < 0 → shrink factor (be more conservative).
        - If LLM is under-confident, error > 0 → increase factor.
        - Adjustment is proportional but clamped to [factor_min, factor_max].
    """
    if len(trade_history) < min_trades:
        return baseline

    recent = [
        t for t in trade_history
        if t.get("predicted_prob") is not None and t.get("resolved_prob") is not None
    ][-window:]

    if len(recent) < min_trades:
        return baseline

    errors = [
        t["resolved_prob"] - t["predicted_prob"]
        for t in recent
    ]
    mean_error = sum(errors) / len(errors)

    # Scale: each 0.1 mean error shifts factor by ~0.05
    adjustment = mean_error * 0.5
    new_factor = baseline + adjustment
    new_factor = max(factor_min, min(factor_max, new_factor))

    logging.getLogger(__name__).debug(
        f"[CalibDynamic] window={len(recent)} mean_err={mean_error:.4f} "
        f"factor={new_factor:.4f}"
    )
    return round(new_factor, 4)


def get_calibration_factor(trade_history: list | None = None) -> float:
    """Return dynamic CALIBRATION_FACTOR if enough history, else static baseline."""
    if trade_history:
        return _compute_dynamic_calibration_factor(trade_history)
    return CALIBRATION_FACTOR  # Shrink LLM/heuristic probs toward 50% to correct overconfidence
                            # (Grok: predicted 71.5% → actual 46.2%, ratio ~0.65)
MIN_EDGE = 0.02            # Global minimum (paper mode: more trades = more data)
MAX_EDGE_CAP = 0.10        # Cap edges >10% (overconfident forecaster at >10%: 0% WR)
MAX_POSITION_PCT = 0.05    # Max 5% of portfolio per position
KELLY_FRACTION = 0.25      # Quarter-Kelly: conservative (Grok uses 0.75 but we need data first)
MIN_BET_CENTS = 5
MAX_BET_CENTS = 200        # $2 max per trade (2% of $100 bankroll — conservative until profitable)
MAX_POSITIONS = 15         # Max open positions (Grok rec) — overridden by dynamic_max_positions()


def dynamic_max_positions(balance: float) -> int:
    """PROC-002 Task 3.1: Scale max positions with balance. Min 5, max 20."""
    if balance <= 50:
        return 5
    elif balance <= 100:
        return int(5 + (balance - 50) * 0.2)  # 5-15
    elif balance <= 200:
        return int(15 + (balance - 100) * 0.05)  # 15-20
    else:
        return 20

# ── Risk/Reward filters (TRADE-003: fix loss 2x > win asymmetry) ──
# BUY_NO at >50¢ means you risk more than you win. Require bigger edge to justify.
MAX_NO_PRICE_CENTS = 80     # Hard cap: never buy NO above 80¢ (relaxed for paper mode data)
NO_PRICE_EDGE_SCALE = True  # Scale min edge up with BUY_NO price
# If NO price is 50-65¢, require edge >= 3% + 0.1% per cent above 50
# e.g., 55¢ → 3.5% min edge, 60¢ → 4% min edge, 65¢ → 4.5%
MAX_RISK_REWARD_RATIO = 5.0 # Skip if (cost / potential_win) > 5.0 (allows BUY_NO up to ~83¢)

# ── Parlay strategy (from v3 data) ──
PARLAY_ONLY_NO = True       # On multi-leg parlays, primarily take BUY_NO
PARLAY_YES_EXCEPTION = True # Allow BUY_YES on 2-leg parlays if edge > 5%

# ── Position management / trailing stop (TRADE-017) ──
TRAILING_STOP_ENABLED = True
PROFIT_TAKE_PCT = 0.30       # Take profit when position is +30% (e.g., bought NO@60¢, now worth 78¢)
TRAILING_STOP_PCT = 0.15     # Trail by 15% from peak unrealized profit
MIN_PROFIT_TO_TRAIL = 0.10   # Start trailing only after +10% unrealized
EARLY_EXIT_NEAR_EXPIRY_HOURS = 2  # Force exit if <2h to expiry and in profit
HARD_STOP_LOSS_PCT = -0.30   # Hard stop: exit if position is -30% or worse

# ── Market scanning filters ──
MIN_VOLUME = 200
MIN_LIQUIDITY = 1000
MAX_DAYS_TO_EXPIRY = 30
MIN_DAYS_TO_EXPIRY = 0.02  # ~30 minutes
MIN_PRICE_CENTS = 5
MAX_PRICE_CENTS = 50   # Grok rec C: raise to 50¢ for more volume (breakeven WR = 50%)

# ── Circuit breaker / daily loss ──
CIRCUIT_BREAKER_THRESHOLD = 5  # Pause after N consecutive losses
CIRCUIT_BREAKER_COOLDOWN_HOURS = 4
DAILY_LOSS_LIMIT_CENTS = 1500  # $15 daily loss limit (15% of $100 bankroll per Grok pointer)

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

# Crypto series tickers (hourly + daily contracts, trade 24/7)
CRYPTO_SERIES_TICKERS = [
    "KXBTCD", "KXETHD", "KXSOLD",           # Daily range contracts
    "KXBTCMAX100", "KXETHMAX100",            # Max price contracts
    "KXBTCMIN100", "KXETHMIN100",            # Min price contracts
]

# ── Crypto volatility defaults (v2 calibrated) ──
BTC_HOURLY_VOL = 0.0096  # ~0.96% hourly std dev (Grok review: actual avg from Babypips data)
ETH_HOURLY_VOL = 0.0118  # ~1.18% hourly std dev (Grok review: actual avg, more volatile than BTC)
CRYPTO_FAT_TAIL_MULTIPLIER = 1.0  # Disabled after v2 disaster analysis

# ── Logging paths ──
PROJECT_ROOT = Path(__file__).parent  # Standalone repo: logs live inside the repo dir
_DATA_DIR = PROJECT_ROOT / "data" / "trading"
TRADE_LOG_FILE = _DATA_DIR / "kalshi-unified-trades.jsonl"
V3_TRADE_LOG   = _DATA_DIR / "kalshi-v3-trades.jsonl"
CYCLE_LOG_FILE = _DATA_DIR / "kalshi-unified-cycles.jsonl"
SKIP_LOG_FILE  = _DATA_DIR / "kalshi-unified-skips.jsonl"
LEGACY_TRADE_LOG = PROJECT_ROOT / "kalshi-trades.jsonl"

# ── Alert files (v2 compat) ──
CIRCUIT_BREAKER_STATE_FILE = Path(__file__).parent / "kalshi-circuit-breaker.json"
DAILY_LOSS_PAUSE_FILE = Path(__file__).parent / "kalshi-daily-pause.json"

# ── Alert files (GROK-TRADE-002) ──
DRAWDOWN_ALERT_FILE = Path(__file__).parent / "kalshi-drawdown.alert"
API_ERROR_ALERT_FILE = Path(__file__).parent / "kalshi-api-error.alert"
MAX_EXPOSURE_ALERT_FILE = Path(__file__).parent / "kalshi-max-exposure.alert"

# ── Alert + log files (GROK-TRADE-004: post-trade monitoring) ──
DECISION_LOG_FILE = _DATA_DIR / "kalshi-decisions.jsonl"
LOSS_STREAK_ALERT_FILE = Path(__file__).parent / "kalshi-loss-streak.alert"
HIGH_EDGE_CLUSTER_ALERT_FILE = Path(__file__).parent / "kalshi-high-edge-cluster.alert"

# ── Risk limits (GROK-TRADE-002: portfolio-level) ──
MAX_EXPOSURE_PCT = 0.50          # Don't trade if total open positions value > 50% of balance
MAX_CATEGORY_EXPOSURE_PCT = 0.30 # Max 30% in any single category (crypto, weather, sports)
MAX_DAILY_TRADES = 200           # Hard cap on trades per day (relaxed for paper mode data collection)
MAX_DAILY_EXPOSURE_USD = 50.0    # GROK-TRADE-004: Absolute $ cap on daily new exposure

# ── Paper trade state (bankroll/positions tracking for dashboard) ──
PAPER_STATE_FILE = _DATA_DIR / "paper-trade-state.json"
PAPER_STARTING_BANKROLL_CENTS = 10000  # $100 virtual bankroll (matches planned live bankroll)

# ── Structured logging (GROK-TRADE-002) ──
AUTOTRADER_LOG_FILE = _DATA_DIR / "kalshi-autotrader.log"

# ── Drawdown tracking (GROK-TRADE-002) ──
DRAWDOWN_PEAK_BALANCE = 0.0      # Tracked in-memory, seeded from balance on startup
DRAWDOWN_WINDOW_HOURS = 24       # Rolling 24h window for drawdown alerts
DRAWDOWN_THRESHOLD_PCT = 0.15    # Alert if portfolio drops >15% from peak

# ── API error rate tracking (GROK-TRADE-002) ──
API_ERROR_WINDOW: list = []      # list of (timestamp, is_error) tuples
API_ERROR_RATE_WINDOW_SEC = 300  # 5 minute rolling window
API_ERROR_RATE_THRESHOLD = 0.10  # Alert if >10% of API calls fail

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
            if key.startswith("sk-ant-oat"):
                auth_h = {"Authorization": f"Bearer {key}", "anthropic-beta": "oauth-2025-04-20"}
            else:
                auth_h = {"x-api-key": key}
            return {
                "provider": "anthropic", "api_key": key,
                "base_url": "https://api.anthropic.com/v1/messages",
                "model": env_vars.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
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
# GRACEFUL SHUTDOWN (GROK-TRADE-002)
# ============================================================================

shutdown_requested = False

def _shutdown_handler(signum, frame):
    """Signal handler for graceful shutdown (SIGTERM/SIGINT)."""
    global shutdown_requested
    shutdown_requested = True
    sig_name = signal.Signals(signum).name
    print(f"\n⚠️  Received {sig_name} — graceful shutdown requested...")
    structured_log("shutdown_requested", {"signal": sig_name})

# Register signal handlers
signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)


# ============================================================================
# STRUCTURED JSON LOGGING (GROK-TRADE-002)
# ============================================================================

class JSONFormatter(logging.Formatter):
    """Outputs log records as JSON lines for machine-readable log files."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
            "data": getattr(record, "data", {}),
        }
        return json.dumps(log_entry)


class ColoredConsoleFormatter(logging.Formatter):
    """Human-readable colored console formatter."""
    COLORS = {
        "DEBUG": "\033[90m",     # gray
        "INFO": "\033[36m",      # cyan
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        event = getattr(record, "event", "")
        data = getattr(record, "data", {})
        data_str = ""
        if data:
            # Compact key=value representation
            parts = []
            for k, v in data.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            data_str = " | " + ", ".join(parts)
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return f"{color}[{ts}] [{record.levelname[0]}] {event}{data_str}{self.RESET}"


def setup_structured_logger() -> logging.Logger:
    """Set up structured JSON logger with rotating file + colored console handlers."""
    logger = logging.getLogger("kalshi_autotrader")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on re-init
    if logger.handlers:
        return logger

    # 1. Rotating JSON file handler (10MB, 5 backups)
    AUTOTRADER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        str(AUTOTRADER_LOG_FILE),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # 2. Colored console handler (INFO+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredConsoleFormatter())
    logger.addHandler(console_handler)

    return logger


# Initialize the structured logger
_structured_logger = setup_structured_logger()


def structured_log(event: str, data: dict = None, level: str = "info"):
    """
    Log a structured event. Key events:
    cycle_start, cycle_end, trade_executed, trade_skipped,
    circuit_breaker, daily_loss, position_exit, api_error, settlement,
    risk_limit_hit, drawdown_alert, shutdown_requested, shutdown_complete
    """
    record = logging.LogRecord(
        name="kalshi_autotrader",
        level=getattr(logging, level.upper(), logging.INFO),
        pathname="",
        lineno=0,
        msg=event,
        args=(),
        exc_info=None,
    )
    record.event = event
    record.data = data or {}
    _structured_logger.handle(record)


# ============================================================================
# ALERT FILE HELPERS (GROK-TRADE-002)
# ============================================================================

def write_alert(alert_file: Path, reason: str, data: dict = None):
    """Write a .alert file for watchdog consumption."""
    alert_content = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        **(data or {}),
    }
    try:
        with open(alert_file, "w") as f:
            json.dump(alert_content, f, indent=2)
        structured_log("alert_created", {"file": str(alert_file), "reason": reason})
    except Exception as e:
        structured_log("alert_write_error", {"file": str(alert_file), "error": str(e)}, level="error")


def clear_alert(alert_file: Path):
    """Remove an alert file when condition clears."""
    try:
        if alert_file.exists():
            alert_file.unlink()
            structured_log("alert_cleared", {"file": str(alert_file)})
    except Exception:
        pass


def track_api_error(is_error: bool):
    """Track API call success/failure for error rate alerting."""
    now = time.time()
    API_ERROR_WINDOW.append((now, is_error))
    # Prune old entries outside the window
    cutoff = now - API_ERROR_RATE_WINDOW_SEC
    while API_ERROR_WINDOW and API_ERROR_WINDOW[0][0] < cutoff:
        API_ERROR_WINDOW.pop(0)
    # Check error rate
    if len(API_ERROR_WINDOW) >= 10:  # Need at least 10 calls to judge
        errors = sum(1 for _, e in API_ERROR_WINDOW if e)
        rate = errors / len(API_ERROR_WINDOW)
        if rate > API_ERROR_RATE_THRESHOLD:
            write_alert(API_ERROR_ALERT_FILE, f"API error rate {rate:.0%} > {API_ERROR_RATE_THRESHOLD:.0%}",
                       {"error_rate": round(rate, 3), "total_calls": len(API_ERROR_WINDOW), "errors": errors})
            structured_log("api_error_rate_high", {"rate": round(rate, 3), "errors": errors,
                                                    "total": len(API_ERROR_WINDOW)}, level="warning")
        else:
            clear_alert(API_ERROR_ALERT_FILE)


def check_drawdown(current_balance: float):
    """Check if portfolio has dropped >15% from peak in rolling 24h window."""
    global DRAWDOWN_PEAK_BALANCE
    if current_balance <= 0:
        return
    # Update peak
    if current_balance > DRAWDOWN_PEAK_BALANCE:
        DRAWDOWN_PEAK_BALANCE = current_balance
    # Check drawdown
    if DRAWDOWN_PEAK_BALANCE > 0:
        drawdown = (DRAWDOWN_PEAK_BALANCE - current_balance) / DRAWDOWN_PEAK_BALANCE
        if drawdown > DRAWDOWN_THRESHOLD_PCT:
            write_alert(DRAWDOWN_ALERT_FILE,
                       f"Drawdown {drawdown:.1%} > {DRAWDOWN_THRESHOLD_PCT:.0%} threshold",
                       {"peak_balance": round(DRAWDOWN_PEAK_BALANCE, 2),
                        "current_balance": round(current_balance, 2),
                        "drawdown_pct": round(drawdown * 100, 1)})
            structured_log("drawdown_alert", {"drawdown_pct": round(drawdown * 100, 1),
                                               "peak": round(DRAWDOWN_PEAK_BALANCE, 2),
                                               "current": round(current_balance, 2)}, level="warning")
        else:
            clear_alert(DRAWDOWN_ALERT_FILE)


def check_exposure_alert(positions: list, balance: float):
    """Check if total exposure > 50% of balance and write alert if so."""
    if balance <= 0:
        return
    total_exposure = sum(abs(p.get("market_exposure", 0)) for p in positions) / 100.0
    exposure_pct = total_exposure / balance
    if exposure_pct > MAX_EXPOSURE_PCT:
        write_alert(MAX_EXPOSURE_ALERT_FILE,
                   f"Exposure {exposure_pct:.0%} > {MAX_EXPOSURE_PCT:.0%} limit",
                   {"total_exposure": round(total_exposure, 2),
                    "balance": round(balance, 2),
                    "exposure_pct": round(exposure_pct * 100, 1)})
        structured_log("max_exposure_alert", {"exposure_pct": round(exposure_pct * 100, 1),
                                               "total_exposure": round(total_exposure, 2)}, level="warning")
    else:
        clear_alert(MAX_EXPOSURE_ALERT_FILE)


# ============================================================================
# RISK LIMIT CHECKS (GROK-TRADE-002)
# ============================================================================

def check_risk_limits(positions: list, balance: float, daily_trades: int) -> list:
    """
    Check portfolio-level risk limits before trading.
    Returns list of reasons to skip trading (empty = OK to trade).
    """
    reasons = []

    # 1. Max daily trades
    if daily_trades >= MAX_DAILY_TRADES:
        reason = f"Daily trade limit reached ({daily_trades}/{MAX_DAILY_TRADES})"
        reasons.append(reason)
        structured_log("risk_limit_hit", {"limit": "max_daily_trades",
                                           "current": daily_trades, "max": MAX_DAILY_TRADES}, level="warning")

    # 2. Max total exposure
    if balance > 0:
        total_exposure = sum(abs(p.get("market_exposure", 0)) for p in positions) / 100.0
        exposure_pct = total_exposure / balance
        if exposure_pct > MAX_EXPOSURE_PCT:
            reason = f"Total exposure {exposure_pct:.0%} > {MAX_EXPOSURE_PCT:.0%} limit"
            reasons.append(reason)
            structured_log("risk_limit_hit", {"limit": "max_exposure",
                                               "exposure_pct": round(exposure_pct * 100, 1),
                                               "max_pct": MAX_EXPOSURE_PCT * 100}, level="warning")

    # 3. Max category exposure
    if balance > 0:
        category_exposure = defaultdict(float)
        for p in positions:
            ticker = p.get("ticker", "").upper()
            exposure_val = abs(p.get("market_exposure", 0)) / 100.0
            # Classify category from ticker
            if any(x in ticker for x in ("BTC", "ETH", "SOL", "CRYPTO")):
                category_exposure["crypto"] += exposure_val
            elif any(x in ticker for x in ("HIGH", "LOW", "WEATHER", "TEMP")):
                category_exposure["weather"] += exposure_val
            elif any(x in ticker for x in ("NBA", "NFL", "NHL", "MLB", "CBB", "SOCCER", "SPORT", "NCAA")):
                category_exposure["sports"] += exposure_val
            else:
                category_exposure["other"] += exposure_val

        for cat, cat_exposure in category_exposure.items():
            cat_pct = cat_exposure / balance
            if cat_pct > MAX_CATEGORY_EXPOSURE_PCT:
                reason = f"Category '{cat}' exposure {cat_pct:.0%} > {MAX_CATEGORY_EXPOSURE_PCT:.0%} limit"
                reasons.append(reason)
                structured_log("risk_limit_hit", {"limit": "max_category_exposure",
                                                   "category": cat,
                                                   "exposure_pct": round(cat_pct * 100, 1),
                                                   "max_pct": MAX_CATEGORY_EXPOSURE_PCT * 100}, level="warning")

    return reasons


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
                track_api_error(True)  # GROK-TRADE-002: track API errors
                structured_log("api_error", {"endpoint": endpoint_name,
                                              "status": resp.status_code, "attempt": attempt + 1}, level="error")
                return {"error": f"Server error {resp.status_code}"}

            latency = (time.time() - total_start) * 1000
            record_api_latency(endpoint_name, latency)
            record_api_call("kalshi")
            track_api_error(False)  # GROK-TRADE-002: track API success
            return resp.json()

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            track_api_error(True)  # GROK-TRADE-002: track API errors
            structured_log("api_error", {"endpoint": endpoint_name, "error": "Timeout"}, level="error")
            return {"error": "Timeout"}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            track_api_error(True)  # GROK-TRADE-002: track API errors
            structured_log("api_error", {"endpoint": endpoint_name, "error": "Connection error"}, level="error")
            return {"error": "Connection error"}
        except Exception as e:
            track_api_error(True)  # GROK-TRADE-002: track API errors
            structured_log("api_error", {"endpoint": endpoint_name, "error": str(e)}, level="error")
            return {"error": str(e)}

    track_api_error(True)  # GROK-TRADE-002: track API errors
    return {"error": "Max retries exceeded"}


def get_balance() -> float:
    """Get account balance in dollars."""
    result = kalshi_api("GET", "/trade-api/v2/portfolio/balance")
    if "error" in result:
        print(f"❌ Balance error: {result['error']}")
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


def sell_position(ticker: str, side: str, price_cents: int, count: int, dry_run: bool = True) -> dict:
    """Sell/exit an existing position."""
    if dry_run:
        return {"dry_run": True, "ticker": ticker, "action": "sell", "side": side,
                "price": price_cents, "count": count, "status": "simulated"}
    body = {
        "ticker": ticker, "action": "sell", "side": side, "type": "limit",
        "count": count,
        "yes_price": price_cents if side == "yes" else (100 - price_cents),
    }
    return kalshi_api("POST", "/trade-api/v2/portfolio/orders", body=body)


# ============================================================================
# POSITION MANAGEMENT — Trailing Stop / Early Exit (TRADE-017)
# ============================================================================

# Track peak unrealized profit per position (in-memory, resets on restart)
_position_peaks: dict[str, float] = {}  # ticker → peak unrealized profit ratio

def manage_positions(positions: list, dry_run: bool = True) -> int:
    """
    Monitor open positions and exit when profitable.
    
    For BUY_NO positions:
    - Take profit at +30% (bought NO@60¢ → current value 78¢+)
    - Trailing stop: once +10% profit, trail by 15% from peak
    - Early exit: <2h to expiry and in any profit → close
    
    Returns number of positions exited.
    """
    if not TRAILING_STOP_ENABLED or not positions:
        return 0

    exits = 0
    now = datetime.now(timezone.utc)

    for pos in positions:
        ticker = pos.get("ticker", "")
        if not ticker:
            continue

        # Position details from Kalshi API
        # market_positions have: ticker, market_exposure, total_traded,
        # realized_pnl, rest_resting_contracts, fees_paid, etc.
        # We need to check the current market price vs our entry price
        
        # Get our side and contracts
        yes_contracts = pos.get("position", 0)  # positive = long YES, negative = short YES (= long NO)
        no_contracts = abs(yes_contracts) if yes_contracts < 0 else 0
        yes_held = yes_contracts if yes_contracts > 0 else 0
        
        if yes_held == 0 and no_contracts == 0:
            continue

        # Get current market price
        try:
            market_data = kalshi_api("GET", f"/trade-api/v2/markets/{ticker}")
            if "error" in market_data or "market" not in market_data:
                continue
            mkt = market_data["market"]
        except Exception:
            continue

        current_yes = mkt.get("yes_bid", 0) or mkt.get("last_price", 50)
        current_no = 100 - current_yes
        
        # Calculate cost basis from fills (approximate from total_traded / contracts)
        total_traded_cents = pos.get("total_traded", 0)  # total cost in cents
        total_contracts = abs(yes_contracts)
        if total_contracts == 0:
            continue
        
        entry_price = total_traded_cents / total_contracts  # avg entry price per contract
        
        # Determine our side
        if yes_held > 0:
            side = "yes"
            current_value = current_yes
            contracts = yes_held
        else:
            side = "no"
            current_value = current_no
            contracts = no_contracts

        # Unrealized profit ratio
        if entry_price <= 0:
            continue
        unrealized_pnl_ratio = (current_value - entry_price) / entry_price

        # Track peak
        peak = _position_peaks.get(ticker, unrealized_pnl_ratio)
        if unrealized_pnl_ratio > peak:
            peak = unrealized_pnl_ratio
            _position_peaks[ticker] = peak

        # Check time to expiry
        close_time = mkt.get("close_time") or mkt.get("expiration_time", "")
        hours_to_expiry = float('inf')
        if close_time:
            try:
                exp = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                hours_to_expiry = (exp - now).total_seconds() / 3600
            except Exception:
                pass

        # Decision logic
        should_exit = False
        exit_reason = ""

        # 1. Take profit: position is +30%
        if unrealized_pnl_ratio >= PROFIT_TAKE_PCT:
            should_exit = True
            exit_reason = f"TAKE_PROFIT +{unrealized_pnl_ratio:.0%} (threshold: {PROFIT_TAKE_PCT:.0%})"

        # 2. Hard stop-loss: position is -30% or worse
        elif unrealized_pnl_ratio <= HARD_STOP_LOSS_PCT:
            should_exit = True
            exit_reason = f"STOP_LOSS {unrealized_pnl_ratio:.0%} (limit: {HARD_STOP_LOSS_PCT:.0%})"

        # 3. Trailing stop: peaked above +10%, now dropped 15% from peak
        elif peak >= MIN_PROFIT_TO_TRAIL and (peak - unrealized_pnl_ratio) >= TRAILING_STOP_PCT:
            should_exit = True
            exit_reason = f"TRAILING_STOP peak={peak:.0%} current={unrealized_pnl_ratio:.0%} drop={peak-unrealized_pnl_ratio:.0%}"

        # 4. Early exit near expiry: <2h to expiry and in any profit
        elif hours_to_expiry < EARLY_EXIT_NEAR_EXPIRY_HOURS and unrealized_pnl_ratio > 0.01:
            should_exit = True
            exit_reason = f"EARLY_EXIT {hours_to_expiry:.1f}h to expiry, +{unrealized_pnl_ratio:.0%} profit"

        if should_exit:
            sell_price = current_value  # limit sell at current bid
            print(f"   🔔 EXIT: {ticker}")
            print(f"      Side: {side.upper()} x{contracts}")
            print(f"      Entry: {entry_price:.0f}¢ → Current: {current_value:.0f}¢ ({unrealized_pnl_ratio:+.1%})")
            print(f"      Reason: {exit_reason}")

            result = sell_position(ticker, side, int(sell_price), contracts, dry_run)

            if dry_run:
                print(f"      🧪 DRY RUN: Simulated exit")
            else:
                if "error" in result:
                    print(f"      ❌ Exit failed: {result['error']}")
                    continue
                else:
                    print(f"      ✅ Exit order placed")

            # GROK-TRADE-002: Structured log for position exit
            structured_log("position_exit", {
                "ticker": ticker, "side": side, "contracts": contracts,
                "entry_price": round(entry_price), "exit_price": int(sell_price),
                "unrealized_pnl_pct": round(unrealized_pnl_ratio * 100, 1),
                "reason": exit_reason, "dry_run": dry_run,
            })

            # Log the exit
            exit_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "EXIT",
                "ticker": ticker,
                "side": side,
                "contracts": contracts,
                "entry_price": round(entry_price),
                "exit_price": int(sell_price),
                "unrealized_pnl_pct": round(unrealized_pnl_ratio * 100, 1),
                "reason": exit_reason,
                "dry_run": dry_run,
            }
            try:
                with open(TRADE_LOG_FILE, "a") as f:
                    f.write(json.dumps(exit_entry) + "\n")
            except Exception:
                pass

            # Clean up peak tracking
            _position_peaks.pop(ticker, None)
            exits += 1

            # Update paper state: close position at exit price
            try:
                paper_state = load_paper_state()
                for pos in paper_state.get("positions", []):
                    if pos.get("ticker") == ticker and pos.get("status") == "open":
                        pos["status"] = "exited"
                        pos["exited_at"] = datetime.now(timezone.utc).isoformat()
                        pos["exit_price_cents"] = int(sell_price)
                        pos["exit_reason"] = exit_reason
                        exit_proceeds = int(sell_price) * contracts
                        cost_paid = pos.get("cost_cents", 0)
                        pnl = exit_proceeds - cost_paid
                        pos["pnl_cents"] = pnl
                        paper_state["current_balance_cents"] += exit_proceeds
                        paper_state["stats"]["pnl_cents"] = paper_state["stats"].get("pnl_cents", 0) + pnl
                        if pnl >= 0:
                            paper_state["stats"]["wins"] = paper_state["stats"].get("wins", 0) + 1
                        else:
                            paper_state["stats"]["losses"] = paper_state["stats"].get("losses", 0) + 1
                        break
                # Remove the exited position from active list
                paper_state["positions"] = [p for p in paper_state["positions"]
                                              if not (p.get("ticker") == ticker and p.get("status") == "exited")]
                save_paper_state(paper_state)
            except Exception as e_ps:
                print(f"      ⚠️ Paper state update error: {e_ps}")

            time.sleep(0.5)  # Rate limit
        else:
            # Log status for monitoring
            if unrealized_pnl_ratio > 0.05 or unrealized_pnl_ratio < -0.10:
                status = "📈" if unrealized_pnl_ratio > 0 else "📉"
                print(f"   {status} {ticker}: {side.upper()} entry={entry_price:.0f}¢ now={current_value:.0f}¢ ({unrealized_pnl_ratio:+.1%}) peak={peak:+.1%}")

    return exits


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
# DYNAMIC VOLATILITY (PROC-002 Task 5.1 — fetch from CoinGecko)
# ============================================================================

_vol_cache = {}  # {asset: (timestamp, hourly_vol)}

def get_dynamic_hourly_vol(asset: str = "btc") -> float:
    """Compute realized hourly vol from CoinGecko 7d OHLC data.
    Returns hourly std dev of log returns. Falls back to static constants."""
    global _vol_cache
    
    # Cache for 1 hour
    now = time.time()
    if asset in _vol_cache and (now - _vol_cache[asset][0]) < 3600:
        return _vol_cache[asset][1]
    
    coin_id = "bitcoin" if asset == "btc" else "ethereum"
    fallback = BTC_HOURLY_VOL if asset == "btc" else ETH_HOURLY_VOL
    
    try:
        ohlc = get_crypto_ohlc(coin_id, days=7)
        if not ohlc or len(ohlc) < 24:
            return fallback
        
        # CoinGecko 7d OHLC gives ~4h candles (42 candles for 7 days)
        # Compute log returns from close prices
        closes = [c[4] for c in ohlc if c[4] and c[4] > 0]
        if len(closes) < 10:
            return fallback
        
        log_returns = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
        if not log_returns:
            return fallback
        
        # Std dev of log returns
        mean_ret = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_ret) ** 2 for r in log_returns) / len(log_returns)
        candle_vol = math.sqrt(variance)
        
        # Scale to hourly: CoinGecko 7d gives ~4h candles
        # hourly_vol = candle_vol / sqrt(candle_hours)
        total_hours = (ohlc[-1][0] - ohlc[0][0]) / (1000 * 3600)  # ms to hours
        candle_hours = total_hours / len(log_returns) if len(log_returns) > 0 else 4
        hourly_vol = candle_vol / math.sqrt(max(1, candle_hours))
        
        # Sanity bounds: 0.3% - 5% hourly
        hourly_vol = max(0.003, min(0.05, hourly_vol))
        
        _vol_cache[asset] = (now, hourly_vol)
        structured_log("dynamic_vol", {
            "asset": asset, "hourly_vol_pct": round(hourly_vol * 100, 3),
            "candles": len(closes), "candle_hours": round(candle_hours, 1),
            "fallback_pct": round(fallback * 100, 3)
        })
        return hourly_vol
        
    except Exception as e:
        structured_log("dynamic_vol_error", {"asset": asset, "error": str(e)}, level="warning")
        return fallback


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

    # Dynamic min edge per regime (tightened per Grok analysis 2026-02-16)
    if result["regime"] in ("trending_bullish", "trending_bearish"):
        result["dynamic_min_edge"] = 0.07  # Trending = clearer signal, moderate edge ok
    elif result["regime"] == "choppy":
        result["dynamic_min_edge"] = 0.10  # Choppy = danger zone, require high edge (was 0.08)
    else:
        result["dynamic_min_edge"] = 0.08  # Sideways = uncertain, need decent edge (was 0.06)

    if vol_class in ("high", "very_high"):
        result["dynamic_min_edge"] += 0.02  # Extra penalty for high volatility (was 0.01)

    # PROC-002 Task 5.2: Vol regime classifier — scale MIN_EDGE and Kelly by vol factor
    # Compare dynamic vol to assumed vol, adjust thresholds
    for asset_key in ("btc", "eth"):
        assumed = BTC_HOURLY_VOL if asset_key == "btc" else ETH_HOURLY_VOL
        current = get_dynamic_hourly_vol(asset_key)
        vol_factor = current / assumed if assumed > 0 else 1.0
        if vol_factor > 1.5:
            # High vol regime: tighten edge, reduce Kelly
            result["dynamic_min_edge"] *= (1 + (vol_factor - 1) * 0.4)
            result["vol_kelly_scale"] = 1.0 / vol_factor  # de-risk
            result["vol_regime"] = "high_vol"
        elif vol_factor < 0.7:
            # Low vol regime: loosen edge slightly, increase Kelly
            result["dynamic_min_edge"] *= 0.8
            result["vol_kelly_scale"] = min(1.2, 1.0 / vol_factor)
            result["vol_regime"] = "low_vol"
        else:
            result["vol_kelly_scale"] = 1.0
            result["vol_regime"] = "normal_vol"
        result["vol_factor"] = round(vol_factor, 2)
        break  # Use first asset for now (most positions are BTC)

    # Store ATR info for asset-specific filtering (Grok recommendation #4)
    result["avg_candle_range_pct"] = avg_range
    result["btc_high_vol_skip"] = (avg_range > 0.02)  # ATR > 2% = skip BTC trades (1% was too aggressive, normal BTC ATR ~1.2%)

    result["dynamic_min_edge"] = max(0.05, min(0.20, result["dynamic_min_edge"]))
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
    system_prompt = """You are an expert quantitative forecaster for prediction markets.
Your job: estimate the BASE probability an event resolves YES. Be objective and calibrated.

TICKER FORMAT: KXBTCD-26FEB2717-T66999.99 = "BTC price on 2026-Feb-27 at 17:00 UTC, target $66,999.99".
Use the EXPIRY field — it is the definitive resolution time.

For crypto price markets, reason through:
1. Current price vs target: how much % move is required?
2. Time until expiry: more time = more uncertainty
3. Volatility and momentum: what direction is the market moving?
4. Give your honest base probability estimate based on price gap, time, and momentum only.
   Do NOT apply specific multipliers or numerical discounts — just give your best estimate.

Rules:
- Current price already ABOVE target → high YES probability
- Current price far BELOW target → low YES probability
- <6 hours to expiry: probability should reflect current state strongly (>75% or <25%)
- Provide an honest estimate; the system will apply risk adjustments separately

End with:
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
    crypto_prices = {}
    if context:
        if context.get("crypto_prices"):
            cp = context["crypto_prices"]
            crypto_prices = cp
            extra_context += f"\nCurrent crypto prices: BTC ${cp.get('btc', 0):,.0f}, ETH ${cp.get('eth', 0):,.2f}"
        if context.get("sentiment"):
            s = context["sentiment"]
            extra_context += f"\nFear & Greed Index: {s.get('value', 50)} ({s.get('classification', 'Neutral')})"
            if s.get('value', 50) < 20:
                extra_context += "  ⚠️ EXTREME FEAR — discount all crypto UP moves heavily"
            elif s.get('value', 50) < 35:
                extra_context += "  ⚠️ FEAR — be skeptical of crypto UP bets"
        if context.get("news_sentiment"):
            ns = context["news_sentiment"]
            extra_context += f"\nCrypto news sentiment: {ns.get('sentiment', 'neutral')} (conf: {ns.get('confidence', 0.5):.0%})"
        if context.get("momentum"):
            m = context["momentum"]
            extra_context += f"\nMomentum: BTC dir={m.get('btc_direction', 0):.2f}, ETH dir={m.get('eth_direction', 0):.2f}"
        if context.get("regime"):
            r = context["regime"]
            extra_context += f"\nMarket regime: {r.get('regime', 'unknown')} ({r.get('confidence', 0):.0%} conf), vol: {r.get('volatility', 'normal')}"

    # For crypto price markets: calculate the gap between current and target
    price_gap_note = ""
    if crypto_prices and market.category in ("crypto", "cryptocurrency"):
        ticker_upper = market.ticker.upper()
        current = 0
        if "BTC" in ticker_upper:
            current = crypto_prices.get("btc", 0)
        elif "ETH" in ticker_upper:
            current = crypto_prices.get("eth", 0)
        if current > 0:
            import re as _re
            nums = _re.findall(r'[\d,]+\.?\d*', market.title.replace(',', ''))
            targets = [float(n.replace(',', '')) for n in nums if float(n.replace(',', '')) > 100]
            if targets:
                target = max(targets)
                gap_pct = (target - current) / current * 100
                direction = "above" if "above" in market.title.lower() or "over" in market.title.lower() else "below"
                price_gap_note = f"\n⚡ PRICE GAP: current=${current:,.2f}, target=${target:,.0f}, gap={gap_pct:+.1f}% ({'UP' if gap_pct>0 else 'DOWN'} needed for YES/{direction})"

    user_prompt = f"""Analyze this prediction market and estimate the true probability:

MARKET: {market.title}
{f'DETAILS: {market.subtitle}' if market.subtitle else ''}
CATEGORY: {market.category}
YES PRICE: {market.yes_price}¢ (market implies {market.market_prob:.0%} probability)
NO PRICE: {market.no_price}¢
VOLUME: {market.volume:,} contracts
EXPIRY: {expiry_str} ({time_desc} remaining)
TICKER: {market.ticker}
{extra_context}{price_gap_note}

Today: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

What is the TRUE probability this resolves YES? Be specific about price levels and realistic about required moves."""

    result = call_claude(system_prompt, user_prompt, max_tokens=1500)

    if result.get("error"):
        return ForecastResult(probability=market.market_prob, reasoning=f"Error: {result['error']}",
                              confidence="low", model_used="error", tokens_used=0)

    content = result["content"]
    raw_prob = parse_probability(content) or market.market_prob

    # Apply calibration: direction depends on market_prob
    # For unlikely events (<30%): calibrate toward 0 (LLM overestimates tail risk)
    # For likely events (>70%): calibrate toward 1 (LLM underestimates near-certainty)
    # For middle range: shrink toward 50% (standard overconfidence correction)
    if market.market_prob < 0.30:
        prob = raw_prob * CALIBRATION_FACTOR  # compress toward 0%
    elif market.market_prob > 0.70:
        prob = 1.0 - (1.0 - raw_prob) * CALIBRATION_FACTOR  # compress toward 100%
    else:
        prob = 0.5 + (raw_prob - 0.5) * CALIBRATION_FACTOR  # compress toward 50%
    prob = max(0.03, min(0.97, prob))

    # Apply regime discounts algorithmically (not via LLM prompt)
    if context and market.category in ("crypto", "cryptocurrency"):
        # Sentiment discount for crypto UP moves
        fear_val = 50
        if context.get("sentiment"):
            fear_val = context["sentiment"].get("value", 50)

        # Determine if this is an "UP move" market (YES = price goes up)
        is_up_market = False
        gap_pct = 0.0
        ticker_upper = market.ticker.upper()
        current_price = 0.0
        if "BTC" in ticker_upper:
            current_price = context.get("crypto_prices", {}).get("btc", 0)
        elif "ETH" in ticker_upper:
            current_price = context.get("crypto_prices", {}).get("eth", 0)
        if current_price > 0:
            import re as _re2
            nums = _re2.findall(r'[\d,]+\.?\d*', market.title.replace(',', ''))
            targets = [float(n.replace(',', '')) for n in nums if float(n.replace(',', '')) > 100]
            if targets:
                target_price = max(targets)
                gap_pct = (target_price - current_price) / current_price * 100
                is_up_market = gap_pct > 0  # YES requires upward move

        if is_up_market and gap_pct > 0.5:  # Only apply for genuine up-moves (>0.5%)
            if fear_val < 20:    # Extreme Fear: big discount
                regime_discount = 0.5
            elif fear_val < 35:  # Fear: moderate discount
                regime_discount = 0.7
            else:
                regime_discount = 1.0
            # Also apply bearish momentum discount
            if context.get("momentum"):
                btc_dir = context["momentum"].get("btc_direction", 0)
                eth_dir = context["momentum"].get("eth_direction", 0)
                if "BTC" in ticker_upper and btc_dir < -0.3:
                    regime_discount *= 0.85
                elif "ETH" in ticker_upper and eth_dir < -0.3:
                    regime_discount *= 0.85
            if regime_discount < 1.0:
                prob = max(0.05, prob * regime_discount)
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
    system_prompt = """You are a critical analyst reviewing probability forecasts for prediction markets.
Your job: catch SEVERE errors only. Minor uncertainty or suboptimal analysis is acceptable.

ONLY mark SHOULD_TRADE: no for these CRITICAL issues:
- The forecaster's final probability directly contradicts its own reasoning (e.g., says "very unlikely" then outputs 60%)
- Wrong direction: forecaster recommends BUY_YES but the bet should be BUY_NO based on price gap
- Gross hallucination: fabricated data, wrong current price by >10%, completely wrong event

Do NOT veto for: slightly different probability estimates, methodological preferences, cautious wording, or minor inconsistencies.
A trade with genuine edge (>5%) and correct direction should NOT be vetoed unless there is a CRITICAL error above.

MAJOR_FLAWS should only list truly critical issues. If analysis is roughly reasonable, output MAJOR_FLAWS: NONE.

End with:
ADJUSTED_PROBABILITY: XX%
MAJOR_FLAWS: [critical_flaw1] (or NONE)
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
    raw_adj = parse_probability(content) or forecast.probability
    # Apply calibration (same direction-aware formula as forecaster)
    if market.market_prob < 0.30:
        adj_prob = raw_adj * CALIBRATION_FACTOR
    elif market.market_prob > 0.70:
        adj_prob = 1.0 - (1.0 - raw_adj) * CALIBRATION_FACTOR
    else:
        adj_prob = 0.5 + (raw_adj - 0.5) * CALIBRATION_FACTOR
    adj_prob = max(0.03, min(0.97, adj_prob))
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
    # If no major flaws found, override SHOULD_TRADE to True (can't veto without evidence)
    if not major_flaws:
        should_trade = True

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

    # Get hourly vol using EWMA (GROK-TRADE-006: dynamic vol estimation)
    # EWMA gives more weight to recent returns, captures vol clustering in crypto
    # PROC-002 Task 5.1: use dynamic vol from CoinGecko, fallback to static
    hourly_vol = get_dynamic_hourly_vol(asset) if asset in ("btc", "eth") else 0.005
    ohlc_data = (context or {}).get("ohlc", {}).get(asset, [])
    if ohlc_data and len(ohlc_data) >= 10:
        try:
            closes = [c[4] for c in ohlc_data[-48:] if c and len(c) >= 5 and c[4] and c[4] > 0]
            if len(closes) >= 5:
                log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
                # EWMA volatility (lambda=0.94, standard RiskMetrics)
                ewma_lambda = 0.94
                ewma_var = log_returns[0] ** 2  # Initialize with first squared return
                for r in log_returns[1:]:
                    ewma_var = ewma_lambda * ewma_var + (1 - ewma_lambda) * r ** 2
                ewma_vol = math.sqrt(ewma_var)
                default_vol = hourly_vol
                # Allow wider range: 0.3x to 3x default (EWMA can detect regime shifts)
                hourly_vol = max(default_vol * 0.3, min(default_vol * 3.0, ewma_vol))
        except Exception:
            pass

    # Student's t probability model (replaces log-normal per Grok calibration 2026-02-16)
    # t-distribution captures fat tails in crypto hourly returns (df=4-5 typical)
    T = minutes_left / 60.0
    sigma = hourly_vol * math.sqrt(T)
    if sigma <= 0 or current_price <= 0 or strike <= 0:
        return market_prob, "low", ["Invalid model params"], ["Bad data"]

    log_ratio = math.log(current_price / strike)
    # Standardized distance in sigma units
    z = log_ratio / sigma

    # Student's t CDF approximation (df=4 for crypto fat tails)
    # df=4 gives heavier tails than normal: P(|X|>2σ) ≈ 13% vs 5% for normal
    CRYPTO_T_DF = 4
    try:
        from scipy.stats import t as t_dist
        prob_above = max(0.05, min(0.95, t_dist.cdf(z, df=CRYPTO_T_DF)))
    except ImportError:
        # Fallback: approximate t-CDF using normal CDF with wider tails
        # For df=4, scale z by ~0.85 to approximate fatter tails
        tail_factor = 0.85  # Makes distribution wider (more probability in tails)
        z_adjusted = z * tail_factor
        def norm_cdf(x):
            a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
            p_val = 0.3275911
            sign = 1 if x >= 0 else -1
            x = abs(x)
            t = 1.0 / (1.0 + p_val * x)
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
            return 0.5 * (1.0 + sign * y)
        prob_above = max(0.05, min(0.95, norm_cdf(z_adjusted)))

    # Momentum adjustment
    momentum = (context or {}).get("momentum", {}).get(asset, {})
    mom_dir = momentum.get("composite_direction", 0)
    mom_adj = mom_dir * 0.03  # Max ±3% adjustment
    prob_above = max(0.05, min(0.95, prob_above + mom_adj))

    # Sentiment adjustment (Fear & Greed Index)
    # F&G 10 = extreme fear → bearish bias; F&G 90 = extreme greed → bullish bias
    # FIXED: Previous /500 was too weak (only -8% at extreme fear)
    # New: at F&G=10 → -16% adj; at F&G=90 → +16% adj; at F&G=50 → 0
    # Extreme fear (F&G < 20) gets EXTRA penalty since markets trend DOWN in fear
    fng = (context or {}).get("sentiment", {})
    sentiment_val = fng.get("value", 50)
    sentiment_adj = (sentiment_val - 50) / 250  # -20% to +20% range (was /500 = -8%, way too weak)
    # Extra bearish penalty in extreme fear — the market trends DOWN hard
    if sentiment_val < 20:
        extreme_fear_penalty = (20 - sentiment_val) / 100  # up to -20% extra at F&G=0
        sentiment_adj -= extreme_fear_penalty
    prob_above += sentiment_adj

    # Regime adjustment: choppy + high vol = shrink toward 50% more (less directional confidence)
    regime = (context or {}).get("regime", {})
    if isinstance(regime, dict) and regime.get("regime") == "choppy" and regime.get("volatility") in ("high", "very_high"):
        # In choppy high-vol, further shrink away from extremes
        prob_above = 0.5 + (prob_above - 0.5) * 0.80  # 20% additional shrink

    # News sentiment adjustment
    news = (context or {}).get("news_sentiment")
    if news and news.get("edge_adjustment"):
        prob_above = max(0.05, min(0.95, prob_above + news["edge_adjustment"] * 0.5))

    prob_above = max(0.05, min(0.95, prob_above))

    # Calibration scaling — use global CALIBRATION_FACTOR (same as LLM forecaster)
    prob_above = 0.5 + (prob_above - 0.5) * CALIBRATION_FACTOR
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


def make_trade_decision(market: MarketInfo, forecast: ForecastResult, critic: CriticResult,
                        balance: float) -> TradeDecision:
    """Compare probability vs market price and decide. Uses split thresholds (v3 data-driven)."""
    final_prob = 0.6 * forecast.probability + 0.4 * critic.adjusted_probability

    # Overconfidence decay: shrink toward 50% for BUY_YES on markets >72h out only
    # (BUY_NO direction: NO decay — in bearish regime, long-dated NO is actually MORE certain)
    try:
        close_str = getattr(market, 'expiry', '') or ''
        if close_str:
            close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            hours_to_close = (close_dt - datetime.now(timezone.utc)).total_seconds() / 3600
            if hours_to_close > 72 and final_prob > market.market_prob:
                # Only apply decay for BUY_YES (final_prob > market = YES looks underpriced)
                decay_rate = 0.002  # -0.2% per hour past 72h (reduced from 0.3%)
                decay_hours = min(hours_to_close - 72, 200)
                decay = decay_rate * decay_hours
                final_prob = final_prob * (1 - decay) + 0.5 * decay
                final_prob = max(0.01, min(0.99, final_prob))
    except Exception:
        pass
    market_prob = market.market_prob
    edge_yes = final_prob - market_prob
    
    # Subtract round-trip fees/slippage (Grok review #2: edge overstated by 1-3%)
    ROUND_TRIP_FEE = 0.007  # ~0.7% taker fees on Kalshi
    if edge_yes > 0:
        edge_yes -= ROUND_TRIP_FEE
    elif edge_yes < 0:
        edge_yes += ROUND_TRIP_FEE  # For BUY_NO, edge_yes is negative

    # Dynamic edge minimum based on liquidity (Grok review: illiquid markets need higher bar)
    # Low volume (<1k contracts) → add 1% to min edge. Very low (<500) → add 2%.
    if market.volume < 500:
        liq_edge_bump = 0.02
    elif market.volume < 1000:
        liq_edge_bump = 0.01
    else:
        liq_edge_bump = 0.0
    dyn_min_yes = MIN_EDGE_BUY_YES + liq_edge_bump
    dyn_min_no  = MIN_EDGE_BUY_NO  + liq_edge_bump

    # Split thresholds
    if edge_yes > 0:
        if edge_yes < dyn_min_yes:
            return TradeDecision(action="SKIP", edge=edge_yes, kelly_size=0, contracts=0,
                                price_cents=0, reason=f"BUY_YES edge {edge_yes:.1%} < {dyn_min_yes:.0%} (vol={market.volume})",
                                forecast=forecast, critic=critic)
    else:
        if abs(edge_yes) < dyn_min_no:
            return TradeDecision(action="SKIP", edge=edge_yes, kelly_size=0, contracts=0,
                                price_cents=0, reason=f"BUY_NO edge {abs(edge_yes):.1%} < {dyn_min_no:.0%} (vol={market.volume})",
                                forecast=forecast, critic=critic)

    if not critic.should_trade:
        return TradeDecision(action="SKIP", edge=edge_yes, kelly_size=0, contracts=0, price_cents=0,
                            reason=f"Critic vetoed: {', '.join(critic.major_flaws[:2]) or 'concerns'}",
                            forecast=forecast, critic=critic)
    # Only block on 3+ critical flaws (critic now only lists truly critical ones)
    if len(critic.major_flaws) >= 3:
        return TradeDecision(action="SKIP", edge=edge_yes, kelly_size=0, contracts=0, price_cents=0,
                            reason=f"Too many flaws: {', '.join(critic.major_flaws[:2])}",
                            forecast=forecast, critic=critic)
    # Low-confidence filter disabled in paper mode — we need data on all edge levels
    # if forecast.confidence == "low" and edge_yes > 0 and abs(edge_yes) < 0.10:
    #     return TradeDecision(action="SKIP", ...)

    # Cap edges
    if abs(edge_yes) > MAX_EDGE_CAP:
        edge_yes = MAX_EDGE_CAP if edge_yes > 0 else -MAX_EDGE_CAP
        final_prob = market_prob + edge_yes

    # Parlay BUY_YES filter
    market_type = classify_market_type(market)
    if edge_yes > 0 and PARLAY_ONLY_NO and market_type == "combo":
        num_legs = estimate_combo_legs(market)
        allow = PARLAY_YES_EXCEPTION and num_legs <= 2 and edge_yes > 0.05 and market.yes_price >= 30
        if not allow:
            return TradeDecision(action="SKIP", edge=edge_yes, kelly_size=0, contracts=0, price_cents=0,
                                reason=f"Parlay BUY_YES blocked (9.7% WR)", forecast=forecast, critic=critic)

    # Decide side
    if edge_yes > 0:
        action, side_price, edge = "BUY_YES", market.yes_price, edge_yes
    else:
        action, side_price, edge = "BUY_NO", market.no_price, abs(edge_yes)

    # ── Risk/Reward filters (TRADE-003) ──
    # 1. Hard cap on BUY_NO price
    if action == "BUY_NO" and side_price > MAX_NO_PRICE_CENTS:
        return TradeDecision(action="SKIP", edge=edge, kelly_size=0, contracts=0, price_cents=side_price,
                            reason=f"BUY_NO price {side_price}¢ > {MAX_NO_PRICE_CENTS}¢ cap (bad risk/reward)",
                            forecast=forecast, critic=critic)

    # 2. Scaled edge requirement for expensive BUY_NO
    if action == "BUY_NO" and NO_PRICE_EDGE_SCALE and side_price > 50:
        scaled_min_edge = 0.03 + (side_price - 50) * 0.001  # 3% base + 0.1% per cent above 50
        if edge < scaled_min_edge:
            return TradeDecision(action="SKIP", edge=edge, kelly_size=0, contracts=0, price_cents=side_price,
                                reason=f"BUY_NO {side_price}¢ needs {scaled_min_edge:.1%} edge, got {edge:.1%}",
                                forecast=forecast, critic=critic)

    # 3. General risk/reward ratio check
    potential_win = 100 - side_price  # cents won if correct
    risk_reward = side_price / potential_win if potential_win > 0 else 999
    if risk_reward > MAX_RISK_REWARD_RATIO:
        return TradeDecision(action="SKIP", edge=edge, kelly_size=0, contracts=0, price_cents=side_price,
                            reason=f"Risk/reward {risk_reward:.2f} > {MAX_RISK_REWARD_RATIO} ({side_price}¢ risk / {potential_win}¢ win)",
                            forecast=forecast, critic=critic)

    # Kelly sizing (PROC-002 Task 5.2: apply vol regime scaling)
    kelly_frac = calculate_kelly(final_prob if action == "BUY_YES" else (1 - final_prob), side_price)

    # PROC-002 Task 4.2: Recovery mode — halve Kelly when drawdown > 10%
    if DRAWDOWN_PEAK_BALANCE > 0 and balance < DRAWDOWN_PEAK_BALANCE:
        current_drawdown = (DRAWDOWN_PEAK_BALANCE - balance) / DRAWDOWN_PEAK_BALANCE
        if current_drawdown > 0.20:
            kelly_frac *= 0.25  # Severe drawdown: quarter Kelly
            structured_log("recovery_mode", {"drawdown_pct": round(current_drawdown * 100, 1),
                                             "kelly_scale": 0.25}, level="warning")
        elif current_drawdown > 0.10:
            kelly_frac *= 0.50  # Moderate drawdown: halve Kelly
            structured_log("recovery_mode", {"drawdown_pct": round(current_drawdown * 100, 1),
                                             "kelly_scale": 0.50}, level="warning")

    # PROC-002 Task 5.2: Scale Kelly by vol regime factor
    asset = "btc" if "btc" in market.ticker.lower() or "BTC" in market.ticker else "eth"
    dyn_vol = get_dynamic_hourly_vol(asset)
    assumed_vol = BTC_HOURLY_VOL if asset == "btc" else ETH_HOURLY_VOL
    vol_factor = dyn_vol / assumed_vol if assumed_vol > 0 else 1.0
    if vol_factor > 1.5:
        kelly_frac /= vol_factor  # de-risk in high vol
    elif vol_factor < 0.7:
        kelly_frac *= min(1.2, 1.0 / vol_factor)  # slightly more aggressive in low vol
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
    # Crypto bonus: always tradeable, boost for hourly contracts
    ticker = market.ticker.upper()
    if any(x in ticker for x in ("KXBTC", "KXETH", "KXSOL")):
        score += 5  # Crypto markets get priority (trade 24/7)
        if dte < 0.1:  # Hourly contracts — our bread and butter
            score += 3
    return score


def scan_all_markets() -> list:
    """Scan all open Kalshi markets with pagination + sports tickers."""
    all_markets = []
    seen = set()
    cursor = None

    print("📡 Scanning Kalshi markets...")
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

    print(f"   General: {len(all_markets)} markets")

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
        print(f"   Sports: +{sports_found} additional markets")

    # Crypto series scan (hourly/daily contracts — trade 24/7)
    crypto_found = 0
    for ct in CRYPTO_SERIES_TICKERS:
        path = f"/trade-api/v2/markets?limit=200&series_ticker={ct}&status=open"
        result = kalshi_api("GET", path)
        if "error" not in result:
            for r in result.get("markets", []):
                m = parse_market(r)
                if m and m.ticker not in seen:
                    all_markets.append(m)
                    seen.add(m.ticker)
                    crypto_found += 1
        time.sleep(0.2)
    if crypto_found:
        print(f"   Crypto: +{crypto_found} additional markets")

    # Filter
    filtered = filter_markets(all_markets)
    print(f"   Filtered: {len(filtered)}/{len(all_markets)} pass criteria")
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
        # GROK-TRADE-004: Create alert file for loss streak
        write_alert(LOSS_STREAK_ALERT_FILE,
                   f"Circuit breaker triggered: {losses} consecutive losses",
                   {"consecutive_losses": losses, "threshold": CIRCUIT_BREAKER_THRESHOLD})
        return True, losses, f"TRIGGERED: {losses} consecutive losses"

    # Clear loss streak alert if we're under threshold
    clear_alert(LOSS_STREAK_ALERT_FILE)
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
                        # API returns {"market": {"result": "yes"/"no", ...}}
                        market_data = result.get("market", result)
                        market_result = market_data.get("result", "")
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
                            # SQLite settlement
                            if _DB_AVAILABLE:
                                try:
                                    contracts   = entry.get("contracts", 0)
                                    price_cents = entry.get("price_cents", 50)
                                    if entry["result_status"] == "won":
                                        pnl_c = contracts * (100 - price_cents)
                                    else:
                                        pnl_c = -(contracts * price_cents)
                                    entry["pnl_cents"] = pnl_c
                                    _db.upsert_trade_by_ticker_ts(entry)
                                except Exception:
                                    pass
                    time.sleep(0.2)  # Rate limit
                updated_lines.append(json.dumps(entry) + "\n")
            except Exception:
                updated_lines.append(line)

        if updated_count > 0:
            with open(TRADE_LOG_FILE, "w") as f:
                f.writelines(updated_lines)
            # GROK-TRADE-002: structured log for settlements
            structured_log("settlement", {"updated": updated_count, "wins": wins, "losses": losses})

            # Update paper trade state with settlements
            try:
                paper_state = load_paper_state()
                for line in updated_lines:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("settled_at") and entry.get("ticker"):
                            won = entry.get("result_status") == "won"
                            paper_trade_settle(paper_state, entry["ticker"], won)
                    except Exception:
                        continue
            except Exception as e:
                print(f"⚠️ Paper state settlement update error: {e}")

    except Exception as e:
        print(f"⚠️ Settlement check error: {e}")
        structured_log("api_error", {"context": "settlement_check", "error": str(e)}, level="error")

    return {"updated": updated_count, "wins": wins, "losses": losses}


# ============================================================================
# PAPER TRADE STATE (bankroll/positions tracking for dashboard)
# ============================================================================

def load_paper_state() -> dict:
    """Load paper trade state from file, or create fresh if missing."""
    if PAPER_STATE_FILE.exists():
        try:
            with open(PAPER_STATE_FILE) as f:
                state = json.load(f)
            # Migration: ensure all fields exist
            state.setdefault("starting_balance_cents", PAPER_STARTING_BANKROLL_CENTS)
            state.setdefault("current_balance_cents", PAPER_STARTING_BANKROLL_CENTS)
            state.setdefault("positions", [])
            state.setdefault("stats", {
                "total_trades": 0, "wins": 0, "losses": 0, "pending": 0,
                "win_rate": 0.0, "pnl_cents": 0, "peak_balance_cents": PAPER_STARTING_BANKROLL_CENTS,
            })
            return state
        except Exception:
            pass

    # Fresh state
    return {
        "session_id": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "starting_balance_cents": PAPER_STARTING_BANKROLL_CENTS,
        "current_balance_cents": PAPER_STARTING_BANKROLL_CENTS,
        "mode": "paper",
        "strategy_version": "v3-unified",
        "positions": [],
        "stats": {
            "total_trades": 0, "wins": 0, "losses": 0, "pending": 0,
            "win_rate": 0.0, "pnl_cents": 0,
            "gross_profit_cents": 0, "gross_loss_cents": 0,
            "peak_balance_cents": PAPER_STARTING_BANKROLL_CENTS,
            "max_drawdown_cents": 0,
        },
        "trade_history": [],
    }


def save_paper_state(state: dict):
    """Save paper trade state to file."""
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Update computed fields
    stats = state.get("stats", {})
    total = stats.get("wins", 0) + stats.get("losses", 0)
    stats["win_rate"] = round(stats["wins"] / total, 4) if total > 0 else 0.0
    stats["pending"] = len(state.get("positions", []))
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Track peak balance / max drawdown
    bal = state.get("current_balance_cents", 0)
    peak = stats.get("peak_balance_cents", bal)
    if bal > peak:
        stats["peak_balance_cents"] = bal
        peak = bal
    dd = peak - bal
    if dd > stats.get("max_drawdown_cents", 0):
        stats["max_drawdown_cents"] = dd
    # Also track percentage drawdown (Grok review bug #4)
    dd_pct = (dd / peak * 100) if peak > 0 else 0
    if dd_pct > stats.get("max_drawdown_pct", 0):
        stats["max_drawdown_pct"] = round(dd_pct, 2)

    with open(PAPER_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def paper_trade_open(state: dict, ticker: str, action: str, price_cents: int, contracts: int,
                     title: str = "", edge: float = 0.0, expiry: str = ""):
    """Record a new paper trade: deduct cost from bankroll, add to positions."""
    # Check max positions limit (Grok review bug #3)
    open_positions = [p for p in state.get("positions", []) if p.get("status") == "open"]
    current_balance = state.get("current_balance_cents", 10000) / 100
    dyn_max = dynamic_max_positions(current_balance)
    if len(open_positions) >= dyn_max:
        print(f"  ⚠️ Max positions ({dyn_max}, balance ${current_balance:.0f}) reached, skipping {ticker}")
        return
    
    cost = contracts * price_cents
    # Don't allow negative balance (Grok review: overbetting fix)
    if state["current_balance_cents"] - cost < 0:
        print(f"  ⚠️ Insufficient balance for {ticker} (need {cost}c, have {state['current_balance_cents']}c)")
        return
    
    state["current_balance_cents"] -= cost

    position = {
        "ticker": ticker,
        "action": action,  # BUY_YES or BUY_NO
        "price_cents": price_cents,
        "contracts": contracts,
        "cost_cents": cost,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "title": title[:100],
        "edge": round(edge, 4),
        "expiry": expiry,
        "status": "open",
    }
    state["positions"].append(position)
    state["stats"]["total_trades"] = state["stats"].get("total_trades", 0) + 1

    # Keep recent trade history (last 200)
    history = state.setdefault("trade_history", [])
    history.append({
        "timestamp": position["opened_at"],
        "ticker": ticker,
        "action": action,
        "price_cents": price_cents,
        "contracts": contracts,
        "cost_cents": cost,
        "status": "open",
    })
    if len(history) > 200:
        state["trade_history"] = history[-200:]

    save_paper_state(state)


def paper_trade_settle(state: dict, ticker: str, won: bool):
    """Settle a paper trade: if won, add payout (100¢/contract); if lost, cost already deducted."""
    settled = False
    for pos in state.get("positions", []):
        if pos.get("ticker") == ticker and pos.get("status") == "open":
            pos["status"] = "won" if won else "lost"
            pos["settled_at"] = datetime.now(timezone.utc).isoformat()
            contracts = pos.get("contracts", 1)
            cost = pos.get("cost_cents", 0)

            if won:
                # Payout: 100¢ per contract for winning side
                payout = contracts * 100
                profit = payout - cost
                state["current_balance_cents"] += payout
                state["stats"]["wins"] = state["stats"].get("wins", 0) + 1
                state["stats"]["pnl_cents"] = state["stats"].get("pnl_cents", 0) + profit
                state["stats"]["gross_profit_cents"] = state["stats"].get("gross_profit_cents", 0) + profit
                pos["pnl_cents"] = profit
            else:
                # Loss: cost already deducted at open time, nothing to add back
                state["stats"]["losses"] = state["stats"].get("losses", 0) + 1
                state["stats"]["pnl_cents"] = state["stats"].get("pnl_cents", 0) - cost
                state["stats"]["gross_loss_cents"] = state["stats"].get("gross_loss_cents", 0) + cost
                pos["pnl_cents"] = -cost

            # Update trade history
            for h in reversed(state.get("trade_history", [])):
                if h.get("ticker") == ticker and h.get("status") == "open":
                    h["status"] = "won" if won else "lost"
                    h["pnl_cents"] = pos["pnl_cents"]
                    break

            settled = True
            # Don't break — settle ALL matching positions for this ticker

    if settled:
        # Remove settled positions from active positions
        state["positions"] = [p for p in state["positions"] if not (p.get("ticker") == ticker and p.get("status") != "open")]
        save_paper_state(state)

    return settled


# ============================================================================
# LOGGING
# ============================================================================

def log_trade(market: MarketInfo, decision: TradeDecision, order_result: dict, dry_run: bool):
    """Log trade to JSONL files."""
    for log_path in [TRADE_LOG_FILE, LEGACY_TRADE_LOG, V3_TRADE_LOG]:
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

    # SQLite
    if _DB_AVAILABLE and entry.get("action") in ("BUY_YES", "BUY_NO"):
        try:
            _db.upsert_trade_by_ticker_ts(entry)
        except Exception as _e:
            print(f"⚠️  DB insert_trade error: {_e}")


def log_cycle(stats: dict):
    CYCLE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **stats}
    with open(CYCLE_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    if _DB_AVAILABLE:
        try:
            _db.insert_cycle(entry)
        except Exception:
            pass


def log_skip(ticker: str, reason: str, details: dict = None):
    SKIP_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "ticker": ticker,
             "reason": reason, **(details or {})}
    with open(SKIP_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================================================================
# POST-TRADE MONITORING (GROK-TRADE-004)
# ============================================================================

def log_decision(market: MarketInfo, decision: TradeDecision, outcome: str):
    """
    Log every trade decision (vetoed OR passed) to a dedicated JSONL file.
    outcome: "executed", "vetoed", "skipped_risk", "skipped_edge", "skipped_parlay"
    """
    DECISION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": market.ticker,
        "title": market.title[:80],
        "category": market.category,
        "outcome": outcome,
        "action": decision.action,
        "edge": round(decision.edge, 4),
        "price_cents": decision.price_cents,
        "contracts": decision.contracts,
        "reason": decision.reason,
        "forecast_prob": round(decision.forecast.probability, 4) if decision.forecast else None,
        "forecast_confidence": decision.forecast.confidence if decision.forecast else None,
        "critic_should_trade": decision.critic.should_trade if decision.critic else None,
        "critic_flaws": decision.critic.major_flaws if decision.critic else [],
    }
    try:
        with open(DECISION_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        structured_log("decision_log_error", {"error": str(e)}, level="error")


def check_high_edge_cluster():
    """
    GROK-TRADE-004: Alert if we see clusters of high-edge skipped trades.
    If 5+ trades in the last hour had edge > 15% but were skipped/vetoed,
    the forecaster may be miscalibrated or market conditions are unusual.
    """
    if not DECISION_LOG_FILE.exists():
        return
    try:
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        high_edge_skips = 0
        with open(DECISION_LOG_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    ts = datetime.fromisoformat(entry["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts >= one_hour_ago and entry.get("outcome") != "executed":
                        if abs(entry.get("edge", 0)) > 0.15:
                            high_edge_skips += 1
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        if high_edge_skips >= 5:
            write_alert(HIGH_EDGE_CLUSTER_ALERT_FILE,
                       f"{high_edge_skips} high-edge trades (>15%) skipped/vetoed in last hour — possible forecaster miscalibration",
                       {"count": high_edge_skips, "threshold": 5, "edge_threshold": 0.15})
            structured_log("high_edge_cluster", {"count": high_edge_skips}, level="warning")
        else:
            clear_alert(HIGH_EDGE_CLUSTER_ALERT_FILE)
    except Exception as e:
        structured_log("high_edge_cluster_check_error", {"error": str(e)}, level="error")


def check_daily_exposure_cap(daily_trades_cost_cents: int) -> bool:
    """
    GROK-TRADE-004: Check if daily new exposure exceeds absolute $ cap.
    Returns True if we should pause trading.
    """
    daily_cost_usd = daily_trades_cost_cents / 100.0
    if daily_cost_usd >= MAX_DAILY_EXPOSURE_USD:
        structured_log("daily_exposure_cap", {
            "daily_cost_usd": round(daily_cost_usd, 2),
            "cap_usd": MAX_DAILY_EXPOSURE_USD
        }, level="warning")
        return True
    return False


def get_daily_trades_cost() -> int:
    """Sum cost_cents of all trades executed today."""
    if not TRADE_LOG_FILE.exists():
        return 0
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total_cost = 0
    try:
        with open(TRADE_LOG_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("timestamp", "").startswith(today_str) and entry.get("action") != "SKIP":
                        total_cost += entry.get("cost_cents", 0)
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception:
        pass
    return total_cost


# ============================================================================
# MAIN TRADING CYCLE
# ============================================================================

def run_cycle(dry_run: bool = True, max_markets: int = 30, max_trades: int = 10):
    """
    One complete trading cycle:
    1. Check safety (circuit breaker, daily loss, holiday)
    2. Gather context (prices, momentum, sentiment, regime)
    3. Scan & rank markets
    4. For top N: Forecast → Critique → Decide → Execute
    5. Log everything
    """
    cycle_start = time.time()

    # GROK-TRADE-002: check shutdown before starting
    if shutdown_requested:
        print("⚠️  Shutdown requested — skipping cycle")
        return

    # GROK-TRADE-002: structured log cycle start
    structured_log("cycle_start", {"dry_run": dry_run, "max_markets": max_markets, "max_trades": max_trades})

    print("=" * 70)
    print(f"🤖 KALSHI AUTOTRADER — Unified")
    print(f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'🧪 DRY RUN (PAPER)' if dry_run else '🔴 LIVE TRADING'}")
    print("=" * 70)

    # ── Check holiday ──
    # NOTE: Crypto markets (KXBTC, KXETH, KXSOL) trade 24/7 including holidays.
    # We only skip non-crypto markets on holidays, not the entire cycle.
    is_holiday_today = False
    holiday_name = ""
    if HOLIDAY_CHECK_AVAILABLE:
        is_holiday_today, holiday_name = is_market_holiday()
        if is_holiday_today:
            print(f"🎄 Market holiday: {holiday_name} — crypto-only mode (non-crypto skipped)")

    # ── Settlement tracker ──
    settlement = update_trade_results()
    if settlement["updated"]:
        print(f"📊 Settled: {settlement['updated']} trades ({settlement['wins']}W/{settlement['losses']}L)")

    # ── Circuit breaker ── (skip in paper mode — we want max data)
    cb_paused, cb_losses, cb_msg = check_circuit_breaker()
    print(f"🔒 Circuit breaker: {cb_msg}")
    if cb_paused and not dry_run:
        structured_log("circuit_breaker", {"paused": True, "losses": cb_losses, "message": cb_msg}, level="warning")
        return
    elif cb_paused:
        structured_log("circuit_breaker", {"paused": True, "losses": cb_losses, "paper_mode": True})
        print("   📝 Paper mode — ignoring circuit breaker, continuing for data collection")

    # ── Daily loss limit ── (skip in paper mode — no real money at risk)
    dl_paused, dl_pnl = check_daily_loss_limit()
    print(f"📊 Daily PnL: ${dl_pnl['net_pnl_cents']/100:+.2f} ({dl_pnl['trades_today']} trades)")
    if dl_paused and not dry_run:
        print(f"🛑 Daily loss limit reached (-${DAILY_LOSS_LIMIT_CENTS/100:.2f})")
        structured_log("daily_loss", {"paused": True, "pnl": dl_pnl}, level="warning")
        return
    elif dl_paused:
        structured_log("daily_loss", {"paused": True, "pnl": dl_pnl, "paper_mode": True})
        print("   📝 Paper mode — ignoring daily loss limit, continuing for data collection")

    # ── LLM status ──
    use_heuristic = True
    if LLM_CONFIG:
        print(f"✅ LLM: {LLM_CONFIG['provider']} / {LLM_CONFIG['model']}")
        use_heuristic = False
    else:
        print("⚠️  No LLM API — using HEURISTIC forecaster")

    # ── Balance ──
    balance = get_balance()
    print(f"💰 Real balance: ${balance:.2f}")
    if dry_run:
        # In paper mode, use the paper state balance (tracks P&L from settlements)
        paper_st = load_paper_state()
        paper_balance = paper_st.get("current_balance_cents", PAPER_STARTING_BANKROLL_CENTS) / 100.0
        if paper_balance > 0:
            balance = paper_balance
        elif balance < 1.0:
            balance = VIRTUAL_BALANCE
        print(f"   📝 Paper balance: ${balance:.2f}")

    # ── Positions ──
    # In paper mode, use paper state positions (not real API positions which may
    # contain stale real-money positions like KXTRUMPFIRE)
    if dry_run:
        paper_state = load_paper_state()
        paper_positions = [p for p in paper_state.get("positions", []) if p.get("status") == "open"]
        # Convert paper positions to the format manage_positions expects
        positions = []
        for pp in paper_positions:
            contracts = pp.get("contracts", 1)
            is_no = pp.get("action") == "BUY_NO"
            positions.append({
                "ticker": pp.get("ticker", ""),
                "position": -contracts if is_no else contracts,
                "market_exposure": pp.get("cost_cents", 0),
                "total_traded": pp.get("cost_cents", 0),
            })
        num_positions = len(positions)
    else:
        positions = get_positions()
        num_positions = len(positions)
    dyn_max_pos = dynamic_max_positions(balance)
    print(f"📊 Open positions: {num_positions}/{dyn_max_pos} (dynamic, balance ${balance:.0f})")

    # ── Position Management: Trailing Stop / Early Exit (TRADE-017) ──
    if TRAILING_STOP_ENABLED and positions:
        print(f"\n🔔 Managing {num_positions} open positions...")
        exits = manage_positions(positions, dry_run)
        if exits > 0:
            print(f"   ✅ Exited {exits} position(s)")
            # Refresh positions after exits
            if dry_run:
                paper_state = load_paper_state()
                positions = [p for p in paper_state.get("positions", []) if p.get("status") == "open"]
                num_positions = len(positions)
            else:
                positions = get_positions()
                num_positions = len(positions)
            print(f"📊 Open positions after exits: {num_positions}/{dyn_max_pos}")

    if num_positions >= dyn_max_pos:
        print("⚠️ Max positions reached")
        return

    # ── GROK-TRADE-002: Drawdown & exposure alerts ──
    check_drawdown(balance)
    check_exposure_alert(positions, balance)

    # ── GROK-TRADE-002: Risk limit checks ──
    risk_reasons = check_risk_limits(positions, balance, dl_pnl.get("trades_today", 0))
    if risk_reasons and not dry_run:
        for reason in risk_reasons:
            print(f"🛡️ Risk limit: {reason}")
        print("⚠️ Risk limits exceeded — skipping cycle")
        structured_log("cycle_end", {"skipped": True, "reason": "risk_limits", "details": risk_reasons})
        return
    elif risk_reasons:
        for reason in risk_reasons:
            print(f"🛡️ Risk limit (paper mode, ignoring): {reason}")

    # ── Gather context (v2 signals) ──
    context = {}

    # Crypto prices
    prices = get_crypto_prices()
    if prices:
        context["crypto_prices"] = prices
        print(f"📈 BTC: ${prices['btc']:,.0f} | ETH: ${prices['eth']:,.0f}")
    else:
        print("⚠️ No crypto prices available")

    # Fear & Greed
    fng = get_fear_greed_index()
    context["sentiment"] = fng
    print(f"😱 Fear & Greed: {fng['value']} ({fng['classification']})")

    # News sentiment
    if NEWS_SEARCH_AVAILABLE:
        try:
            news = get_crypto_sentiment("both")
            context["news_sentiment"] = news
            icon = "🟢" if news["sentiment"] == "bullish" else ("🔴" if news["sentiment"] == "bearish" else "⚪")
            print(f"📰 News: {icon} {news['sentiment'].upper()} ({news['confidence']*100:.0f}%)")
        except Exception:
            context["news_sentiment"] = None
    else:
        context["news_sentiment"] = None

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
        print(f"📊 {asset} Momentum: {label} (dir={cd:+.2f}, str={cs:.2f}) {aligned}")

    for asset, regime in [("BTC", btc_regime), ("ETH", eth_regime)]:
        print(f"   {asset} Regime: {regime['regime']} ({regime['confidence']:.0%}), vol: {regime['volatility']}")

    # ── Scan markets ──
    markets = scan_all_markets()
    if not markets:
        print("❌ No tradeable markets found!")
        return

    # On holidays, pre-filter to crypto-only (they trade 24/7)
    if is_holiday_today:
        crypto_markets = [m for m in markets if any(x in m.ticker.upper() for x in ("KXBTC", "KXETH", "KXSOL", "CRYPTO"))]
        print(f"🎄 Holiday filter: {len(crypto_markets)} crypto markets out of {len(markets)} total")
        if crypto_markets:
            markets = crypto_markets
        # If no crypto markets available, keep all (holiday skip will filter later)

    # Score and rank
    scored = [(m, score_market(m)) for m in markets]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_markets = scored[:max_markets]

    print(f"\n🎯 TOP {len(top_markets)} MARKETS:")
    print("-" * 70)
    for i, (m, s) in enumerate(top_markets[:10], 1):
        print(f"  {i:2d}. [{s:.1f}] {m.ticker} — {m.title[:65]}")
        print(f"      YES:{m.yes_price}¢ Vol:{m.volume:,} DTE:{m.days_to_expiry:.1f}d")
    print("-" * 70)

    # ── Analyze markets: Forecast → Critique → Decide ──
    trades_executed = 0
    trades_skipped = 0
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
        # GROK-TRADE-002: check shutdown between market analyses
        if shutdown_requested:
            print(f"\n⚠️  Shutdown requested — stopping market analysis")
            structured_log("cycle_interrupted", {"reason": "shutdown", "markets_analyzed": i - 1})
            break

        if trades_executed >= max_trades:
            print(f"\n⏹️ Max trades ({max_trades}) reached")
            break

        if market.ticker in existing_tickers:
            continue

        print(f"\n{'='*50}")
        print(f"🔍 [{i}/{len(top_markets)}] {market.ticker}")
        print(f"   {market.title[:75]}")
        print(f"   YES:{market.yes_price}¢ NO:{market.no_price}¢ Vol:{market.volume:,}")

        # Build market-specific context
        mkt_context = dict(context)
        mkt_type = classify_market_type(market)

        # Skip non-crypto on holidays (crypto trades 24/7)
        if is_holiday_today and mkt_type != "crypto":
            print(f"   ⏭️ Holiday skip (non-crypto): {holiday_name}")
            trades_skipped += 1
            log_skip(market.ticker, f"holiday_{holiday_name}", {})
            continue

        if mkt_type == "crypto":
            asset = "btc" if "BTC" in market.ticker.upper() else "eth"
            mkt_context["momentum"] = context.get("momentum", {}).get(asset, {})
            mkt_context["regime"] = context.get("regime", {}).get(asset, {})

        # Asset-specific volatility filter — DISABLED per Grok rec C (2026-02-16)
        # Collecting data in all regimes to prove strategy. Will re-enable if choppy BTC drags performance.
        # Original filters: btc_high_vol_skip and choppy+high-vol skip
        if mkt_type == "crypto":
            asset_regime = mkt_context.get("regime", {})
            asset_name = "btc" if "BTC" in market.ticker.upper() else "eth"
            # Log regime for data collection but DON'T skip
            if asset_name == "btc" and asset_regime.get("btc_high_vol_skip", False):
                avg_range = asset_regime.get("avg_candle_range_pct", 0)
                print(f"   ⚠️ BTC high-vol regime (ATR {avg_range:.2%}) — trading anyway per Grok rec C")
            if asset_regime.get("regime") == "choppy" and asset_regime.get("volatility") in ("high", "very_high"):
                print(f"   ⚠️ Choppy+high-vol for {asset_name.upper()} — trading anyway per Grok rec C")

        # Step 1: FORECAST
        if use_heuristic:
            print(f"   🧮 Heuristic ({mkt_type}/{detect_sport(market)})...")
            forecast = heuristic_forecast(market, mkt_context)
        else:
            print(f"   🧠 LLM Forecasting...")
            forecast = forecast_market_llm(market, mkt_context)
        total_tokens += forecast.tokens_used
        print(f"   📊 Forecast: {forecast.probability:.1%} ({forecast.confidence})")
        print(f"   📝 Factors: {', '.join(forecast.key_factors[:3]) if forecast.key_factors else 'N/A'}")

        # Quick edge check before critic
        quick_edge = abs(forecast.probability - market.market_prob)
        if quick_edge < MIN_EDGE_BUY_NO * 0.5:
            print(f"   ⏭️ Quick skip: edge {quick_edge:.1%} too small")
            trades_skipped += 1
            log_skip(market.ticker, f"quick_edge_{quick_edge:.3f}", {"market_prob": market.market_prob})
            continue

        # Step 2: CRITIQUE
        if use_heuristic:
            critic = heuristic_critique(market, forecast)
        else:
            print(f"   🔎 LLM Critiquing...")
            critic = critique_forecast_llm(market, forecast)
        total_tokens += critic.tokens_used
        print(f"   📊 Critic: {critic.adjusted_probability:.1%} | Flaws: {len(critic.major_flaws)} | Trade: {'✅' if critic.should_trade else '❌'}")

        # Step 3: TRADE DECISION
        decision = make_trade_decision(market, forecast, critic, balance)
        print(f"   📋 DECISION: {decision.action} — {decision.reason}")

        if decision.action == "SKIP":
            trades_skipped += 1
            # GROK-TRADE-002: structured log for trade skip
            structured_log("trade_skipped", {
                "ticker": market.ticker, "reason": decision.reason,
                "edge": round(decision.edge, 4),
            })
            # GROK-TRADE-004: log vetoed/skipped decision with reason classification
            skip_outcome = "vetoed" if ("Critic vetoed" in decision.reason or "flaws" in decision.reason.lower()) else \
                           "skipped_parlay" if "Parlay" in decision.reason else \
                           "skipped_risk" if ("risk" in decision.reason.lower() or "cap" in decision.reason.lower() or "price" in decision.reason.lower()) else \
                           "skipped_edge"
            log_decision(market, decision, skip_outcome)
            log_trade(market, decision, {}, dry_run)
            continue

        # GROK-TRADE-004: Check daily absolute exposure cap
        daily_cost = get_daily_trades_cost()
        new_cost = decision.contracts * decision.price_cents
        if check_daily_exposure_cap(daily_cost + new_cost) and not dry_run:
            print(f"   ⛔ Daily exposure cap reached (${(daily_cost + new_cost)/100:.2f} >= ${MAX_DAILY_EXPOSURE_USD})")
            log_decision(market, decision, "skipped_risk")
            continue

        # Step 4: EXECUTE
        side = "yes" if decision.action == "BUY_YES" else "no"
        cost = decision.contracts * decision.price_cents
        print(f"   💰 {decision.action} × {decision.contracts} @ {decision.price_cents}¢ = ${cost/100:.2f}")

        order_result = place_order(market.ticker, side, decision.price_cents, decision.contracts, dry_run)

        if dry_run:
            print(f"   🧪 DRY RUN: Simulated")
            # Track in paper portfolio
            try:
                paper_state = load_paper_state()
                paper_trade_open(paper_state, market.ticker, decision.action,
                                 decision.price_cents, decision.contracts,
                                 title=market.title, edge=decision.edge,
                                 expiry=market.expiry or "")
                print(f"   📊 Paper bankroll: ${paper_state['current_balance_cents']/100:.2f}")
            except Exception as e:
                print(f"   ⚠️ Paper state update error: {e}")
        else:
            if "error" in order_result:
                print(f"   ❌ Order failed: {order_result['error']}")
            else:
                print(f"   ✅ Placed! ID: {order_result.get('order', {}).get('order_id', 'N/A')}")

        trades_executed += 1
        existing_tickers.add(market.ticker)
        log_trade(market, decision, order_result, dry_run)
        # GROK-TRADE-004: log executed decision
        log_decision(market, decision, "executed")
        # GROK-TRADE-002: structured log for trade execution
        structured_log("trade_executed", {
            "ticker": market.ticker, "action": decision.action,
            "contracts": decision.contracts, "price_cents": decision.price_cents,
            "edge": round(decision.edge, 4), "cost_cents": cost,
            "dry_run": dry_run,
        })
        time.sleep(1)

    # ── Weather opportunities ──
    if WEATHER_ENABLED:
        weather_opps = find_weather_opportunities()
        if weather_opps:
            print(f"\n🌡️ Weather: {len(weather_opps)} opportunities found")
            for wo in weather_opps[:3]:
                if trades_executed >= max_trades:
                    break
                wm = wo["market"]
                if wm.ticker in existing_tickers:
                    continue
                print(f"   🌡️ {wm.ticker}: {wo['side'].upper()} @{wo['price']}¢ edge:{wo['edge']*100:.1f}% ({wo['city']})")
                # Simple forecast/critic for weather
                wf = ForecastResult(probability=wo["our_prob"], reasoning="NWS forecast",
                                    confidence="medium", key_factors=["NWS"], model_used="nws-weather")
                wc = CriticResult(adjusted_probability=wo["our_prob"], should_trade=True)
                wd = make_trade_decision(wm, wf, wc, balance)
                if wd.action != "SKIP":
                    order = place_order(wm.ticker, wo["side"], wo["price"], wd.contracts, dry_run)
                    log_trade(wm, wd, order, dry_run)
                    if dry_run:
                        try:
                            ps = load_paper_state()
                            paper_trade_open(ps, wm.ticker, wd.action, wo["price"], wd.contracts,
                                             title=wm.title, edge=wo["edge"], expiry=wm.expiry or "")
                        except Exception:
                            pass
                    trades_executed += 1
                    existing_tickers.add(wm.ticker)

    # ── Cycle summary ──
    duration = time.time() - cycle_start
    print(f"\n{'='*70}")
    print(f"📊 CYCLE SUMMARY")
    print(f"   Forecaster: {'🧮 HEURISTIC' if use_heuristic else '🧠 LLM'}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Markets scanned: {len(markets)}")
    print(f"   Markets analyzed: {min(len(top_markets), max_markets)}")
    print(f"   Trades executed: {trades_executed}")
    print(f"   Trades skipped: {trades_skipped}")
    if not use_heuristic:
        print(f"   Tokens: {total_tokens:,} (~${total_tokens * 0.000004:.4f})")

    # Paper portfolio summary
    if dry_run:
        try:
            ps = load_paper_state()
            ps_bal = ps["current_balance_cents"] / 100
            ps_pnl = ps["stats"].get("pnl_cents", 0) / 100
            ps_wr = ps["stats"].get("win_rate", 0) * 100
            ps_w = ps["stats"].get("wins", 0)
            ps_l = ps["stats"].get("losses", 0)
            ps_open = len(ps.get("positions", []))
            print(f"   📊 Paper: ${ps_bal:.2f} (PnL: ${ps_pnl:+.2f}, WR: {ps_wr:.1f}% [{ps_w}W/{ps_l}L], {ps_open} open)")
        except Exception:
            pass

    # Latency summary
    avg_lat = get_avg_latency("markets_search")
    if avg_lat > 0:
        print(f"   Avg API latency: {avg_lat:.0f}ms")
    print(f"{'='*70}")

    cycle_stats = {
        "dry_run": dry_run,
        "forecaster": "heuristic" if use_heuristic else "llm",
        "duration_s": round(duration, 1),
        "markets_scanned": len(markets),
        "markets_analyzed": min(len(top_markets), max_markets),
        "trades_executed": trades_executed,
        "trades_skipped": trades_skipped,
        "tokens": total_tokens,
        "balance": balance,
        "positions": num_positions,
    }
    log_cycle(cycle_stats)
    # GROK-TRADE-002: structured log for cycle end
    structured_log("cycle_end", cycle_stats)
    # GROK-TRADE-004: post-cycle monitoring checks
    check_high_edge_cluster()


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
        """)

    parser.add_argument("--live", action="store_true", help="Enable LIVE trading (default: paper)")
    parser.add_argument("--markets", type=int, default=20, help="Max markets to analyze (default: 20)")
    parser.add_argument("--max-trades", type=int, default=5, help="Max trades per cycle (default: 5)")
    parser.add_argument("--loop", type=int, default=0, help="Loop interval in seconds (0 = single run)")
    parser.add_argument("--min-edge", type=float, default=None, help="Override minimum edge")
    parser.add_argument("--kelly", type=float, default=None, help="Override Kelly fraction")

    args = parser.parse_args()

    # Override globals
    global MIN_EDGE, MIN_EDGE_BUY_YES, MIN_EDGE_BUY_NO, KELLY_FRACTION, DRY_RUN
    if args.min_edge is not None:
        MIN_EDGE = args.min_edge
        MIN_EDGE_BUY_YES = max(args.min_edge, MIN_EDGE_BUY_YES)
        MIN_EDGE_BUY_NO = args.min_edge
    if args.kelly is not None:
        KELLY_FRACTION = args.kelly
    if args.live:
        DRY_RUN = False

    dry_run = not args.live

    if args.live:
        print("⚠️  LIVE TRADING MODE! Press Ctrl+C within 5s to abort...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    if args.loop > 0:
        print(f"🔄 Loop mode: every {args.loop}s")
        while not shutdown_requested:
            try:
                run_cycle(dry_run=dry_run, max_markets=args.markets, max_trades=args.max_trades)
                # GROK-TRADE-002: check shutdown during sleep
                if shutdown_requested:
                    break
                print(f"\n⏰ Next cycle in {args.loop}s...")
                # Sleep in small increments to respond to shutdown quickly
                for _ in range(args.loop):
                    if shutdown_requested:
                        break
                    time.sleep(1)
            except KeyboardInterrupt:
                # GROK-TRADE-002: KeyboardInterrupt also triggers graceful shutdown
                break
            except Exception as e:
                print(f"\n❌ Cycle error: {e}")
                structured_log("api_error", {"context": "cycle_loop", "error": str(e)}, level="error")
                traceback.print_exc()
                time.sleep(30)
        # GROK-TRADE-002: graceful shutdown — log final state
        structured_log("shutdown_complete", {
            "dry_run": dry_run,
            "reason": "signal" if shutdown_requested else "user_interrupt",
        })
        print("\n✅ Graceful shutdown complete")
    else:
        run_cycle(dry_run=dry_run, max_markets=args.markets, max_trades=args.max_trades)
        # GROK-TRADE-002: log shutdown for single-run mode too
        if shutdown_requested:
            structured_log("shutdown_complete", {"dry_run": dry_run, "reason": "signal"})
            print("\n✅ Graceful shutdown complete")


if __name__ == "__main__":
    main()
