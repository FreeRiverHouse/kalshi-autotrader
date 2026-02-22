#!/usr/bin/env python3
"""
Kalshi AutoTrader v2 - Fixed Algorithm with Proper Probability Model

FIXES:
1. Proper probability calculation using log-normal price model
2. Correct time-to-expiry handling (hourly contracts!)
3. Feedback loop - updates trade results automatically
4. Better edge calculation with realistic volatility
5. Trend detection to avoid betting against momentum

Author: Clawd (Fixed after 0% win rate disaster)
"""

import os
import requests
import json
import sys
import time
import math
import threading
from datetime import datetime, timezone, timedelta
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from pathlib import Path
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler

# Import news sentiment analysis (T661 - Grok Fundamental strategy)
try:
    # Add scripts dir to path for local import
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("crypto_news_search", 
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto-news-search.py"))
    crypto_news_module = module_from_spec(spec)
    spec.loader.exec_module(crypto_news_module)
    get_crypto_sentiment = crypto_news_module.get_crypto_sentiment
    NEWS_SEARCH_AVAILABLE = True
except Exception as e:
    NEWS_SEARCH_AVAILABLE = False
    print(f"⚠️ News search module not available: {e}")
    def get_crypto_sentiment(asset="both"):
        return {"sentiment": "neutral", "confidence": 0.5, "edge_adjustment": 0, "should_trade": True, "reasons": []}

# Import NWS weather forecast module (T422 - Weather markets based on PredictionArena research)
try:
    weather_spec = spec_from_file_location("nws_weather_forecast",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "nws-weather-forecast.py"))
    weather_module = module_from_spec(weather_spec)
    weather_spec.loader.exec_module(weather_module)
    parse_kalshi_weather_ticker = weather_module.parse_kalshi_weather_ticker
    calculate_weather_edge = weather_module.calculate_weather_edge
    fetch_forecast = weather_module.fetch_forecast
    NWS_POINTS = weather_module.NWS_POINTS
    WEATHER_AVAILABLE = True
except Exception as e:
    WEATHER_AVAILABLE = False
    print(f"⚠️ Weather forecast module not available: {e}")

# Import market holiday checker (T414 - Auto-pause during market holidays)
try:
    holiday_spec = spec_from_file_location("check_market_holiday",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "check-market-holiday.py"))
    holiday_module = module_from_spec(holiday_spec)
    holiday_spec.loader.exec_module(holiday_module)
    is_market_holiday = holiday_module.is_holiday
    HOLIDAY_CHECK_AVAILABLE = True
except Exception as e:
    HOLIDAY_CHECK_AVAILABLE = False
    print(f"⚠️ Holiday check module not available: {e}")
    def is_market_holiday(check_date=None):
        return False, None

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

# Trading parameters - MORE CONSERVATIVE
MIN_EDGE = 0.04  # 4% minimum edge (was 10% — too high, generated 0 crypto trades)
MAX_EDGE = 0.20  # 20% max edge — edges above this are likely model errors, not real alpha
                  # Reduced from 25%: backtest (TRADE-003) found weather edges >25% were false positives
                  # and crypto edges >20% are also suspicious (model overconfidence near strikes)
MAX_POSITION_PCT = 0.03  # 3% max per position (default, see per-asset below)
KELLY_FRACTION = 0.05  # Very conservative Kelly (default, see per-asset below)
MIN_BET_CENTS = 5
MIN_TIME_TO_EXPIRY_MINUTES = 45  # Increased from 30
MAX_POSITIONS = 30

# BUGFIX 2026-07-17: Strike distance and probability sanity filters
# These would have prevented ALL 41 losing BTC trades (0% WR disaster).
MIN_STRIKE_DISTANCE_PCT = 0.03  # Don't trade strikes <3% from current price
MAX_PROB_DISAGREEMENT = 0.25    # Skip if model differs >25% from market probability
MIN_NO_PRICE_CENTS = 8          # Don't buy NO contracts at <8 cents (market is usually right)

# Per-asset position sizing (T441)
# Weather markets: higher Kelly (NWS forecasts are reliable), lower max position (less liquid)
# BTC: standard sizing (most liquid, most traded)
# ETH: slightly lower (more volatile than BTC)
# KEY INSIGHT: Different assets have different edge sources and liquidity profiles
ASSET_CONFIG = {
    "btc": {
        "kelly_fraction": 0.10,     # Higher Kelly - crypto is our main focus
        "max_position_pct": 0.05,   # 5% max per position
        "min_edge": 0.04,           # 4% min edge — realistic for crypto (was 10%, too high, produced 0 trades)
        "max_edge": 0.20,           # TRADE-005: Cap at 20% — edges above this are model overconfidence
    },
    "eth": {
        "kelly_fraction": 0.08,     # Slightly lower than BTC (more volatile)
        "max_position_pct": 0.04,   # 4% max per position
        "min_edge": 0.05,           # 5% min edge (was 12%, too high)
        "max_edge": 0.20,           # TRADE-005: Cap at 20%
    },
    "sol": {  # T423 - Solana support
        "kelly_fraction": 0.05,     # Conservative Kelly (high volatility, less predictable)
        "max_position_pct": 0.03,   # 3% max per position (newer market, less liquid)
        "min_edge": 0.06,           # 6% min edge (was 15%, too high)
        "max_edge": 0.20,           # TRADE-005: Cap at 20%
    },
    "weather": {
        # UPDATED 2026-02-08: Backtest showed 17.9% WR with old params, 75%+ with new
        # Key insight: NWS forecasts have 2.8°F MAE, must require buffer from strike
        "kelly_fraction": 0.05,     # Reduced from 0.08 (more conservative after analysis)
        "max_position_pct": 0.02,   # 2% max (weather markets less liquid)
        "min_edge": 0.15,           # Increased from 0.10 (require stronger edge)
        "max_edge": 0.15,           # TRADE-005: Strictest cap — backtest showed >25% edges were false positives
    },
    # Default for unknown assets
    "default": {
        "kelly_fraction": 0.03,     # Very conservative for unknown assets
        "max_position_pct": 0.02,   # 2% max
        "min_edge": 0.15,           # 15% min edge (be extra careful)
        "max_edge": 0.15,           # TRADE-005: Strict cap for unknown assets
    }
}

def get_asset_config(asset_type):
    """Get per-asset configuration with fallback to default."""
    asset_key = asset_type.lower() if asset_type else "default"
    return ASSET_CONFIG.get(asset_key, ASSET_CONFIG["default"])

# Volatility assumptions (DEFAULTS - overridden by dynamic calculation when OHLC available)
# BUGFIX 2026-07-17: Reduced from 0.005/0.007 — old values produced ~40% prob for 0.6% moves
# in 55min, generating massive fake edges. Empirical BTC hourly vol is ~0.25-0.35%.
BTC_HOURLY_VOL = 0.003  # ~0.3% hourly volatility (empirical, validated against Kalshi outcomes)
ETH_HOURLY_VOL = 0.004  # ~0.4% hourly volatility (empirical fallback)
SOL_HOURLY_VOL = 0.008  # ~0.8% hourly volatility (T423 - Solana is more volatile)

# Fat-tail adjustment: DISABLED (set to 1.0)
# BUGFIX 2026-07-17: The 1.4x multiplier was the #1 cause of the 0% win rate disaster.
# It inflated sigma so much that the model assigned ~48% probability to BTC dropping
# 0.6% in under 1 hour. Combined with buying NO at 1-2 cents, every single trade lost.
# The lognormal model WITHOUT fat-tail adjustment already slightly overestimates
# tail probabilities for crypto. Fat tails matter for multi-day horizons, not hourly.
CRYPTO_FAT_TAIL_MULTIPLIER = float(os.getenv("FAT_TAIL_MULT", "1.0"))  # 1.0 = disabled (was 1.4)

# Momentum config
MOMENTUM_TIMEFRAMES = ["1h", "4h", "24h"]
MOMENTUM_WEIGHT = {"1h": 0.5, "4h": 0.3, "24h": 0.2}  # Short-term matters more for hourly contracts

# Weather market config (T422 - Based on PredictionArena research)
# Key insight: NWS forecasts are accurate within ±2-3°F for <48h predictions
# Edge source: favorite-longshot bias + forecast accuracy
WEATHER_ENABLED = os.getenv("WEATHER_ENABLED", "false").lower() in ("true", "1", "yes")  # Disabled by default: 20% WR, model is broken (see docs/autotrader-improvement-plan.md)
WEATHER_CITIES = ["NYC", "CHI", "DEN"]  # Removed MIA (0% WR), focus on higher-performing cities
WEATHER_MAX_HOURS_TO_SETTLEMENT = 48  # Only trade within 48h of settlement (highest forecast accuracy)

# ============== WEATHER SAFETY GUARDS (2026-02-08 Backtest-Validated) ==============
# These filters improved win rate from 17.9% to 75%+ in backtesting
# Key finding: 0°F gap = 6.4% WR, 2°F gap = 40% WR, 7°F+ gap = 100% WR
WEATHER_MIN_FORECAST_STRIKE_GAP = float(os.getenv("WEATHER_MIN_GAP", "2.0"))  # Minimum °F between forecast and strike
WEATHER_MAX_MARKET_CONVICTION = float(os.getenv("WEATHER_MAX_CONVICTION", "0.85"))  # Don't bet against 85%+ conviction
WEATHER_MIN_OUR_PROB = float(os.getenv("WEATHER_MIN_PROB", "0.05"))  # Reject trades where our_prob < 5%
WEATHER_FORECAST_UNCERTAINTY_OVERRIDE = float(os.getenv("WEATHER_UNCERTAINTY", "4.0"))  # Override default 2.0 with 4.0 based on MAE

# Legacy constants for backwards compatibility - now use ASSET_CONFIG["weather"] (T441)
WEATHER_MIN_EDGE = ASSET_CONFIG["weather"]["min_edge"]
WEATHER_KELLY_FRACTION = ASSET_CONFIG["weather"]["kelly_fraction"]

# Logging
TRADE_LOG_FILE = "scripts/kalshi-trades-v2.jsonl"
SKIP_LOG_FILE = "scripts/kalshi-skips.jsonl"
EXECUTION_LOG_FILE = "scripts/kalshi-execution-log.jsonl"

# Dry run mode - log trades without executing
DRY_RUN = os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes")
DRY_RUN_LOG_FILE = "scripts/kalshi-trades-dryrun.jsonl"

# Circuit breaker config (consecutive losses)
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))  # Auto-pause after N consecutive losses
CIRCUIT_BREAKER_COOLDOWN_HOURS = float(os.getenv("CIRCUIT_BREAKER_COOLDOWN_HOURS", "4"))  # Require cooldown period after trigger
CIRCUIT_BREAKER_ALERT_FILE = "scripts/kalshi-circuit-breaker.alert"
CIRCUIT_BREAKER_STATE_FILE = "scripts/kalshi-circuit-breaker.json"
CIRCUIT_BREAKER_HISTORY_FILE = "scripts/kalshi-circuit-breaker-history.jsonl"  # T471: History logging

# Regime change alerting
REGIME_STATE_FILE = "scripts/kalshi-regime-state.json"
REGIME_ALERT_FILE = "scripts/kalshi-regime-change.alert"
REGIME_ALERT_COOLDOWN = 3600  # 1 hour cooldown between alerts

# Momentum direction change alerting
MOMENTUM_STATE_FILE = "scripts/kalshi-momentum-state.json"
MOMENTUM_ALERT_FILE = "scripts/kalshi-momentum-change.alert"
MOMENTUM_ALERT_COOLDOWN = 1800  # 30 min cooldown (more frequent than regime)

# Whipsaw detection (T393) - momentum flip twice in 24h
WHIPSAW_ALERT_FILE = "scripts/kalshi-whipsaw.alert"
WHIPSAW_ALERT_COOLDOWN = 7200  # 2 hour cooldown (rare event)
WHIPSAW_WINDOW_HOURS = 24  # Look for 2 flips within this window

# Latency alerting
LATENCY_ALERT_FILE = "scripts/kalshi-latency.alert"

# Health status endpoint (T472)
HEALTH_STATUS_FILE = "data/trading/autotrader-health.json"

# HTTP Health Server (T828)
HEALTH_SERVER_PORT = int(os.environ.get("HEALTH_SERVER_PORT", 8089))
HEALTH_SERVER_ENABLED = os.environ.get("HEALTH_SERVER_ENABLED", "true").lower() == "true"

# ML Feature Logging (T331) - structured data for ML model training
ML_FEATURE_LOG_FILE = "data/trading/ml-training-data.jsonl"

# Extreme volatility alerting (T294)
EXTREME_VOL_ALERT_FILE = "scripts/kalshi-extreme-vol.alert"
EXTREME_VOL_ALERT_COOLDOWN = 3600  # 1 hour cooldown

# Full momentum alignment alerting (T301)
MOMENTUM_ALIGN_ALERT_FILE = "scripts/kalshi-momentum-aligned.alert"
MOMENTUM_ALIGN_ALERT_COOLDOWN = 7200  # 2 hour cooldown (rare event)
MOMENTUM_ALIGN_MIN_STRENGTH = 0.5  # Minimum composite strength to alert

# Momentum reversion detection (T302) - extended moves often precede reversals
REVERSION_ALERT_FILE = "scripts/kalshi-momentum-reversion.alert"
REVERSION_ALERT_COOLDOWN = 3600  # 1 hour cooldown
REVERSION_4H_THRESHOLD = 0.02  # 2% move in 4h triggers reversion watch
REVERSION_8H_THRESHOLD = 0.03  # 3% move in 8h triggers high confidence reversion
REVERSION_STRENGTH_THRESHOLD = 0.7  # Momentum strength for reversion signal

# Momentum divergence detection (T303) - price vs momentum disagreement
DIVERGENCE_ALERT_FILE = "scripts/kalshi-momentum-divergence.alert"
DIVERGENCE_ALERT_COOLDOWN = 3600  # 1 hour cooldown
DIVERGENCE_LOOKBACK = 8  # Number of candles to analyze for divergence
DIVERGENCE_MIN_PRICE_MOVE = 0.008  # 0.8% minimum price move to detect
DIVERGENCE_STATE_FILE = "scripts/kalshi-divergence-state.json"

# Portfolio concentration limits (T480) - prevent over-concentration in correlated assets
# Key insight: Don't put all eggs in one basket - diversify across asset classes
CONCENTRATION_MAX_ASSET_CLASS_PCT = 0.50  # Max 50% of portfolio in any single asset class
CONCENTRATION_MAX_CORRELATED_PCT = 0.30   # Default max 30% in highly correlated positions (dynamic via T483)
CONCENTRATION_WARN_PCT = 0.40             # Warn when approaching 40% in any asset class
CONCENTRATION_ALERT_FILE = "scripts/kalshi-concentration.alert"
CONCENTRATION_ALERT_COOLDOWN = 3600       # 1 hour cooldown between alerts
CORRELATION_DATA_FILE = "data/trading/asset-correlation.json"  # Dynamic BTC/ETH correlation (T483)

# Auto-rebalancing (T816) - automatically reduce concentration when it exceeds threshold
AUTO_REBALANCE_ENABLED = os.getenv("AUTO_REBALANCE_ENABLED", "false").lower() == "true"
AUTO_REBALANCE_THRESHOLD = float(os.getenv("AUTO_REBALANCE_THRESHOLD", "0.45"))  # Auto-rebalance when >45%
AUTO_REBALANCE_DRY_RUN = os.getenv("AUTO_REBALANCE_DRY_RUN", "true").lower() == "true"  # Preview without executing
AUTO_REBALANCE_LOG_FILE = "scripts/kalshi-rebalance.log"
AUTO_REBALANCE_ALERT_FILE = "scripts/kalshi-rebalance.alert"

# VIX integration for regime detection (T611) - use fear index to adjust sizing/edge
# VIX >25: high fear = more conservative (higher edge required)
# VIX <15: low fear = normal sizing
# Moderate positive correlation (0.40) with crypto means VIX spikes may precede vol spikes
VIX_CORRELATION_FILE = "data/trading/vix-correlation.json"
VIX_HIGH_FEAR_THRESHOLD = 25.0  # VIX >= 25 = high fear, reduce position size
VIX_ELEVATED_THRESHOLD = 20.0  # VIX 20-25 = elevated, increase min edge
VIX_LOW_FEAR_THRESHOLD = 15.0  # VIX < 15 = low fear, normal operation
VIX_CACHE_MAX_AGE_HOURS = 24  # Only use VIX data if less than 24h old

LATENCY_THRESHOLD_MS = int(os.getenv("LATENCY_THRESHOLD_MS", "2000"))  # Alert if avg latency > 2s
LATENCY_ALERT_COOLDOWN = 3600  # 1 hour cooldown
LATENCY_CHECK_WINDOW = 10  # Check last N trades

# Streak record alerting (T288)
STREAK_STATE_FILE = "scripts/kalshi-streak-records.json"

# Market holiday pausing (T414) - skip trading on US market holidays
# Crypto markets have unusual patterns during holidays (lower liquidity, erratic moves)
HOLIDAY_PAUSE_ENABLED = os.getenv("HOLIDAY_PAUSE_ENABLED", "true").lower() == "true"
HOLIDAY_REDUCE_SIZE = os.getenv("HOLIDAY_REDUCE_SIZE", "true").lower() == "true"  # Alternative: reduce size instead of pausing
HOLIDAY_SIZE_REDUCTION = 0.5  # 50% position size reduction if trading on holidays
STREAK_ALERT_FILE = "scripts/kalshi-streak-record.alert"

# Streak position analysis (T770) - context for trade decisions
STREAK_POSITION_ANALYSIS_FILE = "data/trading/streak-position-analysis.json"
STREAK_TILT_THRESHOLD = 3  # Warn when entering trade after N consecutive losses
STREAK_HOT_HAND_THRESHOLD = 3  # Note when entering trade after N consecutive wins
# Streak-based position sizing (T388) - reduce size to prevent tilt/hot-hand-fallacy
STREAK_SIZE_REDUCTION = float(os.environ.get("STREAK_SIZE_REDUCTION", "0.7"))  # 70% of normal size

# Tilt risk alerting (T774) - alert Mattia when trading in risky state
TILT_RISK_ALERT_FILE = "scripts/kalshi-tilt-risk.alert"

# API Latency Profiling (T279)
LATENCY_PROFILE_FILE = "scripts/kalshi-latency-profile.json"
LATENCY_PROFILE_WINDOW = 100  # Keep last N calls per endpoint
API_LATENCY_LOG = defaultdict(list)  # endpoint -> list of (timestamp, latency_ms)

# Latency-based position sizing (T801) - reduce position when API is slow
LATENCY_POSITION_SIZING_ENABLED = os.getenv("LATENCY_POSITION_SIZING", "true").lower() in ("true", "1", "yes")
LATENCY_SIZE_THRESHOLDS = {
    500: 0.75,    # >500ms avg: reduce Kelly by 25%
    1000: 0.50,   # >1000ms avg: reduce Kelly by 50%
    2000: 0.0,    # >2000ms avg: skip trade entirely (return 0)
}
LATENCY_CRITICAL_ENDPOINTS = ["order", "markets_search"]  # Endpoints that matter for trade execution

# Trading Window Schedule (T789) - skip bad hours/days based on historical performance
TRADING_SCHEDULE_FILE = "data/trading/trading-recommendations.json"
TRADING_SCHEDULE_ENABLED = os.getenv("TRADING_SCHEDULE_ENABLED", "true").lower() in ("true", "1", "yes")
TRADING_SCHEDULE_SKIP_LOG = "scripts/kalshi-schedule-skips.jsonl"

# ============== TRADING WINDOW SCHEDULE (T789) ==============

def load_trading_schedule():
    """
    Load trading window recommendations from JSON file.
    Returns schedule dict or None if not available.
    """
    schedule_path = Path(__file__).parent.parent / TRADING_SCHEDULE_FILE
    if not schedule_path.exists():
        return None
    
    try:
        with open(schedule_path) as f:
            data = json.load(f)
        return data.get("recommendations", {}).get("schedule")
    except Exception as e:
        print(f"⚠️ Failed to load trading schedule: {e}")
        return None

def check_trading_schedule():
    """
    Check if current time is within optimal trading windows.
    Also checks for US market holidays (T414).
    
    Returns:
        tuple: (should_trade, reason)
        - should_trade: True if we should trade now, False otherwise
        - reason: Explanation of why trading is allowed/blocked
    """
    from datetime import date
    
    # Check for market holidays first (T414)
    if HOLIDAY_PAUSE_ENABLED and HOLIDAY_CHECK_AVAILABLE:
        is_hol, holiday_name = is_market_holiday()
        if is_hol:
            return False, f"🎄 {holiday_name} - paused (lower liquidity, unusual patterns)"
    
    if not TRADING_SCHEDULE_ENABLED:
        # Still report if it's a holiday even when schedule is disabled
        if HOLIDAY_CHECK_AVAILABLE:
            is_hol, holiday_name = is_market_holiday()
            if is_hol:
                if HOLIDAY_REDUCE_SIZE:
                    return True, f"⚠️ {holiday_name} - trading with reduced size"
                else:
                    return True, f"⚠️ {holiday_name} - schedule check disabled"
        return True, "Schedule check disabled"
    
    schedule = load_trading_schedule()
    if not schedule:
        return True, "No schedule data available"
    
    now = datetime.now(timezone.utc)
    current_hour = now.hour
    current_day = now.weekday()
    
    active_hours = schedule.get("active_hours", list(range(24)))
    avoid_days = schedule.get("avoid_days", [])
    
    # Check if current day should be avoided
    if current_day in avoid_days:
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_name = day_names[current_day]
        return False, f"{day_name} is in avoid_days (historical poor performance)"
    
    # Check if current hour is active
    if current_hour not in active_hours:
        return False, f"{current_hour:02d}:00 UTC not in active_hours (poor historical win rate)"
    
    return True, f"✓ {current_hour:02d}:00 UTC is an optimal trading window"


def is_holiday_trading():
    """Check if we're trading during a market holiday (T414)"""
    if HOLIDAY_CHECK_AVAILABLE:
        is_hol, _ = is_market_holiday()
        return is_hol
    return False

def log_schedule_skip(reason: str):
    """Log when we skip trading due to schedule."""
    log_path = Path(__file__).parent / Path(TRADING_SCHEDULE_SKIP_LOG).name
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "hour": datetime.now(timezone.utc).hour,
        "day": datetime.now(timezone.utc).weekday()
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============== API LATENCY PROFILING (T279) ==============

def record_api_latency(endpoint: str, latency_ms: float):
    """
    Record API call latency for profiling.
    
    Args:
        endpoint: API endpoint name (e.g., "balance", "positions", "order")
        latency_ms: Time taken for the call in milliseconds
    """
    global API_LATENCY_LOG
    timestamp = datetime.now(timezone.utc).isoformat()
    API_LATENCY_LOG[endpoint].append((timestamp, latency_ms))
    
    # Keep only last N entries per endpoint
    if len(API_LATENCY_LOG[endpoint]) > LATENCY_PROFILE_WINDOW:
        API_LATENCY_LOG[endpoint] = API_LATENCY_LOG[endpoint][-LATENCY_PROFILE_WINDOW:]


def calculate_latency_stats(latencies: list) -> dict:
    """
    Calculate latency statistics from a list of latency values.
    
    Args:
        latencies: List of latency values in ms
    
    Returns:
        Dict with min, avg, p50, p95, p99, max, count
    """
    if not latencies:
        return {"count": 0}
    
    sorted_latencies = sorted(latencies)
    count = len(sorted_latencies)
    
    return {
        "count": count,
        "min_ms": round(sorted_latencies[0], 1),
        "avg_ms": round(sum(sorted_latencies) / count, 1),
        "p50_ms": round(sorted_latencies[int(count * 0.5)], 1),
        "p95_ms": round(sorted_latencies[min(int(count * 0.95), count - 1)], 1),
        "p99_ms": round(sorted_latencies[min(int(count * 0.99), count - 1)], 1),
        "max_ms": round(sorted_latencies[-1], 1)
    }


def get_latency_profile() -> dict:
    """
    Get latency profile for all tracked endpoints.
    
    Returns:
        Dict with endpoint -> stats mapping
    """
    profile = {}
    for endpoint, entries in API_LATENCY_LOG.items():
        latencies = [lat for _, lat in entries]
        profile[endpoint] = calculate_latency_stats(latencies)
    return profile


def print_latency_summary():
    """Print formatted latency profiling summary to console."""
    profile = get_latency_profile()
    if not profile:
        return
    
    print("\n📊 API LATENCY PROFILE:")
    print("=" * 70)
    print(f"{'Endpoint':<25} {'Calls':>6} {'Min':>8} {'Avg':>8} {'P95':>8} {'Max':>8}")
    print("-" * 70)
    
    # Sort by avg latency descending (slowest first)
    sorted_endpoints = sorted(profile.items(), key=lambda x: x[1].get("avg_ms", 0), reverse=True)
    
    for endpoint, stats in sorted_endpoints:
        if stats["count"] > 0:
            print(f"{endpoint:<25} {stats['count']:>6} {stats['min_ms']:>7.1f}ms {stats['avg_ms']:>7.1f}ms "
                  f"{stats['p95_ms']:>7.1f}ms {stats['max_ms']:>7.1f}ms")
    
    print("=" * 70)
    
    # Calculate totals
    total_calls = sum(s.get("count", 0) for s in profile.values())
    all_latencies = []
    for entries in API_LATENCY_LOG.values():
        all_latencies.extend([lat for _, lat in entries])
    
    if all_latencies:
        total_stats = calculate_latency_stats(all_latencies)
        print(f"{'TOTAL':<25} {total_calls:>6} {total_stats['min_ms']:>7.1f}ms {total_stats['avg_ms']:>7.1f}ms "
              f"{total_stats['p95_ms']:>7.1f}ms {total_stats['max_ms']:>7.1f}ms")


def save_latency_profile():
    """Save latency profile to file for later analysis."""
    profile = get_latency_profile()
    if not profile:
        return
    
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints": profile,
        "raw_data": {endpoint: entries[-20:] for endpoint, entries in API_LATENCY_LOG.items()}  # Last 20 per endpoint
    }
    
    with open(LATENCY_PROFILE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def identify_bottlenecks() -> list:
    """
    Identify API endpoints that may be bottlenecks.
    
    Returns:
        List of (endpoint, issue, details) tuples
    """
    profile = get_latency_profile()
    bottlenecks = []
    
    for endpoint, stats in profile.items():
        if stats["count"] < 3:
            continue  # Not enough data
        
        # Flag slow endpoints
        if stats["avg_ms"] > 1000:
            bottlenecks.append((endpoint, "slow_avg", f"Avg {stats['avg_ms']:.0f}ms > 1000ms threshold"))
        
        if stats["p95_ms"] > 2000:
            bottlenecks.append((endpoint, "slow_p95", f"P95 {stats['p95_ms']:.0f}ms > 2000ms threshold"))
        
        # Flag high variance (p95 >> avg)
        if stats["avg_ms"] > 0 and stats["p95_ms"] / stats["avg_ms"] > 3:
            bottlenecks.append((endpoint, "high_variance", f"P95/Avg ratio {stats['p95_ms']/stats['avg_ms']:.1f}x"))
    
    return bottlenecks


# ============== ENDPOINT-SPECIFIC TIMEOUTS (T805) ==============

# Configurable timeout per endpoint type
# Critical endpoints (orders) get shorter timeout to fail fast
# Read-only endpoints can tolerate slightly longer waits
ENDPOINT_TIMEOUTS = {
    "order": 8,           # Order placement - fail fast to avoid partial fills
    "balance": 10,        # Balance check - medium priority
    "positions": 10,      # Position check - medium priority  
    "markets_search": 15, # Market search - can be slower
    "fills": 12,          # Fill history - medium priority
    "default": 10         # Fallback for unknown endpoints
}

# Timeout alert threshold - create alert when this many timeouts happen in TIMEOUT_ALERT_WINDOW
TIMEOUT_ALERT_THRESHOLD = 3
TIMEOUT_ALERT_WINDOW_SECONDS = 300  # 5 minutes
TIMEOUT_ALERT_COOLDOWN = 3600  # 1 hour between alerts

# Track recent timeouts for alerting
RECENT_TIMEOUTS = []  # List of (timestamp, endpoint) tuples
LAST_TIMEOUT_ALERT = None


def get_endpoint_timeout(endpoint_name: str) -> int:
    """Get configured timeout for an endpoint type."""
    return ENDPOINT_TIMEOUTS.get(endpoint_name, ENDPOINT_TIMEOUTS["default"])


def record_timeout(endpoint: str):
    """
    Record a timeout event and create alert if threshold exceeded.
    
    Creates kalshi-timeout.alert if multiple timeouts happen in short window.
    """
    global RECENT_TIMEOUTS, LAST_TIMEOUT_ALERT
    
    now = time.time()
    RECENT_TIMEOUTS.append((now, endpoint))
    
    # Clean old timeouts outside window
    cutoff = now - TIMEOUT_ALERT_WINDOW_SECONDS
    RECENT_TIMEOUTS = [(t, e) for t, e in RECENT_TIMEOUTS if t > cutoff]
    
    # Check if we should alert
    if len(RECENT_TIMEOUTS) >= TIMEOUT_ALERT_THRESHOLD:
        # Check cooldown
        if LAST_TIMEOUT_ALERT is None or (now - LAST_TIMEOUT_ALERT) > TIMEOUT_ALERT_COOLDOWN:
            create_timeout_alert(RECENT_TIMEOUTS)
            LAST_TIMEOUT_ALERT = now


def create_timeout_alert(timeouts: list):
    """Create alert file for multiple API timeouts."""
    alert_file = Path(__file__).parent / "kalshi-timeout.alert"
    
    endpoints_hit = {}
    for _, endpoint in timeouts:
        endpoints_hit[endpoint] = endpoints_hit.get(endpoint, 0) + 1
    
    endpoint_summary = ", ".join(f"{e}: {c}x" for e, c in endpoints_hit.items())
    
    alert_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "api_timeout_cluster",
        "timeout_count": len(timeouts),
        "window_seconds": TIMEOUT_ALERT_WINDOW_SECONDS,
        "endpoints_affected": endpoints_hit,
        "message": f"⚠️ API TIMEOUT CLUSTER: {len(timeouts)} timeouts in {TIMEOUT_ALERT_WINDOW_SECONDS//60}min ({endpoint_summary})"
    }
    
    with open(alert_file, "w") as f:
        json.dump(alert_data, f, indent=2)
    
    print(f"⚠️ TIMEOUT ALERT: {len(timeouts)} timeouts - {endpoint_summary}")


# ============== LATENCY-BASED POSITION SIZING (T801) ==============

def get_latency_position_multiplier() -> tuple[float, str]:
    """
    Get position size multiplier based on current API latency.
    
    Reduces position size when critical API endpoints are slow, because:
    - High latency = stale prices (execution risk)
    - High latency = higher probability of order rejection
    - High latency = potential network issues
    
    Returns:
        tuple: (multiplier 0.0-1.0, reason string)
    
    Thresholds (configurable via LATENCY_SIZE_THRESHOLDS):
        - >500ms avg: 0.75 (reduce by 25%)
        - >1000ms avg: 0.50 (reduce by 50%)
        - >2000ms avg: 0.0 (skip trade entirely)
    """
    if not LATENCY_POSITION_SIZING_ENABLED:
        return 1.0, "disabled"
    
    # Get current latency profile
    profile = get_latency_profile()
    if not profile:
        return 1.0, "no_data"
    
    # Calculate avg latency for critical endpoints
    total_latency = 0.0
    endpoint_count = 0
    
    for endpoint in LATENCY_CRITICAL_ENDPOINTS:
        if endpoint in profile:
            stats = profile[endpoint]
            if stats["count"] >= 3:  # Need sufficient data
                total_latency += stats["avg_ms"]
                endpoint_count += 1
    
    if endpoint_count == 0:
        return 1.0, "insufficient_data"
    
    avg_latency = total_latency / endpoint_count
    
    # Apply threshold-based reduction
    # Check thresholds in descending order (highest first)
    for threshold_ms, multiplier in sorted(LATENCY_SIZE_THRESHOLDS.items(), reverse=True):
        if avg_latency > threshold_ms:
            reason = f"latency_{threshold_ms}ms_threshold"
            return multiplier, reason
    
    return 1.0, "normal"


# ============== LATENCY-BASED EXCHANGE PRIORITIZATION (T396) ==============

# Exchange latency endpoint mapping
EXCHANGE_LATENCY_ENDPOINTS = {
    "binance": "ext_binance",
    "coingecko": "ext_coingecko",
    "coinbase": "ext_coinbase"
}

# Default order (used when no latency data available)
DEFAULT_EXCHANGE_ORDER = ["binance", "coingecko", "coinbase"]

# Enable/disable dynamic prioritization
DYNAMIC_EXCHANGE_ORDER = os.getenv("DYNAMIC_EXCHANGE_ORDER", "true").lower() in ("true", "1", "yes")


def get_optimal_exchange_order() -> list[str]:
    """
    Get optimal exchange order based on historical latency.
    
    Returns exchanges sorted by avg latency (fastest first).
    Falls back to default order if insufficient data.
    
    Returns:
        List of exchange names in priority order
    """
    if not DYNAMIC_EXCHANGE_ORDER:
        return DEFAULT_EXCHANGE_ORDER.copy()
    
    profile = get_latency_profile()
    if not profile:
        return DEFAULT_EXCHANGE_ORDER.copy()
    
    # Get latency stats for each exchange
    exchange_latencies = []
    for exchange, endpoint in EXCHANGE_LATENCY_ENDPOINTS.items():
        if endpoint in profile:
            stats = profile[endpoint]
            if stats.get("count", 0) >= 3:  # Need at least 3 calls for reliable data
                exchange_latencies.append({
                    "exchange": exchange,
                    "avg_ms": stats.get("avg_ms", float('inf')),
                    "p95_ms": stats.get("p95_ms", float('inf')),
                    "count": stats.get("count", 0)
                })
    
    # If we don't have enough data, use default order
    if len(exchange_latencies) < 2:
        return DEFAULT_EXCHANGE_ORDER.copy()
    
    # Sort by avg latency (fastest first)
    exchange_latencies.sort(key=lambda x: x["avg_ms"])
    
    # Build ordered list, keeping any missing exchanges at the end
    ordered = [e["exchange"] for e in exchange_latencies]
    for default_ex in DEFAULT_EXCHANGE_ORDER:
        if default_ex not in ordered:
            ordered.append(default_ex)
    
    return ordered


def print_exchange_priority():
    """Print current exchange priority order with latency stats."""
    order = get_optimal_exchange_order()
    profile = get_latency_profile()
    
    print("\n🔄 EXCHANGE PRIORITY (latency-based):")
    for i, exchange in enumerate(order, 1):
        endpoint = EXCHANGE_LATENCY_ENDPOINTS.get(exchange, "")
        if endpoint and endpoint in profile:
            stats = profile[endpoint]
            avg = stats.get("avg_ms", 0)
            p95 = stats.get("p95_ms", 0)
            count = stats.get("count", 0)
            print(f"  {i}. {exchange}: avg={avg:.0f}ms, p95={p95:.0f}ms ({count} calls)")
        else:
            print(f"  {i}. {exchange}: (no data)")


# ============== API RATE LIMIT MONITORING (T308) ==============

# Rate limit tracking
API_RATE_LIMITS = {
    "kalshi": {"calls_per_hour": 0, "limit": 1000, "remaining": None, "reset_time": None},
    "coingecko": {"calls_per_hour": 0, "limit": 30, "remaining": None, "reset_time": None},  # Free tier: 10-30/min
    "binance": {"calls_per_hour": 0, "limit": 1200, "remaining": None, "reset_time": None},
    "coinbase": {"calls_per_hour": 0, "limit": 10000, "remaining": None, "reset_time": None},
    "feargreed": {"calls_per_hour": 0, "limit": 100, "remaining": None, "reset_time": None}
}
API_RATE_WINDOW_START = time.time()
RATE_LIMIT_ALERT_FILE = "scripts/kalshi-rate-limit.alert"
RATE_LIMIT_ALERT_THRESHOLD = 0.8  # Alert at 80% of limit
RATE_LIMIT_LOG_FILE = "scripts/kalshi-api-rate-log.jsonl"


def record_api_call(source: str, response_headers: dict = None):
    """
    Record an API call for rate limit tracking.
    
    Args:
        source: API source name (kalshi, coingecko, binance, coinbase, feargreed)
        response_headers: Optional response headers to extract rate limit info
    """
    global API_RATE_LIMITS, API_RATE_WINDOW_START
    
    # Reset hourly counters if window expired
    if time.time() - API_RATE_WINDOW_START > 3600:
        for src in API_RATE_LIMITS:
            API_RATE_LIMITS[src]["calls_per_hour"] = 0
        API_RATE_WINDOW_START = time.time()
    
    if source not in API_RATE_LIMITS:
        return
    
    API_RATE_LIMITS[source]["calls_per_hour"] += 1
    
    # Extract rate limit headers if provided
    if response_headers:
        # Kalshi uses X-Ratelimit-* headers
        if "x-ratelimit-remaining" in response_headers:
            API_RATE_LIMITS[source]["remaining"] = int(response_headers.get("x-ratelimit-remaining", 0))
        if "x-ratelimit-limit" in response_headers:
            API_RATE_LIMITS[source]["limit"] = int(response_headers.get("x-ratelimit-limit", 1000))
        if "x-ratelimit-reset" in response_headers:
            API_RATE_LIMITS[source]["reset_time"] = response_headers.get("x-ratelimit-reset")
        
        # CoinGecko uses x-cg-* headers
        if "x-cg-demo-api-calls-left" in response_headers:
            API_RATE_LIMITS[source]["remaining"] = int(response_headers.get("x-cg-demo-api-calls-left", 0))


def check_rate_limits() -> list:
    """
    Check if any API is approaching rate limits.
    
    Returns:
        List of (source, usage_pct, message) for APIs near limit
    """
    warnings = []
    
    for source, data in API_RATE_LIMITS.items():
        calls = data["calls_per_hour"]
        limit = data["limit"]
        remaining = data["remaining"]
        
        # Check based on remaining (from headers) if available
        if remaining is not None and limit > 0:
            usage_pct = 1 - (remaining / limit)
            if usage_pct >= RATE_LIMIT_ALERT_THRESHOLD:
                warnings.append((source, usage_pct, f"{source}: {remaining}/{limit} remaining ({usage_pct*100:.0f}% used)"))
        # Otherwise check based on our hourly count
        elif calls > 0 and limit > 0:
            usage_pct = calls / limit
            if usage_pct >= RATE_LIMIT_ALERT_THRESHOLD:
                warnings.append((source, usage_pct, f"{source}: {calls}/{limit} calls/hour ({usage_pct*100:.0f}% used)"))
    
    return warnings


def write_rate_limit_alert(warnings: list):
    """Write rate limit alert file for heartbeat pickup."""
    alert_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "warnings": [{"source": w[0], "usage_pct": round(w[1]*100, 1), "message": w[2]} for w in warnings],
        "full_status": {src: {k: v for k, v in data.items()} for src, data in API_RATE_LIMITS.items()}
    }
    
    try:
        with open(RATE_LIMIT_ALERT_FILE, "w") as f:
            json.dump(alert_data, f, indent=2)
        print(f"⚠️ RATE LIMIT ALERT written: {[w[2] for w in warnings]}")
    except Exception as e:
        print(f"Failed to write rate limit alert: {e}")


def log_rate_limits():
    """Log current rate limit status to JSONL file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window_start": datetime.fromtimestamp(API_RATE_WINDOW_START, timezone.utc).isoformat(),
        "sources": {src: {
            "calls": data["calls_per_hour"],
            "limit": data["limit"],
            "remaining": data["remaining"],
            "usage_pct": round((data["calls_per_hour"] / data["limit"] * 100) if data["limit"] > 0 else 0, 1)
        } for src, data in API_RATE_LIMITS.items()}
    }
    
    try:
        with open(RATE_LIMIT_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Failed to log rate limits: {e}")


def get_rate_limit_summary() -> str:
    """Get formatted rate limit summary for console output."""
    lines = ["📊 API RATE LIMITS:"]
    for source, data in API_RATE_LIMITS.items():
        calls = data["calls_per_hour"]
        limit = data["limit"]
        remaining = data["remaining"]
        
        if remaining is not None:
            status = f"{remaining} remaining"
        else:
            status = f"{calls}/{limit} calls/hour"
        
        usage_pct = (calls / limit * 100) if limit > 0 else 0
        indicator = "🟢" if usage_pct < 50 else "🟡" if usage_pct < 80 else "🔴"
        lines.append(f"  {indicator} {source}: {status}")
    
    return "\n".join(lines)


# ============== EXTERNAL API CACHING (T427) ==============

# Cache for external API responses to reduce redundant calls
EXT_API_CACHE = {}  # key -> (timestamp, data)
EXT_API_CACHE_TTL = 60  # 60 second TTL for cached responses


def get_cached_response(cache_key: str):
    """
    Get cached API response if still valid.
    
    Args:
        cache_key: Unique key for the cached data
        
    Returns:
        Cached data if valid, None if expired or missing
    """
    if cache_key not in EXT_API_CACHE:
        return None
    
    cached_time, cached_data = EXT_API_CACHE[cache_key]
    if time.time() - cached_time > EXT_API_CACHE_TTL:
        del EXT_API_CACHE[cache_key]
        return None
    
    record_api_latency(f"{cache_key}_cache_hit", 0)  # Track cache hits
    return cached_data


def set_cached_response(cache_key: str, data: any):
    """
    Cache an API response.
    
    Args:
        cache_key: Unique key for the cached data
        data: Response data to cache
    """
    EXT_API_CACHE[cache_key] = (time.time(), data)


def get_cache_stats() -> dict:
    """Get cache statistics."""
    valid_entries = sum(1 for key, (ts, _) in EXT_API_CACHE.items() 
                       if time.time() - ts <= EXT_API_CACHE_TTL)
    return {
        "total_entries": len(EXT_API_CACHE),
        "valid_entries": valid_entries,
        "keys": list(EXT_API_CACHE.keys())
    }


# ============== REGIME CHANGE ALERTING ==============

def load_regime_state() -> dict:
    """Load previous regime state from file."""
    try:
        with open(REGIME_STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"btc": None, "eth": None, "last_alert": 0}


def save_regime_state(state: dict):
    """Save current regime state to file."""
    with open(REGIME_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ============== VIX INTEGRATION (T611) ==============

def load_vix_data() -> dict:
    """
    Load VIX correlation data for regime enhancement.
    
    Returns dict with:
        - vix_current: Current VIX level
        - vix_regime: 'low_fear', 'moderate', 'elevated', 'high_fear'
        - valid: Whether data is fresh enough to use
        - edge_adjustment: Suggested edge adjustment based on VIX
        - size_multiplier: Suggested position size multiplier
    """
    result = {
        "vix_current": None,
        "vix_regime": "unknown",
        "valid": False,
        "edge_adjustment": 0,
        "size_multiplier": 1.0,
        "correlation": None
    }
    
    try:
        if not os.path.exists(VIX_CORRELATION_FILE):
            return result
        
        # Check file freshness
        file_mtime = os.path.getmtime(VIX_CORRELATION_FILE)
        file_age_hours = (time.time() - file_mtime) / 3600
        
        if file_age_hours > VIX_CACHE_MAX_AGE_HOURS:
            # Data too old, don't use it
            return result
        
        with open(VIX_CORRELATION_FILE, "r") as f:
            data = json.load(f)
        
        vix_current = data.get("vix_current")
        vix_regime = data.get("vix_regime", "unknown")
        correlation = data.get("correlation", 0)
        
        if vix_current is None:
            return result
        
        result["vix_current"] = vix_current
        result["vix_regime"] = vix_regime
        result["correlation"] = correlation
        result["valid"] = True
        
        # Calculate adjustments based on VIX level
        # VIX >= 25 (high_fear): Be more conservative
        #   - Higher edge required (+3%)
        #   - Smaller position size (0.7x)
        # VIX 20-25 (elevated): Moderate caution
        #   - Slightly higher edge (+1.5%)
        #   - Slightly smaller size (0.85x)
        # VIX 15-20 (moderate): Normal operation
        # VIX < 15 (low_fear): Can be slightly more aggressive
        #   - Slightly lower edge (-0.5%)
        
        if vix_current >= VIX_HIGH_FEAR_THRESHOLD:
            result["edge_adjustment"] = 0.03  # +3% minimum edge
            result["size_multiplier"] = 0.70  # 70% position size
        elif vix_current >= VIX_ELEVATED_THRESHOLD:
            result["edge_adjustment"] = 0.015  # +1.5% minimum edge
            result["size_multiplier"] = 0.85  # 85% position size
        elif vix_current < VIX_LOW_FEAR_THRESHOLD:
            result["edge_adjustment"] = -0.005  # -0.5% edge (can be more aggressive)
            result["size_multiplier"] = 1.0  # Normal size
        else:
            # Moderate zone (15-20): normal
            result["edge_adjustment"] = 0
            result["size_multiplier"] = 1.0
        
        return result
        
    except (json.JSONDecodeError, KeyError, OSError) as e:
        # Silently fail and return defaults
        return result


def get_vix_regime_emoji(vix_regime: str) -> str:
    """Get emoji for VIX regime display."""
    return {
        "low_fear": "🟢",
        "moderate": "🟡",
        "elevated": "🟠",
        "high_fear": "🔴"
    }.get(vix_regime, "⚪")


def check_regime_change(current_regimes: dict) -> list:
    """
    Check if regime changed for any asset and return changes.
    
    Args:
        current_regimes: Dict with btc and eth regime dicts from detect_market_regime()
    
    Returns:
        List of (asset, old_regime, new_regime) tuples for any changes
    """
    state = load_regime_state()
    changes = []
    
    for asset in ["btc", "eth"]:
        new_regime = current_regimes.get(asset, {}).get("regime", "unknown")
        old_regime = state.get(asset)
        
        if old_regime and old_regime != new_regime:
            changes.append((asset.upper(), old_regime, new_regime))
        
        # Update state
        state[asset] = new_regime
    
    save_regime_state(state)
    return changes


def write_regime_alert(changes: list, regime_details: dict):
    """
    Write regime change alert file for heartbeat pickup.
    
    Args:
        changes: List of (asset, old_regime, new_regime) tuples
        regime_details: Full regime data for context
    """
    state = load_regime_state()
    now = time.time()
    
    # Check cooldown
    if now - state.get("last_alert", 0) < REGIME_ALERT_COOLDOWN:
        print(f"   ⏳ Regime alert on cooldown ({int((REGIME_ALERT_COOLDOWN - (now - state['last_alert']))/60)}min left)")
        return
    
    alert_lines = ["🔄 MARKET REGIME CHANGE DETECTED\n"]
    
    for asset, old, new in changes:
        # Get direction indicator
        if new in ("trending_bullish",):
            emoji = "📈"
        elif new in ("trending_bearish",):
            emoji = "📉"
        elif new == "choppy":
            emoji = "⚡"
        else:
            emoji = "➡️"
        
        alert_lines.append(f"{emoji} {asset}: {old} → {new}")
        
        # Add context if available
        details = regime_details.get(asset.lower(), {}).get("details", {})
        if details:
            chg_4h = details.get("change_4h", 0) * 100
            chg_24h = details.get("change_24h", 0) * 100
            alert_lines.append(f"   4h: {chg_4h:+.2f}% | 24h: {chg_24h:+.2f}%")
        
        # Add trading implication
        new_edge = regime_details.get(asset.lower(), {}).get("dynamic_min_edge", MIN_EDGE)
        alert_lines.append(f"   New MIN_EDGE: {new_edge*100:.0f}%\n")
    
    alert_lines.append(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    
    with open(REGIME_ALERT_FILE, "w") as f:
        f.write("\n".join(alert_lines))
    
    # Update last alert time
    state["last_alert"] = now
    save_regime_state(state)
    
    print(f"   📢 Regime change alert written to {REGIME_ALERT_FILE}")


# ============== MOMENTUM DIRECTION CHANGE ALERTING ==============

def load_momentum_state() -> dict:
    """Load previous momentum direction state from file."""
    try:
        with open(MOMENTUM_STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"btc": None, "eth": None, "last_alert": 0}


def save_momentum_state(state: dict):
    """Save current momentum direction state to file."""
    with open(MOMENTUM_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_momentum_direction_label(composite_dir: float) -> str:
    """Convert composite direction to label (bullish/bearish/neutral)."""
    if composite_dir > 0.1:
        return "bullish"
    elif composite_dir < -0.1:
        return "bearish"
    else:
        return "neutral"


def check_momentum_change(momentum_data: dict) -> list:
    """
    Check if momentum direction changed for any asset.
    
    Only alerts on significant direction flips:
    - bullish → bearish
    - bearish → bullish
    (Ignores neutral transitions to reduce noise)
    
    Returns:
        List of (asset, old_dir, new_dir, details) tuples for significant changes
    """
    state = load_momentum_state()
    changes = []
    
    for asset in ["btc", "eth"]:
        momentum = momentum_data.get(asset, {})
        composite_dir = momentum.get("composite_direction", 0)
        new_label = get_momentum_direction_label(composite_dir)
        old_label = state.get(asset)
        
        # Only alert on bullish↔bearish flips (not neutral transitions)
        if old_label and new_label != old_label:
            if (old_label == "bullish" and new_label == "bearish") or \
               (old_label == "bearish" and new_label == "bullish"):
                details = {
                    "composite_dir": composite_dir,
                    "composite_str": momentum.get("composite_strength", 0),
                    "alignment": momentum.get("alignment", False),
                    "timeframes": momentum.get("timeframes", {})
                }
                changes.append((asset.upper(), old_label, new_label, details))
        
        # Update state
        state[asset] = new_label
    
    save_momentum_state(state)
    return changes


def write_momentum_alert(changes: list):
    """
    Write momentum direction change alert file for heartbeat pickup.
    
    Args:
        changes: List of (asset, old_dir, new_dir, details) tuples
    """
    state = load_momentum_state()
    now = time.time()
    
    # Check cooldown
    if now - state.get("last_alert", 0) < MOMENTUM_ALERT_COOLDOWN:
        print(f"   ⏳ Momentum alert on cooldown ({int((MOMENTUM_ALERT_COOLDOWN - (now - state['last_alert']))/60)}min left)")
        return
    
    alert_lines = ["📊 MOMENTUM DIRECTION CHANGE\n"]
    
    for asset, old_dir, new_dir, details in changes:
        # Direction emoji
        if new_dir == "bullish":
            emoji = "🟢📈"
            action = "Consider YES bets"
        else:
            emoji = "🔴📉"
            action = "Consider NO bets"
        
        alert_lines.append(f"{emoji} {asset}: {old_dir.upper()} → {new_dir.upper()}")
        
        # Timeframe breakdown
        tfs = details.get("timeframes", {})
        tf_parts = []
        for tf in ["1h", "4h", "24h"]:
            tf_data = tfs.get(tf, {})
            pct = tf_data.get("pct_change", 0) * 100
            tf_parts.append(f"{tf}: {pct:+.2f}%")
        if tf_parts:
            alert_lines.append(f"   {' | '.join(tf_parts)}")
        
        alert_lines.append(f"   Composite: {details['composite_dir']:+.2f} | Strength: {details['composite_str']:.2f}")
        alert_lines.append(f"   💡 {action}\n")
    
    alert_lines.append(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    
    with open(MOMENTUM_ALERT_FILE, "w") as f:
        f.write("\n".join(alert_lines))
    
    # Update last alert time
    state["last_alert"] = now
    save_momentum_state(state)
    
    print(f"   📢 Momentum change alert written to {MOMENTUM_ALERT_FILE}")


# ============== WHIPSAW DETECTION (T393) ==============

def check_whipsaw(changes: list):
    """
    Detect whipsaw pattern: momentum flips direction twice within 24h.
    
    Whipsaw indicates choppy market conditions - consider reducing position sizes
    or pausing trading for that asset.
    
    Args:
        changes: List of (asset, old_dir, new_dir, details) tuples from current check
    """
    if not changes:
        return
    
    state = load_momentum_state()
    now = time.time()
    
    # Initialize history if not present
    if "direction_history" not in state:
        state["direction_history"] = {"btc": [], "eth": []}
    
    whipsaws_detected = []
    window_seconds = WHIPSAW_WINDOW_HOURS * 3600
    
    for asset, old_dir, new_dir, details in changes:
        asset_lower = asset.lower()
        history = state["direction_history"].get(asset_lower, [])
        
        # Add current change to history
        history.append({
            "timestamp": now,
            "old": old_dir,
            "new": new_dir
        })
        
        # Clean old entries (older than window)
        history = [h for h in history if now - h["timestamp"] < window_seconds]
        state["direction_history"][asset_lower] = history
        
        # Check for whipsaw: 2+ significant flips in window
        # Look for pattern: bullish→bearish→bullish or bearish→bullish→bearish
        if len(history) >= 2:
            # Count direction flips (not just changes - actual reversals)
            flip_count = 0
            for i, change in enumerate(history):
                if change["old"] in ("bullish", "bearish") and change["new"] in ("bullish", "bearish"):
                    # This is a significant flip (not involving neutral)
                    flip_count += 1
            
            if flip_count >= 2:
                # Whipsaw detected!
                hours_window = (now - history[0]["timestamp"]) / 3600
                whipsaws_detected.append({
                    "asset": asset,
                    "flips": flip_count,
                    "window_hours": hours_window,
                    "history": history[-3:],  # Last 3 changes
                    "latest_direction": new_dir
                })
    
    save_momentum_state(state)
    
    # Write alert if whipsaw detected
    if whipsaws_detected:
        write_whipsaw_alert(whipsaws_detected)


def write_whipsaw_alert(whipsaws: list):
    """Write whipsaw alert file for heartbeat pickup."""
    state = load_momentum_state()
    now = time.time()
    
    # Check cooldown
    if now - state.get("last_whipsaw_alert", 0) < WHIPSAW_ALERT_COOLDOWN:
        remaining = int((WHIPSAW_ALERT_COOLDOWN - (now - state["last_whipsaw_alert"])) / 60)
        print(f"   ⏳ Whipsaw alert on cooldown ({remaining}min left)")
        return
    
    alert_lines = ["⚠️ WHIPSAW DETECTED - CHOPPY MARKET\n"]
    
    for ws in whipsaws:
        alert_lines.append(f"🔀 {ws['asset']}: {ws['flips']} direction flips in {ws['window_hours']:.1f}h")
        alert_lines.append(f"   Current direction: {ws['latest_direction']}")
        alert_lines.append(f"   Recent changes: {' → '.join(h['new'] for h in ws['history'])}")
        alert_lines.append("")
    
    alert_lines.append("💡 RECOMMENDATION:")
    alert_lines.append("   • Reduce position sizes for affected assets")
    alert_lines.append("   • Consider pausing trading until momentum stabilizes")
    alert_lines.append("   • Higher MIN_EDGE may be appropriate")
    alert_lines.append("")
    alert_lines.append(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    
    with open(WHIPSAW_ALERT_FILE, "w") as f:
        f.write("\n".join(alert_lines))
    
    # Update last alert time
    state["last_whipsaw_alert"] = now
    save_momentum_state(state)
    
    print(f"   ⚠️ Whipsaw alert written to {WHIPSAW_ALERT_FILE}")


# ============== FULL MOMENTUM ALIGNMENT ALERTING (T301) ==============

def check_momentum_alignment_alert(momentum_data: dict):
    """
    Check if all timeframes are aligned with strong signal.
    Alert for high-conviction opportunity when 1h/4h/24h all agree.
    """
    if not momentum_data:
        return
    
    alerts = []
    now = time.time()
    
    # Check cooldown
    try:
        if os.path.exists(MOMENTUM_ALIGN_ALERT_FILE):
            mtime = os.path.getmtime(MOMENTUM_ALIGN_ALERT_FILE)
            if now - mtime < MOMENTUM_ALIGN_ALERT_COOLDOWN:
                return  # On cooldown
    except Exception:
        pass
    
    for asset in ["btc", "eth"]:
        momentum = momentum_data.get(asset, {})
        
        # Check for full alignment with strong signal
        is_aligned = momentum.get("alignment", False)
        composite_dir = momentum.get("composite_direction", 0)
        composite_str = momentum.get("composite_strength", 0)
        timeframes = momentum.get("timeframes", {})
        
        if not is_aligned or composite_str < MOMENTUM_ALIGN_MIN_STRENGTH:
            continue
        
        # Determine direction
        if composite_dir > 0.3:
            direction = "🟢 BULLISH"
            signal = "Strong upward momentum across all timeframes"
            action = "Consider YES bets on upside targets"
        elif composite_dir < -0.3:
            direction = "🔴 BEARISH" 
            signal = "Strong downward momentum across all timeframes"
            action = "Consider NO bets on upside targets"
        else:
            continue  # Not strong enough
        
        alerts.append({
            "asset": asset.upper(),
            "direction": direction,
            "signal": signal,
            "action": action,
            "composite_dir": composite_dir,
            "composite_str": composite_str,
            "timeframes": timeframes
        })
    
    if alerts:
        write_momentum_alignment_alert(alerts)


def write_momentum_alignment_alert(alerts: list):
    """Write full momentum alignment alert file for heartbeat pickup."""
    alert_lines = [
        "🎯 FULL MOMENTUM ALIGNMENT DETECTED!\n",
        "All timeframes (1h/4h/24h) agree - HIGH CONVICTION signal\n"
    ]
    
    for alert in alerts:
        asset = alert["asset"]
        direction = alert["direction"]
        signal = alert["signal"]
        action = alert["action"]
        composite_dir = alert["composite_dir"]
        composite_str = alert["composite_str"]
        timeframes = alert["timeframes"]
        
        alert_lines.append(f"📊 {asset}: {direction}")
        alert_lines.append(f"   {signal}")
        alert_lines.append("")
        
        # Show individual timeframes
        for tf, data in timeframes.items():
            tf_dir = data.get("direction", 0)
            tf_str = data.get("strength", 0)
            tf_label = "↑" if tf_dir > 0 else "↓" if tf_dir < 0 else "→"
            alert_lines.append(f"   {tf}: {tf_label} dir={tf_dir:+.2f} str={tf_str:.2f}")
        
        alert_lines.append(f"\n   Composite: {composite_dir:+.2f} | Strength: {composite_str:.2f}")
        alert_lines.append(f"   💡 {action}\n")
    
    alert_lines.append(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    alert_lines.append("\n⚠️ This is a rare high-confidence signal. Use appropriate position sizing!")
    
    alert_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "momentum_alignment",
        "alerts": alerts,
        "message": "\n".join(alert_lines)
    }
    
    with open(MOMENTUM_ALIGN_ALERT_FILE, "w") as f:
        json.dump(alert_data, f, indent=2)
    
    print(f"   🎯 Full momentum alignment alert written!")


# ============== MOMENTUM REVERSION DETECTION (T302) ==============

def detect_momentum_reversion(ohlc_data: dict, momentum_data: dict) -> list:
    """
    Detect extended momentum that often precedes reversals.
    
    Triggers when:
    - 4h move > 2% (REVERSION_4H_THRESHOLD)
    - OR 8h move > 3% (REVERSION_8H_THRESHOLD)
    - AND momentum strength is high (> 0.7)
    
    Extended moves tend to mean-revert in crypto, especially in hourly contracts.
    This signals a potential contrarian opportunity.
    
    Returns:
        List of reversion signals with asset, direction, confidence, suggested action
    """
    reversions = []
    
    for asset in ["btc", "eth"]:
        ohlc = ohlc_data.get(asset, [])
        momentum = momentum_data.get(asset, {})
        
        if not ohlc or len(ohlc) < 8:
            continue
        
        # Get current and historical prices
        current_price = ohlc[-1][4] if ohlc[-1] else None
        if not current_price:
            continue
        
        # 4-hour price change
        price_4h_ago = ohlc[-4][4] if len(ohlc) >= 4 else current_price
        change_4h = (current_price - price_4h_ago) / price_4h_ago if price_4h_ago else 0
        
        # 8-hour price change (if available)
        price_8h_ago = ohlc[-8][4] if len(ohlc) >= 8 else price_4h_ago
        change_8h = (current_price - price_8h_ago) / price_8h_ago if price_8h_ago else 0
        
        # Get momentum strength
        composite_str = momentum.get("composite_strength", 0)
        composite_dir = momentum.get("composite_direction", 0)
        
        # Check for extended move
        abs_4h = abs(change_4h)
        abs_8h = abs(change_8h)
        
        is_extended_4h = abs_4h >= REVERSION_4H_THRESHOLD
        is_extended_8h = abs_8h >= REVERSION_8H_THRESHOLD
        is_strong_momentum = composite_str >= REVERSION_STRENGTH_THRESHOLD
        
        if not (is_extended_4h or is_extended_8h):
            continue
        
        if not is_strong_momentum:
            continue  # Weak momentum = probably not a reversion candidate
        
        # Determine reversion direction (opposite to move)
        if change_4h > 0:
            reversion_dir = "bearish"
            current_trend = "bullish"
            contrarian_action = "Consider NO bets on upside / YES bets on downside"
            emoji = "🔻"
        else:
            reversion_dir = "bullish"
            current_trend = "bearish"
            contrarian_action = "Consider YES bets on upside recovery"
            emoji = "🔺"
        
        # Calculate confidence based on extension degree
        confidence = "medium"
        if is_extended_8h:
            confidence = "high"
        if abs_4h > REVERSION_4H_THRESHOLD * 1.5:  # 3%+ in 4h
            confidence = "very_high"
        
        reversions.append({
            "asset": asset.upper(),
            "current_trend": current_trend,
            "reversion_dir": reversion_dir,
            "change_4h": change_4h,
            "change_8h": change_8h,
            "momentum_strength": composite_str,
            "confidence": confidence,
            "action": contrarian_action,
            "emoji": emoji,
            "current_price": current_price
        })
    
    return reversions


def check_reversion_alert(ohlc_data: dict, momentum_data: dict):
    """
    Check for momentum reversion signals and write alert if found.
    """
    if not ohlc_data or not momentum_data:
        return
    
    now = time.time()
    
    # Check cooldown
    try:
        if os.path.exists(REVERSION_ALERT_FILE):
            mtime = os.path.getmtime(REVERSION_ALERT_FILE)
            if now - mtime < REVERSION_ALERT_COOLDOWN:
                return  # On cooldown
    except Exception:
        pass
    
    reversions = detect_momentum_reversion(ohlc_data, momentum_data)
    
    if reversions:
        write_reversion_alert(reversions)


def write_reversion_alert(reversions: list):
    """Write momentum reversion alert file for heartbeat pickup."""
    alert_lines = [
        "⚡ MOMENTUM REVERSION SIGNAL!\n",
        "Extended move detected - potential mean reversion opportunity\n"
    ]
    
    for rev in reversions:
        asset = rev["asset"]
        emoji = rev["emoji"]
        current_trend = rev["current_trend"]
        confidence = rev["confidence"]
        change_4h = rev["change_4h"]
        change_8h = rev["change_8h"]
        mom_str = rev["momentum_strength"]
        action = rev["action"]
        price = rev["current_price"]
        
        conf_emoji = "🟢" if confidence == "very_high" else "🟡" if confidence == "high" else "⚪"
        
        alert_lines.append(f"{emoji} {asset}: Extended {current_trend.upper()} move")
        alert_lines.append(f"   4h: {change_4h:+.2%} | 8h: {change_8h:+.2%}")
        alert_lines.append(f"   Momentum strength: {mom_str:.2f}")
        alert_lines.append(f"   Confidence: {conf_emoji} {confidence.upper()}")
        alert_lines.append(f"   Current price: ${price:,.0f}")
        alert_lines.append(f"\n   💡 Contrarian: {action}\n")
    
    alert_lines.append(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    alert_lines.append("\n⚠️ Reversion signals are contrarian. Use smaller position sizes!")
    alert_lines.append("📊 Extended moves often revert, but can also accelerate (momentum).")
    
    alert_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "momentum_reversion",
        "reversions": reversions,
        "message": "\n".join(alert_lines)
    }
    
    with open(REVERSION_ALERT_FILE, "w") as f:
        json.dump(alert_data, f, indent=2)
    
    print(f"   ⚡ Momentum reversion alert written ({len(reversions)} signals)!")


def get_reversion_edge_adjustment(asset: str, ohlc_data: dict, momentum_data: dict) -> dict:
    """
    Get edge adjustment for reversion signals.
    
    Returns:
        dict with:
        - has_signal: bool
        - adjustment: float (positive = bonus for contrarian, negative = penalty for with-trend)
        - reason: str
    """
    result = {"has_signal": False, "adjustment": 0.0, "reason": ""}
    
    reversions = detect_momentum_reversion(ohlc_data, momentum_data)
    
    for rev in reversions:
        if rev["asset"].lower() == asset.lower():
            result["has_signal"] = True
            
            # Give bonus for contrarian bets
            if rev["confidence"] == "very_high":
                result["adjustment"] = 0.02  # +2% edge bonus for strong contrarian
            elif rev["confidence"] == "high":
                result["adjustment"] = 0.01  # +1% edge bonus
            else:
                result["adjustment"] = 0.005  # +0.5% for medium confidence
            
            result["reason"] = f"Extended {rev['current_trend']} move ({rev['change_4h']:+.1%} 4h) - reversion likely"
            break
    
    return result


# ============== MOMENTUM DIVERGENCE DETECTION (T303) ==============

def calculate_rsi(ohlc_data: list, period: int = 14) -> list:
    """
    Calculate RSI (Relative Strength Index) for momentum comparison.
    
    Args:
        ohlc_data: List of [timestamp, open, high, low, close] candles
        period: RSI period (default 14)
    
    Returns:
        List of RSI values (same length as input, first 'period' entries are None)
    """
    if not ohlc_data or len(ohlc_data) < period + 1:
        return [None] * len(ohlc_data) if ohlc_data else []
    
    closes = [c[4] for c in ohlc_data]
    rsi_values = [None] * len(closes)
    
    # Calculate price changes
    gains = []
    losses = []
    
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))
    
    # Initial averages
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(closes)):
        if avg_loss == 0:
            rsi_values[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))
        
        # Update averages with smoothing
        if i < len(gains):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    return rsi_values


def detect_momentum_divergence(ohlc_data: dict, momentum_data: dict) -> list:
    """
    Detect price vs momentum divergence - classic reversal signal.
    
    Bullish divergence: Price makes LOWER low, but momentum (RSI) makes HIGHER low
                        → Signals upward reversal (weakening selling pressure)
    
    Bearish divergence: Price makes HIGHER high, but momentum (RSI) makes LOWER high
                        → Signals downward reversal (weakening buying pressure)
    
    Args:
        ohlc_data: Dict with 'btc' and 'eth' OHLC lists
        momentum_data: Dict with momentum info per asset
    
    Returns:
        List of divergence signals with asset, type, confidence, action
    """
    divergences = []
    
    for asset in ["btc", "eth"]:
        ohlc = ohlc_data.get(asset, [])
        
        if not ohlc or len(ohlc) < DIVERGENCE_LOOKBACK:
            continue
        
        # Use recent candles only
        recent = ohlc[-DIVERGENCE_LOOKBACK:]
        closes = [c[4] for c in recent]
        highs = [c[2] for c in recent]
        lows = [c[3] for c in recent]
        
        # Calculate RSI for recent period
        rsi_values = calculate_rsi(ohlc, period=6)  # Shorter period for responsiveness
        recent_rsi = rsi_values[-DIVERGENCE_LOOKBACK:] if len(rsi_values) >= DIVERGENCE_LOOKBACK else []
        
        # Filter out None values for RSI analysis
        valid_rsi_indices = [i for i, r in enumerate(recent_rsi) if r is not None]
        if len(valid_rsi_indices) < 4:
            continue  # Need enough data points
        
        # Find swing highs and lows in price
        # Looking at first half vs second half of recent data
        mid = DIVERGENCE_LOOKBACK // 2
        
        # Price extremes
        first_half_low = min(lows[:mid])
        second_half_low = min(lows[mid:])
        first_half_high = max(highs[:mid])
        second_half_high = max(highs[mid:])
        
        # RSI extremes (only where RSI is valid)
        first_half_rsi_valid = [recent_rsi[i] for i in valid_rsi_indices if i < mid]
        second_half_rsi_valid = [recent_rsi[i] for i in valid_rsi_indices if i >= mid]
        
        if not first_half_rsi_valid or not second_half_rsi_valid:
            continue
        
        first_half_rsi_low = min(first_half_rsi_valid)
        second_half_rsi_low = min(second_half_rsi_valid)
        first_half_rsi_high = max(first_half_rsi_valid)
        second_half_rsi_high = max(second_half_rsi_valid)
        
        current_price = closes[-1]
        price_move_pct = (current_price - closes[0]) / closes[0] if closes[0] else 0
        
        # Skip if price move is too small
        if abs(price_move_pct) < DIVERGENCE_MIN_PRICE_MOVE:
            continue
        
        # Check for BULLISH divergence: price lower low, RSI higher low
        price_lower_low = second_half_low < first_half_low
        rsi_higher_low = second_half_rsi_low > first_half_rsi_low + 3  # 3-point threshold
        
        if price_lower_low and rsi_higher_low:
            # Calculate confidence based on divergence strength
            price_drop = (first_half_low - second_half_low) / first_half_low
            rsi_rise = second_half_rsi_low - first_half_rsi_low
            
            confidence = "medium"
            if price_drop > 0.015 and rsi_rise > 5:  # Strong divergence
                confidence = "high"
            if price_drop > 0.025 and rsi_rise > 8:  # Very strong
                confidence = "very_high"
            
            divergences.append({
                "asset": asset.upper(),
                "type": "bullish",
                "emoji": "🟢",
                "signal": "Price lower low + RSI higher low",
                "action": "Consider YES bets on upside recovery",
                "confidence": confidence,
                "price_drop": price_drop,
                "rsi_rise": rsi_rise,
                "current_price": current_price,
                "current_rsi": recent_rsi[-1] if recent_rsi[-1] else 0
            })
        
        # Check for BEARISH divergence: price higher high, RSI lower high
        price_higher_high = second_half_high > first_half_high
        rsi_lower_high = second_half_rsi_high < first_half_rsi_high - 3  # 3-point threshold
        
        if price_higher_high and rsi_lower_high:
            price_rise = (second_half_high - first_half_high) / first_half_high
            rsi_drop = first_half_rsi_high - second_half_rsi_high
            
            confidence = "medium"
            if price_rise > 0.015 and rsi_drop > 5:
                confidence = "high"
            if price_rise > 0.025 and rsi_drop > 8:
                confidence = "very_high"
            
            divergences.append({
                "asset": asset.upper(),
                "type": "bearish",
                "emoji": "🔴",
                "signal": "Price higher high + RSI lower high",
                "action": "Consider NO bets on continued upside",
                "confidence": confidence,
                "price_rise": price_rise,
                "rsi_drop": rsi_drop,
                "current_price": current_price,
                "current_rsi": recent_rsi[-1] if recent_rsi[-1] else 0
            })
    
    return divergences


def check_divergence_alert(ohlc_data: dict, momentum_data: dict):
    """
    Check for momentum divergence signals and write alert if found.
    """
    if not ohlc_data:
        return
    
    now = time.time()
    
    # Check cooldown
    try:
        if os.path.exists(DIVERGENCE_ALERT_FILE):
            mtime = os.path.getmtime(DIVERGENCE_ALERT_FILE)
            if now - mtime < DIVERGENCE_ALERT_COOLDOWN:
                return  # On cooldown
    except Exception:
        pass
    
    divergences = detect_momentum_divergence(ohlc_data, momentum_data)
    
    if divergences:
        write_divergence_alert(divergences)


def write_divergence_alert(divergences: list):
    """Write momentum divergence alert file for heartbeat pickup."""
    alert_lines = [
        "📊 MOMENTUM DIVERGENCE DETECTED!\n",
        "Price and momentum disagree - potential reversal signal\n"
    ]
    
    for div in divergences:
        asset = div["asset"]
        div_type = div["type"]
        emoji = div["emoji"]
        signal = div["signal"]
        confidence = div["confidence"]
        action = div["action"]
        price = div["current_price"]
        rsi = div["current_rsi"]
        
        conf_emoji = "🟢" if confidence == "very_high" else "🟡" if confidence == "high" else "⚪"
        
        alert_lines.append(f"{emoji} {asset}: {div_type.upper()} DIVERGENCE")
        alert_lines.append(f"   {signal}")
        alert_lines.append(f"   RSI: {rsi:.1f} | Price: ${price:,.0f}")
        
        if div_type == "bullish":
            alert_lines.append(f"   Price drop: {div['price_drop']:.2%} | RSI rise: +{div['rsi_rise']:.1f}")
        else:
            alert_lines.append(f"   Price rise: {div['price_rise']:.2%} | RSI drop: -{div['rsi_drop']:.1f}")
        
        alert_lines.append(f"   Confidence: {conf_emoji} {confidence.upper()}")
        alert_lines.append(f"\n   💡 {action}\n")
    
    alert_lines.append(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    alert_lines.append("\n⚠️ Divergence is a leading indicator. Wait for price confirmation!")
    alert_lines.append("📈 Classic technical pattern: divergence often precedes reversals.")
    
    alert_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "momentum_divergence",
        "divergences": divergences,
        "message": "\n".join(alert_lines)
    }
    
    with open(DIVERGENCE_ALERT_FILE, "w") as f:
        json.dump(alert_data, f, indent=2)
    
    print(f"   📊 Momentum divergence alert written ({len(divergences)} signals)!")


def get_divergence_edge_adjustment(asset: str, ohlc_data: dict, momentum_data: dict) -> dict:
    """
    Get edge adjustment for divergence signals.
    
    Returns:
        dict with:
        - has_signal: bool
        - adjustment: float (bonus for trades aligned with divergence signal)
        - reason: str
    """
    result = {"has_signal": False, "adjustment": 0.0, "reason": ""}
    
    divergences = detect_momentum_divergence(ohlc_data, momentum_data)
    
    for div in divergences:
        if div["asset"].lower() == asset.lower():
            result["has_signal"] = True
            
            # Bonus based on confidence
            if div["confidence"] == "very_high":
                result["adjustment"] = 0.02  # +2% edge bonus
            elif div["confidence"] == "high":
                result["adjustment"] = 0.015  # +1.5% edge bonus
            else:
                result["adjustment"] = 0.01  # +1% for medium confidence
            
            result["reason"] = f"{div['type'].capitalize()} divergence detected (RSI vs price disagree)"
            result["divergence_type"] = div["type"]
            break
    
    return result


# ============== COMPOSITE SIGNAL SCORING (T460) ==============

def calculate_composite_signal_score(asset: str, side: str, ohlc_data: dict, 
                                      momentum_data: dict) -> dict:
    """
    Calculate composite signal score combining divergence, reversion, and momentum alignment.
    
    Multiple confirming signals = higher conviction = edge bonus.
    
    Args:
        asset: 'btc' or 'eth'
        side: 'yes' or 'no' (the bet direction)
        ohlc_data: Dict with 'btc' and 'eth' OHLC lists
        momentum_data: Dict with momentum info per asset
    
    Returns:
        dict with:
        - confirming_signals: int (0-3)
        - total_bonus: float (combined edge bonus)
        - confidence: str ('low', 'medium', 'high', 'very_high')
        - signals: list of signal descriptions
        - details: dict with individual signal info
    """
    result = {
        "confirming_signals": 0,
        "total_bonus": 0.0,
        "confidence": "low",
        "signals": [],
        "details": {
            "momentum_aligned": False,
            "reversion_signal": False,
            "divergence_signal": False
        }
    }
    
    # Track individual signals that confirm our bet direction
    momentum = momentum_data.get(asset, {}) if momentum_data else {}
    mom_dir = momentum.get("composite_direction", 0)
    mom_str = momentum.get("composite_strength", 0)
    mom_alignment = momentum.get("alignment", False)
    
    # Signal 1: MOMENTUM ALIGNMENT
    # For YES bets, bullish momentum confirms; for NO bets, bearish confirms
    momentum_confirms = False
    momentum_bonus = 0.0
    
    if side == "yes" and mom_dir > 0.2 and mom_str > 0.3:
        # Bullish momentum confirms YES bet
        momentum_confirms = True
        momentum_bonus = 0.02 if mom_alignment else 0.01
        result["signals"].append(f"Bullish momentum ({mom_dir:+.2f}, aligned={mom_alignment})")
    elif side == "no" and mom_dir < -0.2 and mom_str > 0.3:
        # Bearish momentum confirms NO bet
        momentum_confirms = True
        momentum_bonus = 0.02 if mom_alignment else 0.01
        result["signals"].append(f"Bearish momentum ({mom_dir:+.2f}, aligned={mom_alignment})")
    
    if momentum_confirms:
        result["confirming_signals"] += 1
        result["details"]["momentum_aligned"] = True
    
    # Signal 2: REVERSION SIGNAL
    # Reversion suggests contrarian move - confirms if our bet aligns with expected reversion
    reversion = get_reversion_edge_adjustment(asset, ohlc_data, momentum_data)
    reversion_confirms = False
    reversion_bonus = 0.0
    
    if reversion.get("has_signal"):
        # Reversion direction: 'bullish' = expecting price to bounce up, 'bearish' = expecting drop
        # Get the reversion direction from the detect function
        reversions = detect_momentum_reversion(ohlc_data, momentum_data)
        for rev in reversions:
            if rev["asset"].lower() == asset.lower():
                rev_dir = rev.get("reversion_dir", "")
                if (side == "yes" and rev_dir == "bullish") or (side == "no" and rev_dir == "bearish"):
                    reversion_confirms = True
                    reversion_bonus = reversion.get("adjustment", 0.01)
                    result["signals"].append(f"Reversion signal: {rev_dir} ({rev.get('confidence', 'medium')})")
                    result["details"]["reversion_signal"] = True
                break
    
    if reversion_confirms:
        result["confirming_signals"] += 1
    
    # Signal 3: DIVERGENCE SIGNAL
    # Bullish divergence = favor YES, Bearish divergence = favor NO
    divergence = get_divergence_edge_adjustment(asset, ohlc_data, momentum_data)
    divergence_confirms = False
    divergence_bonus = 0.0
    
    if divergence.get("has_signal"):
        div_type = divergence.get("divergence_type", "")
        if (side == "yes" and div_type == "bullish") or (side == "no" and div_type == "bearish"):
            divergence_confirms = True
            divergence_bonus = divergence.get("adjustment", 0.01)
            result["signals"].append(f"Divergence: {div_type} ({divergence.get('reason', '')})")
            result["details"]["divergence_signal"] = True
    
    if divergence_confirms:
        result["confirming_signals"] += 1
    
    # Calculate total bonus with synergy effect
    # Base: sum of individual bonuses
    # Synergy: +25% bonus per additional confirming signal above 1
    base_bonus = momentum_bonus + reversion_bonus + divergence_bonus
    
    if result["confirming_signals"] >= 2:
        synergy_multiplier = 1.0 + 0.25 * (result["confirming_signals"] - 1)
        result["total_bonus"] = base_bonus * synergy_multiplier
    else:
        result["total_bonus"] = base_bonus
    
    # Determine confidence level
    if result["confirming_signals"] >= 3:
        result["confidence"] = "very_high"
    elif result["confirming_signals"] == 2:
        result["confidence"] = "high"
    elif result["confirming_signals"] == 1:
        result["confidence"] = "medium"
    else:
        result["confidence"] = "low"
    
    return result


# ============== PROPER PROBABILITY MODEL ==============

def kelly_criterion_check(our_prob: float, price_cents: int) -> float:
    """
    Calculate correct Kelly fraction for a binary market bet.
    
    Kelly formula: f* = (b*p - q) / b
    where:
        b = net odds (payout / stake - 1) = (100/price - 1) for Kalshi
        p = true probability of winning
        q = 1 - p
    
    Returns: Optimal fraction to bet (0 if trade is -EV, never negative)
    
    CRITICAL: This was missing! The old code used fake edges of 60-90%
    without checking if Kelly is positive. Most weather trades had negative Kelly
    because our_prob was <20% while buying at 8-10 cents (which requires >8-10% prob to be +EV).
    """
    if price_cents <= 0 or price_cents >= 100 or our_prob <= 0:
        return 0.0
    
    b = (100 - price_cents) / price_cents  # Net odds ratio
    p = our_prob
    q = 1 - p
    
    f = (b * p - q) / b
    return max(0.0, f)  # Never bet negative Kelly


def get_dynamic_hourly_vol(ohlc_data: list, asset: str) -> float:
    """
    Calculate realized hourly volatility from recent OHLC data.
    
    Falls back to hardcoded defaults if OHLC data is unavailable.
    This replaces the static BTC_HOURLY_VOL/ETH_HOURLY_VOL constants
    for more accurate probability calculations.
    """
    defaults = {"btc": BTC_HOURLY_VOL, "eth": ETH_HOURLY_VOL, "sol": SOL_HOURLY_VOL}
    
    if not ohlc_data or len(ohlc_data) < 10:
        return defaults.get(asset, 0.003)
    
    # Calculate from OHLC: use close-to-close returns
    try:
        rv = calculate_realized_volatility(ohlc_data, hours=24)
        if rv and rv > 0:
            # BUGFIX 2026-07-17: The old code did rv / sqrt(24) which is wrong.
            # calculate_realized_volatility returns the stdev of per-candle log returns.
            # CoinGecko days=7 gives hourly candles → rv IS already hourly vol.
            # CoinGecko days=1 gives 30min candles → rv is 30min vol, need to adjust.
            # For safety, we just use rv directly (it's already per-candle vol).
            # Since most OHLC data is hourly candles, rv ≈ hourly vol.
            hourly_vol = rv  # rv is per-candle vol (hourly candles → hourly vol)
            
            # TIGHTER sanity bounds: don't let dynamic vol go crazy
            # The old 5x max allowed 0.025 hourly vol (2.5%/hr = 12.2%/day = INSANE)
            # which produced ~40% prob for tiny moves, generating phantom edges.
            default_vol = defaults.get(asset, 0.003)
            min_vol = default_vol * 0.5   # Min 50% of default
            max_vol = default_vol * 2.5   # Max 2.5x default (was 5x — way too loose)
            clamped = max(min_vol, min(max_vol, hourly_vol))
            return clamped
    except Exception:
        pass
    
    return defaults.get(asset, 0.003)


def norm_cdf(x):
    """Standard normal CDF approximation (no scipy needed)"""
    # Abramowitz and Stegun approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)


def calculate_prob_above_strike(current_price: float, strike: float, 
                                 minutes_to_expiry: float, hourly_vol: float,
                                 fat_tail: bool = True) -> float:
    """
    Calculate probability that price will be ABOVE strike at expiry.
    Uses log-normal price model (simplified Black-Scholes) with fat-tail adjustment.
    
    P(S_T > K) = N(d2) where:
    d2 = (ln(S/K) + (r - σ²/2)T) / (σ√T)
    
    For short-term crypto, r ≈ 0 (no drift assumption for hourly)
    
    Fat-tail adjustment: crypto returns have excess kurtosis, so we multiply
    sigma by CRYPTO_FAT_TAIL_MULTIPLIER to widen the distribution and produce
    less extreme probabilities (closer to 50% for near-strike prices).
    This creates more edge opportunities compared to the pure lognormal model.
    """
    if minutes_to_expiry <= 0:
        return 1.0 if current_price > strike else 0.0
    
    # Time in hours
    T = minutes_to_expiry / 60.0
    
    # Annualized volatility (hourly vol * sqrt(24*365) for proper scaling)
    # But for short periods, we use hourly vol directly scaled by sqrt(time)
    sigma = hourly_vol * math.sqrt(T)  # Vol for this time period
    
    # Apply fat-tail adjustment for crypto (makes distribution wider = less extreme probabilities)
    if fat_tail:
        sigma = sigma * CRYPTO_FAT_TAIL_MULTIPLIER
    
    if sigma <= 0:
        return 1.0 if current_price > strike else 0.0
    
    # d2 with zero drift (conservative for hourly)
    # d2 = ln(S/K) / σ - σ/2
    log_ratio = math.log(current_price / strike)
    d2 = log_ratio / sigma - sigma / 2
    
    prob_above = norm_cdf(d2)
    
    # BUGFIX 2026-07-17: Tighter bounds [0.05, 0.95].
    # With the old [0.03, 0.97], a prob of 0.03 against a 1¢ contract still
    # generated a (1-0.03) - 0.99 = massive fake edge on the other side.
    # 5% floor means we never claim >95% certainty about any direction
    # in an hourly crypto contract (which is realistic).
    return max(0.05, min(0.95, prob_above))


def get_trend_adjustment(prices_1h: list) -> float:
    """
    Calculate trend adjustment based on recent price action.
    Returns adjustment to probability (-0.1 to +0.1).
    Positive = bullish (price trending up)
    """
    if not prices_1h or len(prices_1h) < 3:
        return 0.0
    
    # Simple: compare current to average of last hour
    current = prices_1h[-1]
    avg = sum(prices_1h) / len(prices_1h)
    
    pct_change = (current - avg) / avg
    
    # Cap adjustment at ±10%
    return max(-0.1, min(0.1, pct_change * 5))


# ============== MULTI-TIMEFRAME MOMENTUM ==============

# Cached OHLC config (T381)
OHLC_CACHE_DIR = Path(__file__).parent.parent / "data" / "ohlc"
OHLC_CACHE_MAX_AGE_HOURS = 24  # Consider cache stale after 24h
COIN_ID_TO_CACHE_FILE = {
    "bitcoin": "btc-ohlc.json",
    "ethereum": "eth-ohlc.json"
}


def load_cached_ohlc(coin_id: str) -> tuple[list, bool]:
    """
    Load OHLC data from local cache file.
    
    Args:
        coin_id: CoinGecko coin id ("bitcoin" or "ethereum")
    
    Returns:
        (ohlc_data, is_fresh): OHLC list in CoinGecko format, whether cache is fresh
    """
    cache_file = COIN_ID_TO_CACHE_FILE.get(coin_id)
    if not cache_file:
        return [], False
    
    cache_path = OHLC_CACHE_DIR / cache_file
    if not cache_path.exists():
        return [], False
    
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        
        # Check freshness
        updated_at = data.get("updated_at", "")
        is_fresh = False
        if updated_at:
            try:
                # Parse ISO timestamp
                updated_time = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - updated_time).total_seconds() / 3600
                is_fresh = age_hours < OHLC_CACHE_MAX_AGE_HOURS
            except (ValueError, TypeError):
                pass
        
        # Convert cache format to CoinGecko API format: [[timestamp, open, high, low, close], ...]
        candles = data.get("candles", [])
        ohlc_data = []
        for c in candles:
            ohlc_data.append([
                c.get("timestamp"),
                c.get("open"),
                c.get("high"),
                c.get("low"),
                c.get("close")
            ])
        
        if ohlc_data:
            symbol = data.get("symbol", coin_id.upper())
            status = "fresh" if is_fresh else "stale"
            print(f"[OHLC] Loaded {len(ohlc_data)} cached {symbol} candles ({status})")
        
        return ohlc_data, is_fresh
    
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"[WARN] Failed to load cached OHLC for {coin_id}: {e}")
        return [], False


def get_crypto_ohlc(coin_id: str = "bitcoin", days: int = 1) -> list:
    """Get crypto OHLC data, preferring local cache to reduce API calls.
    
    Args:
        coin_id: CoinGecko coin id ("bitcoin" or "ethereum")
        days: Number of days (valid: 1, 7, 14, 30, 90, 180, 365, max)
    
    Strategy:
        1. Try local cache first (data/ohlc/*.json)
        2. If cache is fresh (< 24h old), use it
        3. If cache is stale but exists, use it with warning
        4. Fall back to live CoinGecko API if cache unavailable
    """
    # Try cached data first (T381)
    cached_data, is_fresh = load_cached_ohlc(coin_id)
    if cached_data and is_fresh:
        record_api_latency("ohlc_cache_hit", 0)  # Track cache usage
        return cached_data
    
    # Try live API with latency tracking
    start = time.time()
    try:
        # CoinGecko OHLC endpoint - days=1 gives ~48 candles (30min intervals)
        # days=7 gives hourly candles - better for 24h momentum
        valid_days = min(7, max(1, days))  # Clamp to valid values
        resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={valid_days}",
            timeout=10
        )
        latency = (time.time() - start) * 1000
        record_api_latency(f"ext_ohlc_{coin_id[:3]}", latency)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                print(f"[OHLC] Fetched {len(data)} live {coin_id.upper()} candles from CoinGecko ({latency:.0f}ms)")
                return data
        print(f"[WARN] {coin_id.upper()} OHLC API returned {resp.status_code}")
    except Exception as e:
        latency = (time.time() - start) * 1000
        record_api_latency(f"ext_ohlc_{coin_id[:3]}_error", latency)
        print(f"[WARN] CoinGecko OHLC fetch failed: {e}")
    
    # Fall back to stale cache if available
    if cached_data:
        print(f"[WARN] Using stale cached OHLC for {coin_id} (API unavailable)")
        return cached_data
    
    return []


def get_btc_ohlc(days: int = 1) -> list:
    """Get BTC OHLC data (wrapper for backwards compatibility)"""
    return get_crypto_ohlc("bitcoin", days)


def get_eth_ohlc(days: int = 1) -> list:
    """Get ETH OHLC data"""
    return get_crypto_ohlc("ethereum", days)


def calculate_momentum(ohlc_data: list, timeframe: str) -> dict:
    """
    Calculate momentum metrics for a specific timeframe.
    
    Returns:
        direction: -1 (bearish), 0 (neutral), 1 (bullish)
        strength: 0.0 to 1.0
        pct_change: actual % change
    """
    if not ohlc_data or len(ohlc_data) < 4:
        return {"direction": 0, "strength": 0.0, "pct_change": 0.0}
    
    now_ms = time.time() * 1000
    
    # Map timeframe to hours
    hours_map = {"1h": 1, "4h": 4, "24h": 24}
    hours = hours_map.get(timeframe, 1)
    cutoff_ms = now_ms - (hours * 60 * 60 * 1000)
    
    # Get prices in timeframe
    prices_in_range = [c[4] for c in ohlc_data if c[0] >= cutoff_ms]  # Close prices
    
    if len(prices_in_range) < 2:
        # Fall back to last N candles
        n_candles = min(hours, len(ohlc_data))
        prices_in_range = [c[4] for c in ohlc_data[-n_candles:]]
    
    if len(prices_in_range) < 2:
        return {"direction": 0, "strength": 0.0, "pct_change": 0.0}
    
    start_price = prices_in_range[0]
    end_price = prices_in_range[-1]
    
    pct_change = (end_price - start_price) / start_price
    
    # Calculate direction and strength
    if abs(pct_change) < 0.001:  # <0.1% is neutral
        direction = 0
        strength = 0.0
    elif pct_change > 0:
        direction = 1
        strength = min(1.0, abs(pct_change) * 20)  # 5% = full strength
    else:
        direction = -1
        strength = min(1.0, abs(pct_change) * 20)
    
    return {
        "direction": direction,
        "strength": strength,
        "pct_change": pct_change
    }


def get_multi_timeframe_momentum(ohlc_data: list) -> dict:
    """
    Calculate momentum across multiple timeframes.
    
    Returns:
        composite_direction: weighted direction (-1 to 1)
        composite_strength: weighted strength (0 to 1)
        timeframes: individual timeframe data
        alignment: True if all timeframes agree
    """
    result = {
        "composite_direction": 0.0,
        "composite_strength": 0.0,
        "timeframes": {},
        "alignment": False
    }
    
    if not ohlc_data:
        return result
    
    directions = []
    total_weight = 0
    composite_dir = 0.0
    composite_str = 0.0
    
    for tf in MOMENTUM_TIMEFRAMES:
        mom = calculate_momentum(ohlc_data, tf)
        result["timeframes"][tf] = mom
        
        weight = MOMENTUM_WEIGHT.get(tf, 0.33)
        composite_dir += mom["direction"] * mom["strength"] * weight
        composite_str += mom["strength"] * weight
        total_weight += weight
        
        if mom["strength"] > 0.1:  # Only count if meaningful
            directions.append(mom["direction"])
    
    if total_weight > 0:
        result["composite_direction"] = composite_dir / total_weight
        result["composite_strength"] = composite_str / total_weight
    
    # Check alignment - all non-zero directions should match
    if len(directions) >= 2:
        result["alignment"] = len(set(directions)) == 1
    
    return result


def adjust_probability_with_momentum(prob: float, strike: float, current_price: float, 
                                     momentum: dict, side: str) -> float:
    """
    Adjust probability based on multi-timeframe momentum.
    
    For YES (betting price goes UP):
        - Bullish momentum increases our probability
        - Bearish momentum decreases it
    
    For NO (betting price stays DOWN/below):
        - Bearish momentum increases our probability
        - Bullish momentum decreases it
    """
    composite_dir = momentum.get("composite_direction", 0)
    composite_str = momentum.get("composite_strength", 0)
    alignment = momentum.get("alignment", False)
    
    # BUGFIX 2026-07-17: Reduced from ±10%/±5% to ±3%/±1.5%
    # The old ±10% adjustment could push prob_below from 0.05 to 0.15,
    # turning a 4% edge into a 14% fake edge. For hourly contracts,
    # momentum should only fine-tune probability, not dominate it.
    max_adj = 0.03 if alignment else 0.015
    
    # Calculate adjustment
    adjustment = composite_dir * composite_str * max_adj
    
    # Apply adjustment based on side
    if side == "yes":
        # Bullish momentum helps YES bets
        adjusted = prob + adjustment
    else:  # side == "no"
        # Bearish momentum helps NO bets (so we flip the sign)
        adjusted = prob - adjustment
    
    # TRADE-005: Tighter probability bounds to prevent edge inflation
    # Probabilities near 0 or 1 generate huge edges against cheap contracts.
    # Cap at [0.03, 0.97] to keep edges realistic.
    return max(0.03, min(0.97, adjusted))


# ============== MARKET REGIME DETECTION ==============

def detect_market_regime(ohlc_data: list, momentum: dict) -> dict:
    """
    Detect market regime based on price action and momentum.
    
    Regimes:
        - "trending_bullish": Strong uptrend, predictable direction
        - "trending_bearish": Strong downtrend, predictable direction
        - "sideways": No clear trend, range-bound
        - "choppy": High volatility with no trend (hardest to trade)
    
    Returns:
        regime: str (one of above)
        confidence: float (0-1)
        volatility: str ("low", "normal", "high")
        dynamic_min_edge: float (adjusted MIN_EDGE for this regime)
    """
    result = {
        "regime": "sideways",
        "confidence": 0.5,
        "volatility": "normal",
        "dynamic_min_edge": MIN_EDGE,  # default
        "details": {}
    }
    
    if not ohlc_data or len(ohlc_data) < 24:
        return result
    
    # Calculate price changes over different periods
    current_price = ohlc_data[-1][4] if ohlc_data[-1] else None  # Close price
    if not current_price:
        return result
    
    # 4-hour price change (last ~4 hourly candles)
    price_4h_ago = ohlc_data[-4][4] if len(ohlc_data) >= 4 else current_price
    change_4h = (current_price - price_4h_ago) / price_4h_ago if price_4h_ago else 0
    
    # 24-hour price change
    price_24h_ago = ohlc_data[0][4] if len(ohlc_data) >= 24 else current_price
    change_24h = (current_price - price_24h_ago) / price_24h_ago if price_24h_ago else 0
    
    # Calculate volatility (using range-based proxy)
    ranges = []
    for candle in ohlc_data[-24:]:
        if candle and len(candle) >= 4:
            high, low = candle[2], candle[3]
            if low > 0:
                ranges.append((high - low) / low)
    
    avg_range = sum(ranges) / len(ranges) if ranges else 0
    
    # Classify volatility (buckets per T285)
    # very_low: <0.3%, low: 0.3-0.5%, normal: 0.5-1%, high: 1-2%, very_high: >2%
    if avg_range < 0.003:  # < 0.3% avg range
        vol_class = "very_low"
    elif avg_range < 0.005:  # 0.3% - 0.5%
        vol_class = "low"
    elif avg_range < 0.01:  # 0.5% - 1%
        vol_class = "normal"
    elif avg_range < 0.02:  # 1% - 2%
        vol_class = "high"
    else:  # > 2%
        vol_class = "very_high"
    
    result["volatility"] = vol_class
    result["details"]["avg_range"] = avg_range
    result["details"]["change_4h"] = change_4h
    result["details"]["change_24h"] = change_24h
    
    # Get momentum data
    mom_dir = momentum.get("composite_direction", 0) if momentum else 0
    mom_str = momentum.get("composite_strength", 0) if momentum else 0
    mom_aligned = momentum.get("alignment", False) if momentum else False
    
    result["details"]["momentum_dir"] = mom_dir
    result["details"]["momentum_str"] = mom_str
    result["details"]["momentum_aligned"] = mom_aligned
    
    # Determine regime
    abs_4h = abs(change_4h)
    abs_24h = abs(change_24h)
    
    # Strong trend: consistent direction + meaningful price change
    is_bullish = change_4h > 0.005 and change_24h > 0.01 and mom_dir > 0.2
    is_bearish = change_4h < -0.005 and change_24h < -0.01 and mom_dir < -0.2
    
    if is_bullish and mom_aligned:
        result["regime"] = "trending_bullish"
        result["confidence"] = min(0.9, 0.5 + abs_24h * 10 + mom_str * 0.3)
    elif is_bearish and mom_aligned:
        result["regime"] = "trending_bearish"
        result["confidence"] = min(0.9, 0.5 + abs_24h * 10 + mom_str * 0.3)
    elif vol_class == "high" and abs_24h < 0.02:
        # High volatility but no directional move = choppy
        result["regime"] = "choppy"
        result["confidence"] = 0.7
    else:
        result["regime"] = "sideways"
        result["confidence"] = 0.6
    
    # Calculate dynamic MIN_EDGE based on regime
    # Trending markets = easier to predict = lower edge required
    # Choppy/sideways = harder = higher edge required
    if result["regime"] in ("trending_bullish", "trending_bearish"):
        # Lower edge in trending (easier), even lower if high confidence
        base_edge = 0.07  # 7% base for trending
        confidence_adj = (1 - result["confidence"]) * 0.03  # up to 3% more if low confidence
        result["dynamic_min_edge"] = base_edge + confidence_adj
    elif result["regime"] == "choppy":
        # Choppy = higher edge required, but not impossibly high
        result["dynamic_min_edge"] = 0.08  # 8% minimum (was 15% - way too high, blocked all trades)
    else:  # sideways
        # Sideways = moderate edge
        result["dynamic_min_edge"] = 0.06  # 6% minimum (was 12%)
    
    # Volatility adjustment
    if vol_class == "high":
        result["dynamic_min_edge"] += 0.01  # +1% for high vol (was +2%, too conservative)
    elif vol_class == "low":
        result["dynamic_min_edge"] -= 0.01  # -1% for low vol (more predictable)
    
    # VIX integration (T611) - use fear index to adjust edge requirements
    # VIX spikes (>25) often precede crypto volatility, so be more conservative
    vix_data = load_vix_data()
    if vix_data["valid"]:
        result["details"]["vix_current"] = vix_data["vix_current"]
        result["details"]["vix_regime"] = vix_data["vix_regime"]
        result["details"]["vix_size_multiplier"] = vix_data["size_multiplier"]
        
        # Apply VIX edge adjustment
        result["dynamic_min_edge"] += vix_data["edge_adjustment"]
        
        # Store the VIX size multiplier for use in position sizing
        result["vix_size_multiplier"] = vix_data["size_multiplier"]
    else:
        result["vix_size_multiplier"] = 1.0  # No VIX data, normal sizing
    
    # Ensure min edge stays in reasonable bounds
    # In paper mode, use lower floor to generate more data
    min_floor = 0.03 if DRY_RUN else 0.05
    result["dynamic_min_edge"] = max(min_floor, min(0.20, result["dynamic_min_edge"]))
    
    return result


def get_regime_for_asset(asset: str, ohlc_cache: dict, momentum_cache: dict) -> dict:
    """Get market regime for a specific asset (btc or eth)."""
    ohlc = ohlc_cache.get(asset, [])
    momentum = momentum_cache.get(asset, {})
    return detect_market_regime(ohlc, momentum)


# ============== VOLATILITY REBALANCING (T237) ==============

def calculate_realized_volatility(ohlc_data: list, hours: int = 24) -> float:
    """
    Calculate realized volatility from OHLC data.
    
    Uses log returns of hourly close prices.
    
    Args:
        ohlc_data: List of [timestamp, open, high, low, close] from CoinGecko
        hours: Number of hours to use (default 24)
    
    Returns:
        Hourly realized volatility as decimal (e.g., 0.008 = 0.8%)
    """
    if not ohlc_data or len(ohlc_data) < 2:
        return None
    
    # Get last N candles
    candles = ohlc_data[-min(hours, len(ohlc_data)):]
    
    # Extract close prices
    closes = [c[4] for c in candles if len(c) >= 5 and c[4] and c[4] > 0]
    
    if len(closes) < 2:
        return None
    
    # Calculate log returns
    log_returns = []
    for i in range(1, len(closes)):
        if closes[i] > 0 and closes[i-1] > 0:
            log_returns.append(math.log(closes[i] / closes[i-1]))
    
    if not log_returns:
        return None
    
    # Calculate standard deviation of returns (realized volatility)
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
    realized_vol = math.sqrt(variance)
    
    return realized_vol


def get_volatility_advantage(ohlc_data: dict) -> dict:
    """
    Compare realized vs assumed volatility for each asset.
    
    Returns advantage score for each asset:
    - Positive = realized > assumed (favor YES bets, more likely to break strikes)
    - Negative = realized < assumed (favor NO bets, less likely to break strikes)
    - Zero = neutral or no data
    
    Also determines which asset has better trading conditions overall.
    
    Returns:
        {
            "btc": {"realized": 0.006, "assumed": 0.005, "ratio": 1.2, "advantage": "yes"},
            "eth": {"realized": 0.008, "assumed": 0.007, "ratio": 1.14, "advantage": "yes"},
            "preferred_asset": "btc",  # Asset with higher vol ratio
            "vol_bonus": {"btc": 0.01, "eth": 0.005}  # Edge bonus for each asset
        }
    """
    result = {
        "btc": {"realized": None, "assumed": BTC_HOURLY_VOL, "ratio": 1.0, "advantage": "neutral"},
        "eth": {"realized": None, "assumed": ETH_HOURLY_VOL, "ratio": 1.0, "advantage": "neutral"},
        "preferred_asset": None,
        "vol_bonus": {"btc": 0, "eth": 0}
    }
    
    # Calculate realized volatility for each asset
    for asset in ["btc", "eth"]:
        ohlc = ohlc_data.get(asset, [])
        realized = calculate_realized_volatility(ohlc, hours=24)
        
        if realized is not None:
            assumed = BTC_HOURLY_VOL if asset == "btc" else ETH_HOURLY_VOL
            ratio = realized / assumed if assumed > 0 else 1.0
            
            result[asset]["realized"] = realized
            result[asset]["ratio"] = ratio
            
            # Determine advantage direction
            if ratio > 1.15:  # >15% higher realized vol
                result[asset]["advantage"] = "yes"  # Favor YES bets (more movement)
            elif ratio < 0.85:  # >15% lower realized vol
                result[asset]["advantage"] = "no"   # Favor NO bets (less movement)
            else:
                result[asset]["advantage"] = "neutral"
            
            # Calculate edge bonus (max ±2% bonus)
            # Higher ratio = bonus for YES, Lower ratio = bonus for NO
            vol_diff = (ratio - 1.0)  # Positive if realized > assumed
            result["vol_bonus"][asset] = min(0.02, max(-0.02, vol_diff * 0.1))
    
    # Determine preferred asset (one with higher vol ratio = more edge opportunities)
    btc_ratio = result["btc"]["ratio"]
    eth_ratio = result["eth"]["ratio"]
    
    if btc_ratio > eth_ratio * 1.1:  # BTC has 10%+ higher ratio
        result["preferred_asset"] = "btc"
    elif eth_ratio > btc_ratio * 1.1:  # ETH has 10%+ higher ratio
        result["preferred_asset"] = "eth"
    else:
        result["preferred_asset"] = None  # No clear preference
    
    return result


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


def api_request(method: str, path: str, body: dict = None, max_retries: int = 3) -> dict:
    """Make authenticated API request with exponential backoff retry, configurable timeout (T805), and latency tracking"""
    url = f"{BASE_URL}{path}"
    
    # Extract endpoint name for profiling (e.g., "/trade-api/v2/portfolio/balance" -> "balance")
    endpoint_name = path.split("/")[-1].split("?")[0]
    if "orders" in path:
        endpoint_name = "order"
    elif "positions" in path:
        endpoint_name = "positions"
    elif "balance" in path:
        endpoint_name = "balance"
    elif "markets" in path and "{" not in path:
        endpoint_name = "markets_search"
    elif "fills" in path:
        endpoint_name = "fills"
    
    # Get endpoint-specific timeout (T805)
    timeout_seconds = get_endpoint_timeout(endpoint_name)
    
    total_start = time.time()
    
    for attempt in range(max_retries):
        # Generate fresh signature for each attempt (timestamp changes)
        timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
        signature = sign_request(method, path.split('?')[0], timestamp)
        headers = {
            "KALSHI-ACCESS-KEY": API_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }
        
        attempt_start = time.time()
        
        try:
            if method == "GET":
                resp = requests.get(url, headers=headers, timeout=timeout_seconds)
            elif method == "POST":
                resp = requests.post(url, headers=headers, json=body, timeout=timeout_seconds)
            
            attempt_latency = (time.time() - attempt_start) * 1000  # Convert to ms
            
            # Check for server errors (5xx) - retry these
            if resp.status_code >= 500:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + (time.time() % 1)  # Exponential backoff with jitter
                    print(f"[RETRY] API {resp.status_code} error, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    total_latency = (time.time() - total_start) * 1000
                    record_api_latency(f"{endpoint_name}_failed", total_latency)
                    return {"error": f"API error {resp.status_code} after {max_retries} retries"}
            
            # Success - record latency and rate limit
            total_latency = (time.time() - total_start) * 1000
            record_api_latency(endpoint_name, total_latency)
            record_api_call("kalshi", dict(resp.headers))  # Track rate limit headers
            
            # Client errors (4xx) - don't retry, return as-is
            return resp.json()
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + (time.time() % 1)
                print(f"[RETRY] Timeout ({timeout_seconds}s), waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            # Final timeout - record for profiling and alerting (T805)
            total_latency = (time.time() - total_start) * 1000
            record_api_latency(f"{endpoint_name}_timeout", total_latency)
            record_timeout(endpoint_name)  # Track for timeout cluster alerting
            return {"error": f"Timeout after {max_retries} retries ({timeout_seconds}s each)"}
            
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + (time.time() % 1)
                print(f"[RETRY] Connection error, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            total_latency = (time.time() - total_start) * 1000
            record_api_latency(f"{endpoint_name}_conn_error", total_latency)
            return {"error": f"Connection error after {max_retries} retries"}
            
        except Exception as e:
            total_latency = (time.time() - total_start) * 1000
            record_api_latency(f"{endpoint_name}_error", total_latency)
            return {"error": str(e)}
    
    return {"error": "Max retries exceeded"}


def get_balance() -> dict:
    return api_request("GET", "/trade-api/v2/portfolio/balance")


def get_positions() -> list:
    result = api_request("GET", "/trade-api/v2/portfolio/positions")
    return result.get("market_positions", [])


def get_fills(limit=100) -> list:
    result = api_request("GET", f"/trade-api/v2/portfolio/fills?limit={limit}")
    return result.get("fills", [])


def get_market(ticker: str) -> dict:
    result = api_request("GET", f"/trade-api/v2/markets/{ticker}")
    return result.get("market", {})


def search_markets(series: str = None, limit: int = 50) -> list:
    path = f"/trade-api/v2/markets?limit={limit}&status=open"
    if series:
        path += f"&series_ticker={series}"
    result = api_request("GET", path)
    return result.get("markets", [])


def place_order(ticker: str, side: str, count: int, price_cents: int) -> dict:
    body = {
        "ticker": ticker,
        "action": "buy",
        "side": side,
        "count": count,
        "type": "limit",
        "yes_price": price_cents if side == "yes" else 100 - price_cents
    }
    return api_request("POST", "/trade-api/v2/portfolio/orders", body)


def sell_position(ticker: str, side: str, count: int, price_cents: int = None) -> dict:
    """
    Sell/exit an existing position.
    If price_cents is None, use market order (sell at best available price).
    """
    # For market-like execution, we use aggressive limit prices
    if price_cents is None:
        # Sell YES at 1 cent (aggressive), sell NO at 1 cent (which means yes_price = 99)
        price_cents = 1 if side == "yes" else 99
    
    body = {
        "ticker": ticker,
        "action": "sell",
        "side": side,
        "count": count,
        "type": "limit",
        "yes_price": price_cents if side == "yes" else 100 - price_cents
    }
    return api_request("POST", "/trade-api/v2/portfolio/orders", body)


# ============== PORTFOLIO CONCENTRATION MONITORING (T480) ==============

def classify_asset_class(ticker: str) -> str:
    """
    Classify a position into an asset class for concentration tracking.
    
    Returns:
        Asset class: "btc", "eth", "sol", "weather", or "other"
    """
    ticker_upper = ticker.upper()
    if "KXBTC" in ticker_upper:
        return "btc"
    elif "KXETH" in ticker_upper:
        return "eth"
    elif "KXSOL" in ticker_upper:  # T423 - Solana support
        return "sol"
    elif any(city in ticker_upper for city in ["KXHIGH", "KXLOW", "NYC", "MIA", "DEN", "CHI", "LAX"]):
        return "weather"
    else:
        return "other"


def get_correlated_group(ticker: str) -> str:
    """
    Group positions by correlation for concentration limits.
    
    Correlated groups:
    - "crypto": BTC + ETH + SOL (highly correlated, move together)
    - "weather": All weather markets (moderately correlated by region)
    - "other": Everything else
    """
    asset_class = classify_asset_class(ticker)
    if asset_class in ("btc", "eth", "sol"):  # T423 - SOL added to crypto group
        return "crypto"
    elif asset_class == "weather":
        return "weather"
    else:
        return "other"


def calculate_portfolio_concentration(positions: list, portfolio_value_cents: int) -> dict:
    """
    Calculate current portfolio concentration by asset class and correlation group.
    
    Args:
        positions: List of open positions from get_positions()
        portfolio_value_cents: Total portfolio value in cents
        
    Returns:
        Dict with concentration metrics:
        {
            "by_asset_class": {"btc": 0.35, "eth": 0.15, "weather": 0.20, ...},
            "by_correlation_group": {"crypto": 0.50, "weather": 0.20, ...},
            "total_exposure_cents": 1500,
            "position_count": {"btc": 5, "eth": 3, "weather": 2, ...},
            "largest_position_pct": 0.05,
            "largest_asset_class": "btc",
            "largest_correlated_group": "crypto"
        }
    """
    if portfolio_value_cents <= 0 or not positions:
        return {
            "by_asset_class": {},
            "by_correlation_group": {},
            "total_exposure_cents": 0,
            "position_count": {},
            "largest_position_pct": 0,
            "largest_asset_class": None,
            "largest_correlated_group": None
        }
    
    # Calculate exposure by asset class and correlation group
    asset_exposure = defaultdict(int)  # asset_class -> cents
    group_exposure = defaultdict(int)  # correlation_group -> cents
    position_count = defaultdict(int)  # asset_class -> count
    
    largest_position_cents = 0
    
    for pos in positions:
        ticker = pos.get("ticker", "")
        position_qty = pos.get("position", 0)  # Can be negative for NO positions
        
        if position_qty == 0:
            continue
        
        # Get position value in cents
        # For Kalshi: position value = |quantity| * current_price
        # We use market mid-price as proxy (would need orderbook for exact)
        market = get_market(ticker)
        if not market:
            continue
        
        # Get current price - use yes_bid for YES positions, no_bid (= 100 - yes_ask) for NO
        if position_qty > 0:
            # YES position - value at yes_bid
            price_cents = market.get("yes_bid", 50)
        else:
            # NO position - value at no_bid (= 100 - yes_ask)
            price_cents = 100 - market.get("yes_ask", 50)
        
        position_value_cents = abs(position_qty) * price_cents
        
        # Classify and accumulate
        asset_class = classify_asset_class(ticker)
        corr_group = get_correlated_group(ticker)
        
        asset_exposure[asset_class] += position_value_cents
        group_exposure[corr_group] += position_value_cents
        position_count[asset_class] += 1
        
        if position_value_cents > largest_position_cents:
            largest_position_cents = position_value_cents
    
    # Calculate percentages
    total_exposure = sum(asset_exposure.values())
    
    asset_pct = {k: v / portfolio_value_cents for k, v in asset_exposure.items()}
    group_pct = {k: v / portfolio_value_cents for k, v in group_exposure.items()}
    
    # Find largest concentrations
    largest_asset = max(asset_pct.items(), key=lambda x: x[1], default=(None, 0))
    largest_group = max(group_pct.items(), key=lambda x: x[1], default=(None, 0))
    
    return {
        "by_asset_class": dict(asset_pct),
        "by_correlation_group": dict(group_pct),
        "total_exposure_cents": total_exposure,
        "position_count": dict(position_count),
        "largest_position_pct": largest_position_cents / portfolio_value_cents if portfolio_value_cents > 0 else 0,
        "largest_asset_class": largest_asset[0],
        "largest_asset_class_pct": largest_asset[1],
        "largest_correlated_group": largest_group[0],
        "largest_correlated_group_pct": largest_group[1]
    }


def check_concentration_limits(
    positions: list, 
    portfolio_value_cents: int,
    new_trade_asset: str,
    new_trade_value_cents: int
) -> tuple[bool, str, dict]:
    """
    Check if a new trade would exceed portfolio concentration limits.
    
    Args:
        positions: Current open positions
        portfolio_value_cents: Total portfolio value
        new_trade_asset: Asset type of proposed trade ("btc", "eth", "weather")
        new_trade_value_cents: Value of proposed trade
        
    Returns:
        Tuple of (can_trade, reason, concentration_metrics)
        - can_trade: True if trade is allowed, False if blocked
        - reason: Explanation string
        - concentration_metrics: Current concentration data for logging
    """
    concentration = calculate_portfolio_concentration(positions, portfolio_value_cents)
    
    # Calculate what concentration would be after this trade
    new_asset_class = new_trade_asset.lower()
    new_corr_group = "crypto" if new_asset_class in ("btc", "eth") else new_asset_class
    
    current_asset_pct = concentration["by_asset_class"].get(new_asset_class, 0)
    current_group_pct = concentration["by_correlation_group"].get(new_corr_group, 0)
    
    if portfolio_value_cents > 0:
        new_trade_pct = new_trade_value_cents / portfolio_value_cents
        projected_asset_pct = current_asset_pct + new_trade_pct
        projected_group_pct = current_group_pct + new_trade_pct
    else:
        projected_asset_pct = 1.0  # New trade would be 100% of portfolio
        projected_group_pct = 1.0
    
    # Check concentration limits
    
    # 1. Check asset class limit (max 50%)
    if projected_asset_pct > CONCENTRATION_MAX_ASSET_CLASS_PCT:
        reason = f"Would exceed {CONCENTRATION_MAX_ASSET_CLASS_PCT*100:.0f}% asset class limit " \
                 f"({new_asset_class.upper()}: {current_asset_pct*100:.1f}% → {projected_asset_pct*100:.1f}%)"
        return False, reason, concentration
    
    # 2. Check correlated group limit (dynamic based on BTC/ETH correlation - T483)
    if new_corr_group == "crypto":
        dynamic_limit, corr_data = get_dynamic_crypto_correlation_limit()
        if corr_data.get("status") == "success":
            print(f"   📊 Dynamic crypto limit: {dynamic_limit*100:.0f}% (correlation: {corr_data.get('correlation', 0):.3f})")
    else:
        dynamic_limit = CONCENTRATION_MAX_CORRELATED_PCT
        corr_data = {}
    
    if projected_group_pct > dynamic_limit:
        reason = f"Would exceed {dynamic_limit*100:.0f}% correlation group limit " \
                 f"({new_corr_group}: {current_group_pct*100:.1f}% → {projected_group_pct*100:.1f}%)"
        return False, reason, concentration
    
    # 3. Check warning threshold (40%) - allowed but warn
    if projected_asset_pct > CONCENTRATION_WARN_PCT:
        write_concentration_alert(new_asset_class, projected_asset_pct, concentration)
        reason = f"WARN: Approaching concentration limit ({new_asset_class.upper()}: {projected_asset_pct*100:.1f}%)"
        # Still allow the trade, just with warning
        return True, reason, concentration
    
    return True, "OK", concentration


def write_concentration_alert(asset_class: str, concentration_pct: float, metrics: dict):
    """Write concentration warning alert for heartbeat pickup."""
    alert_path = Path(__file__).parent / CONCENTRATION_ALERT_FILE.split("/")[-1]
    
    # Check cooldown
    if alert_path.exists():
        last_alert = alert_path.stat().st_mtime
        if time.time() - last_alert < CONCENTRATION_ALERT_COOLDOWN:
            return  # Still in cooldown
    
    alert_data = {
        "type": "concentration_warning",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "asset_class": asset_class,
        "concentration_pct": concentration_pct,
        "threshold": CONCENTRATION_WARN_PCT,
        "message": f"⚠️ Portfolio concentration warning: {asset_class.upper()} at {concentration_pct*100:.1f}% " \
                   f"(warn threshold: {CONCENTRATION_WARN_PCT*100:.0f}%, max: {CONCENTRATION_MAX_ASSET_CLASS_PCT*100:.0f}%)",
        "metrics": metrics
    }
    
    with open(alert_path, "w") as f:
        json.dump(alert_data, f, indent=2)
    
    print(f"⚠️ Concentration alert written: {asset_class.upper()} at {concentration_pct*100:.1f}%")


def get_dynamic_crypto_correlation_limit() -> tuple[float, dict]:
    """
    Get dynamic crypto correlation limit based on BTC/ETH correlation (T483).
    
    Reads data from btc-eth-correlation.py output file.
    
    Returns:
        Tuple of (limit_pct, correlation_data)
        - limit_pct: Max percentage for combined crypto exposure (0.30-0.60)
        - correlation_data: Full correlation analysis for logging
    """
    correlation_file = Path(__file__).parent.parent / CORRELATION_DATA_FILE
    
    default_limit = CONCENTRATION_MAX_CORRELATED_PCT
    
    if not correlation_file.exists():
        return default_limit, {"status": "no_data", "reason": "correlation file not found"}
    
    try:
        with open(correlation_file, "r") as f:
            data = json.load(f)
        
        # Check if data is stale (>24h old)
        generated_at = data.get("generated_at", "")
        if generated_at:
            try:
                gen_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - gen_dt).total_seconds() / 3600
                if age_hours > 24:
                    return default_limit, {"status": "stale", "reason": f"correlation data is {age_hours:.0f}h old"}
            except:
                pass
        
        if data.get("status") != "success":
            return default_limit, {"status": "error", "reason": data.get("message", "unknown error")}
        
        # Get adjustment from correlation analysis
        adjustment = data.get("adjustment", {})
        crypto_limit = adjustment.get("crypto_group_limit", 30) / 100.0  # Convert to decimal
        
        return crypto_limit, {
            "status": "success",
            "correlation": data.get("correlation", {}).get("value", 0),
            "interpretation": data.get("correlation", {}).get("interpretation", "unknown"),
            "crypto_limit_pct": crypto_limit * 100,
            "risk_level": adjustment.get("risk_level", "unknown"),
            "reason": adjustment.get("adjustment_reason", "")
        }
        
    except Exception as e:
        return default_limit, {"status": "error", "reason": str(e)}


# T482: Concentration history tracking
CONCENTRATION_HISTORY_FILE = "data/trading/concentration-history.jsonl"


def log_concentration_snapshot(concentration: dict, portfolio_value_cents: int):
    """
    Log concentration snapshot to history file for dashboard tracking (T482, T764).
    
    Appends one line per cycle with timestamp and concentration metrics.
    Always logs, even when portfolio is empty (0% concentration) to track state over time.
    """
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "portfolio_value_cents": portfolio_value_cents,
        "by_asset_class": concentration.get("by_asset_class", {}),
        "by_correlation_group": concentration.get("by_correlation_group", {}),
        "total_exposure_cents": concentration.get("total_exposure_cents", 0),
        "position_count": concentration.get("position_count", 0),
        "largest_asset_class": concentration.get("largest_asset_class", "none"),
        "largest_asset_class_pct": concentration.get("largest_asset_class_pct", 0),
        "largest_correlated_group": concentration.get("largest_correlated_group", "none"),
        "largest_correlated_group_pct": concentration.get("largest_correlated_group_pct", 0),
        "exposure_pct": concentration["total_exposure_cents"] / portfolio_value_cents if portfolio_value_cents > 0 and concentration.get("total_exposure_cents", 0) > 0 else 0
    }
    
    # Ensure directory exists
    history_path = Path(CONCENTRATION_HISTORY_FILE)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Append to JSONL
    with open(history_path, "a") as f:
        f.write(json.dumps(snapshot) + "\n")


def print_concentration_summary(concentration: dict):
    """Print a human-readable concentration summary."""
    if not concentration["by_asset_class"]:
        print("   📊 Portfolio: No open positions")
        return
    
    print("   📊 Portfolio Concentration:")
    
    # By asset class
    for asset, pct in sorted(concentration["by_asset_class"].items(), key=lambda x: -x[1]):
        count = concentration["position_count"].get(asset, 0)
        bar_len = int(pct * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        status = "⚠️" if pct > CONCENTRATION_WARN_PCT else "✓" if pct < 0.25 else ""
        print(f"      {asset.upper():8} [{bar}] {pct*100:5.1f}% ({count} pos) {status}")
    
    # Correlation groups
    if concentration["by_correlation_group"]:
        largest_group = concentration["largest_correlated_group"]
        largest_pct = concentration["largest_correlated_group_pct"]
        if largest_pct > CONCENTRATION_WARN_PCT:
            print(f"      ⚠️ Correlated exposure: {largest_group} at {largest_pct*100:.1f}%")


# ============== REBALANCING SUGGESTIONS (T481) ==============

def get_rebalancing_suggestions(
    positions: list, 
    portfolio_value_cents: int, 
    concentration: dict,
    target_pct: float = None
) -> dict:
    """
    Generate suggestions for reducing portfolio concentration.
    
    When concentration exceeds warning threshold, this function suggests
    which positions to reduce and how much capital that would free up.
    
    Args:
        positions: List of open positions
        portfolio_value_cents: Total portfolio value in cents
        concentration: Current concentration metrics from calculate_portfolio_concentration()
        target_pct: Target concentration percentage (default: CONCENTRATION_WARN_PCT - 0.05)
        
    Returns:
        Dict with rebalancing suggestions:
        {
            "needs_rebalancing": True/False,
            "over_concentrated_assets": ["btc", "crypto"],
            "suggestions": [
                {
                    "ticker": "KXBTC-...",
                    "asset_class": "btc",
                    "current_value_cents": 500,
                    "suggested_reduction_qty": 5,
                    "reduction_value_cents": 250,
                    "reason": "oldest",  # or "lowest_edge" or "largest"
                    "position_age_hours": 48.5,
                    "current_qty": 10,
                    "side": "yes"
                }
            ],
            "total_freed_capital_cents": 500,
            "projected_concentration": {"btc": 0.25, ...}
        }
    """
    if target_pct is None:
        target_pct = CONCENTRATION_WARN_PCT - 0.05  # 35% if warn is 40%
    
    result = {
        "needs_rebalancing": False,
        "over_concentrated_assets": [],
        "suggestions": [],
        "total_freed_capital_cents": 0,
        "projected_concentration": {}
    }
    
    if not positions or portfolio_value_cents <= 0:
        return result
    
    # Find over-concentrated assets/groups
    over_limit_assets = []
    for asset, pct in concentration.get("by_asset_class", {}).items():
        if pct > CONCENTRATION_WARN_PCT:
            over_limit_assets.append(("asset", asset, pct))
    
    for group, pct in concentration.get("by_correlation_group", {}).items():
        if pct > CONCENTRATION_WARN_PCT:
            over_limit_assets.append(("group", group, pct))
    
    if not over_limit_assets:
        return result
    
    result["needs_rebalancing"] = True
    result["over_concentrated_assets"] = [name for _, name, _ in over_limit_assets]
    
    # Get fills for position age calculation
    try:
        fills = get_fills(limit=200)
    except Exception:
        fills = []
    
    # Build position age map (ticker -> oldest fill timestamp)
    position_ages = {}
    for fill in fills:
        ticker = fill.get("ticker", "")
        if fill.get("action") == "buy":  # Entry fill
            ts = fill.get("created_time", "")
            if ts and (ticker not in position_ages or ts < position_ages[ticker]):
                position_ages[ticker] = ts
    
    # Collect positions in over-concentrated assets
    rebalance_candidates = []
    
    for pos in positions:
        ticker = pos.get("ticker", "")
        position_qty = pos.get("position", 0)
        
        if position_qty == 0:
            continue
        
        asset_class = classify_asset_class(ticker)
        corr_group = get_correlated_group(ticker)
        
        # Check if this position is in an over-concentrated asset/group
        is_over = any(
            (kind == "asset" and name == asset_class) or 
            (kind == "group" and name == corr_group)
            for kind, name, _ in over_limit_assets
        )
        
        if not is_over:
            continue
        
        # Get market info
        try:
            market = get_market(ticker)
            if not market:
                continue
        except Exception:
            continue
        
        # Calculate position value
        if position_qty > 0:
            side = "yes"
            price_cents = market.get("yes_bid", 50)
        else:
            side = "no"
            price_cents = 100 - market.get("yes_ask", 50)
        
        position_value_cents = abs(position_qty) * price_cents
        
        # Calculate position age in hours
        age_hours = 0
        if ticker in position_ages:
            try:
                from datetime import datetime
                entry_time = datetime.fromisoformat(position_ages[ticker].replace("Z", "+00:00"))
                now = datetime.now(entry_time.tzinfo)
                age_hours = (now - entry_time).total_seconds() / 3600
            except Exception:
                pass
        
        rebalance_candidates.append({
            "ticker": ticker,
            "asset_class": asset_class,
            "corr_group": corr_group,
            "current_value_cents": position_value_cents,
            "current_qty": abs(position_qty),
            "side": side,
            "price_cents": price_cents,
            "position_age_hours": age_hours,
            "market_title": market.get("title", ticker)[:50]
        })
    
    if not rebalance_candidates:
        return result
    
    # For each over-concentrated asset, calculate reduction needed
    for kind, asset_name, current_pct in over_limit_assets:
        # How much do we need to reduce?
        excess_pct = current_pct - target_pct
        if excess_pct <= 0:
            continue
        
        excess_cents = int(excess_pct * portfolio_value_cents)
        
        # Filter candidates for this asset
        if kind == "asset":
            candidates = [c for c in rebalance_candidates if c["asset_class"] == asset_name]
        else:
            candidates = [c for c in rebalance_candidates if c["corr_group"] == asset_name]
        
        if not candidates:
            continue
        
        # Sort by multiple criteria and pick best to reduce
        # Priority: 1. Oldest (reduce locked capital), 2. Largest (biggest impact), 3. Lowest value per contract
        
        # Strategy 1: Oldest positions (free up locked capital)
        oldest = sorted(candidates, key=lambda x: -x["position_age_hours"])
        
        # Strategy 2: Largest positions (biggest single reduction)
        largest = sorted(candidates, key=lambda x: -x["current_value_cents"])
        
        # Generate suggestions - prefer oldest, then largest
        remaining_to_reduce = excess_cents
        seen_tickers = set()
        
        for priority, source, reason in [(1, oldest, "oldest"), (2, largest, "largest")]:
            if remaining_to_reduce <= 0:
                break
            
            for candidate in source:
                if remaining_to_reduce <= 0:
                    break
                if candidate["ticker"] in seen_tickers:
                    continue
                
                seen_tickers.add(candidate["ticker"])
                
                # Calculate how many contracts to sell
                max_reduction = min(candidate["current_value_cents"], remaining_to_reduce)
                reduction_qty = max(1, int(max_reduction / candidate["price_cents"]))
                reduction_qty = min(reduction_qty, candidate["current_qty"])  # Don't sell more than we have
                reduction_value = reduction_qty * candidate["price_cents"]
                
                suggestion = {
                    "ticker": candidate["ticker"],
                    "market_title": candidate["market_title"],
                    "asset_class": candidate["asset_class"],
                    "current_value_cents": candidate["current_value_cents"],
                    "current_qty": candidate["current_qty"],
                    "suggested_reduction_qty": reduction_qty,
                    "reduction_value_cents": reduction_value,
                    "reason": reason,
                    "position_age_hours": candidate["position_age_hours"],
                    "side": candidate["side"]
                }
                
                result["suggestions"].append(suggestion)
                result["total_freed_capital_cents"] += reduction_value
                remaining_to_reduce -= reduction_value
    
    # Calculate projected concentration after rebalancing
    if result["suggestions"]:
        total_reduction = sum(s["reduction_value_cents"] for s in result["suggestions"])
        new_portfolio = portfolio_value_cents + total_reduction  # Freed capital adds to available
        
        for asset, pct in concentration.get("by_asset_class", {}).items():
            current_value = int(pct * portfolio_value_cents)
            reduction = sum(
                s["reduction_value_cents"] 
                for s in result["suggestions"] 
                if s["asset_class"] == asset
            )
            new_value = current_value - reduction
            result["projected_concentration"][asset] = new_value / new_portfolio if new_portfolio > 0 else 0
    
    return result


def print_rebalancing_suggestions(suggestions: dict):
    """Print human-readable rebalancing suggestions."""
    if not suggestions.get("needs_rebalancing"):
        return
    
    print("\n   🔄 REBALANCING SUGGESTIONS:")
    print(f"      Over-concentrated: {', '.join(suggestions['over_concentrated_assets']).upper()}")
    
    if not suggestions.get("suggestions"):
        print("      No specific positions to reduce (positions may have settled)")
        return
    
    for i, s in enumerate(suggestions["suggestions"][:5], 1):  # Show top 5
        age_str = f"{s['position_age_hours']:.0f}h old" if s['position_age_hours'] > 0 else "recent"
        print(f"      {i}. [{s['reason'].upper():7}] Sell {s['suggested_reduction_qty']}x {s['side'].upper()} on {s['ticker'][:25]}")
        print(f"         → Free ${s['reduction_value_cents']/100:.2f} ({age_str})")
    
    total_free = suggestions["total_freed_capital_cents"] / 100
    print(f"\n      💰 Total freed capital: ${total_free:.2f}")
    
    if suggestions.get("projected_concentration"):
        print("      📊 Projected after rebalance:")
        for asset, pct in sorted(suggestions["projected_concentration"].items(), key=lambda x: -x[1]):
            status = "✓" if pct < CONCENTRATION_WARN_PCT else "⚠️"
            print(f"         {asset.upper()}: {pct*100:.1f}% {status}")


# ============== AUTO-REBALANCING (T816) ==============

def execute_auto_rebalancing(
    positions: list,
    portfolio_value_cents: int,
    concentration: dict,
    dry_run: bool = None
) -> dict:
    """
    Automatically execute rebalancing when concentration exceeds threshold.
    
    This is the automated version of get_rebalancing_suggestions() that actually
    executes the sells instead of just printing them.
    
    Args:
        positions: List of open positions
        portfolio_value_cents: Total portfolio value in cents
        concentration: Current concentration metrics
        dry_run: Override AUTO_REBALANCE_DRY_RUN setting
        
    Returns:
        Dict with execution results:
        {
            "executed": True/False,
            "dry_run": True/False,
            "trades": [...],
            "total_freed_cents": int,
            "errors": [...]
        }
    """
    if dry_run is None:
        dry_run = AUTO_REBALANCE_DRY_RUN
    
    result = {
        "executed": False,
        "dry_run": dry_run,
        "trades": [],
        "total_freed_cents": 0,
        "errors": []
    }
    
    # Check if auto-rebalancing is enabled
    if not AUTO_REBALANCE_ENABLED:
        return result
    
    # Check if any asset exceeds the auto-rebalance threshold
    needs_rebalancing = False
    over_threshold_assets = []
    
    for asset, pct in concentration.get("by_asset_class", {}).items():
        if pct > AUTO_REBALANCE_THRESHOLD:
            needs_rebalancing = True
            over_threshold_assets.append((asset, pct))
    
    for group, pct in concentration.get("by_correlation_group", {}).items():
        if pct > AUTO_REBALANCE_THRESHOLD:
            needs_rebalancing = True
            over_threshold_assets.append((group, pct))
    
    if not needs_rebalancing:
        return result
    
    # Get rebalancing suggestions
    suggestions = get_rebalancing_suggestions(
        positions, 
        portfolio_value_cents, 
        concentration,
        target_pct=AUTO_REBALANCE_THRESHOLD - 0.10  # Target 10% below threshold (35% if threshold is 45%)
    )
    
    if not suggestions.get("needs_rebalancing") or not suggestions.get("suggestions"):
        return result
    
    # Build alert message BEFORE executing
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    over_str = ", ".join([f"{a[0].upper()}: {a[1]*100:.1f}%" for a in over_threshold_assets])
    
    alert_lines = [
        f"🔄 AUTO-REBALANCE {'(DRY RUN)' if dry_run else 'EXECUTING'}",
        f"⏰ Time: {timestamp}",
        f"⚠️ Over threshold: {over_str}",
        f"📊 Threshold: {AUTO_REBALANCE_THRESHOLD*100:.0f}%",
        "",
        "📋 Planned sells:"
    ]
    
    for s in suggestions["suggestions"][:5]:  # Show top 5
        alert_lines.append(
            f"  • {s['suggested_reduction_qty']}x {s['side'].upper()} on {s['ticker'][:30]} "
            f"(~${s['reduction_value_cents']/100:.2f})"
        )
    
    total_free = suggestions["total_freed_capital_cents"] / 100
    alert_lines.append("")
    alert_lines.append(f"💰 Total to free: ${total_free:.2f}")
    
    if suggestions.get("projected_concentration"):
        alert_lines.append("📈 Projected after:")
        for asset, pct in sorted(suggestions["projected_concentration"].items(), key=lambda x: -x[1]):
            alert_lines.append(f"  • {asset.upper()}: {pct*100:.1f}%")
    
    alert_message = "\n".join(alert_lines)
    
    # Write alert for heartbeat/Telegram pickup
    try:
        alert_path = Path(__file__).parent / "kalshi-rebalance.alert"
        with open(alert_path, "w") as f:
            f.write(alert_message)
        print(f"   📤 Rebalance alert written to {alert_path}")
    except Exception as e:
        print(f"   ⚠️ Failed to write rebalance alert: {e}")
    
    # Log to rebalance log file
    log_entry = {
        "timestamp": timestamp,
        "dry_run": dry_run,
        "over_threshold_assets": [(a, p) for a, p in over_threshold_assets],
        "suggestions": suggestions["suggestions"],
        "total_to_free_cents": suggestions["total_freed_capital_cents"],
        "projected_concentration": suggestions.get("projected_concentration", {})
    }
    
    try:
        log_path = Path(__file__).parent / "kalshi-rebalance.log"
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"   ⚠️ Failed to write rebalance log: {e}")
    
    # If dry run, don't execute
    if dry_run:
        print("\n   🔄 AUTO-REBALANCE (DRY RUN - no trades executed)")
        print_rebalancing_suggestions(suggestions)
        print(f"   ℹ️  Set AUTO_REBALANCE_DRY_RUN=false to execute")
        return result
    
    # Execute the sells
    print("\n   🔄 AUTO-REBALANCE EXECUTING...")
    result["executed"] = True
    
    for suggestion in suggestions["suggestions"]:
        ticker = suggestion["ticker"]
        side = suggestion["side"]
        qty = suggestion["suggested_reduction_qty"]
        
        print(f"      Selling {qty}x {side.upper()} on {ticker[:30]}...", end=" ")
        
        try:
            # Execute sell at market (aggressive price)
            sell_result = sell_position(ticker, side, qty)
            
            if sell_result.get("order"):
                order_id = sell_result["order"].get("order_id", "unknown")
                result["trades"].append({
                    "ticker": ticker,
                    "side": side,
                    "qty": qty,
                    "order_id": order_id,
                    "expected_value_cents": suggestion["reduction_value_cents"],
                    "reason": suggestion["reason"],
                    "status": "success"
                })
                result["total_freed_cents"] += suggestion["reduction_value_cents"]
                print(f"✓ Order {order_id}")
            else:
                error_msg = sell_result.get("error", "Unknown error")
                result["errors"].append({
                    "ticker": ticker,
                    "error": error_msg
                })
                print(f"✗ {error_msg}")
                
        except Exception as e:
            result["errors"].append({
                "ticker": ticker,
                "error": str(e)
            })
            print(f"✗ {e}")
    
    # Update log entry with execution results
    log_entry["executed"] = True
    log_entry["trades"] = result["trades"]
    log_entry["errors"] = result["errors"]
    log_entry["total_freed_cents"] = result["total_freed_cents"]
    
    try:
        log_path = Path(__file__).parent / "kalshi-rebalance.log"
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
    
    # Summary
    success_count = len(result["trades"])
    error_count = len(result["errors"])
    freed = result["total_freed_cents"] / 100
    
    print(f"\n   📊 Rebalance complete: {success_count} sells, {error_count} errors, ~${freed:.2f} freed")
    
    return result


def check_and_auto_rebalance(positions: list, portfolio_value_cents: int) -> dict:
    """
    Convenience function to check concentration and auto-rebalance if needed.
    Call this from the main trading cycle.
    
    Returns:
        Result dict from execute_auto_rebalancing or empty dict if not needed
    """
    if not AUTO_REBALANCE_ENABLED:
        return {}
    
    if not positions or portfolio_value_cents <= 0:
        return {}
    
    # Calculate current concentration
    concentration = calculate_portfolio_concentration(positions, portfolio_value_cents)
    
    # Check if any asset exceeds threshold
    max_concentration = 0
    for pct in concentration.get("by_asset_class", {}).values():
        max_concentration = max(max_concentration, pct)
    for pct in concentration.get("by_correlation_group", {}).values():
        max_concentration = max(max_concentration, pct)
    
    if max_concentration <= AUTO_REBALANCE_THRESHOLD:
        return {}
    
    # Execute auto-rebalancing
    return execute_auto_rebalancing(positions, portfolio_value_cents, concentration)


# ============== STOP-LOSS MONITORING ==============

# Stop-loss parameters (configurable via environment)
STOP_LOSS_THRESHOLD = float(os.getenv("STOP_LOSS_THRESHOLD", "0.50"))  # Exit if position value drops X% (default 50%)
MIN_STOP_LOSS_VALUE = int(os.getenv("MIN_STOP_LOSS_VALUE", "5"))  # Don't exit positions worth less than X cents
STOP_LOSS_LOG_FILE = "scripts/kalshi-stop-loss.log"
STOP_LOSS_ALERT_FILE = Path(__file__).parent / "kalshi-stop-loss.alert"


def check_stop_losses(positions: list, prices: dict) -> list:
    """
    Check all open positions for stop-loss triggers.
    
    For Kalshi binary options:
    - We paid X cents to enter
    - Current market value is Y cents (current bid)
    - If Y < X * (1 - STOP_LOSS_THRESHOLD), exit
    
    Returns list of positions that should be exited.
    """
    positions_to_exit = []
    
    if not positions:
        return positions_to_exit
    
    for pos in positions:
        ticker = pos.get("ticker", "")
        position = pos.get("position", 0)  # Positive = YES, Negative = NO
        
        if position == 0:
            continue
        
        # Get current market info
        market = get_market(ticker)
        if not market:
            continue
        
        # Determine our side and current market value
        if position > 0:
            side = "yes"
            contracts = position
            # Current value is what we can sell for (yes_bid)
            current_value = market.get("yes_bid", 0)
        else:
            side = "no"
            contracts = abs(position)
            # For NO positions, value is 100 - yes_ask (what we can sell NO for)
            current_value = 100 - market.get("yes_ask", 100)
        
        # Get our entry price from trade log
        entry_price = get_entry_price_for_position(ticker, side)
        if entry_price is None:
            continue  # Can't find entry, skip
        
        # Calculate if we should exit
        # Stop-loss: if current value is below threshold of entry
        stop_loss_price = entry_price * (1 - STOP_LOSS_THRESHOLD)
        
        if current_value < stop_loss_price and current_value >= MIN_STOP_LOSS_VALUE:
            loss_pct = (entry_price - current_value) / entry_price * 100
            positions_to_exit.append({
                "ticker": ticker,
                "side": side,
                "contracts": contracts,
                "entry_price": entry_price,
                "current_value": current_value,
                "loss_pct": loss_pct,
                "stop_loss_trigger": stop_loss_price
            })
    
    return positions_to_exit


def get_entry_price_for_position(ticker: str, side: str) -> float:
    """
    Look up the entry price for a position from trade log.
    Returns average entry price if multiple entries, or None if not found.
    """
    log_path = Path(TRADE_LOG_FILE)
    if not log_path.exists():
        return None
    
    total_cost = 0
    total_contracts = 0
    
    with open(log_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if (entry.get("type") == "trade" and 
                    entry.get("ticker") == ticker and 
                    entry.get("side") == side and
                    entry.get("result_status") == "pending"):
                    total_cost += entry.get("price_cents", 0) * entry.get("contracts", 0)
                    total_contracts += entry.get("contracts", 0)
            except:
                continue
    
    if total_contracts > 0:
        return total_cost / total_contracts
    return None


def execute_stop_losses(stop_loss_positions: list) -> int:
    """
    Execute stop-loss orders for positions that triggered.
    Returns number of positions exited.
    """
    exited = 0
    
    for pos in stop_loss_positions:
        ticker = pos["ticker"]
        side = pos["side"]
        contracts = pos["contracts"]
        entry = pos["entry_price"]
        current = pos["current_value"]
        loss_pct = pos["loss_pct"]
        
        print(f"\n⚠️ STOP-LOSS TRIGGERED: {ticker}")
        print(f"   Side: {side.upper()} | Contracts: {contracts}")
        print(f"   Entry: {entry:.0f}¢ → Current: {current:.0f}¢ ({loss_pct:.1f}% loss)")
        
        # Execute sell order (with latency tracking)
        order_start = time.time()
        result = sell_position(ticker, side, contracts)
        order_end = time.time()
        latency_ms = int((order_end - order_start) * 1000)
        
        if "error" in result:
            print(f"   ❌ Failed to exit: {result['error']}")
            # Log failed stop-loss execution (T329)
            log_execution(
                ticker=ticker, side=side, count=contracts, price_cents=int(current),
                status="error", latency_ms=latency_ms, error=result['error']
            )
            continue
        
        order = result.get("order", {})
        order_status = order.get("status", "unknown")
        if order_status in ["executed", "pending"]:
            print(f"   ✅ Stop-loss order placed (status: {order_status}) ⏱️ {latency_ms}ms")
            exited += 1
            
            # Log execution success (T329)
            log_execution(
                ticker=ticker, side=side, count=contracts, price_cents=int(current),
                status=order_status, latency_ms=latency_ms,
                order_id=order.get("order_id", order.get("id"))
            )
            
            # Log the stop-loss (with latency)
            log_stop_loss({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "stop_loss",
                "ticker": ticker,
                "side": side,
                "contracts": contracts,
                "entry_price": entry,
                "exit_price": current,
                "loss_pct": loss_pct,
                "order_status": order_status,
                "latency_ms": latency_ms
            })
            
            # Write Telegram alert file for heartbeat to pick up
            write_stop_loss_alert(ticker, side, contracts, entry, current, loss_pct)
    
    return exited


def log_stop_loss(data: dict):
    """Log stop-loss event to file"""
    log_path = Path(STOP_LOSS_LOG_FILE)
    with open(log_path, "a") as f:
        f.write(json.dumps(data) + "\n")
    
    # Also log to main trade file
    log_trade(data)


def write_stop_loss_alert(ticker: str, side: str, contracts: int, 
                          entry_price: float, exit_price: float, loss_pct: float):
    """
    Write stop-loss alert file for heartbeat to pick up and send to Telegram.
    Includes ticker, position info, and loss amount.
    """
    # Calculate loss in cents
    loss_cents = (entry_price - exit_price) * contracts
    
    # Determine asset from ticker
    asset = "ETH" if "KXETHD" in ticker else "BTC"
    
    alert_content = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "stop_loss",
        "ticker": ticker,
        "asset": asset,
        "side": side.upper(),
        "contracts": contracts,
        "entry_price_cents": entry_price,
        "exit_price_cents": exit_price,
        "loss_pct": round(loss_pct, 1),
        "loss_cents": round(loss_cents, 2),
        "message": f"🚨 STOP-LOSS TRIGGERED\n\n"
                   f"Ticker: {ticker}\n"
                   f"Side: {side.upper()} | Contracts: {contracts}\n"
                   f"Entry: {entry_price:.0f}¢ → Exit: {exit_price:.0f}¢\n"
                   f"Loss: {loss_pct:.1f}% (${loss_cents/100:.2f})"
    }
    
    with open(STOP_LOSS_ALERT_FILE, "w") as f:
        json.dump(alert_content, f, indent=2)
    
    print(f"📢 Alert file written: {STOP_LOSS_ALERT_FILE}")


# ============== EXTERNAL DATA ==============

def get_prices_coingecko() -> dict:
    """Get BTC/ETH/SOL prices from CoinGecko with latency tracking and caching"""
    # Check cache first (T427)
    cached = get_cached_response("prices_coingecko")
    if cached:
        return cached
    
    start = time.time()
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd",  # T423: Added SOL
            timeout=5
        )
        latency = (time.time() - start) * 1000
        record_api_latency("ext_coingecko", latency)
        record_api_call("coingecko", dict(resp.headers))  # Track rate limit
        if resp.status_code == 200:
            data = resp.json()
            result = {
                "btc": data["bitcoin"]["usd"],
                "eth": data["ethereum"]["usd"],
                "sol": data.get("solana", {}).get("usd"),  # T423: SOL price
                "source": "coingecko"
            }
            set_cached_response("prices_coingecko", result)  # Cache the result
            return result
    except Exception as e:
        latency = (time.time() - start) * 1000
        record_api_latency("ext_coingecko_error", latency)
        print(f"[PRICE] CoinGecko error: {e}")
    return None


def get_prices_binance() -> dict:
    """Get BTC/ETH/SOL prices from Binance with latency tracking and caching"""
    # Check cache first (T427)
    cached = get_cached_response("prices_binance")
    if cached:
        return cached
    
    start = time.time()
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/price?symbols=[\"BTCUSDT\",\"ETHUSDT\",\"SOLUSDT\"]",  # T423: Added SOL
            timeout=5
        )
        latency = (time.time() - start) * 1000
        record_api_latency("ext_binance", latency)
        record_api_call("binance", dict(resp.headers))  # Track rate limit
        if resp.status_code == 200:
            data = resp.json()
            prices = {"source": "binance"}
            for item in data:
                if item["symbol"] == "BTCUSDT":
                    prices["btc"] = float(item["price"])
                elif item["symbol"] == "ETHUSDT":
                    prices["eth"] = float(item["price"])
                elif item["symbol"] == "SOLUSDT":  # T423: SOL price
                    prices["sol"] = float(item["price"])
            if "btc" in prices and "eth" in prices:
                set_cached_response("prices_binance", prices)  # Cache the result (SOL optional)
                return prices
    except Exception as e:
        latency = (time.time() - start) * 1000
        record_api_latency("ext_binance_error", latency)
        print(f"[PRICE] Binance error: {e}")
    return None


def get_prices_coinbase() -> dict:
    """Get BTC/ETH prices from Coinbase with latency tracking and caching"""
    # Check cache first (T427)
    cached = get_cached_response("prices_coinbase")
    if cached:
        return cached
    
    start = time.time()
    try:
        btc_resp = requests.get(
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            timeout=5
        )
        eth_resp = requests.get(
            "https://api.coinbase.com/v2/prices/ETH-USD/spot",
            timeout=5
        )
        sol_resp = requests.get(  # T423: SOL price
            "https://api.coinbase.com/v2/prices/SOL-USD/spot",
            timeout=5
        )
        latency = (time.time() - start) * 1000
        record_api_latency("ext_coinbase", latency)
        record_api_call("coinbase", dict(btc_resp.headers))  # Track rate limit
        if btc_resp.status_code == 200 and eth_resp.status_code == 200:
            result = {
                "btc": float(btc_resp.json()["data"]["amount"]),
                "eth": float(eth_resp.json()["data"]["amount"]),
                "source": "coinbase"
            }
            # T423: Add SOL if available (optional)
            if sol_resp.status_code == 200:
                result["sol"] = float(sol_resp.json()["data"]["amount"])
            set_cached_response("prices_coinbase", result)  # Cache the result
            return result
    except Exception as e:
        latency = (time.time() - start) * 1000
        record_api_latency("ext_coinbase_error", latency)
        print(f"[PRICE] Coinbase error: {e}")
    return None


def get_crypto_prices(max_retries: int = 3) -> dict:
    """
    Get current BTC/ETH prices from multiple exchanges.
    Aggregates prices from Binance, CoinGecko, and Coinbase for accuracy.
    Uses median price when multiple sources available.
    
    Exchange order is dynamically prioritized based on latency (T396).
    """
    all_prices = []
    sources_used = []
    
    # Get optimal exchange order based on latency (T396)
    exchange_order = get_optimal_exchange_order()
    
    # Map exchange names to fetch functions
    exchange_funcs = {
        "binance": get_prices_binance,
        "coingecko": get_prices_coingecko,
        "coinbase": get_prices_coinbase,
    }
    
    # Build ordered list of (name, func) tuples
    price_funcs = [(name, exchange_funcs[name]) for name in exchange_order if name in exchange_funcs]
    
    for source_name, fetch_func in price_funcs:
        for attempt in range(max_retries):
            result = fetch_func()
            if result:
                all_prices.append(result)
                sources_used.append(source_name)
                break
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 0.5 + (time.time() % 0.5)
                time.sleep(wait_time)
    
    if not all_prices:
        print("[PRICE] ERROR: All exchanges failed!")
        return None
    
    # Aggregate prices using median for robustness
    btc_prices = [p["btc"] for p in all_prices if "btc" in p]
    eth_prices = [p["eth"] for p in all_prices if "eth" in p]
    
    def median(values):
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n == 0:
            return None
        if n % 2 == 1:
            return sorted_vals[n // 2]
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    
    btc_median = median(btc_prices) if btc_prices else None
    eth_median = median(eth_prices) if eth_prices else None
    
    if btc_median is None or eth_median is None:
        print(f"[PRICE] WARNING: Missing prices - BTC: {btc_prices}, ETH: {eth_prices}")
        return None
    
    # Log price spread for monitoring
    if len(btc_prices) > 1:
        btc_spread = (max(btc_prices) - min(btc_prices)) / btc_median * 100
        if btc_spread > 0.5:  # >0.5% spread is unusual
            print(f"[PRICE] WARNING: BTC spread {btc_spread:.2f}% across exchanges")
    
    print(f"[PRICE] BTC: ${btc_median:,.0f} | ETH: ${eth_median:,.0f} ({len(sources_used)} sources: {', '.join(sources_used)})")
    
    return {
        "btc": btc_median,
        "eth": eth_median,
        "sources": sources_used,
        "source_count": len(sources_used)
    }


def get_fear_greed(max_retries: int = 2) -> int:
    """Get Fear & Greed Index (0-100) with retry logic, latency tracking and caching"""
    # Check cache first (T427) - F&G updates daily, 5 min cache is fine
    cached = get_cached_response("fear_greed")
    if cached is not None:
        return cached
    
    start = time.time()
    for attempt in range(max_retries):
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
            latency = (time.time() - start) * 1000
            record_api_latency("ext_fear_greed", latency)
            record_api_call("feargreed", dict(resp.headers))  # Track rate limit
            if resp.status_code >= 500:
                raise requests.exceptions.RequestException(f"Server error {resp.status_code}")
            value = int(resp.json()["data"][0]["value"])
            set_cached_response("fear_greed", value)  # Cache the result
            return value
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + (time.time() % 1)
                print(f"[RETRY] F&G error: {e}, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                continue
    latency = (time.time() - start) * 1000
    record_api_latency("ext_fear_greed_error", latency)
    return 50  # Default neutral


# ============== FEEDBACK LOOP ==============

def update_trade_results():
    """
    Check settled markets and update trade log with actual results.
    THIS IS THE MISSING PIECE!
    """
    log_path = Path(TRADE_LOG_FILE)
    if not log_path.exists():
        return {"updated": 0, "wins": 0, "losses": 0}
    
    # Read all trades
    trades = []
    with open(log_path) as f:
        for line in f:
            trades.append(json.loads(line.strip()))
    
    # Find pending trades
    pending_trades = [t for t in trades if t.get("type") == "trade" and t.get("result_status") == "pending"]
    
    updated = 0
    wins = 0
    losses = 0
    
    for trade in pending_trades:
        ticker = trade.get("ticker")
        if not ticker:
            continue
            
        # Check market status
        market = get_market(ticker)
        status = market.get("status")
        result = market.get("result")
        
        if status == "finalized" and result:
            # Market has settled!
            our_side = trade.get("side", "no")
            
            # Did we win?
            we_won = (our_side == result)
            
            # Update trade record
            trade["result_status"] = "win" if we_won else "loss"
            trade["market_result"] = result
            trade["settled_at"] = datetime.now(timezone.utc).isoformat()
            
            if we_won:
                # Win: we get $1 per contract
                trade["profit_cents"] = (100 * trade.get("contracts", 0)) - trade.get("cost_cents", 0)
                wins += 1
            else:
                # Loss: we lose our cost
                trade["profit_cents"] = -trade.get("cost_cents", 0)
                losses += 1
            
            updated += 1
    
    # Write back updated trades
    if updated > 0:
        with open(log_path, "w") as f:
            for trade in trades:
                f.write(json.dumps(trade) + "\n")
    
    return {"updated": updated, "wins": wins, "losses": losses}


def update_dryrun_trade_results():
    """
    Check settled markets and update DRY RUN trade log with actual results.
    This is the missing feedback loop for paper trading!
    Without this, we can never know if our predictions were correct.
    """
    log_path = Path(DRY_RUN_LOG_FILE)
    if not log_path.exists():
        return {"updated": 0, "wins": 0, "losses": 0}
    
    # Read all trades
    trades = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                trades.append(json.loads(line))
    
    # Find pending trades
    pending_trades = [t for t in trades if t.get("type") == "trade" and t.get("result_status") == "pending"]
    
    if not pending_trades:
        return {"updated": 0, "wins": 0, "losses": 0}
    
    # Check unique tickers only (avoid redundant API calls)
    unique_tickers = set(t.get("ticker") for t in pending_trades if t.get("ticker"))
    
    updated = 0
    wins = 0
    losses = 0
    
    # Fetch market results for each unique ticker
    ticker_results = {}
    for ticker in unique_tickers:
        try:
            market = get_market(ticker)
            status = market.get("status")
            result = market.get("result")
            
            if status == "finalized" and result:
                ticker_results[ticker] = result
        except Exception as e:
            print(f"   ⚠️ Failed to check {ticker}: {e}")
    
    if not ticker_results:
        return {"updated": 0, "wins": 0, "losses": 0}
    
    # Update all trades for settled tickers
    for trade in trades:
        if trade.get("type") != "trade" or trade.get("result_status") != "pending":
            continue
        
        ticker = trade.get("ticker")
        if ticker not in ticker_results:
            continue
        
        result = ticker_results[ticker]
        our_side = trade.get("side", "no")
        
        # Did we win?
        we_won = (our_side == result)
        
        # Update trade record
        trade["result_status"] = "win" if we_won else "loss"
        trade["market_result"] = result
        trade["settled_at"] = datetime.now(timezone.utc).isoformat()
        
        if we_won:
            trade["profit_cents"] = (100 * trade.get("contracts", 1)) - trade.get("cost_cents", 0)
            wins += 1
        else:
            trade["profit_cents"] = -trade.get("cost_cents", 0)
            losses += 1
        
        updated += 1
    
    # Write back updated trades
    if updated > 0:
        with open(log_path, "w") as f:
            for trade in trades:
                f.write(json.dumps(trade) + "\n")
        print(f"   📊 [PAPER] Updated {updated} dry-run trades: {wins}W / {losses}L")
    
    return {"updated": updated, "wins": wins, "losses": losses}


def get_trade_stats() -> dict:
    """Calculate win/loss stats from trade log"""
    stats = {"total": 0, "wins": 0, "losses": 0, "pending": 0, "profit_cents": 0}
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
                    stats["profit_cents"] += entry.get("profit_cents", 0)  # Already negative
                else:
                    stats["pending"] += 1
    
    return stats


def log_skip(skip_data: dict):
    """Log a skipped trade to the skip log file."""
    with open(SKIP_LOG_FILE, "a") as f:
        f.write(json.dumps(skip_data) + "\n")


def log_trade(trade_data: dict):
    """Log a trade"""
    log_path = Path(TRADE_LOG_FILE)
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(trade_data) + "\n")


def log_dry_run_trade(trade_data: dict):
    """Log a simulated trade (dry run mode) with deduplication.
    
    Avoids re-logging the same ticker/side within the last N entries,
    as paper mode runs every minute and would otherwise create many duplicates.
    """
    trade_data["dry_run"] = True
    log_path = Path(DRY_RUN_LOG_FILE)
    log_path.parent.mkdir(exist_ok=True)
    
    # Deduplication: check last 10 entries for same ticker+side
    ticker = trade_data.get("ticker", "")
    side = trade_data.get("side", "")
    
    if ticker and side and log_path.exists():
        try:
            with open(log_path) as f:
                lines = f.readlines()
            # Check last 10 entries
            recent = lines[-10:] if len(lines) >= 10 else lines
            for line in recent:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("ticker") == ticker and entry.get("side") == side:
                        # Same ticker+side recently logged, skip
                        print(f"   🔄 Dedup: skipping {ticker} {side} (already in last 10 entries)")
                        return
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass  # If dedup check fails, proceed with logging
    
    with open(log_path, "a") as f:
        f.write(json.dumps(trade_data) + "\n")


def log_execution(ticker: str, side: str, count: int, price_cents: int, 
                  status: str, latency_ms: int = None, error: str = None, 
                  retries: int = 0, order_id: str = None):
    """
    Log execution attempt for success rate tracking (T329).
    
    Status values: executed, pending, rejected, error, timeout
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "side": side,
        "count": count,
        "price_cents": price_cents,
        "status": status,
        "latency_ms": latency_ms,
        "retries": retries,
    }
    if error:
        entry["error"] = error
    if order_id:
        entry["order_id"] = order_id
    
    log_path = Path(EXECUTION_LOG_FILE)
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_ml_features(trade_data: dict):
    """
    Log ML-friendly feature vectors for model training (T331).
    
    This creates a clean dataset with:
    - Input features (all available at decision time)
    - Target: predicted_prob vs market_prob (outcome added by settlement tracker)
    - Metadata for filtering/analysis
    
    Fields logged:
    - id: unique trade identifier (timestamp_ticker)
    - timestamp: ISO timestamp
    - asset: btc/eth
    - side: yes/no
    - target_label: 1 if our prediction says YES, 0 if NO (filled by settlement)
    
    Feature groups:
    - price_*: price-related features
    - momentum_*: momentum indicators
    - regime_*: market regime features  
    - vol_*: volatility features
    - time_*: time-based features
    - prob_*: probability calculations
    """
    # Generate unique ID
    ts = trade_data.get("timestamp", datetime.now(timezone.utc).isoformat())
    ticker = trade_data.get("ticker", "unknown")
    trade_id = f"{ts}_{ticker}"
    
    ml_record = {
        # Metadata
        "id": trade_id,
        "timestamp": ts,
        "asset": trade_data.get("asset", "btc"),
        "ticker": ticker,
        "side": trade_data.get("side", ""),
        
        # Price features (normalized) - may be None for weather markets
        "price_current": trade_data.get("current_price") or 0,
        "price_strike": trade_data.get("strike") or 0,
        "price_distance_pct": 0,  # calculated below
        "price_above_strike": 1 if (trade_data.get("current_price") or 0) > (trade_data.get("strike") or 0) else 0,
        
        # Weather-specific features (T422)
        "is_weather_market": 1 if trade_data.get("asset") == "weather" else 0,
        "forecast_temp": trade_data.get("forecast_temp"),
        "forecast_uncertainty": trade_data.get("forecast_uncertainty"),
        "city": trade_data.get("city"),
        
        # Probability features
        "prob_predicted": trade_data.get("our_prob", 0),  # Our model's probability
        "prob_market": trade_data.get("market_prob", 0),  # Market's implied probability
        "prob_base": trade_data.get("base_prob", trade_data.get("our_prob", 0)),  # Before adjustments
        "edge": trade_data.get("edge", 0),
        "edge_with_bonus": trade_data.get("edge_with_bonus", trade_data.get("edge", 0)),
        
        # Momentum features
        "momentum_direction": trade_data.get("momentum_dir", 0),  # -1 bearish, 0 neutral, 1 bullish
        "momentum_strength": trade_data.get("momentum_str", 0),   # 0-1 strength
        "momentum_aligned": 1 if trade_data.get("momentum_aligned", False) else 0,
        "momentum_full_alignment": 1 if trade_data.get("full_alignment", False) else 0,
        
        # Regime features (one-hot encoded)
        "regime": trade_data.get("regime", "unknown"),
        "regime_trending_bullish": 1 if trade_data.get("regime") == "trending_bullish" else 0,
        "regime_trending_bearish": 1 if trade_data.get("regime") == "trending_bearish" else 0,
        "regime_sideways": 1 if trade_data.get("regime") == "sideways" else 0,
        "regime_choppy": 1 if trade_data.get("regime") == "choppy" else 0,
        "regime_confidence": trade_data.get("regime_confidence", 0),
        
        # VIX features (T611) - fear index integration
        "vix_current": trade_data.get("vix_current"),
        "vix_low_fear": 1 if trade_data.get("vix_regime") == "low_fear" else 0,
        "vix_moderate": 1 if trade_data.get("vix_regime") == "moderate" else 0,
        "vix_elevated": 1 if trade_data.get("vix_regime") == "elevated" else 0,
        "vix_high_fear": 1 if trade_data.get("vix_regime") == "high_fear" else 0,
        "vix_multiplier": trade_data.get("vix_multiplier", 1.0),
        
        # Volatility features
        "vol_ratio": trade_data.get("vol_ratio", 1.0),  # realized/assumed
        "vol_aligned": 1 if trade_data.get("vol_aligned", False) else 0,
        "vol_bonus": trade_data.get("vol_bonus", 0),
        
        # Time features
        "minutes_to_expiry": trade_data.get("minutes_to_expiry", 0),
        "hour_utc": datetime.fromisoformat(ts.replace("Z", "+00:00")).hour if "T" in ts else 0,
        "day_of_week": datetime.fromisoformat(ts.replace("Z", "+00:00")).weekday() if "T" in ts else 0,
        
        # Position sizing features
        "contracts": trade_data.get("contracts", 0),
        "price_cents": trade_data.get("price_cents", 0),
        "cost_cents": trade_data.get("cost_cents", 0),
        "kelly_fraction_used": trade_data.get("kelly_fraction_used", KELLY_FRACTION),
        "size_multiplier": trade_data.get("size_multiplier_total", 1.0),
        
        # Streak position features (T770) - trading psychology
        "streak_current": trade_data.get("streak_current", 0),
        "streak_type_win": 1 if trade_data.get("streak_type") == "win" else 0,
        "streak_type_loss": 1 if trade_data.get("streak_type") == "loss" else 0,
        "streak_tilt_risk": 1 if trade_data.get("streak_tilt_risk", False) else 0,
        "streak_hot_hand": 1 if trade_data.get("streak_hot_hand", False) else 0,
        "streak_continuation_prob": trade_data.get("streak_continuation_prob"),
        
        # Target (filled by settlement tracker)
        "actual_outcome": None,  # 1=won, 0=lost (filled later)
        "settlement_price": None,  # Final BTC/ETH price at settlement
        "profit_cents": None,  # Actual P&L
    }
    
    # Calculate price distance percentage (crypto only - weather has no strike)
    strike = trade_data.get("strike")
    current = trade_data.get("current_price")
    if strike and current and strike > 0:
        ml_record["price_distance_pct"] = (current - strike) / strike * 100
    
    # Write to ML log file
    log_path = Path(ML_FEATURE_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(ml_record) + "\n")
    
    print(f"   📊 ML features logged ({len([k for k,v in ml_record.items() if v is not None])} features)")


# ============== OPPORTUNITY FINDING ==============

def parse_time_to_expiry(market: dict) -> float:
    """Get minutes to expiry from market data"""
    try:
        close_time_str = market.get("close_time")
        if not close_time_str:
            return 999
        close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (close_time - now).total_seconds() / 60
    except:
        return 999


def find_opportunities(markets: list, prices: dict, momentum_data: dict = None, 
                       ohlc_data: dict = None, verbose: bool = True,
                       news_sentiment: dict = None) -> list:
    """Find trading opportunities with PROPER probability model + momentum adjustment + regime detection + volatility rebalancing + news sentiment (T661)
    
    Args:
        markets: List of market dicts from Kalshi API
        prices: Dict with "btc" and "eth" prices
        momentum_data: Dict with "btc" and "eth" momentum dicts
        ohlc_data: Dict with "btc" and "eth" OHLC data for regime detection
        news_sentiment: Dict with news analysis results (T661 - Grok Fundamental)
    """
    opportunities = []
    skip_reasons = {
        "not_crypto": 0,
        "no_strike": 0,
        "too_close_expiry": 0,
        "momentum_conflict": 0,
        "insufficient_edge": []
    }
    
    # Calculate market regime for each asset to get dynamic MIN_EDGE
    regime_cache = {}
    if ohlc_data and momentum_data:
        for asset in ["btc", "eth"]:
            regime_cache[asset] = detect_market_regime(
                ohlc_data.get(asset, []), 
                momentum_data.get(asset, {})
            )
            if verbose and regime_cache[asset]:
                r = regime_cache[asset]
                print(f"   {asset.upper()} regime: {r['regime']} (conf={r['confidence']:.0%}, vol={r['volatility']}, edge={r['dynamic_min_edge']*100:.1f}%)")
    
    # Calculate divergence signals for each asset (T457)
    # Bearish divergence = favor NO bets, Bullish divergence = favor YES bets
    divergence_cache = {}
    if ohlc_data and momentum_data:
        for asset in ["btc", "eth"]:
            divergence_cache[asset] = get_divergence_edge_adjustment(
                asset, ohlc_data, momentum_data
            )
            if verbose and divergence_cache[asset].get("has_signal"):
                d = divergence_cache[asset]
                div_type = d.get("divergence_type", "unknown")
                print(f"   ⚠️ {asset.upper()} DIVERGENCE: {div_type} (+{d['adjustment']*100:.1f}% edge bonus for {'NO' if div_type == 'bearish' else 'YES'} bets)")
    
    # Calculate volatility advantage for asset rebalancing (T237)
    vol_advantage = {}
    if ohlc_data:
        vol_advantage = get_volatility_advantage(ohlc_data)
        if verbose:
            btc_v = vol_advantage.get("btc", {})
            eth_v = vol_advantage.get("eth", {})
            pref = vol_advantage.get("preferred_asset")
            if btc_v.get("realized") and eth_v.get("realized"):
                pref_str = f" → Prefer {pref.upper()}" if pref else " → No preference"
                print(f"   📊 Vol Rebalance: BTC ratio={btc_v['ratio']:.2f} ({btc_v['advantage']}) | ETH ratio={eth_v['ratio']:.2f} ({eth_v['advantage']}){pref_str}")
    
    for m in markets:
        ticker = m.get("ticker", "")
        subtitle = m.get("subtitle", "")
        yes_bid = m.get("yes_bid", 0)
        yes_ask = m.get("yes_ask", 0)
        no_bid = m.get("no_bid", 0)
        no_ask = m.get("no_ask", 0)
        
        # Detect asset type from ticker
        if ticker.startswith("KXBTCD"):
            asset = "btc"
            current_price = prices.get("btc", 0)
            # Use dynamic volatility from OHLC data when available
            hourly_vol = get_dynamic_hourly_vol(ohlc_data.get("btc", []) if ohlc_data else [], "btc")
        elif ticker.startswith("KXETHD"):
            asset = "eth"
            current_price = prices.get("eth", 0)
            hourly_vol = get_dynamic_hourly_vol(ohlc_data.get("eth", []) if ohlc_data else [], "eth")
        else:
            skip_reasons["not_crypto"] += 1
            continue
        
        if not current_price:
            continue
        
        # Get momentum for this asset
        momentum = momentum_data.get(asset) if momentum_data else None
        
        # Extract momentum info for filtering
        mom_dir = momentum.get("composite_direction", 0) if momentum else 0
        mom_str = momentum.get("composite_strength", 0) if momentum else 0
        mom_alignment = momentum.get("alignment", False) if momentum else False
        
        # Parse strike from subtitle
        if "$" not in subtitle:
            skip_reasons["no_strike"] += 1
            continue
        
        try:
            strike_str = subtitle.split("$")[1].split(" ")[0].replace(",", "")
            strike = float(strike_str)
        except:
            skip_reasons["no_strike"] += 1
            continue
        
        # Get time to expiry
        minutes_left = parse_time_to_expiry(m)
        if minutes_left < MIN_TIME_TO_EXPIRY_MINUTES:
            skip_reasons["too_close_expiry"] += 1
            continue
        
        # Calculate BASE probability (from Black-Scholes model)
        base_prob_above = calculate_prob_above_strike(
            current_price, strike, minutes_left, hourly_vol
        )
        base_prob_below = 1 - base_prob_above
        
        # APPLY MOMENTUM ADJUSTMENT to probabilities
        if momentum:
            prob_above = adjust_probability_with_momentum(
                base_prob_above, strike, current_price, momentum, "yes"
            )
            prob_below = adjust_probability_with_momentum(
                base_prob_below, strike, current_price, momentum, "no"
            )
        else:
            prob_above = base_prob_above
            prob_below = base_prob_below
        
        # Market implied probabilities
        market_prob_yes = yes_ask / 100 if yes_ask else 0.5
        market_prob_no = (100 - yes_bid) / 100 if yes_bid else 0.5
        
        # ============ PROBABILITY SANITY CHECK (BUGFIX 2026-07-17) ============
        # If our model disagrees with the market by >25 percentage points,
        # our model is almost certainly wrong. The market aggregates thousands
        # of traders — when there's a 25%+ disagreement, it's a model bug.
        # This would have caught ALL 41 losing BTC trades (avg disagreement: 46%).
        yes_disagreement = abs(prob_above - market_prob_yes)
        no_disagreement = abs(prob_below - market_prob_no)
        if min(yes_disagreement, no_disagreement) > MAX_PROB_DISAGREEMENT:
            skip_reasons["model_market_disagree"] = skip_reasons.get("model_market_disagree", 0) + 1
            continue
        
        # Calculate edges for both sides (now momentum-adjusted)
        yes_edge = prob_above - market_prob_yes
        no_edge = prob_below - market_prob_no
        
        found_opp = False
        
        # MOMENTUM CONFLICT CHECK: Skip YES if strong bearish momentum
        # Skip NO if strong bullish momentum (unless we have alignment which gives confidence)
        skip_due_to_momentum = False
        
        # Get dynamic MIN_EDGE from regime detection (falls back to static MIN_EDGE)
        regime = regime_cache.get(asset, {})
        dynamic_edge = regime.get("dynamic_min_edge", MIN_EDGE)
        
        # ============ STRIKE DISTANCE FILTER (BUGFIX 2026-07-17) ============
        # Don't trade strikes that are too close to current price.
        # For hourly contracts, price rarely moves >1% in an hour.
        # Buying NO on a strike that's only 0.5-2% below current price = guaranteed loss.
        strike_distance_pct = abs(current_price - strike) / current_price
        
        if strike_distance_pct < MIN_STRIKE_DISTANCE_PCT:
            skip_reasons["too_close_strike"] = skip_reasons.get("too_close_strike", 0) + 1
            continue
        
        # Check for YES opportunity (we think it'll be above strike)
        # Skip extreme prices (no profit potential or bad risk/reward)
        # BUGFIX 2026-07-17: Increased minimum from 5c to MIN_NO_PRICE_CENTS for NO contracts.
        # When market prices NO at 1-5 cents, it's usually correct (1-5% prob).
        # Our model should not override the market on low-probability events.
        # For YES: keep 5c minimum (YES at 5c = 5% prob, more reasonable edge source)
        yes_extreme = yes_ask and (yes_ask <= 5 or yes_ask >= 95)
        no_price = 100 - yes_bid if yes_bid else None
        no_extreme = not no_price or no_price <= MIN_NO_PRICE_CENTS or no_price >= 95
        
        if yes_extreme:
            skip_reasons["extreme_price"] = skip_reasons.get("extreme_price", 0) + 1
        elif prob_above > market_prob_yes + dynamic_edge:
            # KELLY CHECK: Skip if Kelly fraction is negative (trade is -EV)
            kelly_f = kelly_criterion_check(prob_above, yes_ask) if yes_ask else 0
            if kelly_f <= 0:
                skip_reasons["negative_kelly"] = skip_reasons.get("negative_kelly", 0) + 1
            # Skip YES if momentum is strongly bearish (dir < -0.3 with strength > 0.3)
            elif mom_dir < -0.3 and mom_str > 0.3:
                skip_due_to_momentum = True
                skip_reasons["momentum_conflict"] += 1
            else:
                edge = prob_above - market_prob_yes
                # Bonus edge if momentum aligns (bullish momentum for YES)
                momentum_bonus = 0.02 if (mom_dir > 0.2 and mom_alignment) else 0
                # Volatility rebalance bonus (T237): favor YES when realized vol > assumed
                vol_info = vol_advantage.get(asset, {}) if vol_advantage else {}
                vol_bonus = max(0, vol_advantage.get("vol_bonus", {}).get(asset, 0)) if vol_info.get("advantage") == "yes" else 0
                vol_aligned = vol_info.get("advantage") == "yes"
                # News sentiment bonus (T661 - Grok Fundamental strategy)
                # Bullish news gives edge bonus for YES bets
                news_bonus = 0
                news_info = {}
                if news_sentiment:
                    news_info = {
                        "sentiment": news_sentiment.get("sentiment", "neutral"),
                        "confidence": news_sentiment.get("confidence", 0.5),
                        "reasons": news_sentiment.get("reasons", [])[:2]  # Keep top 2
                    }
                    if news_sentiment.get("sentiment") == "bullish":
                        news_bonus = news_sentiment.get("edge_adjustment", 0)  # Positive for YES
                    elif news_sentiment.get("sentiment") == "bearish":
                        news_bonus = -abs(news_sentiment.get("edge_adjustment", 0)) * 0.5  # Penalty for YES
                # Divergence bonus (T457)
                # Bullish divergence = price down but RSI up = reversal up likely = favor YES
                divergence_bonus = 0
                divergence_info = divergence_cache.get(asset, {})
                if divergence_info.get("has_signal"):
                    if divergence_info.get("divergence_type") == "bullish":
                        divergence_bonus = divergence_info.get("adjustment", 0)  # Bonus for YES
                    elif divergence_info.get("divergence_type") == "bearish":
                        divergence_bonus = -divergence_info.get("adjustment", 0) * 0.5  # Penalty for YES
                # Composite signal scoring (T460)
                # Multiple confirming signals = higher conviction = extra synergy bonus
                composite_signal = calculate_composite_signal_score(asset, "yes", ohlc_data, momentum_data)
                composite_bonus = composite_signal.get("total_bonus", 0) - (momentum_bonus + divergence_bonus)  # Avoid double-counting
                composite_bonus = max(0, composite_bonus)  # Only add synergy bonus, not reduce
                opportunities.append({
                    "ticker": ticker,
                    "asset": asset,
                    "side": "yes",
                    "price": yes_ask,
                    "edge": edge,
                    "edge_with_bonus": edge + momentum_bonus + vol_bonus + news_bonus + divergence_bonus + composite_bonus,
                    "our_prob": prob_above,
                    "base_prob": base_prob_above,
                    "market_prob": market_prob_yes,
                    "strike": strike,
                    "current": current_price,
                    "minutes_left": minutes_left,
                    "momentum_dir": mom_dir,
                    "momentum_str": mom_str,
                    "momentum_aligned": mom_alignment and mom_dir > 0.2,
                    "full_alignment": mom_alignment,  # T395: all timeframes agree
                    "regime": regime.get("regime", "unknown"),
                    "regime_confidence": regime.get("confidence", 0),
                    "volatility": regime.get("volatility", "normal"),  # T293
                    "vix_size_multiplier": regime.get("vix_size_multiplier", 1.0),  # T611: VIX-based sizing
                    "vix_current": regime.get("details", {}).get("vix_current"),  # T611
                    "vix_regime": regime.get("details", {}).get("vix_regime", "unknown"),  # T611
                    "dynamic_min_edge": dynamic_edge,
                    "vol_ratio": vol_info.get("ratio", 1.0),
                    "vol_aligned": vol_aligned,
                    "vol_bonus": vol_bonus,
                    "news_bonus": news_bonus,
                    "news_sentiment": news_info.get("sentiment", "neutral"),
                    "news_confidence": news_info.get("confidence", 0.5),
                    "news_reasons": news_info.get("reasons", []),
                    "divergence_bonus": divergence_bonus,
                    "divergence_aligned": divergence_info.get("divergence_type") == "bullish",
                    "divergence_type": divergence_info.get("divergence_type", "none"),
                    "divergence_reason": divergence_info.get("reason", ""),
                    # Composite signal scoring (T460)
                    "composite_signals": composite_signal.get("confirming_signals", 0),
                    "composite_confidence": composite_signal.get("confidence", "low"),
                    "composite_bonus": composite_bonus,
                    "composite_reasons": composite_signal.get("signals", [])
                })
                found_opp = True
        
        # Check for NO opportunity (we think it'll be below strike)
        # Skip extreme NO prices (no profit potential or bad risk/reward)
        # Also skip if no_price is None (no bid available)
        if no_extreme:
            if no_price:  # Only count as extreme if we have a price
                skip_reasons["extreme_price"] = skip_reasons.get("extreme_price", 0) + 1
        elif prob_below > market_prob_no + dynamic_edge:
            # KELLY CHECK: Skip if Kelly fraction is negative (trade is -EV)
            no_price_for_kelly = 100 - yes_bid if yes_bid else 0
            kelly_f_no = kelly_criterion_check(prob_below, no_price_for_kelly) if no_price_for_kelly else 0
            if kelly_f_no <= 0:
                skip_reasons["negative_kelly"] = skip_reasons.get("negative_kelly", 0) + 1
            # Skip NO if momentum is strongly bullish (dir > 0.3 with strength > 0.3)
            elif mom_dir > 0.3 and mom_str > 0.3:
                if not skip_due_to_momentum:  # Don't double count
                    skip_reasons["momentum_conflict"] += 1
            else:
                edge = prob_below - market_prob_no
                # Bonus edge if momentum aligns (bearish momentum for NO)
                momentum_bonus = 0.02 if (mom_dir < -0.2 and mom_alignment) else 0
                # Volatility rebalance bonus (T237): favor NO when realized vol < assumed
                vol_info = vol_advantage.get(asset, {}) if vol_advantage else {}
                # For NO bets, we want NEGATIVE vol_bonus (realized < assumed = less movement = good for NO)
                vol_bonus = abs(min(0, vol_advantage.get("vol_bonus", {}).get(asset, 0))) if vol_info.get("advantage") == "no" else 0
                vol_aligned = vol_info.get("advantage") == "no"
                # News sentiment bonus (T661 - Grok Fundamental strategy)
                # Bearish news gives edge bonus for NO bets
                news_bonus = 0
                news_info = {}
                if news_sentiment:
                    news_info = {
                        "sentiment": news_sentiment.get("sentiment", "neutral"),
                        "confidence": news_sentiment.get("confidence", 0.5),
                        "reasons": news_sentiment.get("reasons", [])[:2]  # Keep top 2
                    }
                    if news_sentiment.get("sentiment") == "bearish":
                        news_bonus = abs(news_sentiment.get("edge_adjustment", 0))  # Positive for NO
                    elif news_sentiment.get("sentiment") == "bullish":
                        news_bonus = -abs(news_sentiment.get("edge_adjustment", 0)) * 0.5  # Penalty for NO
                # Divergence bonus (T457)
                # Bearish divergence = price up but RSI down = reversal down likely = favor NO
                divergence_bonus = 0
                divergence_info = divergence_cache.get(asset, {})
                if divergence_info.get("has_signal"):
                    if divergence_info.get("divergence_type") == "bearish":
                        divergence_bonus = divergence_info.get("adjustment", 0)  # Bonus for NO
                    elif divergence_info.get("divergence_type") == "bullish":
                        divergence_bonus = -divergence_info.get("adjustment", 0) * 0.5  # Penalty for NO
                # Composite signal scoring (T460)
                # Multiple confirming signals = higher conviction = extra synergy bonus
                composite_signal = calculate_composite_signal_score(asset, "no", ohlc_data, momentum_data)
                composite_bonus = composite_signal.get("total_bonus", 0) - (momentum_bonus + divergence_bonus)  # Avoid double-counting
                composite_bonus = max(0, composite_bonus)  # Only add synergy bonus, not reduce
                opportunities.append({
                    "ticker": ticker,
                    "asset": asset,
                    "side": "no",
                    "price": no_price,
                    "edge": edge,
                    "edge_with_bonus": edge + momentum_bonus + vol_bonus + news_bonus + divergence_bonus + composite_bonus,
                    "our_prob": prob_below,
                    "base_prob": base_prob_below,
                    "market_prob": market_prob_no,
                    "strike": strike,
                    "current": current_price,
                    "minutes_left": minutes_left,
                    "momentum_dir": mom_dir,
                    "momentum_str": mom_str,
                    "momentum_aligned": mom_alignment and mom_dir < -0.2,
                    "full_alignment": mom_alignment,  # T395: all timeframes agree
                    "regime": regime.get("regime", "unknown"),
                    "regime_confidence": regime.get("confidence", 0),
                    "volatility": regime.get("volatility", "normal"),  # T293
                    "vix_size_multiplier": regime.get("vix_size_multiplier", 1.0),  # T611: VIX-based sizing
                    "vix_current": regime.get("details", {}).get("vix_current"),  # T611
                    "vix_regime": regime.get("details", {}).get("vix_regime", "unknown"),  # T611
                    "dynamic_min_edge": dynamic_edge,
                    "vol_ratio": vol_info.get("ratio", 1.0),
                    "vol_aligned": vol_aligned,
                    "vol_bonus": vol_bonus,
                    "news_bonus": news_bonus,
                    "news_sentiment": news_info.get("sentiment", "neutral"),
                    "news_confidence": news_info.get("confidence", 0.5),
                    "news_reasons": news_info.get("reasons", []),
                    "divergence_bonus": divergence_bonus,
                    "divergence_aligned": divergence_info.get("divergence_type") == "bearish",
                    "divergence_type": divergence_info.get("divergence_type", "none"),
                    "divergence_reason": divergence_info.get("reason", ""),
                    # Composite signal scoring (T460)
                    "composite_signals": composite_signal.get("confirming_signals", 0),
                    "composite_confidence": composite_signal.get("confidence", "low"),
                    "composite_bonus": composite_bonus,
                    "composite_reasons": composite_signal.get("signals", [])
                })
                found_opp = True
        
        # Log skip reason if no opportunity found (but not due to extreme price)
        if not found_opp and not (yes_extreme and no_extreme):
            best_edge = max(yes_edge, no_edge)
            best_side = "YES" if yes_edge > no_edge else "NO"
            # Only log if best side wasn't skipped due to extreme price
            skip_due_to_price = (best_side == "YES" and yes_extreme) or (best_side == "NO" and no_extreme)
            if not skip_due_to_price:
                skip_reasons["insufficient_edge"].append({
                    "ticker": ticker,
                    "strike": strike,
                    "best_side": best_side,
                    "best_edge": best_edge,
                    "required_edge": dynamic_edge,  # Use dynamic edge from regime
                    "regime": regime.get("regime", "unknown"),
                    "minutes_left": int(minutes_left)
                })
    
    # Log skip summary if verbose
    if verbose and (skip_reasons["insufficient_edge"] or skip_reasons["too_close_expiry"] or skip_reasons["momentum_conflict"] or skip_reasons.get("extreme_price") or skip_reasons.get("negative_kelly") or skip_reasons.get("too_close_strike") or skip_reasons.get("model_market_disagree")):
        print(f"\n📋 Skip Summary:")
        if skip_reasons["not_crypto"]:
            print(f"   - Not crypto markets: {skip_reasons['not_crypto']}")
        if skip_reasons["no_strike"]:
            print(f"   - No strike parsed: {skip_reasons['no_strike']}")
        if skip_reasons["too_close_expiry"]:
            print(f"   - Too close to expiry (<{MIN_TIME_TO_EXPIRY_MINUTES}min): {skip_reasons['too_close_expiry']}")
        if skip_reasons.get("too_close_strike"):
            print(f"   - Strike too close (<{MIN_STRIKE_DISTANCE_PCT*100:.0f}% from price): {skip_reasons['too_close_strike']}")
        if skip_reasons.get("model_market_disagree"):
            print(f"   - Model vs market disagree (>{MAX_PROB_DISAGREEMENT*100:.0f}%): {skip_reasons['model_market_disagree']}")
        if skip_reasons.get("extreme_price"):
            print(f"   - Extreme price (YES≤5¢/NO≤8¢ or ≥95¢): {skip_reasons['extreme_price']}")
        if skip_reasons["momentum_conflict"]:
            print(f"   - Momentum conflict (betting against trend): {skip_reasons['momentum_conflict']}")
        if skip_reasons.get("negative_kelly"):
            print(f"   - Negative Kelly (trade is -EV despite apparent edge): {skip_reasons['negative_kelly']}")
        
        if skip_reasons["insufficient_edge"]:
            print(f"   - Insufficient edge (need >{MIN_EDGE*100:.0f}%): {len(skip_reasons['insufficient_edge'])}")
            # Show top 5 closest to having edge
            sorted_by_edge = sorted(skip_reasons["insufficient_edge"], key=lambda x: x["best_edge"], reverse=True)
            for skip in sorted_by_edge[:5]:
                edge_pct = skip["best_edge"] * 100
                gap = (MIN_EDGE - skip["best_edge"]) * 100
                print(f"      {skip['ticker']} | Strike ${skip['strike']:,.0f} | {skip['best_side']} edge {edge_pct:+.1f}% (need {gap:.1f}% more) | {skip['minutes_left']}min left")
    
    # ============== EDGE SANITY CHECKS (TRADE-005) ==============
    # Filter out suspiciously high edges (likely model errors, not real alpha)
    # Backtest found that edges >25% for weather were false positives (17.9% WR).
    # For crypto, edges >20% are also suspicious — our Black-Scholes model overestimates
    # edge near extreme strikes where the lognormal assumption breaks down.
    #
    # Per-asset max edge caps:
    #   BTC/ETH/SOL: 20% — crypto model has fat-tail adjustment but still overestimates
    #   Weather:     15% — NWS forecast ±2.8°F MAE makes high edges unreliable
    #   Default:     15% — unknown assets get strictest cap
    ASSET_MAX_EDGE = {
        "btc": 0.20,
        "eth": 0.20,
        "sol": 0.20,
        "weather": 0.15,
        "default": 0.15,
    }
    
    before_cap = len(opportunities)
    filtered_opps = []
    capped_details = []
    for o in opportunities:
        asset_type = o.get("asset", "default")
        asset_cap = ASSET_MAX_EDGE.get(asset_type, ASSET_MAX_EDGE["default"])
        raw_edge = o.get("edge", 0)
        edge_with_bonus = o.get("edge_with_bonus", raw_edge)
        
        # Check raw edge against per-asset cap
        if raw_edge > asset_cap:
            capped_details.append(f"{o.get('ticker','')} {o.get('side','').upper()} edge={raw_edge*100:.1f}% > {asset_cap*100:.0f}% cap ({asset_type})")
            continue
        
        # Also sanity-check: edge_with_bonus shouldn't exceed global MAX_EDGE
        # Bonuses can push total edge unrealistically high
        if edge_with_bonus > MAX_EDGE:
            # Clamp the bonus edge rather than reject outright
            o["edge_with_bonus"] = MAX_EDGE
            o["edge_bonus_clamped"] = True
        
        filtered_opps.append(o)
    
    opportunities = filtered_opps
    if before_cap > len(opportunities):
        capped = before_cap - len(opportunities)
        print(f"   ⚠️ Edge cap: filtered {capped} trades with suspicious edges:")
        for detail in capped_details[:5]:  # Show top 5
            print(f"      {detail}")
    
    # Sort by edge (with momentum bonus for aligned trades)
    opportunities.sort(key=lambda x: x.get("edge_with_bonus", x["edge"]), reverse=True)
    return opportunities


# ============== WEATHER MARKET OPPORTUNITIES (T422) ==============

def find_weather_opportunities(verbose: bool = True) -> list:
    """
    Find trading opportunities in weather markets using NWS forecasts.
    
    Based on PredictionArena research:
    - NWS forecasts are accurate within ±2-3°F for <48h predictions
    - Markets systematically underprice high-probability outcomes (favorite-longshot bias)
    - Best edge on high/low temp markets for next 1-2 days
    
    Returns: List of opportunity dicts compatible with execute_opportunity()
    """
    if not WEATHER_AVAILABLE:
        if verbose:
            print("⚠️ Weather module not available, skipping weather markets")
        return []
    
    if not WEATHER_ENABLED:
        if verbose:
            print("⏸️ Weather trading disabled (set WEATHER_ENABLED=true to enable)")
        return []
    
    opportunities = []
    skip_reasons = {"no_forecast": 0, "too_far": 0, "insufficient_edge": 0, "extreme_price": 0, "parse_error": 0}
    
    # Map city codes to series tickers
    city_series = {
        "NYC": ["KXHIGHNY", "KXLOWTNYC", "KXLOWNY"],
        "MIA": ["KXHIGHMIA", "KXLOWMIA"],
        "DEN": ["KXHIGHDEN", "KXLOWDEN"],
        "CHI": ["KXHIGHCHI", "KXLOWCHI"],
        "LAX": ["KXHIGHLAX", "KXLOWLAX"],
        "HOU": ["KXHIGHHOU", "KXLOWHOU"],
    }
    
    markets_scanned = 0
    
    for city in WEATHER_CITIES:
        series_list = city_series.get(city, [])
        if not series_list:
            continue
        
        # Pre-fetch forecast for this city
        try:
            forecast = fetch_forecast(city)
            if not forecast:
                skip_reasons["no_forecast"] += 1
                continue
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Failed to fetch {city} forecast: {e}")
            skip_reasons["no_forecast"] += 1
            continue
        
        for series in series_list:
            try:
                # Fetch markets for this series
                url = f"{BASE_URL}/trade-api/v2/markets"
                params = {"series_ticker": series, "limit": 20, "status": "open"}
                resp = requests.get(url, params=params, timeout=10)
                
                if resp.status_code != 200:
                    continue
                
                markets = resp.json().get("markets", [])
                markets_scanned += len(markets)
                
                for m in markets:
                    ticker = m.get("ticker")
                    title = m.get("title", "")
                    yes_bid = m.get("yes_bid")
                    yes_ask = m.get("yes_ask")
                    close_time_str = m.get("close_time")
                    
                    if not ticker or yes_bid is None:
                        continue
                    
                    # Skip extreme prices (bad risk/reward)
                    if yes_ask and (yes_ask <= 5 or yes_ask >= 95):
                        skip_reasons["extreme_price"] += 1
                        continue
                    if yes_bid and (yes_bid <= 5 or yes_bid >= 95):
                        skip_reasons["extreme_price"] += 1
                        continue
                    
                    # Check time to settlement
                    if close_time_str:
                        try:
                            close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                            hours_to_close = (close_time - datetime.now(timezone.utc)).total_seconds() / 3600
                            if hours_to_close > WEATHER_MAX_HOURS_TO_SETTLEMENT:
                                skip_reasons["too_far"] += 1
                                continue
                            if hours_to_close < 1:  # Too close to settlement
                                skip_reasons["too_far"] += 1
                                continue
                        except:
                            pass
                    
                    # Parse ticker to understand the market
                    parsed = parse_kalshi_weather_ticker(ticker, title)
                    if not parsed:
                        skip_reasons["parse_error"] += 1
                        continue
                    
                    # Calculate edge using NWS forecast
                    edge_result = calculate_weather_edge(parsed, yes_bid)
                    if not edge_result:
                        skip_reasons["no_forecast"] += 1
                        continue
                    
                    recommendation = edge_result.get("recommendation")
                    edge = edge_result.get("edge", 0)
                    
                    if not recommendation or edge < WEATHER_MIN_EDGE:
                        skip_reasons["insufficient_edge"] += 1
                        continue
                    
                    # ============== SAFETY GUARDS (2026-02-08 Backtest-Validated) ==============
                    # These filters improved win rate from 17.9% to 75%+ in backtesting
                    
                    our_prob = edge_result.get("calculated_probability", 0)
                    market_prob = edge_result.get("market_probability", 0.5)
                    forecast_temp = edge_result.get("forecast_temp")
                    side = "yes" if recommendation == "BUY_YES" else "no"
                    
                    # 1. Reject near-zero probability trades (our_prob < 5%)
                    # These are trades where we have no confidence but edge appears high due to formula
                    if our_prob < WEATHER_MIN_OUR_PROB:
                        skip_reasons["low_prob"] = skip_reasons.get("low_prob", 0) + 1
                        if verbose:
                            print(f"   ⛔ {ticker}: Rejected - our_prob {our_prob:.1%} < {WEATHER_MIN_OUR_PROB:.0%} threshold")
                        continue
                    
                    # 2. Don't bet against extreme market conviction (>85%)
                    # When market says 85%+ YES, betting NO rarely succeeds
                    # When market says 85%+ NO (15% YES), betting YES rarely succeeds
                    if side == "no" and market_prob > WEATHER_MAX_MARKET_CONVICTION:
                        skip_reasons["high_conviction"] = skip_reasons.get("high_conviction", 0) + 1
                        if verbose:
                            print(f"   ⛔ {ticker}: Rejected - market conviction {market_prob:.0%} > {WEATHER_MAX_MARKET_CONVICTION:.0%} (betting NO)")
                        continue
                    if side == "yes" and (1 - market_prob) > WEATHER_MAX_MARKET_CONVICTION:
                        skip_reasons["high_conviction"] = skip_reasons.get("high_conviction", 0) + 1
                        if verbose:
                            print(f"   ⛔ {ticker}: Rejected - market conviction {(1-market_prob):.0%} > {WEATHER_MAX_MARKET_CONVICTION:.0%} (betting YES)")
                        continue
                    
                    # 3. Require minimum gap between forecast and strike
                    # Backtesting: 0°F gap = 6.4% WR, 2°F gap = 40% WR, 7°F+ gap = 100% WR
                    if forecast_temp is not None and WEATHER_MIN_FORECAST_STRIKE_GAP > 0:
                        midpoint = parsed.get("midpoint")
                        if midpoint is not None:
                            gap = abs(forecast_temp - midpoint)
                            if gap < WEATHER_MIN_FORECAST_STRIKE_GAP:
                                skip_reasons["insufficient_gap"] = skip_reasons.get("insufficient_gap", 0) + 1
                                if verbose:
                                    print(f"   ⛔ {ticker}: Rejected - gap {gap:.1f}°F < {WEATHER_MIN_FORECAST_STRIKE_GAP:.1f}°F minimum")
                                continue
                    
                    # ============== END SAFETY GUARDS ==============
                    
                    # Determine price for the order
                    price = yes_ask if side == "yes" else (100 - yes_bid if yes_bid else None)
                    
                    if not price or price <= 0:
                        continue
                    
                    # Build opportunity dict (compatible with execute_opportunity)
                    opp = {
                        "ticker": ticker,
                        "asset": "weather",  # Special asset type
                        "side": side,
                        "price": price,
                        "edge": edge,
                        "edge_with_bonus": edge,  # No momentum/news bonus for weather
                        "our_prob": our_prob,
                        "market_prob": market_prob,
                        "forecast_temp": forecast_temp,
                        "uncertainty": edge_result.get("uncertainty"),
                        "city": city,
                        "is_high_temp": parsed.get("is_high_temp", True),
                        "market_type": parsed.get("market_type", "threshold"),
                        "title": title,
                        "forecast_strike_gap": abs(forecast_temp - parsed.get("midpoint", forecast_temp)) if forecast_temp else None,
                        # Weather-specific Kelly (slightly higher confidence)
                        "kelly_override": WEATHER_KELLY_FRACTION,
                        # Reason for trade log
                        "reason": f"Weather: {city} {'high' if parsed.get('is_high_temp') else 'low'} temp, NWS forecast {forecast_temp}°F, gap {abs(forecast_temp - parsed.get('midpoint', forecast_temp)) if forecast_temp else '?'}°F, edge {edge*100:.1f}%",
                    }
                    opportunities.append(opp)
                    
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ Error scanning {series}: {e}")
    
    if verbose:
        print(f"🌡️ Weather markets: Scanned {markets_scanned} | Found {len(opportunities)} opportunities")
        if skip_reasons["insufficient_edge"]:
            print(f"   Skipped: {skip_reasons['insufficient_edge']} insufficient edge, {skip_reasons['too_far']} too far, {skip_reasons['extreme_price']} extreme price")
    
    # Sort by edge
    opportunities.sort(key=lambda x: x["edge"], reverse=True)
    return opportunities


# ============== MAIN LOOP ==============

def run_cycle():
    """Run one trading cycle"""
    now = datetime.now(timezone.utc)
    print(f"\n{'='*60}")
    print(f"🤖 KALSHI AUTOTRADER v2 - {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*60}")
    
    # Check trading window schedule (T789) - skip bad hours/days
    should_trade, schedule_reason = check_trading_schedule()
    if not should_trade:
        print(f"⏰ Schedule: {schedule_reason}")
        print("💤 Skipping this cycle (outside optimal trading window)")
        log_schedule_skip(schedule_reason)
        return
    else:
        print(f"⏰ Schedule: {schedule_reason}")
    
    # Update trade results first (FEEDBACK LOOP!)
    update_result = update_trade_results()
    if update_result["updated"] > 0:
        print(f"📊 Updated {update_result['updated']} trades: {update_result['wins']}W / {update_result['losses']}L")
        # Check for streak records when trades settle (T288)
        streak_status = check_streak_records()
        print(f"🎖️ {streak_status}")
    
    # Also update dry-run trades (PAPER TRADING FEEDBACK LOOP!)
    if DRY_RUN:
        dryrun_result = update_dryrun_trade_results()
        if dryrun_result["updated"] > 0:
            print(f"🧪 Paper trades settled: {dryrun_result['updated']} ({dryrun_result['wins']}W / {dryrun_result['losses']}L)")
    
    # Get stats
    stats = get_trade_stats()
    win_rate = stats["wins"] / (stats["wins"] + stats["losses"]) * 100 if (stats["wins"] + stats["losses"]) > 0 else 0
    print(f"📈 History: {stats['total']} trades | {stats['wins']}W/{stats['losses']}L | {win_rate:.0f}% WR")
    print(f"💵 P/L: ${stats['profit_cents']/100:+.2f} | Pending: {stats['pending']}")
    
    # Check circuit breaker (consecutive loss protection)
    cb_paused, cb_losses, cb_message = check_circuit_breaker()
    print(f"🔒 Circuit Breaker: {cb_message}")
    if cb_paused:
        print("⏸️ Trading paused by circuit breaker. Waiting for a win to settle...")
        return
    
    # Get balance
    bal = get_balance()
    if "error" in bal:
        print(f"❌ Balance error: {bal['error']}")
        return
    
    cash = bal.get("balance", 0) / 100
    portfolio = bal.get("portfolio_value", 0) / 100
    print(f"💰 Cash: ${cash:.2f} | Portfolio: ${portfolio:.2f}")
    
    # Get prices
    prices = get_crypto_prices()
    if not prices:
        print("❌ Failed to get crypto prices")
        return
    
    print(f"📈 BTC: ${prices['btc']:,.0f} | ETH: ${prices['eth']:,.0f}")
    
    # Get news sentiment (T661 - Grok Fundamental strategy)
    news_sentiment = None
    if NEWS_SEARCH_AVAILABLE:
        try:
            news_sentiment = get_crypto_sentiment("both")
            sentiment_icon = "🟢" if news_sentiment["sentiment"] == "bullish" else ("🔴" if news_sentiment["sentiment"] == "bearish" else "⚪")
            print(f"📰 News Sentiment: {sentiment_icon} {news_sentiment['sentiment'].upper()} ({news_sentiment['confidence']*100:.0f}% conf)")
            if news_sentiment.get("edge_adjustment"):
                print(f"   Edge adjustment: {news_sentiment['edge_adjustment']*100:+.2f}%")
            if news_sentiment.get("event_warning"):
                print(f"   {news_sentiment['event_warning']}")
            if not news_sentiment.get("should_trade", True):
                print("   ⚠️ High-impact event approaching - reduced confidence")
        except Exception as e:
            print(f"📰 News check failed: {e}")
            news_sentiment = None
    
    # Get OHLC data for momentum calculation (7 days gives us hourly candles, enough for 24h momentum)
    btc_ohlc = get_btc_ohlc(days=7)
    eth_ohlc = get_eth_ohlc(days=7)
    
    btc_momentum = get_multi_timeframe_momentum(btc_ohlc)
    eth_momentum = get_multi_timeframe_momentum(eth_ohlc)
    
    momentum_data = {"btc": btc_momentum, "eth": eth_momentum}
    ohlc_data = {"btc": btc_ohlc, "eth": eth_ohlc}
    
    # Check for full momentum alignment (T301) - high-conviction signal
    check_momentum_alignment_alert(momentum_data)
    
    # Check for momentum reversion signals (T302) - contrarian opportunity
    check_reversion_alert(ohlc_data, momentum_data)
    
    # Check for momentum divergence (T303) - price vs momentum disagreement
    check_divergence_alert(ohlc_data, momentum_data)
    
    # Display momentum info for both
    for asset, momentum in [("BTC", btc_momentum), ("ETH", eth_momentum)]:
        if momentum["timeframes"]:
            print(f"📊 {asset} Momentum (1h/4h/24h):")
            for tf in ["1h", "4h", "24h"]:
                tf_data = momentum["timeframes"].get(tf, {})
                dir_symbol = "🟢" if tf_data.get("direction", 0) > 0 else ("🔴" if tf_data.get("direction", 0) < 0 else "⚪")
                pct = tf_data.get("pct_change", 0) * 100
                print(f"   {tf}: {dir_symbol} {pct:+.2f}% (str: {tf_data.get('strength', 0):.2f})")
            
            composite_dir = "BULLISH" if momentum["composite_direction"] > 0.1 else ("BEARISH" if momentum["composite_direction"] < -0.1 else "NEUTRAL")
            aligned_str = "✓ ALIGNED" if momentum["alignment"] else ""
            print(f"   → Composite: {composite_dir} (dir: {momentum['composite_direction']:.2f}, str: {momentum['composite_strength']:.2f}) {aligned_str}")
    
    # Get positions
    positions = get_positions()
    print(f"📋 Open positions: {len(positions)}")
    
    # Always get portfolio value for concentration tracking (T764)
    bal_for_conc = get_balance()
    portfolio_val = bal_for_conc.get("portfolio_value", 0)
    
    # Show portfolio concentration if we have positions (T480)
    if positions:
        concentration = calculate_portfolio_concentration(positions, portfolio_val)
        print_concentration_summary(concentration)
        
        # Show rebalancing suggestions if over-concentrated (T481)
        rebal_suggestions = get_rebalancing_suggestions(positions, portfolio_val, concentration)
        if rebal_suggestions.get("needs_rebalancing"):
            print_rebalancing_suggestions(rebal_suggestions)
        
        # Auto-rebalancing if enabled and over threshold (T816)
        if AUTO_REBALANCE_ENABLED:
            rebal_result = check_and_auto_rebalance(positions, portfolio_val)
            if rebal_result.get("executed"):
                # Refresh positions after rebalancing
                positions = get_positions()
                concentration = calculate_portfolio_concentration(positions, portfolio_val)
                print(f"   🔄 Positions refreshed after auto-rebalance: {len(positions)} open")
    else:
        # Empty portfolio - create zero-concentration state (T764)
        concentration = {
            "by_asset_class": {},
            "by_correlation_group": {},
            "total_exposure_cents": 0,
            "position_count": 0,
            "largest_asset_class": "none",
            "largest_asset_class_pct": 0,
            "largest_correlated_group": "none",
            "largest_correlated_group_pct": 0
        }
    
    # Always log concentration to history for dashboard tracking (T482, T764)
    log_concentration_snapshot(concentration, portfolio_val)
    
    # CHECK STOP-LOSSES for open positions
    if positions:
        stop_loss_candidates = check_stop_losses(positions, prices)
        if stop_loss_candidates:
            print(f"\n🚨 Stop-loss check: {len(stop_loss_candidates)} position(s) below threshold")
            exited = execute_stop_losses(stop_loss_candidates)
            if exited > 0:
                print(f"✅ Exited {exited} position(s) via stop-loss")
                # Refresh positions after exits
                positions = get_positions()
    
    if len(positions) >= MAX_POSITIONS:
        print("⚠️ Max positions reached, skipping")
        return
    
    # Find opportunities from BOTH BTC and ETH markets
    btc_markets = search_markets("KXBTCD", limit=50)
    eth_markets = search_markets("KXETHD", limit=50)
    all_markets = btc_markets + eth_markets
    print(f"🔍 Scanning {len(btc_markets)} BTC + {len(eth_markets)} ETH = {len(all_markets)} total markets...")
    
    # Pass OHLC data for regime detection
    ohlc_data = {"btc": btc_ohlc, "eth": eth_ohlc}
    
    # Show dynamic volatility (new: replaces hardcoded vol)
    btc_dyn_vol = get_dynamic_hourly_vol(btc_ohlc, "btc")
    eth_dyn_vol = get_dynamic_hourly_vol(eth_ohlc, "eth")
    print(f"📊 Dynamic Vol: BTC={btc_dyn_vol*100:.3f}%/hr (fat-tail adj: {btc_dyn_vol*CRYPTO_FAT_TAIL_MULTIPLIER*100:.3f}%) | ETH={eth_dyn_vol*100:.3f}%/hr")
    
    # CHECK FOR REGIME CHANGES (alert on shift)
    btc_regime = detect_market_regime(btc_ohlc, btc_momentum)
    eth_regime = detect_market_regime(eth_ohlc, eth_momentum)
    current_regimes = {"btc": btc_regime, "eth": eth_regime}
    
    print(f"📊 Market Regimes: BTC={btc_regime['regime']} ({btc_regime['confidence']:.0%}) | ETH={eth_regime['regime']} ({eth_regime['confidence']:.0%})")
    
    regime_changes = check_regime_change(current_regimes)
    if regime_changes:
        print(f"🔄 REGIME CHANGE: {', '.join([f'{a}: {old}→{new}' for a, old, new in regime_changes])}")
        write_regime_alert(regime_changes, current_regimes)
    
    # CHECK FOR MOMENTUM DIRECTION CHANGES (bullish↔bearish flips)
    momentum_changes = check_momentum_change(momentum_data)
    if momentum_changes:
        print(f"📊 MOMENTUM FLIP: {', '.join([f'{a}: {old}→{new}' for a, old, new, _ in momentum_changes])}")
        write_momentum_alert(momentum_changes)
        # Check for whipsaw pattern (T393) - 2+ flips in 24h
        check_whipsaw(momentum_changes)
    
    crypto_opportunities = find_opportunities(all_markets, prices, momentum_data=momentum_data, ohlc_data=ohlc_data, news_sentiment=news_sentiment)
    
    # Also scan weather markets (T422 - Based on PredictionArena research)
    weather_opportunities = find_weather_opportunities(verbose=True)
    
    # Merge all opportunities
    all_opportunities = crypto_opportunities + weather_opportunities
    
    # Sort by edge (with bonus)
    all_opportunities.sort(key=lambda x: x.get("edge_with_bonus", x.get("edge", 0)), reverse=True)
    
    if not all_opportunities:
        print("😴 No opportunities found (crypto or weather)")
        return
    
    # In paper mode, trade top N opportunities for maximum data collection
    if DRY_RUN and len(all_opportunities) > 1:
        max_paper_trades = min(5, len(all_opportunities))  # Top 5 per cycle
        print(f"\n🧪 PAPER MODE: Will trade top {max_paper_trades} of {len(all_opportunities)} opportunities!")
    
    # Take best opportunity
    best = all_opportunities[0]
    asset_type = best.get('asset', 'btc')
    asset_label = asset_type.upper()
    
    # Display varies for crypto vs weather
    if asset_type == "weather":
        # Weather opportunity display
        city = best.get('city', '?')
        temp_type = "High" if best.get('is_high_temp', True) else "Low"
        forecast = best.get('forecast_temp', '?')
        uncertainty = best.get('uncertainty', '?')
        print(f"\n🌡️ Best opportunity: {best['ticker']} (WEATHER)")
        print(f"   Side: {best['side'].upper()} @ {best['price']}¢")
        print(f"   {city} {temp_type} Temp | NWS Forecast: {forecast}°F ± {uncertainty}°F")
        print(f"   Our prob: {best['our_prob']*100:.1f}% vs Market: {best['market_prob']*100:.1f}%")
        print(f"   Edge: {best['edge']*100:.1f}%")
        print(f"   💡 Weather edge source: NWS forecast accuracy + favorite-longshot bias")
    else:
        # Crypto opportunity display
        mom_aligned = best.get('momentum_aligned', False)
        mom_badge = "🎯 MOMENTUM ALIGNED!" if mom_aligned else ""
        print(f"\n🎯 Best opportunity: {best['ticker']} {mom_badge}")
        print(f"   Side: {best['side'].upper()} @ {best['price']}¢")
        print(f"   Strike: ${best['strike']:,.0f} | {asset_label}: ${best['current']:,.0f}")
        print(f"   Base prob: {best.get('base_prob', best['our_prob'])*100:.1f}% → Adjusted: {best['our_prob']*100:.1f}% vs Market: {best['market_prob']*100:.1f}%")
        print(f"   Edge: {best['edge']*100:.1f}% (w/bonus: {best.get('edge_with_bonus', best['edge'])*100:.1f}%) | Time left: {best['minutes_left']:.0f}min")
        print(f"   Momentum: dir={best.get('momentum_dir', 0):.2f} str={best.get('momentum_str', 0):.2f}")
        regime_str = best.get('regime', 'unknown')
        regime_conf = best.get('regime_confidence', 0)
        dynamic_edge = best.get('dynamic_min_edge', MIN_EDGE)
        print(f"   Regime: {regime_str} (conf={regime_conf:.0%}) | Min edge: {dynamic_edge*100:.1f}%")
        # Volatility rebalance info (T237)
        vol_ratio = best.get('vol_ratio', 1.0)
        vol_aligned = best.get('vol_aligned', False)
        vol_bonus = best.get('vol_bonus', 0)
        vol_badge = "📊 VOL ALIGNED!" if vol_aligned else ""
        print(f"   Vol ratio: {vol_ratio:.2f} | Vol bonus: +{vol_bonus*100:.1f}% {vol_badge}")
        # News sentiment info (T661 - Grok Fundamental)
        news_sent = best.get('news_sentiment', 'neutral')
        news_bonus = best.get('news_bonus', 0)
        news_conf = best.get('news_confidence', 0.5)
        if news_bonus != 0:
            news_icon = "📰🟢" if news_bonus > 0 else "📰🔴"
            print(f"   {news_icon} News: {news_sent.upper()} ({news_conf*100:.0f}%) → edge {news_bonus*100:+.2f}%")
        # Composite signal info (T460)
        composite_signals = best.get('composite_signals', 0)
        composite_conf = best.get('composite_confidence', 'low')
        composite_bonus = best.get('composite_bonus', 0)
        composite_reasons = best.get('composite_reasons', [])
        if composite_signals >= 2:
            # Multiple confirming signals - highlight this
            stars = "⭐" * composite_signals
            print(f"   {stars} COMPOSITE: {composite_signals} signals ({composite_conf}) → synergy +{composite_bonus*100:.2f}%")
            for reason in composite_reasons[:3]:
                print(f"      • {reason}")
    
    # Calculate bet size (Kelly with volatility adjustment - T293, T441)
    if cash < MIN_BET_CENTS / 100:
        print("❌ Insufficient cash")
        return
    
    # Get per-asset configuration (T441)
    asset_cfg = get_asset_config(asset_type)
    
    # Base Kelly calculation - use per-asset config, override from opportunity if present
    kelly_fraction = best.get("kelly_override", asset_cfg["kelly_fraction"])
    max_position_pct = asset_cfg["max_position_pct"]
    
    # Skip volatility adjustments for weather markets (different edge source)
    # Initialize multipliers (used in logging)
    regime_multiplier = 1.0
    vol_multiplier = 1.0
    
    if asset_type == "weather":
        # Weather uses NWS forecast accuracy - simpler sizing
        adjusted_kelly = kelly_fraction
        print(f"   📊 Weather Kelly: {kelly_fraction*100:.1f}% | Max: {max_position_pct*100:.1f}% (NWS-based)")
    else:
        # Volatility-adjusted position sizing (T293) for crypto
        # Reduce size in choppy/high-vol regimes, increase when volatility aligns
        regime = best.get("regime", "unknown")
        volatility = best.get("volatility", "normal")
        vol_aligned = best.get("vol_aligned", False)
        
        # Regime adjustment: reduce in choppy markets, slight boost in trending
        regime_multiplier = 1.0
        if regime == "choppy":
            regime_multiplier = 0.5  # Half size in choppy (hardest to trade)
        elif regime == "sideways":
            regime_multiplier = 0.75  # Reduced size in sideways
        elif regime in ("trending_bullish", "trending_bearish"):
            regime_multiplier = 1.1  # Slight boost in trending (cleaner signals)
        
        # Volatility alignment bonus: increase size when vol favors our direction
        vol_multiplier = 1.0
        if vol_aligned:
            vol_multiplier = 1.15  # 15% boost when volatility aligns
        elif volatility == "high":
            vol_multiplier = 0.8  # Reduce in high vol if not aligned
        
        # Apply adjustments to Kelly fraction
        adjusted_kelly = kelly_fraction * regime_multiplier * vol_multiplier
    
    # Get streak context for position sizing (T388: streak-based size adjustment)
    streak_ctx = get_streak_position_context()
    
    # Streak-based position sizing (T388) - reduce size during tilt risk or hot hand
    streak_multiplier = 1.0
    if streak_ctx.get("tilt_risk") or streak_ctx.get("hot_hand"):
        streak_multiplier = STREAK_SIZE_REDUCTION
        adjusted_kelly = adjusted_kelly * streak_multiplier
        streak_reason = "tilt_risk" if streak_ctx.get("tilt_risk") else "hot_hand"
        streak_count = streak_ctx.get("current_streak", 0)
        print(f"   ⚠️ Streak size adjustment: {STREAK_SIZE_REDUCTION:.0%} ({streak_reason}, {streak_count} streak)")
    
    # Latency-based position sizing (T801) - reduce size when API is slow
    latency_multiplier, latency_reason = get_latency_position_multiplier()
    if latency_multiplier == 0.0:
        # Skip trade entirely when latency is critically high
        print(f"   ⛔ SKIPPED: API latency too high ({latency_reason})")
        skip_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "latency_skip",
            "ticker": best["ticker"],
            "asset": asset_type,
            "reason": latency_reason,
            "edge": best["edge"],
        }
        log_skip(skip_data)
        return
    elif latency_multiplier < 1.0:
        adjusted_kelly = adjusted_kelly * latency_multiplier
        print(f"   ⚡ Latency size adjustment: {latency_multiplier:.0%} ({latency_reason})")
    
    # VIX-based position sizing (T611) - reduce size when fear index is elevated
    vix_multiplier = best.get("vix_size_multiplier", 1.0)
    vix_current = best.get("vix_current")
    vix_regime = best.get("vix_regime", "unknown")
    if vix_multiplier != 1.0 and vix_current:
        adjusted_kelly = adjusted_kelly * vix_multiplier
        vix_emoji = get_vix_regime_emoji(vix_regime)
        print(f"   {vix_emoji} VIX size adjustment: {vix_multiplier:.0%} (VIX={vix_current:.1f}, {vix_regime})")
    
    # Log the adjustment (T441: show per-asset config)
    total_multiplier = regime_multiplier * vol_multiplier * streak_multiplier * latency_multiplier * vix_multiplier
    if asset_type != "weather":
        print(f"   📊 {asset_type.upper()} Kelly: {kelly_fraction*100:.1f}% | Max: {max_position_pct*100:.1f}%")
    if abs(total_multiplier - 1.0) > 0.01:
        adj_direction = "↑" if total_multiplier > 1.0 else "↓"
        multiplier_parts = f"regime={regime_multiplier:.0%}, vol={vol_multiplier:.0%}"
        if streak_multiplier != 1.0:
            multiplier_parts += f", streak={streak_multiplier:.0%}"
        if latency_multiplier != 1.0:
            multiplier_parts += f", latency={latency_multiplier:.0%}"
        if vix_multiplier != 1.0:
            multiplier_parts += f", vix={vix_multiplier:.0%}"
        print(f"   Position size: {adj_direction} {total_multiplier:.0%} ({multiplier_parts})")
    
    # Use correct Kelly criterion: f* = (b*p - q) / b, then scale by our conservative fraction
    correct_kelly_f = kelly_criterion_check(best["our_prob"], best["price"])
    if correct_kelly_f <= 0:
        print(f"⛔ Kelly says don't bet (f*={correct_kelly_f:.4f}) — skipping")
        return
    # Scale by our conservative fraction (adjusted_kelly is already regime/streak/latency adjusted)
    kelly_bet = cash * min(correct_kelly_f, adjusted_kelly) * best["edge"]
    # Apply per-asset max position limit (T441)
    bet_size = max(MIN_BET_CENTS / 100, min(kelly_bet, cash * max_position_pct))
    
    # Apply holiday position size reduction (T414)
    holiday_multiplier = 1.0
    if HOLIDAY_REDUCE_SIZE and is_holiday_trading():
        holiday_multiplier = HOLIDAY_SIZE_REDUCTION
        bet_size = bet_size * holiday_multiplier
        print(f"   🎄 Holiday size reduction: {HOLIDAY_SIZE_REDUCTION*100:.0f}%")
    
    contracts = int(bet_size * 100 / best["price"])
    
    # In DRY_RUN/paper mode, force at least 1 contract for data collection
    if contracts < 1 and DRY_RUN:
        contracts = 1
        print(f"   🧪 DRY RUN: Forcing 1 contract for data collection")
    
    if contracts < 1:
        print("❌ Bet too small")
        return
    
    # Check portfolio concentration limits (T480)
    trade_value_cents = contracts * best["price"]
    bal_data = get_balance()
    portfolio_value = bal_data.get("portfolio_value", 0)  # Already in cents
    
    can_trade, conc_reason, conc_metrics = check_concentration_limits(
        positions,
        portfolio_value,
        asset_type,
        trade_value_cents
    )
    
    # Print concentration summary
    print_concentration_summary(conc_metrics)
    
    if not can_trade:
        print(f"⛔ BLOCKED by concentration limits: {conc_reason}")
        
        # Show rebalancing suggestions when blocked (T481)
        rebal = get_rebalancing_suggestions(positions, portfolio_value, conc_metrics)
        if rebal.get("needs_rebalancing"):
            print_rebalancing_suggestions(rebal)
        
        # Try auto-rebalancing if enabled to unblock future trades (T816)
        if AUTO_REBALANCE_ENABLED:
            rebal_result = check_and_auto_rebalance(positions, portfolio_value)
            if rebal_result.get("executed") and not rebal_result.get("dry_run"):
                print("   ℹ️  Auto-rebalance executed - retry trade on next cycle")
        
        # Log the skip with concentration data
        skip_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "concentration_skip",
            "ticker": best["ticker"],
            "asset": asset_type,
            "reason": conc_reason,
            "would_be_contracts": contracts,
            "would_be_value_cents": trade_value_cents,
            "concentration_metrics": conc_metrics,
            "rebalancing_suggestions": rebal.get("suggestions", [])[:3]  # Top 3 suggestions
        }
        with open(SKIP_LOG_FILE, "a") as f:
            f.write(json.dumps(skip_data) + "\n")
        return
    
    if "WARN" in conc_reason:
        print(f"   ⚠️ {conc_reason}")
    
    # streak_ctx already available from position sizing (T388)
    
    # Build trade data for logging
    trade_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "trade",
        "ticker": best["ticker"],
        "asset": best.get("asset", "btc"),
        "side": best["side"],
        "contracts": contracts,
        "price_cents": best["price"],
        "cost_cents": contracts * best["price"],
        "edge": best["edge"],
        "edge_with_bonus": best.get("edge_with_bonus", best["edge"]),
        "our_prob": best["our_prob"],
        "base_prob": best.get("base_prob", best["our_prob"]),
        "market_prob": best["market_prob"],
        "strike": best.get("strike"),  # None for weather
        "current_price": best.get("current"),  # None for weather
        "minutes_to_expiry": best.get("minutes_left"),  # None for weather
        # Weather-specific fields (T422)
        "forecast_temp": best.get("forecast_temp"),
        "forecast_uncertainty": best.get("uncertainty"),
        "city": best.get("city"),
        "momentum_dir": best.get("momentum_dir", 0),
        "momentum_str": best.get("momentum_str", 0),
        "momentum_aligned": best.get("momentum_aligned", False),
        "full_alignment": best.get("full_alignment", False),  # T395: all timeframes agree
        "regime": best.get("regime", "unknown"),
        "regime_confidence": best.get("regime_confidence", 0),
        "dynamic_min_edge": best.get("dynamic_min_edge", MIN_EDGE),
        # Volatility rebalance data (T237)
        "vol_ratio": best.get("vol_ratio", 1.0),
        "vol_aligned": best.get("vol_aligned", False),
        "vol_bonus": best.get("vol_bonus", 0),
        # Position sizing multipliers (T390, T441)
        "kelly_fraction_base": asset_cfg["kelly_fraction"],  # Per-asset base (T441)
        "kelly_fraction_used": adjusted_kelly,
        "max_position_pct": max_position_pct,  # Per-asset max (T441)
        "regime_multiplier": regime_multiplier,
        "vol_multiplier": vol_multiplier,
        "streak_multiplier": streak_multiplier,  # T388: streak-based size adjustment
        "latency_multiplier": latency_multiplier,  # T801: latency-based position sizing
        "latency_reason": latency_reason,  # T801: why latency adjustment applied
        "vix_multiplier": vix_multiplier,  # T611: VIX-based position sizing
        "vix_current": vix_current,  # T611: Current VIX level
        "vix_regime": vix_regime,  # T611: VIX regime (low_fear/moderate/elevated/high_fear)
        "holiday_multiplier": holiday_multiplier,  # T414: holiday position size reduction
        "is_holiday": is_holiday_trading(),  # T414: trading during market holiday
        "size_multiplier_total": total_multiplier * holiday_multiplier,
        # Price source data (T386)
        "price_sources": prices.get("sources", []),
        # News sentiment data (T661 - Grok Fundamental)
        "news_bonus": best.get("news_bonus", 0),
        "news_sentiment": best.get("news_sentiment", "neutral"),
        "news_confidence": best.get("news_confidence", 0.5),
        "news_reasons": best.get("news_reasons", []),
        # Portfolio concentration data (T480)
        "concentration_asset_pct": conc_metrics.get("by_asset_class", {}).get(asset_type, 0),
        "concentration_corr_group_pct": conc_metrics.get("by_correlation_group", {}).get(
            "crypto" if asset_type in ("btc", "eth") else asset_type, 0
        ),
        "concentration_warning": "WARN" in conc_reason,
        # Streak position context (T770) - trading psychology data
        "streak_context": streak_ctx.get("streak_context", "fresh_start"),
        "streak_current": streak_ctx.get("current_streak", 0),
        "streak_type": streak_ctx.get("streak_type"),
        "streak_tilt_risk": streak_ctx.get("tilt_risk", False),
        "streak_hot_hand": streak_ctx.get("hot_hand", False),
        "streak_continuation_prob": streak_ctx.get("continuation_probability"),
        "result_status": "pending"
    }
    
    if DRY_RUN:
        # Dry run mode - log without executing
        print(f"\n🧪 [DRY RUN] Would place order: {contracts} contracts @ {best['price']}¢")
        print(f"   🧪 Simulating execution (no real money at risk)")
        log_dry_run_trade(trade_data)
        print(f"✅ [DRY RUN] Trade logged to {DRY_RUN_LOG_FILE}")
        
        # Paper mode: also log additional opportunities for more data
        if len(all_opportunities) > 1:
            extra_count = 0
            for extra_opp in all_opportunities[1:5]:  # Top 5 total
                extra_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "trade",
                    "ticker": extra_opp["ticker"],
                    "asset": extra_opp.get("asset", "btc"),
                    "side": extra_opp["side"],
                    "contracts": 1,
                    "price_cents": extra_opp["price"],
                    "cost_cents": extra_opp["price"],
                    "edge": extra_opp["edge"],
                    "edge_with_bonus": extra_opp.get("edge_with_bonus", extra_opp["edge"]),
                    "our_prob": extra_opp["our_prob"],
                    "base_prob": extra_opp.get("base_prob", extra_opp["our_prob"]),
                    "market_prob": extra_opp["market_prob"],
                    "strike": extra_opp.get("strike", 0),
                    "current_price": extra_opp.get("current_price", 0),
                    "minutes_to_expiry": extra_opp.get("minutes_left", 0),
                    "momentum_aligned": extra_opp.get("momentum_aligned", False),
                    "regime": extra_opp.get("regime", "unknown"),
                    "dynamic_min_edge": extra_opp.get("dynamic_min_edge", 0),
                    "result_status": "pending",
                    "dry_run": True
                }
                log_dry_run_trade(extra_data)
                extra_count += 1
                print(f"   🧪 +{extra_count}: {extra_opp['ticker']} {extra_opp['side'].upper()} @{extra_opp['price']}¢ edge:{extra_opp['edge']*100:.1f}%")
            if extra_count:
                print(f"   📊 Total paper trades this cycle: {extra_count + 1}")
        return
    
    # Print streak position warning/note if applicable (T770)
    print_streak_position_warning(streak_ctx)
    
    print(f"\n💸 Placing order: {contracts} contracts @ {best['price']}¢")
    
    # Place order (with latency tracking)
    order_start = time.time()
    result = place_order(best["ticker"], best["side"], contracts, best["price"])
    order_end = time.time()
    latency_ms = int((order_end - order_start) * 1000)
    
    if "error" in result:
        print(f"❌ Order error: {result['error']}")
        # Log failed execution (T329)
        log_execution(
            ticker=best["ticker"],
            side=best["side"],
            count=contracts,
            price_cents=best["price"],
            status="error",
            latency_ms=latency_ms,
            error=result['error']
        )
        return
    
    order = result.get("order", {})
    order_status = order.get("status", "unknown")
    order_id = order.get("order_id", order.get("id"))
    
    if order_status == "executed":
        print(f"✅ Order executed!")
        
        # Log trade (with momentum + regime data + latency)
        print(f"   ⏱️  Order latency: {latency_ms}ms")
        trade_data["latency_ms"] = latency_ms
        log_trade(trade_data)
        
        # Log ML features for future model training (T331)
        log_ml_features(trade_data)
        
        # Log successful execution (T329)
        log_execution(
            ticker=best["ticker"],
            side=best["side"],
            count=contracts,
            price_cents=best["price"],
            status="executed",
            latency_ms=latency_ms,
            order_id=order_id
        )
        
        # Check for latency alerts (after logging trade with new latency data)
        check_latency_alert()
        
        # Check for extreme volatility alerts (T294)
        check_extreme_vol_alert(trade_data)
        
        # Check for tilt risk alerts (T774) - alert when trading in risky state
        if streak_ctx.get("tilt_risk"):
            write_tilt_risk_alert(trade_data, streak_ctx)
    else:
        print(f"⏳ Order status: {order_status}")
        # Log pending/other status (T329)
        log_execution(
            ticker=best["ticker"],
            side=best["side"],
            count=contracts,
            price_cents=best["price"],
            status=order_status,
            latency_ms=latency_ms,
            order_id=order_id
        )


# ============== HEALTH STATUS ENDPOINT (T472) ==============
def get_today_stats() -> dict:
    """Get today's trading statistics"""
    log_path = Path(TRADE_LOG_FILE)
    if not log_path.exists():
        return {"trades": 0, "won": 0, "lost": 0, "pending": 0, "win_rate": 0.0, "pnl_cents": 0}
    
    today = datetime.now(timezone.utc).date()
    trades_today = []
    
    with open(log_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("type") == "trade":
                    ts = entry.get("timestamp", "")
                    if ts.startswith(str(today)):
                        trades_today.append(entry)
            except:
                pass
    
    won = sum(1 for t in trades_today if t.get("result_status") == "won")
    lost = sum(1 for t in trades_today if t.get("result_status") == "lost")
    pending = sum(1 for t in trades_today if t.get("result_status") == "pending")
    settled = won + lost
    win_rate = (won / settled * 100) if settled > 0 else 0.0
    
    pnl = 0
    for t in trades_today:
        if t.get("result_status") == "won":
            # When winning: get paid 100¢ per contract minus entry price
            pnl += (100 - t.get("price_cents", 0)) * t.get("contracts", 0)
        elif t.get("result_status") == "lost":
            # When losing: lose entry price
            pnl -= t.get("price_cents", 0) * t.get("contracts", 0)
    
    return {
        "trades": len(trades_today),
        "won": won,
        "lost": lost,
        "pending": pending,
        "win_rate": round(win_rate, 1),
        "pnl_cents": pnl
    }


# ============== HTTP HEALTH SERVER (T828) ==============

class HealthHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for /health endpoint."""
    
    # Track server start time
    server_start_time = None
    
    def log_message(self, format, *args):
        """Suppress default logging to avoid noise."""
        pass
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self.send_health_response()
        elif self.path == "/ready":
            self.send_ready_response()
        elif self.path == "/metrics":
            self.send_prometheus_metrics()
        else:
            self.send_error(404, "Not Found")
    
    def send_health_response(self):
        """Return health status JSON."""
        try:
            health_path = Path(HEALTH_STATUS_FILE)
            if health_path.exists():
                with open(health_path) as f:
                    health_data = json.load(f)
            else:
                health_data = {"is_running": True, "status": "starting"}
            
            # Add server uptime
            if HealthHandler.server_start_time:
                uptime_seconds = (datetime.now(timezone.utc) - HealthHandler.server_start_time).total_seconds()
                health_data["uptime_seconds"] = int(uptime_seconds)
                health_data["uptime_human"] = str(timedelta(seconds=int(uptime_seconds)))
            
            # Add server info
            health_data["health_server"] = {
                "port": HEALTH_SERVER_PORT,
                "endpoints": ["/health", "/ready", "/metrics"]
            }
            
            response = json.dumps(health_data, indent=2)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(response))
            self.end_headers()
            self.wfile.write(response.encode())
            
        except Exception as e:
            error_response = json.dumps({"error": str(e), "is_running": False})
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(error_response.encode())
    
    def send_ready_response(self):
        """Simple readiness check - returns 200 if server is up."""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")
    
    def send_prometheus_metrics(self):
        """Return metrics in Prometheus text exposition format (T858)."""
        try:
            # Load health status for metrics
            health_path = Path(HEALTH_STATUS_FILE)
            health = {}
            if health_path.exists():
                with open(health_path) as f:
                    health = json.load(f)
            
            # Build Prometheus metrics (text format)
            lines = [
                "# HELP kalshi_autotrader_info Autotrader metadata",
                "# TYPE kalshi_autotrader_info gauge",
                f'kalshi_autotrader_info{{dry_run="{str(health.get("dry_run", True)).lower()}"}} 1',
                "",
                "# HELP kalshi_trades_today_total Total trades today by outcome",
                "# TYPE kalshi_trades_today_total counter",
                f'kalshi_trades_today_total{{outcome="won"}} {health.get("today_won", 0)}',
                f'kalshi_trades_today_total{{outcome="lost"}} {health.get("today_lost", 0)}',
                f'kalshi_trades_today_total{{outcome="pending"}} {health.get("today_pending", 0)}',
                "",
                "# HELP kalshi_win_rate_today Current day win rate (0-1)",
                "# TYPE kalshi_win_rate_today gauge",
                f'kalshi_win_rate_today {health.get("win_rate_today", 0) / 100 if health.get("win_rate_today") else 0}',
                "",
                "# HELP kalshi_pnl_today_cents PnL today in cents",
                "# TYPE kalshi_pnl_today_cents gauge",
                f'kalshi_pnl_today_cents {health.get("pnl_today_cents", 0)}',
                "",
                "# HELP kalshi_positions_count Number of open positions",
                "# TYPE kalshi_positions_count gauge",
                f'kalshi_positions_count {health.get("positions_count", 0) or 0}',
                "",
                "# HELP kalshi_cash_cents Available cash balance in cents",
                "# TYPE kalshi_cash_cents gauge",
                f'kalshi_cash_cents {health.get("cash_cents", 0) or 0}',
                "",
                "# HELP kalshi_circuit_breaker_active Circuit breaker status (1=paused, 0=running)",
                "# TYPE kalshi_circuit_breaker_active gauge",
                f'kalshi_circuit_breaker_active {1 if health.get("circuit_breaker_active") else 0}',
                "",
                "# HELP kalshi_consecutive_losses Current consecutive loss streak",
                "# TYPE kalshi_consecutive_losses gauge",
                f'kalshi_consecutive_losses {health.get("consecutive_losses", 0)}',
                "",
                "# HELP kalshi_cycle_count Total trading cycles completed",
                "# TYPE kalshi_cycle_count counter",
                f'kalshi_cycle_count {health.get("cycle_count", 0)}',
                "",
            ]
            
            # Add uptime if available
            if HealthHandler.server_start_time:
                uptime = (datetime.now(timezone.utc) - HealthHandler.server_start_time).total_seconds()
                lines.extend([
                    "# HELP kalshi_uptime_seconds Autotrader uptime in seconds",
                    "# TYPE kalshi_uptime_seconds counter",
                    f'kalshi_uptime_seconds {int(uptime)}',
                    "",
                ])
            
            response = "\n".join(lines)
            
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", len(response))
            self.end_headers()
            self.wfile.write(response.encode())
            
        except Exception as e:
            error_msg = f"# Error generating metrics: {e}\n"
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(error_msg.encode())


def start_health_server():
    """Start the HTTP health server in a background thread."""
    if not HEALTH_SERVER_ENABLED:
        print(f"   ⚠️ Health server disabled (set HEALTH_SERVER_ENABLED=true to enable)")
        return None
    
    try:
        server = HTTPServer(("0.0.0.0", HEALTH_SERVER_PORT), HealthHandler)
        HealthHandler.server_start_time = datetime.now(timezone.utc)
        
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        
        print(f"   📡 Health server started on http://0.0.0.0:{HEALTH_SERVER_PORT}/health (/metrics for Prometheus)")
        return server
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"   ⚠️ Health server port {HEALTH_SERVER_PORT} already in use (another instance running?)")
        else:
            print(f"   ⚠️ Failed to start health server: {e}")
        return None
    except Exception as e:
        print(f"   ⚠️ Failed to start health server: {e}")
        return None


def write_health_status(cycle_count: int):
    """Write autotrader health status to JSON file for external monitoring (T472)"""
    try:
        # Gather health data
        today_stats = get_today_stats()
        
        # Get balance (safely)
        try:
            balance_data = get_balance()
            cash_cents = balance_data.get("available_balance", 0)
        except Exception:
            cash_cents = None
        
        # Get positions count (safely)
        try:
            positions = get_positions()
            positions_count = len(positions)
        except Exception:
            positions_count = None
        
        # Get circuit breaker status (returns 3 values: paused, losses, message)
        is_paused, _, pause_reason = check_circuit_breaker()
        
        # Get consecutive losses
        consecutive_losses = get_consecutive_losses()
        
        # Build health status
        health = {
            "is_running": True,
            "last_cycle_time": datetime.now(timezone.utc).isoformat(),
            "cycle_count": cycle_count,
            "dry_run": DRY_RUN,
            "trades_today": today_stats["trades"],
            "today_won": today_stats["won"],
            "today_lost": today_stats["lost"],
            "today_pending": today_stats["pending"],
            "win_rate_today": today_stats["win_rate"],
            "pnl_today_cents": today_stats["pnl_cents"],
            "positions_count": positions_count,
            "cash_cents": cash_cents,
            "circuit_breaker_active": is_paused,
            "circuit_breaker_reason": pause_reason if is_paused else None,
            "consecutive_losses": consecutive_losses,
            "status": "🧪 dry_run" if DRY_RUN else ("⏸️ paused" if is_paused else "✅ running")
        }
        
        # Ensure directory exists
        health_path = Path(HEALTH_STATUS_FILE)
        health_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write atomically (write to temp then rename)
        temp_path = health_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(health, f, indent=2)
        temp_path.rename(health_path)
        
    except Exception as e:
        # Don't crash the autotrader for health status failures
        print(f"⚠️ Failed to write health status: {e}")


def main():
    """Main entry point"""
    print("🚀 Starting Kalshi AutoTrader v2")
    print("   With PROPER probability model and feedback loop!")
    print("   Now trading: BTC (KXBTCD) + ETH (KXETHD) markets!")
    print("   📊 API latency profiling enabled (T279)")
    if DRY_RUN:
        print("   🧪 DRY RUN MODE - No real trades will be executed!")
        print(f"   📝 Trades logged to: {DRY_RUN_LOG_FILE}")
    
    # Start HTTP health server (T828)
    health_server = start_health_server()
    
    print("   Press Ctrl+C to stop\n")
    
    cycle_count = 0
    
    while True:
        try:
            run_cycle()
            cycle_count += 1
            
            # Write health status after each cycle (T472)
            write_health_status(cycle_count)
            
            # Print and save latency profile every 6 cycles (30 mins)
            if cycle_count % 6 == 0:
                print_latency_summary()
                save_latency_profile()
                
                # Check for bottlenecks
                bottlenecks = identify_bottlenecks()
                if bottlenecks:
                    print("\n⚠️ BOTTLENECKS DETECTED:")
                    for endpoint, issue, details in bottlenecks:
                        print(f"   • {endpoint}: {issue} - {details}")
                
                # Check rate limits (T308)
                print(get_rate_limit_summary())
                log_rate_limits()
                rate_warnings = check_rate_limits()
                if rate_warnings:
                    print("\n⚠️ RATE LIMIT WARNING:")
                    for source, pct, msg in rate_warnings:
                        print(f"   • {msg}")
                    write_rate_limit_alert(rate_warnings)
                
        except KeyboardInterrupt:
            print("\n\n👋 Stopping autotrader...")
            # Print final latency summary
            print("\n📊 FINAL LATENCY REPORT:")
            print_latency_summary()
            save_latency_profile()
            break
        except Exception as e:
            import traceback
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
        
        # Wait between cycles - shorter in paper mode for more data
        if DRY_RUN:
            print("\n💤 Sleeping 1 minute (paper mode - fast cycle)...")
            time.sleep(60)
        else:
            print("\n💤 Sleeping 5 minutes...")
            time.sleep(300)


# ============== SAFETY LIMITS (Added after 0% WR disaster) ==============
MAX_DAILY_LOSS_CENTS = 500  # $5 max daily loss
MAX_TRADES_PER_HOUR = 3     # Limit trade frequency
PAPER_TRADE_MODE = True     # Start in paper trade mode!

def check_daily_loss():
    """Check if we've hit daily loss limit"""
    log_path = Path(TRADE_LOG_FILE)
    if not log_path.exists():
        return 0
    
    today = datetime.now(timezone.utc).date()
    daily_loss = 0
    
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("type") == "trade":
                ts = entry.get("timestamp", "")
                if ts.startswith(str(today)):
                    pnl = entry.get("profit_cents", 0)
                    if pnl < 0:
                        daily_loss += abs(pnl)
    
    return daily_loss


def check_hourly_trades():
    """Count trades in last hour"""
    log_path = Path(TRADE_LOG_FILE)
    if not log_path.exists():
        return 0
    
    hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    count = 0
    
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("type") == "trade":
                ts = entry.get("timestamp", "")
                try:
                    trade_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if trade_time > hour_ago:
                        count += 1
                except:
                    pass
    
    return count


def get_consecutive_losses() -> int:
    """
    Count consecutive losses from most recent settled trades.
    Returns the current loss streak count (0 if last trade was a win).
    """
    log_path = Path(TRADE_LOG_FILE)
    if not log_path.exists():
        return 0
    
    # Read all trades and get settled ones
    settled_trades = []
    with open(log_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("type") == "trade" and entry.get("result_status") in ("won", "lost"):
                    settled_trades.append(entry)
            except:
                pass
    
    if not settled_trades:
        return 0
    
    # Sort by timestamp descending (most recent first)
    settled_trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Count consecutive losses from most recent
    consecutive_losses = 0
    for trade in settled_trades:
        if trade.get("result_status") == "lost":
            consecutive_losses += 1
        else:
            break  # Hit a win, stop counting
    
    return consecutive_losses


def load_circuit_breaker_state() -> dict:
    """Load circuit breaker state from file."""
    try:
        with open(CIRCUIT_BREAKER_STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "paused": False, 
            "pause_time": None, 
            "streak_at_pause": 0,
            "forgiven_losses": 0,  # T762: Losses forgiven after cooldown release
            "last_cooldown_release": None  # T762: Track last cooldown release for decay
        }


def save_circuit_breaker_state(state: dict):
    """Save circuit breaker state to file."""
    with open(CIRCUIT_BREAKER_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_effective_losses(raw_losses: int, state: dict) -> int:
    """
    T762: Calculate effective loss count with forgiveness and time decay.
    
    - Subtracts forgiven_losses from raw count
    - Applies time-based decay: -1 loss every 2 hours since last cooldown release
    
    Returns effective loss count (min 0).
    """
    forgiven = state.get("forgiven_losses", 0)
    last_release_str = state.get("last_cooldown_release")
    
    # Apply forgiven losses
    effective = raw_losses - forgiven
    
    # Apply time-based decay if we had a cooldown release
    if last_release_str:
        try:
            last_release = datetime.fromisoformat(last_release_str)
            if last_release.tzinfo is None:
                last_release = last_release.replace(tzinfo=timezone.utc)
            hours_since_release = (datetime.now(timezone.utc) - last_release).total_seconds() / 3600
            # Decay 1 loss every 2 hours
            decay = int(hours_since_release / 2)
            effective -= decay
        except:
            pass
    
    return max(0, effective)


def check_circuit_breaker() -> tuple:
    """
    Check if circuit breaker should trigger or is active.
    
    T762: Now includes time-based decay and loss forgiveness after cooldown release.
    
    Returns:
        (should_pause: bool, consecutive_losses: int, message: str)
    """
    state = load_circuit_breaker_state()
    raw_consecutive_losses = get_consecutive_losses()
    
    # If already paused, check if we should resume
    if state.get("paused"):
        pause_time_str = state.get("pause_time")
        cooldown_elapsed = False
        hours_since_pause = 0
        
        if pause_time_str:
            try:
                pause_time = datetime.fromisoformat(pause_time_str)
                if pause_time.tzinfo is None:
                    pause_time = pause_time.replace(tzinfo=timezone.utc)
                hours_since_pause = (datetime.now(timezone.utc) - pause_time).total_seconds() / 3600
                cooldown_elapsed = hours_since_pause >= CIRCUIT_BREAKER_COOLDOWN_HOURS
            except:
                pass
        
        # Resume if EITHER we got a win OR cooldown period has elapsed
        if raw_consecutive_losses == 0:
            # We got a win! Resume trading and clear forgiveness
            trades_skipped = get_trades_skipped_count()
            log_circuit_breaker_event(
                event_type="release",
                consecutive_losses=state.get("streak_at_pause", 0),
                release_reason="win",
                trades_skipped=trades_skipped,
                trigger_time=pause_time_str
            )
            state["paused"] = False
            state["pause_time"] = None
            state["streak_at_pause"] = 0
            state["forgiven_losses"] = 0  # T762: Clear on win
            state["last_cooldown_release"] = None
            save_circuit_breaker_state(state)
            return (False, 0, "✅ Circuit breaker released - got a win!")
        elif cooldown_elapsed:
            # T762: Cooldown elapsed - forgive the losses that triggered the pause
            # This prevents immediate re-trigger after cooldown
            trades_skipped = get_trades_skipped_count()
            streak_at_pause = state.get("streak_at_pause", 0)
            
            log_circuit_breaker_event(
                event_type="release",
                consecutive_losses=raw_consecutive_losses,
                release_reason="cooldown",
                trades_skipped=trades_skipped,
                trigger_time=pause_time_str,
                forgiven_losses=streak_at_pause  # T762: Log forgiveness
            )
            
            state["paused"] = False
            state["pause_time"] = None
            state["forgiven_losses"] = raw_consecutive_losses  # T762: Forgive current streak
            state["last_cooldown_release"] = datetime.now(timezone.utc).isoformat()  # T762: Track for decay
            state["streak_at_pause"] = 0
            save_circuit_breaker_state(state)
            return (False, 0, f"⏰ Circuit breaker released - {CIRCUIT_BREAKER_COOLDOWN_HOURS}h cooldown elapsed, {raw_consecutive_losses} losses forgiven")
        else:
            hours_remaining = CIRCUIT_BREAKER_COOLDOWN_HOURS - hours_since_pause
            return (True, raw_consecutive_losses, f"⏸️ Circuit breaker ACTIVE ({raw_consecutive_losses} losses, {hours_remaining:.1f}h until cooldown, or win to resume)")
    
    # T762: Calculate effective losses with forgiveness and decay
    effective_losses = get_effective_losses(raw_consecutive_losses, state)
    
    # Check if we should trigger circuit breaker (using effective losses)
    if effective_losses >= CIRCUIT_BREAKER_THRESHOLD:
        trigger_time = datetime.now(timezone.utc).isoformat()
        state["paused"] = True
        state["pause_time"] = trigger_time
        state["streak_at_pause"] = raw_consecutive_losses
        # T762: Clear forgiveness when triggering fresh
        state["forgiven_losses"] = 0
        state["last_cooldown_release"] = None
        save_circuit_breaker_state(state)
        
        # Log trigger event to history (T471)
        log_circuit_breaker_event(
            event_type="trigger",
            consecutive_losses=raw_consecutive_losses,
            effective_losses=effective_losses,  # T762: Log effective too
            trigger_time=trigger_time
        )
        
        # Write alert file for heartbeat pickup
        write_circuit_breaker_alert(raw_consecutive_losses)
        
        return (True, raw_consecutive_losses, f"🚨 CIRCUIT BREAKER TRIGGERED! {effective_losses} effective losses (raw: {raw_consecutive_losses})")
    
    # T762: Show both raw and effective in status message
    if raw_consecutive_losses > 0 and effective_losses < raw_consecutive_losses:
        return (False, effective_losses, f"✓ Streak: {effective_losses} effective losses (raw: {raw_consecutive_losses}, forgiven/decayed: {raw_consecutive_losses - effective_losses})")
    return (False, effective_losses, f"✓ Streak: {effective_losses} losses (threshold: {CIRCUIT_BREAKER_THRESHOLD})")


def write_circuit_breaker_alert(consecutive_losses: int):
    """Write circuit breaker alert file for heartbeat notification."""
    alert_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "circuit_breaker",
        "consecutive_losses": consecutive_losses,
        "threshold": CIRCUIT_BREAKER_THRESHOLD,
        "cooldown_hours": CIRCUIT_BREAKER_COOLDOWN_HOURS,
        "message": f"🚨 CIRCUIT BREAKER TRIGGERED!\n\n"
                   f"AutoTrader paused after {consecutive_losses} consecutive losses.\n\n"
                   f"This is a safety measure to prevent tilt trading.\n"
                   f"Trading will resume after:\n"
                   f"• A winning trade settles, OR\n"
                   f"• {CIRCUIT_BREAKER_COOLDOWN_HOURS}h cooldown period elapses\n\n"
                   f"Actions you can take:\n"
                   f"• Wait for pending trades to settle (a win will resume)\n"
                   f"• Wait for cooldown ({CIRCUIT_BREAKER_COOLDOWN_HOURS}h from now)\n"
                   f"• Manually reset: `rm scripts/kalshi-circuit-breaker.json`\n"
                   f"• Adjust threshold: `export CIRCUIT_BREAKER_THRESHOLD=10`\n"
                   f"• Adjust cooldown: `export CIRCUIT_BREAKER_COOLDOWN_HOURS=2`"
    }
    
    with open(CIRCUIT_BREAKER_ALERT_FILE, "w") as f:
        json.dump(alert_data, f, indent=2)
    
    print(f"📝 Circuit breaker alert written to {CIRCUIT_BREAKER_ALERT_FILE}")


# ============== CIRCUIT BREAKER HISTORY LOGGING (T471) ==============

def log_circuit_breaker_event(
    event_type: str,  # "trigger" or "release"
    consecutive_losses: int,
    release_reason: str = None,  # "win", "cooldown", "manual"
    trades_skipped: int = 0,
    trigger_time: str = None,
    release_time: str = None,
    effective_losses: int = None,  # T762: Effective count after forgiveness/decay
    forgiven_losses: int = None,  # T762: How many losses were forgiven
):
    """
    Log circuit breaker trigger/release events to history file.
    
    Tracks:
    - trigger_time: When circuit breaker was triggered
    - release_time: When circuit breaker was released
    - release_reason: Why it was released (win/cooldown/manual)
    - streak_at_trigger: Consecutive losses when triggered
    - trades_skipped: How many trading opportunities were skipped while paused
    - T762: effective_losses and forgiven_losses for decay tracking
    """
    now = datetime.now(timezone.utc).isoformat()
    
    entry = {
        "timestamp": now,
        "event_type": event_type,
        "consecutive_losses": consecutive_losses,
        "threshold": CIRCUIT_BREAKER_THRESHOLD,
        "cooldown_hours": CIRCUIT_BREAKER_COOLDOWN_HOURS,
    }
    
    # T762: Add effective/forgiven losses if provided
    if effective_losses is not None:
        entry["effective_losses"] = effective_losses
    if forgiven_losses is not None:
        entry["forgiven_losses"] = forgiven_losses
    
    if event_type == "trigger":
        entry["trigger_time"] = trigger_time or now
    elif event_type == "release":
        entry["release_time"] = release_time or now
        entry["release_reason"] = release_reason
        entry["trigger_time"] = trigger_time
        entry["trades_skipped"] = trades_skipped
        
        # Calculate pause duration if we have trigger_time
        if trigger_time:
            try:
                trigger_dt = datetime.fromisoformat(trigger_time)
                release_dt = datetime.fromisoformat(release_time or now)
                if trigger_dt.tzinfo is None:
                    trigger_dt = trigger_dt.replace(tzinfo=timezone.utc)
                if release_dt.tzinfo is None:
                    release_dt = release_dt.replace(tzinfo=timezone.utc)
                entry["pause_duration_hours"] = round((release_dt - trigger_dt).total_seconds() / 3600, 2)
            except:
                pass
    
    # Append to history file
    try:
        with open(CIRCUIT_BREAKER_HISTORY_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"📊 Circuit breaker event logged: {event_type}")
    except Exception as e:
        print(f"⚠️ Failed to log circuit breaker event: {e}")


def get_trades_skipped_count() -> int:
    """
    Count trading opportunities that were skipped while circuit breaker was active.
    This is tracked via skip logs with reason containing 'circuit_breaker'.
    """
    try:
        state = load_circuit_breaker_state()
        pause_time_str = state.get("pause_time")
        if not pause_time_str:
            return 0
        
        pause_time = datetime.fromisoformat(pause_time_str)
        if pause_time.tzinfo is None:
            pause_time = pause_time.replace(tzinfo=timezone.utc)
        
        # Count opportunities that would have been taken since pause
        # We estimate based on cycles run (each cycle is ~1 min)
        hours_paused = (datetime.now(timezone.utc) - pause_time).total_seconds() / 3600
        # Rough estimate: ~60 cycles per hour, avg 0.5 opportunities per cycle
        return int(hours_paused * 30)  # Estimated skipped opportunities
    except:
        return 0


# ============== LATENCY ALERTING (T295) ==============

def get_recent_latencies(n: int = LATENCY_CHECK_WINDOW) -> list:
    """
    Get latencies from the last N trades.
    Returns list of latency_ms values.
    """
    log_path = Path(TRADE_LOG_FILE)
    if not log_path.exists():
        return []
    
    trades_with_latency = []
    with open(log_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("type") == "trade" and "latency_ms" in entry:
                    trades_with_latency.append(entry)
            except:
                pass
    
    # Sort by timestamp descending (most recent first)
    trades_with_latency.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Return latencies from last N trades
    return [t["latency_ms"] for t in trades_with_latency[:n]]


def check_latency_alert():
    """
    Check if average latency exceeds threshold and write alert if needed.
    Has cooldown to prevent alert spam.
    """
    latencies = get_recent_latencies()
    if len(latencies) < 3:
        # Not enough data to determine trend
        return
    
    avg_latency = sum(latencies) / len(latencies)
    
    if avg_latency <= LATENCY_THRESHOLD_MS:
        return  # All good
    
    # Check cooldown
    alert_path = Path(LATENCY_ALERT_FILE)
    if alert_path.exists():
        try:
            stat = alert_path.stat()
            age_seconds = time.time() - stat.st_mtime
            if age_seconds < LATENCY_ALERT_COOLDOWN:
                return  # Cooldown active
        except:
            pass
    
    # Write alert
    write_latency_alert(avg_latency, latencies)


def write_latency_alert(avg_latency: float, latencies: list):
    """Write latency alert file for heartbeat pickup."""
    max_latency = max(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    
    alert_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "latency",
        "avg_latency_ms": round(avg_latency),
        "max_latency_ms": max_latency,
        "min_latency_ms": min_latency,
        "threshold_ms": LATENCY_THRESHOLD_MS,
        "sample_size": len(latencies),
        "message": f"⚠️ HIGH ORDER LATENCY\n\n"
                   f"Average latency: {avg_latency:.0f}ms (threshold: {LATENCY_THRESHOLD_MS}ms)\n"
                   f"Last {len(latencies)} trades: min {min_latency}ms / max {max_latency}ms\n\n"
                   f"This could indicate:\n"
                   f"• Kalshi API slowdown\n"
                   f"• Network connectivity issues\n"
                   f"• Rate limiting\n\n"
                   f"Check autotrader logs for details."
    }
    
    with open(LATENCY_ALERT_FILE, "w") as f:
        json.dump(alert_data, f, indent=2)
    
    print(f"📝 Latency alert written: avg {avg_latency:.0f}ms > {LATENCY_THRESHOLD_MS}ms threshold")


# ============== EXTREME VOLATILITY ALERTING (T294) ==============

def check_extreme_vol_alert(trade_data: dict):
    """
    Check if trade was placed during extreme volatility and alert if so.
    Alerts when volatility is "very_high" (>2% hourly range).
    """
    volatility = trade_data.get("volatility", "normal")
    
    # Only alert on very_high volatility
    if volatility != "very_high":
        return
    
    # Check cooldown
    try:
        if os.path.exists(EXTREME_VOL_ALERT_FILE):
            mtime = os.path.getmtime(EXTREME_VOL_ALERT_FILE)
            if time.time() - mtime < EXTREME_VOL_ALERT_COOLDOWN:
                print(f"   ℹ️ Extreme vol alert on cooldown ({(EXTREME_VOL_ALERT_COOLDOWN - (time.time() - mtime)):.0f}s remaining)")
                return
    except Exception:
        pass
    
    # Write alert
    write_extreme_vol_alert(trade_data)


def write_extreme_vol_alert(trade_data: dict):
    """Write extreme volatility alert file for heartbeat pickup."""
    ticker = trade_data.get("ticker", "unknown")
    asset = trade_data.get("asset", "btc").upper()
    side = trade_data.get("side", "unknown")
    edge = trade_data.get("edge", 0)
    price = trade_data.get("price_cents", 0)
    contracts = trade_data.get("contracts", 0)
    regime = trade_data.get("regime", "unknown")
    vol_ratio = trade_data.get("vol_ratio", 1.0)
    
    alert_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "extreme_volatility",
        "ticker": ticker,
        "asset": asset,
        "side": side,
        "edge_pct": round(edge * 100, 1),
        "regime": regime,
        "vol_ratio": round(vol_ratio, 2),
        "trade_details": {
            "contracts": contracts,
            "price_cents": price,
            "cost_cents": contracts * price
        },
        "message": f"⚠️ EXTREME VOLATILITY TRADE\n\n"
                   f"📊 {asset} is experiencing very high volatility (>2% hourly range)\n\n"
                   f"Trade placed:\n"
                   f"• Ticker: {ticker}\n"
                   f"• Side: {side.upper()}\n"
                   f"• Contracts: {contracts} @ {price}¢\n"
                   f"• Edge: {edge*100:.1f}%\n"
                   f"• Regime: {regime}\n"
                   f"• Vol ratio: {vol_ratio:.2f}x\n\n"
                   f"⚠️ High volatility increases both opportunity and risk.\n"
                   f"Monitor this position closely!"
    }
    
    with open(EXTREME_VOL_ALERT_FILE, "w") as f:
        json.dump(alert_data, f, indent=2)
    
    print(f"📝 🌋 Extreme volatility alert written: {asset} @ very_high vol")


# ============== STREAK POSITION CONTEXT (T770) ==============

def get_streak_position_context() -> dict:
    """
    Get current streak position context for trade decisions.
    
    Returns dict with:
    - streak_context: str like "after_3_losses", "after_2_wins", "fresh_start"
    - current_streak: int - current streak count
    - streak_type: str - "win" or "loss"
    - tilt_risk: bool - True if after STREAK_TILT_THRESHOLD+ losses
    - hot_hand: bool - True if after STREAK_HOT_HAND_THRESHOLD+ wins
    - continuation_probability: float or None - probability of streak continuing
    """
    # Get current streak info
    streaks = calculate_current_streaks()
    current = streaks.get("current_streak", 0)
    streak_type = streaks.get("current_streak_type")
    
    # Build context string
    if current == 0 or streak_type is None:
        context = "fresh_start"
    else:
        context = f"after_{current}_{streak_type}{'es' if streak_type == 'loss' else 's'}"
    
    # Check for tilt risk (after multiple losses)
    tilt_risk = streak_type == "loss" and current >= STREAK_TILT_THRESHOLD
    
    # Check for hot hand (after multiple wins)
    hot_hand = streak_type == "win" and current >= STREAK_HOT_HAND_THRESHOLD
    
    # Try to load continuation probability from streak analysis
    continuation_prob = None
    try:
        analysis_path = Path(STREAK_POSITION_ANALYSIS_FILE)
        if analysis_path.exists():
            with open(analysis_path) as f:
                analysis = json.load(f)
            
            # Find matching context in analysis
            by_context = analysis.get("by_context", {})
            for ctx_key, ctx_data in by_context.items():
                # Match patterns like "after_3_losses" 
                if streak_type and current > 0:
                    ctx_streak = int(ctx_key.split("_")[1]) if "_" in ctx_key else 0
                    ctx_type = "win" if "win" in ctx_key else ("loss" if "loss" in ctx_key else None)
                    if ctx_type == streak_type and ctx_streak == current:
                        continuation_prob = ctx_data.get("continuation_rate", ctx_data.get("win_rate"))
                        break
    except Exception as e:
        # Silently fail if analysis file doesn't exist or is malformed
        pass
    
    return {
        "streak_context": context,
        "current_streak": current,
        "streak_type": streak_type,
        "tilt_risk": tilt_risk,
        "hot_hand": hot_hand,
        "continuation_probability": continuation_prob,
        "longest_win_streak": streaks.get("longest_win_streak", 0),
        "longest_loss_streak": streaks.get("longest_loss_streak", 0)
    }


def print_streak_position_warning(streak_ctx: dict):
    """Print warning/note about streak position when entering trade."""
    if streak_ctx.get("tilt_risk"):
        print(f"\n⚠️  TILT RISK WARNING: Entering trade after {streak_ctx['current_streak']} consecutive losses!")
        print(f"    💡 Consider: Is this a high-conviction trade? Strategy might need review.")
        if streak_ctx.get("continuation_probability") is not None:
            print(f"    📊 Historical continuation rate: {streak_ctx['continuation_probability']:.1%}")
    
    elif streak_ctx.get("hot_hand"):
        print(f"\n🔥 HOT HAND: {streak_ctx['current_streak']} consecutive wins!")
        print(f"    💡 Note: Stay disciplined - don't let success lead to overconfidence.")
        if streak_ctx.get("continuation_probability") is not None:
            print(f"    📊 Historical continuation rate: {streak_ctx['continuation_probability']:.1%}")


def write_tilt_risk_alert(trade_data: dict, streak_ctx: dict):
    """
    Write tilt risk alert file for heartbeat to send to Telegram (T774).
    Called when a trade is placed while in tilt risk state.
    """
    alert_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "tilt_risk_trade",
        "ticker": trade_data.get("ticker"),
        "asset": trade_data.get("asset"),
        "side": trade_data.get("side"),
        "contracts": trade_data.get("contracts"),
        "price_cents": trade_data.get("price_cents"),
        "edge": trade_data.get("edge"),
        "streak_context": streak_ctx.get("streak_context"),
        "consecutive_losses": streak_ctx.get("current_streak"),
        "continuation_probability": streak_ctx.get("continuation_probability"),
        "message": (
            f"⚠️ TILT RISK TRADE PLACED\n\n"
            f"Ticker: {trade_data.get('ticker')}\n"
            f"Side: {trade_data.get('side', '').upper()}\n"
            f"Contracts: {trade_data.get('contracts')}\n"
            f"Edge: {trade_data.get('edge', 0):.1%}\n\n"
            f"📉 After {streak_ctx.get('current_streak')} consecutive losses!\n"
            f"Context: {streak_ctx.get('streak_context')}\n"
            + (f"📊 Historical continuation: {streak_ctx.get('continuation_probability'):.1%}\n" 
               if streak_ctx.get('continuation_probability') is not None else "")
            + "\n💡 Consider reviewing strategy if losses continue."
        )
    }
    
    with open(TILT_RISK_ALERT_FILE, "w") as f:
        json.dump(alert_data, f, indent=2)
    
    print(f"📝 Tilt risk alert written to {TILT_RISK_ALERT_FILE}")


# ============== STREAK RECORD ALERTING (T288) ==============

def load_streak_records() -> dict:
    """Load streak records from file."""
    try:
        with open(STREAK_STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"longest_win_streak": 0, "longest_loss_streak": 0, "updated_at": None}


def save_streak_records(records: dict):
    """Save streak records to file."""
    records["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(STREAK_STATE_FILE, "w") as f:
        json.dump(records, f, indent=2)


def calculate_current_streaks() -> dict:
    """
    Calculate current win/loss streak and longest streaks from trade log.
    Returns dict with: current_streak, current_streak_type, longest_win_streak, longest_loss_streak
    """
    log_path = Path(TRADE_LOG_FILE)
    if not log_path.exists():
        return {
            "current_streak": 0,
            "current_streak_type": None,
            "longest_win_streak": 0,
            "longest_loss_streak": 0
        }
    
    # Read all settled trades
    settled_trades = []
    with open(log_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("type") == "trade" and entry.get("result_status") in ("won", "lost", "win", "loss"):
                    settled_trades.append(entry)
            except:
                pass
    
    if not settled_trades:
        return {
            "current_streak": 0,
            "current_streak_type": None,
            "longest_win_streak": 0,
            "longest_loss_streak": 0
        }
    
    # Sort by timestamp ascending (oldest first)
    settled_trades.sort(key=lambda x: x.get("timestamp", ""))
    
    # Calculate streaks
    current_streak = 0
    current_type = None
    longest_win = 0
    longest_loss = 0
    
    for trade in settled_trades:
        # Normalize result status (win/won, loss/lost)
        result = trade.get("result_status", "")
        is_win = result in ("won", "win")
        is_loss = result in ("lost", "loss")
        
        if is_win:
            if current_type == "win":
                current_streak += 1
            else:
                current_streak = 1
                current_type = "win"
            longest_win = max(longest_win, current_streak)
        elif is_loss:
            if current_type == "loss":
                current_streak += 1
            else:
                current_streak = 1
                current_type = "loss"
            longest_loss = max(longest_loss, current_streak)
    
    return {
        "current_streak": current_streak,
        "current_streak_type": current_type,
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss
    }


def check_streak_records() -> str:
    """
    Check if current streaks have hit new records. Alert if so.
    Call this after trades settle.
    
    Returns: Status message
    """
    current = calculate_current_streaks()
    saved = load_streak_records()
    
    alerts = []
    
    # Check for new win streak record
    if current["longest_win_streak"] > saved["longest_win_streak"] and current["longest_win_streak"] >= 3:
        alerts.append({
            "type": "win_record",
            "new_record": current["longest_win_streak"],
            "old_record": saved["longest_win_streak"],
            "emoji": "🏆🔥"
        })
        saved["longest_win_streak"] = current["longest_win_streak"]
    
    # Check for new loss streak record (not something to celebrate, but important to track)
    if current["longest_loss_streak"] > saved["longest_loss_streak"] and current["longest_loss_streak"] >= 3:
        alerts.append({
            "type": "loss_record",
            "new_record": current["longest_loss_streak"],
            "old_record": saved["longest_loss_streak"],
            "emoji": "💀📉"
        })
        saved["longest_loss_streak"] = current["longest_loss_streak"]
    
    # Save updated records
    if alerts:
        save_streak_records(saved)
        write_streak_alert(alerts, current)
        msg = f"🎖️ New streak record(s): {', '.join([a['type'] for a in alerts])}"
        print(msg)
        return msg
    
    return f"✓ Streaks: current={current['current_streak']} ({current['current_streak_type']}), " \
           f"best win={current['longest_win_streak']}, worst loss={current['longest_loss_streak']}"


def write_streak_alert(alerts: list, current: dict):
    """Write streak record alert file for heartbeat pickup."""
    
    messages = []
    for alert in alerts:
        if alert["type"] == "win_record":
            messages.append(
                f"🏆🔥 NEW WIN STREAK RECORD!\n\n"
                f"Consecutive wins: {alert['new_record']}\n"
                f"Previous record: {alert['old_record']}\n\n"
                f"The strategy is on fire! 🎉"
            )
        elif alert["type"] == "loss_record":
            messages.append(
                f"💀📉 NEW LOSS STREAK RECORD\n\n"
                f"Consecutive losses: {alert['new_record']}\n"
                f"Previous worst: {alert['old_record']}\n\n"
                f"Consider pausing to review strategy.\n"
                f"Circuit breaker: {CIRCUIT_BREAKER_THRESHOLD} consecutive losses"
            )
    
    alert_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "streak_record",
        "alerts": alerts,
        "current_streak": current["current_streak"],
        "current_streak_type": current["current_streak_type"],
        "longest_win_streak": current["longest_win_streak"],
        "longest_loss_streak": current["longest_loss_streak"],
        "message": "\n\n---\n\n".join(messages)
    }
    
    with open(STREAK_ALERT_FILE, "w") as f:
        json.dump(alert_data, f, indent=2)
    
    print(f"📝 Streak record alert written to {STREAK_ALERT_FILE}")


# ============== MAIN ENTRY POINT ==============
def reset_circuit_breaker_manual():
    """
    T762: Manually reset the circuit breaker.
    Call via: python3 kalshi-autotrader-v2.py --reset-circuit-breaker
    """
    state = load_circuit_breaker_state()
    raw_losses = get_consecutive_losses()
    
    if not state.get("paused"):
        print("ℹ️ Circuit breaker is not currently active.")
        print(f"   Current streak: {raw_losses} losses")
        return
    
    # Log the manual release
    log_circuit_breaker_event(
        event_type="release",
        consecutive_losses=raw_losses,
        release_reason="manual",
        trades_skipped=get_trades_skipped_count(),
        trigger_time=state.get("pause_time"),
        forgiven_losses=raw_losses
    )
    
    # Reset state with forgiveness for current losses
    state["paused"] = False
    state["pause_time"] = None
    state["streak_at_pause"] = 0
    state["forgiven_losses"] = raw_losses  # Forgive current streak
    state["last_cooldown_release"] = datetime.now(timezone.utc).isoformat()
    save_circuit_breaker_state(state)
    
    print(f"✅ Circuit breaker manually reset!")
    print(f"   {raw_losses} losses forgiven")
    print(f"   Trading will resume on next cycle")


if __name__ == "__main__":
    import sys
    
    # T762: Handle --reset-circuit-breaker command
    if len(sys.argv) > 1 and sys.argv[1] == "--reset-circuit-breaker":
        reset_circuit_breaker_manual()
    # T789: Handle --ignore-schedule flag
    elif "--ignore-schedule" in sys.argv:
        TRADING_SCHEDULE_ENABLED = False
        print("⚠️ Trading schedule check DISABLED (--ignore-schedule)")
        main()
    else:
        main()
