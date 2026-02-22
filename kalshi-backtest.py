#!/usr/bin/env python3
"""
Kalshi AutoTrader - Backtesting Framework (TRADE-003)

Replay historical trade data to evaluate strategy performance across
parameter variations. Uses actual trade logs (dry-run + live) and
finetuning signal data to simulate different strategy configurations.

Features:
  - Replay historical trades with configurable strategy parameters
  - Sweep MIN_EDGE, KELLY_FRACTION, momentum/regime filters, etc.
  - Evaluate: win rate, PnL, Sharpe ratio, max drawdown, expectation
  - Per-asset and per-regime breakdown
  - Detailed reports with actionable parameter recommendations
  - Monte Carlo bootstrap confidence intervals

Data sources:
  - scripts/kalshi-trades-v2.jsonl        (live trades)
  - scripts/kalshi-trades-dryrun.jsonl    (paper trades)
  - data/trading/kalshi-trades-*.jsonl    (daily snapshots)
  - data/trading/ml-training-data.jsonl   (ML feature vectors)
  - data/finetuning/*.jsonl               (signal events)

Usage:
    python scripts/kalshi-backtest.py                   # Full backtest with defaults
    python scripts/kalshi-backtest.py --sweep            # Parameter sweep
    python scripts/kalshi-backtest.py --report           # Analyze existing trades only
    python scripts/kalshi-backtest.py --monte-carlo 1000 # Bootstrap confidence intervals
    python scripts/kalshi-backtest.py --compare v1 v2    # Compare strategy versions

Author: Clawdbot (TRADE-003)
"""

import json
import math
import sys
import os
import copy
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

# ============== CONFIGURATION ==============

# Default strategy parameters (mirrors kalshi-autotrader-v2.py defaults)
DEFAULT_PARAMS = {
    "min_edge": 0.04,
    "max_edge": 0.20,  # TRADE-005: Reduced from 0.25 — backtest found >25% edges were false positives
    "kelly_fraction": 0.05,
    "max_position_pct": 0.03,
    "min_time_to_expiry_minutes": 45,
    "fat_tail_multiplier": 1.4,
    # Per-asset overrides
    "btc_kelly": 0.10,
    "btc_max_position_pct": 0.05,
    "btc_min_edge": 0.04,
    "eth_kelly": 0.08,
    "eth_max_position_pct": 0.04,
    "eth_min_edge": 0.05,
    "sol_kelly": 0.05,
    "sol_max_position_pct": 0.03,
    "sol_min_edge": 0.06,
    "weather_kelly": 0.05,
    "weather_max_position_pct": 0.02,
    "weather_min_edge": 0.15,
    # Filters
    "use_momentum_filter": True,
    "momentum_conflict_threshold": 0.3,
    "use_regime_filter": True,
    "use_kelly_check": True,
    "use_news_bonus": True,
    "use_divergence_bonus": True,
    "use_composite_scoring": True,
    # Streak management
    "streak_tilt_reduction": 0.7,
    "streak_tilt_threshold": 3,
    # Position sizing multipliers
    "regime_choppy_multiplier": 0.5,
    "regime_trending_multiplier": 1.0,
    "regime_sideways_multiplier": 0.75,
}

# Parameter sweep ranges
SWEEP_RANGES = {
    "min_edge": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
    "kelly_fraction": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30],
    "fat_tail_multiplier": [1.0, 1.2, 1.4, 1.6, 1.8],
    "momentum_conflict_threshold": [0.2, 0.3, 0.4, 0.5],
    "min_time_to_expiry_minutes": [30, 45, 60, 90],
    "streak_tilt_reduction": [0.5, 0.7, 0.85, 1.0],
}

# File paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
LIVE_TRADES_FILE = PROJECT_ROOT / "scripts" / "kalshi-trades-v2.jsonl"
DRYRUN_TRADES_FILE = PROJECT_ROOT / "scripts" / "kalshi-trades-dryrun.jsonl"
ML_DATA_FILE = PROJECT_ROOT / "data" / "trading" / "ml-training-data.jsonl"
DAILY_TRADES_DIR = PROJECT_ROOT / "data" / "trading"
FINETUNING_DIR = PROJECT_ROOT / "data" / "finetuning"

REPORT_OUTPUT = PROJECT_ROOT / "data" / "trading" / "backtest-report.json"
SWEEP_OUTPUT = PROJECT_ROOT / "data" / "trading" / "backtest-sweep.json"


# ============== DATA STRUCTURES ==============

@dataclass
class Trade:
    """Normalized trade record for backtesting."""
    timestamp: str
    ticker: str
    asset: str  # btc, eth, sol, weather
    side: str  # yes, no
    contracts: int
    price_cents: int
    cost_cents: int
    edge: float
    edge_with_bonus: float
    our_prob: float
    base_prob: float
    market_prob: float
    strike: Optional[float]
    current_price: Optional[float]
    minutes_to_expiry: Optional[float]
    # Momentum
    momentum_dir: float = 0.0
    momentum_str: float = 0.0
    momentum_aligned: bool = False
    full_alignment: bool = False
    # Regime
    regime: str = "unknown"
    regime_confidence: float = 0.0
    dynamic_min_edge: float = 0.04
    # Volatility
    vol_ratio: float = 1.0
    vol_aligned: bool = False
    vol_bonus: float = 0.0
    # News
    news_bonus: float = 0.0
    news_sentiment: str = "neutral"
    news_confidence: float = 0.5
    # Divergence
    divergence_bonus: float = 0.0
    divergence_type: str = "none"
    # Composite
    composite_signals: int = 0
    composite_confidence: str = "low"
    composite_bonus: float = 0.0
    # Sizing
    kelly_fraction_used: float = 0.05
    size_multiplier_total: float = 1.0
    regime_multiplier: float = 1.0
    streak_multiplier: float = 1.0
    vix_multiplier: float = 1.0
    latency_multiplier: float = 1.0
    holiday_multiplier: float = 1.0
    # Streak context
    streak_current: int = 0
    streak_type: str = ""
    streak_tilt_risk: bool = False
    # Result (ground truth)
    result_status: str = "pending"  # pending, win, loss
    profit_cents: Optional[int] = None
    # Source
    source: str = "unknown"  # live, dryrun, daily
    dry_run: bool = False


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    params: dict
    total_trades: int = 0
    included_trades: int = 0
    excluded_trades: int = 0
    wins: int = 0
    losses: int = 0
    pending: int = 0
    total_cost_cents: int = 0
    total_profit_cents: int = 0
    gross_win_cents: int = 0
    gross_loss_cents: int = 0
    win_rate: float = 0.0
    expectation_per_trade_cents: float = 0.0
    roi_pct: float = 0.0
    max_drawdown_cents: int = 0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_edge: float = 0.0
    avg_win_cents: float = 0.0
    avg_loss_cents: float = 0.0
    best_trade_cents: int = 0
    worst_trade_cents: int = 0
    # Per-asset breakdown
    by_asset: dict = field(default_factory=dict)
    # Per-regime breakdown
    by_regime: dict = field(default_factory=dict)
    # Per-side breakdown
    by_side: dict = field(default_factory=dict)
    # Equity curve (cumulative PnL over time)
    equity_curve: list = field(default_factory=list)
    # Trade list for drill-down
    trades_detail: list = field(default_factory=list)
    # Exclusion reasons
    exclusion_reasons: dict = field(default_factory=dict)


# ============== DATA LOADING ==============

def load_trades_from_jsonl(filepath: Path, source: str = "unknown") -> list[Trade]:
    """Load trades from a JSONL file, normalizing to Trade dataclass."""
    trades = []
    if not filepath.exists():
        return trades

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Only process actual trade entries
            if d.get("type") not in ("trade", None):
                continue
            # ML data file doesn't have "type" field
            if source == "ml" and "prob_predicted" in d:
                pass  # Allow
            elif d.get("type") != "trade":
                continue

            # Skip if no result (can't evaluate)
            result = d.get("result_status", "pending")

            # Determine asset
            asset = d.get("asset", "")
            if not asset:
                ticker = d.get("ticker", "")
                if "KXBTC" in ticker:
                    asset = "btc"
                elif "KXETH" in ticker:
                    asset = "eth"
                elif "KXSOL" in ticker:
                    asset = "sol"
                elif any(x in ticker for x in ["HIGH", "LOW"]):
                    asset = "weather"
                else:
                    asset = "other"

            try:
                trade = Trade(
                    timestamp=d.get("timestamp", ""),
                    ticker=d.get("ticker", ""),
                    asset=asset,
                    side=d.get("side", ""),
                    contracts=d.get("contracts", 1),
                    price_cents=d.get("price_cents", 0),
                    cost_cents=d.get("cost_cents", 0),
                    edge=d.get("edge", 0),
                    edge_with_bonus=d.get("edge_with_bonus", d.get("edge", 0)),
                    our_prob=d.get("our_prob", d.get("prob_predicted", 0)),
                    base_prob=d.get("base_prob", d.get("prob_base", 0)),
                    market_prob=d.get("market_prob", d.get("prob_market", 0)),
                    strike=d.get("strike"),
                    current_price=d.get("current_price", d.get("price_current")),
                    minutes_to_expiry=d.get("minutes_to_expiry"),
                    momentum_dir=d.get("momentum_dir", d.get("momentum_direction", 0)),
                    momentum_str=d.get("momentum_str", d.get("momentum_strength", 0)),
                    momentum_aligned=bool(d.get("momentum_aligned", False)),
                    full_alignment=bool(d.get("full_alignment", d.get("momentum_full_alignment", False))),
                    regime=d.get("regime", "unknown"),
                    regime_confidence=d.get("regime_confidence", 0),
                    dynamic_min_edge=d.get("dynamic_min_edge", 0.04),
                    vol_ratio=d.get("vol_ratio", 1.0),
                    vol_aligned=bool(d.get("vol_aligned", False)),
                    vol_bonus=d.get("vol_bonus", 0),
                    news_bonus=d.get("news_bonus", 0),
                    news_sentiment=d.get("news_sentiment", "neutral"),
                    news_confidence=d.get("news_confidence", 0.5),
                    divergence_bonus=d.get("divergence_bonus", 0),
                    divergence_type=d.get("divergence_type", "none"),
                    composite_signals=d.get("composite_signals", 0),
                    composite_confidence=d.get("composite_confidence", "low"),
                    composite_bonus=d.get("composite_bonus", 0),
                    kelly_fraction_used=d.get("kelly_fraction_used", 0.05),
                    size_multiplier_total=d.get("size_multiplier_total", 1.0),
                    regime_multiplier=d.get("regime_multiplier", 1.0),
                    streak_multiplier=d.get("streak_multiplier", 1.0),
                    vix_multiplier=d.get("vix_multiplier", 1.0),
                    latency_multiplier=d.get("latency_multiplier", 1.0),
                    holiday_multiplier=d.get("holiday_multiplier", 1.0),
                    streak_current=d.get("streak_current", 0),
                    streak_type=d.get("streak_type", ""),
                    streak_tilt_risk=bool(d.get("streak_tilt_risk", False)),
                    result_status=result,
                    profit_cents=d.get("profit_cents"),
                    source=source,
                    dry_run=bool(d.get("dry_run", False)),
                )
                trades.append(trade)
            except Exception:
                continue

    return trades


def load_all_trades() -> list[Trade]:
    """Load trades from all available sources, deduplicated by timestamp+ticker."""
    all_trades = []

    # 1. Live trades (highest priority)
    if LIVE_TRADES_FILE.exists():
        live = load_trades_from_jsonl(LIVE_TRADES_FILE, "live")
        all_trades.extend(live)
        print(f"  📄 Live trades: {len(live)}")

    # 2. Dry-run trades
    if DRYRUN_TRADES_FILE.exists():
        dryrun = load_trades_from_jsonl(DRYRUN_TRADES_FILE, "dryrun")
        all_trades.extend(dryrun)
        print(f"  📄 Dry-run trades: {len(dryrun)}")

    # 3. Daily trade snapshots
    daily_count = 0
    if DAILY_TRADES_DIR.exists():
        for f in sorted(DAILY_TRADES_DIR.glob("kalshi-trades-2026-*.jsonl")):
            daily = load_trades_from_jsonl(f, "daily")
            all_trades.extend(daily)
            daily_count += len(daily)
    if daily_count:
        print(f"  📄 Daily snapshot trades: {daily_count}")

    # 4. ML training data (has structured features)
    if ML_DATA_FILE.exists():
        ml = load_trades_from_jsonl(ML_DATA_FILE, "ml")
        all_trades.extend(ml)
        if ml:
            print(f"  📄 ML training data: {len(ml)}")

    # Deduplicate by (timestamp, ticker, side) - prefer live > dryrun > daily > ml
    source_priority = {"live": 0, "dryrun": 1, "daily": 2, "ml": 3}
    seen = {}
    for t in all_trades:
        key = (t.timestamp[:19], t.ticker, t.side)  # Truncate timestamp for matching
        existing = seen.get(key)
        if existing is None or source_priority.get(t.source, 99) < source_priority.get(existing.source, 99):
            seen[key] = t

    deduped = sorted(seen.values(), key=lambda t: t.timestamp)
    print(f"  ✅ Total unique trades: {len(deduped)}")

    # Breakdown
    settled = [t for t in deduped if t.result_status in ("win", "loss")]
    pending = [t for t in deduped if t.result_status == "pending"]
    print(f"     Settled: {len(settled)} ({sum(1 for t in settled if t.result_status=='win')}W / {sum(1 for t in settled if t.result_status=='loss')}L)")
    print(f"     Pending: {len(pending)}")

    return deduped


def load_signal_events() -> dict:
    """Load signal events from finetuning data for context analysis."""
    events = {
        "regime_changes": [],
        "momentum_changes": [],
        "whipsaws": [],
        "divergences": [],
        "vol_recalibrations": [],
    }

    mapping = {
        "regime-change.jsonl": "regime_changes",
        "momentum-change.jsonl": "momentum_changes",
        "whipsaw.jsonl": "whipsaws",
        "momentum-divergence.jsonl": "divergences",
        "vol-recalibration.jsonl": "vol_recalibrations",
    }

    for filename, key in mapping.items():
        filepath = FINETUNING_DIR / filename
        if not filepath.exists():
            continue
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    events[key].append(d)
                except json.JSONDecodeError:
                    continue

    total = sum(len(v) for v in events.values())
    if total:
        print(f"  📊 Signal events loaded: {total} total")
        for k, v in events.items():
            if v:
                print(f"     {k}: {len(v)}")

    return events


# ============== STRATEGY SIMULATION ==============

def norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)


def recalculate_prob(trade: Trade, params: dict) -> float:
    """
    Recalculate our probability using the given fat_tail_multiplier.
    Only applies to crypto trades where we have strike + current_price.
    For weather or trades without strike, return the original our_prob.
    """
    if trade.asset == "weather" or trade.strike is None or trade.current_price is None:
        return trade.our_prob
    if trade.minutes_to_expiry is None or trade.minutes_to_expiry <= 0:
        return trade.our_prob

    hourly_vol_map = {"btc": 0.005, "eth": 0.007, "sol": 0.012}
    hourly_vol = hourly_vol_map.get(trade.asset, 0.005)

    T = trade.minutes_to_expiry / 60.0
    sigma = hourly_vol * math.sqrt(T) * params.get("fat_tail_multiplier", 1.4)

    if sigma <= 0:
        return 1.0 if trade.current_price > trade.strike else 0.0

    if trade.strike <= 0 or trade.current_price <= 0:
        return trade.our_prob

    log_ratio = math.log(trade.current_price / trade.strike)
    d2 = log_ratio / sigma - sigma / 2
    prob_above = norm_cdf(d2)
    prob_above = max(0.01, min(0.99, prob_above))

    if trade.side == "yes":
        return prob_above
    else:
        return 1.0 - prob_above


def kelly_criterion_check(our_prob: float, price_cents: int) -> float:
    """Calculate Kelly fraction for a binary bet."""
    if price_cents <= 0 or price_cents >= 100 or our_prob <= 0:
        return 0.0
    b = (100 - price_cents) / price_cents
    p = our_prob
    q = 1 - p
    f = (b * p - q) / b
    return max(0.0, f)


def should_include_trade(trade: Trade, params: dict) -> tuple[bool, str]:
    """
    Apply strategy filters to decide whether this trade should be taken.
    Returns (include, reason_if_excluded).
    """
    asset = trade.asset
    side = trade.side

    # 1. Minimum edge check (per-asset)
    asset_min_edge = params.get(f"{asset}_min_edge", params.get("min_edge", 0.04))
    if trade.edge < asset_min_edge:
        return False, f"edge_{trade.edge:.3f}_below_{asset_min_edge:.3f}"

    # 2. Maximum edge check (suspiciously high = model error)
    # TRADE-005: Per-asset max edge caps — weather has strictest cap (false positives in backtest)
    BACKTEST_ASSET_MAX_EDGE = {
        "btc": 0.20, "eth": 0.20, "sol": 0.20,
        "weather": 0.15, "other": 0.15,
    }
    global_max_edge = params.get("max_edge", 0.20)
    asset_max_edge = BACKTEST_ASSET_MAX_EDGE.get(asset, 0.15)
    effective_max_edge = min(global_max_edge, asset_max_edge)
    if trade.edge > effective_max_edge:
        return False, f"edge_{trade.edge:.3f}_above_max_{effective_max_edge:.3f}_{asset}"

    # 3. Time to expiry check
    min_expiry = params.get("min_time_to_expiry_minutes", 45)
    if trade.minutes_to_expiry is not None and trade.minutes_to_expiry < min_expiry:
        return False, f"expiry_{trade.minutes_to_expiry:.0f}min_below_{min_expiry}min"

    # 4. Momentum conflict filter
    if params.get("use_momentum_filter", True):
        threshold = params.get("momentum_conflict_threshold", 0.3)
        if side == "yes" and trade.momentum_dir < -threshold and trade.momentum_str > threshold:
            return False, "momentum_conflict_yes_vs_bearish"
        if side == "no" and trade.momentum_dir > threshold and trade.momentum_str > threshold:
            return False, "momentum_conflict_no_vs_bullish"

    # 5. Kelly criterion check (is trade +EV?)
    if params.get("use_kelly_check", True):
        kelly_f = kelly_criterion_check(trade.our_prob, trade.price_cents)
        if kelly_f <= 0:
            return False, "negative_kelly"

    # 6. Regime-based dynamic edge override
    if params.get("use_regime_filter", True):
        if trade.regime == "choppy":
            dynamic_edge = params.get("min_edge", 0.04) * 1.5  # Higher bar in choppy
            if trade.edge < dynamic_edge:
                return False, f"regime_choppy_edge_{trade.edge:.3f}_below_{dynamic_edge:.3f}"

    return True, ""


def simulate_position_size(trade: Trade, params: dict, bankroll_cents: int,
                           current_streak: int = 0) -> int:
    """
    Calculate position size (cost in cents) for this trade given parameters.
    """
    asset = trade.asset
    kelly_frac = params.get(f"{asset}_kelly", params.get("kelly_fraction", 0.05))
    max_pos_pct = params.get(f"{asset}_max_position_pct", params.get("max_position_pct", 0.03))

    # Regime multiplier
    regime_mult = 1.0
    if trade.regime == "choppy":
        regime_mult = params.get("regime_choppy_multiplier", 0.5)
    elif trade.regime in ("trending_bullish", "trending_bearish"):
        regime_mult = params.get("regime_trending_multiplier", 1.0)
    elif trade.regime == "sideways":
        regime_mult = params.get("regime_sideways_multiplier", 0.75)

    # Streak tilt reduction
    streak_mult = 1.0
    tilt_threshold = params.get("streak_tilt_threshold", 3)
    if abs(current_streak) >= tilt_threshold:
        streak_mult = params.get("streak_tilt_reduction", 0.7)

    effective_kelly = kelly_frac * regime_mult * streak_mult
    effective_kelly = min(effective_kelly, max_pos_pct)

    bet_cents = int(bankroll_cents * effective_kelly)
    bet_cents = max(5, bet_cents)  # Minimum 5 cents

    # Cap at actual trade cost
    contracts = max(1, bet_cents // trade.price_cents) if trade.price_cents > 0 else 1
    cost = contracts * trade.price_cents

    return cost


def run_backtest(trades: list[Trade], params: dict,
                 initial_bankroll_cents: int = 10000,
                 verbose: bool = False) -> BacktestResult:
    """
    Run a full backtest simulation over historical trades.

    Replays trades chronologically, applying strategy filters and position sizing.
    """
    result = BacktestResult(params=params)
    bankroll = initial_bankroll_cents
    peak_bankroll = bankroll
    max_dd_cents = 0
    equity = [bankroll]
    pnl_series = []
    streak = 0  # positive = wins, negative = losses
    exclusion_reasons = defaultdict(int)

    # Per-category stats
    cat_stats = lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0, "cost": 0}
    by_asset = defaultdict(cat_stats)
    by_regime = defaultdict(cat_stats)
    by_side = defaultdict(cat_stats)

    for trade in trades:
        # Skip pending trades (no ground truth)
        if trade.result_status not in ("win", "loss"):
            result.pending += 1
            continue

        # Optionally recalculate probability with different fat_tail
        if params.get("fat_tail_multiplier") != 1.4:
            recalc_prob = recalculate_prob(trade, params)
            # Update edge based on recalculated probability
            trade = copy.copy(trade)
            trade.our_prob = recalc_prob
            if trade.side == "yes":
                trade.edge = recalc_prob - trade.market_prob
            else:
                trade.edge = recalc_prob - (1.0 - trade.market_prob if trade.market_prob < 1 else 0)

        # Apply strategy filters
        include, reason = should_include_trade(trade, params)
        if not include:
            result.excluded_trades += 1
            exclusion_reasons[reason] += 1
            continue

        # Calculate position size
        cost = simulate_position_size(trade, params, bankroll, streak)
        if cost <= 0 or cost > bankroll:
            result.excluded_trades += 1
            exclusion_reasons["insufficient_bankroll"] += 1
            continue

        result.included_trades += 1
        result.total_trades += 1
        result.total_cost_cents += cost

        # Determine PnL
        contracts = max(1, cost // trade.price_cents) if trade.price_cents > 0 else 1
        if trade.result_status == "win":
            pnl = (100 * contracts) - cost  # Win = payout - cost
            result.wins += 1
            result.gross_win_cents += pnl
            streak = max(0, streak) + 1
        else:
            pnl = -cost  # Loss = lost cost
            result.losses += 1
            result.gross_loss_cents += abs(pnl)
            streak = min(0, streak) - 1

        result.total_profit_cents += pnl
        bankroll += pnl
        pnl_series.append(pnl)

        # Track equity curve
        equity.append(bankroll)
        peak_bankroll = max(peak_bankroll, bankroll)
        dd = peak_bankroll - bankroll
        if dd > max_dd_cents:
            max_dd_cents = dd

        # Per-category tracking
        for cat_key, cat_dict in [(trade.asset, by_asset),
                                   (trade.regime, by_regime),
                                   (trade.side, by_side)]:
            cat_dict[cat_key]["trades"] += 1
            cat_dict[cat_key]["cost"] += cost
            cat_dict[cat_key]["pnl"] += pnl
            if trade.result_status == "win":
                cat_dict[cat_key]["wins"] += 1
            else:
                cat_dict[cat_key]["losses"] += 1

        # Trade detail (limit storage)
        if len(result.trades_detail) < 500:
            result.trades_detail.append({
                "timestamp": trade.timestamp,
                "ticker": trade.ticker,
                "asset": trade.asset,
                "side": trade.side,
                "edge": round(trade.edge, 4),
                "regime": trade.regime,
                "result": trade.result_status,
                "pnl_cents": pnl,
                "bankroll_after": bankroll,
            })

    # Compute aggregate metrics
    settled = result.wins + result.losses
    if settled > 0:
        result.win_rate = result.wins / settled
        result.expectation_per_trade_cents = result.total_profit_cents / settled
    if result.total_cost_cents > 0:
        result.roi_pct = (result.total_profit_cents / result.total_cost_cents) * 100

    result.max_drawdown_cents = max_dd_cents
    if peak_bankroll > 0:
        result.max_drawdown_pct = (max_dd_cents / peak_bankroll) * 100

    # Sharpe ratio (annualized, approximate)
    if pnl_series and len(pnl_series) > 1:
        import statistics
        mean_pnl = statistics.mean(pnl_series)
        std_pnl = statistics.stdev(pnl_series)
        if std_pnl > 0:
            # Approximate annualization: assume ~5 trades/day, 365 days
            trades_per_year = 5 * 365
            result.sharpe_ratio = (mean_pnl / std_pnl) * math.sqrt(trades_per_year)

    # Profit factor
    if result.gross_loss_cents > 0:
        result.profit_factor = result.gross_win_cents / result.gross_loss_cents

    # Average edge
    edges = [t.edge for t in trades if t.result_status in ("win", "loss")]
    if edges:
        result.avg_edge = sum(edges) / len(edges)

    # Avg win/loss size
    if result.wins > 0:
        result.avg_win_cents = result.gross_win_cents / result.wins
    if result.losses > 0:
        result.avg_loss_cents = result.gross_loss_cents / result.losses

    # Best/worst trade
    if pnl_series:
        result.best_trade_cents = max(pnl_series)
        result.worst_trade_cents = min(pnl_series)

    # Format category stats with win rates
    def fmt_cat(stats_dict):
        out = {}
        for k, v in stats_dict.items():
            total = v["wins"] + v["losses"]
            v["win_rate"] = v["wins"] / total if total > 0 else 0
            v["roi_pct"] = (v["pnl"] / v["cost"] * 100) if v["cost"] > 0 else 0
            out[k] = v
        return out

    result.by_asset = fmt_cat(by_asset)
    result.by_regime = fmt_cat(by_regime)
    result.by_side = fmt_cat(by_side)
    result.equity_curve = equity
    result.exclusion_reasons = dict(exclusion_reasons)

    return result


# ============== PARAMETER SWEEP ==============

def run_parameter_sweep(trades: list[Trade], sweep_param: str = None,
                        verbose: bool = True) -> list[dict]:
    """
    Run backtests across multiple parameter values.
    If sweep_param is specified, only sweep that parameter.
    Otherwise sweep all parameters in SWEEP_RANGES.
    """
    results = []
    params_to_sweep = {sweep_param: SWEEP_RANGES[sweep_param]} if sweep_param else SWEEP_RANGES

    total_runs = sum(len(v) for v in params_to_sweep.values())
    run_idx = 0

    for param_name, values in params_to_sweep.items():
        for value in values:
            run_idx += 1
            test_params = dict(DEFAULT_PARAMS)
            test_params[param_name] = value

            # For per-asset min_edge, also update the individual overrides
            if param_name == "min_edge":
                test_params["btc_min_edge"] = value
                test_params["eth_min_edge"] = value * 1.25
                test_params["sol_min_edge"] = value * 1.5

            bt = run_backtest(trades, test_params)

            summary = {
                "param": param_name,
                "value": value,
                "trades": bt.included_trades,
                "win_rate": round(bt.win_rate, 4),
                "pnl_cents": bt.total_profit_cents,
                "roi_pct": round(bt.roi_pct, 2),
                "sharpe": round(bt.sharpe_ratio, 3),
                "max_dd_pct": round(bt.max_drawdown_pct, 2),
                "profit_factor": round(bt.profit_factor, 3),
                "expectation": round(bt.expectation_per_trade_cents, 2),
            }
            results.append(summary)

            if verbose:
                print(f"  [{run_idx}/{total_runs}] {param_name}={value}: "
                      f"{bt.included_trades} trades, WR={bt.win_rate:.1%}, "
                      f"PnL=${bt.total_profit_cents/100:+.2f}, "
                      f"Sharpe={bt.sharpe_ratio:.2f}, "
                      f"MaxDD={bt.max_drawdown_pct:.1f}%")

    return results


# ============== MONTE CARLO BOOTSTRAP ==============

def monte_carlo_bootstrap(trades: list[Trade], params: dict,
                          n_simulations: int = 1000,
                          verbose: bool = True) -> dict:
    """
    Bootstrap confidence intervals for backtest metrics.

    Resamples settled trades with replacement to estimate
    the distribution of win rate, PnL, and Sharpe ratio.
    """
    # Filter to settled, included trades
    settled = [t for t in trades if t.result_status in ("win", "loss")]
    included = [t for t in settled if should_include_trade(t, params)[0]]

    if len(included) < 10:
        print(f"  ⚠️ Only {len(included)} settled+included trades, need ≥10 for bootstrap")
        return {}

    win_rates = []
    pnls = []
    sharpes = []

    for i in range(n_simulations):
        # Resample with replacement
        sample = random.choices(included, k=len(included))
        bt = run_backtest(sample, params)
        win_rates.append(bt.win_rate)
        pnls.append(bt.total_profit_cents)
        sharpes.append(bt.sharpe_ratio)

        if verbose and (i + 1) % 200 == 0:
            print(f"  [{i+1}/{n_simulations}] bootstrapping...")

    def percentiles(data, pcts=(5, 25, 50, 75, 95)):
        s = sorted(data)
        n = len(s)
        return {f"p{p}": s[int(n * p / 100)] for p in pcts}

    result = {
        "n_simulations": n_simulations,
        "n_trades": len(included),
        "win_rate": {
            "mean": sum(win_rates) / len(win_rates),
            **percentiles(win_rates),
        },
        "pnl_cents": {
            "mean": sum(pnls) / len(pnls),
            **percentiles(pnls),
        },
        "sharpe_ratio": {
            "mean": sum(sharpes) / len(sharpes),
            **percentiles(sharpes),
        },
    }

    if verbose:
        wr = result["win_rate"]
        pnl = result["pnl_cents"]
        sr = result["sharpe_ratio"]
        print(f"\n  📊 Monte Carlo ({n_simulations} sims, {len(included)} trades):")
        print(f"     Win Rate:  {wr['mean']:.1%}  [{wr['p5']:.1%} - {wr['p95']:.1%}] 90% CI")
        print(f"     PnL:       ${pnl['mean']/100:+.2f}  [${pnl['p5']/100:+.2f} - ${pnl['p95']/100:+.2f}]")
        print(f"     Sharpe:    {sr['mean']:.2f}  [{sr['p5']:.2f} - {sr['p95']:.2f}]")

    return result


# ============== REPORTING ==============

def format_report(bt: BacktestResult, signals: dict = None) -> str:
    """Format a human-readable backtest report."""
    lines = []
    lines.append("=" * 70)
    lines.append("📊 KALSHI AUTOTRADER - BACKTEST REPORT")
    lines.append("=" * 70)

    # Parameters
    lines.append(f"\n⚙️  Parameters:")
    key_params = ["min_edge", "kelly_fraction", "fat_tail_multiplier",
                  "min_time_to_expiry_minutes", "momentum_conflict_threshold"]
    for p in key_params:
        lines.append(f"     {p}: {bt.params.get(p, 'N/A')}")

    # Summary
    lines.append(f"\n📈 SUMMARY:")
    lines.append(f"     Total trades evaluated: {bt.total_trades + bt.excluded_trades + bt.pending}")
    lines.append(f"     Included trades:        {bt.included_trades}")
    lines.append(f"     Excluded trades:        {bt.excluded_trades}")
    lines.append(f"     Pending (no result):    {bt.pending}")
    lines.append(f"")
    lines.append(f"     Wins:     {bt.wins}")
    lines.append(f"     Losses:   {bt.losses}")
    lines.append(f"     Win Rate: {bt.win_rate:.1%}")
    lines.append(f"")
    lines.append(f"     Total PnL:      ${bt.total_profit_cents/100:+.2f}")
    lines.append(f"     Total Cost:     ${bt.total_cost_cents/100:.2f}")
    lines.append(f"     ROI:            {bt.roi_pct:+.1f}%")
    lines.append(f"     Expectation:    ${bt.expectation_per_trade_cents/100:+.4f} per trade")
    lines.append(f"")
    lines.append(f"     Profit Factor:  {bt.profit_factor:.2f}")
    lines.append(f"     Sharpe Ratio:   {bt.sharpe_ratio:.2f}")
    lines.append(f"     Max Drawdown:   ${bt.max_drawdown_cents/100:.2f} ({bt.max_drawdown_pct:.1f}%)")
    lines.append(f"")
    lines.append(f"     Avg Edge:       {bt.avg_edge:.2%}")
    lines.append(f"     Avg Win:        ${bt.avg_win_cents/100:.2f}")
    lines.append(f"     Avg Loss:       ${bt.avg_loss_cents/100:.2f}")
    lines.append(f"     Best Trade:     ${bt.best_trade_cents/100:+.2f}")
    lines.append(f"     Worst Trade:    ${bt.worst_trade_cents/100:+.2f}")

    # Per-asset breakdown
    if bt.by_asset:
        lines.append(f"\n🔍 BY ASSET:")
        lines.append(f"     {'Asset':<10} {'Trades':>7} {'WR':>7} {'PnL':>10} {'ROI':>8}")
        lines.append(f"     {'─'*10} {'─'*7} {'─'*7} {'─'*10} {'─'*8}")
        for asset, stats in sorted(bt.by_asset.items(), key=lambda x: -x[1]["pnl"]):
            lines.append(f"     {asset.upper():<10} {stats['trades']:>7} "
                         f"{stats['win_rate']:>6.0%} ${stats['pnl']/100:>+9.2f} "
                         f"{stats['roi_pct']:>+7.1f}%")

    # Per-regime breakdown
    if bt.by_regime:
        lines.append(f"\n🌊 BY REGIME:")
        lines.append(f"     {'Regime':<20} {'Trades':>7} {'WR':>7} {'PnL':>10} {'ROI':>8}")
        lines.append(f"     {'─'*20} {'─'*7} {'─'*7} {'─'*10} {'─'*8}")
        for regime, stats in sorted(bt.by_regime.items(), key=lambda x: -x[1]["pnl"]):
            lines.append(f"     {regime:<20} {stats['trades']:>7} "
                         f"{stats['win_rate']:>6.0%} ${stats['pnl']/100:>+9.2f} "
                         f"{stats['roi_pct']:>+7.1f}%")

    # Per-side breakdown
    if bt.by_side:
        lines.append(f"\n↕️  BY SIDE:")
        lines.append(f"     {'Side':<10} {'Trades':>7} {'WR':>7} {'PnL':>10} {'ROI':>8}")
        lines.append(f"     {'─'*10} {'─'*7} {'─'*7} {'─'*10} {'─'*8}")
        for side, stats in sorted(bt.by_side.items(), key=lambda x: -x[1]["pnl"]):
            lines.append(f"     {side.upper():<10} {stats['trades']:>7} "
                         f"{stats['win_rate']:>6.0%} ${stats['pnl']/100:>+9.2f} "
                         f"{stats['roi_pct']:>+7.1f}%")

    # Exclusion reasons
    if bt.exclusion_reasons:
        lines.append(f"\n🚫 EXCLUSION REASONS:")
        for reason, count in sorted(bt.exclusion_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"     {count:>5}x  {reason}")

    # Signal context
    if signals:
        lines.append(f"\n📡 SIGNAL CONTEXT (finetuning data):")
        for k, v in signals.items():
            if v:
                lines.append(f"     {k}: {len(v)} events")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def format_sweep_report(sweep_results: list[dict]) -> str:
    """Format parameter sweep results."""
    lines = []
    lines.append("=" * 70)
    lines.append("🔬 PARAMETER SWEEP RESULTS")
    lines.append("=" * 70)

    # Group by parameter
    by_param = defaultdict(list)
    for r in sweep_results:
        by_param[r["param"]].append(r)

    for param, runs in by_param.items():
        lines.append(f"\n📊 {param.upper()}:")
        lines.append(f"   {'Value':>10} {'Trades':>7} {'WR':>7} {'PnL':>10} {'ROI':>8} {'Sharpe':>8} {'MaxDD':>8} {'PF':>7}")
        lines.append(f"   {'─'*10} {'─'*7} {'─'*7} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")

        best = max(runs, key=lambda r: r["pnl_cents"])

        for r in runs:
            marker = " ★" if r == best else ""
            lines.append(
                f"   {str(r['value']):>10} {r['trades']:>7} "
                f"{r['win_rate']:>6.0%} ${r['pnl_cents']/100:>+9.2f} "
                f"{r['roi_pct']:>+7.1f}% {r['sharpe']:>7.2f} "
                f"{r['max_dd_pct']:>7.1f}% {r['profit_factor']:>6.2f}{marker}"
            )

        # Recommendation
        if best["pnl_cents"] > 0:
            lines.append(f"\n   ★ BEST: {param}={best['value']} "
                         f"(WR={best['win_rate']:.0%}, PnL=${best['pnl_cents']/100:+.2f}, "
                         f"Sharpe={best['sharpe']:.2f})")
        else:
            lines.append(f"   ⚠️  All negative PnL for {param} sweep - strategy needs work")

    # Overall best combination
    if sweep_results:
        overall_best = max(sweep_results, key=lambda r: r["pnl_cents"])
        lines.append(f"\n{'=' * 70}")
        lines.append(f"🏆 OVERALL BEST: {overall_best['param']}={overall_best['value']}")
        lines.append(f"   WR={overall_best['win_rate']:.0%}, "
                     f"PnL=${overall_best['pnl_cents']/100:+.2f}, "
                     f"Sharpe={overall_best['sharpe']:.2f}, "
                     f"MaxDD={overall_best['max_dd_pct']:.1f}%")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def generate_recommendations(bt: BacktestResult, sweep_results: list[dict] = None) -> list[str]:
    """Generate actionable recommendations based on backtest results."""
    recs = []

    # Win rate analysis
    if bt.win_rate < 0.40:
        recs.append("⚠️  Win rate below 40%. Consider: "
                     "increasing MIN_EDGE, "
                     "using stricter momentum filters, "
                     "or pausing trading until model improves.")
    elif bt.win_rate < 0.50:
        recs.append("📊 Win rate 40-50%. Marginal. "
                     "Focus on increasing average win size (higher edge trades).")
    elif bt.win_rate > 0.55:
        recs.append("✅ Win rate above 55%. Model has predictive edge. "
                     "Consider slightly increasing Kelly fraction.")

    # PnL analysis
    if bt.total_profit_cents < 0:
        recs.append("🔴 Negative PnL. Key focus: reduce losses. "
                     "Try: higher MIN_EDGE, smaller position sizes, "
                     "or avoiding choppy regime trades.")
    elif bt.roi_pct < 5:
        recs.append("🟡 Low ROI. Consider: concentrating on highest-edge trades only, "
                     "or increasing position size on winning setups.")

    # Drawdown analysis
    if bt.max_drawdown_pct > 30:
        recs.append("⚠️  Max drawdown >30%. Reduce position sizes. "
                     f"Consider Kelly fraction ≤ {max(0.03, bt.params.get('kelly_fraction', 0.05) * 0.7):.2f}")

    # Profit factor
    if bt.profit_factor < 1.0:
        recs.append("🔴 Profit factor < 1.0 (losing money). "
                     "Average loss > average win. "
                     "Need tighter stop-losses or better entry criteria.")
    elif bt.profit_factor > 1.5:
        recs.append("✅ Profit factor > 1.5. Good risk/reward. "
                     "Strategy is profitable — focus on execution reliability.")

    # Per-asset recommendations
    for asset, stats in bt.by_asset.items():
        if stats["trades"] >= 5 and stats["win_rate"] < 0.30:
            recs.append(f"🔴 {asset.upper()} win rate only {stats['win_rate']:.0%}. "
                        f"Consider disabling {asset.upper()} trading or "
                        f"increasing {asset}_min_edge.")
        if stats["trades"] >= 5 and stats["win_rate"] > 0.60:
            recs.append(f"✅ {asset.upper()} win rate {stats['win_rate']:.0%}. "
                        f"Consider allocating more capital to {asset.upper()} trades.")

    # Per-regime recommendations
    for regime, stats in bt.by_regime.items():
        if stats["trades"] >= 5 and stats["win_rate"] < 0.30:
            recs.append(f"⚠️  '{regime}' regime has {stats['win_rate']:.0%} WR. "
                        f"Consider skipping trades in this regime.")

    # Sweep-based recommendations
    if sweep_results:
        by_param = defaultdict(list)
        for r in sweep_results:
            by_param[r["param"]].append(r)

        for param, runs in by_param.items():
            best = max(runs, key=lambda r: r["pnl_cents"])
            current = DEFAULT_PARAMS.get(param)
            if current is not None and best["value"] != current and best["pnl_cents"] > 0:
                improvement = best["pnl_cents"] - min(r["pnl_cents"] for r in runs if r["value"] == current)
                if improvement > 50:  # Only recommend if >$0.50 improvement
                    recs.append(f"💡 Consider changing {param}: "
                                f"{current} → {best['value']} "
                                f"(+${improvement/100:.2f} PnL)")

    return recs


# ============== MAIN ==============

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kalshi AutoTrader Backtesting Framework")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--sweep-param", type=str, help="Sweep a specific parameter only")
    parser.add_argument("--report", action="store_true", help="Analyze existing trades (no simulation)")
    parser.add_argument("--monte-carlo", type=int, metavar="N", help="Run N Monte Carlo simulations")
    parser.add_argument("--compare", nargs=2, metavar=("V1", "V2"), help="Compare two strategy versions")
    parser.add_argument("--initial-bankroll", type=int, default=10000, help="Initial bankroll in cents (default: $100)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", type=str, help="Output file path")

    # Parameter overrides
    parser.add_argument("--min-edge", type=float, help="Override min_edge")
    parser.add_argument("--kelly", type=float, help="Override kelly_fraction")
    parser.add_argument("--fat-tail", type=float, help="Override fat_tail_multiplier")
    parser.add_argument("--min-expiry", type=int, help="Override min_time_to_expiry_minutes")
    parser.add_argument("--no-momentum", action="store_true", help="Disable momentum filter")
    parser.add_argument("--no-regime", action="store_true", help="Disable regime filter")
    parser.add_argument("--no-kelly-check", action="store_true", help="Disable Kelly criterion check")
    parser.add_argument("--max-edge", type=float, help="Override max_edge (default 0.25, use 1.0 to include all)")
    parser.add_argument("--include-weather", action="store_true", help="Relax max_edge for weather trades (they have high calculated edges)")

    args = parser.parse_args()

    print("🔄 Loading historical trade data...")
    trades = load_all_trades()
    signals = load_signal_events()

    if not trades:
        print("❌ No trade data found. Nothing to backtest.")
        sys.exit(1)

    # Build parameters
    params = dict(DEFAULT_PARAMS)
    if args.min_edge is not None:
        params["min_edge"] = args.min_edge
        params["btc_min_edge"] = args.min_edge
        params["eth_min_edge"] = args.min_edge * 1.25
        params["sol_min_edge"] = args.min_edge * 1.5
    if args.kelly is not None:
        params["kelly_fraction"] = args.kelly
        params["btc_kelly"] = args.kelly
        params["eth_kelly"] = args.kelly * 0.8
    if args.fat_tail is not None:
        params["fat_tail_multiplier"] = args.fat_tail
    if args.min_expiry is not None:
        params["min_time_to_expiry_minutes"] = args.min_expiry
    if args.no_momentum:
        params["use_momentum_filter"] = False
    if args.no_regime:
        params["use_regime_filter"] = False
    if args.no_kelly_check:
        params["use_kelly_check"] = False
    if args.max_edge is not None:
        params["max_edge"] = args.max_edge
    if args.include_weather:
        params["max_edge"] = 1.0  # Weather trades often have very high calculated edges

    # ---- MODE: Parameter Sweep ----
    if args.sweep or args.sweep_param:
        print(f"\n🔬 Running parameter sweep{'(' + args.sweep_param + ')' if args.sweep_param else ''}...")
        sweep_results = run_parameter_sweep(trades, args.sweep_param, verbose=True)

        report = format_sweep_report(sweep_results)
        print(f"\n{report}")

        # Also run a base backtest for recommendations
        print(f"\n📊 Running base backtest for comparison...")
        base_bt = run_backtest(trades, params)
        recs = generate_recommendations(base_bt, sweep_results)

        if recs:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recs:
                print(f"   {rec}")

        # Save results
        output_path = args.output or str(SWEEP_OUTPUT)
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sweep_results": sweep_results,
            "base_params": params,
            "recommendations": recs,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Sweep results saved to: {output_path}")

    # ---- MODE: Monte Carlo ----
    elif args.monte_carlo:
        print(f"\n🎲 Running Monte Carlo bootstrap ({args.monte_carlo} simulations)...")
        mc_results = monte_carlo_bootstrap(trades, params, args.monte_carlo, verbose=True)

        if args.json:
            print(json.dumps(mc_results, indent=2))

    # ---- MODE: Compare strategies ----
    elif args.compare:
        v1_name, v2_name = args.compare
        print(f"\n⚔️  Comparing strategies: {v1_name} vs {v2_name}")

        # V1: conservative (high edge, low kelly)
        v1_params = dict(DEFAULT_PARAMS)
        v2_params = dict(DEFAULT_PARAMS)

        if v1_name.lower() == "v1":
            v1_params["min_edge"] = 0.10
            v1_params["btc_min_edge"] = 0.10
            v1_params["eth_min_edge"] = 0.12
            v1_params["kelly_fraction"] = 0.05
        elif v1_name.lower() == "conservative":
            v1_params["min_edge"] = 0.08
            v1_params["btc_min_edge"] = 0.08
            v1_params["eth_min_edge"] = 0.10
            v1_params["kelly_fraction"] = 0.03

        if v2_name.lower() == "v2":
            v2_params["min_edge"] = 0.04
            v2_params["btc_min_edge"] = 0.04
            v2_params["eth_min_edge"] = 0.05
            v2_params["kelly_fraction"] = 0.08
        elif v2_name.lower() == "aggressive":
            v2_params["min_edge"] = 0.02
            v2_params["btc_min_edge"] = 0.02
            v2_params["eth_min_edge"] = 0.03
            v2_params["kelly_fraction"] = 0.12

        bt1 = run_backtest(trades, v1_params)
        bt2 = run_backtest(trades, v2_params)

        print(f"\n{'Metric':<25} {v1_name:>15} {v2_name:>15} {'Delta':>15}")
        print(f"{'─'*25} {'─'*15} {'─'*15} {'─'*15}")

        metrics = [
            ("Trades", bt1.included_trades, bt2.included_trades, "int"),
            ("Win Rate", bt1.win_rate, bt2.win_rate, "pct"),
            ("PnL", bt1.total_profit_cents, bt2.total_profit_cents, "cents"),
            ("ROI", bt1.roi_pct, bt2.roi_pct, "pct_val"),
            ("Sharpe", bt1.sharpe_ratio, bt2.sharpe_ratio, "float"),
            ("Max DD", bt1.max_drawdown_pct, bt2.max_drawdown_pct, "pct_val"),
            ("Profit Factor", bt1.profit_factor, bt2.profit_factor, "float"),
            ("Avg Edge", bt1.avg_edge, bt2.avg_edge, "pct"),
            ("Avg Win", bt1.avg_win_cents, bt2.avg_win_cents, "cents"),
            ("Avg Loss", bt1.avg_loss_cents, bt2.avg_loss_cents, "cents"),
        ]

        for name, v1, v2, fmt in metrics:
            delta = v2 - v1
            if fmt == "pct":
                print(f"{name:<25} {v1:>14.1%} {v2:>14.1%} {delta:>+14.1%}")
            elif fmt == "cents":
                print(f"{name:<25} ${v1/100:>13.2f} ${v2/100:>13.2f} ${delta/100:>+13.2f}")
            elif fmt == "pct_val":
                print(f"{name:<25} {v1:>13.1f}% {v2:>13.1f}% {delta:>+13.1f}%")
            elif fmt == "float":
                print(f"{name:<25} {v1:>14.3f} {v2:>14.3f} {delta:>+14.3f}")
            else:
                print(f"{name:<25} {v1:>14} {v2:>14} {delta:>+14}")

        winner = v1_name if bt1.total_profit_cents > bt2.total_profit_cents else v2_name
        print(f"\n🏆 Winner by PnL: {winner}")

    # ---- MODE: Report (or default) ----
    else:
        print(f"\n📊 Running backtest with {'custom' if any([args.min_edge, args.kelly, args.fat_tail]) else 'default'} parameters...")
        bt = run_backtest(trades, params, args.initial_bankroll, verbose=args.verbose)

        report = format_report(bt, signals)
        print(f"\n{report}")

        recs = generate_recommendations(bt)
        if recs:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recs:
                print(f"   {rec}")

        # Monte Carlo if enough data
        settled_count = bt.wins + bt.losses
        if settled_count >= 20:
            print(f"\n🎲 Quick Monte Carlo (100 sims)...")
            monte_carlo_bootstrap(trades, params, 100, verbose=True)

        # Save report
        output_path = args.output or str(REPORT_OUTPUT)
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "summary": {
                "total_trades": bt.total_trades,
                "included": bt.included_trades,
                "excluded": bt.excluded_trades,
                "pending": bt.pending,
                "wins": bt.wins,
                "losses": bt.losses,
                "win_rate": round(bt.win_rate, 4),
                "pnl_cents": bt.total_profit_cents,
                "roi_pct": round(bt.roi_pct, 2),
                "sharpe": round(bt.sharpe_ratio, 3),
                "max_dd_pct": round(bt.max_drawdown_pct, 2),
                "profit_factor": round(bt.profit_factor, 3),
                "expectation_per_trade": round(bt.expectation_per_trade_cents, 2),
            },
            "by_asset": bt.by_asset,
            "by_regime": bt.by_regime,
            "by_side": bt.by_side,
            "exclusion_reasons": bt.exclusion_reasons,
            "recommendations": recs,
            "equity_curve_sample": bt.equity_curve[::max(1, len(bt.equity_curve)//50)],  # 50 points
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Report saved to: {output_path}")

        if args.json:
            print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
