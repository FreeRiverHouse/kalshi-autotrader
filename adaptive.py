"""
Adaptive parameter tuning for Kalshi AutoTrader.

Reads rolling trade performance from SQLite and adjusts algo parameters
to maximize win rate and profit factor. Runs every ADAPT_EVERY_N settled trades.

Parameters tuned:
  - MIN_EDGE_BUY_YES / MIN_EDGE_BUY_NO  (edge thresholds)
  - CALIBRATION_FACTOR_YES / CALIBRATION_FACTOR_NO  (forecast shrinkage)
  - KELLY_FRACTION  (position sizing aggressiveness)
"""

import logging
from typing import Optional

import db as _db

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ADAPT_EVERY_N = 20    # trigger every N new settled trades
ADAPT_WINDOW  = 50    # look at last N settled trades for metrics

# Clamp ranges (safety rails — never go outside these)
CLAMPS = {
    "MIN_EDGE_BUY_YES":      (0.05,  0.15),   # 5% – 15%
    "MIN_EDGE_BUY_NO":       (0.02,  0.08),   # 2% – 8%
    "CALIBRATION_FACTOR_YES": (0.25,  0.60),
    "CALIBRATION_FACTOR_NO":  (0.50,  0.85),
    "KELLY_FRACTION":         (0.10,  0.40),
}


def _clamp(name: str, value: float) -> float:
    lo, hi = CLAMPS.get(name, (0.0, 1.0))
    return max(lo, min(hi, value))


def _get_rolling_trades(window: int = ADAPT_WINDOW) -> list[dict]:
    """Fetch last `window` settled trades from SQLite."""
    with _db.get_conn() as conn:
        rows = conn.execute("""
            SELECT action, result_status, edge, forecast_prob, pnl_cents, cost_cents
            FROM trades
            WHERE result_status IN ('won', 'lost')
            ORDER BY settled_at DESC
            LIMIT ?
        """, (window,)).fetchall()
    return [dict(r) for r in rows]


def compute_adaptive_params(current: dict) -> dict:
    """
    Compute adjusted parameters based on rolling trade performance.

    Args:
        current: dict with keys MIN_EDGE_BUY_YES, MIN_EDGE_BUY_NO,
                 CALIBRATION_FACTOR_YES, CALIBRATION_FACTOR_NO, KELLY_FRACTION

    Returns:
        dict with same keys (adjusted) + _meta with diagnostics
    """
    trades = _get_rolling_trades(ADAPT_WINDOW)
    if len(trades) < 10:
        return {**current, "_meta": {"skipped": True, "reason": f"only {len(trades)} trades"}}

    # ── Split by side ─────────────────────────────────────────────────────────
    yes_trades = [t for t in trades if t["action"] == "BUY_YES"]
    no_trades  = [t for t in trades if t["action"] == "BUY_NO"]

    yes_won  = sum(1 for t in yes_trades if t["result_status"] == "won")
    no_won   = sum(1 for t in no_trades  if t["result_status"] == "won")
    yes_total = len(yes_trades)
    no_total  = len(no_trades)

    yes_wr = (yes_won / yes_total * 100) if yes_total > 0 else None
    no_wr  = (no_won  / no_total  * 100) if no_total > 0  else None

    # ── Gross profit/loss ─────────────────────────────────────────────────────
    gross_profit = sum(t["pnl_cents"] for t in trades if (t["pnl_cents"] or 0) > 0)
    gross_loss   = abs(sum(t["pnl_cents"] for t in trades if (t["pnl_cents"] or 0) < 0)) or 1
    profit_factor = gross_profit / gross_loss

    # ── Calibration errors (predicted vs actual) ──────────────────────────────
    def _mean_error(side_trades: list) -> Optional[float]:
        """Mean error = actual_outcome - forecast_prob. Negative = overconfident."""
        errs = []
        for t in side_trades:
            fp = t.get("forecast_prob")
            if fp is None:
                continue
            actual = 1.0 if t["result_status"] == "won" else 0.0
            errs.append(actual - fp)
        return sum(errs) / len(errs) if errs else None

    yes_mean_err = _mean_error(yes_trades)
    no_mean_err  = _mean_error(no_trades)

    # ── Adjust MIN_EDGE_BUY_YES ───────────────────────────────────────────────
    new_min_edge_yes = current["MIN_EDGE_BUY_YES"]
    yes_reason = "no change"
    if yes_total >= 5:
        if yes_wr < 35:
            new_min_edge_yes += 0.01   # tighten: WR too low
            yes_reason = f"WR {yes_wr:.0f}% < 35% → +1%"
        elif yes_wr > 55:
            new_min_edge_yes -= 0.005  # loosen: WR is great
            yes_reason = f"WR {yes_wr:.0f}% > 55% → -0.5%"
        else:
            yes_reason = f"WR {yes_wr:.0f}% in range"
    new_min_edge_yes = _clamp("MIN_EDGE_BUY_YES", new_min_edge_yes)

    # ── Adjust MIN_EDGE_BUY_NO ────────────────────────────────────────────────
    new_min_edge_no = current["MIN_EDGE_BUY_NO"]
    no_reason = "no change"
    if no_total >= 5:
        if no_wr < 55:
            new_min_edge_no += 0.005   # tighten
            no_reason = f"WR {no_wr:.0f}% < 55% → +0.5%"
        elif no_wr > 75:
            new_min_edge_no -= 0.003   # loosen: performing well
            no_reason = f"WR {no_wr:.0f}% > 75% → -0.3%"
        else:
            no_reason = f"WR {no_wr:.0f}% in range"
    new_min_edge_no = _clamp("MIN_EDGE_BUY_NO", new_min_edge_no)

    # ── Adjust CALIBRATION_FACTOR_YES ─────────────────────────────────────────
    new_cal_yes = current["CALIBRATION_FACTOR_YES"]
    cal_yes_reason = "no data"
    if yes_mean_err is not None:
        new_cal_yes = current["CALIBRATION_FACTOR_YES"] + yes_mean_err * 0.3
        cal_yes_reason = f"mean_err={yes_mean_err:+.3f} → adj {yes_mean_err*0.3:+.3f}"
    new_cal_yes = _clamp("CALIBRATION_FACTOR_YES", new_cal_yes)

    # ── Adjust CALIBRATION_FACTOR_NO ──────────────────────────────────────────
    new_cal_no = current["CALIBRATION_FACTOR_NO"]
    cal_no_reason = "no data"
    if no_mean_err is not None:
        new_cal_no = current["CALIBRATION_FACTOR_NO"] + no_mean_err * 0.3
        cal_no_reason = f"mean_err={no_mean_err:+.3f} → adj {no_mean_err*0.3:+.3f}"
    new_cal_no = _clamp("CALIBRATION_FACTOR_NO", new_cal_no)

    # ── Adjust KELLY_FRACTION ─────────────────────────────────────────────────
    new_kelly = current["KELLY_FRACTION"]
    kelly_reason = "stable"
    if profit_factor > 1.5:
        new_kelly = current["KELLY_FRACTION"] * 1.10  # +10%
        kelly_reason = f"PF {profit_factor:.2f} > 1.5 → +10%"
    elif profit_factor < 0.8:
        new_kelly = current["KELLY_FRACTION"] * 0.85  # -15%
        kelly_reason = f"PF {profit_factor:.2f} < 0.8 → -15%"
    new_kelly = _clamp("KELLY_FRACTION", new_kelly)

    result = {
        "MIN_EDGE_BUY_YES":       round(new_min_edge_yes, 4),
        "MIN_EDGE_BUY_NO":        round(new_min_edge_no, 4),
        "CALIBRATION_FACTOR_YES": round(new_cal_yes, 4),
        "CALIBRATION_FACTOR_NO":  round(new_cal_no, 4),
        "KELLY_FRACTION":         round(new_kelly, 4),
        "_meta": {
            "skipped": False,
            "window": len(trades),
            "yes_wr": round(yes_wr, 1) if yes_wr is not None else None,
            "no_wr": round(no_wr, 1) if no_wr is not None else None,
            "yes_total": yes_total,
            "no_total": no_total,
            "profit_factor": round(profit_factor, 3),
            "yes_mean_err": round(yes_mean_err, 4) if yes_mean_err is not None else None,
            "no_mean_err": round(no_mean_err, 4) if no_mean_err is not None else None,
            "reasons": {
                "MIN_EDGE_BUY_YES": yes_reason,
                "MIN_EDGE_BUY_NO": no_reason,
                "CALIBRATION_FACTOR_YES": cal_yes_reason,
                "CALIBRATION_FACTOR_NO": cal_no_reason,
                "KELLY_FRACTION": kelly_reason,
            },
        },
    }

    # ── Log changes to db ─────────────────────────────────────────────────────
    settled_count = _db.get_metrics().get("settled", 0)
    for key in CLAMPS:
        old_val = current[key]
        new_val = result[key]
        if abs(old_val - new_val) > 1e-6:
            reason = result["_meta"]["reasons"].get(key, "")
            _db.log_param_change(settled_count, key, old_val, new_val, reason)
            log.info(f"[Adaptive] {key}: {old_val:.4f} → {new_val:.4f} ({reason})")

    return result
