#!/usr/bin/env python3
"""
SQLite database layer for Kalshi AutoTrader.
Single file, zero dependencies beyond stdlib.

DB path: data/trading/trades.db
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timezone

# Use SSD for DB if available, fallback to repo dir
_SSD_DATA = Path("/Volumes/DATI-SSD/kalshi-logs")
DB_PATH = (_SSD_DATA / "trades.db") if _SSD_DATA.exists() else Path(__file__).parent / "data" / "trading" / "trades.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    settled_at      TEXT,
    ticker          TEXT NOT NULL,
    title           TEXT,
    action          TEXT NOT NULL,       -- BUY_YES | BUY_NO
    price_cents     INTEGER DEFAULT 0,
    contracts       INTEGER DEFAULT 0,
    cost_cents      INTEGER DEFAULT 0,
    edge            REAL DEFAULT 0,
    forecast_prob   REAL,
    critic_prob     REAL,
    result_status   TEXT DEFAULT 'pending', -- won | lost | pending
    pnl_cents       INTEGER,
    category        TEXT,
    regime          TEXT,
    reason          TEXT,
    minutes_to_expiry INTEGER,
    cycle_id        TEXT,
    market_prob     REAL,
    extra           TEXT                -- JSON blob for future fields
);

CREATE TABLE IF NOT EXISTS cycles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    cycle_id        TEXT UNIQUE,
    markets_scanned INTEGER DEFAULT 0,
    trades_attempted INTEGER DEFAULT 0,
    trades_placed   INTEGER DEFAULT 0,
    balance_cents   INTEGER,
    duration_ms     INTEGER,
    error           TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_timestamp     ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_settled_at    ON trades(settled_at);
CREATE INDEX IF NOT EXISTS idx_trades_result_status ON trades(result_status);
CREATE INDEX IF NOT EXISTS idx_trades_action        ON trades(action);
CREATE INDEX IF NOT EXISTS idx_trades_ticker        ON trades(ticker);

CREATE TABLE IF NOT EXISTS param_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT    NOT NULL DEFAULT (datetime('now', 'utc')),
    settled_count  INTEGER NOT NULL,
    param_name     TEXT    NOT NULL,
    old_value      REAL    NOT NULL,
    new_value      REAL    NOT NULL,
    reason         TEXT
);
"""


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ── Write ──────────────────────────────────────────────────────────────────────

def insert_trade(t: dict) -> int:
    """Insert a new trade. Returns rowid."""
    sql = """
    INSERT INTO trades
      (timestamp, settled_at, ticker, title, action, price_cents, contracts,
       cost_cents, edge, forecast_prob, critic_prob, result_status, pnl_cents,
       category, regime, reason, minutes_to_expiry, cycle_id, market_prob, extra)
    VALUES
      (:timestamp, :settled_at, :ticker, :title, :action, :price_cents, :contracts,
       :cost_cents, :edge, :forecast_prob, :critic_prob, :result_status, :pnl_cents,
       :category, :regime, :reason, :minutes_to_expiry, :cycle_id, :market_prob, :extra)
    """
    row = {
        "timestamp":         t.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "settled_at":        t.get("settled_at"),
        "ticker":            t.get("ticker", ""),
        "title":             t.get("title"),
        "action":            t.get("action", ""),
        "price_cents":       t.get("price_cents", 0),
        "contracts":         t.get("contracts", 0),
        "cost_cents":        t.get("cost_cents", 0),
        "edge":              t.get("edge", 0),
        "forecast_prob":     t.get("forecast_prob"),
        "critic_prob":       t.get("critic_prob"),
        "result_status":     t.get("result_status", "pending"),
        "pnl_cents":         t.get("pnl_cents"),
        "category":          t.get("category"),
        "regime":            t.get("regime"),
        "reason":            t.get("reason"),
        "minutes_to_expiry": t.get("minutes_to_expiry"),
        "cycle_id":          t.get("cycle_id"),
        "market_prob":       t.get("market_prob"),
        "extra":             json.dumps({k: v for k, v in t.items() if k not in {
            "timestamp","settled_at","ticker","title","action","price_cents","contracts",
            "cost_cents","edge","forecast_prob","critic_prob","result_status","pnl_cents",
            "category","regime","reason","minutes_to_expiry","cycle_id","market_prob"
        }}) if t else None,
    }
    with get_conn() as conn:
        cur = conn.execute(sql, row)
        return cur.lastrowid


def settle_trade(ticker: str, result_status: str, settled_at: str, pnl_cents: int):
    """Update a pending trade to won/lost."""
    with get_conn() as conn:
        conn.execute("""
            UPDATE trades
            SET result_status=?, settled_at=?, pnl_cents=?
            WHERE ticker=? AND result_status='pending'
            ORDER BY timestamp DESC
            LIMIT 1
        """, (result_status, settled_at, pnl_cents, ticker))


def upsert_trade_by_ticker_ts(t: dict):
    """Insert or update based on (ticker, timestamp). Used during JSONL migration."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM trades WHERE ticker=? AND timestamp=?",
            (t.get("ticker",""), t.get("timestamp",""))
        ).fetchone()
        if row:
            conn.execute("""
                UPDATE trades SET
                  settled_at=?, result_status=?, pnl_cents=?,
                  action=?, price_cents=?, contracts=?, cost_cents=?,
                  edge=?, forecast_prob=?, category=?, title=?, market_prob=?,
                  regime=?, reason=?, minutes_to_expiry=?
                WHERE id=?
            """, (
                t.get("settled_at"), t.get("result_status","pending"), t.get("pnl_cents"),
                t.get("action",""), t.get("price_cents",0), t.get("contracts",0),
                t.get("cost_cents",0), t.get("edge",0), t.get("forecast_prob"),
                t.get("category"), t.get("title"), t.get("market_prob"),
                t.get("regime"), t.get("reason"), t.get("minutes_to_expiry"),
                row["id"]
            ))
        else:
            insert_trade(t)


def insert_cycle(c: dict):
    # Accept both canonical keys and autotrader keys (balance$/duration_s/trades_executed)
    balance_raw = c.get("balance_cents")
    if balance_raw is None and c.get("balance") is not None:
        balance_raw = int(round(c["balance"] * 100))
    duration_raw = c.get("duration_ms")
    if duration_raw is None and c.get("duration_s") is not None:
        duration_raw = int(c["duration_s"] * 1000)
    trades_placed = c.get("trades_placed") or c.get("trades_executed", 0)
    with get_conn() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO cycles
              (timestamp, cycle_id, markets_scanned, trades_attempted,
               trades_placed, balance_cents, duration_ms, error)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            c.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            c.get("cycle_id"),
            c.get("markets_scanned", 0),
            c.get("trades_attempted", 0),
            trades_placed,
            balance_raw,
            duration_raw,
            c.get("error"),
        ))


# ── Read ───────────────────────────────────────────────────────────────────────

def get_trades(action=None, result_status=None, limit=200, offset=0) -> list[dict]:
    where, params = [], []
    if action:
        where.append("action=?"); params.append(action)
    if result_status:
        where.append("result_status=?"); params.append(result_status)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    params += [limit, offset]
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM trades {where_sql} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params
        ).fetchall()
    return [dict(r) for r in rows]


def get_settled_trades() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE result_status IN ('won','lost') ORDER BY settled_at"
        ).fetchall()
    return [dict(r) for r in rows]


def get_pending_trades() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE result_status='pending' ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_metrics() -> dict:
    with get_conn() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM trades WHERE action IN ('BUY_YES','BUY_NO')").fetchone()[0]
        settled  = conn.execute("SELECT COUNT(*) FROM trades WHERE result_status IN ('won','lost')").fetchone()[0]
        won      = conn.execute("SELECT COUNT(*) FROM trades WHERE result_status='won'").fetchone()[0]
        lost     = conn.execute("SELECT COUNT(*) FROM trades WHERE result_status='lost'").fetchone()[0]
        pending  = conn.execute("SELECT COUNT(*) FROM trades WHERE result_status='pending'").fetchone()[0]

        no_total = conn.execute("SELECT COUNT(*) FROM trades WHERE action='BUY_NO' AND result_status IN ('won','lost')").fetchone()[0]
        no_won   = conn.execute("SELECT COUNT(*) FROM trades WHERE action='BUY_NO' AND result_status='won'").fetchone()[0]
        yes_total= conn.execute("SELECT COUNT(*) FROM trades WHERE action='BUY_YES' AND result_status IN ('won','lost')").fetchone()[0]
        yes_won  = conn.execute("SELECT COUNT(*) FROM trades WHERE action='BUY_YES' AND result_status='won'").fetchone()[0]

        cost_row = conn.execute("SELECT SUM(cost_cents) FROM trades WHERE result_status IN ('won','lost')").fetchone()
        total_cost = cost_row[0] or 0

        pnl_row  = conn.execute("SELECT SUM(pnl_cents) FROM trades WHERE result_status IN ('won','lost')").fetchone()
        net_pnl  = pnl_row[0] or 0

        profit_row = conn.execute("SELECT SUM(pnl_cents) FROM trades WHERE result_status='won'").fetchone()
        loss_row   = conn.execute("SELECT SUM(pnl_cents) FROM trades WHERE result_status='lost'").fetchone()

        avg_edge_row = conn.execute(
            "SELECT AVG(edge) FROM trades WHERE edge IS NOT NULL"
        ).fetchone()

        cats = conn.execute("""
            SELECT category,
                   COUNT(*) as trades,
                   SUM(CASE WHEN result_status='won' THEN 1 ELSE 0 END) as won,
                   SUM(COALESCE(pnl_cents,0)) as pnl
            FROM trades
            WHERE result_status IN ('won','lost')
            GROUP BY category
        """).fetchall()

        bankroll_series = conn.execute("""
            SELECT settled_at as t, SUM(COALESCE(pnl_cents,0))
                   OVER (ORDER BY settled_at ROWS UNBOUNDED PRECEDING) as cumulative_pnl
            FROM trades
            WHERE result_status IN ('won','lost') AND settled_at IS NOT NULL
            ORDER BY settled_at
        """).fetchall()

    return {
        "total_trades":      total,
        "settled":           settled,
        "won":               won,
        "lost":              lost,
        "pending":           pending,
        "win_rate":          round(won / settled * 100, 1) if settled else 0,
        "no_win_rate":       round(no_won / no_total * 100, 1) if no_total else 0,
        "yes_win_rate":      round(yes_won / yes_total * 100, 1) if yes_total else 0,
        "no_trades":         no_total,
        "yes_trades":        yes_total,
        "roi_pct":           round(net_pnl / total_cost * 100, 2) if total_cost else 0,
        "net_pnl_cents":     net_pnl,
        "gross_profit_cents": profit_row[0] or 0,
        "gross_loss_cents":   abs(loss_row[0] or 0),
        "avg_edge":          round((avg_edge_row[0] or 0) * 100, 2),
        "by_category":       {r["category"] or "other": {"trades": r["trades"], "won": r["won"], "pnl": r["pnl"]} for r in cats},
        "bankroll_series_raw": [(r[0], r[1]) for r in bankroll_series],
    }


def get_roi_series() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT settled_at, cost_cents, pnl_cents,
                   ROW_NUMBER() OVER (ORDER BY settled_at) as n
            FROM trades
            WHERE result_status IN ('won','lost') AND settled_at IS NOT NULL
            ORDER BY settled_at
        """).fetchall()

    total_cost, total_pnl, series = 0, 0, []
    for r in rows:
        total_cost += r["cost_cents"] or 0
        total_pnl  += r["pnl_cents"]  or 0
        roi = (total_pnl / total_cost * 100) if total_cost else 0
        series.append({
            "t": (r["settled_at"] or "")[:19].replace("T", " "),
            "v": round(roi, 2),
            "n": r["n"],
        })
    return series


def get_cycle_series(limit: int = 1000) -> list[dict]:
    """Cycle timestamps + balance_cents for time-series chart. Returns oldest→newest."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT timestamp, balance_cents, trades_placed, markets_scanned, duration_ms
            FROM cycles
            WHERE balance_cents IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [
        {
            "t":  r["timestamp"][:19].replace("T", " "),
            "b":  round(r["balance_cents"] / 100, 2),
            "tp": r["trades_placed"] or 0,
            "ms": r["markets_scanned"] or 0,
            "ms_dur": r["duration_ms"] or 0,
        }
        for r in reversed(rows)
    ]


def get_daily_stats() -> list[dict]:
    """Daily aggregated trade stats."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                DATE(timestamp) as day,
                COUNT(*) as trades,
                SUM(CASE WHEN result_status='won' THEN 1 ELSE 0 END) as won,
                SUM(CASE WHEN result_status='lost' THEN 1 ELSE 0 END) as lost,
                SUM(COALESCE(pnl_cents,0)) as pnl,
                COALESCE(SUM(cost_cents),0) as cost
            FROM trades
            GROUP BY DATE(timestamp)
            ORDER BY day
        """).fetchall()
    result = []
    for r in rows:
        settled = (r["won"] or 0) + (r["lost"] or 0)
        result.append({
            "day":      r["day"],
            "trades":   r["trades"],
            "won":      r["won"] or 0,
            "lost":     r["lost"] or 0,
            "pnl_usd":  round((r["pnl"] or 0) / 100, 2),
            "cost_usd": round((r["cost"] or 0) / 100, 2),
            "win_rate": round((r["won"] or 0) / settled * 100, 1) if settled > 0 else 0,
        })
    return result


def get_hourly_distribution() -> list[dict]:
    """Trades per hour of day (0-23) for activity heatmap — PST (UTC-8)."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                CAST(strftime('%H', datetime(timestamp, '-8 hours')) AS INTEGER) as hour,
                COUNT(*) as trades,
                SUM(CASE WHEN result_status='won' THEN 1 ELSE 0 END) as won
            FROM trades
            GROUP BY hour
            ORDER BY hour
        """).fetchall()
    result = [{"hour": h, "trades": 0, "won": 0, "win_rate": 0} for h in range(24)]
    for r in rows:
        settled = (r["won"] or 0) + (r["trades"] - (r["won"] or 0))
        result[r["hour"]] = {
            "hour":     r["hour"],
            "trades":   r["trades"],
            "won":      r["won"] or 0,
            "win_rate": round((r["won"] or 0) / r["trades"] * 100, 1) if r["trades"] > 0 else 0,
        }
    return result


def get_cycle_hourly_distribution() -> list[dict]:
    """Cycle activity per hour of day (0-23) — PST (UTC-8)."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                CAST(strftime('%H', datetime(timestamp, '-8 hours')) AS INTEGER) as hour,
                COUNT(*) as cycles,
                SUM(trades_placed) as trades_placed
            FROM cycles
            GROUP BY hour
            ORDER BY hour
        """).fetchall()
    result = [{"hour": h, "cycles": 0, "trades_placed": 0} for h in range(24)]
    for r in rows:
        result[r["hour"]] = {
            "hour":          r["hour"],
            "cycles":        r["cycles"],
            "trades_placed": r["trades_placed"] or 0,
        }
    return result


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_calibration_data() -> list[dict]:
    """Forecast accuracy per decile bin. Returns predicted vs actual win rate."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                forecast_prob,
                result_status
            FROM trades
            WHERE result_status IN ('won','lost')
              AND forecast_prob IS NOT NULL
            ORDER BY forecast_prob
        """).fetchall()

    # 10 bins: 0-10%, 10-20%, ..., 90-100%
    bins = [{"label": f"{i*10}-{(i+1)*10}%", "predicted": i*10 + 5.0,
             "wins": 0, "total": 0} for i in range(10)]
    for r in rows:
        prob = r["forecast_prob"] or 0.0
        idx = min(int(prob * 10), 9)
        bins[idx]["total"] += 1
        if r["result_status"] == "won":
            bins[idx]["wins"] += 1

    result = []
    for b in bins:
        actual = round(b["wins"] / b["total"] * 100, 1) if b["total"] > 0 else None
        result.append({
            "label":     b["label"],
            "predicted": b["predicted"],
            "actual":    actual,
            "count":     b["total"],
        })
    return result


def get_streak_analysis() -> dict:
    """Win/loss streak analysis from settled trades ordered by settled_at."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT result_status FROM trades
            WHERE result_status IN ('won','lost')
              AND settled_at IS NOT NULL
            ORDER BY settled_at ASC
        """).fetchall()

    statuses = [r["result_status"] for r in rows]
    if not statuses:
        return {"longestWin": 0, "longestLoss": 0, "current": 0,
                "currentType": "none", "totalTrades": 0}

    longest_win = longest_loss = cur = 0
    cur_type = statuses[0]
    cur = 1

    def update_longest(t, n, lw, ll):
        if t == "won":
            return max(lw, n), ll
        return lw, max(ll, n)

    for s in statuses[1:]:
        if s == cur_type:
            cur += 1
        else:
            longest_win, longest_loss = update_longest(cur_type, cur, longest_win, longest_loss)
            cur_type = s
            cur = 1
    longest_win, longest_loss = update_longest(cur_type, cur, longest_win, longest_loss)

    return {
        "longestWin":  longest_win,
        "longestLoss": longest_loss,
        "current":     cur,
        "currentType": cur_type,
        "totalTrades": len(statuses),
    }


def get_edge_distribution() -> dict:
    """Edge histogram by bucket, split by BUY_YES / BUY_NO."""
    buckets = [
        {"label": "<1%",   "min": -99,  "max": 0.01},
        {"label": "1-3%",  "min": 0.01, "max": 0.03},
        {"label": "3-5%",  "min": 0.03, "max": 0.05},
        {"label": "5-10%", "min": 0.05, "max": 0.10},
        {"label": ">10%",  "min": 0.10, "max": 99},
    ]
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT action, edge, result_status
            FROM trades
            WHERE edge IS NOT NULL
              AND result_status IN ('won', 'lost')
        """).fetchall()

    yes_dist = {b["label"]: {"total": 0, "won": 0} for b in buckets}
    no_dist  = {b["label"]: {"total": 0, "won": 0} for b in buckets}

    for r in rows:
        e = r["edge"] or 0.0
        for b in buckets:
            if b["min"] <= e < b["max"]:
                target = yes_dist if r["action"] == "BUY_YES" else no_dist
                target[b["label"]]["total"] += 1
                if r["result_status"] == "won":
                    target[b["label"]]["won"] += 1
                break

    def fmt(d):
        out = []
        for label, v in d.items():
            wr = round(v["won"] / v["total"] * 100, 1) if v["total"] else None
            out.append({"label": label, "total": v["total"],
                        "won": v["won"], "winRate": wr})
        return out

    return {"yes": fmt(yes_dist), "no": fmt(no_dist)}


def get_risk_metrics() -> dict:
    """Sharpe, Sortino, Calmar, max drawdown from settled trades."""
    import math
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT pnl_cents, cost_cents, settled_at
            FROM trades
            WHERE result_status IN ('won','lost')
              AND pnl_cents IS NOT NULL AND cost_cents > 0
            ORDER BY settled_at ASC
        """).fetchall()

    if not rows:
        return {"sharpe": 0, "sortino": 0, "calmar": 0,
                "maxDrawdownPct": 0, "maxDrawdownCents": 0,
                "profitFactor": 0, "avgDurationHours": None}

    returns = [r["pnl_cents"] / r["cost_cents"] for r in rows]
    n = len(returns)
    mean_r = sum(returns) / n
    variance = sum((x - mean_r) ** 2 for x in returns) / n if n > 1 else 0
    std_r = math.sqrt(variance)
    downside = [min(x, 0) for x in returns]
    downside_var = sum(x ** 2 for x in downside) / n
    downside_std = math.sqrt(downside_var)

    scale = math.sqrt(252)  # annualise (approx daily trades)
    sharpe  = round(mean_r / std_r * scale, 3) if std_r else 0
    sortino = round(mean_r / downside_std * scale, 3) if downside_std else 0

    # Max drawdown on cumulative PnL
    cum, peak, max_dd = 0, 0, 0
    for r in rows:
        cum += r["pnl_cents"]
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    peak_pos = max(cum + max_dd, 1)
    max_dd_pct = round(max_dd / peak_pos * 100, 2) if peak_pos else 0

    total_pnl = sum(r["pnl_cents"] for r in rows)
    # annualised return approx
    annualised = mean_r * 252
    calmar = round(annualised / (max_dd_pct / 100), 3) if max_dd_pct else 0

    gross_profit = sum(r["pnl_cents"] for r in rows if r["pnl_cents"] > 0)
    gross_loss   = abs(sum(r["pnl_cents"] for r in rows if r["pnl_cents"] < 0))
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss else 0

    # Avg duration (hours) for settled trades with timestamps
    with get_conn() as conn:
        dur_rows = conn.execute("""
            SELECT
              (julianday(settled_at) - julianday(timestamp)) * 24 as dur_h
            FROM trades
            WHERE result_status IN ('won','lost')
              AND settled_at IS NOT NULL AND timestamp IS NOT NULL
        """).fetchall()
    valid_durs = [r["dur_h"] for r in dur_rows if r["dur_h"] is not None and r["dur_h"] >= 0]
    avg_dur = round(sum(valid_durs) / len(valid_durs), 2) if valid_durs else None

    return {
        "sharpe":           sharpe,
        "sortino":          sortino,
        "calmar":           calmar,
        "maxDrawdownPct":   max_dd_pct,
        "maxDrawdownCents": max_dd,
        "profitFactor":     profit_factor,
        "avgDurationHours": avg_dur,
        "totalPnlCents":    total_pnl,
    }


# ── Adaptive Param History ─────────────────────────────────────────────────────

def log_param_change(settled_count: int, param_name: str,
                     old_value: float, new_value: float, reason: str = "") -> None:
    """Record a parameter adjustment made by the adaptive tuner."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO param_history (settled_count, param_name, old_value, new_value, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            (settled_count, param_name, old_value, new_value, reason),
        )


def get_param_history(limit: int = 50) -> list[dict]:
    """Return last N parameter adjustments, newest first."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT timestamp, settled_count, param_name, old_value, new_value, reason
            FROM param_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ── Migration ──────────────────────────────────────────────────────────────────

def migrate_from_jsonl(jsonl_path: Path, cycles_path: Path | None = None) -> tuple[int, int]:
    """Import existing JSONL into SQLite. Returns (trades_imported, cycles_imported)."""
    init_db()
    t_count = 0
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                if t.get("action") not in ("BUY_YES", "BUY_NO"):
                    continue
                # Compute pnl_cents if missing
                if t.get("pnl_cents") is None:
                    if t.get("result_status") == "won":
                        t["pnl_cents"] = t.get("contracts", 0) * (100 - t.get("price_cents", 50))
                    elif t.get("result_status") == "lost":
                        t["pnl_cents"] = -t.get("cost_cents", 0)
                upsert_trade_by_ticker_ts(t)
                t_count += 1
            except Exception as e:
                print(f"  skip: {e}")

    c_count = 0
    if cycles_path and cycles_path.exists():
        for line in cycles_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                insert_cycle(json.loads(line))
                c_count += 1
            except Exception:
                pass

    return t_count, c_count


if __name__ == "__main__":
    from pathlib import Path as P
    DATA = P(__file__).parent / "data" / "trading"
    print("Initialising DB…")
    init_db()
    print(f"DB: {DB_PATH}")
    t, c = migrate_from_jsonl(
        DATA / "kalshi-unified-trades.jsonl",
        DATA / "kalshi-unified-cycles.jsonl",
    )
    print(f"Imported {t} trades, {c} cycles")
    m = get_metrics()
    print(f"Metrics: settled={m['settled']}, WR={m['win_rate']}%, ROI={m['roi_pct']}%")
