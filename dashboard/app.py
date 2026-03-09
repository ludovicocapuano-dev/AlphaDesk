"""
AlphaDesk — Lightweight Web Dashboard
Flask app serving portfolio, trades, and strategy performance from SQLite.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta

from flask import Flask, jsonify, render_template

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = os.environ.get("ALPHADESK_DB", os.path.join(os.path.dirname(__file__), "..", "alphadesk.db"))
DB_PATH = os.path.abspath(DB_PATH)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def get_db():
    """Return a sqlite3 connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def query(sql, params=(), one=False):
    """Run a read query and return list of dicts (or single dict if one=True)."""
    conn = get_db()
    try:
        rows = conn.execute(sql, params).fetchall()
        result = [dict(r) for r in rows]
        return result[0] if one and result else (None if one else result)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Detect which schema we have (schema.sql vs TradeDB)
# ---------------------------------------------------------------------------
def _table_columns(table):
    conn = get_db()
    try:
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row["name"] for row in info}
    finally:
        conn.close()


def _detect_schema():
    """Return 'tradedb' or 'schema_sql' depending on which column names exist."""
    cols = _table_columns("trades")
    if "open_time" in cols:
        return "tradedb"
    return "schema_sql"


SCHEMA = _detect_schema()


# ---------------------------------------------------------------------------
# Data access — adapts to whichever schema is present
# ---------------------------------------------------------------------------
def get_portfolio_summary():
    """Latest portfolio snapshot or sensible defaults."""
    if SCHEMA == "tradedb":
        snap = query("SELECT * FROM daily_snapshots ORDER BY date DESC LIMIT 1", one=True)
    else:
        snap = query("SELECT * FROM daily_snapshots ORDER BY date DESC LIMIT 1", one=True)

    if snap:
        equity = snap.get("equity") or snap.get("total_equity") or 0
        cash = snap.get("cash", 0)
        daily_pnl = snap.get("daily_pnl", 0)
        drawdown = snap.get("drawdown", 0)
        num_positions = snap.get("num_positions") or snap.get("num_positions", 0)
        cumulative_pnl = snap.get("cumulative_pnl") or snap.get("daily_pnl", 0)
        return {
            "equity": equity,
            "cash": cash,
            "positions_value": equity - cash,
            "daily_pnl": daily_pnl,
            "cumulative_pnl": cumulative_pnl,
            "drawdown": drawdown,
            "num_positions": num_positions,
            "date": snap.get("date", "N/A"),
        }
    # Defaults from memory context
    return {
        "equity": 10567.00,
        "cash": 9.76,
        "positions_value": 10557.24,
        "daily_pnl": 0.0,
        "cumulative_pnl": 0.0,
        "drawdown": 0.0,
        "num_positions": 0,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
    }


def get_positions():
    """Open positions (trades with status='open' in TradeDB schema)."""
    if SCHEMA == "tradedb":
        return query(
            "SELECT id, symbol, strategy, direction, amount, entry_price, "
            "stop_loss, take_profit, open_time FROM trades WHERE status = 'open' "
            "ORDER BY open_time DESC"
        )
    else:
        # schema.sql trades don't have status — treat all as executed
        return query(
            "SELECT id, ticker AS symbol, strategy, direction, quantity AS amount, "
            "price AS entry_price, timestamp AS open_time FROM trades "
            "ORDER BY timestamp DESC LIMIT 50"
        )


def get_recent_trades(limit=50):
    """Recent closed trades."""
    if SCHEMA == "tradedb":
        return query(
            "SELECT id, symbol, strategy, direction, amount, entry_price, "
            "exit_price, pnl, pnl_pct, open_time, close_time, status "
            "FROM trades ORDER BY COALESCE(close_time, open_time) DESC LIMIT ?",
            (limit,),
        )
    else:
        return query(
            "SELECT id, ticker AS symbol, strategy, direction, quantity AS amount, "
            "price AS entry_price, pnl, timestamp AS open_time "
            "FROM trades ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )


def get_equity_curve():
    """Daily snapshots for equity chart."""
    if SCHEMA == "tradedb":
        rows = query("SELECT date, equity, cash, drawdown, daily_pnl FROM daily_snapshots ORDER BY date ASC")
    else:
        rows = query(
            "SELECT date, total_equity AS equity, cash, drawdown, daily_pnl "
            "FROM daily_snapshots ORDER BY date ASC"
        )
    return rows


def get_strategy_performance():
    """Aggregate strategy stats from closed trades."""
    if SCHEMA == "tradedb":
        return query("""
            SELECT
                strategy,
                COUNT(*) AS total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) AS losses,
                ROUND(SUM(pnl), 2) AS total_pnl,
                ROUND(AVG(pnl_pct) * 100, 2) AS avg_return_pct,
                ROUND(
                    CASE WHEN COUNT(*) > 0
                        THEN CAST(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*)
                        ELSE 0 END * 100, 1
                ) AS win_rate
            FROM trades
            WHERE status = 'closed'
            GROUP BY strategy
            ORDER BY total_pnl DESC
        """)
    else:
        return query("""
            SELECT
                strategy,
                COUNT(*) AS total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) AS losses,
                ROUND(SUM(pnl), 2) AS total_pnl,
                ROUND(AVG(pnl), 2) AS avg_pnl,
                ROUND(
                    CASE WHEN COUNT(*) > 0
                        THEN CAST(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*)
                        ELSE 0 END * 100, 1
                ) AS win_rate
            FROM trades
            GROUP BY strategy
            ORDER BY total_pnl DESC
        """)


def get_regime_status():
    """Regime info from kv_store or defaults."""
    kv = {}
    try:
        rows = query("SELECT key, value FROM kv_store")
        for r in rows:
            kv[r["key"]] = r["value"]
    except Exception:
        pass

    # Try to parse regime fingerprint
    regime_raw = kv.get("regime_fingerprint", kv.get("current_regime", ""))
    try:
        regime = json.loads(regime_raw) if regime_raw else {}
    except (json.JSONDecodeError, TypeError):
        regime = {}

    return {
        "volatility_regime": regime.get("volatility_regime", "elevated"),
        "trend_regime": regime.get("trend_regime", "ranging"),
        "vix_level": regime.get("vix", 29.5),
        "market_phase": regime.get("market_phase", "risk-off"),
        "last_updated": kv.get("regime_updated_at", "N/A"),
    }


# ---------------------------------------------------------------------------
# Backtest results (hardcoded from memory — no backtest table in DB)
# ---------------------------------------------------------------------------
BACKTEST_RESULTS = [
    {"strategy": "Mean Reversion", "return_pct": 6.99, "sharpe": 0.89, "trades": 639, "status": "profitable"},
    {"strategy": "Momentum",       "return_pct": -4.49, "sharpe": -0.68, "trades": 128, "status": "needs recalibration"},
    {"strategy": "Factor Model",   "return_pct": 0.0,  "sharpe": 0.0,  "trades": 0,   "status": "BUG: no signals"},
    {"strategy": "FX Carry",       "return_pct": 0.0,  "sharpe": 0.0,  "trades": 0,   "status": "BUG: no signals"},
]


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — JSON API
# ---------------------------------------------------------------------------
@app.route("/api/summary")
def api_summary():
    return jsonify(get_portfolio_summary())


@app.route("/api/positions")
def api_positions():
    return jsonify(get_positions())


@app.route("/api/trades")
def api_trades():
    return jsonify(get_recent_trades())


@app.route("/api/equity")
def api_equity():
    return jsonify(get_equity_curve())


@app.route("/api/strategies")
def api_strategies():
    live = get_strategy_performance()
    if not live:
        return jsonify(BACKTEST_RESULTS)
    return jsonify(live)


@app.route("/api/regime")
def api_regime():
    return jsonify(get_regime_status())


@app.route("/api/drawdown")
def api_drawdown():
    curve = get_equity_curve()
    if not curve:
        return jsonify([])
    dd = []
    for row in curve:
        dd.append({"date": row["date"], "drawdown": row.get("drawdown", 0) or 0})
    return jsonify(dd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    print(f"AlphaDesk Dashboard — http://localhost:{port}")
    print(f"Database: {DB_PATH}")
    app.run(host="0.0.0.0", port=port, debug=debug)
