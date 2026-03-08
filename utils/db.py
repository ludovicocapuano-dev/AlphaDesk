"""
AlphaDesk — Trade Journal Database
SQLite-based trade logging for audit, analytics, and backtesting.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("alphadesk.db")


class TradeDB:
    """SQLite trade journal and signal log."""

    def __init__(self, db_path: str = "data/alphadesk.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    risk_reward REAL,
                    metadata TEXT,
                    executed INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    open_time TEXT NOT NULL,
                    close_time TEXT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    amount REAL NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    etoro_position_id TEXT,
                    status TEXT DEFAULT 'open',
                    metadata TEXT,
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                );

                CREATE TABLE IF NOT EXISTS daily_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    equity REAL,
                    cash REAL,
                    num_positions INTEGER,
                    drawdown REAL,
                    daily_pnl REAL,
                    var_95 REAL,
                    strategy_exposures TEXT,
                    metadata TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
            """)
        logger.info(f"Database initialized: {self.db_path}")

    def log_signal(self, signal) -> int:
        """Log a generated signal. Returns signal ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO signals
                   (timestamp, symbol, strategy, signal_type, confidence,
                    entry_price, stop_loss, take_profit, risk_reward, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal.timestamp.isoformat(),
                    signal.symbol,
                    signal.strategy_name,
                    signal.signal.name,
                    signal.confidence,
                    signal.entry_price,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.risk_reward_ratio,
                    json.dumps(signal.metadata),
                ),
            )
            return cursor.lastrowid

    def log_trade_open(self, signal_id: int, trade_data: dict) -> int:
        """Log a trade execution."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO trades
                   (signal_id, open_time, symbol, strategy, direction,
                    amount, entry_price, stop_loss, take_profit,
                    etoro_position_id, status, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)""",
                (
                    signal_id,
                    datetime.utcnow().isoformat(),
                    trade_data["symbol"],
                    trade_data["strategy"],
                    trade_data["direction"],
                    trade_data["amount"],
                    trade_data.get("entry_price"),
                    trade_data.get("stop_loss"),
                    trade_data.get("take_profit"),
                    trade_data.get("etoro_position_id"),
                    json.dumps(trade_data.get("metadata", {})),
                ),
            )
            return cursor.lastrowid

    def log_trade_close(self, trade_id: int, exit_price: float, pnl: float):
        """Log trade closure."""
        with sqlite3.connect(self.db_path) as conn:
            # Get entry price for pnl_pct
            row = conn.execute(
                "SELECT entry_price FROM trades WHERE id = ?", (trade_id,)
            ).fetchone()
            pnl_pct = pnl / row[0] if row and row[0] else 0

            conn.execute(
                """UPDATE trades
                   SET close_time = ?, exit_price = ?, pnl = ?,
                       pnl_pct = ?, status = 'closed'
                   WHERE id = ?""",
                (datetime.utcnow().isoformat(), exit_price, pnl, pnl_pct, trade_id),
            )

    def save_daily_snapshot(self, summary: dict):
        """Save end-of-day portfolio snapshot."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO daily_snapshots
                   (date, equity, cash, num_positions, drawdown,
                    daily_pnl, var_95, strategy_exposures)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().strftime("%Y-%m-%d"),
                    summary.get("equity"),
                    summary.get("cash"),
                    summary.get("num_positions"),
                    summary.get("current_drawdown"),
                    summary.get("daily_pnl", 0),
                    summary.get("daily_var_95"),
                    json.dumps(summary.get("strategy_exposures", {})),
                ),
            )

    def get_strategy_performance(self, strategy: str, days: int = 90) -> dict:
        """Get historical performance for a strategy (for Kelly sizing)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT pnl_pct FROM trades
                   WHERE strategy = ? AND status = 'closed'
                     AND close_time >= datetime('now', ?)
                   ORDER BY close_time DESC""",
                (strategy, f"-{days} days"),
            ).fetchall()

        if not rows:
            return {"win_rate": 0.5, "avg_win": 0.02, "avg_loss": 0.01, "trades": 0, "n_trades": 0}

        returns = [r[0] for r in rows if r[0] is not None]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]

        n = len(returns)
        return {
            "win_rate": len(wins) / n if n else 0.5,
            "avg_win": sum(wins) / len(wins) if wins else 0.02,
            "avg_loss": sum(losses) / len(losses) if losses else -0.01,
            "trades": n,
            "n_trades": n,  # alias for Kelly sizer
            "total_return": sum(returns),
            "sharpe": (
                (sum(returns) / len(returns)) / (
                    (sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns))**0.5
                ) if len(returns) > 1 else 0
            ),
        }
