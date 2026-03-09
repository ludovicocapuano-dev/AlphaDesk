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

                -- Local position state: persists across restarts
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    etoro_position_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    instrument_id INTEGER,
                    strategy TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    amount REAL NOT NULL,
                    entry_price REAL,
                    open_rate REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    open_time TEXT NOT NULL,
                    status TEXT DEFAULT 'open',
                    close_time TEXT,
                    pnl REAL,
                    metadata TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
                CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
                CREATE INDEX IF NOT EXISTS idx_positions_etoro_id ON positions(etoro_position_id);
            """)

            # Migration: add columns to trades table if missing (old schema compat)
            for col, default in [("status", "'open'"), ("etoro_position_id", "NULL")]:
                try:
                    conn.execute(f"SELECT {col} FROM trades LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {col} TEXT DEFAULT {default}")
                    logger.info(f"Migrated trades table: added {col} column")

            # Create index on status only after migration
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
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

    # ────────────────────── Position Persistence ──────────────────────

    def save_position(self, position_data: dict) -> int:
        """Save or update a locally tracked position."""
        etoro_id = position_data.get("etoro_position_id")
        with sqlite3.connect(self.db_path) as conn:
            # Upsert: update if etoro_position_id exists, insert otherwise
            if etoro_id:
                existing = conn.execute(
                    "SELECT id FROM positions WHERE etoro_position_id = ?", (etoro_id,)
                ).fetchone()
                if existing:
                    conn.execute(
                        """UPDATE positions SET
                           amount = ?, open_rate = ?, stop_loss = ?, take_profit = ?,
                           metadata = ?
                           WHERE etoro_position_id = ?""",
                        (
                            position_data.get("amount"),
                            position_data.get("open_rate"),
                            position_data.get("stop_loss"),
                            position_data.get("take_profit"),
                            json.dumps(position_data.get("metadata", {})),
                            etoro_id,
                        ),
                    )
                    return existing[0]

            cursor = conn.execute(
                """INSERT INTO positions
                   (etoro_position_id, symbol, instrument_id, strategy, direction,
                    amount, entry_price, open_rate, stop_loss, take_profit,
                    open_time, status, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)""",
                (
                    etoro_id,
                    position_data["symbol"],
                    position_data.get("instrument_id"),
                    position_data["strategy"],
                    position_data["direction"],
                    position_data["amount"],
                    position_data.get("entry_price"),
                    position_data.get("open_rate"),
                    position_data.get("stop_loss"),
                    position_data.get("take_profit"),
                    position_data.get("open_time", datetime.utcnow().isoformat()),
                    json.dumps(position_data.get("metadata", {})),
                ),
            )
            return cursor.lastrowid

    def close_position(self, etoro_position_id: str, pnl: float = None):
        """Mark a position as closed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE positions SET status = 'closed', close_time = ?, pnl = ?
                   WHERE etoro_position_id = ?""",
                (datetime.utcnow().isoformat(), pnl, str(etoro_position_id)),
            )

    def get_open_positions(self) -> List[dict]:
        """Load all locally tracked open positions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM positions WHERE status = 'open'"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_strategy_for_position(self, etoro_position_id: str) -> Optional[str]:
        """Look up which strategy opened a position."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT strategy FROM positions WHERE etoro_position_id = ?",
                (str(etoro_position_id),),
            ).fetchone()
            return row[0] if row else None

    def reconcile_positions(self, etoro_positions: List[dict]) -> dict:
        """
        Reconcile local DB with eToro API positions.
        - Marks closed positions that are no longer in eToro
        - Enriches eToro positions with strategy tags from DB
        Returns: {"enriched": [...], "closed_stale": int, "unknown": int}
        """
        local_open = {
            str(p["etoro_position_id"]): p
            for p in self.get_open_positions()
            if p.get("etoro_position_id")
        }

        etoro_ids = set()
        enriched = []
        unknown = 0

        for pos in etoro_positions:
            pos_id = str(pos.get("positionID", pos.get("positionId", "")))
            etoro_ids.add(pos_id)

            if pos_id in local_open:
                # Enrich with local data (strategy, etc.)
                local = local_open[pos_id]
                pos["strategy_tag"] = local["strategy"]
                pos["local_id"] = local["id"]
            else:
                # Position opened outside our system (manual trade)
                pos["strategy_tag"] = "manual"
                unknown += 1

            enriched.append(pos)

        # Mark positions that are in DB but no longer in eToro as closed
        closed_stale = 0
        for local_id, local_pos in local_open.items():
            if local_id not in etoro_ids:
                self.close_position(local_id)
                closed_stale += 1
                logger.info(f"Position {local_pos['symbol']} (eToro {local_id}) closed externally")

        return {"enriched": enriched, "closed_stale": closed_stale, "unknown": unknown}

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
