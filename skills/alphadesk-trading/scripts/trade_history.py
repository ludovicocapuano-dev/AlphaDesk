#!/usr/bin/env python3
"""AlphaDesk — Trade History Query."""

import json
import os
import sqlite3
import sys

sys.path.insert(0, "/root/AlphaDesk")
os.chdir("/root/AlphaDesk")

from config.settings import config


def trade_history(days: int = 30, strategy: str = None):
    db_path = config.db_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    results = {"status": "ok", "trades": [], "performance": [], "signals_recent": []}

    try:
        # Recent trades
        query = """
            SELECT * FROM trades
            WHERE opened_at >= datetime('now', ?)
        """
        params = [f"-{days} days"]
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        query += " ORDER BY opened_at DESC LIMIT 50"

        rows = conn.execute(query, params).fetchall()
        results["trades"] = [dict(r) for r in rows]

        # Strategy performance summary
        perf_query = """
            SELECT strategy,
                   COUNT(*) as trades,
                   AVG(pnl_pct) as avg_pnl_pct,
                   SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
                   SUM(pnl_dollars) as total_pnl
            FROM trades
            WHERE closed_at IS NOT NULL
            GROUP BY strategy
            ORDER BY total_pnl DESC
        """
        rows = conn.execute(perf_query).fetchall()
        results["performance"] = [dict(r) for r in rows]

        # Recent signals
        sig_query = """
            SELECT symbol, strategy_name, signal, confidence, executed,
                   created_at
            FROM signals
            ORDER BY created_at DESC LIMIT 20
        """
        rows = conn.execute(sig_query).fetchall()
        results["signals_recent"] = [dict(r) for r in rows]

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
    finally:
        conn.close()

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    strategy = sys.argv[2] if len(sys.argv) > 2 else None
    trade_history(days, strategy)
