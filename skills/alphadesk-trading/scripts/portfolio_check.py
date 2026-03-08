#!/usr/bin/env python3
"""AlphaDesk — Portfolio Check (read-only snapshot)."""

import asyncio
import json
import os
import sys

sys.path.insert(0, "/root/AlphaDesk")
os.chdir("/root/AlphaDesk")

from config.settings import config
from core.etoro_client import EtoroClient
from risk.portfolio_risk import PortfolioRiskManager
from utils.db import TradeDB


async def portfolio_check():
    etoro = EtoroClient(
        user_key=config.etoro.user_key,
        api_key=config.etoro.api_key,
        base_url=config.etoro.base_url,
        environment=config.etoro.environment,
        timeout=config.etoro.request_timeout,
        max_retries=config.etoro.max_retries,
    )
    risk = PortfolioRiskManager(config.risk)
    db = TradeDB(config.db_path)

    try:
        portfolio = await etoro.get_portfolio()
        cp = portfolio.get("clientPortfolio", portfolio)
        positions = cp.get("positions", [])
        credit = cp.get("credit", 0)

        total_invested = sum(p.get("initialAmountInDollars", 0) for p in positions)
        balance = {
            "cash": credit,
            "equity": credit + total_invested,
            "invested": total_invested,
        }
        risk.update_state(balance, positions)
        summary = risk.get_portfolio_summary()

        # Get drawdown action
        dd_action = risk.get_drawdown_action()

        print(json.dumps({
            "status": "ok",
            "equity": balance["equity"],
            "cash": balance["cash"],
            "invested": balance["invested"],
            "n_positions": len(positions),
            "positions": [
                {
                    "symbol": p.get("symbol", "?"),
                    "instrumentId": p.get("instrumentId"),
                    "amount": p.get("initialAmountInDollars", 0),
                    "currentRate": p.get("currentRate", 0),
                    "openRate": p.get("openRate", 0),
                    "pnl": p.get("netProfit", 0),
                    "pnl_pct": (p.get("netProfit", 0) / p.get("initialAmountInDollars", 1)) * 100
                    if p.get("initialAmountInDollars", 0) > 0 else 0,
                    "strategy": p.get("strategy_tag", "unknown"),
                }
                for p in positions
            ],
            "drawdown": summary.get("current_drawdown", 0),
            "drawdown_level": dd_action.level if dd_action else 0,
            "drawdown_action": dd_action.size_multiplier if dd_action else 1.0,
            "is_halted": risk.state.is_halted,
        }, indent=2, default=str))

    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
    finally:
        await etoro.close()


if __name__ == "__main__":
    asyncio.run(portfolio_check())
