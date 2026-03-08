#!/usr/bin/env python3
"""AlphaDesk — Risk Monitor (read-only assessment)."""

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


async def risk_check():
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

        # Drawdown analysis
        dd_action = risk.get_drawdown_action()
        should_reduce, reduction = risk.should_reduce_all()

        # Strategy qualification
        strategy_names = ["momentum", "mean_reversion", "factor_model", "fx_carry", "pead"]
        strategy_health = []
        for name in strategy_names:
            perf = db.get_strategy_performance(name, days=180)
            strategy_health.append({
                "name": name,
                "trades": perf["trades"],
                "sharpe": perf.get("sharpe", 0),
                "win_rate": perf.get("win_rate", 0),
                "qualified": perf["trades"] >= 50 and perf.get("sharpe", 0) > 0.3,
                "avg_pnl_pct": perf.get("avg_pnl_pct", 0),
            })

        # Position concentration
        concentration = {}
        for p in positions:
            sym = p.get("symbol", "?")
            amt = p.get("initialAmountInDollars", 0)
            concentration[sym] = concentration.get(sym, 0) + amt

        top_positions = sorted(concentration.items(), key=lambda x: x[1], reverse=True)[:5]

        print(json.dumps({
            "status": "ok",
            "equity": balance["equity"],
            "drawdown_pct": summary.get("current_drawdown", 0),
            "drawdown_level": dd_action.level if dd_action else 0,
            "drawdown_action": {
                "size_multiplier": dd_action.size_multiplier if dd_action else 1.0,
                "allowed_strategies": dd_action.allowed_strategies if dd_action else "all",
                "halt_hours": dd_action.halt_hours if dd_action else 0,
            } if dd_action else None,
            "should_reduce": should_reduce,
            "reduction_pct": reduction,
            "is_halted": risk.state.is_halted,
            "n_positions": len(positions),
            "top_positions": [{"symbol": s, "amount": a} for s, a in top_positions],
            "strategy_health": strategy_health,
            "alerts": _generate_alerts(summary, dd_action, strategy_health),
        }, indent=2, default=str))

    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
    finally:
        await etoro.close()


def _generate_alerts(summary, dd_action, strategy_health):
    alerts = []
    dd = summary.get("current_drawdown", 0)
    if dd > 0.05:
        alerts.append(f"Drawdown at {dd:.1%} — Level {dd_action.level if dd_action else 0}")
    if dd > 0.15:
        alerts.append("CRITICAL: Drawdown > 15% — consider manual intervention")

    for sh in strategy_health:
        if sh["trades"] > 20 and sh["sharpe"] < 0:
            alerts.append(f"{sh['name']}: negative Sharpe ({sh['sharpe']:.2f}) over 180 days")

    return alerts


if __name__ == "__main__":
    asyncio.run(risk_check())
