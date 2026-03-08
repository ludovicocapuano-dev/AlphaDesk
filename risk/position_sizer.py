"""
AlphaDesk — Position Sizer
Kelly Criterion and ATR-based position sizing.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("alphadesk.risk.sizer")


class PositionSizer:
    """Compute optimal position sizes using Kelly Criterion and risk limits."""

    def __init__(self, max_risk_per_trade: float = 0.05,
                 kelly_fraction: float = 0.5,
                 slippage_equity: float = 0.001,
                 slippage_fx: float = 0.0005):
        self.max_risk_per_trade = max_risk_per_trade
        self.kelly_fraction = kelly_fraction  # Half-Kelly for safety
        self.slippage_equity = slippage_equity
        self.slippage_fx = slippage_fx

    def kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Compute Kelly fraction for position sizing.

        Kelly % = W - [(1-W) / R]
        where W = win rate, R = win/loss ratio

        We use half-Kelly for safety.
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0

        r = avg_win / abs(avg_loss)  # Win/loss ratio
        kelly = win_rate - ((1 - win_rate) / r)

        # Apply fraction and cap
        sized = kelly * self.kelly_fraction
        return max(0, min(self.max_risk_per_trade, sized))

    def atr_based_size(self, account_equity: float, entry_price: float,
                        atr: float, atr_multiplier: float = 2.0,
                        risk_pct: float = None) -> dict:
        """
        ATR-based position sizing.
        Determines position size so that a move of atr_multiplier * ATR
        equals the max acceptable loss.

        Returns:
            dict with dollar_amount, units, risk_dollars, stop_distance
        """
        risk_pct = risk_pct or self.max_risk_per_trade
        risk_dollars = account_equity * risk_pct
        stop_distance = atr * atr_multiplier

        if stop_distance <= 0 or entry_price <= 0:
            return {"dollar_amount": 0, "units": 0, "risk_dollars": 0, "stop_distance": 0}

        # Number of units where loss at stop = risk_dollars
        units = risk_dollars / stop_distance
        dollar_amount = units * entry_price

        return {
            "dollar_amount": round(dollar_amount, 2),
            "units": round(units, 4),
            "risk_dollars": round(risk_dollars, 2),
            "stop_distance": round(stop_distance, 6),
        }

    def compute_trade_size(self, account_equity: float, signal,
                            historical_performance: dict = None,
                            asset_type: str = "equity") -> dict:
        """
        Compute final trade size combining Kelly and ATR methods.

        Args:
            account_equity: Total account equity
            signal: TradeSignal object
            historical_performance: dict with win_rate, avg_win, avg_loss
            asset_type: "equity" or "fx"

        Returns:
            dict with recommended position details
        """
        # Start with signal's suggested size
        base_size_pct = signal.suggested_size_pct

        # Adjust with Kelly if we have performance data
        if historical_performance:
            kelly = self.kelly_size(
                historical_performance.get("win_rate", 0.5),
                historical_performance.get("avg_win", 0.02),
                historical_performance.get("avg_loss", 0.01),
            )
            # Blend signal suggestion with Kelly
            base_size_pct = 0.6 * base_size_pct + 0.4 * kelly

        # Adjust for confidence
        adjusted_pct = base_size_pct * signal.confidence

        # ATR-based sanity check
        atr = signal.metadata.get("atr", 0)
        if atr > 0 and signal.entry_price > 0:
            atr_size = self.atr_based_size(
                account_equity, signal.entry_price, atr
            )
            max_atr_pct = atr_size["dollar_amount"] / account_equity if account_equity > 0 else 0
            adjusted_pct = min(adjusted_pct, max_atr_pct)

        # Hard cap
        adjusted_pct = min(adjusted_pct, self.max_risk_per_trade)

        # Apply slippage budget
        slippage = self.slippage_fx if asset_type == "fx" else self.slippage_equity
        dollar_amount = account_equity * adjusted_pct * (1 - slippage)

        # Minimum trade size check
        if dollar_amount < 50:  # eToro minimum
            return {"execute": False, "reason": "Below minimum trade size"}

        # Cap to available cash (passed via metadata or default to dollar_amount)
        available_cash = signal.metadata.get("available_cash", dollar_amount)
        if dollar_amount > available_cash:
            if available_cash < 50:
                return {"execute": False, "reason": f"Insufficient cash (${available_cash:.0f})"}
            dollar_amount = available_cash * 0.90  # Keep 10% cash buffer

        return {
            "execute": True,
            "dollar_amount": round(dollar_amount, 2),
            "position_pct": round(adjusted_pct, 4),
            "risk_dollars": round(account_equity * adjusted_pct * (
                abs(signal.entry_price - signal.stop_loss) / signal.entry_price
                if signal.entry_price > 0 else 0
            ), 2),
            "confidence": signal.confidence,
            "strategy": signal.strategy_name,
        }
