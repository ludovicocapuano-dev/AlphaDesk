"""
AlphaDesk — Portfolio Risk Manager
VaR, drawdown control, correlation limits, circuit breakers.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("alphadesk.risk.portfolio")


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk assessment."""
    equity: float = 0
    cash: float = 0
    positions: List[dict] = field(default_factory=list)
    peak_equity: float = 0
    current_drawdown: float = 0
    daily_pnl: float = 0
    strategy_exposures: Dict[str, float] = field(default_factory=dict)
    is_halted: bool = False
    halt_until: Optional[datetime] = None


class PortfolioRiskManager:
    """
    Portfolio-level risk management:
    - Drawdown monitoring with circuit breakers
    - VaR computation (parametric + historical)
    - Correlation-based exposure limits
    - Strategy-level allocation enforcement
    """

    def __init__(self, config):
        self.config = config
        self.state = PortfolioState()
        self._pnl_history: List[float] = []
        self._equity_history: List[Tuple[datetime, float]] = []

    def update_state(self, account_data: dict, positions: List[dict]):
        """Update portfolio state from eToro account data."""
        self.state.equity = account_data.get("equity", 0)
        self.state.cash = account_data.get("availableBalance", 0)
        self.state.positions = positions

        # Update peak equity
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity

        # Compute drawdown
        if self.state.peak_equity > 0:
            self.state.current_drawdown = (
                (self.state.peak_equity - self.state.equity) / self.state.peak_equity
            )

        # Track equity history
        self._equity_history.append((datetime.utcnow(), self.state.equity))

        # Strategy exposures
        self._compute_strategy_exposures()

    def _compute_strategy_exposures(self):
        """Compute gross exposure per strategy."""
        exposures: Dict[str, float] = {}
        for pos in self.state.positions:
            strategy = pos.get("strategy_tag", "unknown")
            amount = abs(pos.get("investedAmount", 0))
            exposures[strategy] = exposures.get(strategy, 0) + amount

        if self.state.equity > 0:
            self.state.strategy_exposures = {
                k: v / self.state.equity for k, v in exposures.items()
            }

    # ────────────────────── Risk Checks ──────────────────────

    def check_can_trade(self, signal) -> Tuple[bool, str]:
        """
        Master risk check before executing any trade.
        Returns (allowed, reason).
        """
        # 1. Halt check
        if self.state.is_halted:
            if self.state.halt_until and datetime.utcnow() < self.state.halt_until:
                return False, f"Trading halted until {self.state.halt_until}"
            else:
                self.state.is_halted = False
                logger.info("Trading halt expired, resuming")

        # 2. Drawdown circuit breakers
        dd = self.state.current_drawdown
        if dd >= self.config.max_drawdown_halt:
            self.state.is_halted = True
            self.state.halt_until = datetime.utcnow() + timedelta(hours=48)
            return False, f"HALT: Drawdown {dd:.1%} exceeds {self.config.max_drawdown_halt:.0%}"

        if dd >= self.config.max_drawdown_reduce:
            # Allow only at 50% normal size
            logger.warning(f"Drawdown warning: {dd:.1%}. Reducing position sizes 50%")
            # Signal to caller to halve sizes — we still allow the trade
            signal.suggested_size_pct *= 0.5

        # 3. Daily VaR limit
        daily_var = self._compute_daily_var()
        if daily_var > self.config.daily_var_limit:
            return False, f"Daily VaR {daily_var:.2%} exceeds limit {self.config.daily_var_limit:.0%}"

        # 4. Strategy allocation limits
        strategy = signal.strategy_name
        current_exposure = self.state.strategy_exposures.get(strategy, 0)
        if current_exposure >= self.config.max_strategy_exposure:
            return False, (f"Strategy {strategy} exposure {current_exposure:.1%} "
                          f"at limit {self.config.max_strategy_exposure:.0%}")

        # 5. Correlation check
        corr_ok, corr_msg = self._check_correlation(signal)
        if not corr_ok:
            return False, corr_msg

        # 6. Mandatory stop loss
        if self.config.mandatory_stop_loss and signal.stop_loss == signal.entry_price:
            return False, "Trade rejected: no stop loss defined"

        return True, "OK"

    def _compute_daily_var(self, confidence: float = 0.95) -> float:
        """
        Compute parametric daily VaR.
        Uses recent daily P&L to estimate portfolio risk.
        """
        if len(self._pnl_history) < 10:
            return 0  # Not enough data

        returns = np.array(self._pnl_history[-60:])  # Last 60 days
        if self.state.equity <= 0:
            return 0

        pct_returns = returns / self.state.equity
        var = np.percentile(pct_returns, (1 - confidence) * 100)
        return abs(var)

    def _check_correlation(self, signal) -> Tuple[bool, str]:
        """Check if new position would exceed correlated exposure limits."""
        # Simplified: count positions in same sector/pair
        same_group = 0
        signal_sector = signal.metadata.get("sector", "")

        for pos in self.state.positions:
            pos_sector = pos.get("sector", "")
            if pos_sector and pos_sector == signal_sector:
                same_group += 1

        max_correlated = 3  # Max 3 positions in same sector
        if same_group >= max_correlated:
            return False, f"Max correlated positions ({max_correlated}) in sector {signal_sector}"

        return True, "OK"

    # ────────────────────── Portfolio Analytics ──────────────────────

    def get_portfolio_summary(self) -> dict:
        """Generate portfolio risk summary for monitoring."""
        total_invested = sum(
            abs(p.get("investedAmount", 0)) for p in self.state.positions
        )

        return {
            "equity": self.state.equity,
            "cash": self.state.cash,
            "total_invested": total_invested,
            "gross_exposure": total_invested / self.state.equity if self.state.equity > 0 else 0,
            "current_drawdown": self.state.current_drawdown,
            "peak_equity": self.state.peak_equity,
            "num_positions": len(self.state.positions),
            "strategy_exposures": self.state.strategy_exposures,
            "daily_var_95": self._compute_daily_var(0.95),
            "is_halted": self.state.is_halted,
            "halt_until": str(self.state.halt_until) if self.state.halt_until else None,
        }

    def record_daily_pnl(self, pnl: float):
        """Record daily P&L for VaR computation."""
        self._pnl_history.append(pnl)
        self.state.daily_pnl = pnl

    def should_reduce_all(self) -> Tuple[bool, float]:
        """Check if we need to reduce all positions (drawdown protection)."""
        dd = self.state.current_drawdown
        if dd >= self.config.max_drawdown_halt:
            return True, 1.0  # Close everything
        elif dd >= self.config.max_drawdown_reduce:
            return True, 0.5  # Cut 50%
        return False, 0.0
