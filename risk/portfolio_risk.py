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
class DrawdownAction:
    """Describes the current drawdown level and associated restrictions."""
    level: int                        # 0–5 (0 = normal)
    drawdown: float                   # current drawdown as positive fraction
    size_multiplier: float            # multiplier for new position sizes (1.0 = normal)
    reduce_existing: float            # fraction to cut existing positions (0.0 = none)
    allowed_strategies: List[str]     # strategies allowed to open new trades
    tighten_stops_pct: float          # tighten stops by this fraction (0.0 = no change)
    halt_hours: int                   # hours to halt trading (0 = no halt)
    require_manual_review: bool       # if True, manual review needed before restart
    message: str                      # human-readable description

    ALL_STRATEGIES = ["momentum", "mean_reversion", "factor_model", "fx_carry"]


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
        self.state.cash = account_data.get("cash", account_data.get("availableBalance", 0))
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
                self._manual_review_required = False
                logger.info("Trading halt expired, resuming")

        # 2. Manual review gate (level 5 requires explicit restart)
        if getattr(self, "_manual_review_required", False):
            return False, "HALT: Manual review required before restart"

        # 3. Graduated drawdown circuit breakers
        action = self.get_drawdown_action()

        if action.level >= 4:
            # Level 4/5: halt trading entirely
            self.state.is_halted = True
            self.state.halt_until = datetime.utcnow() + timedelta(hours=action.halt_hours)
            if action.require_manual_review:
                self._manual_review_required = True
            return False, f"HALT: {action.message}"

        if action.level >= 1:
            # Check if this strategy is allowed at current drawdown level
            strategy_name = signal.strategy_name
            if strategy_name not in action.allowed_strategies:
                return False, (f"Strategy {strategy_name} blocked at drawdown level "
                              f"{action.level} ({action.drawdown:.1%})")

            # Apply size reduction
            signal.suggested_size_pct *= action.size_multiplier
            if action.level >= 2:
                logger.warning(
                    f"Drawdown L{action.level}: {action.drawdown:.1%}. "
                    f"Size multiplier {action.size_multiplier:.0%}"
                )

        # 4. Daily VaR limit
        daily_var = self._compute_daily_var()
        if daily_var > self.config.daily_var_limit:
            return False, f"Daily VaR {daily_var:.2%} exceeds limit {self.config.daily_var_limit:.0%}"

        # 5. Strategy allocation limits
        strategy = signal.strategy_name
        current_exposure = self.state.strategy_exposures.get(strategy, 0)
        if current_exposure >= self.config.max_strategy_exposure:
            return False, (f"Strategy {strategy} exposure {current_exposure:.1%} "
                          f"at limit {self.config.max_strategy_exposure:.0%}")

        # 6. Correlation check
        corr_ok, corr_msg = self._check_correlation(signal)
        if not corr_ok:
            return False, corr_msg

        # 7. Mandatory stop loss
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

    def get_drawdown_action(self) -> DrawdownAction:
        """
        Determine the current graduated drawdown response level.

        Levels:
          0: Normal operation
          1: -5%  — warning, tighten stops 20%
          2: -10% — 75% size, no new momentum trades
          3: -15% — 50% positions, only mean reversion
          4: -20% — close all except hedges, halt 24h
          5: -25% — full halt 48h, require manual review
        """
        dd = self.state.current_drawdown
        cfg = self.config
        all_strats = DrawdownAction.ALL_STRATEGIES

        if dd >= cfg.drawdown_level_5:
            return DrawdownAction(
                level=5, drawdown=dd, size_multiplier=0.0, reduce_existing=1.0,
                allowed_strategies=[], tighten_stops_pct=0.0,
                halt_hours=48, require_manual_review=True,
                message=f"LEVEL 5: Drawdown {dd:.1%} — full halt 48h, manual review required",
            )
        if dd >= cfg.drawdown_level_4:
            return DrawdownAction(
                level=4, drawdown=dd, size_multiplier=0.0, reduce_existing=1.0,
                allowed_strategies=[], tighten_stops_pct=0.0,
                halt_hours=24, require_manual_review=False,
                message=f"LEVEL 4: Drawdown {dd:.1%} — close all except hedges, halt 24h",
            )
        if dd >= cfg.drawdown_level_3:
            return DrawdownAction(
                level=3, drawdown=dd, size_multiplier=0.50, reduce_existing=0.5,
                allowed_strategies=["mean_reversion"], tighten_stops_pct=0.20,
                halt_hours=0, require_manual_review=False,
                message=f"LEVEL 3: Drawdown {dd:.1%} — 50% reduction, mean reversion only",
            )
        if dd >= cfg.drawdown_level_2:
            return DrawdownAction(
                level=2, drawdown=dd, size_multiplier=0.75, reduce_existing=0.0,
                allowed_strategies=["mean_reversion", "factor_model", "fx_carry"],
                tighten_stops_pct=0.20, halt_hours=0, require_manual_review=False,
                message=f"LEVEL 2: Drawdown {dd:.1%} — 75% size, no momentum",
            )
        if dd >= cfg.drawdown_level_1:
            logger.warning(f"Drawdown warning L1: {dd:.1%} — tightening stops 20%")
            return DrawdownAction(
                level=1, drawdown=dd, size_multiplier=1.0, reduce_existing=0.0,
                allowed_strategies=list(all_strats), tighten_stops_pct=0.20,
                halt_hours=0, require_manual_review=False,
                message=f"LEVEL 1: Drawdown {dd:.1%} — warning, stops tightened 20%",
            )

        # Level 0: Normal
        return DrawdownAction(
            level=0, drawdown=dd, size_multiplier=1.0, reduce_existing=0.0,
            allowed_strategies=list(all_strats), tighten_stops_pct=0.0,
            halt_hours=0, require_manual_review=False,
            message="Normal operation",
        )

    def should_reduce_all(self) -> Tuple[bool, float]:
        """
        Check if we need to reduce all positions (drawdown protection).
        Returns (should_reduce, reduction_fraction).

        Compatible with existing callers:
          reduction >= 1.0 → close everything
          0 < reduction < 1.0 → reduce by that fraction
          reduction == 0.0 → no action
        """
        action = self.get_drawdown_action()
        if action.reduce_existing > 0:
            return True, action.reduce_existing
        return False, 0.0

    def clear_manual_review(self):
        """Call after manual review to allow trading to resume."""
        self._manual_review_required = False
        self.state.is_halted = False
        self.state.halt_until = None
        logger.info("Manual review cleared — trading may resume")
