"""
AlphaDesk — Configuration
Replace placeholders with your actual credentials.
NEVER commit this file to a public repository.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EtoroConfig:
    """eToro API configuration."""
    base_url: str = "https://public-api.etoro.com/api/v1"
    ws_url: str = "wss://public-api.etoro.com/ws"
    user_key: str = os.getenv("ETORO_USER_KEY", "YOUR_USER_KEY_HERE")
    api_key: str = os.getenv("ETORO_API_KEY", "YOUR_API_KEY_HERE")
    environment: str = os.getenv("ETORO_ENV", "Demo")  # "Demo" or "Real"
    request_timeout: int = 30
    max_retries: int = 3


@dataclass
class RiskConfig:
    """Risk management parameters — AGGRESSIVE profile."""
    max_risk_per_trade: float = 0.05       # 5% max risk per trade
    max_correlated_exposure: float = 0.15  # 15% max in correlated positions
    max_strategy_exposure: float = 0.25    # 25% gross per strategy
    daily_var_limit: float = 0.03          # 3% daily VaR (95%)
    kelly_fraction: float = 0.5            # Half-Kelly
    slippage_budget_equity: float = 0.001  # 0.1%
    slippage_budget_fx: float = 0.0005     # 0.05%
    mandatory_stop_loss: bool = True

    # Graduated drawdown circuit breakers
    drawdown_level_1: float = 0.05   # -5%  → warning, tighten stops 20%
    drawdown_level_2: float = 0.10   # -10% → 75% size, no momentum
    drawdown_level_3: float = 0.15   # -15% → 50% positions, mean reversion only
    drawdown_level_4: float = 0.20   # -20% → close all except hedges, halt 24h
    drawdown_level_5: float = 0.25   # -25% → full halt 48h, manual review


@dataclass
class StrategyAllocation:
    """Capital allocation per strategy.
    Updated March 2026: Risk-off regime favors mean reversion,
    reduced FX carry due to tariff-driven vol.
    """
    momentum: float = 0.25      # 25% (was 30% — reduced in risk-off)
    mean_reversion: float = 0.30  # 30% (was 20% — ranging markets favor MR)
    factor_model: float = 0.25   # 25% (was 20% — value rotation underway)
    fx_carry: float = 0.20       # 20% (was 30% — tariff uncertainty)

    def get_regime_adjusted(self, vix_level: float = None, trend_regime: str = None) -> dict:
        """Adjust allocations based on market regime.

        Args:
            vix_level: Current VIX value (drives risk-on/risk-off tilt).
            trend_regime: Trend regime string from RegimeFingerprint
                          (e.g. 'strong_up', 'ranging', 'strong_down').

        Returns:
            Dict with strategy name keys and float allocation values summing to 1.0.
        """
        if vix_level is not None and vix_level > 25:  # Risk-off
            return {
                'momentum': 0.10,
                'mean_reversion': 0.40,
                'factor_model': 0.30,
                'fx_carry': 0.20,
            }
        elif vix_level is not None and vix_level < 18:  # Risk-on
            return {
                'momentum': 0.35,
                'mean_reversion': 0.20,
                'factor_model': 0.25,
                'fx_carry': 0.20,
            }
        else:  # Normal — use static defaults
            return {
                'momentum': self.momentum,
                'mean_reversion': self.mean_reversion,
                'factor_model': self.factor_model,
                'fx_carry': self.fx_carry,
            }


@dataclass
class TelegramConfig:
    """Telegram alerting (optional)."""
    bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled: bool = bool(os.getenv("TELEGRAM_BOT_TOKEN", ""))


@dataclass
class AppConfig:
    """Master configuration."""
    etoro: EtoroConfig = field(default_factory=EtoroConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    allocation: StrategyAllocation = field(default_factory=StrategyAllocation)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    # Scheduling
    signal_scan_interval_minutes: int = 15    # Scan for signals every 15 min
    risk_check_interval_minutes: int = 5      # Risk checks every 5 min
    portfolio_rebalance_hour: int = 14        # Daily rebalance at 14:00 UTC

    # Database
    db_path: str = "data/alphadesk.db"
    log_path: str = "logs/alphadesk.log"


# Singleton
config = AppConfig()
