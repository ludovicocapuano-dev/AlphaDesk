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
    max_drawdown_reduce: float = 0.15      # -15% → cut positions 50%
    max_drawdown_halt: float = 0.25        # -25% → close all, pause 48h
    daily_var_limit: float = 0.03          # 3% daily VaR (95%)
    kelly_fraction: float = 0.5            # Half-Kelly
    slippage_budget_equity: float = 0.001  # 0.1%
    slippage_budget_fx: float = 0.0005     # 0.05%
    mandatory_stop_loss: bool = True


@dataclass
class StrategyAllocation:
    """Capital allocation per strategy."""
    momentum: float = 0.30      # 30%
    mean_reversion: float = 0.20  # 20%
    factor_model: float = 0.20   # 20%
    fx_carry: float = 0.30       # 30%


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
