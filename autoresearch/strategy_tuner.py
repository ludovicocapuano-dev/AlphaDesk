"""
AutoResearch — Strategy Parameters (AGENT MODIFIES THIS FILE)

Each dict below controls one strategy. The agent experiments by changing
these values. Valid ranges are shown in comments.

After modifying, run:
    bash autoresearch/run_experiment.sh <strategy_name> <experiment_id>
"""

# ═══════════════════════════════════════════════════════════════════
# MOMENTUM STRATEGY
# ═══════════════════════════════════════════════════════════════════
MOMENTUM_PARAMS = {
    "breakout_period": 20,                # [10, 40]
    "trend_sma": 40,                      # [20, 100]
    "long_trend_sma": 210,                # [100, 252]
    "volume_threshold": 1.0,              # [0.5, 3.0]
    "atr_multiplier": 3.25,               # [1.0, 4.0]
    "min_momentum_3m": 0.03,              # [0.0, 0.15]
    "rsi_overbought": 82,                 # [65, 85]
    "rsi_oversold": 25,                   # [20, 40]
    "max_positions": 12,                  # [3, 15]
    "min_confidence": 0.49,               # [0.3, 0.8]
    "min_rr_ratio": 1.0,                  # [0.8, 3.0]
    "tp_atr_multiplier": 5.5,             # [1.5, 8.0]
}

# ═══════════════════════════════════════════════════════════════════
# MEAN REVERSION STRATEGY
# ═══════════════════════════════════════════════════════════════════
MEAN_REVERSION_PARAMS = {
    "z_entry_long": -2.1,                 # [-3.0, -1.0]
    "z_entry_short": 2.2,                 # [1.0, 3.0]
    "z_exit": 0.15,                       # [0.0, 1.0]
    "z_stop": 3.5,                        # [2.5, 5.0]
    "min_lookback": 45,                   # [30, 120]
    "rsi_long_threshold": 22,             # [20, 45]
    "rsi_short_threshold": 78,            # [55, 80]
    "sl_atr_multiplier": 1.8,             # [1.0, 3.0]
    "max_positions": 10,                  # [3, 12]
    "min_confidence": 0.5,                # [0.3, 0.8]
    "min_rr_ratio": 1.0,                  # [0.5, 2.5]
}

# ═══════════════════════════════════════════════════════════════════
# FACTOR MODEL STRATEGY
# ═══════════════════════════════════════════════════════════════════
FACTOR_MODEL_PARAMS = {
    "value_weight": 0.3,                  # [0.1, 0.6]
    "quality_weight": 0.46,               # [0.1, 0.6]
    "momentum_weight": 0.24,              # [0.1, 0.6]
    "rebalance_days": 19,                 # [5, 63]
    "stop_loss_pct": 0.03,                # [0.03, 0.15]
    "take_profit_pct": 0.28,              # [0.05, 0.3]
    "max_positions": 15,                  # [5, 20]
    "min_data_days": 60,                  # [60, 252]
    "min_composite": 0.46,                # [0.3, 0.7]
}

# ═══════════════════════════════════════════════════════════════════
# FX CARRY STRATEGY
# ═══════════════════════════════════════════════════════════════════
FX_CARRY_PARAMS = {
    "min_carry_spread": 0.01,             # [0.001, 0.03]
    "carry_weight": 0.6,                  # [0.2, 0.9]
    "momentum_weight": 0.5,               # [0.1, 0.8]
    "trend_filter_sma": 20,               # [20, 100]
    "atr_stop_multiplier": 1.72,          # [1.0, 3.0]
    "min_composite_score": 0.1,           # [0.01, 0.15]
    "max_positions": 8,                   # [2, 10]
    "min_confidence": 0.45,               # [0.3, 0.8]
    "min_rr_ratio": 1.0,                  # [0.5, 2.0]
    "max_risk_per_pair": 0.03,            # [0.005, 0.04]
}

# ═══════════════════════════════════════════════════════════════════
# BACKTEST CONFIG
# ═══════════════════════════════════════════════════════════════════
BACKTEST_CONFIG = {
    "initial_capital": 100_000,
    "commission_pct": 0.001,
    "slippage_pct": 0.0005,
    "max_positions": 20,
    "risk_per_trade": 0.05,
    "start_date": "2023-01-01",
    "end_date": "2025-12-31",
}


def get_params(strategy_name):
    """Return params dict for a strategy."""
    return {
        "momentum": MOMENTUM_PARAMS,
        "mean_reversion": MEAN_REVERSION_PARAMS,
        "factor_model": FACTOR_MODEL_PARAMS,
        "fx_carry": FX_CARRY_PARAMS,
    }[strategy_name]


def apply_params(strategy, params):
    """Apply tuner params to a strategy instance."""
    for key, value in params.items():
        if hasattr(strategy, key):
            setattr(strategy, key, value)
