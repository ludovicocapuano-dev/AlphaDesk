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
    "breakout_period": 15,        # [10, 40] — lookback for breakout
    "trend_sma": 40,              # [20, 100] — short-term trend MA
    "long_trend_sma": 200,        # [100, 252] — long-term trend MA
    "volume_threshold": 1.0,      # [1.0, 3.0] — volume surge filter
    "atr_multiplier": 3.0,        # [1.0, 4.0] — stop loss distance
    "min_momentum_3m": 0.02,      # [0.0, 0.15] — min 3-month return
    "rsi_overbought": 82,         # [65, 85] — max RSI for entry
    "rsi_oversold": 25,           # [20, 40] — RSI sweet spot floor
    "max_positions": 12,          # [3, 15]
    "min_confidence": 0.45,       # [0.3, 0.8]
    "min_rr_ratio": 1.0,          # [0.8, 3.0]
    "tp_atr_multiplier": 5.0,     # [1.5, 6.0] — take profit = N * atr * ATR
}

# ═══════════════════════════════════════════════════════════════════
# MEAN REVERSION STRATEGY
# ═══════════════════════════════════════════════════════════════════
MEAN_REVERSION_PARAMS = {
    "z_entry_long": -2.8,         # [-3.0, -1.0] — buy when Z < this
    "z_entry_short": 2.8,         # [1.0, 3.0] — sell when Z > this
    "z_exit": 0.0,                # [0.0, 1.0] — close when Z returns to this
    "z_stop": 4.5,                # [2.5, 5.0] — hard stop Z level
    "min_lookback": 90,           # [30, 120]
    "rsi_long_threshold": 22,     # [20, 45] — RSI < this for long entry
    "rsi_short_threshold": 78,    # [55, 80] — RSI > this for short entry
    "sl_atr_multiplier": 2.5,     # [1.0, 3.0]
    "max_positions": 10,          # [3, 12]
    "min_confidence": 0.4,        # [0.3, 0.8]
    "min_rr_ratio": 2.0,          # [0.5, 2.5]
}

# ═══════════════════════════════════════════════════════════════════
# FACTOR MODEL STRATEGY
# ═══════════════════════════════════════════════════════════════════
FACTOR_MODEL_PARAMS = {
    "value_weight": 0.30,         # [0.1, 0.6] — sum of 3 weights = 1.0
    "quality_weight": 0.35,       # [0.1, 0.6]
    "momentum_weight": 0.35,     # [0.1, 0.6]
    "rebalance_days": 15,         # [5, 63] — rebalance frequency
    "stop_loss_pct": 0.03,        # [0.03, 0.15] — hard stop %
    "take_profit_pct": 0.30,      # [0.05, 0.30] — target %
    "max_positions": 18,          # [5, 20]
    "min_data_days": 60,          # [60, 252] — FIXED: was 252 (too high, caused 0 trades)
    "min_composite": 0.30,        # [0.3, 0.7] — min composite score for entry
}

# ═══════════════════════════════════════════════════════════════════
# FX CARRY STRATEGY
# ═══════════════════════════════════════════════════════════════════
FX_CARRY_PARAMS = {
    "min_carry_spread": 0.015,    # [0.001, 0.03] — min rate differential
    "carry_weight": 0.50,         # [0.2, 0.9] — carry contribution
    "momentum_weight": 0.50,      # [0.1, 0.8] — momentum contribution
    "trend_filter_sma": 20,       # [20, 100] — SMA for trend filter
    "atr_stop_multiplier": 2.0,   # [1.0, 3.0] — stop = N * ATR
    "min_composite_score": 0.12,  # [0.01, 0.15] — min |composite| for entry
    "max_positions": 6,           # [2, 10]
    "min_confidence": 0.6,        # [0.3, 0.8]
    "min_rr_ratio": 1.5,          # [0.5, 2.0]
    "max_risk_per_pair": 0.015,   # [0.005, 0.04]
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
