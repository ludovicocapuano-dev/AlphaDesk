# AutoResearch Plan — AlphaDesk Strategy Optimization

## Objective
Maximize the composite score of AlphaDesk's 4 trading strategies by modifying
ONLY the file `autoresearch/strategy_tuner.py`.

## Metric (DO NOT MODIFY)
Score = 0.40*Sharpe_norm + 0.25*Return_norm + 0.20*(1-MaxDD) + 0.15*TradeActivity
Score = 0 if num_trades < 10. Higher is better. Max theoretical = 1.0.

## How to run an experiment
1. Edit parameters in `autoresearch/strategy_tuner.py`
2. Run: `bash autoresearch/run_experiment.sh <strategy_name> <experiment_id>`
3. Read output and results in `autoresearch/results/<experiment_id>.json`
4. If score improved, keep changes. If not, revert and try something else.

## Priority order

### 1. Factor Model (CRITICAL — 0 trades, BUG)
**Root cause**: `min_data_days=252` is too high. The backtester iterates day-by-day
and early in the period the historical window is < 252 bars, so no signals are ever
generated until very late. Also needs fundamentals cache.

**First experiment**: Set `min_data_days` to 120 and `min_composite` to 0.45.
This should immediately generate trades.

**Then optimize**: value_weight, quality_weight, momentum_weight, stop_loss_pct,
take_profit_pct.

### 2. FX Carry (CRITICAL — 0 trades, BUG)
**Root cause**: The backtester's `_try_evaluate` returned None for FX Carry.
Fixed via monkey-patching in backtest_runner.py.

**First experiment**: Use default params, should now generate trades.
**Then optimize**: min_carry_spread, carry_weight vs momentum_weight,
min_composite_score, atr_stop_multiplier.

### 3. Momentum (Sharpe -0.68, losing money)
**Root cause**: Filters too restrictive in volatile market. Requires
price > SMA50 > SMA200 + breakout + volume surge + RSI < 75 + momentum > 5%.
Very few stocks pass all filters simultaneously.

**Ideas to try**:
- Reduce volume_threshold: 1.5 → 1.2
- Reduce min_momentum_3m: 0.05 → 0.02
- Increase rsi_overbought: 75 → 80
- Reduce breakout_period: 20 → 15
- Widen atr_multiplier: 2.0 → 2.5 (wider stops)

### 4. Mean Reversion (Sharpe 0.89, profitable)
Already working well. Only fine-tune:
- z_entry_long: try -1.8 vs -2.0
- sl_atr_multiplier: try 1.8 vs 1.5
- rsi_long_threshold: try 38 vs 35

## Constraints
- DO NOT modify: prepare_market.py, backtest_runner.py, engine.py, strategies/*
- ONLY modify: autoresearch/strategy_tuner.py
- Budget: max 20 experiments per strategy
- Each experiment takes ~60-120 seconds (data is cached after first run)

## Experiment naming convention
- `baseline_<strategy>` — first run with default params
- `exp_<strategy>_<NNN>` — numbered experiments

## Research strategy
1. First: run baselines for all 4 strategies
2. For broken strategies: fix the obvious bugs first (min_data_days, etc)
3. For underperforming: one-at-a-time sensitivity analysis
4. For profitable: only micro-tuning, don't break what works
5. Log every experiment with parameters and results
