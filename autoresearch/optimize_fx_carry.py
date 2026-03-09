"""
AlphaDesk — FX Carry Strategy Parameter Optimization
Grid search over strategy parameters to maximize Sharpe ratio.
Runs ~400 random-sampled combinations from the full grid.
"""

import itertools
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.engine import BacktestConfig, BacktestEngine
from core.data_engine import DataEngine
from strategies.fx_carry import FXCarryStrategy

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("alphadesk.optimize.fx_carry")
logger.setLevel(logging.INFO)

# ── FX symbols (yfinance format) ──
FX_SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
    "USDCHF=X", "EURGBP=X",
]
_FX_RENAME = {s: s.replace("=X", "") for s in FX_SYMBOLS}

# ── Parameter grid ──
PARAM_GRID = {
    "min_carry_spread": [0.005, 0.01, 0.015, 0.02],
    "momentum_weight": [0.20, 0.30, 0.40, 0.50],
    "trend_filter_sma": [20, 50, 100],
    "atr_stop_multiplier": [1.0, 1.5, 2.0, 2.5],
    "composite_threshold": [0.03, 0.05, 0.08, 0.10],
}

MAX_COMBOS = 400  # Random sample if full grid exceeds this


def fetch_fx_data(start: str = "2023-01-01", end: str = "2025-12-31") -> dict:
    """Fetch FX historical data from yfinance (cached)."""
    cache_path = os.path.join(os.path.dirname(__file__), "cache", "fx_data_cache.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        logger.info("Loading cached FX data...")
        price_data = pd.read_pickle(cache_path)
        if price_data:
            return price_data

    import yfinance as yf

    price_data = {}
    for symbol in FX_SYMBOLS:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)
            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue
            df.columns = [c.lower() for c in df.columns]
            df = DataEngine.compute_indicators(df)
            key = _FX_RENAME[symbol]
            price_data[key] = df
            logger.info(f"Loaded {symbol} -> {key}: {len(df)} bars")
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")

    # Cache to disk
    pd.to_pickle(price_data, cache_path)
    logger.info(f"Cached FX data to {cache_path}")
    return price_data


def run_single_backtest(price_data: dict, params: dict) -> dict:
    """Run a single backtest with given parameter set. Returns metrics."""
    strategy = FXCarryStrategy()

    # Apply parameters
    strategy.min_carry_spread = params["min_carry_spread"]
    strategy.momentum_weight = params["momentum_weight"]
    strategy.carry_weight = 1.0 - params["momentum_weight"]  # They must sum to 1
    strategy.trend_filter_sma = params["trend_filter_sma"]
    strategy.atr_stop_multiplier = params["atr_stop_multiplier"]

    # Monkey-patch the composite threshold in generate_signal_sync
    composite_threshold = params["composite_threshold"]
    original_generate = strategy.generate_signal_sync

    def patched_generate(symbol, instrument_id, df):
        from config.instruments import FX_PAIRS
        meta = FX_PAIRS.get(symbol)
        if meta is None:
            return None
        if instrument_id == 0:
            instrument_id = meta.get("etoro_id", 0)
        if len(df) < 60:
            return None

        # Use historical rates for the current backtest date
        from strategies.fx_carry import get_rates_for_date, Signal
        from strategies.base_strategy import TradeSignal
        date_str = str(df.index[-1])[:10]
        rates = get_rates_for_date(date_str)
        score_data = strategy._score_pair(symbol, meta, df, rates=rates)
        if not score_data:
            return None

        composite = score_data["composite_score"]
        if abs(composite) < composite_threshold:
            return None

        latest = df.iloc[-1]
        entry = latest["close"]
        atr = latest.get("atr", entry * 0.01)
        if pd.isna(atr) or atr == 0:
            atr = entry * 0.01

        if composite > 0:
            direction = Signal.BUY if composite < 0.15 else Signal.STRONG_BUY
            stop_loss = entry - (strategy.atr_stop_multiplier * atr)
            take_profit = entry + (2.5 * strategy.atr_stop_multiplier * atr)
        else:
            direction = Signal.SELL if abs(composite) < 0.15 else Signal.STRONG_SELL
            stop_loss = entry + (strategy.atr_stop_multiplier * atr)
            take_profit = entry - (2.5 * strategy.atr_stop_multiplier * atr)

        confidence = min(0.90, 0.4 + abs(composite) * 0.3)
        if score_data["trend_aligned"]:
            confidence = min(0.95, confidence + 0.15)

        return TradeSignal(
            symbol=symbol,
            instrument_id=instrument_id,
            signal=direction,
            strategy_name=strategy.name,
            confidence=confidence,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            suggested_size_pct=strategy.max_risk_per_pair,
            metadata={
                "carry_differential": score_data["carry_differential"],
                "composite_score": composite,
                "trend_aligned": score_data["trend_aligned"],
            },
        )

    strategy.generate_signal_sync = patched_generate

    # Need to handle the SMA column — if trend_filter_sma is not 20/50/200 we need to add it
    sma_col = f"sma_{params['trend_filter_sma']}"
    patched_data = {}
    for symbol, df in price_data.items():
        if sma_col not in df.columns:
            df = df.copy()
            df[sma_col] = df["close"].rolling(params["trend_filter_sma"]).mean()
        patched_data[symbol] = df

    config = BacktestConfig(
        initial_capital=100_000,
        start_date="2023-01-01",
        end_date="2025-12-31",
    )
    engine = BacktestEngine(config)

    try:
        result = engine.run(strategy, patched_data)
        return {
            "total_return": result.total_return,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "num_trades": result.num_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "avg_holding_days": result.avg_holding_days,
        }
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {
            "total_return": -1.0,
            "sharpe": -99.0,
            "sortino": -99.0,
            "max_drawdown": 1.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_holding_days": 0.0,
        }


def generate_param_combos() -> list:
    """Generate parameter combinations, random-sample if too many."""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    full_grid = list(itertools.product(*values))
    total = len(full_grid)
    logger.info(f"Full grid: {total} combinations")

    if total <= MAX_COMBOS:
        combos = full_grid
    else:
        random.seed(42)
        combos = random.sample(full_grid, MAX_COMBOS)
        logger.info(f"Sampled {MAX_COMBOS} combos from {total}")

    return [dict(zip(keys, c)) for c in combos]


def main():
    print("=" * 70)
    print("FX CARRY STRATEGY — PARAMETER OPTIMIZATION")
    print("=" * 70)

    # 1. Fetch data
    logger.info("Fetching FX data...")
    price_data = fetch_fx_data()
    if not price_data:
        print("ERROR: No FX data available. Exiting.")
        return

    print(f"Loaded {len(price_data)} FX pairs: {list(price_data.keys())}")

    # 2. Generate parameter combos
    combos = generate_param_combos()
    print(f"Testing {len(combos)} parameter combinations...\n")

    # 3. Run grid search
    results = []
    t0 = time.time()

    for i, params in enumerate(combos):
        metrics = run_single_backtest(price_data, params)
        results.append({**params, **metrics})

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(combos) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(combos)}] elapsed={elapsed:.0f}s, ETA={eta:.0f}s | "
                  f"last: ret={metrics['total_return']:+.2%}, sharpe={metrics['sharpe']:.2f}, "
                  f"trades={metrics['num_trades']}")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(combos)} backtests in {elapsed:.1f}s ({elapsed/len(combos):.2f}s each)")

    # 4. Analyze results
    df = pd.DataFrame(results)

    # Filter: require at least 10 trades for statistical significance
    df_valid = df[df["num_trades"] >= 10].copy()

    if df_valid.empty:
        print("\nWARNING: No parameter set produced >= 10 trades.")
        print("Relaxing to >= 3 trades...")
        df_valid = df[df["num_trades"] >= 3].copy()

    if df_valid.empty:
        print("ERROR: No parameter set produced any trades. Check strategy logic.")
        # Show what we got anyway
        df_any = df[df["num_trades"] > 0].sort_values("sharpe", ascending=False)
        if not df_any.empty:
            print("\nBest results (any trades):")
            print(df_any.head(10).to_string(index=False))
        return

    df_valid = df_valid.sort_values("sharpe", ascending=False)

    print("\n" + "=" * 70)
    print("TOP 10 PARAMETER SETS (by Sharpe Ratio)")
    print("=" * 70)

    top10 = df_valid.head(10)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"\n  #{rank}")
        print(f"    min_carry_spread={row['min_carry_spread']:.3f}, "
              f"momentum_weight={row['momentum_weight']:.2f}, "
              f"carry_weight={1-row['momentum_weight']:.2f}")
        print(f"    trend_filter_sma={int(row['trend_filter_sma'])}, "
              f"atr_stop_multiplier={row['atr_stop_multiplier']:.1f}, "
              f"composite_threshold={row['composite_threshold']:.2f}")
        print(f"    Return={row['total_return']:+.2%}, Sharpe={row['sharpe']:.2f}, "
              f"Sortino={row['sortino']:.2f}, MaxDD={row['max_drawdown']:.2%}")
        print(f"    Trades={int(row['num_trades'])}, WinRate={row['win_rate']:.1%}, "
              f"PF={row['profit_factor']:.2f}, AvgHold={row['avg_holding_days']:.0f}d")

    # 5. Best parameters
    best = df_valid.iloc[0]
    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    print(f"  min_carry_spread    = {best['min_carry_spread']}")
    print(f"  momentum_weight     = {best['momentum_weight']}")
    print(f"  carry_weight        = {1 - best['momentum_weight']}")
    print(f"  trend_filter_sma    = {int(best['trend_filter_sma'])}")
    print(f"  atr_stop_multiplier = {best['atr_stop_multiplier']}")
    print(f"  composite_threshold = {best['composite_threshold']}")
    print(f"  ---")
    print(f"  Total Return:  {best['total_return']:+.2%}")
    print(f"  Sharpe Ratio:  {best['sharpe']:.2f}")
    print(f"  Sortino Ratio: {best['sortino']:.2f}")
    print(f"  Max Drawdown:  {best['max_drawdown']:.2%}")
    print(f"  Trades:        {int(best['num_trades'])}")
    print(f"  Win Rate:      {best['win_rate']:.1%}")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")

    # 6. Save full results
    results_path = os.path.join(os.path.dirname(__file__), "results", "fx_carry_optimization.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df.sort_values("sharpe", ascending=False).to_csv(results_path, index=False)
    print(f"\nFull results saved to: {results_path}")

    # 7. Apply best parameters to fx_carry.py
    apply_best_params(best)


def apply_best_params(best):
    """Update fx_carry.py with the optimized parameters."""
    fx_carry_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  "strategies", "fx_carry.py")

    with open(fx_carry_path, "r") as f:
        content = f.read()

    import re

    # Update __init__ parameters
    replacements = [
        (r'self\.min_carry_spread\s*=\s*[\d.]+',
         f"self.min_carry_spread = {best['min_carry_spread']}"),
        (r'self\.momentum_weight\s*=\s*[\d.]+',
         f"self.momentum_weight = {best['momentum_weight']:.2f}"),
        (r'self\.carry_weight\s*=\s*[\d.]+',
         f"self.carry_weight = {1 - best['momentum_weight']:.2f}"),
        (r'self\.trend_filter_sma\s*=\s*\d+',
         f"self.trend_filter_sma = {int(best['trend_filter_sma'])}"),
        (r'self\.atr_stop_multiplier\s*=\s*[\d.]+',
         f"self.atr_stop_multiplier = {best['atr_stop_multiplier']}"),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    # Update composite threshold in generate_signal_sync (the hardcoded 0.05)
    content = re.sub(
        r'if abs\(composite\) < 0\.05:',
        f"if abs(composite) < {best['composite_threshold']}:",
        content,
    )

    # Also update the threshold in _generate_trade_signals
    content = re.sub(
        r'if abs\(composite\) < 0\.05:',
        f"if abs(composite) < {best['composite_threshold']}:",
        content,
    )

    with open(fx_carry_path, "w") as f:
        f.write(content)

    print(f"\nUpdated {fx_carry_path} with optimized parameters.")


if __name__ == "__main__":
    main()
