"""
AlphaDesk — FX Carry Strategy Optimization v2
Second-pass: fine-tune around best region found in v1,
adding take_profit_multiplier and max_risk_per_pair.
"""

import itertools
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.engine import BacktestConfig, BacktestEngine
from core.data_engine import DataEngine
from strategies.fx_carry import FXCarryStrategy
from strategies.base_strategy import Signal, TradeSignal

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("alphadesk.optimize.fx_carry.v2")
logger.setLevel(logging.INFO)

FX_SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCHF=X", "EURGBP=X"]
_FX_RENAME = {s: s.replace("=X", "") for s in FX_SYMBOLS}

# Focused grid around best region + new dimensions
PARAM_GRID = {
    "min_carry_spread": [0.005, 0.010, 0.015, 0.020],
    "momentum_weight": [0.40, 0.50, 0.60, 0.70],
    "trend_filter_sma": [20, 50],
    "atr_stop_multiplier": [2.0, 2.5, 3.0],
    "composite_threshold": [0.02, 0.03, 0.05, 0.08, 0.10],
    "tp_multiplier": [1.5, 2.0, 2.5, 3.0, 4.0],  # take_profit = tp_mult * atr_stop_mult * atr
    "max_risk_per_pair": [0.010, 0.015, 0.020, 0.025],
}

MAX_COMBOS = 500


def fetch_fx_data() -> dict:
    cache_path = os.path.join(os.path.dirname(__file__), "cache", "fx_data_cache.pkl")
    if os.path.exists(cache_path):
        logger.info("Loading cached FX data...")
        return pd.read_pickle(cache_path)

    import yfinance as yf
    price_data = {}
    for symbol in FX_SYMBOLS:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start="2023-01-01", end="2025-12-31")
        if df.empty:
            continue
        df.columns = [c.lower() for c in df.columns]
        df = DataEngine.compute_indicators(df)
        price_data[_FX_RENAME[symbol]] = df
    pd.to_pickle(price_data, cache_path)
    return price_data


def run_single_backtest(price_data: dict, params: dict) -> dict:
    strategy = FXCarryStrategy()
    strategy.min_carry_spread = params["min_carry_spread"]
    strategy.momentum_weight = params["momentum_weight"]
    strategy.carry_weight = 1.0 - params["momentum_weight"]
    strategy.trend_filter_sma = params["trend_filter_sma"]
    strategy.atr_stop_multiplier = params["atr_stop_multiplier"]
    strategy.max_risk_per_pair = params["max_risk_per_pair"]

    composite_threshold = params["composite_threshold"]
    tp_multiplier = params["tp_multiplier"]

    def patched_generate(symbol, instrument_id, df):
        from config.instruments import FX_PAIRS
        from strategies.fx_carry import get_rates_for_date
        meta = FX_PAIRS.get(symbol)
        if meta is None or len(df) < 60:
            return None
        if instrument_id == 0:
            instrument_id = meta.get("etoro_id", 0)

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
            take_profit = entry + (tp_multiplier * strategy.atr_stop_multiplier * atr)
        else:
            direction = Signal.SELL if abs(composite) < 0.15 else Signal.STRONG_SELL
            stop_loss = entry + (strategy.atr_stop_multiplier * atr)
            take_profit = entry - (tp_multiplier * strategy.atr_stop_multiplier * atr)

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

    sma_col = f"sma_{params['trend_filter_sma']}"
    patched_data = {}
    for symbol, df in price_data.items():
        if sma_col not in df.columns:
            df = df.copy()
            df[sma_col] = df["close"].rolling(params["trend_filter_sma"]).mean()
        patched_data[symbol] = df

    config = BacktestConfig(initial_capital=100_000, start_date="2023-01-01", end_date="2025-12-31")
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
        return {"total_return": -1.0, "sharpe": -99.0, "sortino": -99.0,
                "max_drawdown": 1.0, "num_trades": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "avg_holding_days": 0.0}


def main():
    print("=" * 70)
    print("FX CARRY STRATEGY — OPTIMIZATION v2 (expanded grid)")
    print("=" * 70)

    price_data = fetch_fx_data()
    if not price_data:
        print("ERROR: No FX data. Exiting.")
        return

    # Generate combos
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    full_grid = list(itertools.product(*values))
    total = len(full_grid)
    print(f"Full grid: {total} combinations")

    random.seed(42)
    if total > MAX_COMBOS:
        combos = random.sample(full_grid, MAX_COMBOS)
    else:
        combos = full_grid
    combos = [dict(zip(keys, c)) for c in combos]
    print(f"Testing {len(combos)} combinations...\n")

    results = []
    t0 = time.time()

    for i, params in enumerate(combos):
        metrics = run_single_backtest(price_data, params)
        results.append({**params, **metrics})

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(combos) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(combos)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s | "
                  f"ret={metrics['total_return']:+.2%}, sharpe={metrics['sharpe']:.2f}, "
                  f"trades={metrics['num_trades']}")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(combos)} backtests in {elapsed:.1f}s")

    df = pd.DataFrame(results)
    df_valid = df[df["num_trades"] >= 5].sort_values("sharpe", ascending=False)

    if df_valid.empty:
        print("No valid results with >= 5 trades")
        df_valid = df[df["num_trades"] > 0].sort_values("sharpe", ascending=False)

    print("\n" + "=" * 70)
    print("TOP 10 PARAMETER SETS (by Sharpe)")
    print("=" * 70)

    for rank, (_, row) in enumerate(df_valid.head(10).iterrows(), 1):
        print(f"\n  #{rank}: Sharpe={row['sharpe']:.2f}, Return={row['total_return']:+.2%}, "
              f"MaxDD={row['max_drawdown']:.2%}, Trades={int(row['num_trades'])}, "
              f"WR={row['win_rate']:.1%}, PF={row['profit_factor']:.2f}")
        print(f"    spread={row['min_carry_spread']}, mom_w={row['momentum_weight']:.2f}, "
              f"sma={int(row['trend_filter_sma'])}, atr_stop={row['atr_stop_multiplier']}, "
              f"threshold={row['composite_threshold']}, tp_mult={row['tp_multiplier']}, "
              f"risk={row['max_risk_per_pair']}")

    best = df_valid.iloc[0]

    # Save
    results_path = os.path.join(os.path.dirname(__file__), "results", "fx_carry_optimization_v2.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df.sort_values("sharpe", ascending=False).to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Return best params for applying
    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    for k in PARAM_GRID.keys():
        print(f"  {k} = {best[k]}")
    print(f"  carry_weight = {1 - best['momentum_weight']:.2f}")
    print(f"  Return={best['total_return']:+.2%}, Sharpe={best['sharpe']:.2f}, "
          f"MaxDD={best['max_drawdown']:.2%}, Trades={int(best['num_trades'])}")

    return best


if __name__ == "__main__":
    main()
