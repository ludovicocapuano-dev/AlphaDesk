"""
AutoResearch — Backtest Runner (FIXED, do not modify)
Runs a single backtest experiment with params from strategy_tuner.py.

Usage:
    python autoresearch/backtest_runner.py --strategy momentum --experiment-id exp_001
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.engine import BacktestConfig, BacktestEngine
from core.data_engine import DataEngine
from strategies.base_strategy import Signal, TradeSignal
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.factor_model import FactorModelStrategy
from strategies.fx_carry import FXCarryStrategy

from autoresearch.prepare_market import get_universe, compute_score, FX_META
from autoresearch.strategy_tuner import get_params, apply_params, BACKTEST_CONFIG

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("autoresearch.runner")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ══════════════════════════════════════════════════════════════════
# FIX: Add generate_signal_sync to Factor Model and FX Carry
# The backtester calls generate_signal_sync but only Momentum and
# MeanReversion have _evaluate methods. We patch the others here.
# ══════════════════════════════════════════════════════════════════

def _factor_model_signal_sync(self, symbol, instrument_id, df):
    """Sync signal generation for Factor Model in backtests."""
    params = get_params("factor_model")
    min_days = params.get("min_data_days", 252)

    if len(df) < min_days:
        return None

    # Get fundamentals from cache
    fundamentals = _get_cached_fundamentals(symbol)

    scores = self._compute_composite_score(df, fundamentals)
    if scores is None:
        return None

    composite = scores["composite"]
    min_composite = params.get("min_composite", 0.5)

    if composite < min_composite:
        return None

    price = df.iloc[-1]["close"]
    atr = df.iloc[-1].get("atr", price * 0.02)
    stop_pct = params.get("stop_loss_pct", 0.08)
    tp_pct = params.get("take_profit_pct", 0.15)

    return TradeSignal(
        symbol=symbol,
        instrument_id=instrument_id,
        signal=Signal.STRONG_BUY if composite > 0.7 else Signal.BUY,
        strategy_name=self.name,
        confidence=min(0.95, composite),
        entry_price=price,
        stop_loss=price * (1 - stop_pct),
        take_profit=price * (1 + tp_pct),
        suggested_size_pct=0.05,
        metadata={"factor_scores": scores},
    )


def _fx_carry_signal_sync(self, symbol, instrument_id, df):
    """Sync signal generation for FX Carry in backtests."""
    meta = FX_META.get(symbol)
    if meta is None:
        return None

    if len(df) < 60:
        return None

    score_data = self._score_pair(symbol, meta, df)
    if score_data is None:
        return None

    composite = score_data["composite_score"]
    params = get_params("fx_carry")
    min_score = params.get("min_composite_score", 0.05)

    if abs(composite) < min_score:
        return None

    latest = df.iloc[-1]
    entry = latest["close"]
    atr = latest.get("atr", entry * 0.005)

    atr_mult = params.get("atr_stop_multiplier", 1.5)

    if composite > 0:
        direction = Signal.BUY
        stop_loss = entry - (atr_mult * atr)
        take_profit = entry + (2.5 * atr_mult * atr)
    else:
        direction = Signal.SELL
        stop_loss = entry + (atr_mult * atr)
        take_profit = entry - (2.5 * atr_mult * atr)

    confidence = min(0.90, 0.4 + abs(composite) * 0.3)
    if score_data.get("trend_aligned", False):
        confidence = min(0.95, confidence + 0.15)

    return TradeSignal(
        symbol=symbol,
        instrument_id=instrument_id,
        signal=direction,
        strategy_name=self.name,
        confidence=confidence,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        suggested_size_pct=params.get("max_risk_per_pair", 0.015),
        metadata={"composite_score": composite},
    )


# Cache fundamentals to avoid repeated yfinance calls
_fundamentals_cache = {}


def _get_cached_fundamentals(symbol):
    """Fetch and cache fundamentals from yfinance."""
    if symbol in _fundamentals_cache:
        return _fundamentals_cache[symbol]

    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        data = {
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "debt_equity": info.get("debtToEquity"),
            "profit_margin": info.get("profitMargins"),
        }
    except Exception:
        data = {}

    _fundamentals_cache[symbol] = data
    return data


def patch_strategies():
    """Monkey-patch strategies with sync signal methods for backtesting."""
    FactorModelStrategy.generate_signal_sync = _factor_model_signal_sync
    FXCarryStrategy.generate_signal_sync = _fx_carry_signal_sync


def create_strategy(strategy_name):
    """Create strategy instance and apply tuner params."""
    strategies = {
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "factor_model": FactorModelStrategy,
        "fx_carry": FXCarryStrategy,
    }
    strategy = strategies[strategy_name]()
    params = get_params(strategy_name)
    apply_params(strategy, params)
    return strategy


def run_experiment(strategy_name, experiment_id, timeout=300,
                   start_date=None, end_date=None):
    """Run a single backtest experiment.

    Args:
        start_date/end_date: Override BACKTEST_CONFIG dates (for OOS validation).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    patch_strategies()

    start_time = time.time()
    logger.info(f"Experiment {experiment_id}: {strategy_name}")

    # Load params
    params = get_params(strategy_name)
    logger.info(f"Params: {json.dumps(params, indent=2)}")

    # Load market data
    logger.info("Loading market data...")
    bc = BACKTEST_CONFIG
    sd = start_date or bc["start_date"]
    ed = end_date or bc["end_date"]
    price_data = get_universe(strategy_name, sd, ed)

    if not price_data:
        logger.error("No data loaded!")
        return None

    logger.info(f"Loaded {len(price_data)} instruments")

    # Create strategy
    strategy = create_strategy(strategy_name)

    # Create backtester
    config = BacktestConfig(
        initial_capital=bc["initial_capital"],
        commission_pct=bc["commission_pct"],
        slippage_pct=bc["slippage_pct"],
        max_positions=bc["max_positions"],
        risk_per_trade=bc["risk_per_trade"],
        start_date=sd,
        end_date=ed,
    )

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run(strategy, price_data)
    elapsed = time.time() - start_time

    # Compute score
    score = compute_score(result)

    # Print summary
    print(result.summary())
    print(f"SCORE: {score:.4f}")
    print(f"Time: {elapsed:.1f}s")

    # Save results
    result_data = {
        "experiment_id": experiment_id,
        "strategy": strategy_name,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "params": params,
        "metrics": {
            "score": score,
            "sharpe_ratio": round(result.sharpe_ratio, 4),
            "total_return": round(result.total_return, 4),
            "total_pnl": round(result.total_pnl, 2),
            "max_drawdown": round(result.max_drawdown, 4),
            "num_trades": result.num_trades,
            "win_rate": round(result.win_rate, 4),
            "profit_factor": round(result.profit_factor, 4) if result.profit_factor != float('inf') else 999,
            "avg_holding_days": round(result.avg_holding_days, 1),
        },
    }

    result_path = os.path.join(RESULTS_DIR, f"{experiment_id}.json")
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)

    logger.info(f"Results saved: {result_path}")
    return result_data


def compare_results():
    """Print comparison of all experiment results."""
    results = []
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, f)) as fp:
                results.append(json.load(fp))

    if not results:
        print("No results found.")
        return

    print(f"\n{'='*90}")
    print(f"{'ID':<20} {'Strategy':<15} {'Score':>8} {'Sharpe':>8} {'Return':>9} {'MaxDD':>8} {'Trades':>7}")
    print(f"{'='*90}")

    for r in sorted(results, key=lambda x: x["metrics"]["score"], reverse=True):
        m = r["metrics"]
        print(f"{r['experiment_id']:<20} {r['strategy']:<15} {m['score']:>8.4f} "
              f"{m['sharpe_ratio']:>8.2f} {m['total_return']:>+8.2%} "
              f"{m['max_drawdown']:>7.2%} {m['num_trades']:>7d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoResearch Backtest Runner")
    parser.add_argument("--strategy", required=True, help="momentum|mean_reversion|factor_model|fx_carry")
    parser.add_argument("--experiment-id", required=True, help="Unique experiment ID")
    parser.add_argument("--timeout", type=int, default=300, help="Max seconds")
    parser.add_argument("--compare", action="store_true", help="Compare all results")

    args = parser.parse_args()

    if args.compare:
        compare_results()
    else:
        run_experiment(args.strategy, args.experiment_id, args.timeout)
