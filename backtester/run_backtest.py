"""
AlphaDesk — Backtest Runner
Run backtests on historical data and generate reports.

Usage:
    python -m backtester.run_backtest --strategy momentum --start 2023-01-01 --end 2025-12-31
    python -m backtester.run_backtest --strategy all --capital 100000
"""

import argparse
import logging
import sys
import os

import numpy as np
import pandas as pd

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.engine import BacktestConfig, BacktestEngine, BacktestResult
from backtester.charts import generate_backtest_report_html
from core.data_engine import DataEngine
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.factor_model import FactorModelStrategy
from strategies.fx_carry import FXCarryStrategy

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("alphadesk.backtest.runner")


# Symbols to backtest (using yfinance for historical data)
EQUITY_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "JNJ", "WMT", "PG", "XOM", "UNH", "HD",
    "BAC", "DIS", "KO", "PFE", "NFLX",
]

FX_SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
    "USDCHF=X", "EURGBP=X",
]

# yfinance uses "=X" suffix for FX; our FX_PAIRS dict uses plain names
_FX_RENAME = {s: s.replace("=X", "") for s in FX_SYMBOLS}


def fetch_historical_data(symbols: list, start: str, end: str) -> dict:
    """Fetch historical data from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance required: pip install yfinance")
        sys.exit(1)

    price_data = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)
            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue

            df.columns = [c.lower() for c in df.columns]

            # Add technical indicators
            df = DataEngine.compute_indicators(df)
            # Use internal name for FX (strip yfinance "=X" suffix)
            key = _FX_RENAME.get(symbol, symbol)
            price_data[key] = df
            logger.info(f"Loaded {symbol} → {key}: {len(df)} bars")

        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")

    return price_data


def run_strategy_backtest(strategy_name: str, config: BacktestConfig) -> BacktestResult:
    """Run backtest for a specific strategy."""
    strategies = {
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "factor_model": FactorModelStrategy,
        "fx_carry": FXCarryStrategy,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")

    strategy = strategies[strategy_name]()

    # Choose symbols based on strategy
    if strategy_name == "fx_carry":
        symbols = FX_SYMBOLS
    else:
        symbols = EQUITY_SYMBOLS

    # Fetch data
    logger.info(f"Fetching data for {len(symbols)} instruments...")
    price_data = fetch_historical_data(symbols, config.start_date, config.end_date)

    if not price_data:
        raise RuntimeError("No historical data available")

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run(strategy, price_data)

    return result


def run_combined_backtest(config: BacktestConfig) -> dict:
    """Run backtest for all strategies and combine results."""
    results = {}
    all_strategies = ["momentum", "mean_reversion", "factor_model", "fx_carry"]

    for name in all_strategies:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Backtesting: {name}")
            logger.info(f"{'='*50}")
            result = run_strategy_backtest(name, config)
            results[name] = result
            print(result.summary())
        except Exception as e:
            logger.error(f"Backtest failed for {name}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="AlphaDesk Backtester")
    parser.add_argument("--strategy", type=str, default="all",
                        help="Strategy to test: momentum, mean_reversion, factor_model, fx_carry, all")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--output", type=str, default="backtest_report.html", help="Output report path")

    args = parser.parse_args()

    config = BacktestConfig(
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
    )

    os.makedirs("reports", exist_ok=True)

    if args.strategy == "all":
        results = run_combined_backtest(config)

        # Generate report for each strategy
        for name, result in results.items():
            report_path = f"reports/backtest_{name}.html"
            generate_backtest_report_html(result, report_path)
            logger.info(f"Report saved: {report_path}")

        # Summary comparison
        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON")
        print("=" * 70)
        print(f"{'Strategy':<20} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>8}")
        print("-" * 70)
        for name, r in results.items():
            print(f"{name:<20} {r.total_return:>+9.2%} {r.sharpe_ratio:>8.2f} "
                  f"{r.max_drawdown:>7.2%} {r.win_rate:>7.1%} {r.num_trades:>8d}")

    else:
        result = run_strategy_backtest(args.strategy, config)
        print(result.summary())
        report_path = f"reports/backtest_{args.strategy}.html"
        generate_backtest_report_html(result, report_path)
        logger.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
