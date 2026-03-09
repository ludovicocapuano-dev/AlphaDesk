"""
AlphaDesk - CPCV Runner
Run Combinatorial Purged Cross-Validation on any strategy.

Usage:
    python -m backtester.run_cpcv --strategy mean_reversion --start 2023-01-01 --end 2025-12-31
    python -m backtester.run_cpcv --strategy momentum --groups 6 --embargo 5
    python -m backtester.run_cpcv --strategy all
"""

import argparse
import logging
import os
import sys

import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.cpcv import CombinatorialPurgedCV, CPCVResult
from backtester.engine import BacktestConfig
from backtester.run_backtest import (
    EQUITY_SYMBOLS,
    FX_SYMBOLS,
    fetch_historical_data,
)
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.factor_model import FactorModelStrategy
from strategies.fx_carry import FXCarryStrategy

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("alphadesk.backtest.cpcv_runner")

STRATEGIES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "factor_model": FactorModelStrategy,
    "fx_carry": FXCarryStrategy,
}


def run_cpcv(
    strategy_name: str,
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    capital: float = 100_000,
    n_groups: int = 6,
    k_test: int = None,
    embargo_days: int = 5,
) -> CPCVResult:
    """
    Run CPCV for a single strategy.

    Args:
        strategy_name: One of momentum, mean_reversion, factor_model, fx_carry
        start: Start date string
        end: End date string
        capital: Initial capital
        n_groups: Number of groups for CPCV
        k_test: Number of test groups (default: n_groups // 2)
        embargo_days: Purge/embargo window in days

    Returns:
        CPCVResult with all metrics
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}")

    strategy = STRATEGIES[strategy_name]()

    # Choose symbols
    if strategy_name == "fx_carry":
        symbols = FX_SYMBOLS
    else:
        symbols = EQUITY_SYMBOLS

    # Fetch data
    logger.info(f"Fetching data for {len(symbols)} instruments ({start} to {end})...")
    price_data = fetch_historical_data(symbols, start, end)

    if not price_data:
        raise RuntimeError("No historical data available")

    logger.info(f"Loaded {len(price_data)} instruments")

    # Configure
    config = BacktestConfig(
        initial_capital=capital,
        start_date=start,
        end_date=end,
    )

    # Run CPCV
    cpcv = CombinatorialPurgedCV(
        n_groups=n_groups,
        k_test=k_test,
        embargo_days=embargo_days,
        config=config,
    )

    result = cpcv.run(strategy, price_data)
    return result


def main():
    parser = argparse.ArgumentParser(description="AlphaDesk CPCV Runner")
    parser.add_argument("--strategy", type=str, default="mean_reversion",
                        help="Strategy: momentum, mean_reversion, factor_model, fx_carry, all")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--groups", type=int, default=6, help="Number of CPCV groups (N)")
    parser.add_argument("--k-test", type=int, default=None,
                        help="Number of test groups (default: N/2)")
    parser.add_argument("--embargo", type=int, default=5, help="Embargo days")

    args = parser.parse_args()

    if args.strategy == "all":
        all_results = {}
        for name in STRATEGIES:
            print(f"\n{'=' * 70}")
            print(f"  CPCV: {name}")
            print(f"{'=' * 70}")
            try:
                result = run_cpcv(
                    name,
                    start=args.start,
                    end=args.end,
                    capital=args.capital,
                    n_groups=args.groups,
                    k_test=args.k_test,
                    embargo_days=args.embargo,
                )
                all_results[name] = result
                print(result.summary())
            except Exception as e:
                logger.error(f"CPCV failed for {name}: {e}")

        # Comparison table
        if all_results:
            print("\n" + "=" * 80)
            print("  CPCV STRATEGY COMPARISON")
            print("=" * 80)
            print(f"{'Strategy':<20} {'Mean Sharpe':>12} {'Median Sharpe':>14} "
                  f"{'PBO':>8} {'Pct>0':>8} {'Folds':>8}")
            print("-" * 80)
            for name, r in all_results.items():
                valid = r.oos_sharpes[np.isfinite(r.oos_sharpes)]
                if len(valid) > 0:
                    print(f"{name:<20} {np.mean(valid):>+12.3f} {np.median(valid):>+14.3f} "
                          f"{r.pbo:>7.1%} {np.mean(valid > 0):>7.1%} "
                          f"{len(r.fold_results):>8d}")
                else:
                    print(f"{name:<20} {'N/A':>12} {'N/A':>14} {'N/A':>8} {'N/A':>8} "
                          f"{len(r.fold_results):>8d}")
    else:
        result = run_cpcv(
            args.strategy,
            start=args.start,
            end=args.end,
            capital=args.capital,
            n_groups=args.groups,
            k_test=args.k_test,
            embargo_days=args.embargo,
        )
        print(result.summary())

        # Print per-fold detail
        print("\nPER-FOLD DETAIL:")
        print(f"{'Fold':>6} {'Test Groups':<20} {'Sharpe':>10} {'Return':>10} {'Trades':>8}")
        print("-" * 60)
        for f in result.fold_results:
            print(f"{f.fold_id + 1:>6} {str(f.test_groups):<20} "
                  f"{f.oos_sharpe:>+10.3f} {f.oos_return:>+10.2%} {f.num_trades:>8d}")


if __name__ == "__main__":
    main()
