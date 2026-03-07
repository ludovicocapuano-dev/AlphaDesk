"""
AlphaDesk — Synthetic Backtest Runner
Generates realistic synthetic market data when yfinance is unavailable.
Uses geometric Brownian motion with regime switching.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.engine import BacktestConfig, BacktestEngine
from backtester.charts import generate_backtest_report_html
from core.data_engine import DataEngine
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.factor_model import FactorModelStrategy
from strategies.fx_carry import FXCarryStrategy

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("alphadesk.backtest.synthetic")

np.random.seed(42)


def generate_synthetic_equity(symbol: str, start: str, end: str,
                               annual_return: float = 0.10,
                               annual_vol: float = 0.25,
                               initial_price: float = 100) -> pd.DataFrame:
    """
    Generate synthetic equity price data using Geometric Brownian Motion
    with mean-reverting volatility (simple regime switching).
    """
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    dt = 1 / 252
    mu = annual_return
    sigma_base = annual_vol

    prices = [initial_price]
    volumes = []

    # Regime: 0=normal, 1=high_vol, 2=trending_up, 3=correction
    regime = 0
    regime_duration = 0

    for i in range(1, n):
        # Regime switching
        regime_duration += 1
        if regime_duration > np.random.randint(20, 60):
            regime = np.random.choice([0, 1, 2, 3], p=[0.45, 0.20, 0.20, 0.15])
            regime_duration = 0

        if regime == 0:  # Normal
            sigma = sigma_base
            drift = mu
        elif regime == 1:  # High volatility
            sigma = sigma_base * 1.8
            drift = mu * 0.5
        elif regime == 2:  # Trending up
            sigma = sigma_base * 0.8
            drift = mu * 2.5
        else:  # Correction
            sigma = sigma_base * 1.5
            drift = -0.15

        # GBM step
        z = np.random.standard_normal()
        price = prices[-1] * np.exp((drift - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
        prices.append(max(price, 1.0))

        # Volume (log-normal, higher during regime changes)
        base_vol = 1_000_000 * (initial_price / 100)
        vol_mult = 1.5 if regime in [1, 3] else 1.0
        volumes.append(int(np.random.lognormal(np.log(base_vol * vol_mult), 0.3)))

    volumes.insert(0, int(np.random.lognormal(np.log(1_000_000), 0.3)))

    # Build OHLCV
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        intraday_range = close * np.random.uniform(0.005, 0.025)
        high = close + intraday_range * np.random.uniform(0, 1)
        low = close - intraday_range * np.random.uniform(0, 1)
        open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
        data.append({
            "date": date,
            "open": round(open_price, 4),
            "high": round(high, 4),
            "low": round(max(low, 0.5), 4),
            "close": round(close, 4),
            "volume": volumes[i],
        })

    df = pd.DataFrame(data)
    df.set_index("date", inplace=True)
    return df


def generate_synthetic_fx(pair: str, start: str, end: str,
                           initial_rate: float = 1.10,
                           annual_vol: float = 0.08) -> pd.DataFrame:
    """Generate synthetic FX pair data (lower vol than equities)."""
    return generate_synthetic_equity(
        pair, start, end,
        annual_return=0.02,  # FX pairs have low drift
        annual_vol=annual_vol,
        initial_price=initial_rate,
    )


def run_all_backtests():
    """Run backtests on all 4 strategies with synthetic data."""
    start = "2023-01-01"
    end = "2025-12-31"
    capital = 100_000

    config = BacktestConfig(
        initial_capital=capital,
        start_date=start,
        end_date=end,
        commission_pct=0.001,
        slippage_pct=0.0005,
    )

    os.makedirs("reports", exist_ok=True)

    # ── Generate synthetic equity universe ──
    logger.info("Generating synthetic equity data...")
    equity_params = {
        "AAPL":  {"ret": 0.15, "vol": 0.28, "price": 170},
        "MSFT":  {"ret": 0.12, "vol": 0.24, "price": 340},
        "GOOGL": {"ret": 0.10, "vol": 0.26, "price": 135},
        "AMZN":  {"ret": 0.14, "vol": 0.30, "price": 145},
        "NVDA":  {"ret": 0.25, "vol": 0.45, "price": 450},
        "META":  {"ret": 0.18, "vol": 0.35, "price": 350},
        "TSLA":  {"ret": 0.08, "vol": 0.55, "price": 240},
        "JPM":   {"ret": 0.10, "vol": 0.22, "price": 170},
        "V":     {"ret": 0.11, "vol": 0.20, "price": 270},
        "JNJ":   {"ret": 0.05, "vol": 0.16, "price": 160},
        "WMT":   {"ret": 0.08, "vol": 0.18, "price": 165},
        "PG":    {"ret": 0.06, "vol": 0.15, "price": 155},
        "XOM":   {"ret": 0.09, "vol": 0.25, "price": 110},
        "UNH":   {"ret": 0.13, "vol": 0.22, "price": 520},
        "HD":    {"ret": 0.10, "vol": 0.23, "price": 340},
        "BAC":   {"ret": 0.12, "vol": 0.28, "price": 35},
        "DIS":   {"ret": 0.04, "vol": 0.30, "price": 90},
        "KO":    {"ret": 0.07, "vol": 0.14, "price": 60},
        "PFE":   {"ret": -0.05, "vol": 0.28, "price": 28},
        "NFLX":  {"ret": 0.20, "vol": 0.35, "price": 480},
    }

    equity_data = {}
    for symbol, params in equity_params.items():
        df = generate_synthetic_equity(
            symbol, start, end,
            annual_return=params["ret"],
            annual_vol=params["vol"],
            initial_price=params["price"],
        )
        df = DataEngine.compute_indicators(df)
        equity_data[symbol] = df
        logger.info(f"  {symbol}: {len(df)} bars, ${params['price']:.0f} → ${df.iloc[-1]['close']:.2f}")

    # ── Generate synthetic FX data ──
    logger.info("Generating synthetic FX data...")
    fx_params = {
        "EURUSD": {"rate": 1.085, "vol": 0.07},
        "GBPUSD": {"rate": 1.265, "vol": 0.08},
        "USDJPY": {"rate": 149.5, "vol": 0.09},
        "AUDUSD": {"rate": 0.655, "vol": 0.10},
        "USDCHF": {"rate": 0.885, "vol": 0.07},
        "EURGBP": {"rate": 0.858, "vol": 0.06},
    }

    fx_data = {}
    for pair, params in fx_params.items():
        df = generate_synthetic_fx(pair, start, end, params["rate"], params["vol"])
        df = DataEngine.compute_indicators(df)
        fx_data[pair] = df
        logger.info(f"  {pair}: {len(df)} bars, {params['rate']:.3f} → {df.iloc[-1]['close']:.4f}")

    # ── Run backtests ──
    strategies = {
        "momentum": (MomentumStrategy(0.30), equity_data),
        "mean_reversion": (MeanReversionStrategy(0.20), equity_data),
        "factor_model": (FactorModelStrategy(0.20), equity_data),
        "fx_carry": (FXCarryStrategy(0.30), fx_data),
    }

    results = {}
    for name, (strategy, data) in strategies.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"BACKTESTING: {name.upper()}")
        logger.info(f"{'='*60}")

        try:
            engine = BacktestEngine(config)
            result = engine.run(strategy, data)
            results[name] = result

            # Print summary
            print(result.summary())

            # Generate HTML report
            report_path = f"reports/backtest_{name}.html"
            generate_backtest_report_html(result, report_path)
            logger.info(f"Report saved: {report_path}")

        except Exception as e:
            logger.error(f"Backtest failed for {name}: {e}", exc_info=True)

    # ── Comparison Table ──
    if results:
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON — SYNTHETIC DATA")
        print("=" * 80)
        print(f"{'Strategy':<20} {'Return':>10} {'Annual':>10} {'Sharpe':>8} {'Sortino':>8} "
              f"{'MaxDD':>8} {'WinRate':>8} {'PF':>6} {'Trades':>8}")
        print("-" * 80)

        for name, r in results.items():
            years = len(r.daily_returns) / 252
            annual = ((1 + r.total_return) ** (1/years) - 1) if years > 0 else 0
            print(f"{name:<20} {r.total_return:>+9.2%} {annual:>+9.2%} {r.sharpe_ratio:>8.2f} "
                  f"{r.sortino_ratio:>8.2f} {r.max_drawdown:>7.2%} {r.win_rate:>7.1%} "
                  f"{r.profit_factor:>6.2f} {r.num_trades:>8d}")

        print("\n⚠️  Note: Risultati su dati sintetici (GBM con regime switching).")
        print("    I risultati reali dipenderanno dai dati di mercato effettivi.")
        print("    Esegui il backtest sul tuo VPS con yfinance per dati reali.\n")

    return results


if __name__ == "__main__":
    results = run_all_backtests()
