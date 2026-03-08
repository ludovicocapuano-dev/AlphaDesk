"""
AutoResearch — Market Data Preparation (FIXED, do not modify)
Downloads OHLCV data, computes indicators, caches on disk.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_engine import DataEngine

logger = logging.getLogger("autoresearch.prepare")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

EQUITY_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "JNJ", "WMT", "PG", "XOM", "UNH", "HD",
    "BAC", "DIS", "KO", "PFE", "NFLX",
    "GDX", "GLD", "SLV",  # Gold/Silver miners & ETFs
]

FX_SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
    "USDCHF=X", "EURGBP=X",
]

FX_META = {
    "EURUSD=X": {"base": "EUR", "quote": "USD"},
    "GBPUSD=X": {"base": "GBP", "quote": "USD"},
    "USDJPY=X": {"base": "USD", "quote": "JPY"},
    "AUDUSD=X": {"base": "AUD", "quote": "USD"},
    "USDCHF=X": {"base": "USD", "quote": "CHF"},
    "EURGBP=X": {"base": "EUR", "quote": "GBP"},
}

METRIC_WEIGHTS = {
    "sharpe_ratio": 0.40,
    "total_return": 0.25,
    "max_drawdown": -0.20,
    "trade_activity": 0.15,
}


def _fetch_and_cache(symbols, start, end, prefix="equity"):
    """Fetch data from yfinance with parquet caching."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    price_data = {}

    for symbol in symbols:
        cache_file = os.path.join(CACHE_DIR, f"{prefix}_{symbol.replace('=', '_')}_{start}_{end}.parquet")

        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            logger.info(f"Cached: {symbol} ({len(df)} bars)")
        else:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end)
                if df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                df.columns = [c.lower() for c in df.columns]
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df = DataEngine.compute_indicators(df)
                df.to_parquet(cache_file)
                logger.info(f"Downloaded: {symbol} ({len(df)} bars)")
            except Exception as e:
                logger.error(f"Failed {symbol}: {e}")
                continue

        price_data[symbol] = df

    return price_data


def load_equity_data(start="2023-01-01", end="2025-12-31"):
    return _fetch_and_cache(EQUITY_SYMBOLS, start, end, "equity")


def load_fx_data(start="2023-01-01", end="2025-12-31"):
    return _fetch_and_cache(FX_SYMBOLS, start, end, "fx")


def get_universe(strategy_name, start="2023-01-01", end="2025-12-31"):
    if strategy_name == "fx_carry":
        return load_fx_data(start, end)
    return load_equity_data(start, end)


def compute_score(result):
    """Compute composite score from backtest result. Higher is better."""
    if result.num_trades < 10:
        return 0.0

    sharpe = max(-3, min(3, result.sharpe_ratio))
    sharpe_norm = (sharpe + 3) / 6  # -3..3 -> 0..1

    ret = max(-0.5, min(0.5, result.total_return))
    ret_norm = (ret + 0.5)  # -0.5..0.5 -> 0..1

    dd = min(1.0, result.max_drawdown)
    dd_norm = 1 - dd  # 0% dd -> 1, 100% dd -> 0

    trade_norm = min(1.0, result.num_trades / 100)

    score = (
        0.40 * sharpe_norm +
        0.25 * ret_norm +
        0.20 * dd_norm +
        0.15 * trade_norm
    )
    return round(score, 4)


def clear_cache():
    """Remove all cached data files."""
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)
        print("Cache cleared.")
