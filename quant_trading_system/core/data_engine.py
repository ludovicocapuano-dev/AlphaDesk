"""
AlphaDesk — Data Engine
Aggregates market data from multiple sources for strategy consumption.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("alphadesk.data")


class DataEngine:
    """Aggregates and normalizes market data from multiple sources."""

    def __init__(self, etoro_client):
        self.etoro = etoro_client
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._fundamentals_cache: Dict[str, dict] = {}
        self._macro_cache: Dict[str, pd.Series] = {}

    # ────────────────────── Price Data ──────────────────────

    async def get_ohlcv(self, instrument_id: int, symbol: str,
                         period: str = "OneDay", count: int = 252) -> pd.DataFrame:
        """Fetch OHLCV data and return as DataFrame."""
        cache_key = f"{symbol}_{period}_{count}"
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        try:
            candles = await self.etoro.get_candles(instrument_id, period, count)
            df = pd.DataFrame(candles)
            # Normalize column names
            col_map = {
                "open": "open", "high": "high", "low": "low",
                "close": "close", "volume": "volume", "timestamp": "date",
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume", "Timestamp": "date",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns},
                       inplace=True)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

            df = df.sort_index()
            self._price_cache[cache_key] = df
            return df

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    def get_historical_yfinance(self, symbol: str, period: str = "1y",
                                 interval: str = "1d") -> pd.DataFrame:
        """Fallback: fetch data from yfinance for backtesting / supplementary data."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            df.columns = [c.lower() for c in df.columns]
            return df
        except ImportError:
            logger.warning("yfinance not installed, skipping historical fetch")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")
            return pd.DataFrame()

    # ────────────────────── Technical Indicators ──────────────────────

    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to OHLCV DataFrame."""
        if df.empty:
            return df

        df = df.copy()

        # Simple Moving Averages
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["sma_200"] = df["close"].rolling(200).mean()

        # Exponential Moving Averages
        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # RSI (14-period)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        # ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # Z-score (for mean reversion)
        df["zscore"] = (df["close"] - df["sma_20"]) / df["close"].rolling(20).std()

        # Momentum (rate of change)
        df["momentum_1m"] = df["close"].pct_change(21)   # ~1 month
        df["momentum_3m"] = df["close"].pct_change(63)   # ~3 months
        df["momentum_12m"] = df["close"].pct_change(252)  # ~12 months

        # Volume indicators
        if "volume" in df.columns and df["volume"].sum() > 0:
            df["volume_sma20"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma20"]

        # Volatility
        df["volatility_20d"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)

        return df

    # ────────────────────── Macro Data ──────────────────────

    def get_fred_data(self, series_id: str, start_date: str = None) -> pd.Series:
        """Fetch macroeconomic data from FRED."""
        try:
            import pandas_datareader.data as web
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
            data = web.DataReader(series_id, "fred", start_date)
            return data.iloc[:, 0]
        except ImportError:
            logger.warning("pandas-datareader not installed")
            return pd.Series()
        except Exception as e:
            logger.error(f"FRED error for {series_id}: {e}")
            return pd.Series()

    def get_macro_indicators(self) -> dict:
        """Fetch key macro indicators for regime detection."""
        indicators = {}
        fred_series = {
            "DFF": "fed_funds_rate",
            "T10Y2Y": "yield_curve_10y2y",
            "VIXCLS": "vix",
            "CPIAUCSL": "cpi",
            "UNRATE": "unemployment",
            "DGS10": "treasury_10y",
        }
        for series_id, name in fred_series.items():
            data = self.get_fred_data(series_id)
            if not data.empty:
                indicators[name] = {
                    "latest": float(data.iloc[-1]),
                    "change_1m": float(data.iloc[-1] - data.iloc[-22]) if len(data) > 22 else None,
                    "series": data,
                }
        return indicators

    # ────────────────────── Correlation & Cointegration ──────────────────────

    @staticmethod
    def compute_correlation_matrix(returns_dict: Dict[str, pd.Series],
                                    window: int = 60) -> pd.DataFrame:
        """Compute rolling correlation matrix from returns."""
        df = pd.DataFrame(returns_dict)
        return df.rolling(window).corr().dropna()

    @staticmethod
    def test_cointegration(series_a: pd.Series, series_b: pd.Series) -> dict:
        """Test for cointegration between two price series (for pairs trading)."""
        try:
            from statsmodels.tsa.stattools import coint
            score, pvalue, _ = coint(series_a.dropna(), series_b.dropna())
            return {
                "cointegrated": pvalue < 0.05,
                "p_value": pvalue,
                "test_statistic": score,
            }
        except ImportError:
            logger.warning("statsmodels not installed")
            return {"cointegrated": False, "p_value": 1.0}

    # ────────────────────── Cache Management ──────────────────────

    def clear_cache(self):
        """Clear all cached data."""
        self._price_cache.clear()
        self._fundamentals_cache.clear()
        self._macro_cache.clear()
        logger.info("Data cache cleared")
