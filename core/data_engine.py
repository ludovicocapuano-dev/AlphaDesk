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

        # Fractionally differentiated close (FFD, d=0.35)
        # Preserves memory while being stationary — better ML features
        if len(df) > 50:
            ffd = DataEngine.frac_diff(df["close"], d=0.35)
            df["close_ffd"] = ffd

        return df

    # ────────────────────── CUSUM Filter (López de Prado) ──────────────────────

    @staticmethod
    def cusum_filter(close: pd.Series, threshold: float = None) -> pd.DatetimeIndex:
        """
        CUSUM event-driven sampling (Advances in Financial ML, Ch. 2).

        Instead of sampling at fixed time intervals, detect structural
        changes in the price series. Returns timestamps where cumulative
        sum of returns breaches the threshold.

        Args:
            close: Price series with DatetimeIndex.
            threshold: CUSUM threshold. If None, uses 1x daily std dev.

        Returns:
            DatetimeIndex of event timestamps.
        """
        if close.empty or len(close) < 20:
            return pd.DatetimeIndex([])

        if threshold is None:
            threshold = close.pct_change().std()

        events = []
        s_pos, s_neg = 0.0, 0.0
        diff = close.pct_change().dropna()

        for t, val in diff.items():
            s_pos = max(0, s_pos + val)
            s_neg = min(0, s_neg + val)

            if s_pos > threshold:
                events.append(t)
                s_pos = 0.0
            elif s_neg < -threshold:
                events.append(t)
                s_neg = 0.0

        return pd.DatetimeIndex(events)

    # ────────────────────── Fractional Differentiation (López de Prado) ──────────

    @staticmethod
    def frac_diff(series: pd.Series, d: float = 0.35, threshold: float = 1e-4) -> pd.Series:
        """
        Fixed-width window fractional differentiation (FFD).

        Preserves memory (mean-reversion signal) while achieving stationarity.
        d≈0.35 typically preserves >95% correlation with original series
        while passing ADF stationarity test.

        Args:
            series: Price or log-price series.
            d: Fractional differentiation order (0 < d < 1).
               d=0 is original series, d=1 is first difference.
            threshold: Weight cutoff for the FFD window.

        Returns:
            Fractionally differentiated series.
        """
        if series.empty:
            return series

        # Compute FFD weights, cap window to half the series length
        max_window = max(10, len(series) // 2)
        weights = [1.0]
        k = 1
        while True:
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < threshold or k >= max_window:
                break
            weights.append(w)
            k += 1

        weights = np.array(weights[::-1])  # Reverse for convolution
        width = len(weights)

        # Apply weights via dot product
        result = {}
        for i in range(width - 1, len(series)):
            window = series.iloc[i - width + 1: i + 1].values
            if len(window) == width:
                result[series.index[i]] = np.dot(weights, window)

        return pd.Series(result, dtype=float)

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

    # ────────────────────── Covariance Denoising (Marcenko-Pastur) ──────────────────

    @staticmethod
    def denoise_covariance(returns_df: pd.DataFrame, num_factors: int = None) -> pd.DataFrame:
        """
        Denoise a covariance matrix using the Marcenko-Pastur theorem
        (López de Prado, Ch. 2).

        Separates signal eigenvalues from noise eigenvalues based on
        random matrix theory. Noise eigenvalues are shrunk to their
        average, preserving only the signal components.

        Args:
            returns_df: DataFrame of asset returns (T x N).
            num_factors: If None, auto-detect using Marcenko-Pastur bound.

        Returns:
            Denoised covariance matrix as DataFrame.
        """
        cov = returns_df.cov()
        corr = returns_df.corr()
        T, N = returns_df.shape
        q = T / N  # Observations-to-variables ratio

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(corr.values)
        idx = eigenvalues.argsort()[::-1]  # Sort descending
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Marcenko-Pastur upper bound for noise eigenvalues
        lambda_plus = (1 + 1 / np.sqrt(q)) ** 2

        if num_factors is None:
            # Count eigenvalues above the MP noise threshold
            num_factors = int(np.sum(eigenvalues > lambda_plus))
            num_factors = max(1, num_factors)

        # Shrink noise eigenvalues to their average
        noise_eigenvalues = eigenvalues[num_factors:]
        if len(noise_eigenvalues) > 0:
            noise_avg = np.mean(noise_eigenvalues)
            eigenvalues[num_factors:] = noise_avg

        # Reconstruct denoised correlation matrix
        denoised_corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Force diagonal to 1 (correlation matrix property)
        np.fill_diagonal(denoised_corr, 1.0)

        # Convert back to covariance
        stds = np.sqrt(np.diag(cov.values))
        denoised_cov = denoised_corr * np.outer(stds, stds)

        return pd.DataFrame(denoised_cov, index=cov.index, columns=cov.columns)

    # ────────────────────── SADF Structural Break Detection ──────────────────────

    @staticmethod
    def _adf_tstat(y: np.ndarray, max_lags: int = 1) -> float:
        """
        Compute the ADF t-statistic for a single window using OLS.

        Regression: Δy = α + β·y[t-1] + Σ(γ_i·Δy[t-i]) + ε
        Returns the t-stat of β (the coefficient on y[t-1]).
        """
        dy = np.diff(y)  # Δy, length n-1
        n = len(dy)
        if n < max_lags + 2:
            return np.nan

        # Dependent variable: Δy[max_lags:]
        dep = dy[max_lags:]
        T = len(dep)

        # Regressors: intercept, y[t-1], lagged Δy
        y_lag = y[max_lags:-1]  # y[t-1] aligned with dep
        cols = [np.ones(T), y_lag]
        for lag in range(1, max_lags + 1):
            cols.append(dy[max_lags - lag: -lag] if lag < len(dy) - max_lags + 1
                        else dy[max_lags - lag: n - lag])

        X = np.column_stack(cols)

        # OLS: β = (X'X)^{-1} X'y
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            return np.nan

        beta = XtX_inv @ (X.T @ dep)
        residuals = dep - X @ beta
        sigma2 = np.sum(residuals ** 2) / max(T - X.shape[1], 1)
        se = np.sqrt(np.diag(XtX_inv) * sigma2)

        if se[1] == 0:
            return np.nan
        return beta[1] / se[1]

    @staticmethod
    def sadf_test(log_prices: pd.Series, min_window: int = 40,
                  max_lags: int = 1) -> pd.Series:
        """
        Supremum ADF test for structural breaks / bubble detection
        (López de Prado, AFML Ch. 17 / Phillips, Shi, Yu 2015).

        Runs a sequence of ADF tests on expanding windows, starting from
        min_window observations. The supremum of the ADF t-statistics
        at each time point indicates explosive behavior (bubbles).

        SADF > critical value → evidence of a bubble/structural break.
        Critical values (95%): ~1.0 for T=100, ~1.5 for T=200.

        Args:
            log_prices: Log of price series.
            min_window: Minimum window size for the first ADF test.
            max_lags: Maximum number of lags in the ADF regression.

        Returns:
            Series of SADF statistics indexed by date. Higher values
            indicate more evidence of explosive (bubble) behavior.
        """
        values = log_prices.values.astype(float)
        index = log_prices.index
        n = len(values)

        if n < min_window + 2:
            return pd.Series(dtype=float)

        sadf_stats = {}

        for t in range(min_window, n):
            # Generate at most 5 evenly spaced starting points
            num_starts = min(5, t - min_window + 1)
            start_indices = np.linspace(0, t - min_window, num_starts, dtype=int)
            # Deduplicate (linspace can produce repeats for small ranges)
            start_indices = np.unique(start_indices)

            sup_adf = -np.inf
            for s in start_indices:
                window = values[s: t + 1]
                if len(window) < min_window:
                    continue
                adf_stat = DataEngine._adf_tstat(window, max_lags)
                if not np.isnan(adf_stat) and adf_stat > sup_adf:
                    sup_adf = adf_stat

            if sup_adf > -np.inf:
                sadf_stats[index[t]] = sup_adf

        return pd.Series(sadf_stats, dtype=float)

    @staticmethod
    def detect_bubbles(close: pd.Series, min_window: int = 40,
                       threshold: float = 1.0) -> pd.DataFrame:
        """
        Detect bubble periods using SADF.

        Computes SADF on the log of the close price series and flags
        time points where the SADF statistic exceeds the threshold
        as bubble regimes.

        Args:
            close: Price series (not log — log is taken internally).
            min_window: Minimum window for the SADF test.
            threshold: SADF threshold above which a bubble is flagged.
                       Typical 95% critical values: ~1.0 (T=100), ~1.5 (T=200).

        Returns:
            DataFrame with columns: sadf_stat, is_bubble.
        """
        log_prices = np.log(close.replace(0, np.nan).dropna())
        sadf = DataEngine.sadf_test(log_prices, min_window=min_window)

        result = pd.DataFrame({
            "sadf_stat": sadf,
            "is_bubble": sadf > threshold,
        })
        return result

    # ────────────────────── Cache Management ──────────────────────

    def clear_cache(self):
        """Clear all cached data."""
        self._price_cache.clear()
        self._fundamentals_cache.clear()
        self._macro_cache.clear()
        logger.info("Data cache cleared")
