"""
AlphaDesk — Market Regime Detector
Inspired by Chris Turner's "Evolutionary Crypto Trading Bot" approach.

Produces a deterministic regime fingerprint that characterizes the current
market state across three dimensions: volatility, trend, and liquidity.
This fingerprint is attached to every signal and trade decision, enabling
the ML ensemble to learn which strategies work in which regimes.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

logger = logging.getLogger("alphadesk.regime")


class RegimeFingerprint:
    """
    A deterministic, reproducible snapshot of market conditions.

    Components:
    - volatility_regime: low | medium | high | extreme
    - trend_regime: strong_up | weak_up | ranging | weak_down | strong_down
    - liquidity_regime: low | normal | high
    - rate_regime: easing | neutral | tightening
    - correlation_regime: normal | elevated | crisis

    The fingerprint is a JSON-serializable dict with a hash for quick comparison.
    """

    def __init__(self, data: dict):
        self.data = data
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Deterministic hash of the regime state."""
        canonical = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {**self.data, "hash": self.hash}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @property
    def is_extreme(self) -> bool:
        """Check if regime signals extreme conditions (pause trading)."""
        return (
            self.data.get("volatility_regime") == "extreme" or
            self.data.get("correlation_regime") == "crisis"
        )

    @property
    def is_favorable_for_momentum(self) -> bool:
        return (
            self.data.get("trend_regime") in ("strong_up", "strong_down") and
            self.data.get("volatility_regime") in ("low", "medium")
        )

    @property
    def is_favorable_for_mean_reversion(self) -> bool:
        return (
            self.data.get("trend_regime") == "ranging" and
            self.data.get("volatility_regime") in ("medium", "high")
        )

    def __repr__(self):
        return (
            f"Regime(vol={self.data.get('volatility_regime')}, "
            f"trend={self.data.get('trend_regime')}, "
            f"liq={self.data.get('liquidity_regime')}, "
            f"hash={self.hash[:8]})"
        )


class HMMRegimeDetector:
    """
    Hidden Markov Model regime detector.

    Uses a 2-state Gaussian HMM on daily returns to classify the market
    as bull or bear. Falls back to a simple numpy-based approximation
    when hmmlearn is not installed.
    """

    def __init__(self, n_states: int = 2, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self._fitted = False
        self._model = None
        self._state_means = None  # mean return per state after fitting
        self._transition_matrix = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, returns: np.ndarray) -> None:
        """
        Fit the 2-state HMM on a 1-D array of daily returns.

        Args:
            returns: 1-D array of daily log/simple returns (length >= 30).
        """
        returns = np.asarray(returns, dtype=np.float64).ravel()
        returns = returns[~np.isnan(returns)]

        if len(returns) < 30:
            logger.warning("HMM fit skipped: need >= 30 observations, got %d", len(returns))
            return

        if HAS_HMM:
            self._fit_hmmlearn(returns)
        else:
            self._fit_numpy_fallback(returns)

        self._fitted = True
        logger.info(
            "HMM fitted (%s backend): state means %s",
            "hmmlearn" if HAS_HMM else "numpy-fallback",
            self._state_means,
        )

    # ---------- hmmlearn backend ----------

    def _fit_hmmlearn(self, returns: np.ndarray) -> None:
        X = returns.reshape(-1, 1)
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state,
        )
        model.fit(X)
        self._model = model
        self._state_means = model.means_.ravel()
        self._transition_matrix = model.transmat_

    # ---------- numpy fallback ----------

    def _fit_numpy_fallback(self, returns: np.ndarray) -> None:
        """Simple rolling-stats approximation when hmmlearn is unavailable."""
        window = min(20, len(returns) // 2)
        if window < 5:
            window = 5

        roll_mean = pd.Series(returns).rolling(window, min_periods=window).mean().values
        roll_std = pd.Series(returns).rolling(window, min_periods=window).std().values

        valid = ~(np.isnan(roll_mean) | np.isnan(roll_std))
        roll_mean = roll_mean[valid]
        roll_std = roll_std[valid]

        if len(roll_mean) == 0:
            logger.warning("HMM numpy fallback: not enough data after rolling window")
            return

        med_mean = np.median(roll_mean)
        med_std = np.median(roll_std)

        # State assignment: 1=bull (high return, low vol), 0=bear
        states = np.where(
            (roll_mean >= med_mean) & (roll_std <= med_std), 1, 0
        )

        # Compute per-state mean return
        self._state_means = np.array([
            roll_mean[states == 0].mean() if np.any(states == 0) else -0.001,
            roll_mean[states == 1].mean() if np.any(states == 1) else 0.001,
        ])

        # Estimate transition matrix from state sequence
        trans = np.zeros((2, 2))
        for i in range(len(states) - 1):
            trans[states[i], states[i + 1]] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self._transition_matrix = trans / row_sums

        # Store states for predict (last state)
        self._last_fallback_returns = returns
        self._last_fallback_states = states

    # ---------- prediction ----------

    def predict_regime(self, returns: np.ndarray) -> str:
        """
        Predict current regime from the most recent returns.

        Args:
            returns: 1-D array of recent daily returns.

        Returns:
            "bull", "bear", or "unknown" if not fitted.
        """
        if not self._fitted:
            return "unknown"

        returns = np.asarray(returns, dtype=np.float64).ravel()
        returns = returns[~np.isnan(returns)]
        if len(returns) == 0:
            return "unknown"

        if HAS_HMM and self._model is not None:
            state = self._model.predict(returns.reshape(-1, 1))[-1]
        else:
            # Numpy fallback: classify the latest window
            window = 20
            tail = returns[-window:] if len(returns) >= window else returns
            avg_ret = np.mean(tail)
            avg_vol = np.std(tail)
            med_mean = np.mean(self._state_means)
            # Simple threshold: above mean-of-means → bull
            state = 1 if avg_ret >= med_mean else 0

        # Map state to label: higher-mean state = bull
        bull_state = int(np.argmax(self._state_means))
        return "bull" if state == bull_state else "bear"

    def get_transition_probs(self) -> dict:
        """
        Return the transition matrix as a dict.

        Returns:
            {"bear_to_bear": float, "bear_to_bull": float,
             "bull_to_bear": float, "bull_to_bull": float}
            or empty dict if not fitted.
        """
        if not self._fitted or self._transition_matrix is None:
            return {}

        bull = int(np.argmax(self._state_means))
        bear = 1 - bull
        tm = self._transition_matrix
        return {
            "bear_to_bear": float(tm[bear, bear]),
            "bear_to_bull": float(tm[bear, bull]),
            "bull_to_bear": float(tm[bull, bear]),
            "bull_to_bull": float(tm[bull, bull]),
        }


class RegimeDetector:
    """
    Detects current market regime from price data and macro indicators.

    Produces a RegimeFingerprint that is:
    1. Deterministic (same inputs → same output)
    2. Attached to every signal/trade for ML learning
    3. Used to filter strategies dynamically
    """

    def __init__(self):
        self.hmm = HMMRegimeDetector()

    # Volatility thresholds (annualized, based on VIX-equivalent)
    VOL_THRESHOLDS = {
        "low": 0.12,       # < 12%
        "medium": 0.20,    # 12-20%
        "high": 0.30,      # 20-30%
        # > 30% = extreme
    }

    # Trend thresholds (based on SMA positioning)
    TREND_STRONG_THRESHOLD = 0.03    # > 3% above/below SMA
    TREND_WEAK_THRESHOLD = 0.01      # 1-3%

    def detect(self, market_data: Dict[str, pd.DataFrame],
               vix: float = None,
               macro_indicators: dict = None) -> RegimeFingerprint:
        """
        Detect current market regime from available data.

        Args:
            market_data: dict of symbol → OHLCV DataFrame (with indicators)
            vix: Current VIX level (optional, improves accuracy)
            macro_indicators: Macro data dict (optional)

        Returns:
            RegimeFingerprint
        """
        # HMM regime (uses pre-fitted model; default "unknown")
        hmm_regime = "unknown"
        if self.hmm.is_fitted:
            try:
                # Build returns from market_data for prediction
                all_returns = []
                for symbol, df in market_data.items():
                    if not df.empty and len(df) > 1:
                        rets = df["close"].pct_change().dropna().values
                        all_returns.append(rets)
                if all_returns:
                    min_len = min(len(r) for r in all_returns)
                    stacked = np.column_stack([r[-min_len:] for r in all_returns])
                    portfolio_returns = stacked.mean(axis=1)
                    hmm_regime = self.hmm.predict_regime(portfolio_returns)
            except Exception as e:
                logger.warning("HMM prediction failed: %s", e)

        regime = {
            "timestamp": datetime.utcnow().isoformat(),
            "volatility_regime": self._detect_volatility(market_data, vix),
            "trend_regime": self._detect_trend(market_data),
            "liquidity_regime": self._detect_liquidity(market_data),
            "rate_regime": self._detect_rate_regime(macro_indicators),
            "correlation_regime": self._detect_correlation(market_data),
            "hmm_regime": hmm_regime,
        }

        fingerprint = RegimeFingerprint(regime)
        logger.info(f"Regime detected: {fingerprint}")
        return fingerprint

    def fit_hmm(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """
        Fit the HMM on equal-weighted portfolio returns.

        Call this during daily retrain, not every signal scan.

        Args:
            market_data: dict of symbol -> OHLCV DataFrame.
        """
        all_returns = []
        for symbol, df in market_data.items():
            if not df.empty and len(df) > 1:
                rets = df["close"].pct_change().dropna().values
                all_returns.append(rets)

        if not all_returns:
            logger.warning("fit_hmm: no valid return series found")
            return

        min_len = min(len(r) for r in all_returns)
        if min_len < 30:
            logger.warning("fit_hmm: need >= 30 return observations, got %d", min_len)
            return

        stacked = np.column_stack([r[-min_len:] for r in all_returns])
        portfolio_returns = stacked.mean(axis=1)

        self.hmm.fit(portfolio_returns)
        trans = self.hmm.get_transition_probs()
        logger.info("HMM fitted on %d observations, transitions: %s", min_len, trans)

    def _detect_volatility(self, market_data: Dict[str, pd.DataFrame],
                            vix: float = None) -> str:
        """Classify volatility regime."""
        if vix is not None:
            # Direct VIX mapping
            if vix < 12:
                return "low"
            elif vix < 20:
                return "medium"
            elif vix < 30:
                return "high"
            else:
                return "extreme"

        # Fallback: compute from price data
        vols = []
        for symbol, df in market_data.items():
            if "volatility_20d" in df.columns and not df.empty:
                vol = df.iloc[-1]["volatility_20d"]
                if not np.isnan(vol):
                    vols.append(vol)

        if not vols:
            return "medium"

        avg_vol = np.mean(vols)
        if avg_vol < self.VOL_THRESHOLDS["low"]:
            return "low"
        elif avg_vol < self.VOL_THRESHOLDS["medium"]:
            return "medium"
        elif avg_vol < self.VOL_THRESHOLDS["high"]:
            return "high"
        else:
            return "extreme"

    def _detect_trend(self, market_data: Dict[str, pd.DataFrame]) -> str:
        """Classify trend regime across the universe."""
        trend_scores = []

        for symbol, df in market_data.items():
            if df.empty or "sma_50" not in df.columns or "sma_200" not in df.columns:
                continue

            latest = df.iloc[-1]
            price = latest["close"]
            sma50 = latest["sma_50"]
            sma200 = latest["sma_200"]

            if sma200 > 0:
                # Distance from 200 SMA as trend indicator
                dist = (price - sma200) / sma200
                trend_scores.append(dist)

        if not trend_scores:
            return "ranging"

        avg_trend = np.mean(trend_scores)

        if avg_trend > self.TREND_STRONG_THRESHOLD:
            return "strong_up"
        elif avg_trend > self.TREND_WEAK_THRESHOLD:
            return "weak_up"
        elif avg_trend < -self.TREND_STRONG_THRESHOLD:
            return "strong_down"
        elif avg_trend < -self.TREND_WEAK_THRESHOLD:
            return "weak_down"
        else:
            return "ranging"

    def _detect_liquidity(self, market_data: Dict[str, pd.DataFrame]) -> str:
        """Classify liquidity based on volume ratios."""
        vol_ratios = []

        for symbol, df in market_data.items():
            if "volume_ratio" in df.columns and not df.empty:
                vr = df.iloc[-1].get("volume_ratio", 1.0)
                if not np.isnan(vr):
                    vol_ratios.append(vr)

        if not vol_ratios:
            return "normal"

        avg_vr = np.mean(vol_ratios)
        if avg_vr < 0.6:
            return "low"
        elif avg_vr > 1.5:
            return "high"
        else:
            return "normal"

    def _detect_rate_regime(self, macro: dict = None) -> str:
        """Classify interest rate regime."""
        if macro is None:
            return "neutral"

        fed_rate = macro.get("fed_funds_rate", {})
        if isinstance(fed_rate, dict):
            change = fed_rate.get("change_1m")
            if change is not None:
                if change < -0.1:
                    return "easing"
                elif change > 0.1:
                    return "tightening"
        return "neutral"

    def _detect_correlation(self, market_data: Dict[str, pd.DataFrame]) -> str:
        """Detect if cross-asset correlations are abnormally high (crisis signal)."""
        returns_dict = {}
        for symbol, df in market_data.items():
            if not df.empty and len(df) > 20:
                returns_dict[symbol] = df["close"].pct_change().dropna().tail(20)

        if len(returns_dict) < 3:
            return "normal"

        returns_df = pd.DataFrame(returns_dict).dropna()
        if returns_df.shape[1] < 3:
            return "normal"

        corr_matrix = returns_df.corr()
        # Average pairwise correlation (excluding diagonal)
        n = corr_matrix.shape[0]
        if n < 2:
            return "normal"

        upper_tri = corr_matrix.values[np.triu_indices(n, k=1)]
        avg_corr = np.mean(upper_tri)

        if avg_corr > 0.7:
            return "crisis"
        elif avg_corr > 0.5:
            return "elevated"
        else:
            return "normal"
