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


class RegimeDetector:
    """
    Detects current market regime from price data and macro indicators.

    Produces a RegimeFingerprint that is:
    1. Deterministic (same inputs → same output)
    2. Attached to every signal/trade for ML learning
    3. Used to filter strategies dynamically
    """

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
        regime = {
            "timestamp": datetime.utcnow().isoformat(),
            "volatility_regime": self._detect_volatility(market_data, vix),
            "trend_regime": self._detect_trend(market_data),
            "liquidity_regime": self._detect_liquidity(market_data),
            "rate_regime": self._detect_rate_regime(macro_indicators),
            "correlation_regime": self._detect_correlation(market_data),
        }

        fingerprint = RegimeFingerprint(regime)
        logger.info(f"Regime detected: {fingerprint}")
        return fingerprint

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
