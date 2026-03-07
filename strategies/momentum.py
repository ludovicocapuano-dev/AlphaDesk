"""
AlphaDesk — Momentum / Trend Following Strategy
Dual momentum (absolute + relative) with volume confirmation.
Allocation: 30% — Equities US/EU
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal, TradeSignal

logger = logging.getLogger("alphadesk.strategy.momentum")


class MomentumStrategy(BaseStrategy):
    """
    Dual momentum strategy combining:
    1. Absolute momentum: is the asset trending up?
    2. Relative momentum: is it outperforming peers?

    Entry: 20-day breakout + volume surge + trend confirmation
    Exit: trailing stop at 2x ATR or momentum reversal
    """

    def __init__(self, allocation_pct: float = 0.30):
        super().__init__(
            name="momentum",
            allocation_pct=allocation_pct,
            max_positions=8,
        )
        # Parameters
        self.breakout_period = 20        # Days for high/low breakout
        self.trend_sma = 50              # SMA for trend filter
        self.long_trend_sma = 200        # Long-term trend
        self.volume_threshold = 1.5      # Volume must be 1.5x average
        self.atr_multiplier = 2.0        # Stop loss = 2x ATR
        self.min_momentum_3m = 0.05      # Min 5% return over 3 months
        self.rsi_overbought = 75
        self.rsi_oversold = 30

    async def generate_signals(self, data_engine, current_positions: List[dict]) -> List[TradeSignal]:
        """Scan universe for momentum breakouts."""
        from config.instruments import US_EQUITIES, EU_EQUITIES

        signals = []
        universe = {**US_EQUITIES, **EU_EQUITIES}

        for symbol, meta in universe.items():
            instrument_id = meta.get("etoro_id")
            if instrument_id is None:
                continue

            try:
                # Fetch daily data
                df = await data_engine.get_ohlcv(instrument_id, symbol, "OneDay", 252)
                if df.empty or len(df) < self.long_trend_sma:
                    continue

                df = data_engine.compute_indicators(df)
                signal = self._evaluate_momentum(symbol, instrument_id, df)
                if signal:
                    signals.append(signal)

            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}")

        # Rank by relative momentum and filter
        signals.sort(key=lambda s: s.metadata.get("momentum_score", 0), reverse=True)
        return self.filter_signals(signals)

    def _evaluate_momentum(self, symbol: str, instrument_id: int,
                            df: pd.DataFrame) -> Optional[TradeSignal]:
        """Evaluate a single instrument for momentum signal."""
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # ── Absolute momentum check ──
        # Price above both 50 and 200 SMA (bullish trend)
        if latest["close"] <= latest["sma_50"] or latest["close"] <= latest["sma_200"]:
            return None

        # 50 SMA above 200 SMA (golden cross territory)
        if latest["sma_50"] <= latest["sma_200"]:
            return None

        # Positive 3-month momentum
        if latest.get("momentum_3m", 0) < self.min_momentum_3m:
            return None

        # ── Breakout detection ──
        high_20 = df["high"].rolling(self.breakout_period).max().iloc[-2]  # Previous day's 20d high
        breakout = latest["close"] > high_20

        if not breakout:
            return None

        # ── Volume confirmation ──
        volume_ratio = latest.get("volume_ratio", 0)
        has_volume = volume_ratio >= self.volume_threshold if volume_ratio > 0 else True

        # ── RSI filter (avoid overbought) ──
        rsi = latest.get("rsi", 50)
        if rsi > self.rsi_overbought:
            return None

        # ── Compute signal strength ──
        confidence = self._compute_confidence(latest, df)

        # ── Entry / Exit levels ──
        entry = latest["close"]
        atr = latest["atr"]
        stop_loss = entry - (self.atr_multiplier * atr)
        # Take profit at 3x risk (aggressive)
        take_profit = entry + (3.0 * self.atr_multiplier * atr)

        # Momentum composite score for ranking
        momentum_score = (
            0.4 * self._normalize(latest.get("momentum_1m", 0), -0.1, 0.2) +
            0.3 * self._normalize(latest.get("momentum_3m", 0), -0.15, 0.3) +
            0.3 * self._normalize(latest.get("momentum_12m", 0), -0.2, 0.5)
        )

        signal_type = Signal.STRONG_BUY if confidence > 0.8 and has_volume else Signal.BUY

        return TradeSignal(
            symbol=symbol,
            instrument_id=instrument_id,
            signal=signal_type,
            strategy_name=self.name,
            confidence=confidence,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            suggested_size_pct=min(0.05, confidence * 0.06),  # Max 5% per position
            metadata={
                "momentum_score": momentum_score,
                "rsi": rsi,
                "volume_ratio": volume_ratio,
                "atr": atr,
                "sma_50_dist": (entry - latest["sma_50"]) / entry,
                "breakout_level": high_20,
            },
        )

    def _compute_confidence(self, latest: pd.Series, df: pd.DataFrame) -> float:
        """Compute signal confidence 0-1 based on multiple factors."""
        score = 0.0

        # Trend alignment (all MAs aligned)
        if latest["close"] > latest["sma_20"] > latest["sma_50"] > latest["sma_200"]:
            score += 0.25
        elif latest["close"] > latest["sma_50"] > latest["sma_200"]:
            score += 0.15

        # MACD positive and rising
        if latest["macd"] > 0 and latest["macd_histogram"] > 0:
            score += 0.20

        # RSI in sweet spot (40-65)
        rsi = latest.get("rsi", 50)
        if 40 <= rsi <= 65:
            score += 0.15
        elif 30 <= rsi <= 75:
            score += 0.08

        # Strong volume
        vr = latest.get("volume_ratio", 1)
        if vr >= 2.0:
            score += 0.20
        elif vr >= 1.5:
            score += 0.10

        # Low volatility (trend is clean)
        vol = latest.get("volatility_20d", 0.25)
        if vol < 0.20:
            score += 0.10
        elif vol < 0.30:
            score += 0.05

        # Positive multi-timeframe momentum
        if latest.get("momentum_1m", 0) > 0 and latest.get("momentum_3m", 0) > 0:
            score += 0.10

        return min(1.0, score)

    def should_exit(self, position: dict, current_data: pd.DataFrame) -> Optional[TradeSignal]:
        """Check if momentum position should be closed."""
        if current_data.empty:
            return None

        latest = current_data.iloc[-1]
        entry_price = position.get("openRate", position.get("entry_price", 0))

        # Exit conditions:
        # 1. Price drops below 50 SMA
        if latest["close"] < latest.get("sma_50", 0):
            return self._create_exit_signal(position, latest, "Price below SMA50")

        # 2. MACD crossover to negative
        if latest.get("macd_histogram", 0) < 0 and latest.get("macd", 0) < 0:
            return self._create_exit_signal(position, latest, "MACD bearish crossover")

        # 3. RSI extreme overbought
        if latest.get("rsi", 50) > 85:
            return self._create_exit_signal(position, latest, "RSI extreme overbought")

        return None

    def _create_exit_signal(self, position: dict, latest: pd.Series, reason: str) -> TradeSignal:
        return TradeSignal(
            symbol=position.get("symbol", ""),
            instrument_id=position.get("instrumentId", 0),
            signal=Signal.SELL,
            strategy_name=self.name,
            confidence=0.7,
            entry_price=latest["close"],
            stop_loss=latest["close"],
            take_profit=latest["close"],
            suggested_size_pct=1.0,  # Close full position
            metadata={"exit_reason": reason},
        )

    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-1 range."""
        if max_val == min_val:
            return 0.5
        return max(0, min(1, (value - min_val) / (max_val - min_val)))
