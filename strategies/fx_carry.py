"""
AlphaDesk — FX Carry + Momentum Strategy
Carry trade with momentum overlay for forex pairs.
Allocation: 30% — Forex
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal, TradeSignal

logger = logging.getLogger("alphadesk.strategy.fx_carry")


# Historical central bank policy rates for backtesting accuracy
# Each entry: (effective_date, rates_dict) — sorted chronologically
HISTORICAL_RATES = [
    ("2023-01-01", {"USD": 0.0450, "EUR": 0.0250, "GBP": 0.0350, "JPY": -0.001, "AUD": 0.0310, "NZD": 0.0425, "CHF": 0.0100, "CAD": 0.0425}),
    ("2023-05-01", {"USD": 0.0525, "EUR": 0.0375, "GBP": 0.0425, "JPY": -0.001, "AUD": 0.0385, "NZD": 0.0525, "CHF": 0.0150, "CAD": 0.0475}),
    ("2023-09-01", {"USD": 0.0550, "EUR": 0.0450, "GBP": 0.0525, "JPY": -0.001, "AUD": 0.0410, "NZD": 0.0550, "CHF": 0.0175, "CAD": 0.0500}),
    ("2024-03-01", {"USD": 0.0550, "EUR": 0.0450, "GBP": 0.0525, "JPY": 0.0000, "AUD": 0.0435, "NZD": 0.0550, "CHF": 0.0175, "CAD": 0.0500}),
    ("2024-07-01", {"USD": 0.0550, "EUR": 0.0425, "GBP": 0.0500, "JPY": 0.0025, "AUD": 0.0435, "NZD": 0.0550, "CHF": 0.0150, "CAD": 0.0475}),
    ("2024-09-01", {"USD": 0.0500, "EUR": 0.0375, "GBP": 0.0500, "JPY": 0.0025, "AUD": 0.0435, "NZD": 0.0525, "CHF": 0.0125, "CAD": 0.0450}),
    ("2024-12-01", {"USD": 0.0450, "EUR": 0.0325, "GBP": 0.0475, "JPY": 0.0025, "AUD": 0.0435, "NZD": 0.0475, "CHF": 0.0050, "CAD": 0.0350}),
    ("2025-06-01", {"USD": 0.0425, "EUR": 0.0275, "GBP": 0.0425, "JPY": 0.0025, "AUD": 0.0410, "NZD": 0.0400, "CHF": 0.0025, "CAD": 0.0300}),
    ("2026-01-01", {"USD": 0.0375, "EUR": 0.0250, "GBP": 0.0400, "JPY": 0.0025, "AUD": 0.0385, "NZD": 0.0350, "CHF": 0.0025, "CAD": 0.0250}),
]

# Current rates (latest entry) for live trading
POLICY_RATES = HISTORICAL_RATES[-1][1]


def get_rates_for_date(date_str: str) -> dict:
    """Get policy rates applicable for a given date."""
    for i in range(len(HISTORICAL_RATES) - 1, -1, -1):
        if date_str >= HISTORICAL_RATES[i][0]:
            return HISTORICAL_RATES[i][1]
    return HISTORICAL_RATES[0][1]


class FXCarryStrategy(BaseStrategy):
    """
    FX Carry Trade + Momentum:
    - Carry: long high-yield, short low-yield currencies
    - Momentum: 1-month and 3-month price momentum overlay
    - Risk: per-pair stop at 1.5% portfolio, correlation limits

    The carry trade earns the interest rate differential (rollover).
    Momentum filter avoids carry trades against strong trends.
    """

    def __init__(self, allocation_pct: float = 0.30):
        super().__init__(
            name="fx_carry",
            allocation_pct=allocation_pct,
            max_positions=6,
        )
        # Parameters (optimized 2026-03-08, backtest 2023-2025)
        self.min_carry_spread = 0.005    # Min 0.5% rate differential
        self.momentum_weight = 0.70      # Momentum contribution to score
        self.carry_weight = 0.30         # Carry contribution to score
        self.max_risk_per_pair = 0.025   # 2.5% max risk per pair
        self.trend_filter_sma = 50       # SMA for trend direction
        self.atr_stop_multiplier = 2.0   # Stop at 2.0x ATR
        self.tp_multiplier = 2.0         # Take profit at 2.0x stop distance

    async def generate_signals(self, data_engine, current_positions: List[dict]) -> List[TradeSignal]:
        """Score FX pairs by carry + momentum and generate signals."""
        from config.instruments import FX_PAIRS

        scored_pairs = []

        for pair_name, meta in FX_PAIRS.items():
            instrument_id = meta.get("etoro_id")
            if instrument_id is None:
                continue

            try:
                df = await data_engine.get_ohlcv(instrument_id, pair_name, "OneDay", 120)
                if df.empty or len(df) < 60:
                    continue

                df = data_engine.compute_indicators(df)

                # Compute carry + momentum score
                score_data = self._score_pair(pair_name, meta, df)
                if score_data:
                    scored_pairs.append({
                        "pair": pair_name,
                        "instrument_id": instrument_id,
                        "meta": meta,
                        "df": df,
                        **score_data,
                    })

            except Exception as e:
                logger.error(f"FX Carry error for {pair_name}: {e}")

        if not scored_pairs:
            return []

        # Rank by composite score and generate top signals
        scored_pairs.sort(key=lambda x: abs(x["composite_score"]), reverse=True)
        return self._generate_trade_signals(scored_pairs)

    def _score_pair(self, pair_name: str, meta: dict, df: pd.DataFrame,
                     rates: dict = None) -> Optional[dict]:
        """Score a pair on carry and momentum dimensions."""
        base = meta.get("base", "")
        quote = meta.get("quote", "")
        latest = df.iloc[-1]

        # ── CARRY SCORE ──
        active_rates = rates or POLICY_RATES
        base_rate = active_rates.get(base, 0)
        quote_rate = active_rates.get(quote, 0)
        carry_differential = base_rate - quote_rate

        if abs(carry_differential) < self.min_carry_spread:
            carry_score = 0
        else:
            # Positive carry → long pair (buy base, sell quote)
            # Negative carry → short pair
            carry_score = carry_differential * 10  # Scale: 2% diff → 0.2 score

        # ── MOMENTUM SCORE ──
        mom_1m = latest.get("momentum_1m", 0)
        mom_3m = latest.get("momentum_3m", 0)

        if np.isnan(mom_1m):
            mom_1m = 0
        if np.isnan(mom_3m):
            mom_3m = 0

        # Momentum confirms carry direction?
        momentum_score = 0.6 * mom_1m + 0.4 * mom_3m
        # Scale momentum to be comparable to carry
        momentum_score = momentum_score * 5

        # ── TREND FILTER ──
        trend_aligned = True
        if carry_differential > 0:
            # We want to go long → price should be above SMA
            if latest["close"] < latest.get(f"sma_{self.trend_filter_sma}", latest["close"]):
                trend_aligned = False
        elif carry_differential < 0:
            # We want to go short → price should be below SMA
            if latest["close"] > latest.get(f"sma_{self.trend_filter_sma}", latest["close"]):
                trend_aligned = False

        # ── COMPOSITE ──
        composite = (
            self.carry_weight * carry_score +
            self.momentum_weight * momentum_score
        )

        # Penalize if trend is against us
        if not trend_aligned:
            composite *= 0.5

        return {
            "carry_differential": carry_differential,
            "carry_score": carry_score,
            "momentum_score": momentum_score,
            "trend_aligned": trend_aligned,
            "composite_score": composite,
            "base_rate": base_rate,
            "quote_rate": quote_rate,
        }

    def generate_signal_sync(self, symbol: str, instrument_id: int,
                              df: pd.DataFrame) -> Optional[TradeSignal]:
        """Synchronous signal for backtester compatibility."""
        from config.instruments import FX_PAIRS

        meta = FX_PAIRS.get(symbol)
        if meta is None:
            return None
        if instrument_id == 0:
            instrument_id = meta.get("etoro_id", 0)
        if len(df) < 60:
            return None

        # Use historical rates for the current backtest date
        date_str = str(df.index[-1])[:10]
        rates = get_rates_for_date(date_str)
        score_data = self._score_pair(symbol, meta, df, rates=rates)
        if not score_data:
            return None

        composite = score_data["composite_score"]
        if abs(composite) < 0.08:
            return None

        latest = df.iloc[-1]
        entry = latest["close"]
        atr = latest.get("atr", entry * 0.01)
        if pd.isna(atr) or atr == 0:
            atr = entry * 0.01

        if composite > 0:
            direction = Signal.BUY if composite < 0.15 else Signal.STRONG_BUY
            stop_loss = entry - (self.atr_stop_multiplier * atr)
            take_profit = entry + (self.tp_multiplier * self.atr_stop_multiplier * atr)
        else:
            direction = Signal.SELL if abs(composite) < 0.15 else Signal.STRONG_SELL
            stop_loss = entry + (self.atr_stop_multiplier * atr)
            take_profit = entry - (self.tp_multiplier * self.atr_stop_multiplier * atr)

        confidence = min(0.90, 0.4 + abs(composite) * 0.3)
        if score_data["trend_aligned"]:
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
            suggested_size_pct=self.max_risk_per_pair,
            metadata={
                "carry_differential": score_data["carry_differential"],
                "composite_score": composite,
                "trend_aligned": score_data["trend_aligned"],
            },
        )

    def _generate_trade_signals(self, scored_pairs: List[dict]) -> List[TradeSignal]:
        """Generate trade signals from scored pairs."""
        signals = []

        for pair_data in scored_pairs[:self.max_positions]:
            df = pair_data["df"]
            latest = df.iloc[-1]
            composite = pair_data["composite_score"]

            if abs(composite) < 0.08:
                continue  # Not enough edge

            # Direction based on composite score
            if composite > 0:
                direction = Signal.BUY
            else:
                direction = Signal.SELL

            entry = latest["close"]
            atr = latest["atr"]

            if direction == Signal.BUY:
                stop_loss = entry - (self.atr_stop_multiplier * atr)
                take_profit = entry + (self.tp_multiplier * self.atr_stop_multiplier * atr)
            else:
                stop_loss = entry + (self.atr_stop_multiplier * atr)
                take_profit = entry - (self.tp_multiplier * self.atr_stop_multiplier * atr)

            # Confidence based on score magnitude and trend alignment
            confidence = min(0.90, 0.4 + abs(composite) * 0.3)
            if pair_data["trend_aligned"]:
                confidence = min(0.95, confidence + 0.15)

            signals.append(TradeSignal(
                symbol=pair_data["pair"],
                instrument_id=pair_data["instrument_id"],
                signal=direction if abs(composite) < 0.15 else (
                    Signal.STRONG_BUY if composite > 0 else Signal.STRONG_SELL
                ),
                strategy_name=self.name,
                confidence=confidence,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                suggested_size_pct=self.max_risk_per_pair,
                metadata={
                    "carry_differential": pair_data["carry_differential"],
                    "base_rate": pair_data["base_rate"],
                    "quote_rate": pair_data["quote_rate"],
                    "carry_score": pair_data["carry_score"],
                    "momentum_score": pair_data["momentum_score"],
                    "trend_aligned": pair_data["trend_aligned"],
                    "composite_score": composite,
                    "atr": float(atr),
                    "rsi": float(latest.get("rsi", 50)),
                },
            ))

        return self.filter_signals(signals, min_confidence=0.5, min_rr_ratio=1.0)

    def should_exit(self, position: dict, current_data: pd.DataFrame) -> Optional[TradeSignal]:
        """Exit on trend reversal or carry inversion."""
        if current_data.empty:
            return None

        latest = current_data.iloc[-1]

        # Exit if trend has reversed against position
        direction = position.get("direction", "Buy")
        sma50 = latest.get("sma_50", latest["close"])

        if direction == "Buy" and latest["close"] < sma50 and latest.get("rsi", 50) < 40:
            return TradeSignal(
                symbol=position.get("symbol", ""),
                instrument_id=position.get("instrumentId", 0),
                signal=Signal.SELL,
                strategy_name=self.name,
                confidence=0.7,
                entry_price=latest["close"],
                stop_loss=latest["close"],
                take_profit=latest["close"],
                suggested_size_pct=1.0,
                metadata={"exit_reason": "Trend reversal against long carry"},
            )

        if direction == "Sell" and latest["close"] > sma50 and latest.get("rsi", 50) > 60:
            return TradeSignal(
                symbol=position.get("symbol", ""),
                instrument_id=position.get("instrumentId", 0),
                signal=Signal.BUY,
                strategy_name=self.name,
                confidence=0.7,
                entry_price=latest["close"],
                stop_loss=latest["close"],
                take_profit=latest["close"],
                suggested_size_pct=1.0,
                metadata={"exit_reason": "Trend reversal against short carry"},
            )

        return None
