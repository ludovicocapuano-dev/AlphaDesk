"""
AlphaDesk — Mean Reversion Strategy
Z-score based + Pairs Trading with cointegration.
Allocation: 20% — Equities US/EU
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal, TradeSignal

logger = logging.getLogger("alphadesk.strategy.mean_reversion")


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion using:
    1. Single-name Z-score reversion (oversold/overbought)
    2. Pairs trading on cointegrated stocks

    Entry: Z-score < -2 (buy) or Z-score > +2 (sell)
    Exit: Z-score returns to 0 (mean), or hits stop
    Holding: 1-5 days typical
    """

    def __init__(self, allocation_pct: float = 0.20):
        super().__init__(
            name="mean_reversion",
            allocation_pct=allocation_pct,
            max_positions=6,
        )
        # Single-name params
        self.z_entry_long = -2.0    # Buy when Z < -2
        self.z_entry_short = 2.0    # Sell when Z > +2
        self.z_exit = 0.3           # Exit when Z approaches 0
        self.z_stop = 3.5           # Stop loss at extreme Z

        # Pairs trading params
        self.coint_pvalue = 0.05    # Cointegration p-value threshold
        self.pairs_z_entry = 2.0    # Z-score for pair spread
        self.pairs_z_exit = 0.5

        # Filters
        self.min_lookback = 60       # Min days for Z-score calculation
        self.bb_squeeze_threshold = 0.03  # Bollinger Band width for squeeze

    async def generate_signals(self, data_engine, current_positions: List[dict]) -> List[TradeSignal]:
        """Generate mean reversion and pairs trading signals."""
        signals = []

        # 1. Single-name mean reversion
        single_signals = await self._scan_single_name(data_engine)
        signals.extend(single_signals)

        # 2. Pairs trading
        pairs_signals = await self._scan_pairs(data_engine)
        signals.extend(pairs_signals)

        return self.filter_signals(signals, min_confidence=0.55, min_rr_ratio=1.2)

    async def _scan_single_name(self, data_engine) -> List[TradeSignal]:
        """Scan for single-name mean reversion opportunities."""
        from config.instruments import US_EQUITIES, EU_EQUITIES

        signals = []
        universe = {**US_EQUITIES, **EU_EQUITIES}

        for symbol, meta in universe.items():
            instrument_id = meta.get("etoro_id")
            if instrument_id is None:
                continue

            try:
                df = await data_engine.get_ohlcv(instrument_id, symbol, "OneDay", 120)
                if df.empty or len(df) < self.min_lookback:
                    continue

                df = data_engine.compute_indicators(df)
                signal = self._evaluate_zscore(symbol, instrument_id, df)
                if signal:
                    signals.append(signal)

            except Exception as e:
                logger.error(f"MeanRev error for {symbol}: {e}")

        return signals

    def _evaluate_zscore(self, symbol: str, instrument_id: int,
                          df: pd.DataFrame) -> Optional[TradeSignal]:
        """Evaluate Z-score for mean reversion entry."""
        latest = df.iloc[-1]
        zscore = latest.get("zscore", 0)
        rsi = latest.get("rsi", 50)

        # ── Long signal: oversold ──
        if zscore < self.z_entry_long:
            # Confirm with RSI divergence
            if rsi < 35:
                confidence = self._mr_confidence(zscore, rsi, df, direction="long")

                entry = latest["close"]
                atr = latest["atr"]
                stop_loss = entry - (1.5 * atr)   # Tighter stop for MR
                # Target: return to mean (SMA20)
                take_profit = latest["sma_20"]

                return TradeSignal(
                    symbol=symbol,
                    instrument_id=instrument_id,
                    signal=Signal.BUY if confidence < 0.8 else Signal.STRONG_BUY,
                    strategy_name=self.name,
                    confidence=confidence,
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    suggested_size_pct=min(0.04, confidence * 0.05),
                    metadata={
                        "zscore": zscore,
                        "rsi": rsi,
                        "type": "single_name_long",
                        "target_sma20": latest["sma_20"],
                        "bb_width": latest.get("bb_width", 0),
                    },
                )

        # ── Short signal: overbought ──
        elif zscore > self.z_entry_short:
            if rsi > 65:
                confidence = self._mr_confidence(zscore, rsi, df, direction="short")

                entry = latest["close"]
                atr = latest["atr"]
                stop_loss = entry + (1.5 * atr)
                take_profit = latest["sma_20"]

                return TradeSignal(
                    symbol=symbol,
                    instrument_id=instrument_id,
                    signal=Signal.SELL if confidence < 0.8 else Signal.STRONG_SELL,
                    strategy_name=self.name,
                    confidence=confidence,
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    suggested_size_pct=min(0.04, confidence * 0.05),
                    metadata={
                        "zscore": zscore,
                        "rsi": rsi,
                        "type": "single_name_short",
                        "target_sma20": latest["sma_20"],
                    },
                )

        return None

    async def _scan_pairs(self, data_engine) -> List[TradeSignal]:
        """Scan for pairs trading opportunities using cointegration."""
        from config.instruments import US_EQUITIES

        signals = []
        # Group by sector for sector-neutral pairs
        sector_groups: Dict[str, list] = {}
        for symbol, meta in US_EQUITIES.items():
            sector = meta.get("sector", "Other")
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append((symbol, meta))

        for sector, stocks in sector_groups.items():
            if len(stocks) < 2:
                continue

            # Test all pairs within sector
            price_data = {}
            for symbol, meta in stocks:
                instrument_id = meta.get("etoro_id")
                if instrument_id is None:
                    continue
                df = await data_engine.get_ohlcv(instrument_id, symbol, "OneDay", 120)
                if not df.empty and len(df) >= self.min_lookback:
                    price_data[symbol] = df["close"]

            for sym_a, sym_b in combinations(price_data.keys(), 2):
                try:
                    result = data_engine.test_cointegration(
                        price_data[sym_a], price_data[sym_b]
                    )
                    if result["cointegrated"]:
                        pair_signal = self._evaluate_pair_spread(
                            sym_a, sym_b, price_data[sym_a], price_data[sym_b],
                            US_EQUITIES, result["p_value"]
                        )
                        if pair_signal:
                            signals.append(pair_signal)
                except Exception as e:
                    logger.debug(f"Pairs error {sym_a}/{sym_b}: {e}")

        return signals

    def _evaluate_pair_spread(self, sym_a: str, sym_b: str,
                                prices_a: pd.Series, prices_b: pd.Series,
                                universe: dict, coint_pvalue: float) -> Optional[TradeSignal]:
        """Evaluate spread Z-score for a cointegrated pair."""
        # Calculate spread ratio
        ratio = prices_a / prices_b
        spread_mean = ratio.rolling(60).mean()
        spread_std = ratio.rolling(60).std()
        zscore = ((ratio - spread_mean) / spread_std).iloc[-1]

        if abs(zscore) < self.pairs_z_entry:
            return None

        # Determine direction
        if zscore > self.pairs_z_entry:
            # Spread too wide: short A, long B
            direction = Signal.SELL
            primary_sym = sym_a
        else:
            # Spread too narrow: long A, short B
            direction = Signal.BUY
            primary_sym = sym_a

        entry = float(prices_a.iloc[-1])
        atr_approx = float(prices_a.pct_change().std() * prices_a.iloc[-1] * np.sqrt(14))

        return TradeSignal(
            symbol=f"{sym_a}/{sym_b}",
            instrument_id=universe.get(primary_sym, {}).get("etoro_id", 0),
            signal=direction,
            strategy_name=self.name,
            confidence=min(0.85, 0.5 + abs(zscore) * 0.1 + (1 - coint_pvalue) * 0.2),
            entry_price=entry,
            stop_loss=entry * (1 - 0.03) if direction == Signal.BUY else entry * (1 + 0.03),
            take_profit=entry * (1 + 0.02) if direction == Signal.BUY else entry * (1 - 0.02),
            suggested_size_pct=0.03,
            metadata={
                "type": "pairs_trade",
                "pair": f"{sym_a}/{sym_b}",
                "spread_zscore": float(zscore),
                "coint_pvalue": coint_pvalue,
            },
        )

    def _mr_confidence(self, zscore: float, rsi: float,
                        df: pd.DataFrame, direction: str) -> float:
        """Compute confidence for mean reversion signal."""
        score = 0.0
        latest = df.iloc[-1]

        # Z-score extremity
        z_abs = abs(zscore)
        if z_abs >= 3.0:
            score += 0.30
        elif z_abs >= 2.5:
            score += 0.20
        else:
            score += 0.10

        # RSI confirmation
        if direction == "long" and rsi < 25:
            score += 0.20
        elif direction == "long" and rsi < 35:
            score += 0.10
        elif direction == "short" and rsi > 75:
            score += 0.20
        elif direction == "short" and rsi > 65:
            score += 0.10

        # Bollinger Band touch/breach
        if direction == "long" and latest["close"] <= latest.get("bb_lower", 0):
            score += 0.15
        elif direction == "short" and latest["close"] >= latest.get("bb_upper", float("inf")):
            score += 0.15

        # Volume spike (capitulation/exhaustion)
        vr = latest.get("volume_ratio", 1)
        if vr >= 2.0:
            score += 0.15
        elif vr >= 1.5:
            score += 0.08

        # Historical mean reversion success rate at this Z level
        historical_z = df.get("zscore", pd.Series()).dropna()
        if len(historical_z) > 60:
            if direction == "long":
                past_entries = historical_z[historical_z < self.z_entry_long]
            else:
                past_entries = historical_z[historical_z > self.z_entry_short]
            if len(past_entries) > 3:
                score += 0.10  # Pattern has occurred before

        return min(1.0, score)

    def should_exit(self, position: dict, current_data: pd.DataFrame) -> Optional[TradeSignal]:
        """Exit when Z-score reverts to mean."""
        if current_data.empty:
            return None

        latest = current_data.iloc[-1]
        zscore = latest.get("zscore", 0)

        # Exit when Z-score returns near zero
        if abs(zscore) < self.z_exit:
            return TradeSignal(
                symbol=position.get("symbol", ""),
                instrument_id=position.get("instrumentId", 0),
                signal=Signal.NEUTRAL,
                strategy_name=self.name,
                confidence=0.8,
                entry_price=latest["close"],
                stop_loss=latest["close"],
                take_profit=latest["close"],
                suggested_size_pct=1.0,
                metadata={"exit_reason": "Mean reversion target reached", "zscore": zscore},
            )

        return None
