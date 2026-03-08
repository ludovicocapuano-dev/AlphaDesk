"""
AlphaDesk — Post-Earnings Announcement Drift (PEAD) Strategy
Trades earnings surprises, holds ~60 days.

Based on Ball & Brown (1968), Bernard & Thomas (1989):
After earnings surprises, stocks tend to drift in the direction
of the surprise for approximately 60 trading days.
Positive surprise -> long drift. Negative surprise -> short drift.

Allocation: 10% — US Equities only
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from strategies.base_strategy import BaseStrategy, Signal, TradeSignal

logger = logging.getLogger("alphadesk.strategy.pead")


class PEADStrategy(BaseStrategy):
    """
    Post-Earnings Announcement Drift.
    Trades earnings surprises, holds ~60 days.

    Entry: After earnings report with |surprise| > threshold
    Direction: Long if surprise > 0, Short if surprise < 0
    Exit: 60 trading days or stop loss hit
    """

    def __init__(self, allocation_pct: float = 0.10):
        super().__init__(
            name="pead",
            allocation_pct=allocation_pct,
            max_positions=5,
        )
        # Parameters
        self.min_surprise_pct = 3.0      # Minimum 3% earnings surprise to trigger
        self.holding_days = 60           # Hold for ~60 trading days
        self.stop_loss_pct = 0.08        # 8% stop loss
        self.lookback_days = 5           # Scan earnings reported in last 5 trading days

        # Cache earnings data within a single scan cycle
        self._earnings_cache: Dict[str, Optional[pd.DataFrame]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)

    def _invalidate_cache_if_stale(self):
        """Clear cache if it's older than the TTL."""
        if self._cache_timestamp is None:
            return
        if datetime.utcnow() - self._cache_timestamp > self._cache_ttl:
            self._earnings_cache.clear()
            self._cache_timestamp = None
            logger.debug("Earnings cache invalidated (stale)")

    def _fetch_earnings(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch earnings dates for a symbol, using cache."""
        if symbol in self._earnings_cache:
            return self._earnings_cache[symbol]

        try:
            ticker = yf.Ticker(symbol)
            earnings = ticker.earnings_dates
            if earnings is not None and not earnings.empty:
                self._earnings_cache[symbol] = earnings
            else:
                self._earnings_cache[symbol] = None
            return self._earnings_cache[symbol]
        except Exception as e:
            logger.warning(f"Failed to fetch earnings for {symbol}: {e}")
            self._earnings_cache[symbol] = None
            return None

    async def generate_signals(self, data_engine, current_positions: List[dict]) -> List[TradeSignal]:
        """Scan US equities for recent earnings surprises."""
        from config.instruments import US_EQUITIES

        # Invalidate stale cache at the start of each scan
        self._invalidate_cache_if_stale()
        if self._cache_timestamp is None:
            self._cache_timestamp = datetime.utcnow()

        signals = []
        current_symbols = {p.get("symbol") for p in current_positions}
        now = datetime.utcnow()
        cutoff = now - timedelta(days=self.lookback_days + 2)  # calendar days buffer

        for symbol, meta in US_EQUITIES.items():
            instrument_id = meta.get("etoro_id")
            if instrument_id is None:
                continue

            # Skip if we already have a position in this symbol
            if symbol in current_symbols:
                continue

            try:
                earnings_df = self._fetch_earnings(symbol)
                if earnings_df is None:
                    continue

                signal = self._evaluate_earnings(
                    symbol, instrument_id, earnings_df, cutoff, now,
                )
                if signal:
                    signals.append(signal)

            except Exception as e:
                logger.error(f"Error evaluating PEAD for {symbol}: {e}")

        self.log_signals(signals)
        return self.filter_signals(signals, min_confidence=0.5, min_rr_ratio=1.0)

    def _evaluate_earnings(
        self,
        symbol: str,
        instrument_id: int,
        earnings_df: pd.DataFrame,
        cutoff: datetime,
        now: datetime,
    ) -> Optional[TradeSignal]:
        """Check if a symbol has a recent qualifying earnings surprise."""
        # earnings_dates index is a DatetimeIndex of report dates
        # Columns typically include: 'Surprise(%)' (or 'Earnings Surprise(%)'),
        # 'Reported EPS', 'EPS Estimate'

        # Normalize column names — yfinance format can vary
        surprise_col = None
        for col in earnings_df.columns:
            if "surprise" in col.lower() and "%" in col:
                surprise_col = col
                break

        if surprise_col is None:
            logger.debug(f"{symbol}: No surprise column found in earnings data")
            return None

        # Filter to earnings reported within the lookback window
        # Index is timezone-aware in some yfinance versions
        for report_date in earnings_df.index:
            try:
                report_dt = report_date.to_pydatetime().replace(tzinfo=None)
            except Exception:
                continue

            # Only consider past earnings within the lookback window
            if report_dt > now or report_dt < cutoff:
                continue

            row = earnings_df.loc[report_date]
            surprise_pct = row.get(surprise_col)

            if surprise_pct is None or pd.isna(surprise_pct):
                continue

            surprise_pct = float(surprise_pct)

            # Check if surprise exceeds threshold
            if abs(surprise_pct) < self.min_surprise_pct:
                continue

            # We have a qualifying surprise — build the signal
            return self._build_signal(symbol, instrument_id, surprise_pct, report_dt)

        return None

    def _build_signal(
        self,
        symbol: str,
        instrument_id: int,
        surprise_pct: float,
        report_date: datetime,
    ) -> Optional[TradeSignal]:
        """Build a TradeSignal from an earnings surprise."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if hist.empty:
                logger.warning(f"{symbol}: No recent price data")
                return None
            entry_price = float(hist["Close"].iloc[-1])
        except Exception as e:
            logger.warning(f"{symbol}: Failed to get current price: {e}")
            return None

        is_positive = surprise_pct > 0

        # Direction
        if is_positive:
            signal_type = Signal.STRONG_BUY if surprise_pct > 10.0 else Signal.BUY
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            tp_pct = max(0.10, abs(surprise_pct) * 2.0 / 100.0)
            take_profit = entry_price * (1 + tp_pct)
        else:
            signal_type = Signal.STRONG_SELL if surprise_pct < -10.0 else Signal.SELL
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            tp_pct = max(0.10, abs(surprise_pct) * 2.0 / 100.0)
            take_profit = entry_price * (1 - tp_pct)

        # Confidence: base 0.5 + magnitude bonus, capped at 0.9
        confidence = min(0.9, 0.5 + abs(surprise_pct) * 0.05)

        # Position sizing: scale with confidence, max 5% per position
        suggested_size = min(0.05, confidence * 0.05)

        return TradeSignal(
            symbol=symbol,
            instrument_id=instrument_id,
            signal=signal_type,
            strategy_name=self.name,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            suggested_size_pct=suggested_size,
            metadata={
                "surprise_pct": surprise_pct,
                "report_date": report_date.isoformat(),
                "holding_days": self.holding_days,
                "direction": "long" if is_positive else "short",
                "tp_pct": tp_pct,
            },
        )

    def should_exit(self, position: dict, current_data: pd.DataFrame) -> Optional[TradeSignal]:
        """Exit after holding_days elapsed or stop loss hit."""
        if current_data.empty:
            return None

        latest = current_data.iloc[-1]
        current_price = latest["close"]
        entry_price = position.get("openRate", position.get("entry_price", 0))

        if entry_price <= 0:
            return None

        # ── Time-based exit: holding period expired ──
        open_date_str = position.get("openDateTime", position.get("open_date"))
        if open_date_str:
            try:
                if isinstance(open_date_str, str):
                    open_date = datetime.fromisoformat(open_date_str.replace("Z", "+00:00"))
                    open_date = open_date.replace(tzinfo=None)
                else:
                    open_date = open_date_str
                days_held = (datetime.utcnow() - open_date).days
                if days_held >= self.holding_days:
                    return self._create_exit_signal(
                        position, latest,
                        f"Holding period expired ({days_held} days)",
                    )
            except Exception as e:
                logger.warning(f"Could not parse open date: {e}")

        # ── Stop loss exit ──
        is_long = position.get("isBuy", True)
        if is_long:
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct <= -self.stop_loss_pct:
                return self._create_exit_signal(
                    position, latest,
                    f"Stop loss hit ({pnl_pct:.1%})",
                )
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            if pnl_pct <= -self.stop_loss_pct:
                return self._create_exit_signal(
                    position, latest,
                    f"Stop loss hit ({pnl_pct:.1%})",
                )

        return None

    def _create_exit_signal(self, position: dict, latest: pd.Series, reason: str) -> TradeSignal:
        """Build an exit signal for a position."""
        logger.info(f"[pead] Exit {position.get('symbol', '?')}: {reason}")
        return TradeSignal(
            symbol=position.get("symbol", ""),
            instrument_id=position.get("instrumentId", 0),
            signal=Signal.SELL,
            strategy_name=self.name,
            confidence=0.8,
            entry_price=latest["close"],
            stop_loss=latest["close"],
            take_profit=latest["close"],
            suggested_size_pct=1.0,  # Close full position
            metadata={"exit_reason": reason},
        )
