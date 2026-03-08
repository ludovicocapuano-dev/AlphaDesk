"""
AlphaDesk — Multi-Factor Strategy
Fama-French inspired factor scoring: Value, Quality, Momentum.
Allocation: 20% — Equities US/EU, monthly rebalance.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal, TradeSignal

logger = logging.getLogger("alphadesk.strategy.factor")


class FactorModelStrategy(BaseStrategy):
    """
    Multi-factor equity strategy:
    - Value: low P/E, low P/B (buy cheap)
    - Quality: high ROE, low debt/equity (buy strong)
    - Momentum: 12-1 month returns (buy winners)

    Monthly rebalance. Long top quintile, avoid bottom quintile.
    """

    def __init__(self, allocation_pct: float = 0.20):
        super().__init__(
            name="factor_model",
            allocation_pct=allocation_pct,
            max_positions=10,  # Diversified portfolio
        )
        # Factor weights (base — adjusted dynamically by regime)
        self.value_weight = 0.35
        self.quality_weight = 0.30
        self.momentum_weight = 0.35

        # Quality tilt when VIX > 25 (risk-off: quality outperforms)
        self._quality_tilt_weights = {
            "value": 0.25,
            "quality": 0.50,  # Double quality weight in risk-off
            "momentum": 0.25,
        }

        # Rebalance tracking
        self._last_rebalance = None
        self.rebalance_days = 21  # ~monthly

    async def generate_signals(self, data_engine, current_positions: List[dict]) -> List[TradeSignal]:
        """Score all stocks and generate signals for top quintile."""
        from config.instruments import US_EQUITIES, EU_EQUITIES

        # Quality factor tilt: detect high-vol regime from universe data
        self._apply_regime_tilt(data_engine)

        universe = {**US_EQUITIES, **EU_EQUITIES}
        scored_stocks = []

        for symbol, meta in universe.items():
            instrument_id = meta.get("etoro_id")
            if instrument_id is None:
                continue

            try:
                # Fetch price data for momentum
                df = await data_engine.get_ohlcv(instrument_id, symbol, "OneDay", 280)
                if df.empty or len(df) < 252:
                    continue

                df = data_engine.compute_indicators(df)

                # Fetch fundamentals via yfinance fallback
                fundamentals = self._get_fundamentals(symbol, data_engine)

                # Compute factor scores
                factor_score = self._compute_composite_score(df, fundamentals)
                if factor_score is not None:
                    scored_stocks.append({
                        "symbol": symbol,
                        "instrument_id": instrument_id,
                        "score": factor_score,
                        "factors": factor_score,
                        "price": df.iloc[-1]["close"],
                        "atr": df.iloc[-1].get("atr", 0),
                        "df": df,
                    })

            except Exception as e:
                logger.error(f"Factor scoring error for {symbol}: {e}")

        if not scored_stocks:
            return []

        # Rank and select top quintile
        return self._rank_and_select(scored_stocks)

    def _apply_regime_tilt(self, data_engine):
        """Tilt factor weights toward quality when volatility is high (VIX > 25)."""
        try:
            # Check if regime detector flagged high vol
            vol_regime = getattr(data_engine, '_last_vol_regime', None)
            if vol_regime in ("high", "extreme"):
                self.value_weight = self._quality_tilt_weights["value"]
                self.quality_weight = self._quality_tilt_weights["quality"]
                self.momentum_weight = self._quality_tilt_weights["momentum"]
                logger.info("Quality tilt active: risk-off regime detected")
            else:
                # Reset to defaults
                self.value_weight = 0.35
                self.quality_weight = 0.30
                self.momentum_weight = 0.35
        except Exception:
            pass  # Use default weights on error

    def _get_fundamentals(self, symbol: str, data_engine) -> dict:
        """Fetch fundamental data for factor scoring."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "debt_equity": info.get("debtToEquity"),
                "profit_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "dividend_yield": info.get("dividendYield", 0),
                "market_cap": info.get("marketCap"),
            }
        except Exception as e:
            logger.debug(f"Fundamentals unavailable for {symbol}: {e}")
            return {}

    def _compute_composite_score(self, df: pd.DataFrame, fundamentals: dict) -> Optional[dict]:
        """Compute composite factor score for a stock."""
        scores = {}

        # ── VALUE FACTOR ──
        pe = fundamentals.get("pe_ratio")
        pb = fundamentals.get("pb_ratio")

        if pe is not None and pe > 0:
            # Lower P/E is better (inverted score)
            scores["pe_score"] = max(0, 1 - (pe / 50))  # Normalized: PE=50 → 0, PE=5 → 0.9
        if pb is not None and pb > 0:
            scores["pb_score"] = max(0, 1 - (pb / 10))

        value_factors = [v for k, v in scores.items() if k.endswith("_score") and "pe" in k or "pb" in k]
        scores["value"] = np.mean(value_factors) if value_factors else 0.5

        # ── QUALITY FACTOR ──
        roe = fundamentals.get("roe")
        de = fundamentals.get("debt_equity")
        margin = fundamentals.get("profit_margin")

        quality_components = []
        if roe is not None:
            quality_components.append(min(1.0, max(0, roe)))  # ROE as decimal
        if de is not None and de >= 0:
            quality_components.append(max(0, 1 - (de / 300)))  # Lower D/E is better
        if margin is not None:
            quality_components.append(min(1.0, max(0, margin)))

        scores["quality"] = np.mean(quality_components) if quality_components else 0.5

        # ── MOMENTUM FACTOR ──
        latest = df.iloc[-1]
        mom_12m = latest.get("momentum_12m", 0)
        mom_1m = latest.get("momentum_1m", 0)
        # 12-1 month momentum: P(t-21)/P(t-252) - 1 (skip last month)
        if not np.isnan(mom_12m) and not np.isnan(mom_1m):
            mom_12_1 = (1 + mom_12m) / max(1 + mom_1m, 0.01) - 1
        else:
            mom_12_1 = 0

        # Normalize momentum to 0-1
        scores["momentum"] = self._sigmoid(mom_12_1, center=0, scale=0.3)

        # ── COMPOSITE ──
        scores["composite"] = (
            self.value_weight * scores["value"] +
            self.quality_weight * scores["quality"] +
            self.momentum_weight * scores["momentum"]
        )

        return scores

    def _rank_and_select(self, scored_stocks: List[dict]) -> List[TradeSignal]:
        """Rank stocks and generate signals for top quintile.
        Uses denoised covariance (Marcenko-Pastur) for risk-adjusted weighting.
        """
        from core.data_engine import DataEngine

        # Sort by composite score
        scored_stocks.sort(key=lambda x: x["score"]["composite"], reverse=True)

        # Top quintile (top 20%)
        n_select = max(1, len(scored_stocks) // 5)
        top_stocks = scored_stocks[:n_select]

        # Risk-adjusted weighting via denoised covariance
        if len(top_stocks) >= 3:
            returns_dict = {}
            for stock in top_stocks:
                df = stock.get("df")
                if df is not None and len(df) > 60:
                    returns_dict[stock["symbol"]] = df["close"].pct_change().dropna().tail(60)

            if len(returns_dict) >= 3:
                returns_df = pd.DataFrame(returns_dict).dropna()
                if len(returns_df) > 20:
                    try:
                        denoised_cov = DataEngine.denoise_covariance(returns_df)
                        # Inverse-variance weighting from denoised matrix
                        inv_var = 1.0 / np.diag(denoised_cov.values).clip(1e-8)
                        raw_weights = inv_var / inv_var.sum()
                        weight_map = dict(zip(returns_df.columns, raw_weights))
                    except Exception as e:
                        logger.debug(f"Denoised cov failed, using equal weight: {e}")
                        weight_map = {}
                else:
                    weight_map = {}
            else:
                weight_map = {}
        else:
            weight_map = {}

        signals = []
        equal_weight = 1.0 / len(top_stocks) if top_stocks else 0

        for stock in top_stocks:
            price = stock["price"]
            atr = stock["atr"]
            composite = stock["score"]["composite"]

            # Use denoised inverse-variance weight if available, else equal weight
            w = weight_map.get(stock["symbol"], equal_weight)

            # Stop loss: wider for factor strategy (longer holding)
            stop_loss = price * (1 - 0.08)   # 8% max loss
            take_profit = price * (1 + 0.15)  # 15% target

            signal_type = Signal.STRONG_BUY if composite > 0.7 else Signal.BUY

            signals.append(TradeSignal(
                symbol=stock["symbol"],
                instrument_id=stock["instrument_id"],
                signal=signal_type,
                strategy_name=self.name,
                confidence=min(0.95, composite),
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                suggested_size_pct=w * self.allocation_pct,
                metadata={
                    "factor_scores": stock["score"],
                    "rank": scored_stocks.index(stock) + 1,
                    "total_universe": len(scored_stocks),
                    "denoised_weight": w,
                },
            ))

        return signals[:self.max_positions]

    def should_exit(self, position: dict, current_data: pd.DataFrame) -> Optional[TradeSignal]:
        """Factor strategy: exit on monthly rebalance or fundamental deterioration."""
        # Factor positions are managed via rebalance, not individual exits
        # However, hard stop at -10% to protect capital
        if current_data.empty:
            return None

        entry_price = position.get("openRate", 0)
        current_price = current_data.iloc[-1]["close"]

        if entry_price > 0:
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct < -0.10:
                return TradeSignal(
                    symbol=position.get("symbol", ""),
                    instrument_id=position.get("instrumentId", 0),
                    signal=Signal.SELL,
                    strategy_name=self.name,
                    confidence=0.9,
                    entry_price=current_price,
                    stop_loss=current_price,
                    take_profit=current_price,
                    suggested_size_pct=1.0,
                    metadata={"exit_reason": f"Hard stop hit: {pnl_pct:.1%}"},
                )

        return None

    @staticmethod
    def _sigmoid(x: float, center: float = 0, scale: float = 1) -> float:
        """Sigmoid normalization to 0-1."""
        return 1.0 / (1.0 + np.exp(-(x - center) / scale))
