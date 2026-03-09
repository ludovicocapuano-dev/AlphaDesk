"""
AlphaDesk — Main Orchestrator (v2 — Self-Learning Ensemble)
Coordinates strategies, risk management, ML ensemble, and execution.

v2 additions:
- RegimeDetector: fingerprints market state for every decision
- OutcomeLabeler: backfills PnL at 15m/1h/4h/24h horizons
- MLEnsemble: PyTorch meta-model that learns when strategies work
- Feature vectors logged with every signal for closed-loop learning
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import List

from config.settings import config
from core.etoro_client import EtoroClient
from core.data_engine import DataEngine
from core.regime_detector import RegimeDetector
from core.outcome_labeler import OutcomeLabeler
from core.ml_ensemble import MLEnsemble
try:
    from core.meta_labeler import MetaLabeler
    HAS_META_LABELER = True
except ImportError:
    HAS_META_LABELER = False
from risk.portfolio_risk import PortfolioRiskManager
from risk.position_sizer import PositionSizer
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.factor_model import FactorModelStrategy
from strategies.fx_carry import FXCarryStrategy
try:
    from strategies.pead import PEADStrategy
    HAS_PEAD = True
except ImportError:
    HAS_PEAD = False
from strategies.base_strategy import TradeSignal
from utils.db import TradeDB
from utils.logger import setup_logging
from utils.telegram_bot import TelegramNotifier

logger = logging.getLogger("alphadesk.main")


class AlphaDesk:
    """Main trading system orchestrator with self-learning ML ensemble."""

    def __init__(self):
        # Core components
        self.etoro = EtoroClient(
            user_key=config.etoro.user_key,
            api_key=config.etoro.api_key,
            base_url=config.etoro.base_url,
            environment=config.etoro.environment,
            timeout=config.etoro.request_timeout,
            max_retries=config.etoro.max_retries,
        )
        self.data_engine = DataEngine(self.etoro)
        self.risk_manager = PortfolioRiskManager(config.risk)
        self.position_sizer = PositionSizer(
            max_risk_per_trade=config.risk.max_risk_per_trade,
            kelly_fraction=config.risk.kelly_fraction,
        )
        self.db = TradeDB(config.db_path)
        self.notifier = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id,
            enabled=config.telegram.enabled,
        )

        # ── v2: Self-Learning Components ──
        model_dir = os.path.join(os.path.dirname(config.db_path), "models")
        self.regime_detector = RegimeDetector()
        self.outcome_labeler = OutcomeLabeler(config.db_path)
        self.ml_ensemble = MLEnsemble(model_dir=model_dir)

        # Meta-labeler (López de Prado): per-strategy false positive filter
        if HAS_META_LABELER:
            self.meta_labeler = MetaLabeler(model_dir=model_dir)
        else:
            self.meta_labeler = None

        # Cache current regime fingerprint
        self._current_regime = None

        # News sentiment (VADER + RSS, lightweight)
        try:
            from core.news_sentiment import NewsSentiment
            self.news_sentiment = NewsSentiment()
        except ImportError:
            self.news_sentiment = None

        # Strategies
        self.strategies = [
            MomentumStrategy(config.allocation.momentum),
            MeanReversionStrategy(config.allocation.mean_reversion),
            FactorModelStrategy(config.allocation.factor_model),
            FXCarryStrategy(config.allocation.fx_carry),
        ]
        if HAS_PEAD:
            self.strategies.append(PEADStrategy(allocation_pct=0.10))

    async def initialize(self):
        """Initialize: load instrument IDs, update portfolio state."""
        logger.info("=" * 60)
        logger.info("AlphaDesk v2 initializing (Self-Learning Ensemble)")
        logger.info(f"Environment: {config.etoro.environment}")
        logger.info(f"Strategies: {[s.name for s in self.strategies]}")
        logger.info(f"ML Ensemble: {'ACTIVE' if self.ml_ensemble.is_active else 'COLD START'}")
        logger.info(f"Model version: v{self.ml_ensemble.model_version}")

        # Instrument IDs are hardcoded in config/instruments.py (immutable)
        from config.instruments import ALL_IDS
        logger.info(f"Instrument universe: {len(ALL_IDS)} symbols mapped")

        # Update portfolio state
        await self.update_portfolio_state()
        logger.info("Initialization complete")
        logger.info("=" * 60)

    async def update_portfolio_state(self):
        """Refresh portfolio state from eToro, with local DB fallback."""
        try:
            portfolio = await self.etoro.get_portfolio()
            cp = portfolio.get("clientPortfolio", portfolio)
            positions = cp.get("positions", [])
            credit = cp.get("credit", 0)

            # Reconcile eToro positions with local DB (enriches with strategy tags)
            reconciliation = self.db.reconcile_positions(positions)
            positions = reconciliation["enriched"]
            if reconciliation["closed_stale"]:
                logger.info(f"Reconciled: {reconciliation['closed_stale']} positions closed externally")
            if reconciliation["unknown"]:
                logger.info(f"Reconciled: {reconciliation['unknown']} positions not tracked (manual trades)")

            # Include PnL for accurate equity calculation
            try:
                pnl_data = await self.etoro.get_pnl()
                pnl_cp = pnl_data.get("clientPortfolio", pnl_data)
                unrealized_pnl = pnl_cp.get("unrealizedPnL", 0)
            except Exception:
                unrealized_pnl = 0

            # Include mirror (copy trading) available cash
            mirrors = cp.get("mirrors", [])
            mirror_cash = sum(m.get("availableAmount", 0) for m in mirrors)

            # EUR wallet: eToro API only returns USD credit, but account may have
            # EUR funds not visible via API. Read from env/config as override.
            eur_balance = float(os.getenv("ETORO_EUR_BALANCE", "0"))
            eur_usd_rate = float(os.getenv("ETORO_EUR_USD_RATE", "1.085"))
            eur_in_usd = eur_balance * eur_usd_rate

            # Build balance dict for risk manager
            total_invested = sum(p.get("initialAmountInDollars", 0) for p in positions)
            total_cash = credit + mirror_cash + eur_in_usd
            balance = {
                "cash": total_cash,
                "equity": total_cash + total_invested + unrealized_pnl,
                "invested": total_invested,
                "credit_usd": credit,
                "eur_balance": eur_balance,
                "unrealized_pnl": unrealized_pnl,
            }

            self.risk_manager.update_state(balance, positions)
            logger.info(
                f"Portfolio: equity=${balance['equity']:,.2f}, "
                f"cash=${credit:.2f}, "
                f"positions={len(positions)}"
            )
        except Exception as e:
            logger.warning(f"eToro API unavailable: {e} — loading positions from local DB")
            self._load_positions_from_db()

    def _load_positions_from_db(self):
        """Fallback: load last known positions from local DB when eToro API is down."""
        local_positions = self.db.get_open_positions()
        if not local_positions:
            logger.info("No locally tracked positions found")
            return

        # Convert DB rows to eToro-like position dicts for risk manager
        positions = []
        for p in local_positions:
            positions.append({
                "positionID": p.get("etoro_position_id"),
                "instrumentId": p.get("instrument_id"),
                "symbol": p.get("symbol", ""),
                "strategy_tag": p.get("strategy", "unknown"),
                "direction": p.get("direction", "Buy"),
                "investedAmount": p.get("amount", 0),
                "initialAmountInDollars": p.get("amount", 0),
                "openRate": p.get("open_rate") or p.get("entry_price", 0),
                "stopLossRate": p.get("stop_loss"),
                "takeProfitRate": p.get("take_profit"),
            })

        # Use last known equity from daily snapshot
        import sqlite3
        equity = 0
        cash = 0
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                row = conn.execute(
                    "SELECT equity, cash FROM daily_snapshots ORDER BY date DESC LIMIT 1"
                ).fetchone()
                if row:
                    equity = row[0] or 0
                    cash = row[1] or 0
        except Exception:
            pass

        if equity == 0:
            total_invested = sum(p["investedAmount"] for p in positions)
            equity = total_invested + cash

        balance = {"cash": cash, "equity": equity, "invested": equity - cash}
        self.risk_manager.update_state(balance, positions)
        logger.info(
            f"Portfolio (from DB fallback): equity=${equity:,.2f}, "
            f"cash=${cash:.2f}, positions={len(positions)}"
        )

    # ────────────────────── Regime Detection ──────────────────────

    async def detect_regime(self, market_data: dict = None):
        """
        Detect current market regime.
        Called before signal generation to attach fingerprint to every decision.
        """
        try:
            if market_data is None:
                # Build market data from cached OHLCV
                market_data = {}
                from config.instruments import US_EQUITIES
                for symbol, meta in list(US_EQUITIES.items())[:10]:
                    inst_id = meta.get("etoro_id")
                    if inst_id:
                        df = await self.data_engine.get_ohlcv(inst_id, symbol, "OneDay", 60)
                        if not df.empty:
                            df = self.data_engine.compute_indicators(df)
                            market_data[symbol] = df

            if market_data:
                self._current_regime = self.regime_detector.detect(market_data)
                logger.info(f"Regime: {self._current_regime}")

                # Enrich regime with news sentiment
                if self.news_sentiment:
                    try:
                        macro = self.news_sentiment.get_macro_sentiment()
                        self._current_regime.data["news_sentiment"] = macro.get("score", 0)
                        fed = self.news_sentiment.get_fed_sentiment()
                        self._current_regime.data["fed_sentiment"] = fed.get("score", 0)
                        logger.info(
                            f"Sentiment: macro={macro.get('score', 0):.2f} "
                            f"({macro.get('n_articles', 0)} articles), "
                            f"fed={fed.get('score', 0):.2f}"
                        )
                    except Exception as e:
                        logger.debug(f"Sentiment fetch failed: {e}")

                # Check for extreme conditions
                if self._current_regime.is_extreme:
                    logger.warning("EXTREME regime detected — reducing exposure")
                    await self.notifier.notify_risk_alert(
                        "Extreme Regime",
                        f"Regime fingerprint: {self._current_regime}\n"
                        "Action: Reducing new position sizes by 50%"
                    )

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")

    def _apply_regime_allocations(self):
        """Update strategy allocation_pct based on current regime."""
        if self._current_regime is None:
            return

        vix_level = None
        vol_regime = self._current_regime.data.get("volatility_regime")
        # Map volatility regime back to approximate VIX for allocation lookup
        vix_map = {"low": 12, "medium": 18, "high": 25, "extreme": 35}
        if vol_regime in vix_map:
            vix_level = vix_map[vol_regime]

        trend_regime = self._current_regime.data.get("trend_regime")
        adjusted = config.allocation.get_regime_adjusted(
            vix_level=vix_level, trend_regime=trend_regime
        )

        # Apply to each strategy
        name_map = {s.name: s for s in self.strategies}
        for name, weight in adjusted.items():
            if name in name_map:
                name_map[name].allocation_pct = weight

        logger.info(
            f"Regime-adjusted allocations: "
            + ", ".join(f"{k}={v:.0%}" for k, v in adjusted.items())
        )

    def _apply_ic_weighting(self):
        """
        Refine strategy allocations using rolling 60-day Information Coefficient.
        IC = correlation between predicted signal confidence and actual outcome.
        Strategies with higher IC get more allocation (proportional tilt).
        """
        name_map = {s.name: s for s in self.strategies}
        ic_scores = {}

        for strategy_name in name_map:
            perf = self.db.get_strategy_performance(strategy_name, days=60)
            if perf["trades"] < 10:
                ic_scores[strategy_name] = 0.5  # Neutral IC when insufficient data
                continue

            # Use Sharpe as IC proxy (correlation of signal → outcome)
            sharpe = perf.get("sharpe", 0)
            # Map Sharpe to 0-1 IC: Sharpe=0→0.5, Sharpe=1→0.75, Sharpe=-1→0.25
            ic = 0.5 + 0.25 * max(-1, min(1, sharpe))
            ic_scores[strategy_name] = ic

        # Only tilt if we have meaningful IC dispersion
        ic_values = list(ic_scores.values())
        if max(ic_values) - min(ic_values) < 0.1:
            return  # All strategies have similar IC, skip tilt

        # IC-weighted tilt: multiply current allocation by IC, renormalize
        total = 0
        new_alloc = {}
        for name, strategy in name_map.items():
            ic = ic_scores.get(name, 0.5)
            new_alloc[name] = strategy.allocation_pct * ic
            total += new_alloc[name]

        if total > 0:
            for name in new_alloc:
                name_map[name].allocation_pct = new_alloc[name] / total

            logger.info(
                f"IC-weighted allocations: "
                + ", ".join(f"{k}={name_map[k].allocation_pct:.0%}" for k in new_alloc)
            )

    # ────────────────────── Main Cycle ──────────────────────

    async def run_signal_scan(self):
        """
        Main trading cycle (v2 — with ML ensemble):
        1. Update portfolio state
        2. Detect market regime
        3. Check existing positions for exits
        4. Generate new signals from all strategies
        5. ML ensemble: predict outcome probability, adjust confidence
        6. Risk-check and size signals
        7. Execute approved trades
        8. Log features + regime for closed-loop learning
        """
        logger.info("─── Signal Scan v2 Starting ───")

        # 1. Update state
        await self.update_portfolio_state()
        self.data_engine.clear_cache()

        # 2. Detect regime and adjust strategy allocations
        await self.detect_regime()
        # Pass vol regime to data_engine for factor model quality tilt
        if self._current_regime:
            self.data_engine._last_vol_regime = self._current_regime.data.get("volatility_regime")
        self._apply_regime_allocations()
        self._apply_ic_weighting()

        # 3. Check exits on existing positions
        await self._check_exits()

        # 4. Check if trading is halted
        if self.risk_manager.state.is_halted:
            logger.warning("Trading is HALTED — skipping signal generation")
            return

        # 5. Generate signals from all strategies
        all_signals: List[TradeSignal] = []
        for strategy in self.strategies:
            try:
                # Strategy qualification gate (Simons): min 50 trades + Sharpe > 0.3
                perf = self.db.get_strategy_performance(strategy.name, days=180)
                if perf["trades"] > 0 and (perf["trades"] < 50 or perf.get("sharpe", 0) < 0.3):
                    logger.info(
                        f"[{strategy.name}] Unqualified: {perf['trades']} trades, "
                        f"Sharpe {perf.get('sharpe', 0):.2f} (need 50+ trades, Sharpe > 0.3)"
                    )
                    # Still allow signal generation but reduce allocation by 50%
                    strategy.allocation_pct *= 0.5

                # Check regime favorability
                if self._current_regime:
                    if (strategy.name == "momentum" and
                            not self._current_regime.is_favorable_for_momentum and
                            self._current_regime.data.get("trend_regime") not in ("weak_up", "strong_up")):
                        logger.info(f"[{strategy.name}] Skipped — unfavorable regime")
                        continue

                positions = [p for p in self.risk_manager.state.positions
                            if p.get("strategy_tag") == strategy.name]
                signals = await strategy.generate_signals(self.data_engine, positions)
                strategy.log_signals(signals)
                all_signals.extend(signals)
                logger.info(f"[{strategy.name}] Generated {len(signals)} signals")
            except Exception as e:
                logger.error(f"[{strategy.name}] Signal generation failed: {e}")

        if not all_signals:
            logger.info("No signals generated this cycle")
            return

        # 6. ML Ensemble filtering + Risk-check + Execute
        executed = 0
        vetoed = 0
        for signal in all_signals:
            try:
                # ── ML Ensemble prediction ──
                signal_data = self._build_signal_data(signal)
                regime_data = self._current_regime.to_dict() if self._current_regime else {}

                ml_result = self.ml_ensemble.predict(signal_data, regime_data)

                # Veto check
                if ml_result["ml_veto"]:
                    vetoed += 1
                    logger.info(
                        f"ML VETO: {signal.symbol} — "
                        f"P(profit)={ml_result['ml_probability']:.2%}"
                    )
                    # Still log the signal for learning (even vetoed ones)
                    self._log_signal_with_features(signal, ml_result, regime_data, executed=False)
                    continue

                # Update signal confidence with ML adjustment
                if ml_result["ml_active"]:
                    signal.confidence = ml_result["ml_confidence_adj"]

                # ── Meta-labeling filter (per-strategy false positive gate) ──
                if self.meta_labeler is not None:
                    if not self.meta_labeler.should_trade(signal.strategy_name, signal_data):
                        vetoed += 1
                        logger.info(f"META-LABEL VETO: {signal.symbol} ({signal.strategy_name})")
                        self._log_signal_with_features(signal, ml_result, regime_data, executed=False)
                        continue

                # ── Risk check ──
                allowed, reason = self.risk_manager.check_can_trade(signal)
                if not allowed:
                    logger.info(f"Signal rejected: {signal.symbol} — {reason}")
                    self._log_signal_with_features(signal, ml_result, regime_data, executed=False)
                    continue

                # ── Extreme regime: reduce size by 50% ──
                size_multiplier = 1.0
                if self._current_regime and self._current_regime.is_extreme:
                    size_multiplier = 0.5

                # ── Size position ──
                asset_type = "fx" if signal.strategy_name == "fx_carry" else "equity"
                perf = self.db.get_strategy_performance(signal.strategy_name)
                # Pass available cash so sizer doesn't exceed it
                if signal.metadata is None:
                    signal.metadata = {}
                signal.metadata["available_cash"] = self.risk_manager.state.cash
                sizing = self.position_sizer.compute_trade_size(
                    self.risk_manager.state.equity, signal, perf, asset_type
                )

                if not sizing.get("execute"):
                    logger.info(f"Signal skipped: {signal.symbol} — {sizing.get('reason')}")
                    continue

                # Apply regime multiplier
                sizing["dollar_amount"] *= size_multiplier

                # ── Log signal with features (before execution) ──
                signal_id = self._log_signal_with_features(
                    signal, ml_result, regime_data, executed=True
                )

                # ── Spread check (delay if spread > 2x median) ──
                from config.instruments import get_instrument_id
                etoro_id = get_instrument_id(signal.symbol) or signal.instrument_id
                spread_info = await self.etoro.check_spread(etoro_id)
                if not spread_info.get("ok"):
                    logger.warning(f"Wide spread for {signal.symbol}: {spread_info['reason']} — skipping")
                    continue

                # ── Execute trade ──
                is_buy = signal.signal.value in ("BUY", "STRONG_BUY", 1, 2)
                result = await self.etoro.open_position(
                    instrument_id=etoro_id,
                    is_buy=is_buy,
                    amount=sizing["dollar_amount"],
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                )

                # Log trade
                etoro_pos_id = result.get("positionId") or result.get("positionID")
                fill_price = result.get("entryPrice", signal.entry_price)
                shortfall = (fill_price - signal.entry_price) / signal.entry_price if signal.entry_price else 0.0
                self.db.log_trade_open(signal_id, {
                    "symbol": signal.symbol,
                    "strategy": signal.strategy_name,
                    "direction": signal.direction,
                    "amount": sizing["dollar_amount"],
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "etoro_position_id": etoro_pos_id,
                    "shortfall": shortfall,
                })

                # Persist position locally for restart recovery
                self.db.save_position({
                    "etoro_position_id": etoro_pos_id,
                    "symbol": signal.symbol,
                    "instrument_id": signal.instrument_id,
                    "strategy": signal.strategy_name,
                    "direction": signal.direction,
                    "amount": sizing["dollar_amount"],
                    "entry_price": signal.entry_price,
                    "open_rate": fill_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                })

                # Notify
                ml_tag = f" | ML: {ml_result['ml_probability']:.0%}" if ml_result["ml_active"] else ""
                await self.notifier.notify_signal(signal)
                await self.notifier.notify_trade_executed({
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "amount": sizing["dollar_amount"],
                    "ml_probability": ml_result["ml_probability"],
                })

                executed += 1
                logger.info(
                    f"✅ EXECUTED: {signal.direction} {signal.symbol} "
                    f"${sizing['dollar_amount']:,.2f} | {signal.strategy_name}"
                    f"{ml_tag}"
                )

            except Exception as e:
                logger.error(f"Execution failed for {signal.symbol}: {e}")

        logger.info(
            f"─── Scan Complete: {executed}/{len(all_signals)} executed, "
            f"{vetoed} ML-vetoed ───"
        )

    def _build_signal_data(self, signal: TradeSignal) -> dict:
        """Extract ML feature data from a TradeSignal."""
        meta = signal.metadata or {}
        return {
            "momentum_score": meta.get("momentum_score", 0),
            "mr_zscore": meta.get("z_score", 0),
            "factor_score": meta.get("factor_score", 0),
            "fx_carry_score": meta.get("carry_score", 0),
            "confidence": signal.confidence,
            "risk_reward": signal.risk_reward_ratio,
            "atr_pct": meta.get("atr_pct", 0.02),
            "volume_ratio": meta.get("volume_ratio", 1.0),
            "rsi": meta.get("rsi", 50),
            "macd": meta.get("macd", 0),
            "bb_position": meta.get("bb_position", 0.5),
            "momentum_3m": meta.get("momentum_3m", 0),
            "sma_cross": meta.get("sma_cross", False),
            "hour": datetime.utcnow().hour,
        }

    def _log_signal_with_features(self, signal: TradeSignal, ml_result: dict,
                                    regime_data: dict, executed: bool) -> int:
        """Log signal with full feature vector and regime fingerprint for ML learning."""
        import sqlite3

        # Log base signal
        signal_id = self.db.log_signal(signal)

        # Attach regime + feature vector
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute(
                    """UPDATE signals
                       SET regime_fingerprint = ?,
                           feature_vector = ?,
                           executed = ?
                       WHERE id = ?""",
                    (
                        json.dumps(regime_data),
                        json.dumps(ml_result.get("feature_vector", [])),
                        1 if executed else 0,
                        signal_id,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to attach features to signal {signal_id}: {e}")

        return signal_id

    async def _check_exits(self):
        """Check all open positions for exit signals."""
        for position in self.risk_manager.state.positions:
            strategy_name = position.get("strategy_tag", "")
            symbol = position.get("symbol", "")

            for strategy in self.strategies:
                if strategy.name != strategy_name:
                    continue

                try:
                    instrument_id = position.get("instrumentId")
                    if not instrument_id:
                        continue

                    df = await self.data_engine.get_ohlcv(
                        instrument_id, symbol, "OneDay", 60
                    )
                    if df.empty:
                        continue

                    df = self.data_engine.compute_indicators(df)
                    exit_signal = strategy.should_exit(position, df)

                    if exit_signal:
                        pos_id = position.get("positionID", position.get("positionId"))
                        inst_id = position.get("instrumentID", position.get("instrumentId"))
                        await self.etoro.close_position(pos_id, inst_id)

                        # Mark closed in local DB
                        if pos_id:
                            self.db.close_position(str(pos_id))

                        logger.info(
                            f"🔴 EXIT: {symbol} | {exit_signal.metadata.get('exit_reason', '')}"
                        )
                        await self.notifier.notify_risk_alert(
                            "Position Closed",
                            f"{symbol}: {exit_signal.metadata.get('exit_reason', '')}"
                        )

                except Exception as e:
                    logger.error(f"Exit check failed for {symbol}: {e}")

    # ────────────────────── Outcome Labeling ──────────────────────

    async def run_outcome_labeling(self):
        """Label outcomes for past decisions (runs every 15 min after signal scan)."""
        try:
            labeled = await self.outcome_labeler.label_outcomes(self.data_engine)
            if labeled > 0:
                logger.info(f"Labeled {labeled} decision outcomes")
        except Exception as e:
            logger.error(f"Outcome labeling failed: {e}")

    # ────────────────────── Risk & Summary ──────────────────────

    async def run_risk_check(self):
        """Periodic risk monitoring (runs more frequently than signals)."""
        await self.update_portfolio_state()
        summary = self.risk_manager.get_portfolio_summary()

        # Check for drawdown reduction
        should_reduce, reduction = self.risk_manager.should_reduce_all()
        if should_reduce:
            logger.warning(f"⚠️ DRAWDOWN ALERT: Reducing positions by {reduction:.0%}")
            await self.notifier.notify_risk_alert(
                "Drawdown Protection",
                f"Drawdown: {summary['current_drawdown']:.1%}\n"
                f"Action: Reducing all positions by {reduction:.0%}"
            )
            if reduction >= 1.0:
                for pos in self.risk_manager.state.positions:
                    try:
                        pos_id = pos.get("positionID", pos.get("positionId"))
                        inst_id = pos.get("instrumentID", pos.get("instrumentId"))
                        await self.etoro.close_position(pos_id, inst_id)
                        if pos_id:
                            self.db.close_position(str(pos_id))
                    except Exception as e:
                        logger.error(f"Failed to close {pos.get('symbol', pos_id)}: {e}")

    async def run_daily_summary(self):
        """End-of-day summary with ML status."""
        await self.update_portfolio_state()
        summary = self.risk_manager.get_portfolio_summary()
        self.db.save_daily_snapshot(summary)

        # Add ML ensemble status to summary
        ml_status = self.ml_ensemble.get_status()
        summary["ml_ensemble"] = ml_status

        # Add labeling stats
        label_stats = self.outcome_labeler.get_labeling_stats()
        summary["labeling"] = label_stats

        await self.notifier.notify_daily_summary(summary)
        logger.info(
            f"Daily snapshot: equity=${summary['equity']:,.2f}, "
            f"ML v{ml_status['model_version']} "
            f"({'active' if ml_status['active'] else 'cold start'})"
        )

    async def run_daily_retrain(self):
        """Daily ML retrain (03:15 UTC)."""
        from core.daily_retrain import label_and_retrain
        result = await label_and_retrain(self.data_engine)
        return result

    async def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down AlphaDesk...")
        await self.etoro.close()
        logger.info("Shutdown complete")


async def main():
    """Entry point for direct execution."""
    setup_logging(config.log_path)

    desk = AlphaDesk()
    try:
        await desk.initialize()
        await desk.run_signal_scan()
        await desk.run_outcome_labeling()
        await desk.run_daily_summary()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await desk.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
