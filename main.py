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
        # Broker selection: etoro | ibkr_paper | ibkr_live
        broker_mode = os.getenv("BROKER", "etoro").lower()

        if broker_mode.startswith("ibkr"):
            from core.ibkr_client import IBKRClient
            port = 4002 if broker_mode == "ibkr_paper" else 4001
            self.etoro = IBKRClient(
                host=os.getenv("IBKR_HOST", "127.0.0.1"),
                port=int(os.getenv("IBKR_PORT", str(port))),
                client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
                timeout=30,
            )
            logger.info(f"Broker: IBKR ({broker_mode}) via {self.etoro.host}:{self.etoro.port}")
        else:
            self.etoro = EtoroClient(
                user_key=config.etoro.user_key,
                api_key=config.etoro.api_key,
                base_url=config.etoro.base_url,
                environment=config.etoro.environment,
                timeout=config.etoro.request_timeout,
                max_retries=config.etoro.max_retries,
            )
            logger.info(f"Broker: eToro ({config.etoro.environment})")

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

        # AI Agent Manager (optional — needs ANTHROPIC_API_KEY)
        try:
            from core.ai_agents import AIPortfolioManager
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if api_key:
                self.ai_manager = AIPortfolioManager(db_path=config.db_path)
                logger.info("AI Agent Manager: ACTIVE")
            else:
                self.ai_manager = None
                logger.info("AI Agent Manager: DISABLED (no ANTHROPIC_API_KEY)")
        except ImportError:
            self.ai_manager = None
            logger.info("AI Agent Manager: DISABLED (module not found)")

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

        # Apply AutoResearch optimized parameters
        try:
            from autoresearch.strategy_tuner import get_params, apply_params
            for strategy in self.strategies:
                try:
                    params = get_params(strategy.name)
                    apply_params(strategy, params)
                    logger.info(f"[{strategy.name}] Applied tuned params")
                except KeyError:
                    pass  # Strategy not in tuner (e.g., PEAD)
                except Exception as e:
                    logger.warning(f"[{strategy.name}] Failed to apply params: {e}")
        except Exception as e:
            logger.warning(f"AutoResearch params not available: {e}")

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

    def _apply_risk_parity(self):
        """Correlation-aware risk parity: equalize risk contributions.

        Uses scipy optimization to find weights where each strategy
        contributes equally to total portfolio volatility, accounting
        for correlations between strategies. Falls back to inverse-vol
        if insufficient data.
        """
        import numpy as np
        import sqlite3

        name_map = {s.name: s for s in self.strategies if s.name != "pead"}
        strategy_names = list(name_map.keys())
        min_trades = 15
        returns_by_strategy = {}

        for name in strategy_names:
            with sqlite3.connect(self.db.db_path) as conn:
                rows = conn.execute(
                    """SELECT pnl_pct FROM trades
                       WHERE strategy = ? AND status = 'closed'
                         AND close_time >= datetime('now', '-90 days')
                       ORDER BY close_time DESC""",
                    (name,),
                ).fetchall()
            returns = [r[0] for r in rows if r[0] is not None]
            if len(returns) >= min_trades:
                returns_by_strategy[name] = returns

        # Need at least 2 strategies with data
        if len(returns_by_strategy) < 2:
            logger.info("Risk parity: insufficient data — using regime allocations")
            return

        # Build covariance matrix (pad shorter series with zeros)
        valid_names = list(returns_by_strategy.keys())
        max_len = max(len(v) for v in returns_by_strategy.values())
        returns_matrix = np.zeros((max_len, len(valid_names)))
        for i, name in enumerate(valid_names):
            r = returns_by_strategy[name]
            returns_matrix[:len(r), i] = r

        cov = np.cov(returns_matrix, rowvar=False)
        # Ledoit-Wolf shrinkage for stability
        n = len(valid_names)
        shrinkage = 0.3
        cov = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))

        # Optimize for equal risk contribution
        try:
            from scipy.optimize import minimize

            def risk_contrib_obj(w):
                w = np.array(w)
                port_vol = np.sqrt(w @ cov @ w)
                if port_vol < 1e-10:
                    return 1e10
                mrc = cov @ w / port_vol  # marginal risk contribution
                rc = w * mrc              # risk contribution per strategy
                target = port_vol / n     # equal risk target
                return float(np.sum((rc - target) ** 2))

            x0 = np.ones(n) / n
            bounds = [(0.05, 0.60)] * n
            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            result = minimize(risk_contrib_obj, x0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 200})

            if result.success:
                rp_weights = {name: float(w) for name, w in zip(valid_names, result.x)}
            else:
                # Fallback to inverse-vol
                vols = np.sqrt(np.diag(cov))
                inv_vols = 1.0 / np.maximum(vols, 0.01)
                inv_vols /= inv_vols.sum()
                rp_weights = {name: float(w) for name, w in zip(valid_names, inv_vols)}

        except ImportError:
            # scipy not available — naive inverse-vol
            vols = np.sqrt(np.diag(cov))
            inv_vols = 1.0 / np.maximum(vols, 0.01)
            inv_vols /= inv_vols.sum()
            rp_weights = {name: float(w) for name, w in zip(valid_names, inv_vols)}

        # Strategies without enough data get median weight
        if len(rp_weights) < len(strategy_names):
            median_w = float(np.median(list(rp_weights.values())))
            for name in strategy_names:
                if name not in rp_weights:
                    rp_weights[name] = median_w
            # Renormalize
            total_w = sum(rp_weights.values())
            rp_weights = {k: v / total_w for k, v in rp_weights.items()}

        # Blend: 60% risk parity + 40% regime allocation
        for name, strategy in name_map.items():
            blended = 0.6 * rp_weights.get(name, 0.25) + 0.4 * strategy.allocation_pct
            strategy.allocation_pct = blended

        # Renormalize
        total = sum(s.allocation_pct for s in self.strategies)
        if total > 0:
            for s in self.strategies:
                s.allocation_pct /= total

        logger.info(
            f"Risk parity (corr-aware): "
            + ", ".join(f"{k}={rp_weights[k]:.0%}" for k in rp_weights)
            + " | Blended: "
            + ", ".join(f"{s.name}={s.allocation_pct:.0%}" for s in self.strategies)
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
        self._apply_risk_parity()
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
                        logger.info(f"[{strategy.name}] Unfavorable regime — reducing allocation 50%")
                        strategy.allocation_pct *= 0.5

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

                # ── AI Agent validation (for signals > $200) ──
                if hasattr(self, 'ai_manager') and self.ai_manager:
                    # Estimate sizing to decide if AI validation is warranted
                    asset_type_est = "fx" if signal.strategy_name == "fx_carry" else "equity"
                    perf_est = self.db.get_strategy_performance(signal.strategy_name)
                    sizing_estimate = self.position_sizer.compute_trade_size(
                        self.risk_manager.state.equity, signal, perf_est, asset_type_est
                    ).get("dollar_amount", 0)

                    if sizing_estimate > 200:
                        try:
                            ai_decision = await asyncio.wait_for(
                                self._run_ai_evaluation(signal, signal_data, regime_data),
                                timeout=10.0,
                            )
                            if not ai_decision.approved:
                                vetoed += 1
                                logger.info(
                                    f"AI VETO: {signal.symbol} — "
                                    f"score={ai_decision.final_score:.2f}, "
                                    f"{ai_decision.reasoning[:100]}"
                                )
                                self._log_signal_with_features(
                                    signal, ml_result, regime_data, executed=False
                                )
                                continue
                            # Adjust confidence based on AI consensus
                            if ai_decision.confidence:
                                signal.confidence = (signal.confidence + ai_decision.confidence) / 2
                        except asyncio.TimeoutError:
                            logger.warning(f"AI evaluation timed out for {signal.symbol} — proceeding without AI")
                        except Exception as e:
                            logger.warning(f"AI evaluation failed for {signal.symbol}: {e} — proceeding without AI")

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

                # ── Dedup: skip if we already have a position in this instrument ──
                existing_positions = self.risk_manager.state.positions or []
                already_holds = any(
                    p.get("instrumentID") == etoro_id or p.get("instrumentId") == etoro_id
                    for p in existing_positions
                )
                if already_holds:
                    logger.info(f"Skipping {signal.symbol} — already holding position")
                    continue

                # ── Circuit breaker gate ──
                if self._circuit_breaker and not self._circuit_breaker.can_open_position():
                    cb_state = self._circuit_breaker.get_status()
                    logger.warning(
                        f"🚧 CIRCUIT BREAKER (tier {cb_state.get('tier')}): "
                        f"skipping {signal.symbol} — "
                        f"session dd {cb_state.get('drawdown_pct', 0):.1%}"
                    )
                    continue

                # ── Execute trade (no SL/TP via API — managed by software TP/SL) ──
                is_buy = signal.signal.value in ("BUY", "STRONG_BUY", 1, 2)
                result = await self.etoro.open_position(
                    instrument_id=etoro_id,
                    is_buy=is_buy,
                    amount=sizing["dollar_amount"],
                )

                # Verify order was actually executed (not just pending/cancelled)
                order_status = result.get("statusID", result.get("orderForOpen", {}).get("statusID", 0))
                await asyncio.sleep(3)  # Wait for eToro to process

                # Check if position actually appeared in portfolio
                portfolio = await self.etoro.get_portfolio()
                live_positions = portfolio.get("clientPortfolio", {}).get("positions", [])
                position_exists = any(
                    p.get("instrumentID") == etoro_id
                    for p in live_positions
                )

                if not position_exists:
                    logger.warning(
                        f"⚠️ PHANTOM ORDER: {signal.symbol} (id={etoro_id}) — "
                        f"order accepted but position NOT found in portfolio. Skipping."
                    )
                    continue

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
            # AFML features (López de Prado)
            "close_ffd": meta.get("close_ffd", 0),
            "ffd_zscore": meta.get("ffd_zscore", 0),
            "cusum_event": meta.get("cusum_event", False),
        }

    async def _run_ai_evaluation(self, signal, signal_data: dict, regime_data: dict) -> dict:
        """Run AI agent evaluation, handling both sync and async implementations."""
        import inspect
        result = self.ai_manager.evaluate_signal(
            trade_signal=signal,
            market_data=signal_data,
            portfolio_state=self.risk_manager.state,
            regime=regime_data,
        )
        if inspect.isawaitable(result):
            result = await result
        return result

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

    # Software TP/SL with trailing stop, time exit, partial TP
    _TP_SL = {
        3008: {"tp": 65.0, "sl": 52.0, "name": "XLE"},
        1036: {"tp": 175.0, "sl": 143.0, "name": "XOM"},
        6434: {"tp": 340.0, "sl": 280.0, "name": "GOOGL"},
        1022: {"tp": 275.0, "sl": 220.0, "name": "TSLA"},
        1137: {"tp": 205.0, "sl": 165.0, "name": "NVDA"},
        4253: {"tp": 54.0, "sl": 40.0, "name": "SLB"},
        3020: {"tp": 550.0, "sl": 470.0, "name": "GLD"},
    }

    # Trailing stop tiers: (pnl_pct_threshold, lock_in_pct)
    _TRAILING_TIERS = [
        (0.15, 0.10),  # +15% PnL → lock in +10%
        (0.10, 0.05),  # +10% PnL → lock in +5%
        (0.05, 0.00),  # +5% PnL  → lock in breakeven
    ]
    _TIME_EXIT_DAYS = 30
    _TIME_EXIT_FLAT_PCT = 0.02
    _PARTIAL_TP_PCT = 0.50
    _partial_tp_done: dict = {}

    _circuit_breaker = None
    _correlation_monitor = None

    async def _check_circuit_breaker(self):
        """Three-tier drawdown protection. Called from run_risk_check."""
        try:
            if self._circuit_breaker is None:
                from core.circuit_breaker import CircuitBreaker
                self._circuit_breaker = CircuitBreaker(
                    db_path=config.db_path,
                    notifier=self.notifier,
                )

            state = self.risk_manager.state
            equity = state.equity if state else 0
            if equity <= 0:
                return

            positions = state.positions if state else []
            await self._circuit_breaker.check(equity, positions, self.etoro)
        except Exception as e:
            logger.debug(f"Circuit breaker check error: {e}")

    async def run_correlation_check(self):
        """Every 4h: compute rolling correlation matrix of portfolio."""
        try:
            if self._correlation_monitor is None:
                from core.correlation_monitor import CorrelationMonitor
                self._correlation_monitor = CorrelationMonitor(
                    db_path=config.db_path,
                    notifier=self.notifier,
                )

            state = self.risk_manager.state
            positions = state.positions if state else []
            if len(positions) < 3:
                return None
            return await self._correlation_monitor.analyze(positions)
        except Exception as e:
            logger.debug(f"Correlation check error: {e}")
            return None

    async def _check_tp_sl(self):
        """Enhanced exit logic: trailing stop, time-based exit, partial TP."""
        try:
            portfolio_data = await self.etoro.get_portfolio()
            portfolio_positions = portfolio_data.get("clientPortfolio", {}).get("positions", [])
            pnl_data = await self.etoro.get_pnl()
            pnl_positions = pnl_data.get("clientPortfolio", {}).get("positions", [])

            pnl_by_pid = {p.get("positionID"): p for p in pnl_positions if p.get("positionID")}

            from collections import defaultdict
            positions_by_iid = defaultdict(list)
            for p in portfolio_positions:
                iid = p.get("instrumentID", 0)
                if iid in self._TP_SL:
                    positions_by_iid[iid].append(p)

            for pos in portfolio_positions:
                iid = pos.get("instrumentID", 0)
                t = self._TP_SL.get(iid)
                if not t:
                    continue

                pid = pos.get("positionID")
                open_rate = pos.get("openRate", 0)
                amount = pos.get("amount", 0)
                open_dt_str = pos.get("openDateTime")

                pnl_pos = pnl_by_pid.get(pid, {})
                unrealized = pnl_pos.get("unrealizedPnL", {})
                current_rate = unrealized.get("currentRate", 0)
                pnl_dollars = unrealized.get("pnL", 0)

                if current_rate <= 0 or open_rate <= 0:
                    continue

                pnl_pct = pnl_dollars / amount if amount > 0 else 0

                # ── Full Take Profit ──
                if current_rate >= t["tp"]:
                    logger.info(f"TP HIT: {t['name']} @ {current_rate} >= {t['tp']}")
                    try:
                        await self.etoro.close_position(pid, iid)
                        await self.notifier.send_message(
                            f"🎯 *TAKE PROFIT* — {t['name']}\n"
                            f"Chiusa @ ${current_rate:.2f} (target ${t['tp']})\n"
                            f"PnL: ${pnl_dollars:+.2f} ({pnl_pct:+.1%})")
                        self._partial_tp_done.pop(pid, None)
                    except Exception as e:
                        logger.error(f"TP close failed {t['name']}: {e}")
                    continue

                # ── Partial Take Profit (50% of way to TP) ──
                if pid not in self._partial_tp_done:
                    partial_target = open_rate + (t["tp"] - open_rate) * self._PARTIAL_TP_PCT
                    if current_rate >= partial_target and len(positions_by_iid.get(iid, [])) > 1:
                        smallest = min(positions_by_iid[iid], key=lambda p: p.get("amount", float("inf")))
                        smallest_pid = smallest.get("positionID")
                        smallest_amt = smallest.get("amount", 0)
                        logger.info(f"PARTIAL TP: {t['name']} pos {smallest_pid} (${smallest_amt:.0f})")
                        try:
                            await self.etoro.close_position(smallest_pid, iid)
                            await self.notifier.send_message(
                                f"🎯 *PARTIAL TP* — {t['name']}\n"
                                f"Chiusa minore (${smallest_amt:.0f}) @ ${current_rate:.2f}")
                            for p in positions_by_iid[iid]:
                                self._partial_tp_done[p.get("positionID")] = True
                            positions_by_iid[iid] = [
                                p for p in positions_by_iid[iid] if p.get("positionID") != smallest_pid]
                        except Exception as e:
                            logger.error(f"Partial TP failed {t['name']}: {e}")
                        if smallest_pid == pid:
                            continue

                # ── Trailing Stop ──
                effective_sl = t["sl"]
                trailing_label = None
                for tier_pct, lock_pct in self._TRAILING_TIERS:
                    if pnl_pct >= tier_pct:
                        trailing_sl = open_rate * (1 + lock_pct)
                        if trailing_sl > effective_sl:
                            effective_sl = trailing_sl
                            trailing_label = f"trailing +{lock_pct:.0%}" if lock_pct > 0 else "breakeven"
                        break

                if current_rate <= effective_sl:
                    sl_type = f"TRAILING STOP ({trailing_label})" if trailing_label else "STOP LOSS"
                    logger.info(f"{sl_type}: {t['name']} @ {current_rate} <= {effective_sl:.2f}")
                    try:
                        await self.etoro.close_position(pid, iid)
                        await self.notifier.send_message(
                            f"🛑 *{sl_type}* — {t['name']}\n"
                            f"Chiusa @ ${current_rate:.2f} (stop ${effective_sl:.2f})\n"
                            f"PnL: ${pnl_dollars:+.2f} ({pnl_pct:+.1%})")
                        self._partial_tp_done.pop(pid, None)
                    except Exception as e:
                        logger.error(f"{sl_type} close failed {t['name']}: {e}")
                    continue

                # ── Time-based Exit (30+ days) ──
                if open_dt_str:
                    try:
                        from datetime import timezone
                        open_dt = datetime.fromisoformat(open_dt_str.replace("Z", "+00:00"))
                        days_held = (datetime.now(timezone.utc) - open_dt).days
                        if days_held >= self._TIME_EXIT_DAYS:
                            if pnl_pct < -self._TIME_EXIT_FLAT_PCT:
                                reason = f"red ({pnl_pct:+.1%}) after {days_held}d"
                            elif abs(pnl_pct) <= self._TIME_EXIT_FLAT_PCT:
                                reason = f"flat ({pnl_pct:+.1%}) after {days_held}d"
                            else:
                                reason = None
                            if reason:
                                logger.info(f"TIME EXIT: {t['name']} — {reason}")
                                try:
                                    await self.etoro.close_position(pid, iid)
                                    await self.notifier.send_message(
                                        f"⏰ *TIME EXIT* — {t['name']}\n"
                                        f"Chiusa @ ${current_rate:.2f} — {reason}\n"
                                        f"PnL: ${pnl_dollars:+.2f}")
                                    self._partial_tp_done.pop(pid, None)
                                except Exception as e:
                                    logger.error(f"Time exit failed {t['name']}: {e}")
                                continue
                    except (ValueError, TypeError):
                        pass

        except Exception as e:
            logger.debug(f"TP/SL check: {e}")

    async def run_risk_check(self):
        """Periodic risk monitoring (runs more frequently than signals)."""
        await self.update_portfolio_state()
        await self._check_tp_sl()
        await self._check_circuit_breaker()
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

    # ────────────────────── Morning Briefing ──────────────────────

    async def run_morning_briefing(self):
        """Daily GS UHNWI-style briefing dispatcher.
        Weekday → daily macro analysis
        Saturday → weekly recap with accuracy evaluation
        Sunday → forward-looking weekly outlook
        """
        from datetime import datetime
        import sqlite3

        today = datetime.utcnow()
        day_of_week = today.weekday()  # 0=Mon, 5=Sat, 6=Sun

        try:
            if day_of_week == 5:
                await self._run_saturday_recap()
            elif day_of_week == 6:
                await self._run_sunday_outlook()
            else:
                await self._run_daily_briefing()
        except Exception as e:
            logger.error(f"Morning briefing failed: {e}", exc_info=True)
            await self.notifier.send_message(f"⚠️ Briefing error: {e}")

    async def _get_portfolio_snapshot(self) -> str:
        """Get current portfolio state for briefing."""
        try:
            from config.instruments import ALL_IDS
            reverse_map = {v: k for k, v in ALL_IDS.items()}
            portfolio = await self.etoro.get_portfolio()
            positions = portfolio.get("clientPortfolio", {}).get("positions", [])

            lines = []
            total_inv = 0
            total_pnl = 0
            for p in positions:
                iid = p.get("instrumentID", 0)
                ticker = reverse_map.get(iid, f"ID:{iid}")
                amount = p.get("amount", 0)
                rate = p.get("openRate", 0)
                lines.append(f"  {ticker}: ${amount:.0f} @ {rate}")
                total_inv += amount

            # Try to get PnL from PnL endpoint
            try:
                pnl_data = await self.etoro.get_pnl()
                pnl_positions = pnl_data.get("clientPortfolio", {}).get("positions", [])
                total_pnl = sum(p.get("unrealizedPnL", {}).get("pnL", 0) for p in pnl_positions)
            except Exception:
                pass

            header = f"Portfolio: ${total_inv:,.0f} invested | PnL: ${total_pnl:+,.2f}\n"
            return header + "\n".join(lines)
        except Exception as e:
            return f"Portfolio: unavailable ({e})"

    def _get_regime_text(self) -> str:
        """Get current regime for briefing."""
        if self._current_regime:
            r = self._current_regime
            return (
                f"Regime: vol={r.data.get('volatility_regime', '?')}, "
                f"trend={r.data.get('trend_regime', '?')}, "
                f"liquidity={r.data.get('liquidity_regime', '?')}"
            )
        # Compute fresh if cached is None
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            # Can't await here synchronously, return cached or unknown
            return "Regime: computing..."
        except Exception:
            return "Regime: unknown"

    async def _get_uw_text(self) -> str:
        """Get Unusual Whales snapshot for briefing."""
        if not hasattr(self, 'uw_client') or not self.uw_client:
            return "Unusual Whales: not configured"
        try:
            snapshot = await self.uw_client.get_macro_snapshot()
            return f"UW Market Tide: {json.dumps(snapshot, indent=2)[:500]}"
        except Exception as e:
            return f"Unusual Whales: error ({e})"

    async def _get_uw_full_data(self) -> str:
        """Get comprehensive UW data for GS-style analysis."""
        if not hasattr(self, 'uw_client') or not self.uw_client:
            return ""
        try:
            import asyncio
            tasks = {
                "market_tide": self.uw_client.get_market_tide(),
                "sectors": self.uw_client.get_sector_etfs(),
                "congress": self.uw_client.get_congress_signals(),
                "calendar": self.uw_client.get_economic_calendar(),
                "insider": self.uw_client.get_insider_aggregate(),
            }
            # Dark pool on major ETFs
            for etf in ["SPY", "QQQ", "TLT", "GLD", "XLE"]:
                tasks[f"dp_{etf}"] = self.uw_client.get_darkpool_ticker(etf)

            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            data = {}
            for key, result in zip(tasks.keys(), results):
                if not isinstance(result, Exception):
                    data[key] = result

            return json.dumps(data, indent=2, default=str)[:3000]
        except Exception as e:
            return f"UW data error: {e}"

    async def _ai_prompt(self, prompt: str, max_tokens: int = 2000, model: str = "claude-sonnet-4-6") -> str:
        """Call Claude for AI analysis."""
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _save_briefing(self, briefing_type: str, content: str):
        """Save briefing to SQLite for weekly recap."""
        import sqlite3
        with sqlite3.connect(self.db.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS briefings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    type TEXT,
                    content TEXT
                )
            """)
            conn.execute(
                "INSERT INTO briefings (type, content) VALUES (?, ?)",
                (briefing_type, content),
            )

    async def _run_daily_briefing(self):
        """GS UHNWI-style daily macro briefing via Telegram."""
        logger.info("Running daily briefing (GS UHNWI-style)...")

        # Gather data
        portfolio = await self._get_portfolio_snapshot()
        regime = self._get_regime_text()
        uw_basic = await self._get_uw_text()
        uw_full = await self._get_uw_full_data()

        # Part 1: Data summary via Telegram
        data_msg = (
            f"📊 *DAILY BRIEFING* — {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
            f"🌍 {regime}\n\n"
            f"💼 {portfolio}\n\n"
            f"🐋 {uw_basic}"
        )
        await self.notifier.send_message(data_msg)

        # Part 2: AI analysis (Sonnet)
        prompt = f"""You are a senior Goldman Sachs analyst preparing a morning briefing
for an UHNWI client with a $15K quantitative trading portfolio on eToro.

Write a concise, actionable briefing covering:
1. MACRO OVERVIEW — global market conditions, key overnight moves
2. RISK ASSESSMENT — what could go wrong today, tail risks
3. PORTFOLIO REVIEW — comment on current positions
4. ACTIONABLE IDEAS — 2-3 specific trade ideas with entry/exit levels
5. WATCHLIST — key events/data releases today

CURRENT DATA:
{regime}

{portfolio}

UNUSUAL WHALES DATA:
{uw_full[:2000]}

Keep it under 400 words. Be specific with numbers. No disclaimers.
Write in English, professional tone."""

        try:
            analysis = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._ai_prompt_sync(prompt, 2000, "claude-sonnet-4-6")
            )
            await self.notifier.send_message(f"🏦 *GS ANALYSIS*\n\n{analysis}")
            self._save_briefing("daily", f"{data_msg}\n\n---\n\n{analysis}")
            logger.info("Daily briefing sent successfully")
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            self._save_briefing("daily", data_msg)

    def _ai_prompt_sync(self, prompt: str, max_tokens: int = 2000, model: str = "claude-sonnet-4-6") -> str:
        """Synchronous Claude call for use in run_in_executor."""
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    async def _run_saturday_recap(self):
        """Saturday: recap the week's briefings, evaluate accuracy."""
        logger.info("Running Saturday weekly recap...")
        import sqlite3

        # Load week's briefings
        with sqlite3.connect(self.db.db_path) as conn:
            rows = conn.execute(
                """SELECT timestamp, type, content FROM briefings
                   WHERE timestamp >= datetime('now', '-7 days')
                   ORDER BY timestamp""",
            ).fetchall()

        week_briefings = "\n\n---\n\n".join(
            f"[{r[0]}] {r[1]}:\n{r[2][:500]}" for r in rows
        ) if rows else "No briefings this week."

        portfolio = await self._get_portfolio_snapshot()

        prompt = f"""You are a senior Goldman Sachs analyst doing a WEEKLY RECAP for an UHNWI client.

Review this week's daily briefings and evaluate:
1. ACCURACY — which predictions were right/wrong?
2. PORTFOLIO PERFORMANCE — how did positions perform this week?
3. LESSONS LEARNED — what patterns emerged?
4. GRADE — give the week's analysis an A-F grade with justification

WEEK'S BRIEFINGS:
{week_briefings[:3000]}

CURRENT PORTFOLIO:
{portfolio}

Keep under 400 words. Be brutally honest about accuracy. Grade fairly."""

        try:
            analysis = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._ai_prompt_sync(prompt, 2000)
            )
            msg = f"📅 *WEEKLY RECAP* — {datetime.utcnow().strftime('%Y-%m-%d')}\n\n{analysis}"
            await self.notifier.send_message(msg)
            self._save_briefing("weekly_recap", msg)
        except Exception as e:
            logger.error(f"Saturday recap failed: {e}")

    async def _run_sunday_outlook(self):
        """Sunday: forward-looking outlook for the coming week."""
        logger.info("Running Sunday outlook...")
        import sqlite3

        # Load Saturday recap for context
        with sqlite3.connect(self.db.db_path) as conn:
            rows = conn.execute(
                """SELECT content FROM briefings
                   WHERE type = 'weekly_recap'
                   ORDER BY timestamp DESC LIMIT 1""",
            ).fetchall()

        recap = rows[0][0][:1500] if rows else "No recap available."
        portfolio = await self._get_portfolio_snapshot()
        uw_data = await self._get_uw_full_data()

        prompt = f"""You are a senior Goldman Sachs analyst preparing a WEEKLY OUTLOOK for an UHNWI client.

Write a forward-looking analysis for the coming week:
1. KEY EVENTS — economic calendar, earnings, central bank decisions
2. MARKET OUTLOOK — expected direction, key levels to watch
3. PORTFOLIO POSITIONING — what to keep, what to trim, what to add
4. TOP 3 TRADES — specific actionable ideas with entry/SL/TP
5. RISK SCENARIOS — bull case, bear case, base case

LAST WEEK'S RECAP:
{recap}

CURRENT PORTFOLIO:
{portfolio}

UNUSUAL WHALES DATA:
{uw_data[:2000]}

Keep under 500 words. Be specific. Professional GS tone."""

        try:
            analysis = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._ai_prompt_sync(prompt, 2500)
            )
            msg = f"🔮 *WEEKLY OUTLOOK* — {datetime.utcnow().strftime('%Y-%m-%d')}\n\n{analysis}"
            await self.notifier.send_message(msg)
            self._save_briefing("weekly_outlook", msg)
        except Exception as e:
            logger.error(f"Sunday outlook failed: {e}")

    async def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down AlphaDesk...")
        await self.etoro.close()
        if hasattr(self, 'uw_client') and self.uw_client:
            await self.uw_client.close()
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
