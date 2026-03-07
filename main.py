"""
AlphaDesk — Main Orchestrator
Coordinates strategies, risk management, and execution.
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import List

from config.settings import config
from core.etoro_client import EtoroClient
from core.data_engine import DataEngine
from risk.portfolio_risk import PortfolioRiskManager
from risk.position_sizer import PositionSizer
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.factor_model import FactorModelStrategy
from strategies.fx_carry import FXCarryStrategy
from strategies.base_strategy import TradeSignal
from utils.db import TradeDB
from utils.logger import setup_logging
from utils.telegram_bot import TelegramNotifier

logger = logging.getLogger("alphadesk.main")


class AlphaDesk:
    """Main trading system orchestrator."""

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

        # Strategies
        self.strategies = [
            MomentumStrategy(config.allocation.momentum),
            MeanReversionStrategy(config.allocation.mean_reversion),
            FactorModelStrategy(config.allocation.factor_model),
            FXCarryStrategy(config.allocation.fx_carry),
        ]

    async def initialize(self):
        """Initialize: fetch instruments, map IDs, update portfolio state."""
        logger.info("=" * 60)
        logger.info("AlphaDesk initializing...")
        logger.info(f"Environment: {config.etoro.environment}")
        logger.info(f"Strategies: {[s.name for s in self.strategies]}")

        # Fetch instrument metadata and map IDs
        try:
            instruments = await self.etoro.get_instruments()
            self._map_instrument_ids(instruments)
            logger.info(f"Loaded {len(instruments)} instruments from eToro")
        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")

        # Update portfolio state
        await self.update_portfolio_state()
        logger.info("Initialization complete")
        logger.info("=" * 60)

    def _map_instrument_ids(self, instruments: list):
        """Map eToro instrument IDs to our universe."""
        from config.instruments import US_EQUITIES, EU_EQUITIES, FX_PAIRS

        # Build lookup by symbol name
        id_lookup = {}
        for inst in instruments:
            symbol = inst.get("SymbolFull", inst.get("symbolFull", ""))
            inst_id = inst.get("InstrumentID", inst.get("instrumentId"))
            if symbol and inst_id:
                id_lookup[symbol.upper()] = inst_id

        # Map to our universes
        for universe in [US_EQUITIES, EU_EQUITIES]:
            for symbol, meta in universe.items():
                if meta["etoro_id"] is None:
                    mapped_id = id_lookup.get(symbol.upper())
                    if mapped_id:
                        meta["etoro_id"] = mapped_id

        for pair, meta in FX_PAIRS.items():
            if meta["etoro_id"] is None:
                mapped_id = id_lookup.get(pair.upper())
                if mapped_id:
                    meta["etoro_id"] = mapped_id

    async def update_portfolio_state(self):
        """Refresh portfolio state from eToro."""
        try:
            balance = await self.etoro.get_account_balance()
            positions = await self.etoro.get_positions()
            self.risk_manager.update_state(balance, positions)
            logger.info(
                f"Portfolio: equity=${balance.get('equity', 0):,.2f}, "
                f"positions={len(positions)}, "
                f"drawdown={self.risk_manager.state.current_drawdown:.1%}"
            )
        except Exception as e:
            logger.error(f"Failed to update portfolio state: {e}")

    # ────────────────────── Main Cycle ──────────────────────

    async def run_signal_scan(self):
        """
        Main trading cycle:
        1. Update portfolio state
        2. Check existing positions for exits
        3. Generate new signals from all strategies
        4. Risk-check and size signals
        5. Execute approved trades
        """
        logger.info("─── Signal Scan Starting ───")

        # 1. Update state
        await self.update_portfolio_state()
        self.data_engine.clear_cache()

        # 2. Check exits on existing positions
        await self._check_exits()

        # 3. Check if trading is halted
        if self.risk_manager.state.is_halted:
            logger.warning("Trading is HALTED — skipping signal generation")
            return

        # 4. Generate signals from all strategies
        all_signals: List[TradeSignal] = []
        for strategy in self.strategies:
            try:
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

        # 5. Risk-check and execute
        executed = 0
        for signal in all_signals:
            try:
                # Risk check
                allowed, reason = self.risk_manager.check_can_trade(signal)
                if not allowed:
                    logger.info(f"Signal rejected: {signal.symbol} — {reason}")
                    continue

                # Size position
                asset_type = "fx" if signal.strategy_name == "fx_carry" else "equity"
                perf = self.db.get_strategy_performance(signal.strategy_name)
                sizing = self.position_sizer.compute_trade_size(
                    self.risk_manager.state.equity, signal, perf, asset_type
                )

                if not sizing.get("execute"):
                    logger.info(f"Signal skipped: {signal.symbol} — {sizing.get('reason')}")
                    continue

                # Log signal
                signal_id = self.db.log_signal(signal)

                # Execute trade
                result = await self.etoro.open_position(
                    instrument_id=signal.instrument_id,
                    direction=signal.direction,
                    amount=sizing["dollar_amount"],
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                )

                # Log trade
                self.db.log_trade_open(signal_id, {
                    "symbol": signal.symbol,
                    "strategy": signal.strategy_name,
                    "direction": signal.direction,
                    "amount": sizing["dollar_amount"],
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "etoro_position_id": result.get("positionId"),
                })

                # Notify
                await self.notifier.notify_signal(signal)
                await self.notifier.notify_trade_executed({
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "amount": sizing["dollar_amount"],
                })

                executed += 1
                logger.info(
                    f"✅ EXECUTED: {signal.direction} {signal.symbol} "
                    f"${sizing['dollar_amount']:,.2f} | {signal.strategy_name}"
                )

            except Exception as e:
                logger.error(f"Execution failed for {signal.symbol}: {e}")

        logger.info(f"─── Scan Complete: {executed}/{len(all_signals)} executed ───")

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
                        pos_id = position.get("positionId")
                        await self.etoro.close_position(pos_id)
                        logger.info(
                            f"🔴 EXIT: {symbol} | {exit_signal.metadata.get('exit_reason', '')}"
                        )
                        await self.notifier.notify_risk_alert(
                            "Position Closed",
                            f"{symbol}: {exit_signal.metadata.get('exit_reason', '')}"
                        )

                except Exception as e:
                    logger.error(f"Exit check failed for {symbol}: {e}")

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
            # In production, this would close/reduce positions
            if reduction >= 1.0:
                for pos in self.risk_manager.state.positions:
                    try:
                        await self.etoro.close_position(pos.get("positionId"))
                    except Exception as e:
                        logger.error(f"Failed to close {pos.get('symbol')}: {e}")

    async def run_daily_summary(self):
        """End-of-day summary and snapshot."""
        await self.update_portfolio_state()
        summary = self.risk_manager.get_portfolio_summary()
        self.db.save_daily_snapshot(summary)
        await self.notifier.notify_daily_summary(summary)
        logger.info(f"Daily snapshot saved: equity=${summary['equity']:,.2f}")

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
        await desk.run_daily_summary()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await desk.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
